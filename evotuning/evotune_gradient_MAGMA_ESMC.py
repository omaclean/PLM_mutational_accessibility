#!/usr/bin/env python
"""Evotune ESM-C on a local Spike FASTA using Muon + Magma optimisation."""

from __future__ import annotations

import argparse
import csv
import functools
import inspect
import math
import signal
import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

from functions import (
    SequenceRecord,
    build_split_uniqueness_and_divergence_summary,
    build_validation_nearest_training_rows,
    cluster_records_for_training_from_assignments,
    combine_log_history,
    ensure_cuda_available,
    iter_fasta,
    length_summary,
    log_training_parameters,
    merge_log_history_rows,
    normalise_sequence,
    sample_diverse_records,
    parse_bool,
    prepare_records_for_training_from_fasta,
    save_json,
    save_initial_eval_artifacts,
    save_training_artifacts,
    split_train_validation,
    split_train_validation_by_cluster,
    write_clean_fasta,
    write_fasta,
    write_metadata,
    write_split_summary,
    write_validation_nearest_training_csv,
)

if TYPE_CHECKING:
    pass
import os

def _resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    if token:
        return token
    token_file = os.environ.get("HF_TOKEN_FILE", "").strip()
    if token_file:
        p = Path(token_file).expanduser()
        if p.exists():
            return p.read_text().strip()
    cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token.exists():
        return cache_token.read_text().strip()
    return None

hf_token = _resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
os.environ["FSDP_USE_ORIG_PARAMS"] = "true"

# ---------------------------------------------------------------------------
# Hard-coded run configuration for this workspace
# ---------------------------------------------------------------------------
INPUT_FASTA = Path.home() / "earliest_haplo_spike_UK.fa"
OUTPUT_DIR = Path("SC2_out")

# ESM-C model options: "esmc_300m" or "esmc_600m"
MODEL_TYPE = "esmc_300m"

# ESM-C has a max sequence length of 2048 residues (before BOS/EOS tokens)
ESMC_MAX_RESIDUE_LENGTH = 2046  # 2048 - 2 for BOS/EOS

NUM_EPOCHS = 10
TRAIN_FRACTION = 0.8
RANDOM_SEED = 13
MLM_PROBABILITY = 0.15
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
DATALOADER_NUM_WORKERS = 4
TOKENIZE_NUM_PROC = 4
MIN_SEQUENCE_LENGTH = 150

TRUNCATE_TO_MODEL_LIMIT = True
RUN_MMSEQS_CLUSTER_PRECHECK = True
EXPORT_MODEL_EACH_EPOCH = True
RUN_MODE = "standard"
MAX_NEAREST_TRAINING_COMPARISONS = 5_000_000
MAX_SPLIT_SUMMARY_RECORDS = 5_000_000


# ---------------------------------------------------------------------------
# Minimal config stub so HuggingFace Trainer is happy without a real HF config
# ---------------------------------------------------------------------------
class _ESMCConfig:
    """Minimal stand-in for a HuggingFace PretrainedConfig for ESMC."""

    def __init__(self, vocab_size: int, max_position_embeddings: int = 2048):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        # Needed by Trainer / FSDP utilities
        self.model_type = "esmc"
        self.is_encoder_decoder = False
        self.use_cache = False
        self.tie_word_embeddings = False


# ---------------------------------------------------------------------------
# ESMC MLM wrapper — makes ESMC compatible with HuggingFace Trainer
# ---------------------------------------------------------------------------
class ESMCMLMWrapper(torch.nn.Module):
    """
    Wraps an ESMC instance and exposes a forward() that:
      - accepts  (input_ids, attention_mask, labels)
      - returns  a MaskedLMOutput-compatible namedtuple

    The ESMC forward() signature is:
        esmc(sequence_tokens: Tensor) -> ESMCOutput
    where ESMCOutput.sequence_logits has shape [B, L, vocab_size].
    """

    def __init__(self, esmc_client: Any, vocab_size: int, max_position_embeddings: int = 2048) -> None:
        super().__init__()
        self.esmc = esmc_client
        self.config = _ESMCConfig(vocab_size=vocab_size, max_position_embeddings=max_position_embeddings)
        self._gradient_checkpointing_applied = False

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        from transformers.modeling_outputs import MaskedLMOutput

        # ESMC forward: sequence_tokens shape [B, L]
        esmc_out = self.esmc(sequence_tokens=input_ids)
        logits = esmc_out.sequence_logits  # [B, L, vocab_size]

        loss = None
        if labels is not None:
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return MaskedLMOutput(loss=loss, logits=logits)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None) -> None:
        """Enable activation checkpointing on the ESM-C transformer blocks.

        The ESM-C SDK exposes no `gradient_checkpointing_enable()` of its own, so
        rather than silently doing nothing we apply PyTorch's activation
        checkpointing to every UnifiedTransformerBlock. Non-reentrant is the
        default because it composes with FSDP and tolerates block inputs that do
        not require grad; pass `{"use_reentrant": True}` to override.
        """
        kwargs = dict(gradient_checkpointing_kwargs or {})

        inner = self.esmc
        # Prefer a native implementation if a future SDK version grows one.
        if hasattr(inner, "gradient_checkpointing_enable"):
            inner.gradient_checkpointing_enable(**kwargs)
            return
        if hasattr(inner, "model") and hasattr(inner.model, "gradient_checkpointing_enable"):
            inner.model.gradient_checkpointing_enable(**kwargs)
            return

        if self._gradient_checkpointing_applied:
            # Trainer calls this again after our explicit call in main(); wrapping
            # the same blocks twice would recompute them twice.
            return

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        from esm.layers.blocks import UnifiedTransformerBlock

        use_reentrant = bool(kwargs.get("use_reentrant", False))
        impl = CheckpointImpl.REENTRANT if use_reentrant else CheckpointImpl.NO_REENTRANT
        num_blocks = sum(
            1 for module in inner.modules() if isinstance(module, UnifiedTransformerBlock)
        )
        if num_blocks == 0:
            _rank_zero_print(
                "[warning] ESMCMLMWrapper: no UnifiedTransformerBlock layers found; "
                "activation checkpointing was NOT enabled."
            )
            return

        apply_activation_checkpointing(
            inner,
            checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=impl),
            check_fn=lambda module: isinstance(module, UnifiedTransformerBlock),
        )
        self._gradient_checkpointing_applied = True
        _rank_zero_print(
            f"[info] Activation checkpointing enabled on {num_blocks} "
            f"UnifiedTransformerBlock layers (impl={impl.name})."
        )


def _save_native_esmc_weights(trainer: Any, destination: Path) -> None:
    """Gather a full, unsharded native ESM-C state dict and write it on rank 0.

    Under FSDP a FULL_STATE_DICT gather is a collective all-gather that every rank
    must enter together. Gating the gather itself behind a rank-0 check deadlocks
    the other ranks (or, if no gather happens, saves only rank 0's shard), so we
    gather on all ranks and only rank 0 — which receives the materialised
    tensors — writes to disk.
    """
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        # Collective: must run on every rank before any rank returns.
        full_state = accelerator.get_state_dict(trainer.model_wrapped)
        if not _is_rank_zero():
            return
        native_state = {
            key[len("esmc."):]: value
            for key, value in full_state.items()
            if key.startswith("esmc.")
        }
    else:
        if not _is_rank_zero():
            return
        native_state = trainer.model.esmc.state_dict()

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(native_state, destination)


class ExportNativeESMCWeightsCallback(TrainerCallback):
    """Export native ESM-C weights at each trainer save event."""

    def __init__(self, trainer: Any) -> None:
        self._trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        native_weights_path = checkpoint_dir / "esmc_weights.pt"

        # Collective gather across all ranks; the helper writes on rank 0 only.
        _save_native_esmc_weights(self._trainer, native_weights_path)

        if not state.is_world_process_zero:
            return control

        epoch_label = "unknown"
        if state.epoch is not None:
            epoch_label = f"{state.epoch:.2f}".rstrip("0").rstrip(".")

        _rank_zero_print(
            f"[info] Exported native ESM-C weights for epoch {epoch_label} "
            f"to {native_weights_path}"
        )
        return control


class TestCheckpointCleanupCallback(TrainerCallback):
    """Clean up intermediate checkpoint directories immediately after write testing in test mode."""

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
            _rank_zero_print(
                f"[test-mode] Deleted test checkpoint directory {checkpoint_dir} after write verification."
            )
        return control


class GracefulStopOnSignalCallback(TrainerCallback):
    """Request a checkpoint save and a clean stop when a preemption signal arrives."""

    def __init__(self) -> None:
        self.stop_requested = False
        self.is_preempted = False  # Track if the stop was caused by a signal trap
        self.signal_name: str | None = None
        self._register_signal(signal.SIGUSR1)
        self._register_signal(signal.SIGTERM)

    def _register_signal(self, sig: signal.Signals) -> None:
        try:
            signal.signal(sig, self._handle_signal)
        except (ValueError, OSError):
            # Signal registration can fail in non-main threads or restricted runtimes.
            return

    def _handle_signal(self, signum, _frame) -> None:
        self.stop_requested = True
        self.is_preempted = True  # Flag that we are handling a preemption event
        try:
            self.signal_name = signal.Signals(signum).name
        except ValueError:
            self.signal_name = str(signum)
        _rank_zero_print(
            f"[signal] Received {self.signal_name}; will save a checkpoint and stop after the current step."
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_requested:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if hasattr(FSDP, "_orig_fsdp_modules"):
            FSDP.fsdp_modules = FSDP._orig_fsdp_modules
            _rank_zero_print("[info] Restored FSDP.fsdp_modules in on_train_begin callback.")
        return control


class TokenizedSequenceDataset(torch.utils.data.Dataset):
    """Tokenize sequences lazily so the full corpus is never pre-tokenized in memory."""

    def __init__(self, records: list[SequenceRecord], tokenizer: Any, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        encoded = self.tokenizer(
            record.training_sequence,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "special_tokens_mask": encoded["special_tokens_mask"],
        }


class StreamingTokenizedFastaDataset(torch.utils.data.IterableDataset):
    """Stream tokenized FASTA rows without building a Hugging Face Arrow cache.

    Shards sequences across GPU ranks and DataLoader workers.
    """

    def __init__(self, fasta_path: Path, tokenizer: Any, max_length: int, num_records: int, test_mode: bool = False) -> None:
        super().__init__()
        self.fasta_path = fasta_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_records = min(num_records, 50) if test_mode else num_records
        self.test_mode = test_mode

    def __len__(self) -> int:
        return self.num_records

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # Process-level (per-GPU) sharding is handled by HuggingFace Accelerate,
        # which wraps this IterableDataset in an IterableDatasetShard so each rank
        # already sees only its 1/world_size slice. We must NOT shard by rank here
        # as well, or each rank would train on ~1/world_size**2 of the data while
        # __len__ still advertised the per-rank count (truncated epochs / DDP hang).
        # This dataset therefore only shards across DataLoader workers.
        limit = 50 if self.test_mode else None

        for index, (_, sequence) in enumerate(iter_fasta(self.fasta_path)):
            if limit is not None and index >= limit:
                break
            if index >= self.num_records:
                break
            if num_workers > 1 and index % num_workers != worker_id:
                continue

            encoded = self.tokenizer(
                sequence,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )
            yield {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "special_tokens_mask": encoded["special_tokens_mask"],
            }


def _distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_rank_zero() -> bool:
    return _distributed_rank() == 0


def _rank_zero_print(message: str) -> None:
    if _is_rank_zero():
        print(message)


def _sequence_record_to_dict(record: SequenceRecord) -> dict[str, object]:
    return {
        "name": record.name,
        "clean_sequence": record.clean_sequence,
        "training_sequence": record.training_sequence,
        "original_length": record.original_length,
        "truncated_length": record.truncated_length,
        "was_truncated": record.was_truncated,
    }


def _sequence_record_from_dict(payload: dict[str, object]) -> SequenceRecord:
    return SequenceRecord(
        name=str(payload["name"]),
        clean_sequence=str(payload["clean_sequence"]),
        training_sequence=str(payload["training_sequence"]),
        original_length=int(payload["original_length"]),
        truncated_length=int(payload["truncated_length"]),
        was_truncated=bool(payload["was_truncated"]),
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle)
    tmp_path.replace(path)


def _log_preprocessing_progress(message: str) -> None:
    _rank_zero_print(f"[progress] {message}")


def _should_generate_nearest_training_report(
    train_records: Sequence[SequenceRecord],
    val_records: Sequence[SequenceRecord],
) -> tuple[bool, int]:
    comparisons = len(train_records) * len(val_records)
    return comparisons <= MAX_NEAREST_TRAINING_COMPARISONS, comparisons


def _should_build_full_split_summary(train_records: Sequence[SequenceRecord], val_records: Sequence[SequenceRecord]) -> tuple[bool, int]:
    total_records = len(train_records) + len(val_records)
    return total_records <= MAX_SPLIT_SUMMARY_RECORDS, total_records


def _build_lightweight_split_summary(
    train_records: Sequence[SequenceRecord],
    val_records: Sequence[SequenceRecord],
    split_summary_extra: dict[str, object],
) -> dict[str, object]:
    return {
        "exact_uniqueness": {
            "skipped": True,
            "reason": "split summary disabled for large runs",
            "train_records": len(train_records),
            "validation_records": len(val_records),
        },
        "nearest_training_distance": {
            "skipped": True,
            "reason": "nearest training summary disabled for large runs",
        },
        "split_summary_extra": split_summary_extra,
    }


def _stream_filter_fasta_to_disk(
    *,
    input_fasta: Path,
    training_fasta: Path,
    validation_fasta: Path,
    max_residue_length: int,
    min_sequence_length: int,
    truncate_to_model_limit: bool,
    train_fraction: float,
    split_strategy: str,
    random_seed: int,
    max_records: int | None,
    max_val_records: int | None,
) -> dict[str, int]:
    """Filter/truncate a FASTA in a single streaming pass, writing train/val splits.

    Reads the input one record at a time so a corpus far larger than RAM can be
    ingested. The eager path (prepare_records_for_training_from_fasta) materialises
    every record plus a SequenceRecord per record (~3x the file size in Python
    objects), which OOM-kills the job on a 115GB corpus.

    Peak memory is bounded: O(max_records) when a training cap is set (reservoir
    sampling gives a uniform subset of the whole corpus, not just its head), and
    O(max_val_records) otherwise because only the validation set is buffered so it
    can be capped. Returns integer counters describing the pass.
    """
    import random as rand

    stats = {
        "input_records": 0,
        "dropped_empty": 0,
        "dropped_short": 0,
        "dropped_too_long": 0,
        "truncated_records": 0,
        "eligible_records": 0,
        "train_records": 0,
        "validation_records": 0,
    }

    # Independent streams so toggling a cap does not shift the split assignment.
    split_rng = rand.Random(random_seed)
    reservoir_rng = rand.Random(random_seed + 1)
    val_rng = rand.Random(random_seed + 2)
    everything_trains = split_strategy == "FULL"
    cap_total = max_records if (max_records is not None and max_records > 0) else None
    cap_val = max_val_records if (max_val_records is not None and max_val_records > 0) else None

    def _prepare(name: str, sequence: str) -> tuple[str, str] | None:
        seq = normalise_sequence(sequence)
        if not seq:
            stats["dropped_empty"] += 1
            return None
        if min_sequence_length and len(seq) < min_sequence_length:
            stats["dropped_short"] += 1
            return None
        if len(seq) > max_residue_length:
            if not truncate_to_model_limit:
                stats["dropped_too_long"] += 1
                return None
            seq = seq[:max_residue_length]
            stats["truncated_records"] += 1
        return name, seq

    def _is_train(record: tuple[str, str]) -> bool:
        return everything_trains or split_rng.random() < train_fraction

    training_fasta.parent.mkdir(parents=True, exist_ok=True)
    validation_fasta.parent.mkdir(parents=True, exist_ok=True)

    with training_fasta.open("w") as train_handle, validation_fasta.open("w") as val_handle:
        val_reservoir: list[tuple[str, str]] = []
        val_seen = 0

        def _write_train(record: tuple[str, str]) -> None:
            train_handle.write(f">{record[0]}\n{record[1]}\n")
            stats["train_records"] += 1

        def _offer_val(record: tuple[str, str]) -> None:
            # Reservoir-sample the validation stream so it stays <= cap_val.
            nonlocal val_seen
            if cap_val is None:
                val_handle.write(f">{record[0]}\n{record[1]}\n")
                stats["validation_records"] += 1
                return
            val_seen += 1
            if len(val_reservoir) < cap_val:
                val_reservoir.append(record)
            else:
                j = val_rng.randrange(val_seen)
                if j < cap_val:
                    val_reservoir[j] = record

        if cap_total is None:
            # No training cap: stream train straight to disk, buffer only val.
            for name, sequence in iter_fasta(input_fasta):
                stats["input_records"] += 1
                if stats["input_records"] % 5_000_000 == 0:
                    _log_preprocessing_progress(
                        f"Streamed {stats['input_records']:,} records "
                        f"({stats['eligible_records']:,} eligible)"
                    )
                prepared = _prepare(name, sequence)
                if prepared is None:
                    continue
                stats["eligible_records"] += 1
                if _is_train(prepared):
                    _write_train(prepared)
                else:
                    _offer_val(prepared)
        else:
            # Training cap: reservoir-sample the eligible records, then split.
            reservoir: list[tuple[str, str]] = []
            for name, sequence in iter_fasta(input_fasta):
                stats["input_records"] += 1
                if stats["input_records"] % 5_000_000 == 0:
                    _log_preprocessing_progress(
                        f"Streamed {stats['input_records']:,} records "
                        f"({stats['eligible_records']:,} eligible)"
                    )
                prepared = _prepare(name, sequence)
                if prepared is None:
                    continue
                stats["eligible_records"] += 1
                if len(reservoir) < cap_total:
                    reservoir.append(prepared)
                else:
                    j = reservoir_rng.randrange(stats["eligible_records"])
                    if j < cap_total:
                        reservoir[j] = prepared
            for record in reservoir:
                if _is_train(record):
                    _write_train(record)
                else:
                    _offer_val(record)

        for record in val_reservoir:
            val_handle.write(f">{record[0]}\n{record[1]}\n")
            stats["validation_records"] += 1

    if stats["validation_records"] == 0 and validation_fasta.exists():
        validation_fasta.unlink()

    stats["selected_records"] = stats["train_records"] + stats["validation_records"]
    stats["records_after_training_clustering"] = stats["selected_records"]
    return stats


def _compute_streaming_preprocessed_payload(
    *,
    input_fasta: Path,
    max_residue_length: int,
    min_sequence_length: int,
    truncate_to_model_limit: bool,
    max_records: int | None,
    max_val_records: int | None,
    random_seed: int,
    train_fraction: float,
    split_strategy: str,
    training_fasta: Path,
    validation_fasta: Path,
    config_json: Path,
    reports_dir: Path,
    training_parameters: dict[str, object],
    model_type: str,
    num_epochs: int,
    run_mode: str,
    export_model_each_epoch: bool,
) -> dict[str, object]:
    """Low-memory preprocessing for non-clustered runs (split FULL or random).

    Never holds the corpus in RAM, so per-record artifacts (clean FASTA, metadata
    CSV) and the uniqueness / nearest-neighbour reports are skipped — the same
    reports the eager path already auto-skips above MAX_*_RECORDS anyway.
    """
    _log_preprocessing_progress(
        f"Streaming {input_fasta} -> train/validation FASTA "
        f"(cap={max_records if max_records else 'none'}, split={split_strategy})"
    )
    prep_stats = _stream_filter_fasta_to_disk(
        input_fasta=input_fasta,
        training_fasta=training_fasta,
        validation_fasta=validation_fasta,
        max_residue_length=max_residue_length,
        min_sequence_length=min_sequence_length,
        truncate_to_model_limit=truncate_to_model_limit,
        train_fraction=train_fraction,
        split_strategy=split_strategy,
        random_seed=random_seed,
        max_records=max_records,
        max_val_records=max_val_records,
    )
    if prep_stats["train_records"] == 0:
        raise ValueError("No usable sequences remained after preprocessing.")

    _log_preprocessing_progress(
        f"Streaming pass complete: {prep_stats['input_records']:,} read, "
        f"{prep_stats['train_records']:,} train, {prep_stats['validation_records']:,} validation"
    )

    total_selected = prep_stats["train_records"] + prep_stats["validation_records"]
    clustering_summary: dict[str, object] = {
        "used": False,
        "identity_threshold": None,
        "cluster_assignments_csv": None,
        "input_records": prep_stats["input_records"],
        "retained_records": total_selected,
        "retained_fraction": (
            total_selected / prep_stats["input_records"] if prep_stats["input_records"] else 0.0
        ),
    }
    split_summary_extra: dict[str, object] = {
        "cluster_based_split": False,
        "random_seed": random_seed,
        "streaming": True,
        "full_training": split_strategy == "FULL",
        "validation_nearest_training_report_skipped": True,
        "validation_nearest_training_report_reason": "streaming mode: corpus not held in memory",
        "validation_nearest_training_csv": None,
        "split_uniqueness": {
            "skipped": True,
            "reason": "streaming mode: corpus not held in memory",
        },
    }

    save_json(
        {
            "input_fasta": str(input_fasta),
            "streaming": True,
            "cluster_assignments_csv": None,
            "training_parameters": training_parameters,
            "preprocessing": prep_stats,
            "training_clustering": clustering_summary,
            "model_type": model_type,
            "num_epochs": num_epochs,
            "train_fraction": train_fraction,
            "random_seed": random_seed,
            "run_mode": run_mode,
            "split_strategy": split_strategy,
            "max_records": max_records,
            "max_val_records": max_val_records,
            "export_model_each_epoch": export_model_each_epoch,
            "note": (
                "Streaming mode: per-record metadata and uniqueness/nearest-neighbour "
                "reports are skipped because the corpus is never held in memory."
            ),
        },
        config_json,
    )
    save_json(
        {
            "split_strategy": split_strategy,
            "train_records": prep_stats["train_records"],
            "validation_records": prep_stats["validation_records"],
            "train_mean_distance_to_reference": None,
            "validation_mean_distance_to_reference": None,
            **split_summary_extra,
        },
        reports_dir / "split_summary.json",
    )

    return {
        "prep_stats": prep_stats,
        "clustering_summary": clustering_summary,
        "split_summary_extra": split_summary_extra,
        "clean_fasta": None,
        "training_fasta": str(training_fasta),
        "validation_fasta": str(validation_fasta),
        "train_records_count": prep_stats["train_records"],
        "val_records_count": prep_stats["validation_records"],
    }


def _compute_preprocessed_payload(
    *,
    input_fasta: Path,
    cluster_assignments_csv: Path | None,
    cluster_identity_for_training: float | None,
    cluster_identity_for_split: float | None,
    max_residue_length: int,
    min_sequence_length: int,
    truncate_to_model_limit: bool,
    max_records: int | None,
    max_val_records: int | None,
    random_seed: int,
    train_fraction: float,
    split_strategy: str,
    clean_fasta: Path,
    training_fasta: Path,
    validation_fasta: Path,
    metadata_csv: Path,
    config_json: Path,
    reports_dir: Path,
    training_parameters: dict[str, object],
    model_type: str,
    num_epochs: int,
    run_mode: str,
    export_model_each_epoch: bool,
) -> dict[str, object]:
    # Clustered runs need every record in memory to resolve cluster membership, so
    # they keep the eager path (their input FASTA is pre-filtered to the cluster
    # assignments upstream, which bounds its size). Everything else streams.
    use_streaming = (
        cluster_assignments_csv is None
        and cluster_identity_for_training is None
        and cluster_identity_for_split is None
        and split_strategy in ("FULL", "random")
    )
    if use_streaming:
        return _compute_streaming_preprocessed_payload(
            input_fasta=input_fasta,
            max_residue_length=max_residue_length,
            min_sequence_length=min_sequence_length,
            truncate_to_model_limit=truncate_to_model_limit,
            max_records=max_records,
            max_val_records=max_val_records,
            random_seed=random_seed,
            train_fraction=train_fraction,
            split_strategy=split_strategy,
            training_fasta=training_fasta,
            validation_fasta=validation_fasta,
            config_json=config_json,
            reports_dir=reports_dir,
            training_parameters=training_parameters,
            model_type=model_type,
            num_epochs=num_epochs,
            run_mode=run_mode,
            export_model_each_epoch=export_model_each_epoch,
        )

    _log_preprocessing_progress(f"Preparing training records from {input_fasta}")
    records, prep_stats, clustering_summary = prepare_records_for_training_from_fasta(
        input_fasta,
        max_residue_length=max_residue_length,
        min_sequence_length=min_sequence_length,
        truncate_to_model_limit=truncate_to_model_limit,
        cluster_assignments_csv=cluster_assignments_csv,
        cluster_identity_for_training=cluster_identity_for_training,
    )
    records = sample_diverse_records(
        records,
        max_records,
        cluster_assignments_csv=cluster_assignments_csv,
        random_seed=random_seed,
    )

    if clustering_summary["used"]:
        clustering_summary = {
            **clustering_summary,
            "retained_records": len(records),
            "retained_fraction": (
                len(records) / clustering_summary["input_records"]
                if clustering_summary["input_records"]
                else 0.0
            ),
        }
    else:
        clustering_summary = {
            **clustering_summary,
            "input_records": len(records),
            "retained_records": len(records),
            "retained_fraction": 1.0,
        }

    if not records:
        raise ValueError("No usable sequences remained after preprocessing.")

    prep_stats["selected_records"] = len(records)
    prep_stats["records_after_training_clustering"] = len(records)

    _log_preprocessing_progress(f"Resolving train/validation split with strategy={split_strategy}")
    train_records, val_records, split_summary_extra = resolve_train_validation_records(
        records,
        cluster_assignments_csv=cluster_assignments_csv,
        cluster_identity_for_split=cluster_identity_for_split,
        train_fraction=train_fraction,
        random_seed=random_seed,
        split_strategy=split_strategy,
    )

    if max_val_records is not None and max_val_records > 0 and len(val_records) > max_val_records:
        import random as rand
        _log_preprocessing_progress(
            f"Downsampling validation records from {len(val_records):,} to {max_val_records:,} "
            f"using seed {random_seed}."
        )
        rng = rand.Random(random_seed)
        val_records = rng.sample(val_records, max_val_records)

    _log_preprocessing_progress(
        f"Writing cleaned FASTA and metadata for {len(records):,} retained records"
    )
    write_clean_fasta(records, clean_fasta)
    write_fasta(train_records, training_fasta)
    if val_records:
        write_fasta(val_records, validation_fasta)
    elif validation_fasta.exists():
        validation_fasta.unlink()
    write_metadata(records, metadata_csv)

    _log_preprocessing_progress("Saving preprocessing summary report")
    save_json(
        {
            "input_fasta": str(input_fasta),
            "cluster_assignments_csv": (
                str(cluster_assignments_csv) if cluster_assignments_csv else None
            ),
            "cluster_identity_for_training": cluster_identity_for_training,
            "cluster_identity_for_split": cluster_identity_for_split,
            "training_parameters": training_parameters,
            "min_sequence_length": min_sequence_length,
            "preprocessing": prep_stats,
            "training_clustering": clustering_summary,
            "length_summary": length_summary(records),
            "model_type": model_type,
            "num_epochs": num_epochs,
            "train_fraction": train_fraction,
            "random_seed": random_seed,
            "run_mode": run_mode,
            "split_strategy": split_strategy,
            "max_records": max_records,
            "max_val_records": max_val_records,
            "export_model_each_epoch": export_model_each_epoch,
        },
        config_json,
    )

    nearest_training_csv = reports_dir / "validation_nearest_training.csv"
    should_generate_nearest_report, nearest_report_comparisons = _should_generate_nearest_training_report(
        train_records,
        val_records,
    )
    nearest_training_rows: list[dict[str, object]] = []
    if should_generate_nearest_report:
        _log_preprocessing_progress(
            "Building validation-to-training nearest-neighbor report "
            f"for {len(val_records):,} validation records against {len(train_records):,} training records"
        )
        nearest_training_rows = build_validation_nearest_training_rows(train_records, val_records)
        write_validation_nearest_training_csv(nearest_training_rows, nearest_training_csv)
        split_summary_extra["validation_nearest_training_report_skipped"] = False
        split_summary_extra["validation_nearest_training_csv"] = str(nearest_training_csv)
    else:
        _log_preprocessing_progress(
            "Skipping validation-to-training nearest-neighbor report for large run: "
            f"{len(val_records):,} x {len(train_records):,} comparisons = {nearest_report_comparisons:,}"
        )
        split_summary_extra["validation_nearest_training_report_skipped"] = True
        split_summary_extra["validation_nearest_training_report_reason"] = (
            "comparison count exceeded limit"
        )
        split_summary_extra["validation_nearest_training_report_threshold"] = (
            MAX_NEAREST_TRAINING_COMPARISONS
        )
        split_summary_extra["validation_nearest_training_report_comparisons"] = (
            nearest_report_comparisons
        )
        split_summary_extra["validation_nearest_training_csv"] = None
    should_build_full_split_summary, total_split_records = _should_build_full_split_summary(
        train_records,
        val_records,
    )
    if should_build_full_split_summary:
        _log_preprocessing_progress("Building split uniqueness summary")
        split_uniqueness = build_split_uniqueness_and_divergence_summary(
            train_records,
            val_records,
            nearest_rows=nearest_training_rows if should_generate_nearest_report else None,
        )
    else:
        _log_preprocessing_progress(
            "Skipping split uniqueness summary for large run: "
            f"{len(train_records):,} train + {len(val_records):,} validation = {total_split_records:,} records"
        )
        split_uniqueness = _build_lightweight_split_summary(
            train_records,
            val_records,
            split_summary_extra.copy(),
        )
    split_summary_extra["split_uniqueness"] = split_uniqueness

    _log_preprocessing_progress("Writing split summary")
    if should_build_full_split_summary:
        write_split_summary(
            train_records,
            val_records,
            split_strategy,
            reports_dir / "split_summary.json",
            extra_summary=split_summary_extra,
        )
    else:
        save_json(
            {
                "split_strategy": split_strategy,
                "train_records": len(train_records),
                "validation_records": len(val_records),
                "train_mean_distance_to_reference": None,
                "validation_mean_distance_to_reference": None,
                **split_summary_extra,
            },
            reports_dir / "split_summary.json",
        )

    return {
        "prep_stats": prep_stats,
        "clustering_summary": clustering_summary,
        "split_summary_extra": split_summary_extra,
        "clean_fasta": str(clean_fasta),
        "training_fasta": str(training_fasta),
        "validation_fasta": str(validation_fasta),
        "train_records_count": len(train_records),
        "val_records_count": len(val_records),
    }


def _preprocessing_leg_token() -> str:
    """Identify this requeue leg.

    Non-zero ranks must not accept a ready-marker left behind by a previous leg,
    otherwise they race ahead to init_process_group while rank 0 is still
    preprocessing and die on the 1800s TCPStore timeout.
    """
    return f"{os.environ.get('SLURM_JOB_ID', 'manual')}:{os.environ.get('SLURM_RESTART_COUNT', '0')}"


def _preprocessing_cache_key(
    *,
    input_fasta: Path,
    cluster_assignments_csv: Path | None,
    cluster_identity_for_training: float | None,
    cluster_identity_for_split: float | None,
    max_residue_length: int,
    min_sequence_length: int,
    truncate_to_model_limit: bool,
    max_records: int | None,
    max_val_records: int | None,
    random_seed: int,
    train_fraction: float,
    split_strategy: str,
) -> dict[str, object]:
    """Everything that changes what preprocessing would produce."""
    try:
        stat = input_fasta.stat()
        fasta_size: int | None = stat.st_size
    except OSError:
        fasta_size = None
    return {
        "input_fasta": str(input_fasta),
        "input_fasta_size": fasta_size,
        "cluster_assignments_csv": str(cluster_assignments_csv) if cluster_assignments_csv else None,
        "cluster_identity_for_training": cluster_identity_for_training,
        "cluster_identity_for_split": cluster_identity_for_split,
        "max_residue_length": max_residue_length,
        "min_sequence_length": min_sequence_length,
        "truncate_to_model_limit": bool(truncate_to_model_limit),
        "max_records": max_records,
        "max_val_records": max_val_records,
        "random_seed": random_seed,
        "train_fraction": train_fraction,
        "split_strategy": split_strategy,
    }


def _reusable_cache_payload(cache_path: Path, expected_key: dict[str, object]) -> dict | None:
    """Return the cached payload if it is valid for these inputs, else None.

    Rebuilding this cache costs ~14.5 hours on the all-virus corpus, and rank 0
    previously rebuilt it unconditionally on every requeue leg — which both wasted
    most of each allocation and caused the rank race described above.
    """
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text())
    except Exception as exc:
        print(f"[info] Preprocessing cache unreadable ({type(exc).__name__}: {exc}); rebuilding.")
        return None

    # Outputs the cache points at must still be on disk and non-empty.
    for field in ("training_fasta", "validation_fasta", "clean_fasta"):
        raw = payload.get(field)
        if not raw:
            print(f"[info] Preprocessing cache missing '{field}'; rebuilding.")
            return None
        path = Path(raw)
        if not path.exists() or path.stat().st_size == 0:
            print(f"[info] Preprocessing cache references missing/empty {path}; rebuilding.")
            return None

    cached_key = payload.get("cache_key")
    if cached_key is not None:
        mismatch = [k for k, v in expected_key.items() if cached_key.get(k) != v]
        if mismatch:
            print(f"[info] Preprocessing cache key differs on {mismatch}; rebuilding.")
            return None
        return payload

    # Legacy cache written before cache_key existed: fall back to the fields the
    # summaries do record, so an in-flight run can still reuse its own work.
    clustering = payload.get("clustering_summary") or {}
    split_extra = payload.get("split_summary_extra") or {}
    checks = (
        (clustering.get("cluster_assignments_csv"), expected_key["cluster_assignments_csv"], "cluster_assignments_csv"),
        (clustering.get("identity_threshold"), expected_key["cluster_identity_for_training"], "cluster_identity_for_training"),
        (split_extra.get("cluster_identity_threshold"), expected_key["cluster_identity_for_split"], "cluster_identity_for_split"),
    )
    for actual, expected, label in checks:
        if expected is not None and actual is not None and actual != expected:
            print(f"[info] Legacy preprocessing cache differs on {label} ({actual!r} != {expected!r}); rebuilding.")
            return None
    print("[info] Reusing legacy preprocessing cache (no cache_key recorded); it will gain one on next rebuild.")
    return payload


def _prepare_training_inputs_once(
    *,
    cache_path: Path,
    failure_path: Path,
    input_fasta: Path,
    cluster_assignments_csv: Path | None,
    cluster_identity_for_training: float | None,
    cluster_identity_for_split: float | None,
    max_residue_length: int,
    min_sequence_length: int,
    truncate_to_model_limit: bool,
    max_records: int | None,
    max_val_records: int | None,
    random_seed: int,
    train_fraction: float,
    split_strategy: str,
    clean_fasta: Path,
    training_fasta: Path,
    validation_fasta: Path,
    metadata_csv: Path,
    config_json: Path,
    reports_dir: Path,
    training_parameters: dict[str, object],
    model_type: str,
    num_epochs: int,
    run_mode: str,
    export_model_each_epoch: bool,
) -> tuple[Path, Path, int, int, dict[str, int], dict[str, object], dict[str, object]]:
    ready_path = cache_path.with_name(cache_path.stem + ".ready")
    leg_token = _preprocessing_leg_token()
    expected_key = _preprocessing_cache_key(
        input_fasta=input_fasta,
        cluster_assignments_csv=cluster_assignments_csv,
        cluster_identity_for_training=cluster_identity_for_training,
        cluster_identity_for_split=cluster_identity_for_split,
        max_residue_length=max_residue_length,
        min_sequence_length=min_sequence_length,
        truncate_to_model_limit=truncate_to_model_limit,
        max_records=max_records,
        max_val_records=max_val_records,
        random_seed=random_seed,
        train_fraction=train_fraction,
        split_strategy=split_strategy,
    )

    if _is_rank_zero():
        # Retract any marker from a previous leg before deciding, so a rank that
        # checks early cannot mistake it for this leg's signal.
        try:
            ready_path.unlink()
        except FileNotFoundError:
            pass

        try:
            payload = _reusable_cache_payload(cache_path, expected_key)
            if payload is not None:
                _log_preprocessing_progress(
                    f"Reusing preprocessing cache at {cache_path} "
                    f"({payload.get('train_records_count')} train / {payload.get('val_records_count')} val)"
                )
                if failure_path.exists():
                    failure_path.unlink()
                ready_path.write_text(leg_token)
                return _unpack_preprocessed_payload(payload)

            _log_preprocessing_progress(f"Starting preprocessing cache build at {cache_path}")
            payload = _compute_preprocessed_payload(
                input_fasta=input_fasta,
                cluster_assignments_csv=cluster_assignments_csv,
                cluster_identity_for_training=cluster_identity_for_training,
                cluster_identity_for_split=cluster_identity_for_split,
                max_residue_length=max_residue_length,
                min_sequence_length=min_sequence_length,
                truncate_to_model_limit=truncate_to_model_limit,
                max_records=max_records,
                max_val_records=max_val_records,
                random_seed=random_seed,
                train_fraction=train_fraction,
                split_strategy=split_strategy,
                clean_fasta=clean_fasta,
                training_fasta=training_fasta,
                validation_fasta=validation_fasta,
                metadata_csv=metadata_csv,
                config_json=config_json,
                reports_dir=reports_dir,
                training_parameters=training_parameters,
                model_type=model_type,
                num_epochs=num_epochs,
                run_mode=run_mode,
                export_model_each_epoch=export_model_each_epoch,
            )
            payload["cache_key"] = expected_key
            if failure_path.exists():
                failure_path.unlink()
            _write_json_atomic(cache_path, payload)
            ready_path.write_text(leg_token)
            _log_preprocessing_progress(f"Finished preprocessing cache build at {cache_path}")
        except Exception as exc:
            failure_path.write_text(f"{type(exc).__name__}: {exc}\n")
            raise
    else:
        # Wait for THIS leg's marker, not merely for the cache file to exist. On a
        # requeue the cache is already present from the previous leg, so the old
        # existence check let these ranks run ahead to init_process_group while
        # rank 0 was still preprocessing; they then died on the TCPStore timeout.
        print(
            f"[info] Rank {_distributed_rank()} waiting for rank 0 preprocessing "
            f"(leg {leg_token}) marker: {ready_path}",
            flush=True,
        )
        waited = 0
        while True:
            if ready_path.exists() and ready_path.read_text().strip() == leg_token:
                break
            if failure_path.exists():
                raise RuntimeError(
                    f"Rank 0 preprocessing failed: {failure_path.read_text().strip()}"
                )
            time.sleep(5)
            waited += 5
            if waited % 600 == 0:
                print(
                    f"[info] Rank {_distributed_rank()} still waiting for preprocessing "
                    f"({waited // 60} min elapsed)",
                    flush=True,
                )

    payload = json.loads(cache_path.read_text())
    return _unpack_preprocessed_payload(payload)


def _unpack_preprocessed_payload(payload: dict) -> tuple[Path, Path, int, int, dict[str, int], dict[str, object], dict[str, object]]:
    prep_stats = {key: int(value) for key, value in payload["prep_stats"].items()}
    clustering_summary = dict(payload["clustering_summary"])
    split_summary_extra = dict(payload["split_summary_extra"])
    return (
        Path(payload["training_fasta"]),
        Path(payload["validation_fasta"]),
        int(payload["train_records_count"]),
        int(payload["val_records_count"]),
        prep_stats,
        clustering_summary,
        split_summary_extra,
    )


def load_checkpoint_epoch(checkpoint_dir: Path | None) -> float:
    """Return the saved trainer epoch for a checkpoint, or 0.0 when unavailable."""
    if checkpoint_dir is None:
        return 0.0

    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return 0.0

    try:
        payload = json.loads(trainer_state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0.0

    epoch = payload.get("epoch")
    if epoch is None:
        return 0.0

    try:
        return float(epoch)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Optimisers (unchanged from ESM-2 MAGMA script)
# ---------------------------------------------------------------------------
def _fsdp_parameter_gradient_view(
    parameter: torch.Tensor,
    gradient: torch.Tensor | None,
) -> torch.Tensor | None:
    """Match an original FSDP parameter's local view to its gradient view."""
    if gradient is None:
        return None
    if parameter.numel() == 0 or gradient.numel() == 0:
        return None
    if parameter.shape == gradient.shape:
        return gradient
    if parameter.numel() == gradient.numel():
        return gradient.reshape_as(parameter)
    raise RuntimeError(
        "parameter and gradient have different numbers of elements: "
        f"{parameter.numel()} vs {gradient.numel()}. This means the parameter is "
        "sharded while its gradient is not, i.e. FSDP is sharding parameters. "
        "Muon requires whole 2D weight matrices, so this script sets "
        "fsdp='no_shard auto_wrap'; check that the sharding strategy has not been "
        "changed back to full_shard/shard_grad_op."
    )


_FSDP_MODULE_PREFIX = "_fsdp_wrapped_module."
_CHECKPOINT_MODULE_PREFIX = "_checkpoint_wrapped_module."


def _clean_parameter_name(name: str) -> str:
    """Strip FSDP and activation-checkpoint wrapper segments from a parameter name.

    Trainer creates the optimiser *after* `accelerator.prepare()` has wrapped the
    model, and activation checkpointing inserts wrappers of its own, so
    `named_parameters()` yields names such as
    `_fsdp_wrapped_module.esmc.transformer.blocks.0._checkpoint_wrapped_module.attn.…`.
    Matching those raw names against the plain module path silently selects
    nothing, which quietly disables Magma entirely.
    """
    return name.replace(_FSDP_MODULE_PREFIX, "").replace(_CHECKPOINT_MODULE_PREFIX, "")


def _is_magma_target_parameter_name(name: str) -> bool:
    """Return whether an ESM-C attention or feed-forward parameter uses Magma."""
    cleaned = _clean_parameter_name(name)
    return cleaned.startswith("esmc.transformer.blocks.") and (
        ".attn." in cleaned or ".ffn." in cleaned
    )


class MuonAdamW(torch.optim.Optimizer):
    """Hybrid optimiser: Muon for >=2D parameters, AdamW for 1D parameters."""

    def __init__(self, params, lr=3e-4, muon_lr=0.02, momentum=0.95, betas=(0.9, 0.95), weight_decay=0.01):
        defaults = dict(lr=lr, muon_lr=muon_lr, momentum=momentum, betas=betas, weight_decay=weight_decay, initial_lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = _fsdp_parameter_gradient_view(p, p.grad)
                if grad is None:
                    continue
                state = self.state[p]
                factor = state.pop("magma_factor", 1.0)

                if p.ndim >= 2:
                    # Muon Orthogonalisation
                    if state.get("exp_avg") is None or state["exp_avg"].shape != grad.shape:
                        state["exp_avg"] = torch.zeros_like(grad)
                    mu_t = state["exp_avg"]

                    orig_shape = grad.shape
                    grad_2d = grad.view(orig_shape[0], -1)

                    a, b, c = 3.4445, -4.7750, 2.0315
                    X = grad_2d.bfloat16() if grad_2d.dtype != torch.bfloat16 else grad_2d
                    X = X / (X.norm() + 1e-7)
                    transposed = X.size(0) > X.size(1)
                    if transposed:
                        X = X.T

                    for _ in range(5):
                        A = X @ X.T
                        B = b * A + c * (A @ A)
                        X = a * X + B @ X

                    if transposed:
                        X = X.T
                    g_ortho = X.to(grad.dtype).view(orig_shape)

                    mu_t.mul_(group["momentum"]).add_(g_ortho, alpha=1 - group["momentum"])
                    initial_lr = group.get("initial_lr", 1e-4)
                    lr_ratio = group["lr"] / initial_lr if initial_lr > 0 else 1.0
                    p.add_(mu_t, alpha=-group["muon_lr"] * lr_ratio * factor)

                else:
                    # AdamW Fallback
                    if (
                        state.get("exp_avg") is None
                        or state["exp_avg"].shape != grad.shape
                        or state.get("exp_avg_sq") is None
                        or state["exp_avg_sq"].shape != grad.shape
                    ):
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                        state["step"] = 0

                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] / bias_correction1

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)

                    p.data.mul_(1 - factor * group["lr"] * group["weight_decay"])
                    p.addcdiv_(exp_avg, denom, value=-factor * step_size)

        return loss


class MagmaOptimiser(torch.optim.Optimizer):
    """
    Momentum-aligned gradient masking (Magma) drop-in wrapper.
    Ensures internal optimiser states update densely while masking parameter application.
    """

    def __init__(self, base_optimiser, tau=2.0, p=0.5, mask_seed: int = 0):
        if not 0 < p <= 1:
            raise ValueError(f"Magma survival probability must be in (0, 1], got {p}.")
        self.base_optimiser = base_optimiser
        self.tau = tau
        self.p = p
        self.mask_seed = int(mask_seed)
        self._magma_step = 0

        defaults = dict(tau=tau, p=p, mask_seed=mask_seed)
        super().__init__(base_optimiser.param_groups, defaults)
        self.state = self.base_optimiser.state

    def zero_grad(self, set_to_none=True):
        self.base_optimiser.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        state_dict = self.base_optimiser.state_dict()
        state_dict['magma_meta'] = {
            'tau': self.tau,
            'p': self.p,
            'mask_seed': self.mask_seed,
            'step': self._magma_step,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        magma_meta = dict(state_dict.get('magma_meta', {}))
        base_state_dict = {key: value for key, value in state_dict.items() if key != 'magma_meta'}
        self.base_optimiser.load_state_dict(base_state_dict)
        self.state = self.base_optimiser.state

        self.tau = magma_meta.get('tau', self.tau)
        self.p = magma_meta.get('p', self.p)
        self.mask_seed = int(magma_meta.get('mask_seed', self.mask_seed))
        self._magma_step = int(magma_meta.get('step', self._magma_step))
        self.defaults.update(tau=self.tau, p=self.p, mask_seed=self.mask_seed)

    def _sample_mask_like(self, reference: torch.Tensor, *, step: int, param_index: int) -> torch.Tensor:
        seed = (self.mask_seed + 1_000_003 * step + 97_003 * param_index) % (2**63 - 1)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
        probs = torch.full(reference.shape, self.p, dtype=torch.float32, device='cpu')
        mask = torch.bernoulli(probs, generator=generator)
        return mask.to(device=reference.device, dtype=reference.dtype, non_blocking=True)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # ─── Existing Magma Masking Evaluation ───
        param_indices: dict[torch.Tensor, int] = {}
        current_step = self._magma_step
        param_index = 0

        for group in self.param_groups:
            if not group.get("magma_enabled", False):
                continue
            for p in group["params"]:
                param_indices[p] = param_index
                param_index += 1

                if p.grad is None:
                    continue

                grad = _fsdp_parameter_gradient_view(p, p.grad)
                if grad is None:
                    continue
                state = self.state[p]

                if "exp_avg" in state and state["exp_avg"].shape == grad.shape:
                    mu_t = state["exp_avg"]
                else:
                    mu_t = grad

                cos_sim = F.cosine_similarity(
                    mu_t.flatten().unsqueeze(0), grad.flatten().unsqueeze(0)
                ).squeeze()
                s_tilde = torch.sigmoid(cos_sim / self.tau)

                if "magma_s_t" not in state:
                    state["magma_s_t"] = s_tilde.detach().clone()
                else:
                    state["magma_s_t"] = 0.9 * state["magma_s_t"] + 0.1 * s_tilde

                s_t = state["magma_s_t"]

                m_t = self._sample_mask_like(s_t, step=current_step, param_index=param_indices[p])
                # The paper applies block-wise Bernoulli masking and rescales a
                # surviving update by 1 / p. The alignment EMA is intentional
                # damping and is therefore not bias-corrected.
                state["magma_factor"] = (s_t * m_t / self.p).item()

        # Synchronise all param group attributes (like lr, weight_decay, etc.) to the base optimiser
        for group_magma, group_base in zip(self.param_groups, self.base_optimiser.param_groups):
            for key in group_magma.keys():
                if key != "params":
                    group_base[key] = group_magma[key]

        # Execute inner base step with updated inner_group metrics
        self.base_optimiser.step()

        self._magma_step = current_step + 1
        return loss


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evotune ESM-C on a FASTA file.")
    parser.add_argument("--input-fasta", type=Path, default=INPUT_FASTA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--run-mode",
        choices=("standard", "full-finetune"),
        default=RUN_MODE,
        help=(
            "Run preset. 'full-finetune' disables eval splitting, clustering, and "
            "max-record downsampling so every prepared FASTA sequence is used for training."
        ),
    )
    parser.add_argument(
        "--model-type",
        default=MODEL_TYPE,
        help='ESM-C model variant: "esmc_300m" or "esmc_600m"',
    )
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--mlm-probability", type=float, default=MLM_PROBABILITY)
    parser.add_argument("--train-batch-size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=PER_DEVICE_EVAL_BATCH_SIZE)
    parser.add_argument("--dataloader-num-workers", type=int, default=DATALOADER_NUM_WORKERS)
    parser.add_argument("--tokenize-num-proc", type=int, default=TOKENIZE_NUM_PROC)
    parser.add_argument("--require-cuda", type=parse_bool, default=True)
    parser.add_argument("--bf16", type=parse_bool, default=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=MIN_SEQUENCE_LENGTH,
        help="Drop sequences shorter than this amino-acid length before training (default: 150).",
    )
    parser.add_argument("--truncate-to-model-limit", type=parse_bool, default=TRUNCATE_TO_MODEL_LIMIT)
    parser.add_argument(
        "--run-mmseqs-cluster-precheck",
        type=parse_bool,
        default=RUN_MMSEQS_CLUSTER_PRECHECK,
        help="Accepted for CLI compatibility; ignored by this script.",
    )
    parser.add_argument("--cluster-assignments-csv", type=Path, default=None)
    parser.add_argument("--cluster-identity-for-training", type=float, default=None)
    parser.add_argument("--cluster-identity-for-split", type=float, default=None)
    parser.add_argument("--split-strategy", choices=("random", "reference-divergent", "FULL"), default="random")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--max-val-records",
        type=int,
        default=10000,
        help="Limit validation set to at most this many records (default: 10000). Set <= 0 to disable limit.",
    )
    parser.add_argument(
        "--test-mode",
        type=parse_bool,
        default=False,
        help="Skip saving model weights at the end of training (for quick pipeline tests).",
    )
    parser.add_argument(
        "--test-checkpoint-save-and-cleanup",
        type=parse_bool,
        default=False,
        help="When test-mode is true, exercise full checkpoint saving/export logic to verify serialization, then delete saved checkpoints automatically.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test downsampling mode (max 50 sequences).",
    )
    parser.add_argument(
        "--export-model-each-epoch",
        type=parse_bool,
        default=EXPORT_MODEL_EACH_EPOCH,
        help="Export native ESM-C weights into each epoch checkpoint directory.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a Trainer checkpoint directory to resume from "
            "(for example /projects/.../model/checkpoint-50)."
        ),
    )
    return parser


def resolve_run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    split_strategy = args.split_strategy
    cluster_identity_for_training = args.cluster_identity_for_training
    cluster_identity_for_split = args.cluster_identity_for_split
    max_records = args.max_records
    clustering_disabled = False

    if args.run_mode == "full-finetune":
        split_strategy = "FULL"
        cluster_identity_for_training = None
        cluster_identity_for_split = None
        max_records = None
        clustering_disabled = True

    return {
        "split_strategy": split_strategy,
        "cluster_identity_for_training": cluster_identity_for_training,
        "cluster_identity_for_split": cluster_identity_for_split,
        "max_records": max_records,
        "clustering_disabled": clustering_disabled,
    }


def build_training_arguments_kwargs(
    *,
    output_dir: Path,
    num_epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    dataloader_num_workers: int,
    random_seed: int,
    has_eval: bool,
    bf16: bool,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    export_model_each_epoch: bool,
) -> dict[str, Any]:
    eval_value = "epoch" if has_eval else "no"
    return {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": num_epochs,
        "logging_strategy": "epoch",
        "save_strategy": "epoch" if export_model_each_epoch else "no",
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "eval_strategy": eval_value,
        "evaluation_strategy": eval_value,
        "save_total_limit": None if export_model_each_epoch else 1,
        "seed": random_seed,
        "dataloader_num_workers": dataloader_num_workers,
        "disable_tqdm": not _is_rank_zero(),
        "log_on_each_node": False,
        "dispatch_batches": False,
        "report_to": "none",
        "do_train": True,
        "do_eval": has_eval,
        "remove_unused_columns": False,
        "bf16": bf16,
        "bf16_full_eval": bf16 and has_eval,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "ddp_backend": "nccl",
        "ddp_find_unused_parameters": False,
        # NO_SHARD (ZeRO-0) replicates parameters on every rank rather than
        # sharding them. This is required for Muon: its Newton-Schulz iteration
        # orthogonalises the whole 2D weight matrix, but under full_shard each
        # rank only holds an arbitrary contiguous slice of the row-major
        # flattening, which is not a submatrix — reshaping it back to the
        # parameter's 2D shape produces meaningless updates (and crashes outright
        # when a parameter straddles a shard boundary). ESM-C 600m is ~575M
        # parameters, so replicated params + grads + Muon momentum is ~7GB/GPU.
        "fsdp": "no_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": ["UnifiedTransformerBlock"],
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "use_orig_params": True,
        },
    }


def instantiate_training_arguments(training_arguments_cls: type[Any], **kwargs: Any) -> Any:
    supported = inspect.signature(training_arguments_cls.__init__).parameters
    filtered = {key: value for key, value in kwargs.items() if key in supported}

    if "eval_strategy" in supported and "evaluation_strategy" in supported:
        filtered.pop("evaluation_strategy", None)
    elif "eval_strategy" in supported:
        filtered.pop("evaluation_strategy", None)
    elif "evaluation_strategy" in supported:
        filtered.pop("eval_strategy", None)

    return training_arguments_cls(**filtered)


def resolve_train_validation_records(
    records: list[SequenceRecord],
    *,
    cluster_assignments_csv: Path | None,
    cluster_identity_for_split: float | None,
    train_fraction: float,
    random_seed: int,
    split_strategy: str,
) -> tuple[list[SequenceRecord], list[SequenceRecord], dict[str, object]]:
    split_summary_extra: dict[str, object] = {
        "cluster_based_split": False,
        "random_seed": random_seed,
    }

    if split_strategy == "FULL":
        split_summary_extra["full_training"] = True
        return records, [], split_summary_extra

    if cluster_identity_for_split is not None:
        if cluster_assignments_csv is None:
            raise ValueError(
                "`--cluster-assignments-csv` is required when `--cluster-identity-for-split` is set."
            )
        return split_train_validation_by_cluster(
            records,
            cluster_assignments_csv=cluster_assignments_csv,
            identity_threshold=cluster_identity_for_split,
            train_fraction=train_fraction,
            random_seed=random_seed,
            split_strategy=split_strategy,
        )

    train_records, val_records = split_train_validation(
        records,
        train_fraction=train_fraction,
        random_seed=random_seed,
        split_strategy=split_strategy,
    )
    return train_records, val_records, split_summary_extra


# ---------------------------------------------------------------------------
# Custom builder for ESMC 6B model
# ---------------------------------------------------------------------------
def ESMC_6B_builder(device: torch.device | str = "cpu", use_flash_attn: bool = True):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from pathlib import Path
    import os
    import json
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers
    from filelock import FileLock

    token = _resolve_hf_token()

    repo_id = "biohub/esmc-6b-2024-12"
    
    # In distributed environments, ensure rank 0 downloads/checks first.
    # Note: torch.distributed.is_initialized() is False at this point because
    # model loading happens before the Trainer initializes the process group.
    # Therefore, we use a file-based lock for synchronization to avoid a race condition.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    
    import tempfile
    lock_path = Path(tempfile.gettempdir()) / "esmc_6b_download.lock"
    lock = FileLock(str(lock_path))
    
    if global_rank == 0:
        with lock:
            snapshot_path = Path(snapshot_download(
                repo_id=repo_id,
                token=token,
                allow_patterns=["*.json", "*.safetensors", "*.txt", ".gitattributes"]
            ))
    else:
        with lock:
            pass
        snapshot_path = Path(snapshot_download(
            repo_id=repo_id,
            token=token,
            allow_patterns=["*.json", "*.safetensors", "*.txt", ".gitattributes"],
            local_files_only=True
        ))

    with torch.device(device):
        model = ESMC(
            d_model=2560,
            n_heads=40,
            n_layers=80,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()

    index_file = snapshot_path / "model.safetensors.index.json"
    state_dict = {}
    if index_file.exists():
        with open(index_file) as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        shards = sorted(list(set(weight_map.values())))
        for shard in shards:
            shard_path = snapshot_path / shard
            shard_state = load_file(shard_path, device=str(device))
            for k, v in shard_state.items():
                if k.startswith("esmc."):
                    mapped_k = k[5:]
                elif k.startswith("lm_head."):
                    mapped_k = "sequence_head." + k[8:]
                else:
                    mapped_k = k
                state_dict[mapped_k] = v
    else:
        single_safetensors = snapshot_path / "model.safetensors"
        if single_safetensors.exists():
            shard_state = load_file(single_safetensors, device=str(device))
            for k, v in shard_state.items():
                if k.startswith("esmc."):
                    mapped_k = k[5:]
                elif k.startswith("lm_head."):
                    mapped_k = "sequence_head." + k[8:]
                else:
                    mapped_k = k
                state_dict[mapped_k] = v
        else:
            raise FileNotFoundError(f"Weights not found in {snapshot_path}")

    model.load_state_dict(state_dict)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        get_scheduler,
    )
    from esm.models.esmc import ESMC
    from esm.pretrained import register_local_model
    register_local_model("esmc_6b", ESMC_6B_builder)

    args = build_argument_parser().parse_args(argv)

    input_fasta = args.input_fasta.expanduser()
    output_dir = args.output_dir.expanduser()
    run_name = args.run_name
    model_type = args.model_type
    num_epochs = args.num_epochs
    train_fraction = args.train_fraction
    random_seed = args.random_seed
    mlm_probability = args.mlm_probability
    per_device_train_batch_size = args.train_batch_size
    per_device_eval_batch_size = args.eval_batch_size
    dataloader_num_workers = args.dataloader_num_workers
    tokenize_num_proc = args.tokenize_num_proc
    require_cuda = args.require_cuda
    use_bf16 = args.bf16
    gradient_accumulation_steps = args.gradient_accumulation_steps
    gradient_checkpointing = args.gradient_checkpointing
    min_sequence_length = args.min_sequence_length
    export_model_each_epoch = args.export_model_each_epoch
    truncate_to_model_limit = args.truncate_to_model_limit
    _run_mmseqs_cluster_precheck = args.run_mmseqs_cluster_precheck
    cluster_assignments_csv = (
        args.cluster_assignments_csv.expanduser() if args.cluster_assignments_csv else None
    )
    run_mode = args.run_mode
    run_config = resolve_run_configuration(args)
    cluster_identity_for_training = run_config["cluster_identity_for_training"]
    cluster_identity_for_split = run_config["cluster_identity_for_split"]
    split_strategy = run_config["split_strategy"]
    max_records = run_config["max_records"]
    max_val_records = args.max_val_records
    test_mode = args.test_mode
    resume_from_checkpoint = args.resume_from_checkpoint.expanduser() if args.resume_from_checkpoint else None

    if test_mode:
        _rank_zero_print("[test-mode] Model weights will NOT be saved at the end of training.")
    if export_model_each_epoch and test_mode:
        _rank_zero_print("[test-mode] Per-epoch exports are also disabled because trainer checkpoint saving is off.")
    if resume_from_checkpoint is not None:
        if not resume_from_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_from_checkpoint}")
        _rank_zero_print(f"[resume] Resuming from checkpoint: {resume_from_checkpoint}")
    if run_config["clustering_disabled"]:
        _rank_zero_print(
            "[info] run_mode=full-finetune: using all prepared sequences for training; "
            "eval split, clustering, and max-record downsampling are disabled."
        )

    ensure_cuda_available(torch, require_cuda=require_cuda)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    run_name = (
        run_name
        or f"{input_fasta.stem}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output_dir / run_name
    data_dir = run_dir / "data"
    reports_dir = run_dir / "reports"
    model_dir = run_dir / "model"
    for path in (data_dir, reports_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    training_parameters = log_training_parameters(
        model_type=model_type,
        num_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        bf16=use_bf16,
        mlm_probability=mlm_probability,
        train_fraction=train_fraction,
        random_seed=random_seed,
        split_strategy=split_strategy,
        max_records=max_records,
        dataloader_num_workers=dataloader_num_workers,
        tokenize_num_proc=tokenize_num_proc,
    )
    training_parameters["export_model_each_epoch"] = export_model_each_epoch
    training_parameters["run_mode"] = run_mode
    training_parameters["min_sequence_length"] = min_sequence_length
    training_parameters["max_val_records"] = max_val_records

    # -----------------------------------------------------------------------
    # Load ESM-C model and extract tokenizer
    # -----------------------------------------------------------------------
    # Compatibility patch: newer transformers (>=4.40) tries to set special-token
    # attributes via setattr() inside PreTrainedTokenizerBase.__init__, but
    # EsmSequenceTokenizer defines cls_token (and siblings) as read-only properties.
    # We add no-op setters so that setattr() silently succeeds instead of raising.
    try:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer as _EsmTok

        # Patch 1: fix _get_token to read from _special_tokens_map directly.
        # Some versions of the tokenizers library do NOT expose __getattr__ as a
        # callable method (only as the implicit dunder), so calling
        # self.__getattr__(name) explicitly raises AttributeError.
        _HARDCODED_SPECIAL_TOKENS = {
            "mask_token": "<mask>", "cls_token": "<cls>",
            "pad_token": "<pad>", "eos_token": "<eos>",
            "unk_token": "<unk>", "bos_token": "<cls>", "sep_token": "<eos>",
        }

        def _get_token_patched(self, token_name: str) -> str:
            # Use _special_tokens_map safely (may not exist on all builds).
            _private_map = getattr(self, "_special_tokens_map", None)
            token_str = _private_map.get(token_name) if _private_map is not None else None
            # Hardcoded last-resort fallback (safe in subprocesses; no recursion).
            if token_str is None:
                token_str = _HARDCODED_SPECIAL_TOKENS.get(token_name)
            assert isinstance(token_str, str), (
                f"EsmSequenceTokenizer._get_token: could not resolve {token_name!r}; "
                f"_special_tokens_map={_private_map!r}"
            )
            return token_str

        _EsmTok._get_token = _get_token_patched

        # Patch 2: add no-op setters so transformers' __init__ can call
        # self.mask_token = "..." without raising (read-only property guard).
        _SPECIAL_ATTRS = (
            "cls_token", "bos_token", "eos_token", "unk_token",
            "pad_token", "mask_token", "sep_token",
        )

        def _make_special_token_setter(attr_name: str):
            def _setter(self, v):
                if v is None:
                    return
                _private = getattr(self, "_special_tokens_map", None)
                if _private is not None:
                    _private[attr_name] = v
                # Also seed the hardcoded fallback dict for subprocess safety.
                _HARDCODED_SPECIAL_TOKENS[attr_name] = v
            return _setter

        for _attr in _SPECIAL_ATTRS:
            _prop = getattr(_EsmTok, _attr, None)
            if isinstance(_prop, property) and _prop.fset is None:
                setattr(
                    _EsmTok,
                    _attr,
                    property(_prop.fget, _make_special_token_setter(_attr), _prop.fdel, _prop.__doc__),
                )

        _rank_zero_print("[info] EsmSequenceTokenizer property patch applied (transformers compat).")
    except Exception as _patch_exc:
        _rank_zero_print(f"[warning] EsmSequenceTokenizer patch skipped: {_patch_exc}")

    _rank_zero_print(f"[info] Loading ESM-C model: {model_type} on {device}")
    esmc_client = ESMC.from_pretrained(model_type, device=device)
    # Convert model parameters to float32. This is crucial when training with FSDP mixed precision
    # because if the model is wrapped in bfloat16, accelerate will attempt to dynamically upcast
    # module._flat_param to float32. Due to a bug in accelerate's upcasting logic (where it only
    # replaces _flat_param.data and breaks storage sharing with the original parameter views),
    # this leads to a "Cannot writeback when the parameter shape changes" RuntimeError during the forward pass.
    # By wrapping the model in float32 and letting FSDP mixed precision handle bfloat16 casting,
    # we avoid the buggy dynamic upcasting path entirely.
    esmc_client = esmc_client.to(torch.float32)

    # The ESM-C SDK exposes a tokenizer that is HuggingFace-compatible
    # (EsmSequenceTokenizer inherits from PreTrainedTokenizerFast)
    tokenizer = esmc_client.tokenizer

    # EsmSequenceTokenizer declares model_input_names = ["sequence_tokens", ...]
    # but the standard HF DataCollatorForLanguageModeling works with "input_ids".
    # Override it so tokenizer.pad() and the collator agree on the key name.
    tokenizer.model_input_names = ["input_ids", "attention_mask"]

    # Determine vocabulary size and max length from tokenizer/client
    vocab_size: int = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size
    tokenizer_model_max_length = int(getattr(tokenizer, "model_max_length", ESMC_MAX_RESIDUE_LENGTH + 2))
    tokenizer_max_length: int = min(tokenizer_model_max_length, ESMC_MAX_RESIDUE_LENGTH + 2)
    tokenizer.model_max_length = tokenizer_max_length
    max_residue_length = tokenizer_max_length - 2  # account for BOS/EOS

    _rank_zero_print(
        f"[info] vocab_size={vocab_size}, "
        f"tokenizer.model_max_length={tokenizer_model_max_length}, "
        f"effective tokenizer_max_length={tokenizer_max_length} tokens, "
        f"max_residue_length={max_residue_length} AA"
    )
    if tokenizer_max_length < 2048:
        _rank_zero_print(
            f"[warning] tokenizer_max_length={tokenizer_max_length} is less than 2048 — "
            f"sequences longer than {max_residue_length} AA will be truncated. "
            f"Expected ESMC_MAX_RESIDUE_LENGTH=2046+2=2048."
        )
    else:
        _rank_zero_print(f"[info] Max sequence length OK: {max_residue_length} AA ({tokenizer_max_length} tokens incl. BOS/EOS).")

    # Ensure tokenizer has mask_token_id so DataCollatorForLanguageModeling works.
    # After the _get_token patch above this should always succeed; the fallback
    # handles any edge case where _special_tokens_map wasn't seeded in time.
    try:
        _mask_id = tokenizer.mask_token_id
    except (AssertionError, AttributeError, KeyError, TypeError):
        _mask_id = None

    if _mask_id is None:
        # Try to resolve the mask token directly from the vocab and seed the map.
        _mask_str = "<mask>"
        _mask_id_direct = tokenizer.convert_tokens_to_ids(_mask_str)
        if isinstance(_mask_id_direct, int):
            _private = getattr(tokenizer, "_special_tokens_map", None)
            if _private is not None:
                _private["mask_token"] = _mask_str
            _rank_zero_print(f"[info] Seeded mask_token = '{_mask_str}' (id={_mask_id_direct}).")
        else:
            raise RuntimeError(
                "ESMC tokenizer does not expose mask_token_id and '<mask>' is not in "
                "its vocabulary. Cannot use DataCollatorForLanguageModeling."
            )

    # Wrap ESMC for MLM training
    model = ESMCMLMWrapper(esmc_client, vocab_size=vocab_size, max_position_embeddings=tokenizer_max_length)
    num_parameters = sum(p.numel() for p in model.parameters())
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(f"[Rank {global_rank}/Local {local_rank}] Parameters count: {len(list(model.parameters()))}, Total numel: {num_parameters:,}")
    _rank_zero_print(f"[info] Parameters: {num_parameters:,} ({num_parameters / 1e6:.1f}M)")

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    clean_fasta = data_dir / f"{input_fasta.stem}.clean_input.fasta"
    training_fasta = data_dir / f"{input_fasta.stem}.training_input.fasta"
    validation_fasta = data_dir / f"{input_fasta.stem}.validation_input.fasta"
    metadata_csv = data_dir / f"{input_fasta.stem}.sequence_metadata.csv"
    config_json = reports_dir / "run_report.json"
    preprocessing_cache = data_dir / "preprocessed_inputs.json"
    preprocessing_failure = data_dir / "preprocessed_inputs.failed"
    train_fasta_path, val_fasta_path, train_record_count, val_record_count, prep_stats, clustering_summary, split_summary_extra = _prepare_training_inputs_once(
        cache_path=preprocessing_cache,
        failure_path=preprocessing_failure,
        input_fasta=input_fasta,
        cluster_assignments_csv=cluster_assignments_csv,
        cluster_identity_for_training=cluster_identity_for_training,
        cluster_identity_for_split=cluster_identity_for_split,
        max_residue_length=max_residue_length,
        min_sequence_length=min_sequence_length,
        truncate_to_model_limit=truncate_to_model_limit,
        max_records=max_records,
        max_val_records=max_val_records,
        random_seed=random_seed,
        train_fraction=train_fraction,
        split_strategy=split_strategy,
        clean_fasta=clean_fasta,
        training_fasta=training_fasta,
        validation_fasta=validation_fasta,
        metadata_csv=metadata_csv,
        config_json=config_json,
        reports_dir=reports_dir,
        training_parameters=training_parameters,
        model_type=model_type,
        num_epochs=num_epochs,
        run_mode=run_mode,
        export_model_each_epoch=export_model_each_epoch,
    )

    train_dataset = StreamingTokenizedFastaDataset(
        train_fasta_path,
        tokenizer,
        tokenizer_max_length,
        num_records=train_record_count,
        test_mode=args.test,
    )
    val_dataset = (
        StreamingTokenizedFastaDataset(
            val_fasta_path,
            tokenizer,
            tokenizer_max_length,
            num_records=val_record_count,
            test_mode=args.test,
        )
        if val_record_count > 0 and val_fasta_path.exists()
        else None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        return_tensors="pt",
        mlm_probability=mlm_probability,
    )

    # -----------------------------------------------------------------------
    # Training arguments
    # -----------------------------------------------------------------------
    training_kwargs = build_training_arguments_kwargs(
        output_dir=model_dir,
        num_epochs=num_epochs,
        train_batch_size=per_device_train_batch_size,
        eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        random_seed=random_seed,
        has_eval=val_dataset is not None,
        bf16=use_bf16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        export_model_each_epoch=export_model_each_epoch,
    )

    test_checkpoint_save_and_cleanup = args.test_checkpoint_save_and_cleanup

    if test_mode and not test_checkpoint_save_and_cleanup:
        training_kwargs["save_strategy"] = "no"
        training_kwargs["save_total_limit"] = 1
    training_args = instantiate_training_arguments(TrainingArguments, **training_kwargs)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Optimiser setup
    # -----------------------------------------------------------------------
    class MagmaTrainer(Trainer):

        def create_optimizer(self):
            if self.optimizer is None:
                optimizer_model = self.model
                magma_params = []
                dense_params = []
                for name, parameter in optimizer_model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if _is_magma_target_parameter_name(name):
                        magma_params.append(parameter)
                    else:
                        dense_params.append(parameter)

                _rank_zero_print(
                    f"[info] Optimiser parameter groups: {len(magma_params)} Magma-masked "
                    f"(attn/ffn), {len(dense_params)} dense."
                )
                if not magma_params:
                    _rank_zero_print(
                        "[warning] No parameters matched the Magma selector, so Magma masking "
                        "is inert. Check parameter naming against "
                        "_is_magma_target_parameter_name()."
                    )

                base_lr = 1e-4
                base_optimiser = MuonAdamW(
                    [
                        {"params": magma_params, "magma_enabled": True},
                        {"params": dense_params, "magma_enabled": False},
                    ],
                    lr=base_lr,
                    muon_lr=0.02,
                )
                
                # Instantiation with base tracking metrics
                self.optimizer = MagmaOptimiser(base_optimiser, tau=2.0, p=0.5, mask_seed=random_seed)
                for group in self.optimizer.param_groups:
                    group["initial_lr"] = base_lr
                    
            return self.optimizer

        def create_scheduler(self, num_training_steps: int, optimizer=None):
            if self.lr_scheduler is None:
                # Derive warmup from the step count the Trainer actually computes
                # (per-process, world-size and iterable-length aware) so the linear
                # decay reaches zero exactly at the end of training.
                warmup_steps = int(0.1 * num_training_steps)
                self.lr_scheduler = get_scheduler(
                    "linear",
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps,
                )
            return self.lr_scheduler

        def _save_optimizer_and_scheduler(self, output_dir: str):
            """Intercept saving logic to prevent FSDP dimension errors during preemption or tied weight issues."""
            # Find the signal callback in the trainer's callback handler
            signal_callback = next(
                (cb for cb in self.callback_handler.callbacks if isinstance(cb, GracefulStopOnSignalCallback)), 
                None
            )
            
            if signal_callback and signal_callback.is_preempted:
                _rank_zero_print(
                    "[warning] Preemption triggered. Skipping optimizer and scheduler state serialization "
                    "to bypass FSDP unflattening constraints. Model weights will still save cleanly."
                )
                return  # Safe exit, prevents the split_with_sizes RuntimeError
            
            # Standard execution path for normal epoch ends or final saves
            try:
                return super()._save_optimizer_and_scheduler(output_dir)
            except Exception as exc:
                import traceback
                _rank_zero_print(
                    f"[warning] Could not save optimizer/scheduler state "
                    f"({type(exc).__name__}: {exc}). Resuming from this checkpoint will "
                    "restart optimizer momentum from zero; model weights still save cleanly."
                )
                _rank_zero_print(traceback.format_exc())

    # ─── Monkey patch FSDP.fsdp_modules to bypass accelerate's buggy upcasting ───
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if not hasattr(FSDP, "_orig_fsdp_modules"):
        FSDP._orig_fsdp_modules = FSDP.fsdp_modules
    FSDP.fsdp_modules = classmethod(lambda cls, module: [])
    _rank_zero_print("[info] Monkey-patched FSDP.fsdp_modules to bypass accelerate upcasting.")

    trainer = MagmaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # ─── Overriding FSDP plugin mixed precision policy ───
    if getattr(trainer.accelerator.state, "fsdp_plugin", None) is not None:
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        _rank_zero_print("[info] Overriding FSDP plugin mixed precision policy to use float32 parameters.")
        from torch.distributed.fsdp import MixedPrecision
        fsdp_plugin.mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    trainer.add_callback(GracefulStopOnSignalCallback())
    if export_model_each_epoch and (not test_mode or test_checkpoint_save_and_cleanup):
        trainer.add_callback(ExportNativeESMCWeightsCallback(trainer))
    if test_mode and test_checkpoint_save_and_cleanup:
        trainer.add_callback(TestCheckpointCleanupCallback())

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------


    initial_eval_metrics: dict[str, float] | None = None
    initial_eval_epoch = load_checkpoint_epoch(resume_from_checkpoint)
    if val_dataset is not None:
        initial_eval_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
        if initial_eval_metrics and "eval_loss" in initial_eval_metrics:
            _rank_zero_print(f"\n\n{'='*60}\n[info] EXACT INITIAL EVAL LOSS: {initial_eval_metrics['eval_loss']:.10f}\n{'='*60}\n\n")
        save_initial_eval_artifacts(
            initial_eval_metrics,
            reports_dir,
            initial_epoch=initial_eval_epoch,
            model_label=model_type,
        )

    train_result = trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
    )

    if not test_mode or test_checkpoint_save_and_cleanup:
        # Save the wrapped model; the inner ESMC weights are accessible via model.esmc
        final_ckpt_dir = model_dir / "final_checkpoint"
        trainer.save_model(str(final_ckpt_dir))
        # Also save native ESM-C weights for easy reloading. The FSDP full-state
        # gather inside the helper is a collective, so every rank must call it;
        # the helper materialises and writes the weights on rank 0 only.
        esmc_save_path = final_ckpt_dir / "esmc_weights.pt"
        _save_native_esmc_weights(trainer, esmc_save_path)
        if trainer.is_world_process_zero():
            _rank_zero_print(f"[info] ESM-C native weights saved to {esmc_save_path}")
            if test_mode and test_checkpoint_save_and_cleanup:
                import shutil
                if final_ckpt_dir.exists():
                    shutil.rmtree(final_ckpt_dir)
                _rank_zero_print(
                    f"[test-mode] Deleted final checkpoint directory {final_ckpt_dir} after write verification."
                )
    else:
        _rank_zero_print("[test-mode] Skipping trainer.save_model.")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    exported_log_history = combine_log_history(initial_eval_metrics, trainer.state.log_history)
    save_training_artifacts(exported_log_history, reports_dir, model_label=model_type)
    save_json(
        {
            "model_type": model_type,
            "model_num_parameters": num_parameters,
            "model_size_millions": round(num_parameters / 1e6, 2),
            "train_records": train_record_count,
            "validation_records": val_record_count,
            "initial_validation_metrics": initial_eval_metrics,
            "training_metrics": metrics,
            "clustering_used": clustering_summary["used"],
            "cluster_identity_for_training": clustering_summary["identity_threshold"],
            "cluster_identity_for_split": cluster_identity_for_split,
            "run_mode": run_mode,
            "split_strategy": split_strategy,
            "training_parameters": training_parameters,
            "export_model_each_epoch": export_model_each_epoch,
        },
        reports_dir / "training_summary.json",
    )

    # Synchronize all processes to avoid distributed teardown errors (e.g. c10d recvVector)
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
