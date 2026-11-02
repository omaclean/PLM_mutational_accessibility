# SARS-CoV-2 Fine-Tuned Checkpoint: magma_ESMC_600M_99_95perc

This directory packages all configuration, hyperparameters, tensor inventories, and metadata for the primary **SARS-CoV-2 Spike** protein language model fine-tune evaluated across the paper and revision analyses.

## Model Summary

* **Model Family:** ESM-C (EvolutionaryScale)
* **Base Architecture:** `esmc_600m` (UnifiedTransformerBlock layers, rotary embeddings)
* **Total Parameters:** 575,036,992
* **Total Tensors:** 368
* **Target Protein:** SARS-CoV-2 Spike glycoprotein
* **Checkpoint Tag:** `magma_ESMC_600M_99_95perc`

## Clustering & Sequence Split Strategy

The `99_95perc` tag denotes the dual MMseqs2 clustering thresholds used during sequence preprocessing:
* **Training Identity Threshold (`--cluster-identity-for-training`):** `0.99` (99% sequence identity clustering to collapse redundant near-identical isolates and reduce overrepresentation of dominant spike haplotypes).
* **Held-Out Validation Split Threshold (`--cluster-identity-for-split`):** `0.95` (95% sequence identity clustering ensuring validation sequences are genuinely distinct from the training corpus).

## Training Hyperparameters

Read directly from the bundled `training_args.bin` (and transcribed in `training_args.json`):
* **Optimizer:** AdamW (`adamw_torch`)
* **Learning Rate:** `5e-05` (linear schedule)
* **Epochs:** 40
* **Per-device Train Batch Size:** 12
* **Per-device Eval Batch Size:** 2
* **Gradient Accumulation Steps:** 4
* **Precision:** bfloat16 (`bf16=True`, `bf16_full_eval=True`)
* **Random Seed:** 13
* **Weight Decay:** 0.0
* **Max Gradient Norm:** 1.0
* **Save Strategy:** Per epoch (`save_safetensors=True`, `save_total_limit=5`)

## Checkpoint Files & Provenance

| File / Directory | Size | MD5 / Format | Role |
|---|---|---|---|
| `training_args.bin` | 5,176 bytes | `8e6646afaf45f40f5a67e63beaa8b1e7` | Serialized Hugging Face `TrainingArguments` object holding full runtime parameters. Bundled directly in repo. |
| `training_args.json` | 4,065 bytes | JSON text | Human-readable JSON export of `training_args.bin`. |
| `checkpoint_metadata.json` | 60,927 bytes | JSON text | Comprehensive metadata catalog including parameter counts, safetensors header info, and 368 tensor shapes/dtypes. |
| `earlycehck/` | directory | Directory | Early checkpoint validation directory preserved from training run. |
| `model.safetensors` | 1,150,117,424 bytes | MD5: `d6e234f47ab0232a4e84577a7c9a7d7e`<br>SHA256: `519de0fb9e224447c8725ac9686ed306be8f1c110c0da985efd28782c0437301` | Full safetensors weight file (~1.15 GB). Located in external weights store (`my_SC2_finetunes/magma_ESMC_600M_99_95perc/`). |
| `esmc_weights.pt` | 1,150,225,302 bytes | MD5: `404a1097cba81fde4926154ce427f94c`<br>SHA256: `1ab972b65da412ceebc033a8508ab66b7e56586bbeb6486c36d04ff0e80e2dc7` | Native PyTorch checkpoint state dict (~1.15 GB). Located in external weights store (`my_SC2_finetunes/magma_ESMC_600M_99_95perc/`). |

## Role in the Revision Figures

This fine-tuned ESM-C 600M checkpoint is the specific model evaluated in:
* **Figure SA (Spike):** Compared against base untuned ESM-C 600M across 1, 2, and 3 nucleotide step replacement strata.
* **Figure 4 (SARS-CoV-2 Epistasis & Direction):** Scored on Spike background mutations to measure directional capture.
* **Figure SJ & SM:** Evaluated for DMS concordance and within-stratum predictive accuracy.
* **Symmetry Tests:** Analyzed in the directional versus symmetric substitution battery (`SYMMETRY_TESTS.md`).
