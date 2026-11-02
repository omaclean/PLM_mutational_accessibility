# Protein Language Model Fine-Tuning & Evolutionary Simulation Engine

This directory packages the custom fine-tuning code, distributed optimization infrastructure, Slurm configurations, and evolutionary simulation pipelines used throughout the study.

The primary fine-tuning script is [`evotune_gradient_MAGMA_ESMC.py`](evotune_gradient_MAGMA_ESMC.py) (also accessible directly at the repository root as [`evotune_gradient_MAGMA_ESMC.py`](../evotune_gradient_MAGMA_ESMC.py)).

---

## 1. Architecture & The Muon / Magma Optimizer

Fine-tuning ESM-C (and ESM-2) models under evolutionary divergence presents unique optimization challenges:
* Standard AdamW can over-smooth representations or drift when fine-tuning on highly biased viral clades.
* **Muon (Momentum Orthogonalized by Newton-Schulz):** Muon updates 2D weight matrices by orthogonalizing momentum buffers via Newton-Schulz matrix iterations.
* **Magma Integration:** Combines Muon updates for internal 2D transformer projection matrices with AdamW for 1D biases, layernorms, and embedding matrices.

### Distributed Training & FSDP NO_SHARD

Because Muon's Newton-Schulz iteration requires the **complete 2D shape** of each weight matrix, standard parameter-sharding strategies (such as ZeRO-3 / FSDP `FULL_SHARD`) cannot be used directly:
* Sharding slices parameters into 1D chunks across GPUs, destroying the 2D matrix structure needed for orthogonalization.
* `evotune_gradient_MAGMA_ESMC.py` uses PyTorch Fully Sharded Data Parallel in **`no_shard auto_wrap` mode** (ZeRO-0 wrapping `UnifiedTransformerBlock`).
* Each GPU replicates the model parameters (~575M parameters for ESM-C 600M, requiring ~7 GB VRAM per GPU for weights, gradients, and momentum), while data parallelism and gradient synchronization scale efficiently across devices.

---

## 2. Execution Modes & CLI Reference

`evotune_gradient_MAGMA_ESMC.py` exposes two primary execution presets via `--run-mode`:

### Mode A: `standard` (Default)
Includes rigorous data preprocessing and validation hold-out:
* Computes or ingests sequence cluster assignments via MMseqs2.
* Downsamples or clusters training sequences using `--cluster-identity-for-training` (e.g. `0.99` for SARS-CoV-2 spike, `0.90` for diverse viral sets).
* Partitions held-out validation sequences at `--cluster-identity-for-split` (e.g. `0.95` for SARS-CoV-2, `0.70` for viral sets) ensuring validation sequences are genuinely divergent from the training pool.
* Logs training and validation metrics at each epoch, and optionally exports native ESM-C weights per epoch (`--export-model-each-epoch true`).

### Mode B: `full-finetune`
Designed for production fine-tunes on pre-filtered or curated sets:
* Disables evaluation splitting, clustering downsampling, and sequence holdouts (`--split-strategy FULL`).
* Allocates 100% of prepared FASTA sequences to training.
* Maximizes training data efficiency while preserving gradient accumulation and optimizer schedules.

### Common Arguments

```bash
python evotune_gradient_MAGMA_ESMC.py \
    --input-fasta <path/to/sequences.fasta> \
    --output-dir <path/to/output_dir> \
    --run-name <run_identifier> \
    --model-type esmc_600m \
    --run-mode standard \
    --num-epochs 40 \
    --train-batch-size 8 \
    --gradient-accumulation-steps 4 \
    --cluster-identity-for-training 0.99 \
    --cluster-identity-for-split 0.95 \
    --bf16 true \
    --gradient-checkpointing true
```

---

## 3. SARS-CoV-2 Spike Fine-Tuning (`magma_ESMC_600M_99_95perc`)

The primary SARS-CoV-2 Spike model analyzed in this paper (evaluated in Figure SA, Figure 4, Figure SJ, Figure SM, and the symmetry battery) was trained using this engine.

* **Base Model:** ESM-C 600M (`esmc_600m`, 575M parameters).
* **Dataset:** Curated SARS-CoV-2 Spike sequences preprocessed at 99% training cluster identity and 95% validation split identity.
* **Hyperparameters:** 40 epochs, batch size 12 per device, gradient accumulation 4, learning rate 5e-5, AdamW, bfloat16, random seed 13.
* **Slurm Recipe:** [`slurm/evotune_esmc_full_magma.sbatch`](slurm/evotune_esmc_full_magma.sbatch) launched with `--species sc2 --model-type esmc_600m`.
* **Checkpoint & Metadata:** Full parameter inventories, training arguments, and checksums are cataloged in [`checkpoints/magma_ESMC_600M_99_95perc/`](checkpoints/magma_ESMC_600M_99_95perc/).

---

## 4. End-to-End Simulation Control Pipeline (`output/sN_simulation_control.png`)

Figure SN (**Reviewer 2's Simulation Control**) demonstrates that PLMs specifically learn the nucleotide mutation spectrum of their evolutionary corpus rather than generic mutational accessibility, using matched in silico evolutionary simulations where the underlying protein fitness landscape is identical and known.

The complete pipeline to generate `output/sN_simulation_control.png` proceeds in five stages:

```
[Synthetic Founder & Shared Fitness Landscape (beta=2.0)]
                           │
       ┌───────────────────┼───────────────────┬───────────────────┐
       ▼                   ▼                   ▼                   ▼
  Human Germline      SARS-CoV-2          Flu H3N2             JC69 Flat
  (v3_star90.faa)  (v3_sc2_star90.faa) (v3_h3n2_star90.faa) (v3_jc69_star90.faa)
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
 ESM-C 600M FT       ESM-C 600M FT       ESM-C 600M FT       ESM-C 600M FT
 (evotune_grad)      (evotune_grad)      (evotune_grad)      (evotune_grad)
       │                   │                   │                   │
       └───────────────────┼───────────────────┴───────────────────┘
                           ▼
     Founder Scoring (Base + 4 Fine-Tuned Arms, 13,300 replacements)
                           ▼
          results/sim_v3_star90/replacement_scores.tsv.gz
                           ▼
       Directional & Debiasing Symmetry Battery (tools/run_symmetry_sim.py)
                           ▼
        Publication Figure: python figures/make_sN_simulation_control.py
                           ▼
            output/sN_simulation_control.{png, pdf, tsv, md}
```

### Stage 1: Evolutionary Simulation under Shared Fitness Landscape
Sequences (10,000 per arm, length 700 aa) are evolved from a single synthetic founder under one invariant site-specific fitness landscape ($\beta = 2.0$), varying **only** the generating nucleotide mutation matrix:
1. **Human Germline Spectrum:** [`data/spectra/human.csv`](../data/spectra/human.csv)
2. **SARS-CoV-2 Spectrum:** [`data/spectra/sars_cov_2.csv`](../data/spectra/sars_cov_2.csv)
3. **Influenza A H3N2 Spectrum:** [`data/spectra/influenza_h3n2.csv`](../data/spectra/influenza_h3n2.csv)
4. **JC69 Flat Control:** [`data/spectra/jc69.csv`](../data/spectra/jc69.csv) (neutral negative control where all nucleotide substitutions have equal probability)

*Runner Script:* [`simulation/simulate_v2_phylogenies.py`](simulation/simulate_v2_phylogenies.py) calibrated against [`simulation/dms_calibration_targets.json`](simulation/dms_calibration_targets.json).

### Stage 2: ESM-C 600M Fine-Tuning across Evolutionary Arms
Each evolutionary arm is fine-tuned from raw ESM-C 600M using [`evotune_gradient_MAGMA_ESMC.py`](evotune_gradient_MAGMA_ESMC.py):
* 10 epochs, per-device batch size 8, gradient accumulation 4, bfloat16, gradient checkpointing.
* Coordinated via [`simulation/run_gpu_arms.sh`](simulation/run_gpu_arms.sh).

### Stage 3: Founder Scoring & Tabulation
The synthetic founder sequence is scored across all 13,300 single-amino-acid replacements ($700\text{ sites} \times 19\text{ replacements}$) using:
* Untuned base ESM-C 600M (`plm_base`)
* The four fine-tuned arms (`plm_human_ft`, `plm_sc2_ft`, `plm_h3n2_ft`, `plm_jc69_ft`)
* True fitness landscape values (`ground_truth`) and PSSMs
* Shipped table: [`results/sim_v3_star90/replacement_scores.tsv.gz`](../results/sim_v3_star90/replacement_scores.tsv.gz)

### Stage 4: Symmetry Battery & Debiasing Sweeps
The symmetric-versus-directional battery tests whether fine-tuning specifically acquires directional accessibility for the target spectrum:
```bash
python tools/run_symmetry_sim.py --spectrum sars_cov_2
python tools/run_symmetry_sim.py --spectrum human --scale 1.2e-8
python tools/run_symmetry_sim.py --spectrum influenza_h3n2
python tools/run_symmetry_sim.py --spectrum jc69
```
Outputs are written to [`results/sim_v3_star90/symmetry_*.tsv.gz`](../results/sim_v3_star90/).

### Stage 5: Generating Figure SN
To render the final publication-ready four-panel figure:
```bash
python figures/make_sN_simulation_control.py
```
This produces:
* [`output/sN_simulation_control.png`](../output/sN_simulation_control.png): High-resolution raster figure.
* [`output/sN_simulation_control.pdf`](../output/sN_simulation_control.pdf): Vector publication graphic.
* [`output/sN_simulation_control_values.tsv`](../output/sN_simulation_control_values.tsv): Exactly plotted point estimates and confidence intervals.
* [`output/sN_simulation_control_caption.md`](../output/sN_simulation_control_caption.md): Formatted caption text.
* [`output/table_sN_simulation_control.tsv`](../output/table_sN_simulation_control.tsv): Supplementary summary table.

---

## 5. File Inventory

| Path | Description |
|---|---|
| [`evotune_gradient_MAGMA_ESMC.py`](evotune_gradient_MAGMA_ESMC.py) | Main fine-tuning engine for ESM-C with Muon/Magma and FSDP NO_SHARD. |
| [`functions.py`](functions.py) | Supporting data loaders, clustering wrappers, FASTA parsers, and evaluation metrics. |
| [`mmseqs_cluster_table.py`](mmseqs_cluster_table.py) | Automated MMseqs2 clustering and sequence partition generator. |
| [`slurm/evotune_esmc_full_magma.sbatch`](slurm/evotune_esmc_full_magma.sbatch) | Slurm batch runner supporting multiple species presets and distributed multi-GPU nodes. |
| [`slurm/evotune_esmc_test_MAGMA.sbatch`](slurm/evotune_esmc_test_MAGMA.sbatch) | Test launcher for quick verification of distributed checkpointing and pipeline execution. |
| [`simulation/simulate_v2_phylogenies.py`](simulation/simulate_v2_phylogenies.py) | In silico evolutionary sequence simulator for star and branching phylogenies under DMS calibration. |
| [`simulation/run_gpu_arms.sh`](simulation/run_gpu_arms.sh) | Orchestrator for the 4-arm simulation control fine-tuning and founder scoring queue. |
| [`simulation/dms_calibration_targets.json`](simulation/dms_calibration_targets.json) | Calibration specifications matching simulated divergence to empirical DMS preferences. |
| [`checkpoints/magma_ESMC_600M_99_95perc/`](checkpoints/magma_ESMC_600M_99_95perc/) | Full configuration, arguments (`training_args.bin`), and metadata for the paper's Spike model. |
