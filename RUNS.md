# How the upstream tables were produced

> ## ⚠️ `N145S` was a phantom mutation in Figure 1B/C — FIXED 20 Oct 2026
>
> **The J.2 → K transition is NINE replacements, not ten:**
> `K2N`, `T135K`, `S144N`, `N158D`, `I160K`, `Q173R`, `K189R`, `T328A`, `HA2:S49N`.
> Residue 145 is Serine in J.2, J.2.4 and K alike, and in the A/Aichi/2/1968 H3-numbering
> reference. There is no change at 145 on that path, in nucleotide or amino acid.
>
> **Root cause.** The panel diffed a *stored* results table's `ref_aa` against a *live*
> reference FASTA. The `J.2_int` root was corrected at precursor codon 161 (`aat` → `agt`)
> on 15 Jun 2026 and the May tables were never rebuilt, so the 2 Jul 2026 render compared a
> stale J.2 against a current K and emitted both `S144N` and `N145S`. Nothing warned.
>
> **Fix.** `guides/composites/fig01.yaml` places the fine-tuned model (`ESM2_650M_HA80`)
> render (`data/frozen_panels/fig1bc_ranked_mutations_finetuned_esm2_ha80.pdf`), matching the
> Results text (HA2:S49N highest at rank 7, T328A lowest at rank 578); `output/fig01_composite.pdf`
> and `fig01_composite.png` carry exactly the canonical nine. The 2 Jul file is kept as
> `data/frozen_panels/fig1bc_ranked_mutations.pdf` **for provenance only — do not place it.**
>
> **Standing risk this exposed:** re-plotting an old table against newer roots can silently
> invent or erase replacements. Full account in `LINEAGE_DEFINITIONS.md` section 6.


The figure engine reads derived tables only (`results/`). This file records how the
expensive ones were made, because the command lines were never written down.
Nothing here is executed by the figure scripts or the tests.

## The 101-point alpha sweep (Figure 3, S2)

`results/h3n2_13lineage/alpha_sweep_by_lineage.tsv.gz` comes from
`alpha_sweep_fit_metrics_BY_LINEAGE.tsv` in the exploration archive's
`results/nextstrain_13lin_dense/`.

**Status: reconstructed, not recovered.** No shell history or script recorded the
invocation. The run did write `run_manifest.json` — the *parsed* arguments — and the
command below is the invocation that yields exactly those arguments. It has not been
re-run (it needs a GPU, the model checkpoint and GISAID-derived alignments), so treat
it as the best-supported equivalent, not a verified replay.

```bash
python scripts/run_mutational_accessibility.py \
  --analysis-mode MONTHLY_GUIDE \
  --guide-path Sequences/IAV_lineage_guide_nextstrain.csv \
  --mutation-model H3N2 \
  --output-dir results/nextstrain_13lin_dense \
  --alpha-start -1.0 --alpha-stop 1.0 --alpha-step 0.02 \
  --model-tag ESM2_650M_HA80 \
  --base-model esm2_t33_650M_UR50D --model-layer 33 \
  --checkpoint-dir <directory holding the fine-tuned ESM2-HA80 model.safetensors>
```

Every other argument in the manifest equals the script's default
(`--filter-fixed-mutations`, `--min-obs-count 2`, `--rolling-identity-window 30`,
`--alignment-verify-max-cols 100`, `--observed-mutation-selection last`,
`--scatter-alphas -1,0,1`, no `--force-recompute-plm`). The raw (untuned) model
appears in the tables as `<model-tag>_raw`; it is scored in the same invocation.

What the manifest does **not** settle:

* The manifest is dated 10 Sep 2026 but the table is dated 5 Oct 2026. The table was
  re-derived after the AUROC clamp fix (`site_logistic_auroc` is now read from the
  score, not from the slope-constrained logistic fit). The run directory holds a
  `plm_cache/`, so the rebuild was probably done from cached PLM scores without
  re-running the model, but that is an inference: the call that rewrote the table is
  not recorded anywhere.
* The lineage alignments named in the manifest are GISAID-derived and are not
  redistributed. The guide shipped here (`guides/h3n2_13lineage.yaml`) deliberately
  has no `alignment` entries.

## Simulation arms (GPU)

Everything for the simulated sequences lives in the archive's `dev/simulated_phylogenies/` (human and
SARS-CoV-2 arms, fine-tuned earlier) and, for the new arms, `tmp_explore/gpu_sim_arms/`.

* Simulator: `scripts/simulate_v2_phylogenies.py --mode star --matrix <spectrum.csv> --calibration
  dms_calibration_targets.json --beta 2.0 --star-branch-length 33.0 --n-sequences 10000 --length-aa 700
  --seed 20260924`. The flat-spectrum arm used `data/spectra/jc69.csv`; its founder is byte-identical to the
  other arms (md5 43f02a48...). Branch length means accepted amino-acid substitutions per site, so divergence
  is comparable across spectra.
* Fine-tune: `sc2_esmc_evotune/evotune_esmc_magma.py` under `torchrun --nproc-per-node 1`, ESM-C 600M,
  10 epochs, batch 8, accumulation 4, bf16, gradient checkpointing. About 70 minutes per arm on a V100S.
* Scoring: `slurm_mutation_acesiblity.sbatch.sh` with a `--guide-path` whose `reference` is the **founder**
  (the earlier scoring used a tip representative).
* Queue: `tmp_explore/gpu_sim_arms/run_gpu_arms.sh`, run on `gpu1` (an idle V100S on the same filesystem),
  idempotent. Launched 9 Oct 2026 15:38; log `tmp_explore/gpu_sim_arms/logs/queue.log`.
  **Finished 10 Oct 2026 17:13, every step `exit 0`** — both fine-tunes and all four founder
  scorings. Each fine-tune took about **11 h 45 m**, not the ~70 minutes the earlier shared card
  suggested; the estimate above is for the older hardware and should not be used to plan a new
  arm. The outputs have not been analysed yet (see `SYMMETRY_TESTS.md` §8).

## Masked marginals for the DMS figures (GPU), 19–20 Oct 2026

`results/h3n2_13lineage/dms_scores_masked.tsv.gz` and `masked_dms_curves.tsv.gz` come from one
forward pass **per site**, with that site replaced by the mask token, on each of the four DMS
backbones and each of the two models scoring it. Why the estimator was changed at all is in
`mutacc/masked_dms.py`; what it moved is in `docs/notes/MASKED_DMS_PROGRESS.md`.

The scripts take their paths from the environment, because the weights and the sequence
repository are outside this repo (`dev/masked_dms/_env.py` lists the variables and what to
point them at):

```bash
export PLM_REPO=<...>/PLM_SARS-CoV-2
export PLM_MODEL_DIR=<...>/model_weights_topublish
export SC2_FINETUNE=<...>/magma_ESMC_600M_99_95perc

# influenza backbones, ESM-2 via transformers, in the plm_sars environment
python dev/masked_dms/embed_masked_profile.py --dataset MA22
python dev/masked_dms/embed_masked_profile.py --dataset Perth09
# SARS-CoV-2 backbones, ESM-C via EvolutionaryScale's esm, in plm_entropy
python dev/masked_dms/embed_masked_profile_esmc.py

# then, on any machine: the shipped tables
python tools/build_masked_dms_scores.py          # dms_scores_masked
python dev/masked_dms/build_masked_tables.py     # masked_dms_curves, masked_strata
```

About 2 minutes per model per backbone on a V100S. Each run writes its two profiles (masked and
unmasked) and a provenance JSON into `output/unmasked_check/`.

**Every run is gated, and the gate is the point.** The unmasked arm is recomputed alongside the
masked one and compared against the published `plm_cache`; the collapse step then reproduces the
shipped `dms_scores.log_plm` from those unmasked profiles before it writes anything. Both stop
rather than write on a disagreement. Without that, a difference in a figure could be a
tokeniser, a checkpoint or an off-by-one rather than masking. Measured values are tabulated in
`docs/notes/MASKED_DMS_PROGRESS.md`.

Figure 3's lineage roots were scored the same way as a check, not for a published figure:
`dev/masked_fig03/run_all_lineages.sh` (13 lineages x 2 models, about 80 minutes), writing to
`output/unmasked_check/fig03_13lineage/` with a report beside the figure.

## The spectrum comparison (S4, S5) — the one run that needs nothing

`results/spectra_codon_space/` is the exception to everything above: it is cheap, it needs no
GPU, no archive and no GISAID-derived data, and it can simply be rebuilt.

```bash
python tools/export_spectra.py
```

Its only inputs are the 4x4 matrices in `data/spectra/` and the metadata in
`spectra_sources.csv`. The exporter scales every matrix to a common total rate, pushes each
through the standard genetic code (taken from Biopython, never typed out), compares every
spectrum against every other, and writes three tables with a MANIFEST. A test recomputes the
shipped tables from `data/spectra/` and fails if they have gone stale, so there is no way for
this run's output to drift from its input unnoticed.

It reproduces the archive's `a16_cross_species_matrices.py` output to 1e-10, and the
manuscript's published 84.8% and 43.6% (and 51.0% uncompressed) to the digit printed.

## How many times each replacement arose (Table SZ), 21 Oct 2026

Table SZ's *Homoplasies* and *Multiple origins* columns come from the segment 4
mutation-annotated UShER tree — the treepatch repair of 24 Sep 2026, 780,727 nodes and
572,108 tips, carrying 845,167 mutation records. Three files are taken off it in one pass:

```bash
matUtils extract -i <seg4 repair usher.pb> \
    -t seg4.nwk -A seg4_paths.txt -v seg4_ref.vcf -n
```

The reference record is the **first** record of the alignment the tree was built from,
with its gap columns removed. Both checks that can catch a wrong coordinate system are run
before anything is counted and both stop the run: the record must read as one open reading
frame (1,659 bases, 553 codons, one terminal stop), and its base must equal the VCF `REF`
base at every one of those 1,659 positions.

```bash
python tools/export_homoplasy.py \
    --nwk seg4.nwk --paths seg4_paths.txt --vcf seg4_ref.vcf --ref ref_ungapped.fa \
    --accessions ../dev/lineage_panel/clade_accessions.tsv \
    --roots data/roots/h3n2_13lineage \
    --columns results/s11_h3n2_ha_tree/column_codons.tsv.gz \
    --observed results/h3n2_13lineage/replacement_observed.tsv.gz \
    --out-dir results/h3n2_13lineage
```

Four things about this run are worth knowing before the numbers are quoted.

**The whole tree is walked, not a pruned copy.** Pruning to one lineage's tips first would
be wrong for this question: `matUtils` merges the mutations of the paths it prunes away
onto the branch that survives, so two changes at one codon on two branches can become one
branch carrying both — which is exactly the distinction being counted. The walk instead
descends from each lineage's common ancestor into only those children that have a lineage
member below them, so every branch is scored as UShER placed it.

**The UShER reference is not an H3.** `DQ864721` is an H2, and its codon index is not an H3
residue number, so `--columns` supplies the mapping — the `site_number` column of the
shipped S11 table, whose own numbering was checked against the conserved residues Y98,
W153, H183 and Y195. A mature site number `s` is position `s + 16` of the lineage root's
566-residue protein; the offset was confirmed by agreeing with all thirteen root proteins
at 94–98% of positions, against 4% for either neighbouring offset. 535 of the 566 positions
have a reference codon. Positions 1–26 (the signal peptide, and the first ten residues of
mature HA1) and 562–566 do not, and no replacement there can be placed.

**Eight of the thirteen lineages are paraphyletic, by construction.** A sequence is assigned
to the most derived clade it falls in, so `E` means "clade E but not E.2" and `J.2` means
"J.2 but not J.2.4 or K". The common ancestor of such a set is the whole clade's, which is
why `n_mrca_tips` is far larger than the lineage for exactly the eight that have a descendant
lineage in the panel (E, G.1, G.1.1, G.1.3, G.2, J, J.2, J.2.4) and close to it for the five
terminal ones (C.1, D, E.2, G.2.2, K). It is not a misassignment and it does not widen the
walk.

**76,029 of the 78,659 lineage sequences are in the tree.** The shortfall is mostly K
(1,971 of 19,492), which post-dates parts of the tree build. A replacement observed only in
a sequence the tree does not hold has no origin to find; it is reported as unplaced, never
as a single origin.

Result: 3,038 of the 6,130 placed replacements arose two or more times — 50% overall, 23%
(G.1) to 67% (J.2) by lineage. 93% of those are the same codon change in different parts of
the tree; the other 7% reach the same residue by two or more different codon changes.
**The percentage tracks sampling depth almost perfectly** (Spearman 0.96 against sequence
count across the thirteen lineages), so it compares sampling effort as much as biology and
the counts are a floor on recurrence, not an estimate of its rate. The most recurrent
replacement found is position 161 — mature H3 site 145, an antigenic site — gaining N 90
times in J.2, by both `AGT>AAT` and `AGT>AAC`.

### Is the recurrence real, or tree error and sampling depth? (21 Oct 2026)

Two objections have to be answered before any of these counts is quoted, and both were
tested rather than argued.

**The tree is the right one.** The influenza database holds a segment 4 UShER entry
(`trees`, `usher_refset_4_aln_merged_MSA_dedup`, created 2 Oct 2026) but it is **newick
only**: it carries no per-branch mutations, so origins cannot be counted from it. Its
572,119 tips and the 572,108 of the 24 Sep treepatch repair `.pb` share 572,106, so the two
are the same tree and the `.pb` is the only file that can answer this question. The
database is the source for metadata, not for this.

**Tree error inflates the raw counts, but cannot explain the biology.** 9,751 of the 18,188
origins (54%) sit on a terminal branch with a single descendant tip, which is what a
parsimony misplacement produces. Requiring origins to sit on internal branches drops the
homoplasy share everywhere (J.2.4 27.6% → 14.0%, E.2 31.5% → 21.9%) without removing it.
The decisive test is that UShER does not know whether a codon change alters the residue, so
placement error inflates synonymous and replacement recurrence alike: counting recurrence
the same way for synonymous changes — unit `(position, codon)` against the root codon,
beside `(position, amino acid)` against the root residue — gives a baseline that carries
the error and not the selection. **Replacements recur substantially *less* than synonymous
changes**, in all thirteen lineages, by a mean 12.4 percentage points at a common sampling
depth (Wilcoxon signed-rank p = 2.4e-4; Mantel-Haenszel OR 0.454 stratified on the number
of alternative states seen at that position, which rules out the two classes' different
state-space sizes). Error that is blind to the amino acid cannot produce a difference
between the classes, so that difference is not error. It reads as purifying selection.

**Conditioning on frequency removes the signal, and is the wrong test.** At matched
frequency the two classes are indistinguishable (Mantel-Haenszel OR 1.075, 1.085 on
internal branches alone; no stratum significant). This is not evidence against the result:
frequency is how selection acts, so holding it fixed conditions away the thing being
measured. It does mean the raw *Homoplasies* and *Multiple origins* columns carry no
amino-acid-level signal of their own once frequency is known, and must not be cited as
evidence about accessibility or selection.

**Rarefaction makes the column comparable across lineages.** The depth artefact is a
species-accumulation problem. An origin is still visible in a subsample of `n` of a
lineage's `N` tips exactly when at least one of its `s` descendant tips is drawn, so
`P(visible) = 1 - C(N-s, n)/C(N, n)` and a replacement is a homoplasy in that subsample
when two or more origins are visible — a Poisson-binomial over its origins, exact and
needing no re-walk. Rarefied to 778 tips (G.1, the smallest), the correlation with
sampling depth falls from **Spearman 0.96 to 0.23 (p = 0.45)** and the rate flattens to
21–35%, J.2 still the highest at 35.0%.

## Checks that guard the shipped sweep table

`tests/test_data_integrity.py` fails if the AUROC column is ever floored at 0.5, if the
alpha grid differs between lineages, or if the `*_constrained_fit` columns reappear.
`tests/test_fig03.py` pins the lineage means at alpha = 0 to the values quoted in
`LINEAGE_VALIDITY.md` (0.9195 fine-tuned, 0.7721 base).
