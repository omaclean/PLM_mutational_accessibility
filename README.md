# Mutational accessibility and protein language models: figures


Code to redraw the figures of *Protein language models learn underlying mutation biases
alongside fitness landscapes* (MacLean et al.) from the derived tables the analysis produced.

The repository is a **figure engine**. Every figure script takes a *guide file* naming the
lineages, models and tables of one analysis run, and draws what it finds. Nothing in
`figures/` names a lineage, a model, a virus or a count; a test enforces that.

## Set up

```bash
conda env create -f environment.yml && conda activate mutacc
pytest            # about two minutes; no GPU, no network
```

One environment draws every figure and runs every test. Tables ship as gzipped TSV so that
no parquet reader is needed.

## Rebuild every figure

```bash
python tools/build_figures.py                 # build into build/figures/, then compare with output/
python tools/build_figures.py --list          # the steps, in dependency order
python tools/build_figures.py --only sE fig02   # a subset, plus whatever it needs
python tools/build_figures.py --out output    # overwrite the shipped figures (review with git diff)
```

One command, no GPU, no network, no archive: every step reads the derived tables in
`results/` and the frozen panels in `data/`. The scoring runs that made those tables are
recorded in `RUNS.md` and are not repeated here. Steps run four at a time on one BLAS thread
each (`--jobs`). The default build goes to a scratch directory and is **compared** with the
shipped `output/`: every `_values.tsv` and `_caption.md` must match (values to one unit in the
sixth significant digit, captions as text), because every bootstrap is seeded. A difference
means a shipped figure no longer matches the code that claims to draw it, and the exit status
says so. PDFs and PNGs are not compared, since they embed fonts and dates. The slowest
step is S12 (about 9 minutes on one core); the rest finish in a few minutes together.

The composites (Figures 1 and 2) place the panels *this build* made, not the shipped ones.
Retired figures are skipped unless `--include-retired`. Which figure each step draws, and the
guide it reads, are in `tools/build_figures.py`; `tests/test_build_figures.py` fails if a figure
script is added without a step, or a step claims a figure that is not shipped.

## Reproduce a single figure

Each command writes `output/<stem>.pdf` and `.png` (same stem), plus `<stem>_values.tsv`
(the numbers plotted) and `<stem>_caption.md`.

The exception is a script that draws one cell of a composite — `make_fig01bc.py`,
`make_fig01d.py`, `make_fig02ab.py`, `make_fig02c.py`, `make_fig02d.py`. Its PDF and PNG are intermediates:
`panelise.py` places them in the composite and nothing reads them again, so they go to the
gitignored `build/panels/` (`--panels-out` overrides it) and `tools/build_figures.py` deletes
them once both composites are built. `output/` therefore holds finished figures only. Their
`_values.tsv` and `_caption.md` do stay in `output/` with every other figure's: those are the
record a rebuild is checked against, not an intermediate.

> ⚠️ **Supplementary numbers in this table are stale (15 Oct 2026), and the supplement
> is lettered now (20 Oct 2026).** Against the manuscript draft: contact recovery is
> **Figure SH**, the SARS-CoV-2 DMS sweeps **SL**, the within-stratum figure **SM**, the
> simulation control **SN**, the masking check **SO**, the lineage panel **Table SZ**; the
> lineage-set validity figure is dropped. Letters are names, not positions — the final
> S1…SN digits are assigned at submission from the order of first citation. Full mapping:
> [`FIGURE_LABELS.md`](FIGURE_LABELS.md); staged stem fix:
> [`extended_red_team_tests.md`](extended_red_team_tests.md) §A1b. Trust the command and
> the description, not the number in the table.

| Figure | Command | Reads |
|---|---|---|
| **1** tree, ranked mutations, heatmap | `python figures/panelise.py --layout guides/composites/fig01.yaml` | frozen panels, see below |
| **2** composite (model vs accessibility, epoch curve, Grantham control arrows, direction) | `python figures/panelise.py --layout guides/composites/fig02.yaml` | `build/panels/fig02a_scatter_panel.pdf`, `fig02b_...`, `fig02c_...`, `fig02d_...` (intermediates; see below) |
| **3** alpha sweep | `python figures/make_fig03.py --guide guides/h3n2_13lineage.yaml` | `alpha_sweep_by_lineage` |
| **4** (influenza) + **S18** (SARS-CoV-2) DMS fitness vs weighted score | `python figures/make_fig04.py --guide guides/h3n2_13lineage.yaml` | `dms_scores`, `dms_scores_masked`, `dms_alpha_curves`, `dms_peak_summary` |
| **5** + **S9** codon learning (two figures) | `python figures/make_codon_learning.py --guide guides/h3n2_5lineage.yaml` | `replacement_scores` |
| S2 six ranking metrics: PR-AUC, fold enrichment, partial AUROC, the two frequency correlations and picking the winners | `python figures/make_sB_ranking_metrics.py --guide guides/h3n2_13lineage.yaml` | `replacement_scores_masked`, `replacement_observed`, `replacement_frequency`, `alpha_sweep_by_lineage_masked` (`--scoring unmasked` for the unmasked pair) |
| S4 partial correlation | `python figures/make_sD_partial_correlation.py --guide guides/h3n2_5lineage.yaml` | `replacement_scores` |
| S5 spectra in codon space + best generic stand-in | `python figures/make_sE_spectra_codon_space.py` | `spectra_pairwise`, `spectra_sources`, `spectra_generic_stand_in` |
| S6 spectrum swap (two figures) | `python figures/make_sF_spectrum_swap.py --guide guides/h3n2_13lineage.yaml` | `spectrum_swap_strata`, `replacement_scores` |
| S7 epistasis against a null | `python figures/make_sG_epistasis_null.py --guide guides/h3n2_13lineage.yaml` | `epistasis_cells`, `epistasis_focal_sites`, `epistasis_null_draws` |
| S10 DMS agreement vs distance | `python figures/make_sJ_dms_concordance.py --guide guides/h3n2_13lineage.yaml` | `dms_cross_lineage_pairs` |
| S11 nucleotides per codon per branch | `python figures/make_sK_branch_codons.py --branch-table results/s11_h3n2_ha_tree/branch_codons.tsv.gz --column-table results/s11_h3n2_ha_tree/column_codons.tsv.gz --long-range-nodes results/s11_h3n2_ha_tree/long_range_nodes.tsv.gz --baseline-codons results/s11_h3n2_ha_tree/baseline_codons.tsv.gz --plm-null-scores results/h3n2_13lineage/replacement_scores_masked.tsv.gz --year-table results/s11_h3n2_ha_tree/year_counts.tsv.gz` | real tables, 22 Sep tree |
| S3 recovery of a hidden amino acid across viral proteins | `python figures/make_sC_viral_panel.py --results results/viral_panel` | untuned ESM-2 and ESM-C |
| S12 DMS within nucleotide-step strata | `python figures/make_sM_within_stratum.py --guide guides/h3n2_13lineage.yaml` | `dms_scores`, `dms_scores_masked`, `dms_within_stratum`, `dms_peak_summary` |
| S13 contact recovery by region, and at antigenic sites | `python figures/make_sH_contact_recovery.py --guide guides/h3n2_13lineage.yaml` | `contact_*`, `epistasis_focal_sites`, `antigenic_sites`, `antigenic_residue_accuracy` |
| S14 lineage-set validity audit | `python figures/make_s14.py --guide guides/h3n2_13lineage.yaml --guide-compare guides/h3n2_5lineage.yaml` | `audit_*`, `corpus_membership`, `replacement_scores` |
| S1 SARS-CoV-2 spike: Figure 2's question asked of a second virus | `python figures/make_sA_spike.py --guide guides/sars_cov_2_spike.yaml` | `replacement_scores` (spike) |
| lineage panel (a table of the thirteen lineages) | `python figures/make_sZ_lineage_panel.py --guide guides/h3n2_13lineage.yaml --results results/h3n2_13lineage` | `lineage_panel` |
| corpus leakage (a table: which lineage-defining residues the fine-tuning corpus already held) | `python figures/make_sX_corpus_leakage.py --results results/h3n2_13lineage` | `k_corpus_leakage` |
| masking check: what the wild-type (unmasked) estimator would have given | `python figures/make_sO_masked_marginals.py --guide guides/h3n2_13lineage.yaml` | `masked_dms_curves`, `masked_strata` |
| every scan's whole accessibility-weight curve, behind the optimum counts | `python figures/make_sP_alpha_curves.py --guide guides/h3n2_13lineage.yaml` | `dms_scores`, `dms_scores_masked`, `dms_peak_summary` |
| simulation control, four evolutionary arms | `python figures/make_sN_simulation_control.py` | `results/sim_v3_star90/` |
| Figure 1 panel D, codon-to-amino-acid heatmap (a panel, not a figure) | `python figures/make_fig01d.py` | `data/spectra/` |
| Figure 1 panels B and C, every replacement ranked (panels, not figures) | `python figures/make_fig01bc.py` | `fig01bc_ranked_mutations_values.tsv` |
| Figure 1 panels B and C, the values and their caption | `python tools/export_fig01bc_values.py --comparison K` | `replacement_scores` |
| how closely those two quantities agree: both models, raw, either variable logged, both logged, and ranks | `python tools/export_fig01bc_correlations.py` | `replacement_scores` |
| Figure 2 panels A and B, PLM probability against accessibility (panels, not figures); add `--labels text` for the no-legend variant | `python figures/make_fig02ab.py --guide guides/h3n2_5lineage.yaml` | `replacement_scores` |
| Figure 2 panel C, epoch curve (a panel, not a figure) | `python figures/make_fig02c.py` | `data/frozen_panels/fig2c_epoch_values.tsv`, digitised from the published raster by `python tools/digitise_epoch_curve.py` |
| Figure 2 panel D, strata (a panel, not a figure) | `python figures/make_fig02d.py --guide guides/h3n2_5lineage.yaml` | `replacement_scores` |

S5 takes no guide: it is a property of the mutation matrices and the genetic code, with
no lineage set and no protein model. It reads `results/spectra_codon_space/`, which
`tools/export_spectra.py` builds from `data/spectra/` without needing the exploration archive.
Figure 5 evaluates on the primary lineage reference sequence (`J.2_int`, matching Figure 2D and S4)
to eliminate cross-lineage pseudo-replication across closely related variants sharing >98% sequence identity.

`figures/README.md` has the longer table with caveats; `DATA.md` says which table feeds which
figure and where it came from.

## Relation to the preprint

The bioRxiv preprint (July 2026) is not what these figures reproduce. Its Fig 2 is two scatters
and an epoch curve, its Fig 3 is a five-lineage sweep, and its Fig 4 is the epistasis scan. The
within-stratum, comparator, direction, thirteen-lineage and DMS figures here are **revision-stage**
work answering the MBE reviews. `PREPRINT_AND_REVIEWS.md` has the figure-by-figure mapping and the
reviewers' requests; `SYMMETRY_TESTS.md` covers the symmetric-versus-directional tests: what S04
got wrong, the direction-specific tests, and the finding that the SARS-CoV-2 fine-tune learned its
virus's direction while the influenza models show no evidence.

## Which models, which checkpoints

The SARS-CoV-2 fine-tune used everywhere in this repository is **ESM-C 600M**, checkpoint
`my_SC2_finetunes/magma_ESMC_600M_99_95perc/esmc_weights.pt`, with raw ESM-C 600M as its base.
The ESM-2 650M spike fine-tune (`magma_esm-2_650M_95perc`) is **not** the paper's model. S1 first
drew it by mistake (12 Oct 2026), and also showed an ESM-2 influenza fine-tune on spike; both are
gone, and `tests/test_sA_spike.py` pins S1 to the ESM-C pair. The other SARS-CoV-2 uses (Fig 4, S10,
S12, the SARS-CoV-2 symmetry tests) were checked the same day and already use ESM-C. The
influenza models are ESM-2 650M, base and fine-tuned on HA-80.

## What is not here yet

Every planned figure now exists. Two were finished late; status and run times are in
`S3_S12_PROGRESS.md`.

* **S3**: two untuned models (ESM-2 650M, ESM-C 600M) on a panel of viral proteins, every position
  hidden in turn. Scoring needs a GPU; the scores and the sequences ship (`results/viral_panel/`,
  `data/viral_panel/`).
* **S11**: real tables from the 22 Sep 2026 H3N2 HA tree ship in `results/s11_h3n2_ha_tree/`. They must be
  re-made on the 2 Oct tree once a `.pb` exists for it.

## Things to know before quoting a number

* **The across-lineage bands are not generalisation intervals.** The thirteen lineages are
  disjoint sets of sequences but share most of their sites, so a band over lineages
  understates the uncertainty, most for the weaker model. Captions say so.
* **Figure 2 mixes two lineage sets.** Panel A is computed on the five-lineage run, panels B
  and C on the thirteen-lineage run. Each panel's title states its own count.
* **BLOSUM62 is not a biochemistry control.** Empirical substitution matrices already contain
  mutational accessibility. Grantham distance is the control; BLOSUM62 is a comparator.
* **Some panels cannot be regenerated.** The Fig 1A tree and 1D heatmap and the Fig 2C epoch
  plot have no generator; they are frozen files in `data/frozen_panels/`.
* **The five-lineage roots are not what their headers say.** `guides/h3n2_5lineage.yaml` records
  this and `tests/test_data_integrity.py` pins it. Do not quote five-lineage numbers without
  resolving it.
* **Do not quote "fine-tuning is better in N protein regions" from the archive's S13 intervals.**
  Those intervals resample contact *pairs*, but one residue sits in many pairs, so they are too
  narrow. Resampling residues instead widens them about 2.2x, and almost no region then clearly
  separates base from fine-tuned. S13 draws the residue-resampled interval.
* **The antigenic result is a null and says so.** Fine-tuning is not specialised to known
  antigenic sites: the gain there is about the gain elsewhere in the same domain, and neither
  structure's difference excludes zero (p = 0.85 and 0.71). This was figure S8; since it is a
  null it is now panel D of Figure SH, with the five published sites pooled into one class and only the
  zero-sum gauge drawn. Splitting by site asked five uncorrected questions and one site
  came out ahead in both structures, which is what five looks produce; do not quote it.
* **Neither influenza model shows evidence of having learned influenza's mutational direction.**
  The SARS-CoV-2 fine-tune did learn its own virus's direction. `SYMMETRY_TESTS.md` has the
  battery and the numbers; S04's headline "symmetric beats directional" ordering is reproduced by
  controls containing no nucleotide information at all, so it does not support the reading the
  archive gave it.

## Colour plan

Two palettes, and only two. Both live in `mutacc/style.py`; no figure script writes a
colour literal.

**1. The model pair — reserved.** Blue is the untuned model, green is the fine-tuned
one, in every figure in the set.

| Role | Colour | |
|---|---|---|
| `BASE` | `#0072b2` | blue — base / untuned |
| `FINETUNE` | `#009e73` | green — fine-tuned |

These two hues are never spent on anything else. A reader who has seen one figure should
be able to carry that meaning to the next without checking the legend.

**2. Everything else — `CATEGORICAL`.** Any grouping that is *not* a model — assay type,
antigenic stratum, structure, virus — is coloured from this list, in sorted category
order:

| | Colour | |
|---|---|---|
| 1 | `#d55e00` | vermillion |
| 2 | `#cc79a7` | reddish purple |
| 3 | `#6a3d9a` | violet |

It holds only warm and violet hues, so nothing drawn from it can be mistaken for the
model pair. Use `style.colours_for_categories(names)`, which assigns by **sorted name**
so that dropping a category never repaints the ones that remain, and raises rather than
cycling if you hand it more categories than there are colours. More than three groups
means encoding some of them with marker shape instead.

**One channel per variable.** A second channel spent repeating a variable colour already
carries is a channel the reader has to rule out. S10 was redrawn on 13 Oct 2026 for this:
it had been carrying the trait in colour *and* in marker fill while marker shape carried a
fourth variable, and the panels were unreadable. Colour now carries the trait, shape
carries the viral species, and the variable that lost its channel lives in the values table
and the caption. Reach for a redundant encoding only where the figure has to survive a mono
print, and say so in the caption when you do.

Greys stay what they were: `NULL_GREY` for nulls, chance lines and reference series,
`INK` / `INK_MUTED` / `GRID` for text and furniture.

**Every entry has to earn its place.** `tests/test_style.py` pins four properties, and a
new colour must pass all of them:

1. at least 3:1 contrast against white;
2. at least ΔE 42 from both `BASE` and `FINETUNE`, so it cannot be read as a model;
3. at least ΔE 35 from the other category colours;
4. at least 0.045 apart in luminance from them, so a mono print still separates.

**Why not ColorBrewer Dark2.** Its first entry, `#1b9e77`, is ΔE 3.2 from `FINETUNE` in
normal vision — the same teal-green. A Dark2 series would collide with "fine-tuned" on
its very first colour. Brick `#8c564b` was the obvious fourth category colour and was
dropped for failing rule 4: it sits 0.040 from violet in luminance.

**Achromatic figures are not exempt.** The antigenic comparison (now S13 panel D) and S10
were built in ink and grey on the argument that their categories were not model roles. That
is now handled by `CATEGORICAL` instead, so they use colour like everything else — antigenic
residues against the rest of the domain take the first two category colours, not blue and
green, because every point there is already a fine-tuned-minus-base difference and the model
pair would be misread.

## Layout

```
guides/        one YAML per analysis run (and composites/ for panel layouts)
mutacc/        library: style, guide loader, metrics, io
figures/       one script per figure, plus panelise.py
tools/         build_figures.py (rebuild everything), bridges and analysis runners: export_*.py, enlarge_tree_labels.py, run_symmetry_*.py
data/          lineage roots, nucleotide mutation spectra, reference genomes, frozen panels
results/       derived tables (gzipped TSV) with a MANIFEST per run
tests/         code and input-data tests
RUNS.md        how the upstream tables were made
FIGURE_LABELS.md  supplementary figure letters -> content, and why they are letters
```
