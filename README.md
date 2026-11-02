# Mutational accessibility and protein language models — figure engine

Code and derived tables that redraw every figure of **"Protein language models learn
underlying mutation biases alongside fitness landscapes"** (MacLean, Lamb, Lytras,
Mojsiejczuk, Yuan, Hughes & Robertson), bioRxiv
[10.64898/2026.07.06.736753](https://www.biorxiv.org/content/10.64898/2026.07.06.736753), under revision at
*Molecular Biology and Evolution*.

Protein language models score amino acid replacements as pseudo-probabilities. Because they
are trained on natural sequences, those scores conflate structural constraint with
**nucleotide mutation bias** — how easy a replacement is to reach through the genetic code.
This repository holds the analysis that separates the two, and the code that draws it.

## What this repository is, and is not

**It is a figure engine.** Every figure script takes a *guide file* naming the lineages,
models and tables of one analysis run, and draws what it finds. Nothing in `figures/` should
name a lineage, a model, a virus or a count, and `tests/test_figure_rules.py` enforces that —
with one known exception, recorded under [Known gaps](#known-gaps).

**No model runs here.** Scoring the protein language models needed GPUs and sequence data
that cannot be redistributed. What ships is their *output* — the scored tables under
`results/` — together with everything needed to turn those tables into the published
figures. How the tables were made is recorded in [`RUNS.md`](RUNS.md); reproducing them is a
separate, much heavier job.

So a reader can check **every number and every statistic in the paper** from this repository
alone, on a laptop, with no GPU and no network, and rebuild every figure but one: the Fig 1A
clade tree was drawn from GISAID sequences that cannot be redistributed and is placed as a
frozen file. A reader cannot re-score the models from here.

## Quick start

```bash
git clone https://github.com/omaclean/PLM_mutational_accessibility
cd PLM_mutational_accessibility

conda env create -f environment.yml && conda activate mutacc
# or: python -m pip install -r requirements.txt

pytest                        # the full suite; a few minutes, no GPU, no network
python tools/build_figures.py # redraw everything and check it against the shipped figures
```

One environment draws every figure and runs every test. Tables ship as gzipped TSV so that
no parquet reader is needed. Python 3.12; the dependencies are numpy, pandas, scipy,
matplotlib, PyYAML, biopython, Pillow and PyMuPDF, with fonttools and threadpoolctl as
optional extras and pytest for the tests. Nothing heavier, and no machine-learning stack:
no model is run from this repository.

## How to validate this work

`tools/build_figures.py` is the check. It rebuilds every figure into a scratch directory and
**compares the result with the shipped `output/`**: every `_values.tsv` and every
`_caption.md` must match — values to one unit in the sixth significant digit, captions as
text — because every bootstrap in the repository is seeded. A difference means a shipped
figure no longer matches the code that claims to draw it, and the exit status says so.

```bash
python tools/build_figures.py                  # build and compare; non-zero exit on any drift
python tools/build_figures.py --list           # the steps, in dependency order
python tools/build_figures.py --only sE fig02  # a subset, plus whatever it needs
python tools/build_figures.py --out output     # overwrite the shipped figures; review with git diff
```

Steps run four at a time on one BLAS thread each (`--jobs`). The slowest is SM, about nine
minutes on one core; the rest finish in a few minutes together. PDFs and PNGs are not
compared, since they embed fonts and build dates — the `_values.tsv` beside each figure is
the artefact to diff, and it holds exactly the numbers plotted.

The composites (Figures 1 and 2) place the panels *this* build made, not the shipped ones.
Retired figures are skipped unless `--include-retired`.

Three further checks are worth knowing about:

* `pytest` covers the library, every figure script, the guide files and the input data. It
  also checks the prose: `tests/test_document_numbers.py` fails if a document in this
  repository asserts a figure number that contradicts
  `guides/manuscript/figure_numbers.yaml`, and `tests/test_build_figures.py` fails if a
  figure script exists without a build step, or a step claims a figure that is not shipped.
* `tests/test_figure_rules.py` enforces that no figure script names a lineage, a model or a
  virus, and that none writes a colour literal. It currently fails on one script; see
  [Known gaps](#known-gaps).
* Each `results/<run>/` directory carries a `MANIFEST` recording what produced its tables.

## What each figure comes from

Each command writes `output/<stem>.pdf` and `.png`, plus `<stem>_values.tsv` (the numbers
plotted) and `<stem>_caption.md`.

The exception is a script that draws one cell of a composite — `make_fig01bc.py`,
`make_fig01d.py`, `make_fig02ab.py`, `make_fig02c.py`, `make_fig02d.py`. Its PDF and PNG are
intermediates: `panelise.py` places them in the composite and nothing reads them again, so
they go to the gitignored `build/panels/` (`--panels-out` overrides it) and
`tools/build_figures.py` deletes them once both composites are built. `output/` therefore
holds finished figures only. Their `_values.tsv` and `_caption.md` do stay in `output/` with
every other figure's: those are the record a rebuild is checked against, not an intermediate.

> **On the supplementary labels in this table.** The supplement is lettered, not numbered
> (since 20 Oct 2026): `Figure SA` … `Figure SP`, and `Table SW` … `Table SZ`. A letter is a
> *name*, not a position — the final digits are assigned once, at submission, by walking the
> citations in order of first mention. The labels come from
> [`guides/manuscript/figure_numbers.yaml`](guides/manuscript/figure_numbers.yaml), which
> `mutacc.numbering` reads and every shipped caption is checked against;
> [`FIGURE_LABELS.md`](FIGURE_LABELS.md) is the reader's copy, including the translation to
> the numbers the 15 Oct 2026 draft used.

| Figure | Command | Reads |
|---|---|---|
| **1** tree, ranked mutations, heatmap | `python figures/panelise.py --layout guides/composites/fig01.yaml` | the frozen 1A clade tree, plus `build/panels/fig01b_...`, `fig01c_...`, `fig01d_...` (intermediates; see below) |
| **2** composite (model vs accessibility, epoch curve, Grantham control arrows, direction) | `python figures/panelise.py --layout guides/composites/fig02.yaml` | `build/panels/fig02a_scatter_panel.pdf`, `fig02b_...`, `fig02c_...`, `fig02d_...` (intermediates; see below) |
| **3** alpha sweep | `python figures/make_fig03.py --guide guides/h3n2_13lineage.yaml` | `alpha_sweep_by_lineage` |
| **4** (influenza) + **SL** (SARS-CoV-2) DMS fitness vs weighted score | `python figures/make_fig04.py --guide guides/h3n2_13lineage.yaml` | `dms_scores`, `dms_scores_masked`, `dms_alpha_curves`, `dms_peak_summary` |
| **5** + **SI** codon learning (two figures) | `python figures/make_codon_learning.py --guide guides/h3n2_5lineage.yaml` | `replacement_scores` |
| SB six ranking metrics: PR-AUC, fold enrichment, partial AUROC, the two frequency correlations and picking the winners | `python figures/make_sB_ranking_metrics.py --guide guides/h3n2_13lineage.yaml` | `replacement_scores_masked`, `replacement_observed`, `replacement_frequency`, `alpha_sweep_by_lineage_masked` (`--scoring unmasked` for the unmasked pair) |
| SD partial correlation | `python figures/make_sD_partial_correlation.py --guide guides/h3n2_5lineage.yaml` | `replacement_scores` |
| SE spectra in codon space + best generic stand-in | `python figures/make_sE_spectra_codon_space.py` | `spectra_pairwise`, `spectra_sources`, `spectra_generic_stand_in` |
| SF spectrum swap (two figures) | `python figures/make_sF_spectrum_swap.py --guide guides/h3n2_13lineage.yaml` | `spectrum_swap_strata`, `replacement_scores` |
| SG epistasis against a null | `python figures/make_sG_epistasis_null.py --guide guides/h3n2_13lineage.yaml` | `epistasis_cells`, `epistasis_focal_sites`, `epistasis_null_draws` |
| SJ DMS agreement vs distance | `python figures/make_sJ_dms_concordance.py --guide guides/h3n2_13lineage.yaml` | `dms_cross_lineage_pairs` |
| SK nucleotides per codon per branch | `python figures/make_sK_branch_codons.py --branch-table results/s11_h3n2_ha_tree/branch_codons.tsv.gz --column-table results/s11_h3n2_ha_tree/column_codons.tsv.gz --long-range-nodes results/s11_h3n2_ha_tree/long_range_nodes.tsv.gz --baseline-codons results/s11_h3n2_ha_tree/baseline_codons.tsv.gz --year-table results/s11_h3n2_ha_tree/year_counts.tsv.gz --plm-null-scores results/h3n2_13lineage/replacement_scores_masked.tsv.gz --plm-null-guide guides/h3n2_13lineage.yaml --plm-null-lineage J.2` | real tables, the 224,731-tip tree of 24 Sep 2026 |
| SC recovery of a hidden amino acid across viral proteins | `python figures/make_sC_viral_panel.py --results results/viral_panel` | untuned ESM-2 and ESM-C |
| SM DMS within nucleotide-step strata | `python figures/make_sM_within_stratum.py --guide guides/h3n2_13lineage.yaml` | `dms_scores`, `dms_scores_masked`, `dms_within_stratum`, `dms_peak_summary` |
| SH contact recovery by region, and at antigenic sites | `python figures/make_sH_contact_recovery.py --guide guides/h3n2_13lineage.yaml` | `contact_*`, `epistasis_focal_sites`, `antigenic_sites`, `antigenic_residue_accuracy` |
| lineage-set validity audit — **retired** 15 Oct 2026, dropped from the supplement; the findings are in [`LINEAGE_VALIDITY.md`](LINEAGE_VALIDITY.md) §4 | `python figures/make_s14.py --guide guides/h3n2_13lineage.yaml --guide-compare guides/h3n2_5lineage.yaml` | `audit_*`, `corpus_membership`, `replacement_scores` |
| SA SARS-CoV-2 spike: Figure 2's question asked of a second virus | `python figures/make_sA_spike.py --guide guides/sars_cov_2_spike.yaml` | `replacement_scores` (spike) |
| Table SZ, the thirteen lineages | `python figures/make_sZ_lineage_panel.py --guide guides/h3n2_13lineage.yaml --results results/h3n2_13lineage` | `lineage_panel` |
| Table SX, corpus leakage: which lineage-defining residues the fine-tuning corpus already held | `python figures/make_sX_corpus_leakage.py --results results/h3n2_13lineage` | `k_corpus_leakage` |
| SO masking check: what the wild-type (unmasked) estimator would have given | `python figures/make_sO_masked_marginals.py --guide guides/h3n2_13lineage.yaml` | `masked_dms_curves`, `masked_strata` |
| SP every scan's whole accessibility-weight curve, behind SM's optimum counts | `python figures/make_sP_alpha_curves.py --guide guides/h3n2_13lineage.yaml` | `dms_scores`, `dms_scores_masked`, `dms_peak_summary` |
| SN simulation control, four evolutionary arms | `python figures/make_sN_simulation_control.py` | `results/sim_v3_star90/` |
| Figure 1 panel D, codon-to-amino-acid heatmap (a panel, not a figure) | `python figures/make_fig01d.py` | `data/spectra/` |
| Figure 1 panels B and C, every replacement ranked (panels, not figures) | `python figures/make_fig01bc.py` | `fig01bc_ranked_mutations_values.tsv` |
| Figure 1 panels B and C, the values and their caption | `python tools/export_fig01bc_values.py --comparison K` | `replacement_scores` |
| how closely those two quantities agree: both models, raw, either variable logged, both logged, and ranks | `python tools/export_fig01bc_correlations.py` | `replacement_scores` |
| Figure 2 panels A and B, PLM probability against accessibility (panels, not figures); add `--labels text` for the no-legend variant | `python figures/make_fig02ab.py --guide guides/h3n2_5lineage.yaml` | `replacement_scores` |
| Figure 2 panel C, epoch curve (a panel, not a figure) | `python figures/make_fig02c.py` | `data/frozen_panels/fig2c_epoch_values.tsv`, from a real per-epoch ESM-2 650M fine-tuning run on HA-80, evaluated on `J.2_int`; the run is written up in [`RUNS.md`](RUNS.md) and each row names its own checkpoint |
| Figure 2 panel D, strata (a panel, not a figure) | `python figures/make_fig02d.py --guide guides/h3n2_5lineage.yaml` | `replacement_scores` |

SE takes no guide: it is a property of the mutation matrices and the genetic code, with
no lineage set and no protein model. It reads `results/spectra_codon_space/`, which
`tools/export_spectra.py` builds from `data/spectra/`. Figure 5 evaluates on the primary
lineage reference sequence (`J.2_int`, matching Figure 2D and SD) to eliminate cross-lineage
pseudo-replication across closely related variants sharing >98% sequence identity.

[`figures/README.md`](figures/README.md) has the longer table with caveats;
[`DATA.md`](DATA.md) says which table feeds which figure and where it came from.

## Layout

```
guides/        one YAML per analysis run (and composites/ for panel layouts)
mutacc/        library: style, guide loader, metrics, io
figures/       one script per figure, plus panelise.py
tools/         build_figures.py (rebuild everything), bridges and analysis runners
data/          lineage roots, nucleotide mutation spectra, reference genomes, frozen panels,
               db_build/ (which influenza database build the sequence metadata came from)
results/       derived tables (gzipped TSV) with a MANIFEST per run
output/        the shipped figures, and the values tables a rebuild is checked against
tests/         code and input-data tests
docs/notes/    working notes: the plan, the red-team audits, progress logs
```

### Where to read next

| Document | What it answers |
|---|---|
| [`DATA.md`](DATA.md) | What every derived table is, how many rows it has, and where it came from |
| [`RUNS.md`](RUNS.md) | How the upstream scoring runs were done — models, checkpoints, commands |
| [`DB_SOURCING.md`](DB_SOURCING.md) | Where the sequence data came from, which database build, and the metadata defects that shape the figures |
| [`LINEAGE_DEFINITIONS.md`](LINEAGE_DEFINITIONS.md) | The H3N2 J.2 → K transition and its nine clade-defining replacements |
| [`LINEAGE_VALIDITY.md`](LINEAGE_VALIDITY.md) | Whether the lineage sets are independent, and what the fine-tuning corpus already contained |
| [`SYMMETRY_TESTS.md`](SYMMETRY_TESTS.md) | The symmetric-versus-directional battery and what it found |
| [`PREPRINT_AND_REVIEWS.md`](PREPRINT_AND_REVIEWS.md) | Figure-by-figure mapping from the preprint, and the reviewers' requests |
| [`FIGURE_LABELS.md`](FIGURE_LABELS.md) | Supplementary figure letters → content, and why they are letters |
| [`LICENSE-DATA.md`](LICENSE-DATA.md) | What may be reused, and the inputs that are not ours to license |
| `docs/notes/` | Working notes, kept for transparency. Not needed to validate anything. |

## Relation to the preprint

The bioRxiv preprint (July 2026) is **not** what these figures reproduce. Its Fig 2 is two
scatters and an epoch curve, its Fig 3 a five-lineage sweep, its Fig 4 the epistasis scan.
The within-stratum, comparator, direction, thirteen-lineage and DMS figures here are
**revision-stage** work answering the MBE reviews.
[`PREPRINT_AND_REVIEWS.md`](PREPRINT_AND_REVIEWS.md) has the figure-by-figure mapping.

## Which models, which checkpoints

The SARS-CoV-2 fine-tune used everywhere in this repository is **ESM-C 600M**, checkpoint
`my_SC2_finetunes/magma_ESMC_600M_99_95perc/esmc_weights.pt`, with raw ESM-C 600M as its
base. The influenza models are **ESM-2 650M**, base and fine-tuned on HA-80.

The ESM-2 650M spike fine-tune (`magma_esm-2_650M_95perc`) is **not** the paper's model. SA
first drew it by mistake (12 Oct 2026), and also showed an ESM-2 influenza fine-tune on
spike; both are gone, and `tests/test_sA_spike.py` pins SA to the ESM-C pair. The other
SARS-CoV-2 uses (Fig 4, SJ, SM, the symmetry tests) were checked the same day and already
use ESM-C.

## Things to know before quoting a number

This section is the honest part of the repository. Each item is a conclusion that would be
easy to overstate, with the reason not to.

* **The across-lineage bands are not generalisation intervals.** The thirteen lineages are
  disjoint sets of sequences but share most of their sites, so a band over lineages
  understates the uncertainty, most for the weaker model. Captions say so.
* **Figure 2 is one lineage, deliberately.** All four panels are computed on the primary
  reference root `J.2_int` of the five-lineage set, not pooled across lineages. The other
  lineages are scored at the same sites, so they would add near-copies of the same points
  rather than independent ones. The shipped captions say so, and the frozen panels they
  replaced — which did pool five lineages — are kept in `data/frozen_panels/` for comparison.
* **BLOSUM62 is not a biochemistry control.** Empirical substitution matrices already contain
  mutational accessibility. Grantham distance is the control; BLOSUM62 is a comparator.
* **One panel cannot be regenerated: the Fig 1A clade tree.** It was built from GISAID
  sequences that cannot be redistributed, so no generator exists; it is a frozen file in
  `data/frozen_panels/` and this repository holds the only copy. The Fig 1D heatmap and the
  Fig 2C epoch curve were frozen too until October 2026 and are not any more:
  `figures/make_fig01d.py` recomputes the heatmap from `data/spectra/`, and
  `figures/make_fig02c.py` draws the epoch curve from a real per-epoch fine-tuning run. Both
  are build steps, and both are rebuilt and checked by `tools/build_figures.py`.
* **The five-lineage roots are not what their headers say.** `guides/h3n2_5lineage.yaml`
  records this and `tests/test_data_integrity.py` pins it. Do not quote five-lineage numbers
  without resolving it.
* **Do not quote "fine-tuning is better in N protein regions" from the archive's contact-recovery
  intervals.** Those intervals resample contact *pairs*, but one residue sits in many pairs,
  so they are too narrow. Resampling residues instead widens them about 2.2x, and almost no
  region then clearly separates base from fine-tuned. SH draws the residue-resampled
  interval.
* **The antigenic result is a null and says so.** Fine-tuning is not specialised to known
  antigenic sites: the gain there is about the gain elsewhere in the same domain, and neither
  structure's difference excludes zero (p = 0.85 and 0.71). Since it is a null it is now
  panel D of Figure SH, with the five published sites pooled into one class and only the
  zero-sum gauge drawn. Splitting by site asked five uncorrected questions and one site came
  out ahead in both structures, which is what five looks produce; do not quote it.
* **Neither influenza model shows evidence of having learned influenza's mutational
  direction.** The SARS-CoV-2 fine-tune did learn its own virus's direction.
  [`SYMMETRY_TESTS.md`](SYMMETRY_TESTS.md) has the battery and the numbers; the archive's S04 headline
  "symmetric beats directional" ordering is reproduced by controls containing no nucleotide
  information at all, so it does not support the reading the archive gave it.

### A correction worth knowing about

Figures 1B/C once carried a tenth replacement, `N145S`, on the J.2 → K path. It does not
exist. The panel had diffed a *stored* results table's `ref_aa` against a *live* reference
FASTA; the `J.2_int` root was corrected at precursor codon 161 (`aat` → `agt`) on 15 Jun 2026
and the May tables were never rebuilt, so a stale J.2 was compared against a current K.
Nothing warned. The transition is **nine** replacements: `K2N`, `T135K`, `S144N`, `N158D`,
`I160K`, `Q173R`, `K189R`, `T328A`, `HA2:S49N`.

The shipped `output/fig01_composite.pdf` carries exactly those nine. The 2 Jul 2026 render is
kept as `data/frozen_panels/fig1bc_ranked_mutations.pdf` **for provenance only — do not place
it.** The standing risk it exposed: re-plotting an old table against newer roots can silently
invent or erase replacements. Full account in [`LINEAGE_DEFINITIONS.md`](LINEAGE_DEFINITIONS.md)
section 6.

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
carries is a channel the reader has to rule out. SJ was redrawn on 13 Oct 2026 for this:
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

**Achromatic figures are not exempt.** The antigenic comparison (now SH panel D) and SJ
were built in ink and grey on the argument that their categories were not model roles. That
is now handled by `CATEGORICAL` instead, so they use colour like everything else — antigenic
residues against the rest of the domain take the first two category colours, not blue and
green, because every point there is already a fine-tuned-minus-base difference and the model
pair would be misread.

## Known gaps

* **SC** (recovery of a hidden amino acid across viral proteins) needs a GPU to re-score.
  The scores and the sequences ship: `results/viral_panel/`, `data/viral_panel/`.
* **SK** ships real tables from the H3N2 HA tree in `results/s11_h3n2_ha_tree/`:
  `h3n2_1968.pb` — H3N2 only, dated, 1968 onward, **224,731 tips**, pruned from the
  treepatch repair `usher.pb` of **24 Sep 2026**, as that directory's `MANIFEST.tsv` and
  `summary.txt` both record. They must be re-made on the 2 Oct tree once a `.pb` exists for
  it: the database's `trees` table holds newick only, with no per-branch mutations
  ([`DB_SOURCING.md`](DB_SOURCING.md) §4).

Status and run times for both: `docs/notes/S3_S11_PROGRESS.md`.

* **`make_sK_branch_codons.py` breaks the engine's own rule, and its test says so.**
  `tests/test_figure_rules.py::test_no_literals_in_figure_scripts[make_sK_branch_codons.py]`
  fails. SK is the one figure that can be run without a guide — the command in the table
  above passes its tables directly — so the script carries the model keys
  (`ESM2_650M_HA80_raw`, `ESM2_650M_HA80`) and the legend label `HA80 fine-tune` as a
  hard-coded fallback, and overrides the guide's own label even when a guide is given. The
  figure it draws is correct; what is wrong is that the script knows which virus it is
  looking at. Fixing it changes SK's legend text, so it is deferred rather than done
  quietly before submission.

## Licence and citation

Code is [MIT](LICENSE). Derived tables and figures are
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Some inputs are not ours to
license — in particular the GISAID-derived sequence alignments, which are **not
redistributed here**. See [`LICENSE-DATA.md`](LICENSE-DATA.md).

To cite, see [`CITATION.cff`](CITATION.cff), or cite the preprint directly:

> MacLean OA, Lamb K, Lytras S, Mojsiejczuk L, Yuan K, Hughes J, Robertson DL (2026).
> Protein language models learn underlying mutation biases alongside fitness landscapes.
> *bioRxiv* [10.64898/2026.07.06.736753](https://www.biorxiv.org/content/10.64898/2026.07.06.736753).

Questions: [oscar.maclean@glasgow.ac.uk](mailto:oscar.maclean@glasgow.ac.uk).
