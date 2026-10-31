# `clean_repo` — build plan and progress tracker

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


**Purpose.** Turn the 675-file exploration archive at `Mut_acc_PLM_MS/` into a small,
runnable, citable manuscript repo under `clean_repo/`, with one coherent figure set
in one visual theme, readable by someone who is neither an ML person nor an
evolutionary geneticist.

**Status:** building. Main figures 1-4 are ported and tested, and every supplementary figure
now exists. S3 and S11 were finished on 12 Oct (both from real data; see `S3_S11_PROGRESS.md`). A
symmetric-vs-directional test battery has been built and run (`SYMMETRY_TESTS.md`); two GPU
simulation arms finished cleanly on 10 Oct and **their results have still not been looked at**.
Nothing has been committed. See the progress log (section 8) and `README.md`.
**Started:** 2026-10-08. **Last updated:** 2026-10-12.

This document is the single source of truth for the job. Update the checkboxes and
the progress log as work lands; do not keep a second plan anywhere else.

### Read this first: preprint vs revision

`PREPRINT_AND_REVIEWS.md` records, from the two PDFs in `MS_stuff/`, what the bioRxiv preprint
contains and what each MBE reviewer asked (decision 3 Sep 2026, revision due about 2 Nov 2026).
**Several figures this plan calls "published" are not in the preprint.** The preprint's Fig 2 is
the two scatters plus the epoch curve; its Fig 3 is a **five**-lineage sweep (Spearman, PR-AUC,
Pearson). Within-stratum correlations, BLOSUM62/Grantham comparators, the direction analysis, the
thirteen-lineage sweep, DMS (new Fig 4) and the codon analyses are all revision-stage. Where this
file says "published" for one of those, read "archive, revision-stage".

### Relationship to the existing plan

`dev/final_steps/final_steps_plan.html` (5 Oct 2026) already covers repo-release
mechanics — reviewer points, strip list, archive/Zenodo, absolute paths, model
hosting. **That document is still valid and is not superseded.** This one is
narrower and goes deeper on the thing it does not settle: the *figure set*, its
*presentation*, and *test coverage*. Where the two overlap, the mapping table below
wins because it is built on the newer figure list.

---

## 1. Scope and governing principle

### The principle (Oscar, 8 Oct)

> *"The new repo should be able to run all the figures off of any guide file pointing
> at anything, and the premise of the question shouldn't matter. But we separately
> need to make sure that the input data and the code get tested like crazy."*

This is the architectural constraint everything else answers to. Concretely:

1. **The clean repo is a question-agnostic figure engine.** Every figure script takes
   a guide file and a results directory and draws what it finds. Nothing hard-codes
   a lineage set, a model name, a virus, or a count of thirteen.
2. **Scientific disputes do not live in the clean repo.** Whether the 13-lineage set
   is sound, whether a result survives a null — these are *separate workstreams* with
   their own documents. The repo must run correctly whichever way they land.
3. **Testing is the deliverable, not a chore.** Input data and code both. A figure
   that draws cleanly from corrupt input is a worse failure than one that crashes.

This changes the target layout: `config/paths.py` becomes a *guide-file loader*, and
every `make_figN.py` takes `--guide`. See §2.

### In scope

1. Identify the real generator for every planned figure; separate it from the
   abandoned near-duplicates.
2. Port the essential code into `clean_repo/` with repo-relative, guide-driven paths.
3. Re-plot everything through one shared style module.
4. Red-team the result with tests — code **and** input data.

### Out of scope

Tracked in the 5 Oct plan: Zenodo deposition, model checkpoint hosting,
licence/CITATION files, rewriting the old public repos.

Tracked in **[`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md)**: whether the 13-lineage
set is free of leakage and bias, how it standardises against the preprint
methodology, and the codon-test contradiction. **That investigation must not leak
into this repo's design.**

---

## 2. Target layout

```
clean_repo/
├─ README.md                  # paper-only; "reproduce Figure N" table
├─ docs/notes/PLAN.md                    # this file (delete or move to dev/ at release)
├─ environment.yml            # trimmed, pinned
├─ guides/                    # ← the parameterisation. One file = one analysis run.
│  ├─ h3n2_13lineage.yaml     #   lineages, roots, alignments, models, spectra
│  ├─ h3n2_5lineage.yaml      #   the preprint's five-lineage set (G.1, J, J.2, J.2.4, K), still runnable
│  └─ composites/             #   panel layouts for panelise.py (not run guides)
│     (validation lives in mutacc/guide.py, not a separate schema.py)
├─ mutacc/                    # library
│  ├─ guide.py                # guide loader + resolver — the ONLY place paths are set
│  ├─ style.py                # THE theme module — see §4
│  ├─ codons.py  accessibility.py  metrics.py  io.py
├─ figures/
│  ├─ make_fig01.py … make_fig04.py      # each: --guide G --results R --out O
│  ├─ make_sA_spike.py  … make_s14.py
│  ├─ panelise.py             # composite assembly — see §2.1
│  └─ README.md               # figure -> script -> input table -> licence
├─ data/                      # small public inputs only
├─ results/                   # derived tables as gzipped TSV (not parquet: see RUNS.md/DATA.md); no plots, no caches
├─ output/                    # generated figures, PDF + PNG (gitignored)
└─ tests/
   ├─ test_guides.py          # every shipped guide validates and resolves
   ├─ test_data_integrity.py  # ← input-data red team (§5 Phase 6)
   └─ test_figures.py
```

**Nothing below `figures/` may contain a literal lineage name, model name, virus or
count.** If a figure needs to know there are thirteen of something, it counts them
from the guide. A test asserts this by grepping the figure scripts.

### 2.1 Composite assembly — the panelisation code exists

❗ **Correction to the repo's own docs.** Both `README.md` and
`results/final_figures/readme_origin.md` state the composites have no generator and
that the notebook "lives in another repo". It does not — it is at
**`results/Lytras_OG/plots/per_model/ESM2_650M_HA80/panelise.ipynb.ipynb`** (note the
doubled extension, which is likely why greps missed it).

What it is: PyMuPDF (`fitz`) based. A `figure_list` dict maps panel keys
(`1A`, `1B&C`, `1D`, `2A`, `2B`, `2C`, `3`) to file paths, and two functions —
`generate_composite_figure(fig_keys, layout, …)` and `generate_dynamic_row_figure(…)`
— stack them and draw panel letters.

**Note `2C` is `plot_PLM_epoch.png`.** The epoch figure is a *panel of composite
Figure 2*, not a Figure 3 panel — which resolves the confusion behind D3.

Problems to fix when packaging it:

1. **Every path points at the old repo** (`/home3/oml4h/PLM_SARS-CoV-2/…`). The
   `figure_list` dict must become a guide-driven mapping.
2. **`generate_dynamic_row_figure` is defined twice** (cells 3 and 4) with different
   signatures; the second silently overrides the first. Keep one.
3. **Panel letters are placed by hand-tuned absolute coordinates** — e.g.
   `("C", 950, 45)`. If any panel changes size, the letters land in the wrong place
   and nothing warns. Derive positions from the panel rectangles instead.
4. It hard-codes a font path, `/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf`.
5. Adds a **PyMuPDF** dependency that is not in the environment spec.
6. The notebook's final cells only build the `panelstrim` Figures 1 and 3. The
   `panels2` Figure 2 variants (including `Figure_2_Composite_Asymmetric`) came from
   an earlier `generate_composite_figure` run that is not preserved in the saved
   cells. **Reconstruct and record that invocation.**

♻️ **`test_custom_labels.png` and `test_label.png` at the repo root are not litter.**
They are dated 7 Oct and match this notebook's `custom_labels` parameter — they are
recent test output from exactly this work. Removed from the red-herring list.

**Rule inherited from `CLAUDE.md`:** everything on main is reachable from a figure.
If a script, table or directory does not feed a figure or a stated claim, it is
archived, not kept in case.

---

## 3. Figure set

Status key: **R** ready (generator in repo, tracked) · **U** generator in repo but
untracked · **T** generator only in gitignored `tmp_explore/` · **N** no generator ·
**B** needs building.

### Main figures

| Slot | Content | Best existing asset | Generator | St | Action |
|---|---|---|---|---|---|
| **Fig 1** | Tree + overview panels (`1A` tree, `1B&C` ranked mutations, `1D` codon→AA heatmap) | `panelstrim/Figure_1_Composite.pdf` | **`panelise.ipynb.ipynb`** — found, see §2.1 | **U** | `1A` (tree) is GISAID-derived and frozen. The *assembly* is reproducible once packaged. Back the composites up outside git before any cleanup. |
| **Fig 2** | PLM prob vs mutational accessibility within codon-distance strata; + Grantham; + BLOSUM62 as panel 4; asymmetry ("reversible or not") | `Fig2_plm_vs_accessibility_by_codon_stratum`, `Fig2b_within_stratum_correlation`, `S10_biochemical_property_baselines`, `Fig5_aa_pair_direction` | **strata/property panels:** `tmp_explore/strata_explore.py` + `tmp_explore/aa_properties.py` (`PROPERTY_SCORES = ["blosum62","neg_grantham"]`); **asymmetry panel:** `scripts/plot_aa_pair_matrices.py` (panel F) | **T/R** | ⚠️ **No assembling script exists** — see §3.1. Promote generators out of `tmp_explore`. Decide panel count (D2). Caption must state BLOSUM62 is *not* a biochemistry control. |
| **Fig 3** | AUROC of evolutionary prediction vs alpha, 13 lineages | `Fig3_alpha_sweep_13lineages_bootstrapCI` | **`scripts/run_mutational_accessibility.py --regen-figures-only --ci-style boot`, driven by `scripts/build_key_plots.sh` §1b** | **R** | ❗ *Not* `plot_alpha_sweep_ci.py` — the 5 Oct plan names that, and it is wrong (see §3.1). Resolve the epoch question (D3). Re-theme per §4.3. |
| **Fig 4** | **DMS fitness vs the α-weighted score** (D1 ✅) | `results/dms/figures/a31_alpha_vs_dms` | **tables:** `tmp_explore/analyses/a31_dms_inference.py` → `a31_dms_joined.parquet`; **plot:** `tmp_explore/analyses/a31_alpha_vs_dms_plot.py` | **T** | Generators **do** exist — the repo's own docs are wrong (risk 8). Promote both. The plotter self-checks against `a31_alpha_curves.tsv` to 1e-6 — keep that check. |
| **Fig 5** | **Learning codon space** (D9 ✅) | `output/fig05_codon_learning.{pdf,png}` | `figures/make_codon_learning.py --guide guides/h3n2_5lineage.yaml` | **R** | Promoted 13 Oct 2026. Evaluated on single reference sequence (`J.2_int`, matching Fig 2D and S4) to eliminate cross-lineage pseudo-replication across closely related variants sharing >98% sequence identity. Panel A: S→N worked example (AUROC 0.871). Panel B: 54 pairs, 929 easy and 1,079 hard site evaluations. Panel C: pooled AUROC (fine-tuned 0.608 vs null p=0.001; base 0.509 vs null p=0.532). |

### Supplementary figures

> ⚠️ **The slot numbers below are stale as of 15 Oct 2026.** Against the manuscript
> draft's current ordering: contact recovery is **S8** (here S13), the SARS-CoV-2 DMS
> sweeps are **S12** (here S18), the within-stratum figure is **S13** (here S12), and the
> simulation control is **S14** (here S15/S16). S12 and S13 *swap*. The lineage-set
> validity audit is **dropped** from the numbered set and retires to
> `dev/retired_figures/`; its findings stay in
> [`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md) §4. The full mapping, the decisions behind
> it and the staged fix are in
> [`docs/notes/extended_red_team_tests.md`](extended_red_team_tests.md) §A1b. Expect another
> re-ordering once the manuscript is finished — which is why the fix puts the numbers in
> a map file rather than back into filenames.

| Slot | Content | Best existing asset | Generator | St | Action |
|---|---|---|---|---|---|
| **S1** | SC2 mutation correlations | `dev/fig4_gauge_sc2/`, `tmp_explore/sc2/`, `sc2_vs_iav_mutation_spectra.pdf` | `scripts/plot_sc2_vs_iav_heatmaps.py` + tmp_explore a13/a14/a24 | **T/R** | Likely needs building as the SC2 analogue of Fig 2. |
| **S2** | PR-AUC vs normalised PR-AUC vs partial AUROC | `S03b_displaced_metrics_per_lineage.png` | `scripts/plot_fig3_supplementary.py` | **U** | Ready — just track the script and re-theme. |
| **S3** | **How well ESM-2 650M and ESM-C 600M (both untuned) recover a hidden amino acid, across a panel of viral proteins.** Every position masked; dot = mean probability given to the true residue. No fine-tuning (decision 12 Oct) | `tmp_explore/IAV_vs_SC2_criss_cross/` panel (32 UniProt proteins, 21,003 residues) | `score_panel_allsites.py` (GPU) → `tools/export_viral_panel.py` → `figures/make_sC_viral_panel.py` | **R** | **Built 12 Oct.** Scored on `gpu1` (ESM-2 40 min, ESM-C 50 min). Sequences in `data/viral_panel/`, scores in `results/viral_panel/`. Run log: [`S3_S11_PROGRESS.md`](S3_S11_PROGRESS.md). |
| **S4** | Partial Spearman: PLM prob vs Grantham and vs accessibility | `S11_partial_correlation_property_removed.png` | `tmp_explore/strata_explore.py` | **T** | Promote with Fig 2. Caption carries the BLOSUM caveat. |
| **S5** | Heatmap of mutation decompositions, **plus the generic stand-in ranking (B, C) since 14 Oct 2026** | `S08_cross_species_matrix_heatmap.png`, `S09_generic_matrix_comparison.png` | `tmp_explore/analyses/a16_cross_species_matrices.py` | **T** | Promote. Note a snapshot exists at `results/mutation_codon_learning/scripts_snapshot/a16_*`. |
| **S6** | Mutation correlations within strata, several viruses | `a35a/b/c_spectrum_panels_13lin`, `crisscross_stats.tsv` | `tmp_explore/analyses/a35_spectrum_panels.py` (+ `a36_spectrum_delta.py`) | **B** | ⛔ **Nothing in the repo loops over >2 viruses within strata.** `strata_explore.py`'s `--mut-model` is single-choice, never a loop. `a35` varies the *spectrum* (14 conditions × 4 strata) but the protein is always influenza HA. Natural substrate is `crisscross_long.parquet` extended with the S3 viral panel. |
| **S7** | Epistasis distributions against null | `Fig4_epistasis_against_null.png`, `S14`, `S15` | `scripts/figure4_epistasis/fig4_null_analysis.py` | **R** | Best-prepared item. Resolve which gauge is plotted (D6). |
| ~~**S8**~~ | *Withdrawn as a figure 13 Oct 2026 — merged into S13 as panel D.* Antigenic residues vs the rest of the domain | `dev/fig4_gauge/antigenic_sites.{png,pdf}` | `figures/make_sH_contact_recovery.py` (`antigenic_stratum`) + `data/h3_antigenic_sites.csv` | **Merged** | **The result is a null** — risk 9. It did not earn a figure of its own, so it is one panel: the five sites pooled into one class, headline zero-sum gauge only. The per-site A–E and pair-level panels are dropped. |
| **S9** | Codon learning all-pairs tracking | `output/sI_codon_pair_tracking.{pdf,png}` | `figures/make_codon_learning.py` | **R** | Shipped alongside Fig 5. Tracks all 54 amino acid pairs side-by-side with 95% site-bootstrap intervals. |
| **S10** | DMS inter-dataset correlation vs AA distance | `results/dms/figures/dms_correlation_vs_aa_distance` | `scripts/plot_dms_correlations.py`, `build_dms_cross_lineage_pairs.py` | **U** | Track both. Tables already in `results/dms/tables/`. |
| **S11** | nt changes per codon per **internal** branch, H3N2 HA UShER tree, branches <10 nt | — | `tools/branch_codon_tables.py` → `figures/make_sM_within_stratum.py` | **R** | **Real tables exist since 12 Oct** (22 Sep tree). Inputs came from the IAV database, not the archive: see §3.2 and [`S3_S11_PROGRESS.md`](S3_S11_PROGRESS.md). Re-run on the 2 Oct tree once a `.pb` exists. |
| **S12** | DMS performance gap by nt-step | `a31_within_stratum`, `a37_dms_step_collapse` | `scripts/plot_dms_within_stratum.py` (live, Oct 5) + `tmp_explore/analyses/a37_dms_step_collapse.py` | **U/T** | Table `a31_within_stratum.tsv` already carries `dataset, trait, model, role, stratum, n, n_sites, rho, ci_lo, ci_hi`, CI by cluster bootstrap over **sites**. ⚠️ **No significance test exists** — risk 10. |
| **S14** | **Lineage-set validity audit** — codon tiebreak, independence, corpus leakage | `dev/lineage_validity/lineage_validity_summary.{png,pdf}` | `dev/lineage_validity/plot_lineage_validity.py` (+ 4 analysis scripts) | **R(dev)** | ✅ Built 9 Oct, already in house style. 4 panels: **A** strand B replicates on both lineage sets · **B** strand C null on both · **C** advantage is not memorisation · **D** interval depends on resampling unit. Full report `dev/lineage_validity/report.html`. See [`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md) §4. |
| **S13** | Contact performance by region; Zhang zero-sum AUROC; antigenic residues vs the rest (panel D, was S8) | `strata_benchmark_{6WXB,7ZJ7}.tsv`, `positional_accuracy.tsv`, `surface_vs_core_benchmark.tsv` | `dev/fig4_gauge/plot_strata_per_structure.py`, `plot_positional_accuracy.py`, `analyze_surface_vs_core.py` | **R(dev)** | Promote out of `dev/`. "Zhang zero-sum" = the `zerosum` gauge, log P(c) − mean log P(d), Zhang & Ovchinnikov 2024 double-centred Potts gauge. Ties to D6. |
| **S15** → **S14** | **Reviewer 2's Simulation Control** (4 arms: human, SC2, H3N2, JC69 flat neutral control; empirical spectra, directional contrast on founder, partial correlation holding fitness fixed, debiasing recovery) | `clean_repo/results/sim_v3_star90/` (`replacement_scores.tsv.gz`, `empirical_substitution_spectra.tsv.gz`) | `figures/make_sN_simulation_control.py`, `tools/export_table_s15.py` | **R** | Built 13 Oct. Shipped: `output/sN_simulation_control.{pdf,png}`, `output/supl_table_16.tsv`, `output/table_sN_simulation_control.tsv`, and DMS alpha sweep `output/table_s16_dms_alpha_optima.tsv`. ⚠️ **Three corrections, 15 Oct.** The generator was named here as `tools/export_table_s16.py`, which does not exist — it is `export_table_s15.py`. The figure is **S14** in the draft, not S15 or S16. And `make_sN_simulation_control.py` writes the same frame to three filenames; two are being dropped. See [`docs/notes/extended_red_team_tests.md`](extended_red_team_tests.md) §A1b. |

### 3.1 Provenance corrections for the main figures

Three claims in the repo's own documentation are wrong. Fix them at the source, or
the next person repeats the search.

| Claim | Where | Truth |
|---|---|---|
| Fig 3's source stem is `fig3_alpha_sweep_prauc_band_boot` | `figures/key_revision_figures/README.md` table row | `collect_revision_figures.sh:51` copies `fig3_alpha_sweep_**auroc**_band_boot`. The PR-AUC version is now S03. The README *prose* is right; only its table row is wrong. |
| Fig 3's generator is `plot_alpha_sweep_ci.py` | `dev/final_steps/final_steps_plan.html` §04 | Wrong. That script is Sep 10, reads from a different source, writes a *t-interval* band, and its only outputs were deleted. The shipped Fig 3 (Oct 5 15:13) matches the `build_key_plots.sh` run window exactly. **`run_mutational_accessibility.py` via `build_key_plots.sh` is the live generator.** |
| Fig 5's generators are `a38`/`a39` | same | Those two write **zero figures**. `a40_side_analyses_and_figures.py` is the actual plotter. |

**Fig 1 is definitively ungeneratable.** Three independent documents say so, and a
repo-wide grep for `Bio.Phylo|toytree|ete3|baltic|dendropy` returns exactly three
hits, none of which draw the manuscript tree (one draws *simulated* topologies; two
are mmseqs clustering). `scripts/analyze_segment4_tree_origins.py` reads the tree but
draws **scatter plots, not a tree**. Treat Fig 1 as a frozen supplied asset.

**Fig 2 has no assembling script, and spans two lineage sets.** `strata_explore.py`
runs on the **5-lineage** `combined_long_table.csv`; `plot_aa_pair_matrices.py` runs
on the **13-lineage** parquet. Combining them into one multi-panel Fig 2 is currently
a manual step that exists nowhere in code. The legend must state which set each panel
used.

**Fig 2 red herrings.** `tmp_explore/report/build_aa_pair_heatmap.py` is *newer* than
`plot_aa_pair_matrices.py` and also builds 20×20 ordered-pair heatmaps — but it is a
different figure for a different document (within-pair AUC, not asymmetry) and feeds
`codon_or_amino_acid.html`. Likewise `scripts/regress_1nt_stratum.py` is superseded by
`results/mutation_codon_learning/scripts_snapshot/regress_1nt_spectra.py` +
`plot_regress_1nt.py`. And `scripts/pairwise_probability_correlations.py` is dated
**27 April** — pre-revision, almost certainly dead.

**Open gap (unchanged from the 5 Oct plan):** `results/nextstrain_13lin_dense` — the
101-point −1..1 grid that **all of Fig 3 and S01–S04b rest on** — came from a manual
`run_mutational_accessibility.py` invocation whose command line was never recorded.
Only `dev/logs/dense_sweep.log` survives, and it records output but not the command.
`run_all_figures.sh` builds a *different* directory (41-point grid out to α = 3.0).
**Recover and commit that command line.**

### 3.2 S11 — `branch_codon_distance.py` rework

> **Update 12 Oct: the blockers are gone and real tables exist.** The three missing inputs all came
> from the IAV database (`iav-db.rebuilt.db`, read-only): the UShER reference record `DQ864721` (aligned
> row in `sequence_alignment`), the tip list (every tip of the tree), and the frame (checked against 1,489
> VCF sites). Tree: `h3_seg4_tree.pb` of 22 Sep (67,879 tips, all inside the DB's 2 Oct segment-4 tree of
> 572,119 tips; the DB holds that tree as Newick only, so no per-branch mutations until a `.pb` exists).
>
> ⚠️ *Two corrections, 15 Oct.* **(a)** `iav-db.rebuilt.db` no longer exists — it is now
> `iav-db.db` in the same `db_build/` directory (risk 23). **(b)** The 22 Sep 67,879-tip tree
> described here is **not** the one S11 shipped on: `results/s11_h3n2_ha_tree/MANIFEST.tsv` and
> `summary.txt` both name `h3n2_1968.pb`, 224,731 tips, pruned from the 24 Sep treepatch repair.
> The shipped numbers are from that tree and are sound (risk 25).
>
> **A trap found on the way.** The UShER reference is **not an H3 sequence**, so its codon numbers are not
> H3 numbers, and 18 of its 553 codons have no H3 residue at all (alignment gaps). Those columns produced
> most of the apparent multi-position changes: with them, 146 amino-acid-changing codons changed 2
> positions and 261 codons were dropped as stops; without them, 40 and 1. `tools/h3_numbering_from_db.py`
> gives each codon its mature H3 number (checked against the conserved Y98, W153, H183, Y195) and
> `branch_codon_tables.py --skip-unnumbered` leaves the 18 out. **Use that setting for the paper.**
>
> **Result (22 Sep tree, internal branches under 10 nt, H3-only):** 20,589 branches. 15,898 carry 1
> change, 3,016 carry 2, 1,005 carry 3, 427 carry 4, and 243 carry 5 to 9. In those branches 8,788
> amino-acid-changing codons changed 1 position, 40 changed 2 and 1 changed 3 (plus 1 synonymous codon with
> 2). Under a crude even-spread null, about 13.5 same-codon pairs are expected; 42 are seen, so there
> is some clustering, concentrated at antigenic sites (H156 recurs 12 times). The caveat from below
> stands: whether the tree was built with `faToVcf -includeNoAltN` is not recorded.
>
> **Table at the top of the figure (added 12 Oct).** Row 1, one branch at a time: of 8,829 amino-acid-changing
> codons, 8,795 (99.6%) needed 1 nucleotide change from the parent codon, 33 needed 2, 1 needed 3. Row 2,
> oldest H3 node to the 12,665 newest internal nodes: a typical node differs by 82 amino acids; 66 (80%) need 1
> nucleotide, 14 (17%) need 2, 2 (3%) need 3. Changes needing 2 or 3 nucleotides are nearly absent on a single
> branch but are about a fifth of the long-range differences, so they are made in steps. Counted per node
> (shared history is counted for each node); the oldest node's codons are inferred.

**Current state: the script has never successfully produced output.** No
`branch_subs.tsv.gz`, `summary.txt`, `pb.nwk`, `pb_paths.txt` or `pb_ref.vcf` exists
anywhere in the repo. A single `.pyc` dated Oct 5 17:16 (py3.10, while every other
`.pyc` there is 3.12/3.13) shows one run of the *companion*
`site_variation_codon_distance.py`, which imports it. Live intent, never landed.

**Two inputs are missing and are not in the repo at all:**
- `--aln` — the alignment UShER was built from; its *first record must be the UShER
  reference*.
- `--tips` — the target tip names, one per line.

❗ **The plan doc is wrong about the inputs.** `h3_mutations.tsv` and `h3_samples.tsv`
are **not read by this script**. They are consumed by
`results/mutation_codon_learning/scripts_snapshot/a39_per_site_codon_ceiling.py`.
Only `h3_seg4_tree.pb` is an input here, plus `matUtils` on PATH.

#### Why the row count explodes

Row granularity is **one row per (branch, codon) amino-acid-changing event** — not per
site, not per amino-acid pair. There is **no aggregation anywhere**; the script dumps
every event. Compounding factors:

1. The tree has **20,682 internal nodes and 67,879 samples**, and the walk descends
   into every branch containing a target tip.
2. **Recurrence is never collapsed.** The repo's own notes record that *the median
   mutated site has 167 independent origins* on this tree. All 167 become separate
   rows with identical `codon_pos`/`aa_from`/`aa_to`.
3. `--max-branch-nt 10` caps **nucleotide changes per branch, not rows** — a 10-nt
   branch touching 10 codons still yields 10 rows. It barely reduces the count.
4. No restriction to the HA coding region, a lineage, or a date window.
5. The `--purity 0.95` escape hatch: if `--all-tips` defaults to `--tips` and the
   targets already dominate, the start node becomes the root and the walk covers the
   whole tree.

Expect 10⁴–10⁵ rows. **The table you actually want already exists** — it is the
`crosstab(n_obs, n_min)` printed into `summary.txt`. The TSV is an intermediate that
was never meant to be read row by row.

#### Why the cells are ambiguous

| Column | The trap |
|---|---|
| `codon_pos` | **The biggest problem.** It is `(alignment_column // 3) + 1` — an *alignment-column* codon index. Not HA numbering, not mature-H3 numbering, not reference-record numbering. Any gap block upstream shifts every downstream index. The only frame check is `len(ref) % 3 == 0` on the total width, which does not guarantee the frame survives. |
| `n_min` | **The most misleading column.** It is the minimum Hamming distance over *all* codons of `aa_from` × *all* codons of `aa_to`, so it ignores which synonymous codon the ancestor actually had. The docstring claims `obs > min` means "the branch took a longer route than the code requires"; in reality it usually just means the ancestor used a synonymous codon further from the target. |
| `codon_from` / `codon_to` | **Reconstructed, not observed.** Never-mutated positions take the reference; for reference-gap columns that is the majority base over the first 5,000 records, or the literal `"N"`. Combined with UShER's missing-equals-reference convention, a codon can describe sites that were never sequenced. |
| `n_obs` | Positions that *differ*, not mutations placed. Two mutations at one position, or a revert, makes `n_obs` < the true count. |
| `terminal` | **Constant 0** under default settings — terminal branches are skipped before a row is emitted. Only varies with `--include-terminal`. |
| `isolated` | Defined as `branch_nt_total == n_obs`, so it inherits `n_obs`'s undercount: a genuinely isolated codon can read `False`. |
| `branch_nt_total` | All nt mutations on the branch, syn and nonsyn, in and out of this codon — repeated verbatim on every row from that branch. |

Other defects: `dropped_stop` also increments for unknown/ambiguous codons, so
`summary.txt` over-reports stops. And the filter is `len(ms) > max_nt`, i.e. branches
with **≤10** nt are kept — off by one against the stated "<10 nt" spec.

#### Required rework

1. **Branch-wise table.** One row per branch (branches with <10 nt changes — fix the
   off-by-one). Columns stating plainly how many codons on that branch took 1, 2 or 3
   nucleotide changes.
2. **Column-wise table.** One row per codon-alignment column: how many times that
   column underwent a 1-, 2- and 3-nt change over the whole tree.
3. Keep the event dump behind an optional `--dump-events`.
4. Translate `codon_pos` to reference-record codon index, ideally mature-H3 numbering,
   and **assert the frame explicitly**.
5. Add `n_min_from_parent = min(hamming(codon_from, c) for c in AA2CODONS[aa_to])` and
   make *that* the headline comparator; keep the present column as `n_min_code`.
6. Split `dropped_stop` from `dropped_ambiguous`; drop `terminal` unless
   `--include-terminal`; recompute `isolated` from the per-codon mutation count.
7. Every column gets a one-line definition in the figure README.

Move `h3_seg4_tree.pb` out of the repo root into `data/`.

⚠️ **Known trap:** `faToVcf` needs `-includeNoAltN`, or UShER silently reads
uncovered sites as the reference base.

### 3.3 Fig 5 — three codon strands, and why they looked contradictory

**RESOLVED 9 Oct — and the resolution went the other way from the first reading.**
B and C test *different hypotheses*, and running each statistic on *both* lineage
sets shows neither depends on the lineage set:

| Statistic | 5-lineage | 13-lineage |
|---|---|---|
| Strand B (within-replacement AUROC), fine-tune | 0.6088 [0.5710, 0.6387] | **0.6104 [0.5733, 0.6414]** |
| Strand B, base | 0.5106 [0.4852, 0.5407] | 0.5105 [0.4875, 0.5406] |
| Strand C (within-residue rho), fine-tune | −0.030, p = 0.27 | −0.037, p = 0.22 |
| Strand C, base | −0.026, p = 0.30 | −0.046, p = 0.16 |

Strand C is **null on five lineages too**, where B is strongly positive — so it was
never a refutation of B. **Strand B's result stands.** Full detail in
[`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md) §4; figure S14.

The original comparison below is kept because it records what the two strands are.

| Strand | Code | Data | Result |
|---|---|---|---|
| **A** — compression / codon-usage ceiling | `a38_codon_usage_ceiling.py` + `a39_per_site_codon_ceiling.py` (tables, 0 savefig) → **`a40_side_analyses_and_figures.py`** (the actual plotter) | — | Figures `fig1`–`fig4` in `results/mutation_codon_learning/figures/` |
| **B** — synonymous-codon discrimination (the S4 test) | `a20_codon_randomisation.py` → `build_s4_figures.py` → `extract_report_svgs.py` | **5-lineage** parquet | **Positive.** AUC 0.608 [0.572, 0.645] vs conservation-stratified null 0.520; 46/54 pairs above 0.5; sign test p = 1.4e-7. Base model sits on its null. |
| **C** — the 28 Sep re-test | `results/mutation_codon_learning/synonymous_permutation/run_test.py` | **13-lineage** parquet, permutation null, pre-specified, seed 20260928 | **Null.** Its own README opens: *"Result: null. The omnibus test finds nothing, the fine-tuned model is no better than the untuned base."* Pre-specified amino acid not significant on the choice-free statistic (p = 0.13). One hand-pooled variant reaches p = 0.037, and §4 of that README explains why that is not independent support. |

B and C test closely related hypotheses with different statistics and reach opposite
conclusions. Points worth weighing:

- **C is the better-designed test.** It is pre-specified, uses the 13-lineage set, and
  uses a permutation null. Its README argues — correctly — that holding the amino acid
  fixed is *stronger* than partialling Grantham out, because within a synonymous
  stratum BLOSUM62, Grantham and every other 20×20 matrix have variance **exactly
  zero**. There is nothing left to partial.
- **B's chain is broken in two places.** `build_s4_figures.py` calls an
  `insert_s4_figures.py` that does not exist, and `codon_or_amino_acid.html` is
  hand-authored with no builder. The `_s4_*.svg` files are hand-copied and go stale
  silently — the extracted SVGs are dated 5 Oct while the underlying `a20_s4_*.tsv`
  are 8 Sep.
- The 5 Oct plan already made Fig 5 conditional: *"Codon learning is Fig 5 only if the
  synonymous-codon test survives the null; otherwise Supp."*

~~**My read:** on the evidence as it stands, C supersedes B...~~ **Superseded.** That
read was wrong, and the way it was wrong is worth keeping: I treated "newer,
pre-specified, more lineages" as decisive without first checking whether the two
statistics even measure the same thing. They do not. The tiebreak that settled it
took one script and ten minutes — running it before forming a view would have been
the better order.

### 3.4 Items that need a GPU re-run before they can exist

Not blocked on code — blocked on compute. Budget for these early or cut them.

| Item | What is missing |
|---|---|
| ~~**S3** — fine-tuning success across viruses~~ | **Withdrawn 12 Oct.** S3 is now the two untuned models only. The all-site scoring is a short inference job on `gpu1` (about 15 minutes per model), not training. No fine-tunes are needed. |
| **Fig 3 epoch panel** (if D3 says rebuild) | Per-epoch ESM-C 600M checkpoints exist in `dev/epoch_adar/` but that is a **different backbone**. Rebuilding on ESM2-HA80 means a fresh fine-tuning run. |

### 3.5 Red herrings — confirmed dead or superseded

Do not port any of these. Verified by mtime, by grep for the writing script, and by
the presence of a successor.

| Item | Why it is dead |
|---|---|
| `results/dms/figures/a31_fitness_vs_supply_tradeoff.png` (S17) | No generator anywhere; Sep 10, pre-dates the backbone fix; sibling archived as `…_STALE_2026-09-10.png`. **Numerically withdrawn**, not just ungenerated. |
| `dev/a37_recheck/` | A frozen before/after snapshot pair from Sep 23. |
| `dev/fig4_gauge/headroom_by_site.tsv` | 3-row stub (sites 160/144/158 only). |
| `dev/fig4_gauge/generate_og_figures.py`, `plot_gauge_comparison.py`, `plot_comparison_heatmaps.py`, `generate_cells_table.py`, `distance_correlation_summary.py` | The pre-split Sep 23–24 deck, superseded by `plot_5way_gauge_matrix.py` + `build_master_table.py` (Oct 7–8). `README_BASELINE.md` still describes the old deck. |
| `generate_calibrated_heatmaps.py` → `surface_vs_core_heatmap_calibrated.*` | Third parallel surface/core plotter, superseded Sep 24. |
| `dms_correlation_vs_aa_distance.*`, `dms_concordance_spearman_vs_pearson.*` (Sep 28) | Orphaned output of an earlier revision of `plot_dms_correlations.py`; the current script writes `dms_pearson_vs_aa_distance.*` (Oct 7). |
| `results/mutation_codon_learning/synonymous_permutation/superseded/` | Self-declared. |
| `.coverage`, `coverage.xml`, `htmlcov/` | Stale coverage artefacts (18 Sep). *(`test_custom_labels.png` / `test_label.png` were here but are **not** dead — see §2.1.)* |
| `scratch/`, `notes/` | Both empty. |
| `scripts/` growth & renewal family (~7,000 lines) | `growth_core`, `plot_growth_*`, `renewal_transmission_fitness`, `estimate_within_lineage_variant_renewal`, both `generate_*_report`, `test_growth_positive_control`. Not in the figure plan; HA-growth is a workaround for a DB defect, not the goal. Archive with their tests. |
| `scripts/generate_progress_report_20260922.py` (1,650 lines) | Dated progress report. |

⚠️ **Naming trap:** `dev/fig4_gauge/` is **not** the generator for manuscript Figure 4.
It is a structural-validation bake-off that decides *which gauge* the epistasis
figure should plot. Manuscript Fig 4 (epistasis) comes from
`scripts/figure4_epistasis/`. Under the new numbering epistasis drops to
supplementary and DMS becomes Fig 4 — so `fig4_gauge` will support a *supplementary*
figure while carrying "fig4" in its name. **Rename on port.**

### 3.6 Existing figures with no slot in the new list

`S01`–`S07` (alpha-sweep band definitions, per-lineage values, frequency
diagnostics, misclassified replacements), `S12_codon_randomisation`,
`S13_step_penalty`, `S16`–`S18`, `S24`. Each needs one pass asking: *which claim
does this support, and is that claim in the paper?* Several answer reviewer
comments rather than supporting the paper and belong in the **response letter**,
not the supplement.

- [ ] Pass done, each assigned to: figure / response letter / archive.

---

## 4. House style

One module, `mutacc/style.py`, imported by every figure script. No script sets its
own rcParams, dpi, figure size or colour.

### 4.0 What exists today: nothing

There is **no shared theme in the repo**. No `.mplstyle`, no `set_style()`, no style
module. 70 files call `savefig`. The measured spread:

| Thing | How many distinct values in use |
|---|---|
| Hard-coded hex literals | **258** across `scripts/` + `dev/` |
| `figsize` tuples | **79** — nothing repeats more than 5×; range `(6,5)` to `(28,5)` |
| `savefig` dpi | 4 — `300` (95×), `200` (37×), `170` (5×), `190` (1×) |
| `facecolor` "white" | 6+ spellings, incl. near-whites `#F8FAFC`, `#f1f5f9` — some figures are faintly blue-grey |
| `INK` constant | defined **9 times** with **5 different values** |
| `ax.grid(...)` | ~12 spellings |
| Type scale | 2 incompatible ones, both commented "Manuscript type scale" |
| Diverging colormap | 3 in use (`RdBu_r`, `coolwarm`, …) for what is likely one kind of quantity |

Only `bbox_inches="tight"` is already consistent (67/67).

**Four competing base/fine-tune colour schemes are live, and two invert the
hue→role mapping** — one makes the fine-tune blue, viridis makes the *base* blue:

| Scheme | Base | Fine-tune | Where |
|---|---|---|---|
| viridis n=2 (published) | `#31688e` | `#35b779` | `plot_fig3_supplementary`, `growth_core`, `plot_lineage_operational_metrics`, +4 |
| blue/orange A | `#eb6834` | `#2a78d6` | `plot_alpha_sweep_ci`, `plot_growth_rate_analyses` |
| blue/orange B | `#e8702a` | `#1e6fb8` | all of `dev/fig4_gauge/` **and** `dev/fig4_gauge_sc2/` |
| Okabe–Ito | `#0072B2` | `#D55E00` | `plot_dms_within_stratum` only |

The repo already *knows* about the contrast defect my audit found: `plot_fig3_supplementary.py`
carries a 20-line comment noting `#31688e` is 6.00:1 on white while `#35b779` is only
2.56:1, and works around it by giving the two colours **different alphas**. Adopting
Okabe–Ito blue/green removes the need for that workaround rather than papering over it.

### 4.1 Colour — validated, not eyeballed

The current manuscript pair **fails** a colourblind/legibility audit:

| Pair | Lightness | Chroma | CVD sep | Normal-vision | Contrast |
|---|---|---|---|---|---|
| `#31688e` base / `#35b779` fine-tune *(current)* | pass | **FAIL** — `#31688e` reads grey (0.084) | pass ΔE 24.0 | pass | **WARN** — `#35b779` 2.49:1, under the 3:1 floor |
| `#0072b2` / `#009e73` *(Okabe–Ito, proposed)* | pass | pass | pass ΔE 18.0 | pass | **pass** |

✅ **Decided (D8): adopt Okabe–Ito** — designed for colour-vision deficiency and
standard in biology journals. It is the only candidate tested that passes all five
checks. (Six viridis variants were tried; all failed the chroma floor, because
viridis blues are inherently desaturated.)

```python
BASE       = "#0072b2"   # base model            (Okabe–Ito blue)
FINETUNE   = "#009e73"   # fine-tuned model      (bluish green)
ACCESS     = "#d55e00"   # mutational accessibility (vermillion)
BIOCHEM    = "#cc79a7"   # Grantham / BLOSUM     (reddish purple)
NULL_GREY  = "#999999"   # nulls, chance lines, reference
```

Four-slot palette passes every check except a contrast WARN on `#cc79a7` (2.98,
on the line) — acceptable given direct labels are mandatory anyway.

Rules, all non-negotiable:

- **Colour follows the entity, never its rank.** `BASE` is always that blue, in
  every figure, forever. A filter that drops a series must not repaint the rest.
- **Sequential = one hue, light→dark. Diverging = two hues + a neutral grey
  midpoint. Never a rainbow, never a hue at the diverging midpoint.** Heatmaps of
  correlation are diverging and must be centred on zero.
- **Never a dual-axis chart.** Two measures of different scale → two panels.
- **Text wears ink, not series colour.**
- Re-run the validator on any new palette before it ships.

### 4.2 Geometry and type

| Token | Value |
|---|---|
| Single-column width | 3.46 in (88 mm) |
| Double-column width | 7.09 in (180 mm) |
| Font | one sans family throughout; 7 pt tick, 8 pt axis label, 9 pt panel title |
| Panel letters | bold, upper-left, outside the axes |
| Output | PDF (vector, for submission) **and** PNG at 400 dpi, same filename stem |
| Grid/axes | recessive — light grey, no top/right spines |

### 4.3 Audit of the current Fig 3 — what the re-theme has to fix

Read off the actual rendered `Fig3_alpha_sweep_13lineages_bootstrapCI.png`. These
faults are representative, not specific to this figure, and the style module plus
the §4.4 checklist exist to stop each one.

| # | Fault | Fix |
|---|---|---|
| 1 | **No panel letters.** The figure README says "Panel B is AUROC", but there are no A/B/C marks on the figure. A caption cannot reference panels that are not labelled. | Panel letters mandatory, bold, outside the axes. |
| 2 | **Legend carries raw model IDs** — `esm2_t33_650M_UR50D raw base model`, `HA80_fine-tune`. Meaningless to a non-ML reader. | "Base model (ESM-2)" / "Fine-tuned on influenza HA". Checkpoint IDs belong in the methods. |
| 3 | **Panel titles are column expressions** — `Spearman(PLM vs all possible AA freq)`. | Say what is being asked: "Does the model rank all possible replacements correctly?" |
| 4 | **Five lines of methods prose set as a grey footnote under the axes.** This is caption text living inside the image. | Move to the caption. Nothing below the axes but the x-label. |
| 5 | **Red used for a reference line** (`0.5 = chance`). Red is a status colour; spending it on a neutral annotation makes a real warning unreadable later. | Neutral grey `#999999`. Keep the text label — labelling it was right. |
| 6 | **Axis label leaks the variable name** — `mutational accessibility weighting Alpha (mut_acc^alpha) 0= PLM only`. | "Weight on mutational accessibility (α) — 0 = model alone". |
| 7 | **Unlabelled dashed lines** at x=0 and y=0 compete with the labelled chance line. | Label or drop. |
| 8 | **Internal naming in the title** — "Focused alpha-sweep metrics (all models)". | Plain description of what the reader is looking at. |
| 9 | Very wide aspect with dead space reserved for the footnote. | Resize to the column grid in §4.2 once the footnote moves out. |

⚠️ **Verify, do not assume:** the middle panel runs down to AUROC ≈ 0.20 at α = −1,
so it appears to use **unclamped** rank statistics. But the pipeline's *stored*
`site_logistic_auroc` is clamped at 0.5 (risk 4). Confirm which column this panel
actually reads before re-plotting. If any published panel reads the stored column
below α = 0, it is drawing the clamp.

### 4.4 Legibility for a non-specialist reader

The user's explicit requirement. Every panel must satisfy:

- [ ] Y-axis says what it **means** in plain words, not the column name.
      "Model probability of this replacement" beats `plm_prob`.
- [ ] The take-home is written on the panel as a short annotation, not left to the
      caption.
- [ ] Every reference line is labelled with what it is — "chance = 0.5", not a bare
      dashed line.
- [ ] Acronyms expanded on first use in the caption: AUROC, PR-AUC, DMS, PLM, LOO.
- [ ] **α is never shown bare.** It is the accessibility exponent; say so on the
      axis. Never reuse α for a false-positive rate — write FPR as FPR.
- [ ] ≥2 series → legend always present; ≤4 series → also direct-labelled, so
      identity is never carried by colour alone.
- [ ] Direct labels are selective. Never a number on every point.

### 4.5 Migration hazards — things that will fight the new theme

1. 🔴 **The seaborn fallback is actively wrong and reachable today.** Five scripts do
   `try: import seaborn … except: MODEL_COLOURS = [(0.267,0.005,0.329),(0.993,0.906,0.144)]`.
   `sns.color_palette("viridis", 2)` *trims* the ramp and returns blue/green; the
   fallback literals are the raw viridis **endpoints** — dark purple and yellow. So
   without seaborn the same model silently wears a different colour. **The default
   `python` on PATH has no seaborn**, so this fires now, with no error and no warning.
   Fixed in `plot_lineage_operational_metrics.py` only. Still broken in:
   `sweep_misclassified_mutations.py:46`, `plot_frequency_diagnostic.py:53`,
   `plot_lineage_values_at_alpha.py:47`, `plot_geometry_vs_spectrum.py:48`,
   `regress_1nt_stratum.py:59`. All five are marked `# pragma: no cover`, which is
   exactly why it survived. **One-line fix each — highest value per keystroke in the
   whole job.**
2. **`plot_dms_correlations.py` calls `plt.style.use("default")` six times**, per
   figure. It will reset any global theme. Remove those calls or the theme will not
   stick.
3. **`scripts/` is not a package.** Figure scripts are standalone `__main__` entry
   points; `tests/conftest.py` imports them by path via `load_script()`. The shared
   style module must be importable the way `growth_core` already is.
4. **`growth_core.py` is the de facto prototype** — it is the only place with a
   documented house-style block and a `save_figure()` helper, and 7 scripts import it.
   But it is also in the strip list (§3.5). **Lift the style block out of it before
   archiving the growth family**, or the prototype goes with the dead code.
5. **`rasterise_figures.sh` back-fills PNGs from PDFs but skips any PDF whose script
   already wrote a PNG** — so a figure's final raster dpi depends on which path
   produced it. Remove this ambiguity: one save helper, always PDF + PNG.

---

## 5. Work phases

### Phase 0 — Safety (do first, small)
- [ ] Commit a snapshot of the current index **and the 10 untracked `scripts/*.py`**.
      They are not gitignored, merely unadded; one bad `git add -A` / `reset` loses
      them. (`analyse_ha_epistatic_shifts`, `branch_codon_distance`,
      `build_dms_cross_lineage_pairs`, `map_jsd_to_ha_structure`,
      `plot_dms_correlations`, `plot_dms_rolling_concordance`,
      `plot_dms_within_stratum`, `plot_fig3_supplementary`, `rebuild_ranking_metrics`,
      `site_variation_codon_distance`)
- [ ] `git tag exploration-2026-10`.
- [ ] Back up `results/final_figures/panels*/` and `clade_viz_*` **outside git** —
      the repo's own notes say these are the only copies.

### Phase 1 — Freeze the figure list
- [x] **Main set settled: Fig 1–4.** Fig 4 = DMS vs the α-weighted score; Fig 1 keeps
      the composites and packages the panelisation code; codon learning moves out to
      `LINEAGE_VALIDITY.md`. Palette settled: Okabe–Ito.
- [ ] Close D2, D4, D5, D6 (§6). D5 and D6 gate S3 and S7/S13; D2 and D4 are cosmetic
      by comparison.

### Phase 2 — Skeleton
- [x] Create the `clean_repo/` tree.
- [x] **`mutacc/style.py`** — the house style module. Okabe-Ito roles, one dpi, one
      save helper writing PDF + PNG together, `reference_line()` that *requires* a
      label, `panel_letters()`, and `finalise_axes()` that **raises** on a bare column
      name like `plm_prob`. Imports no seaborn at all, which removes the recolour bug
      by construction rather than patching it.
- [x] **`tests/test_style.py`** — 29 tests, all passing. Pins the palette values,
      asserts every series colour clears 3:1 contrast, checks base/fine-tune separate
      in greyscale, and imports the module with `seaborn` blocked from `sys.modules`.
- [x] Rendered a layout proof at `output/demo_fig3_house_style.png` (synthetic data,
      real layout). First render had panel titles colliding with the next panel's
      letter — fixed. This is why §4.6 step "render it and look at it" is mandatory.
- [x] `mutacc/guide.py` — the guide loader and validator (§1 principle). Rejects absolute
      paths and `..`, unknown keys, duplicate ids, bad roles; carries `root_type`.
- [x] `environment.yml`. Tables ship as gzipped TSV, so **one** interpreter
      (`miniconda3/bin/python`) draws every figure and runs every test: risk 18 is solved by
      design rather than documented.

### Phase 3 — Data
- [x] Ship the derived tables the figures read, as gzipped TSV with a `MANIFEST.tsv` per run
      (about 12 MB). Done for Figs 2-4 and S2, S4, S10, S12; codon-learning and figure4_null
      tables are not shipped yet. Still to do: replace the blanket `results/*/tables/` ignore
      with an explicit allow-list when the repo is git-initialised for release.
- [x] `DATA.md`: for every figure, which table it reads and whether that table ships.

### Phase 4 — Port generators (largest block)
One `make_figN.py` per figure, reading only from `results/`. Order: start with the
**R** rows (cheap wins), then **U**, then **T**, then **B**/**N** last.
- [x] Main: Fig 2 · Fig 3 · Fig 4 · Fig 1 composite assembly (§2.1) — all ported, 9 Oct
- [x] Supp done: S2 · S4 · S10 · S12 · S6 · S7 · S13 (S8 merged in as panel D) · S14 · S9/codon learning · S4 (withdrawn) · S5
- [x] S11 built; **real tables from the 22 Sep tree since 12 Oct** (see §3.2 and `S3_S11_PROGRESS.md`). To repeat on the 2 Oct tree once a `.pb` exists.
- [x] S1 built 12 Oct (ESM-C base vs ESM-C spike fine-tune; see log).
- [x] S3 built 12 Oct (untuned ESM-2 and ESM-C, every position; see `S3_S11_PROGRESS.md`).
- [x] S15 built 13 Oct (Reviewer 2 simulation control: 4 evolutionary arms, empirical spectra, directional contrast, partial rho, debiasing recovery). Supl Table 16 exported to `output/`. **All planned figures and supplementary tables now exist.**

### Phase 5 — Re-theme
- [x] Every ported figure goes through `mutacc/style.py` (a test reads the source to enforce it).
- [x] Rendered, opened and fixed for collisions: Figs 2, 3, 4, S2, S12; Fig 1 composite.
      Also S4 (withdrawn) and S5 (12 Oct) — S4 (withdrawn) panel A needed two passes, and the fix became
      `style.reference_line(label_side=…)` rather than a nudge local to the figure.
- [ ] Confirm the same was done for S6, S7, S11, S13, S14 and codon learning. Their workers
      reported collisions fixed, but only S7's spare headroom and crossing leader line are
      recorded as still open (9 Oct). Open each rendered figure once before submission.
- [ ] Same for S1 and S3 when they exist; §4.3 checklist per panel.

### Phase 6 — Red team

**Baseline as of 8 Oct** (`/home3/oml4h/miniconda3/bin/python -m pytest tests/ -q -m "not esm_cpu"`,
2m30s, no GPU):

```
9 failed, 1566 passed, 2 skipped, 13 deselected, 26 xfailed
```
`dev/fig4_gauge/` 103 passed, 4 xfailed · `dev/fig4_gauge_sc2/` 47 passed.

The suite is a genuine strength and must be carried over. Notes:

- ⚠️ **Only `/home3/oml4h/miniconda3/bin/python` can run it.** The default `python`
  on PATH cannot even *collect* the suite (broken system `six`), and `/home3/oml4h/venv`
  has no torch. Pin this in the environment docs.
- ⚠️ `pytest.ini` sets `testpaths = tests`, so **a bare `pytest` never collects the two
  `dev/fig4_gauge*` suites**. 150 passing tests are invisible to CI as configured.
- Of the 9 failures: 4 are stale-world assertions (the repo moved, the test did not —
  e.g. one asserts no dependency manifest is committed, but `requirements.txt` now is);
  **5 are real drift** and need triage before porting, four of them in the DMS work
  (`test_dms_within_stratum`, `test_a31_dms`, `test_fig4_null_analysis` ×2).
- `tests/README.md` carries a current inventory of 21 red-team findings. Read it before
  touching `Functions_HuggingFace.py` or `figure4_epistasis/`.

**Coverage gaps that matter here:** ~34 of 48 scripts in `scripts/` have no test at
all, including almost every figure generator. There is exactly **one** colour
assertion in the repo and **no** image-regression testing of any kind.

- [ ] Triage the 5 real failures in the old repo's suite; fix or document. (Not started: the
      clean repo carries its own suite, 1.5 minutes, all passing.)
- [x] Palette unchanged without seaborn: `tests/test_style.py` blocks seaborn, and
      `tests/test_figure_rules.py` bans seaborn, `savefig`, `rcParams` and hex literals in figures.
- [ ] Fan out adversarial agents over the ported code. Targets: off-by-one in codon
      and HA1/HA0 numbering (the ±16 offset); silent empty-join producing an
      all-NaN panel; metric clamping; seeds not set; axis/label mismatches;
      hard-coded absolute paths; claims in captions that the data does not support.
- [x] A "no absolute paths" test that fails on any `/home` or `/Users` hit (code, config, manifests).
- [x] A golden-value test per ported figure table.
- [ ] Clean-clone verification: fresh clone with no access to `/home3`, build env,
      run tests, run the figure set, diff tables against golden values. **This is the
      only real test of the reviewer's reproducibility complaint.**

### Phase 7 — Single source: the influenza database

Full specification: [`DB_SOURCING.md`](../../DB_SOURCING.md). Opened 15 Oct 2026, nothing
implemented. Summary of why it exists:

Collection dates, subtype, host, segment and the S11 tip set currently reach the
figures by four unrelated paths, none of which a clean checkout can follow. The
database they should all come from is
`/home3/oml4h/RABV-gTK/test_out/IAV_DB_genbank_n_gisaid_rebuild/db_build/iav-db.db`.
Three defects found while scoping this, each its own risk below: the database the
repo names was **renamed out from under it** (risk 23), the per-sequence inputs that
built S15 are **not in this repo** (risk 24), and two documents **disagree about
which tree S11 used** (risk 25).

Design constraint that shapes the whole phase: `meta_data` carries **no index**, so
every `WHERE` is a full scan of a 33 GB row-store, and the file has a live
`-wal`/`-shm` pair — it is being written to. So "query the database directly" means
*one* module opens it, in one guarded pass, into a small pinned snapshot; it does
not mean every tool opening a live 33 GB file. `info.date` is **not** a version
stamp (it reads 26 Aug on a file rebuilt 15 Oct); size, mtime and row hashes are.

- [ ] `mutacc/iavdb.py` — the only module that opens the database. Read-only URI.
- [ ] `tools/export_db_snapshot.py` → `dev/db_snapshot/` + `MANIFEST.json`.
- [ ] `tests/test_iavdb_import.py` — consistency at import ([`DB_SOURCING.md`](../../DB_SOURCING.md) §7).
      The three known date defects become **counted** tests, not prose.
- [ ] Extend `tests/test_document_numbers.py` to run logs, so risk 25 cannot recur.
- [ ] Port `tools/h3_numbering_from_db.py` and `tools/export_lineage_panel.py`.
- [ ] Re-derive the S11 tip set as a query (the `.pb` itself stays external — the
      `trees` table holds newick only, with no per-branch mutations).
- [ ] Freeze the baseline **before any of the above**, then bottom-to-top re-run and
      the comparison table ([`DB_SOURCING.md`](../../DB_SOURCING.md) §8–9).

This phase is the concrete form of the Phase 6 clean-clone item: the metadata path
is where that verification currently fails.

---

## 6. Decisions

### Settled 8 Oct

| # | Question | **Decision** |
|---|---|---|
| **D1** | Fig 4: "DMS vs ___" | ✅ **vs the α-weighted score.** Measured DMS fitness against the combined PLM × accessibility score across the alpha sweep. Generator: `tmp_explore/analyses/a31_alpha_vs_dms_plot.py`, reading `a31_dms_joined.parquet`. |
| **D3** | Fig 3 epoch panel | ✅ **Keep the old figure.** It is panel **2C** of the composite Figure 2, not a Figure 3 panel — see §2.1. Ship `panels2/Figure_2_Composite_Asymmetric` and **package the panelisation code with it**. |
| **D7** | Fig 5 / codon learning | ⚠️ **Revised 9 Oct — my earlier recommendation was wrong.** The audit ([`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md) §4) shows the two analyses test *different hypotheses* and neither depends on the lineage set. Strand B replicates at 13 lineages (**AUROC 0.6104 [0.5733, 0.6414]** vs base 0.5105 [0.4875, 0.5406]). **The codon result stands, so Fig 5 is viable again.** Blocker is now mechanical, not scientific: strand B's figure chain is broken and its tables must be rebuilt from source. See D9. |
| **D9** | **New.** Given D7 reversed, is codon learning Fig 5 or supplementary? | Your call. The result survives, so the scientific objection is gone. Cost to promote: rebuild `a20` tables from source and replace the hand-authored SVGs with scripted matplotlib (D4). |
| **D8** | Palette | ✅ **Okabe–Ito.** `#0072b2` base / `#009e73` fine-tune. Passes all five checks. |

### Still open

| # | Question | Recommendation |
|---|---|---|
| **D2** | Is "reversible or not" a panel of Fig 2 or its own figure? | Panel of Fig 2. Symmetric matrices cannot express direction at all, so it reads as the natural fourth panel beside Grantham/BLOSUM. |
| **D4** | S19–S23 hand-authored SVGs: rebuild in matplotlib, or freeze? | Rebuild — they cannot be re-themed otherwise. **But** they belong to the codon strand, so this may be moot once `LINEAGE_VALIDITY.md` reports. |
| **D5** | S3 "fine-tuning success rate across viruses" | ✅ **Closed 12 Oct:** ESM-2 and ESM-C untuned, mean probability of the true amino acid at every masked position, across the viral panel. No fine-tuning. |
| **D6** | Epistasis gauge: published ΔP, or zero-sum? | Zero-sum. `dev/fig4_gauge` finds zero-sum tracks 3D distance and raw probability does not. Affects both S7 and S13. |
| **D10** | **New 15 Oct.** Where do clade labels come from once everything is sourced from the database? | `meta_data.Clade` is populated on GISAID rows and was **empty in every GenBank row sampled**. If coverage over the thirteen lineages is poor, the database cannot supply the label and we need Nextclade assignments instead, carried in their own column with the dataset version. The first thing the extraction pass reports is clade coverage for human H3N2 segment 4 — that number decides this, so the decision costs one query. See [`DB_SOURCING.md`](../../DB_SOURCING.md) §6.3. |

---

## 7. Risk register

1. **No epoch axis exists for the 13-lineage Fig 3.** `epoch_metric_summary.tsv`
   holds exactly two models — raw ESM-2 (epoch 0) and one final ESM2-HA80
   (epoch 1.0). The old `plot_PLM_epoch.png` has **no generator anywhere**. The only
   genuine per-epoch series is ESM-C 600M in `dev/epoch_adar/` — *a different
   backbone*. So the panel cannot be regenerated as published. This is D3.
2. **Fig 1 and the composites cannot be regenerated** and are the only copies.
   Back them up before any cleanup touches them.
3. **Fig 2/Fig 2b are computed on the 5-lineage run; everything else on 13.** The
   legend must say which, rather than leaving it implicit.
4. **The AUROC clamp — fixed, but the old tables survive.** Historically
   `site_logistic_auroc` was taken from a slope-constrained logistic fit, so wherever
   the score was anti-predictive the fit collapsed and AUROC read exactly 0.5 — a
   floor, not a measurement (170 of 260 rows at α < 0; max error 0.323).
   `run_mutational_accessibility.py:1844` now takes it from the **score** instead and
   keeps the old value beside it as `site_logistic_auroc_constrained_fit`. So the
   current Fig 3 is correct (its middle panel reaches ≈0.20 at α = −1, as it should).
   **Residual hazard:** `alpha_sweep_fit_metrics_BY_LINEAGE.tsv.preclamp_fix` is still
   on disk, and any figure or quoted number sourced from it — or from the
   `*_constrained_fit` columns — is reading the clamp. Delete the `.preclamp_fix`
   table on port, or move it out of `results/`.
5. **BLOSUM62 is not a biochemistry control.** Empirical substitution matrices
   already contain mutational accessibility. Grantham is the control; BLOSUM is a
   comparator. Any caption that blurs this invites the reviewer's objection back.
6. **`S08`–`S18` predate the 13-lineage expansion** and have not been re-checked
   against it.
7. **Environment split:** ESM-C and fair-esm live in *different* conda envs
   (`plm_entropy` vs `plm_sars`) and both import as `esm`. The plan must say which
   env each stage needs; a module-name check cannot tell them apart.
8. **Three docs in the repo carry a stale "no generator" claim** for the DMS
   figures, and they will mislead anyone cleaning up:
   `figures/key_revision_figures/README.md` (note 1 + the S16–S18 row), the
   *identical heredoc text* inside `scripts/collect_revision_figures.sh` (~l.341,
   353–355), and `results/dms/README.md`. Generators were written afterwards and
   never backported. **Only `S17_dms_fitness_supply_tradeoff` is a genuine orphan**
   — and it is also numerically withdrawn (dated Sep 10 11:00, pre-dates the
   backbone fix; its sibling is archived as `…_STALE_2026-09-10.png`). Treat S17 as
   dead, not merely ungenerated. Fix all three docs.
9. ✅ **Resolved 13 Oct 2026: the antigenic null is now one panel of S13, not a figure.**
   *Original note:* **The antigenic-site result is a null, and the figure says so.** +0.032 at the
   sites vs +0.035 elsewhere on 6WXB; +0.040 vs +0.034 on 7ZJ7. Only site C
   replicates on both structures. This null *replaces* an earlier fragile "+0.145
   Head Surface" claim. If S8 goes in, it goes in as a negative result — do not let
   a caption imply otherwise. *Now:* S8 is deleted and the comparison is S13 panel D, with the
   five sites pooled into a single class and only the zero-sum gauge drawn, so it is one test
   rather than five uncorrected looks. The site-C claim is no longer made anywhere in the
   figures; `mutacc.antigenic.site_summary` still computes it and is still tested, but nothing
   draws it. The caption states the null and both permutation p-values (0.85, 0.71).
10. ✅ **Resolved 9 Oct: S12 now has a paired test.** *Original note:* **S12's headline had no test behind it.** "Significant differences between
   fine-tuned models only for cell entry and preference ratio" is currently read off
   non-overlapping confidence intervals in `a31_within_stratum.tsv`. Non-overlapping
   CIs are not a test, and overlapping CIs do not imply no difference. Either add a
   paired test or soften the claim.
11. **`dev/fig4_gauge_sc2/` is a near-clone of `dev/fig4_gauge/` that has
   diverged.** Same ~15 filenames, retargeted from HA/6WXB to spike/6VXX+7KRR, built
   8 Oct. The results genuinely differ in direction: SC2 finds fine-tuning *beats*
   base on zero-sum AUROC (0.7733 vs 0.7622) where HA finds baseline ahead 28/28.
   Nothing enforces mirroring between the two trees, so an edit to one silently
   skews the comparison. Only the SC2 copy has a driver (`run_analysis_and_build.py`).
   Decide whether they merge behind one parameterised entrypoint before porting.
12. **ACE2-binding DMS arms are flagged `interpretable = False`** in
   `a31_dms_inference.py`, which is why those SC2 panels correlate negatively. They
   are not model failures and must not be reported as such.
13. 🔴 **Two different S27 figures are sitting in the supplementary directory.**
   `generate_revision_supplementary_figures.py` (v1, Oct 5) wrote
   `S27_mutation_bias_vs_fitness_distortion.{png,pdf}`;
   `…_v2.py` (Oct 6) wrote `S27_pssm_alignment_vs_simulation_fitness.{png,pdf}`.
   Both are still there. **v2 is the live generator** (the figures README credits v2
   for S25–S28; v1 produced no S28). Delete the v1 outputs, or the wrong S27 ships.
14. **`grep` at the repo root is gitignore-aware here** (ugrep `--ignore-files`), so
   plain greps **silently skip all of `tmp_explore/`** — which is where many figure
   generators live. Use `command grep` when searching for a generator, or you will
   conclude code is missing when it is not. This may well be how several of the stale
   "no generator" claims arose.
15. 🔴 **The across-lineage band does not mean generalisation to a new lineage.**
   Measured 9 Oct: the thirteen lineages are disjoint in sequences but share their
   sites — 80% carry the identical reference codon in all thirteen, and between-lineage
   probability correlation is 0.968. Resampling sites instead of lineages gives an
   interval **about 2.3–2.5× wider for the base model** (the 3.0× first reported came from only 500–600 bootstrap draws; ~1.2× for the fine-tune, ~13× for the
   codon statistic). The understatement is worst where the model is weakest, so the
   band on the *between-model difference* is the one to distrust. Either report
   site-clustered intervals, or state plainly that the band is the precision of the
   mean over thirteen near-duplicate backbones. See [`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md) §4 Q2.
16. ✅ **Training-corpus leakage checked and cleared** — worth stating in the
   manuscript rather than leaving for a reviewer to raise. 95.6% of observed
   replacements do appear in the 8,099-sequence HA-80 corpus, but on those *absent*
   from it the fine-tune still beats base by **+0.143 AUROC [+0.107, +0.182]**,
   reaching 0.889 on variants never seen. Generalisation, not recall.
17. **The five-lineage root files misrepresent themselves, and one is mislabelled.**
   Audited 9 Oct. The thirteen nextstrain roots are fine — every header says
   `nextstrain_clade_root|ref=A/Darwin/6/2021|nuc_mutations_from_root=N`. The
   original five do not: they carry GISAID tip accessions (`EPI4551091`,
   `EPI2990337`, …) with nothing marking the sequence as inferred, and
   **`K.nt.fa`'s header gives the lineage as `J.2.4.1`, not `K`.** The five-lineage
   guide also points into `/home3/oml4h/PLM_SARS-CoV-2/`, so it cannot run from this
   repo. Fig 2/Fig 2b are computed on that five-lineage run (risk 3), so this needs
   resolving before any five-lineage number is quoted again. Guide schema must carry
   `root_type`. See [`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md) §4 Q3b.
19. 🔴 **"Symmetric beats directional" (S4 (withdrawn)) is not evidence about the model.** Controls with no nucleotide content show the same ordering (real data) or the opposite sign (simulation). Do not use S4 (withdrawn)'s titles or the 9Octy README's executive summary. See `SYMMETRY_TESTS.md`.
20. 🔴 **Simulation scoring used a tip, not the founder.** 76, 71, 63 and 42 of 700 residues differ for the human, SARS-CoV-2, H3N2 and branching-ladder arms. Accessibility and the landscape come from the founder. Re-score on the founder before quoting S28 numbers; the GPU queue does this.
21. ⚠️ **Influenza and SARS-CoV-2 differ on learned direction.** The preprint's title claim ("learn underlying mutation biases") has direction-specific support for SARS-CoV-2 only. Reviewer 2 asked for exactly this test (`PREPRINT_AND_REVIEWS.md`).
22. ⚠️ **The preprint transposes two PR-AUC values** (accessibility alone 0.235, base ESM-2 0.130). The sentence built on them is reversed.
18. ✅ **Solved by design 9 Oct: tables ship as gzipped TSV.** *Original note:* **Only one interpreter on this machine can read the parquets.**
   `/home3/oml4h/miniconda3/envs/PRESCOTT/bin/python` is the sole env with both
   `pyarrow` and `scipy`; `plm_entropy`, `plm_sars` and `plm_esmc` all lack pyarrow.
   Combine with the earlier finding that only `miniconda3/bin/python` runs the test
   suite, and **no single interpreter does both**. `environment.yml` must say so.
23. 🔴 **The influenza database was renamed out from under the repo.**
    `iav-db.rebuilt.db` no longer exists; it is now `iav-db.db` (33 GB, 15 Oct 13:45) in
    the same `db_build/` directory. Still naming the old file: §3.2 above, the 15 Oct
    progress entry below, [`docs/notes/S3_S11_PROGRESS.md`](S3_S11_PROGRESS.md) "Inputs", and the
    usage example in `tools/h3_numbering_from_db.py`. **Nothing in the repo that reads
    the database can run as documented.** Two traps beyond the rename: the file has a
    live `-wal`/`-shm` pair, so it is being written to and a long run can see two
    states; and `SELECT * FROM info` still returns `new db | 2026-08-26`, so **`info`
    is not a version stamp** and must not be used as one. Size, mtime and row hashes
    are. [`DB_SOURCING.md`](../../DB_SOURCING.md) §4.
24. 🔴 **S15 cannot be rebuilt from a clean checkout.** Its two per-sequence inputs
    (`clade_accessions.tsv`, `seg4_dates.tsv`) were staged in the *archive's* gitignored
    `dev/lineage_panel/`; `clean_repo/dev/` holds only `retired_figures/` and
    `spectrum_swap_anomaly/`. They are correctly unshippable — GISAID accessions and
    per-sequence dates — but there is currently no documented path from the database to
    them, so the thirteen-row aggregate is an orphan. This is the reviewer's
    reproducibility complaint in miniature, and Phase 7 exists to close it.
25. ⚠️ **Two documents disagree about which tree S11 used.**
    [`docs/notes/S3_S11_PROGRESS.md`](S3_S11_PROGRESS.md) describes `h3_seg4_tree.pb`, 67,879 tips,
    22 Sep. `results/s11_h3n2_ha_tree/MANIFEST.tsv` says `h3n2_1968.pb`, 224,731 tips,
    pruned from the treepatch repair `usher.pb` of 24 Sep — and the shipped
    `summary.txt` agrees with the manifest (224,731 target tips, 50,094 branches
    scored). **The manifest is right; the run log describes a superseded run.** The
    S11 *numbers* are sound. What is missing is any test comparing a run log to a
    manifest, which is why this sat unnoticed; Phase 7 adds one.

---

## 8. Progress log

| Date | What landed |
|---|---|
| 2026-10-08 | Plan created. Repo mapped by four parallel surveys (main figures, DMS/contact, SC2/cross-virus/S11, theme/tests). |
| | **Palette audited** — current pair fails chroma + contrast; Okabe–Ito passes all five checks. |
| | **No shared theme exists**: 258 hex literals, 79 figsizes, 4 dpi values, 4 competing base/FT colour schemes. |
| | **Seaborn fallback bug is live** in 5 scripts and fires on the default interpreter. |
| | **Codon test contradiction found** (§3.3) — blocks Fig 5. |
| | **S11 never produced output**; two required inputs absent from the repo. |
| | **S3 blocked on GPU** — viral panel scored with base models only. |
| | Test baseline: 1566 pass / 9 fail / 26 xfail, 2m30s, miniconda interpreter only. |
| | Antigenic-site file found (unblocks S8, as a null result). 10 live scripts found untracked. |
| | **Panelisation code found** at `results/Lytras_OG/plots/per_model/ESM2_650M_HA80/panelise.ipynb.ipynb`, contradicting two repo docs that said it was in another repo. |
| 2026-10-09 | **Lineage validity audit run and reported** — `dev/lineage_validity/`, report emailed. Five scripts, nine tables, figure **S14** built in house style. Q1 codon contradiction resolved (different hypotheses; strand B replicates at 13 lineages — **D7 reversed, Fig 5 viable again**). Q2 across-lineage band is 3.0× too narrow for the base model (risk 15). Q3 no meaningful corpus leakage (risk 16). |
| 2026-10-09 | **Lineage-root audit:** all 13 nextstrain roots honestly labelled `nextstrain_clade_root` — the set in use is clean. The *original five* carry GISAID tip accessions with no "inferred" marker, `K.nt.fa`'s header says lineage `J.2.4.1` not `K`, and the 5-lineage guide points into the old repo. See risk 17. |
| 2026-10-09 | Style module gained `reference_line(label_loc=…)` after a real label collision. Caught a wrong panel in S14 pre-publication (pandas `.loc[[False, True]]` read as a boolean mask) and a wrong pair count (380 vs the 54 that clear the block filter). |
| 2026-10-08 | **Style module + 29 tests built and passing** (Phase 2). Layout proof rendered. |
| 2026-10-09 | **Guide system, data integrity tests, Fig 3 built.** Loader rejects bad guides; tables ship as gzipped TSV. Fig 3 lineage means at alpha = 0 reproduce the audit's 0.9195 / 0.7721. |
| | **Dense-sweep command reconstructed** from `run_manifest.json` (not replayed; needs GPU). `RUNS.md`. Closes the open gap in §3.1. |
| | **Fig 2 ported.** Published strata correlations reproduce to 4 decimals. New: site-bootstrap intervals are ~10x wider than the spread over lineages (Fig 2b is a revision-stage figure, never in the preprint; its default drew no intervals); ~80% of the amino-acid-pair asymmetry is a per-amino-acid tendency, and the pair-specific part is 0.35 (fine-tuned) vs -0.01 (base). |
| | **Fig 4 and S10 ported** (DMS). Every curve recomputed and agrees with the archive to 1e-6. ACE2-binding panels are labelled as binding, not fitness. |
| | **Fig 1 / Fig 2 composites packaged** (`panelise.py`). The published `panelstrim` Figure 1 is defective (heatmap twice, letters on titles); rebuilt from the intended `panels` version. The notebook never contained the Figure 2 composite call; layout read back from the PDF. Tree labels enlarged x1.65 by `tools/enlarge_tree_labels.py`, which refuses any scale that causes an overlap. |
| | **S2 ported.** AUROC recomputed from per-replacement scores agrees with the shipped sweep table to 8e-7. **S12 ported** with a paired site-bootstrap test: the fine-tuned-minus-base difference excludes zero in 14 of 21 fitness cells, more than the old "only some assays" reading from overlapping intervals. Closes risk 10. |
| 2026-10-09 | **Read the preprint and the MBE decision** (`MS_stuff/`): `PREPRINT_AND_REVIEWS.md`. Corrections: the strata, BLOSUM62/Grantham and direction analyses are revision-stage, not in the preprint (my earlier "published Fig 2b" remarks were wrong); the preprint's Fig 3B has two PR-AUC values transposed; revision due about 2 Nov 2026. Demo layout-proof files deleted. |
| | **S4 (withdrawn) audited** (`9Octy/generate_9octy_expansion.py`, reads `a34`): arithmetic correct (recomputed to 5e-6) and identical to `a35`'s reversible condition, but the headline interpretation is unsupported. No-nucleotide controls (BLOSUM62, Grantham, hydropathy, volume) show the same "symmetric beats directional" ordering by the same amount; in simulation the sign flips for the same controls. |
| | **Symmetry battery built** (`mutacc/accessibility.py`, `mutacc/symmetry.py`, 3 runners, planted-truth tests). Orientation, spectrum-shuffle, direction-flip null, direction contrast, planted-signal calibration, tie robustness. Accessibility reproduces the pipeline's `mut_prob` to 1e-9 (real) and 6e-16 (simulation). Two bugs found by the data and pinned by tests. |
| | **Result:** no evidence that either influenza model learned influenza's direction (flip pct 0.6 to 0.9; contrast 5% of truth, CI includes 0). **SARS-CoV-2 fine-tune did learn its own direction**, specific to SARS-CoV-2 (flip pct 0.016 to 0.062 in every stratum on BA.2 and XBB.1.5; 36 to 38% of truth; no preference for influenza's). Fig 2 panel C pair-specific direction is about half genetic-code geometry (flat spectrum gives 0.18 of the 0.35). |
| | **Simulation flaw found**: models were scored on a tip representative differing from the founder at 6 to 11% of sites, while accessibility uses the founder (affects archive S28). Conclusions survive excluding the 108 affected sites. **GPU arms launched on `gpu1`** (influenza spectrum, flat spectrum fine-tunes; founder re-scoring of all five models). Results pending. |
| | Fig 3 and S2 axes cut at chance; panel titles name the metric with a small plain-words tagline. Shared helpers `panel_title` and `cut_axis_below` in `style.py`. |
| 2026-10-08 | **Decisions D1, D3, D7, D8 settled** (§6). Governing principle added (§1): the repo is a question-agnostic, guide-driven figure engine. `LINEAGE_VALIDITY.md` opened for the codon contradiction and lineage-independence work. Sections renumbered. |
| 2026-10-09 | **Session 3 started (16:40).** Baseline suite launched. Six parallel workers launched, each owning separate new files (no shared-file edits): (1) S7 + S13, (2) S14 + S8, (3) S4 (withdrawn) + S5 + S1, (4) codon learning + S6, (5) S11 rework (synthetic fixtures if real inputs absent), (6) red team of the ported code. Lead (me) merges their reports into this file, README.md, DATA.md and figures/README.md. GPU queue on `gpu1` still on arm 1 (fine-tune, started 15:38, ~70 min each). |
| 2026-10-09 | **S11 rework built, no real output.** `mutacc/branch_codons.py`, `tools/branch_codon_tables.py`, `figures/make_sM_within_stratum.py`, 292 tests pass (planted truths, 180 random trees vs an independent oracle). Real tree `h3_seg4_tree.pb` parses cleanly (88,561 nodes, 67,879 tips, 101,640 mutations). **Blocked:** the UShER reference nucleotide record (the tree's VCF names it `DQ864721`) and a tip list are not in the archive, so no real table exists and none is shipped. Figure runs only from a synthetic fixture, watermarked. Fixed 7 defects of the archive script (root mutations dropped, undo order, `<10` off-by-one, stop/ambiguous mixed, column-as-codon numbering, synonymous-parent ignored, `terminal`/`isolated`). `--numbering-offset 16` is a guess, unverified. To unblock: supply one nt FASTA record in tree coordinates, the tip names and the CDS bounds. |
| 2026-10-09 | **S4 (withdrawn)/S5/S1 worker stopped by a safety flag on the model** before reporting. Not relaunched; status of its partial files unchecked. Open: decide how to proceed. |
| 2026-10-09 | **Cross-worker bug:** `tests/test_figure_rules.py` fails for `make_s15.py` line 273: a literal `'D'` matches a lineage name. S14 worker to fix. |
| 2026-10-09 | **S7 and S13 ported** (`make_s08.py`, `make_s14.py`, `tools/export_s08_s14.py`, 85 tests pass). S7: 72 clade pairs vs a pooled null of 92,682 draws; default gauge zero-sum (`--gauge` argument, not a guide key, because the loader rejects unknown keys); 3 pairs pass pooled BH q<0.05 under zero-sum (S144N->I160K, K189R->Q173R, N158D->Q173R), 0 under wt, 1 under prob, 1 under logit. N158D->I160K changes sign between prob and wt. S13: contact AUROC by region for two structures. 8 new tables in `results/h3n2_13lineage/` (440 MB scan store and structures not shipped). Guide and MANIFEST edited additively. |
| | **Finding (S13):** the archive's strata intervals resample contact pairs, but one residue sits in many pairs. A residue-resampling interval is ~2.2x wider; with it, almost no stratum clearly differs between base and fine-tune (6WXB 0 better / 1 worse; 7ZJ7 0 better). Do not quote "fine-tune better in N strata" from the archive intervals. |
| | **Other issues found:** an all-NaN p/q column passed every comparison silently (fixed, tested); archive doc says fusion peptide `backbone[330:354]`, correct is `[329:354]`; archive `figure4_null` and `fig4_gauge` cell tables disagree on studentised q (the latter lacks structure distances; shipped the former); head boundary 50-320 is hard-coded and unstable. **Open:** set-level null exists for the wt gauge only; studentised and "nearby" tests cannot be recomputed (structure distances not shipped); S7 panel B has spare headroom and one leader line crosses text. |
| | **Suite status (mid-run):** figure-rule tests fail for `make_codon_learning.py`, `make_s09.py`, `make_s15.py` (literal lineage-like names) and `test_no_absolute_paths` fails on `tests/test_branch_codons.py`. All in files owned by workers still running; to re-check once they finish. |
| 2026-10-09 | **S14 and S8 ported** (`make_s15.py`, `make_s09.py`, `mutacc/audit.py`, `mutacc/antigenic.py`, `tools/export_s15_s09.py`; 53 + 28 tests pass). S14 is re-derived from shipped tables; point estimates match the archive to 2e-6 (strand B 0.6104 / 0.5105 on 13 lineages, 0.6088 on 5; strand C all four cells). S8 per-residue gains match to 1e-9; headline gauge +0.032 at antigenic sites vs +0.035 elsewhere (6WXB), +0.040 vs +0.034 (7ZJ7); added a proper contrast with permutation p (0.85, 0.71), none of 10 cells excludes zero. Figure and caption say it is a null. Site C is above the rest on both structures, but that is one site of five, uncorrected. |
| | **Correction to a quoted number:** the archive audit used 500/500/600 bootstrap draws. The base-model "3.0×" interval ratio is partly noise: 2.5× (2,000 draws), 2.4× and 2.3× (10,000, two seeds); fine-tune ~1.2×. Archive intervals also shift slightly (strand B fine-tune [0.5760, 0.6415] vs [0.5733, 0.6414]; corpus advantage [+0.109, +0.178] vs [+0.107, +0.182]). Risk 15 corrected; LINEAGE_VALIDITY §4 Q2 table to say the same. Archive numbers reproduce exactly with its draw counts and seed (slow test pins it). |
| | **Open (S14/S8):** ~~S8 panel D (pair AUROC by antigenic-residue count) has no interval and rests on 136 both-antigenic contacts; needs a residue-clustered bootstrap.~~ *Closed 13 Oct 2026 by deletion: that panel went when S8 was merged into S13.* Gauge display names ("Zero-sum" etc.) are plain-word labels to check against S7/Fig 4. Archive per-residue values are rounded to 4 dp. `test_accessibility::test_every_shipped_spectrum_loads` fails on `data/spectra/spectra_sources.csv` (belongs to the S4 (withdrawn)/S5 worker's area or the base repo: triage). One `test_redteam_data` manifest-hash failure seen once in a full run, not reproduced: watch for a race with workers appending to MANIFEST.tsv. |
| 2026-10-12 | **Session 4.** Picked up the block the stopped S4 (withdrawn)/S5/S1 worker left behind. It had got as far as the engine (`tools/spectra_compare.py`) and `data/spectra/spectra_sources.csv`, with no exporter, figures or tests. The engine was correct: it reproduces the manuscript's **84.8%** self-compression and **43.6%** comparator cell, and all 64 cells and the stand-in table match the archive's `a16` output to **1e-10**. |
| | **S4 (withdrawn) and S5 ported** (`tools/export_spectra.py`, `figures/make_sD_partial_correlation.py`, `figures/make_sF_spectrum_swap.py`, `tests/test_spectra.py`, 26 tests). New results set **`results/spectra_codon_space/`** — the only one computed entirely inside the clean repo, since its inputs are the 4x4 matrices in `data/spectra/`. S5 is the cross-species heatmap; S4 (withdrawn) is the non-exclusivity argument: panel A plots each spectrum's ability to account for the focus spectrum against its free-parameter count, panel B the generic stand-in. |
| | **The S4 (withdrawn) argument, stated plainly:** two spectra with the *same* 12 free parameters score **+83%** and **−61%** against the focus spectrum, while a 1-parameter matrix scores **+44%**. Expressivity does not order the points; identity does. This is the direct answer to reviewer 1 point 1. |
| | **Two caption bugs found and fixed by looking at the rendered output.** S5's caption first attributed the published 43.6% to the *simplest* matrix (JC69, 0 parameters) rather than the comparator the manuscript actually used (K80, 1 parameter) — a false provenance claim. It also cited the *best*-scoring equally-expressive spectrum, which argues the opposite of the sentence it sat in. Both now come from a new `manuscript_role` column in `spectra_sources.csv` and from taking the worst, not the best, rival. Both bugs are pinned by tests that were checked by reintroducing them. |
| | **Both leftover suite failures fixed.** `test_every_shipped_spectrum_loads` globbed `*.csv` and so tried to load the metadata file; it now reads the key list from `spectra_sources.csv` and a new test asserts the files and the table agree. `test_no_absolute_paths_in_code_or_config` failed on two test files that must spell `/home` in order to check for it; the one-off exemption is replaced by an explicit `allow-absolute-path-literals` marker, honoured only under `tests/` and policed by a second test. |
| | **Manifest writer consolidated.** `append_manifest` existed in two copies with different code; both now import one implementation from `tools/export_base.py`, which also validates that no prose field contains a tab or newline. `style.reference_line` gained `label_side` after a real collision in S4 (withdrawn) panel A where both ends of the band above zero were already occupied. |
| | **Remaining supplementary work: S1 and S3 only.** S3 is still blocked on a GPU re-run and on D5. S1 (the SC2 analogue of Fig 2) is unblocked and is the next item. |
| | **GPU arms finished and analysed.** Founder-scored models (base + 4 fine-tunes: human, SC2, H3N2, JC69) and 4 PSSMs assembled into `clean_repo/results/sim_v3_star90/replacement_scores.tsv.gz` (13,300 rows x 19 cols, founder background). Symmetry battery run across all 4 spectra (`sars_cov_2`, `influenza_h3n2`, `human`, `jc69`). Findings: (1) SC2 FT specifically learns SC2 direction (contrast +0.0366, flip rank 1st/2nd across strata, partial rho holding fitness fixed +0.256 at 2 nt); (2) H3N2 FT specifically captures H3N2 mutational accessibility holding fitness fixed (+0.175 at 1 nt, +0.174 at 2 nt vs -0.0005 for flat FT); (3) Flat JC69 is clean negative control (flip 1.000, delta 0.0000, zero spurious correlation with real spectra); (4) Tip-divergence flaw resolved by direct founder scoring. `clean_repo/SYMMETRY_TESTS.md` §§5, 6, 8 updated. |
| 2026-10-12 | **S1 built, then corrected twice.** First draw used the wrong fine-tune (ESM-2 650M spike fine-tune, plus an ESM-2 influenza fine-tune as a foreign-protein arm). The paper's spike fine-tune is ESM-C: `my_SC2_finetunes/magma_ESMC_600M_99_95perc/esmc_weights.pt`. S1 now compares raw ESM-C 600M with that checkpoint, read from `tmp_explore/esmc_matrices/` (scored 9 Sep, after the checkpoint was saved 25 Mar; `score_esmc.py` names the same path). The foreign-protein arm is dropped: no ESM-C influenza fine-tune was scored on spike. Panels A and B now use the Figure 2 role colours (base blue, fine-tune green); the number of changes is read from the three bands, labelled directly. Audit of the same mistake elsewhere: Fig 4, S10, S12 and the SC2 symmetry tests already use ESM-C; the other checkpoint references in the archive (`run_all_figures.sh`, `dev/fig4_gauge_sc2`) name the ESM-C one. Recorded in `README.md` ("Which models, which checkpoints"); `tests/test_sA_spike.py` pins it. Fixed on the way: `make_sA_spike.py` broke the no-literals rule (virus names now come from the guide); the spike guide's unaligned, unverified roots are now a documented exemption in `test_data_integrity.py`. Suite green. |
| 2026-10-12 | **S3 redefined and S11 unblocked.** S3 is now ESM-2 and ESM-C untuned only, every position masked, mean probability of the true residue (no fine-tuning, so no 25 fine-tunes and no D5). Scorer batches masked copies and agrees with a one-at-a-time pass to 3e-7; running on `gpu1` from 17:26. S11 run on real data from the IAV database; the gap-column artefact above found and fixed. Run log with timestamps: [`S3_S11_PROGRESS.md`](S3_S11_PROGRESS.md). |
| 2026-10-12 | **S3 done; S11 table added.** S3 scoring finished 18:55 (both models exit 0); figure, data and 8 integrity tests in. S11 gained the summary table with a second row (oldest H3 node against the newest internal nodes): changes needing 2 or 3 nucleotides are 0.4% on single branches but about 20% of long-range differences. One panel sequence (influenza C HEF) is a UniProt fragment with 2 unknown residues, scored with them removed. |
| 2026-10-13 | **S15 and Supplementary Table 16 delivered.** Built `figures/make_sN_simulation_control.py`, `tests/test_sN_simulation_control.py`, and `results/sim_v3_star90/empirical_substitution_spectra.tsv.gz` (48 rows, 12-class substitution spectrum across all 4 evolutionary arms). Shipped to `output/`: `sN_simulation_control.{pdf,png}`, `sN_simulation_control_values.tsv`, `sN_simulation_control_caption.md`, `supl_table_16.tsv`, and `table_sN_simulation_control.tsv`. Maintained the archive DMS optima table via `tools/export_table_s15.py` (`table_s16_dms_alpha_optima.tsv`). ⚠️ *Corrected 15 Oct: this line originally named a `tools/export_table_s16.py`, which does not exist, and called the same table both S15 and S16. The table number is unresolved and tracked in `docs/notes/extended_red_team_tests.md` §A1b.* All figure rules, AST checks, and simulation pipeline tests passing. |
| 2026-10-13 | **S10 re-encoded: one channel per variable.** The panels carried four marker shapes (one per assay system), three marker fills (solid / open / half-filled, repeating the trait) and the trait colour, plus a per-system text label redrawn on all six panels — too many channels to read. Now **colour is the trait category** (the same three `CATEGORICAL` hues) and **marker shape is the viral species only**: triangle for Influenza A, circle for SARS-CoV-2, taken from the `virus` column with its parenthesised qualifier stripped, so `Influenza A (H3N2)` and `Influenza A (cross-subtype)` share a shape. Marker faces are drawn at 55% alpha inside a full-strength edge, so overlapping points read as a stack. The fill styles and the in-panel system labels are gone; the legend now carries species and trait in three columns. **Cost to note:** the figure no longer distinguishes `HA Entry vs Growth Preference` from `HA Cross-Subtype Entry`, so the caption's caveat that the cross-subtype pairs use a structural alignment and cross cell line can no longer be tied to particular points — the assay stays in `sJ_dms_concordance_values.tsv` for every point, and the caption says so. The mono-print claim is dropped from the caption with the redundant fill. `tests/test_sJ_dms_concordance.py`: three tests that pinned the old shape/fill encoding rewritten, one added to pin the face alpha; 84 tests green with `tests/test_redteam_figures.py`. |
| 2026-10-13 | **S8 merged into S13 and deleted.** The antigenic comparison is a null, so it is now S13 panel D rather than a figure: the five published sites pooled into a single class (one test, not five uncorrected looks) and only the headline zero-sum gauge drawn. `figures/make_s08.py` and `output/s08_*` are gone; `tests/test_s08.py` is now `tests/test_antigenic.py` and covers `mutacc.antigenic` alone. The per-residue intervals still match the archive exactly because `antigenic_stratum` bootstraps every gauge before dropping all but zero-sum — summarising zero-sum alone would draw different random numbers. The per-site and pair-level panels are dropped; `site_summary` and `antigenic_pair_benchmark` survive, tested but undrawn. S13's later panels renumber: positional is now E, the positional difference F. The antigenic panel gets a row of its own, not a fourth column in the top row, which collapsed that row's layout; it runs the full width of that row with its legend inside it, over the empty top of the range: at half width a quarter of the row was blank and the legend sat further from the marks it named than a legend on them. The intervals look longer than they did in the five-gauge panel they came from only because that panel's shared x-axis had to span every gauge; the numbers are the archive's, unchanged. Panel C's "no change" label is dropped (it collided with the panel title; the line at zero is labelled by the axis). Text sizes across the whole figure were below the house style (local overrides at 5 and 6 pt against a house 7 for ticks and legends); they are raised to it, which is why the top row's stratum labels now set the width of that row. `tools/export_s14_s08.py` keeps its name until the figures are renumbered. |
| 2026-10-14 | **S10 cut to two panels.** The figure was three rows (Spearman, Pearson, per-site JSD) x two distance axes. The Pearson row ranks the pairs the same way the Spearman row does, and the JSD row answers a different question in a different unit, so six panels were asking the reader to tell apart three measures that agree. S10 is now **one row: the Spearman rank correlation against substitution count (A) and against percent divergence (B)**, all 26 pairs in both. The other two measures stay in `dms_cross_lineage_pairs` and are named in the caption as present-but-not-plotted; `tests/test_redteam_figures.py` still pins that Pearson falls with divergence too, so the claim that they agree is tested rather than asserted. Dropped with them: the "2 pairs have no JSD" panel note (those two were the `Full Spike Pseudovirus` pair — ~4.5 variants per site over 1,227 sites, too thin for a 20-amino-acid preference profile; **not** an ACE2 problem, 10 of the 11 ACE2-binding pairs did have a JSD). The question is no longer a panel title, because both panels ask it; the y label and the caption carry it. Panel height 2.15 -> 2.6 in. `tests/test_sJ_dms_concordance.py` and `tests/test_redteam_figures.py` updated; validation now checks only the columns the figure draws. |
| 2026-10-14 | **Fig 3 panel subtitles made exact; α = 0 made visible.** The subtitles said "ranking every possible replacement", "distinguishing observed from unobserved replacements" and "frequency of replacements observed" — the last of which did not say the one thing that separates it from panel A. Traced the three columns to source (`Functions_HuggingFace.py:2805-2891`, `scripts/run_mutational_accessibility.py:1823-1864`): all three share the same unit, one candidate replacement (each of the 19 non-reference amino acids at every site, within a lineage), scored as `log10(plm_prob) + α·log10(mut_prob)`. **A** `mut_flat_global_spearman_r` is Spearman(score, observed frequency) over all of them, **zeros included**; **B** `site_logistic_auroc` discriminates `obs_present` 0/1 (the `site_` prefix is legacy — the function feeding it is `compute_mutation_row_logistic_metrics` and the unit is replacements, not sites); **C** `mut_flat_nonzero_pearson_r` is Pearson(score, observed frequency) over the rows with **frequency > 0 only**. Subtitles now say exactly that, naming the swept quantity "PLM score" and defined in the caption as the model's **pseudo-probability** for that amino acid at that site (a score over the 20 amino acids, not a calibrated probability of the replacement happening) times accessibility^α, equal to the pseudo-probability alone at α = 0. Two lines each so one long line cannot set the width of the whole figure; `style.panel_title` pads the title by `TAGLINE_LINE_PAD` per extra line. `obs_freq` is a fraction of aligned sequences, not a count, so the subtitles say "frequency". **α = 0 was a 0.6 pt dotted hairline** and is now a labelled dashed rule in `INK_MUTED` at 1.3 pt, with a white-edged circle on each curve where it crosses, so the model-alone value can be read off directly. Caption updated: it now states the unit, the zeros-in/zeros-out difference between A and C, and what the circles are. |
| 2026-10-14 | **Fig 5 panel C widened.** `width_ratios` 1.38/0.58/4.1 -> 1.22/0.5/4.9. The panel letters were placed with one `dx=-0.28` in *axes* fractions, so on the widest panel that put "C" a long way out into the gap beside it; each letter now gets its own dx (A -0.28, B -0.55, C -0.075) and sits just clear of its own y label. |
| 2026-10-14 | **S9 generic-matrix comparison folded into S5 as panels B and C.** The archive's `S09_generic_matrix_comparison.png` was never ported; its table, `spectra_generic_stand_in`, had been shipping unused since S04 was withdrawn on 13 Oct. S5 is now three stacked panels: the heatmap, then the ranking by mean correlation and by the manuscript's retained-variance statistic. Laid out as two rows, A across the top and B and C beneath, after a first pass stacked all three. Getting the two-row version to hold needed three fixes: the colour bar moved into its own grid column (`cax=`, not `ax=`, which was carving its space out of the shared column widths and squeezing B and C to nothing), the taglines shortened (the constrained layout treats an over-long one as a width the axes must find), and the tagline and panel letters pinned in points rather than axes fractions, since A is three times the height of B and C and one offset cannot serve both. The second label line is "6 nt params", not "6 free nt params", which was wider than the spectrum names; the caption carries the full phrase. Numbers match the archive PNG exactly (E. coli 0.812 / 48.4%, focus 0.494 / -47.9%). No intervals are drawn: these are means over a fixed set of six comparisons, not sampled estimates, and the caption says so. The merged spectrum is derived from the two tables rather than typed, because `test_figure_rules` forbids a virus name in a figure script. Two tests needed scoping, not weakening: the rival-citation check now reads only the heatmap paragraphs (the new paragraph names the merged subtype for an unrelated reason), and the caption-number check now pools every values file the figure writes. Panel subheaders across S5 raised from 6.5 pt muted to 8 pt, kept muted so they do not compete with the 9 pt titles. |
| 2026-10-15 | **New Figure S15: the lineage panel.** The manuscript has never described the lineage set — its methods still carry the five-lineage sequence counts (one of them a literal `<>` placeholder) and no sampling dates at all. `tools/export_lineage_panel.py` joins the sub-alignment accessions to collection dates in the influenza database (`iav-db.rebuilt.db`, `meta_data.collection_date`, read-only) and writes a thirteen-row aggregate, `results/h3n2_13lineage/lineage_panel.tsv.gz`; `figures/make_sZ_lineage_panel.py` draws it as a table. The two per-sequence inputs are GISAID accessions and dates, staged in the archive's gitignored `dev/lineage_panel/` and **explicitly re-ignored** — `.gitignore`'s `!dev/**/*.tsv` would otherwise have committed them. Counts confirmed against `panel_metadata.tsv` and the 78,659 total in `LINEAGE_VALIDITY.md`; replacement counts re-derived from `replacement_observed`. **Three date defects found in the source metadata**: mixed formats (ISO, `12-Dec-2024`, `2024-12`, bare `2024`); partial dates, which would drag a first date back to 1 January; and a handful of GISAID records dated a decade before their own lineage, two of them in 1905 — a year read as a spreadsheet serial number. The figure therefore carries both an absolute first/last and a 1st–99th percentile span, and the caption says to quote the percentile span. Sizes run 779 (G.1) to 25,841 (J.2), a 33-fold range, which is the imbalance Reviewer 2 raised. `tests/test_sZ_lineage_panel.py`, 15 tests. `test_numbering.py::test_the_numbering_is_contiguous_from_one` hard-coded `range(1, 15)`; it now reads the length from the map and still checks for gaps, since adding a figure is the expected change and a gap is the defect. |
| 2026-10-15 | **Extended red-team plan opened; supplementary numbering found to be wrong.** New document [`docs/notes/extended_red_team_tests.md`](extended_red_team_tests.md): a plan to establish whether the suite can actually catch a wrong number, rather than only that the code agrees with itself. Baseline measured — 1,758 tests collected, 1 failure, and that failure is a true positive. A coverage survey of all 16 modules and 50 test files found six tests whose expected value is computed by the code under test, eight that assert less than their names claim (`test_sN_simulation_control.py::test_sN_simulation_control_golden_values` contains no golden number), and five modules with no adversarial file at all. Four confirmed defects beyond the failing test: the `xfail(strict=True)` ratchet described in `test_known_defects.py` **does not exist anywhere in `tests/`**; `test_simulation_pipeline.py` reads a path outside the repo so it fails on a clean checkout; no library version is pinned anywhere while goldens are asserted to 1e-12; and `MANIFEST.tsv` hashes the unshipped archive source rather than the shipped file. Chasing the failing test opened the numbering work: **two supplementary tables are both labelled S16**, and against the draft's ordering four supplementary figures move (S12 and S13 swap) and the lineage-set validity figure is dropped. Decided: numbers come out of filenames into a map file, because the set will be renumbered again once the manuscript is finished. Staged fix in §A1b; nothing renamed yet. |
| 2026-10-15 | **S7 and S13 cut to the Zhang zero-sum gauge alone; three S13 defects fixed.** The gauge is no longer a variable anywhere in the figure code: `make_sG_epistasis_null.select_gauge` and `mutacc.antigenic.select_gauge` drop every other gauge as the table loads, so no function below takes a gauge argument. Removed from S7: `--gauge`, `GAUGE_INFO`, `DEFAULT_GAUGE`, `PUBLISHED_GAUGE`, `gauge_names`, `gauge_summary`, `sign_changes`, and **panel D** (the cross-gauge count) — the figure is now A to C with panel C across the full bottom row. The caption's gauge argument no longer quotes the probability-scale null, which meant reading a gauge the figure does not draw; the most conserved focal site's baseline probability (0.985) carries it alone. Removed from S13: `headline_gauge` and the all-gauge bootstrap. **The shipped tables keep every gauge on purpose** — they are what the archive exported and what `MANIFEST.tsv` pins; the exporters are unchanged and now say not to narrow them, and the cross-gauge facts (3/0/1 pairs by scale; N158D→I160K changing sign) are still asserted against the shipped tables in `test_sG_epistasis_null.py`. **Interval ends moved** by up to 0.0017: the old `antigenic_stratum` bootstrapped all five gauges and discarded four purely so the random stream matched the archive's, and that is the dead code the cut removes. No point estimate moved (a group mean does not depend on the stream) and no conclusion changed — all four antigenic intervals still cross zero. Golden values in `test_antigenic.py` re-pinned to this code, with the means still checked against the archive separately. Three S13 defects found while checking panels A and D: the script **did not run** (its caption text opened "Supplementary figure.", which `write_caption` refuses, so `output/` was stale and self-labelled instead of "Figure S8"); panels A and B **share tick labels but had only 6WXB's contact counts**, which differ a lot between structures (Stalk Core 119 vs 20, Head Surface 28 vs 13), now shown as `n = 119 / 20`; and panel D's bars were unlabelled as intervals, now "95% bootstrap intervals over residues (2,000 resamples), not a test of the difference" with each row's permutation test written out in full beside it. |
| 2026-10-15 | **Phase 7 opened: everything to flow from the influenza database.** New document [`DB_SOURCING.md`](../../DB_SOURCING.md). Collection dates, subtype, host, segment and the S11 tip set currently reach the figures by four unrelated paths; the inventory in §2 of that document shows **five of the nine metadata inputs cannot be reproduced from a clean checkout**. Three defects found while scoping, now risks 23–25: the database the repo names was renamed (`iav-db.rebuilt.db` → `iav-db.db`, 15 Oct 13:45) so no DB-reading code runs as documented; S15's two per-sequence inputs are in the archive's `dev/`, not here; and `docs/notes/S3_S11_PROGRESS.md` describes a 67,879-tip 22 Sep tree where the shipped manifest and `summary.txt` both say 224,731 tips from the 24 Sep treepatch repair — the manifest is right and the S11 numbers are sound, but nothing compares a run log to a manifest. Two properties of the file shape the design: `meta_data` carries **no index** (every `WHERE` is a full scan of 33 GB) and a live `-wal` pair means it is being written to, so the plan is one module, one guarded pass, one pinned snapshot — not live queries from every tool. Also established that **`info` is not a version stamp**: it reads `new db \| 2026-08-26` on a file rebuilt 15 Oct, so staleness must be detected from size, mtime and row hashes. New open decision D10 (clade labels: `meta_data.Clade` was empty in every GenBank row sampled, so coverage decides whether we need Nextclade instead) — answerable with one query from the extraction pass. Nothing implemented; the first action is freezing the baseline, before any code changes. |
| 2026-10-20 | **The supplement is lettered, not numbered.** Supplementary labels in [`guides/manuscript/figure_numbers.yaml`](guides/manuscript/figure_numbers.yaml) became `Figure SA` … `Figure SO` and `Table SZ`; main-text figures keep their digits. The reason is that the draft's citation order is still moving and a digit is a position, so every re-order dragged ~20 captions, five documents and a dozen test assertions behind it — the 15 Oct re-order is exactly that, and it is why three captions shipped with numbers the draft had already changed. A letter is a name: it is assigned once and never moves, and the manuscript's final `S1`…`SN` run is applied in one pass at submission by walking the citations in order of first mention (each letter appears exactly once in [`FIGURE_LABELS.md`](../../FIGURE_LABELS.md), so the substitution is mechanical). `mutacc.numbering` now matches `^Figure S([A-Z])$` and `^Table S([A-Z])$`, reduces a letter to an ordinal (A = 1) so ordering and the no-gaps check are unchanged, and **exempts supplementary tables from the run** — `Table SZ` is parked at the end of the alphabet so adding a figure never moves it. Letters assigned in the 15 Oct order, so SA…SO is S1…S15 and `Table SZ` is the former `Table S1`. New document `FIGURE_LABELS.md` (letter → old number → stem → script → content) with three tests in `test_document_numbers.py` that hold it to the map, plus `test_numbering.py` coverage that a digit supplementary label is now *rejected* and that a table may sit outside the figure run. The 16 shipped captions were re-labelled in place rather than redrawn: `write_caption` takes the label from the map, so the next rebuild writes the same bytes. Stems are untouched — they are still numbered after their old slots and stage 3 of [`docs/notes/extended_red_team_tests.md`](extended_red_team_tests.md) §A1b still owns renaming them. |
| 2026-10-21 | **SJ: the floor-censoring brackets removed.** Two of the 26 DMS pairs (both `HA Cross-Subtype Entry`, 31% of variants at the assay detection floor) carried a vertical I-bar from the plotted coefficient down to the same coefficient recomputed without the floor-piled variants (ρ 0.53→0.34 and 0.55→0.34). The bar was honest about a real sensitivity but drawn in the grammar of an error bar, on a figure where no other mark has a vertical extent: a reader takes it as sampling uncertainty on those two pairs, and as precision on the other 24. Both numbers are still computed and are now **stated in the caption and carried per pair in the values table** (`y_no_floor`, `censored`) rather than drawn, so nothing is lost but the misreading. `FLOOR_BRACKET` and `panel_points` are unchanged — only the drawing, the legend entry, the label-placement obstacles and the y-limit are. `tests/test_sJ_dms_concordance.py` gains `test_no_mark_on_a_panel_has_a_vertical_extent` (no errorbar container, and no line spanning two y values other than the horizontal reference) and `test_the_floor_sensitivity_is_reported_in_words_not_drawn` (both coefficients appear in the caption; the flag and value reach the values table); three bracket-named tests renamed to say "sensitivity". |
