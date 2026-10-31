# Data: which table feeds which figure

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


> ⚠️ **Figure numbers here are stale (15 Oct 2026).** Of the ones named below, **S12
> becomes S13** and **S15 becomes S14** in the manuscript draft's ordering; S2–S5, S10 and
> S11 are unchanged. Mapping and staged fix:
> [`docs/notes/extended_red_team_tests.md`](docs/notes/extended_red_team_tests.md) §A1b. The table-to-content
> pairings below are correct; only the numbers move.

> 📋 **Metadata provenance is being consolidated (plan opened 15 Oct 2026).** Collection
> dates, subtype, host, segment and the S11 tip set currently reach these tables by four
> unrelated paths, and five of the nine metadata inputs cannot be reproduced from a clean
> checkout. They are all moving to one source — the influenza database at
> `…/IAV_DB_genbank_n_gisaid_rebuild/db_build/iav-db.db` — with consistency tests at import
> and a bottom-to-top re-run that confirms no value changes. Inventory, design, tests and
> the re-run plan: [`DB_SOURCING.md`](DB_SOURCING.md). Tracked as [`docs/notes/PLAN.md`](docs/notes/PLAN.md) §5
> Phase 7. **Note for anyone reading a DB path in this repo today: `iav-db.rebuilt.db` no
> longer exists** — it was renamed to `iav-db.db` on 15 Oct (PLAN risk 23).

Every table is gzipped TSV under `results/<run>/`, named in the run's guide, and traced to its
source in that directory's `MANIFEST.tsv` (source path relative to the exploration archive, a
hash of the source, row and column counts, columns dropped). `tools/export_tables.py` makes
them; the figure scripts never read anything else.

## `results/h3n2_13lineage/`

| Table | Rows | Read by | What it is |
|---|---|---|---|
| `alpha_sweep_by_lineage` | 2,652 | Fig 3, S2 (cross-check) | Metrics per lineage, model and accessibility weight, on the 101-point grid. The clamped `*_constrained_fit` columns are removed. |
| `replacement_scores` | 279,604 | Fig 2 (B, C), S2 | Model probability, accessibility, nucleotide-step class and two amino-acid-pair scores for every possible replacement, per lineage and model. |
| `replacement_observed` | 139,802 | S2 | Whether each replacement was observed in each lineage. Does not depend on the model. |
| `replacement_frequency` | 139,802 | Fig 3 (C), S2 (F), S15 | How *often* each replacement was observed, and the sequence depth behind that fraction. Keyed one-to-one against `replacement_observed` and built from the same archive table by `tools/export_sB_frequency.py`, which refuses to write unless `obs_freq > 0` agrees with `obs_present` on every row. Does not depend on the model, so it is stored once per replacement rather than once per model. The "picking the winners" panels rank the observed replacements against each other, which the 0/1 flag cannot do. |
| `lineage_panel` | 13 | S15 | One row per lineage: sequence count, collection-date coverage and span, sites differing from the lineage root, and replacements observed. Aggregated by `tools/export_lineage_panel.py` from the sub-alignment accessions, the sequence database's collection dates and `panel_metadata.tsv`. The two per-sequence inputs hold GISAID accessions and dates and **do not ship**; only this thirteen-row aggregate does. |
| `homoplasy_by_lineage` | 13 | SZ | One row per lineage: how many of its observed replacements arose two or more times independently on the segment 4 UShER tree, split by whether the recurrent origins used one codon route or several, with the coverage gaps that shrink the denominator. Counted by `tools/export_homoplasy.py`; `mutacc/homoplasy.py` holds the definitions. Also carries the **rarefied** columns: `pct_multi_origin` tracks sampling depth at Spearman 0.96 and so cannot be read across lineages, while `rarefied_pct_multi_origin` recomputes it with every lineage subsampled to the smallest one's 778 tips and tracks depth at 0.23. The rarefaction is exact, not simulated, and is taken over the same replacements as the raw column, so the smallest lineage comes back unchanged — the check that pins that is in `tests/test_sZ_lineage_panel.py`. The tree and the per-sequence tip lists are inputs and **do not ship**; only this thirteen-row aggregate and the per-replacement table below do. |
| `homoplasy_replacements` | 6,208 | SZ (diagnostics) | One row per replacement the tree places: its origin count, how many internal branches those sat on, the distinct codon changes that produced it, and whether `replacement_observed` marks it present. The table to read when a lineage's homoplasy count looks surprising — J.2 position 161 (mature H3 site 145) arose 90 times by two codon routes. Its `supports` column holds each origin's descendant tip count, largest first, which is what the rarefaction needs and the reason it requires no second walk of the tree. |
| `dms_scores` | 67,724 | Fig 4, S12 | Model score, accessibility and measured effect for every scanned replacement. Its `log_plm` is the **wild-type (unmasked)** marginal, which is what the archive's curves were computed from; every figure reproduces it as a gate and then draws the masked column below. |
| `dms_scores_masked` | 67,724 | Fig 4, S12 | The **masked marginal** for every row of `dms_scores`, keyed one-to-one against it: the probability the site being scored gets when it is replaced by the mask token and predicted from its context. What the DMS figures report since 19 Oct 2026. Built by `tools/build_masked_dms_scores.py` from the GPU profiles; see `mutacc/masked_dms.py` for why the estimator is not a convention. |
| `dms_alpha_curves` | 3,618 | Fig 4 | The archive's own correlation-vs-weight curves; Fig 4 recomputes them and must agree to 1e-6. |
| `dms_alpha_bands_published`, `dms_alpha_bands_reversible` | 3,618 each | Fig 4 (tests) | The same curves with their site-bootstrap band; used to check the band reproduces. |
| `dms_peak_summary` | 36 | Fig 4, S12 | Panel names, citations, model roles, and the archive's optimum per curve. Its `interpretable` column concerns the optimum, not the assay. **Its `rho_alpha0`, `peak_alpha`, `peak_rho` and `gain` are the wild-type (unmasked) estimator** and are kept because they are the fixed point every switched figure's gate checks against — they are *not* the numbers the manuscript reports. For those read `output/sY_dms_alpha_optima.tsv` (Table SY), written off the installed masked figures. The two estimators disagree in sign on MA22 pH stability: unmasked puts the fine-tune below the base at α = 0 (0.127 vs 0.138), masked puts it above (0.138 vs 0.121). That is the estimator change, not a discrepancy to fix; `tests/test_masked_dms.py` pins it so the tables cannot be quoted interchangeably. |
| `dms_spectrum_influenza_h3n2`, `dms_spectrum_sars_cov_2` | 4 each | Fig 4 (reversible variant) | The 4x4 nucleotide matrices. |
| `dms_within_stratum` | 54 | S12 | The archive's within-step-stratum correlations, **unmasked**; S12 recomputes them from `dms_scores` and must agree to 1e-9 whichever arm it draws. There is no masked counterpart: the masked bars and their intervals are recomputed in the figure. |
| `masked_dms_curves` | 7,236 | SO | The Figure 4 alpha sweep run twice, once under each estimator, over every DMS arm of both viruses. Built by `dev/masked_dms/build_masked_tables.py`, which calls the published sweep code with nothing swapped but `log_plm`. |
| `masked_strata` | 32 | SO | The model-vs-accessibility correlation by codon distance under both estimators, the Figure 2D quantity. From `dev/masked_fig02/compare_fig02.py`. |
| `dms_cross_lineage_pairs` | 26 | S10 | Agreement between pairs of scans of one protein, against how far apart the strains are. |

| `replacement_properties` | 139,802 | symmetry tests | Negative hydropathy and volume distance per replacement: controls with no nucleotide content. |
| `symmetry_<spectrum>_*` | small | `SYMMETRY_TESTS.md` | Output of `tools/run_symmetry_battery.py`: orientation, null, flip, direction, planted, robustness and distinct-value tables, for the influenza and SARS-CoV-2 spectra. |
| `symmetry_dms_*` | small | `SYMMETRY_TESTS.md` | The battery on the four DMS datasets, scored against each virus's own and the other virus's spectrum. |

## `results/sim_v3_star90/`

| Table | Rows | Read by | What it is |
|---|---|---|---|
| `replacement_scores` | 13,300 | S15, symmetry tests | 700 sites x 19 replacements scored on the synthetic founder background: ESM-C 600M base, four fine-tuned arms (human, SARS-CoV-2, H3N2, flat JC69 neutral control), true fitness landscape, and founder codons. Not GISAID-derived. |
| `empirical_substitution_spectra` | 48 | S15 (A) | Counts and percentages for all 12 nucleotide substitution classes across 5,000 evolved sequences in each of the 4 evolutionary arms. |
| `symmetry_*` | small | S15, `SYMMETRY_TESTS.md` | Directional battery and debiasing sweep results across the 4 simulated arms holding true fitness fixed. |

## `results/h3n2_5lineage/`

| Table | Rows | Read by | What it is |
|---|---|---|---|
| `replacement_scores` | 107,540 | Fig 2 (A), S4 | As above, for the original five lineages. |

## `results/spectra_codon_space/`

The only run computed entirely inside this repo: its inputs are the 4x4 nucleotide matrices in
`data/spectra/`, so `tools/export_spectra.py` needs no exploration archive. No protein model is
involved; both figures are properties of the mutation matrices and the genetic code alone.

| Table | Rows | Read by | What it is |
|---|---|---|---|
| `spectra_pairwise` | 64 | S4, S5 | Every spectrum against every other in codon space: percent of variance retained with and without the compress-and-rebuild step, and Pearson r. |
| `spectra_generic_stand_in` | 7 | S4 (B) | Mean and worst agreement of each spectrum with all the others. An organism measured twice is counted once, so `organism_group` in the sources table decides which row is dropped. |
| `spectra_sources` | 8 | S4, S5 | Display label, kind, free nucleotide parameters, citation and plot order per spectrum, plus which one the manuscript's claim is about (`manuscript_role`). The figures take every name from here and hard-code none. |

The published numbers live in `spectra_pairwise`: the focus spectrum's self-compression (84.8%)
and the comparator's score against it (43.6% compressed, 51.0% not). `tests/test_spectra.py`
pins all three.

## `results/viral_panel/` and `data/viral_panel/`

Scores and sequences behind S3. The scoring needs a GPU, so only the output ships. The panel is the
32 viral proteins plus a nine-protein cellular and phage comparison (3 eukaryotic including human,
3 bacterial, 3 bacteriophage) fetched and scored identically, so the viral scores can be read
against something; `group` separates them.

| File | Rows | Read by | What it is |
|---|---|---|---|
| `data/viral_panel/viral_panel.faa`, `viral_panel_sequences.tsv` | 41 | S3 | Every amino-acid sequence scored (25,541 residues): UniProt reviewed entries, with accession, protein name, organism and length. One entry (influenza C HEF, P17005) is a UniProt fragment with 2 unknown residues, scored with them removed (`unknown_residues_removed`). |
| `viral_panel_sites.tsv.gz` | 51,082 | S3 | Probability each untuned model gave the true amino acid at every position when it was hidden, and its single best guess. |
| `viral_panel_summary.tsv.gz` | 82 | S3 | One row per model and protein: mean and median probability of the true residue, top-1 hit rate, composition baseline. |

## `results/s11_h3n2_ha_tree/`

| File | Rows | Read by | What it is |
|---|---|---|---|
| `branch_codons.tsv.gz` | 50,094 | S11 | One row per internal branch of the H3N2 HA tree (224,731 tips, H3N2 only, dated, 1968 onward) with fewer than 10 nucleotide changes: how many codons took 1, 2 or 3 changes. |
| `column_codons.tsv.gz` | 553 | S11 | One row per codon of the tree's reference record, with its mature H3 site number (blank where H3 has no residue) and how often it changed by 1, 2 or 3 nucleotides. |
| `long_range_nodes.tsv.gz` | 44,965 | S11 (table, row 2) | One row per newest internal node (all children are tips): how many amino-acid differences it has from the oldest scored node, by the fewest nucleotide changes each needed (1, 2, 3) and by the number the tree used. The baseline is `KY348531` (A/Hong Kong/01/1968), not the walk's start node: near the root the reconstruction is largely the reference record, which is an H5. |
| `long_range_changes.tsv.gz` | 2,869 | S11 | The distinct (site, new amino acid) differences seen in any newest node, with the fewest nucleotide changes needed from the oldest node's codon and how many newest nodes carry each. |
| `baseline_codons.tsv.gz` | 535 | S11 (table, null row) | The baseline's own codons, one row per scored codon; the null row counts the fewest nucleotide changes from each to each of the other 19 amino acids. |
| `year_counts.tsv.gz` | 59 | S11 (year panel) | Sequences per collection year over the tree's tips. The tree carries no undated tip and none before 1968. |
| `summary.txt` | — | — | The tool's own run summary. |

## Not shipped, and why

* **Sequence alignments.** GISAID-derived; cannot be redistributed. Guides omit `alignment`.
* **Model checkpoints and protein-model profiles.** About 180 GB; scoring needs a GPU. The
  derived scores are shipped instead.
* **The five-lineage sweep tables.** Dated before the AUROC clamp fix, so they would draw a
  flat floor at 0.5 wherever the score is anti-predictive.
* **`alpha_sweep_fit_metrics_BY_LINEAGE.tsv.preclamp_fix`** and the `*_constrained_fit`
  columns. Any number quoted from them is reading the clamp.

## Rounding

`plm_prob` is rounded to 7 significant digits in the shipped scores (below the precision of the
fp32 model; no ties are created). `mut_prob` is **not** rounded: accessibility takes a few
hundred distinct values that differ in the 8th to 10th digit, and rounding merges them into ties
and moves the archive's correlations in the third decimal.

## Frozen panels

`data/frozen_panels/README.md` lists the panels that have no generator, with provenance.

## Reference genomes

`data/reference_genomes/` holds whole genomes that a stated number in the manuscript is computed
from, so that number can be recomputed from a clean checkout. No figure script reads them.

| File | Accession | Backs |
|---|---|---|
| `NC_045512.2.fasta` | `NC_045512.2` (Wuhan-Hu-1, 29,903 nt) | The genome composition behind the `sars_cov_2.csv` scaling factor of 8.186e-7. Added 22 Oct 2026; the composition was previously stated in Methods with no accession and nothing in the repo to check it against. |

Provenance and fetch command: `data/reference_genomes/README.md`. The scaling arithmetic is
derived in `data/spectra/README.md` and pinned by `tests/test_sars_cov_2_scaling.py`.
