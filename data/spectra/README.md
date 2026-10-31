# Nucleotide mutation spectra (4x4, coding sense, order A C G T)

Rows are the base changed from, columns the base changed to. Used by `mutacc/accessibility.py`,
and compared against each other in codon space by `tools/spectra_compare.py` for S4 and S6.

## `spectra_sources.csv`

The list of record: one row per matrix, and the only place a spectrum is named or described.
`tests/test_accessibility.py` asserts that the files on disk and the rows here are the same set,
so an undeclared matrix or a declared one with no file is an error, not a silently plotted subset.

| Column | What it is |
|---|---|
| `key` | File stem of the matrix, `<key>.csv` in this directory. |
| `label` | Display name. The figures take every name from here and hard-code none. |
| `kind` | `virus`, `cellular` or `analytic`. |
| `free_nt_parameters` | How many free parameters the nucleotide model has: 12 for a measured spectrum, 6 strand-symmetric, 1 for K80, 0 for JC69. The x-axis of S4 panel A. |
| `citation`, `doi` | Where the matrix came from. |
| `organism_group` | Blank, or a name shared by every row measuring the same organism. Rows sharing a group are counted **once** when asking which single matrix is the best generic stand-in, so two subtypes of one virus cannot inflate each other's mean agreement. |
| `manuscript_role` | `focus` for the spectrum the manuscript's claim is about, `comparator` for the matrix it was contrasted with, blank otherwise. Exactly one row must be `focus`. S6's caption reads the published numbers against these rows; attributing them to the merely *simplest* matrix instead would be a false provenance claim, and a test pins that it does not. |

| File | Source | Scale |
|---|---|---|
| `influenza_h3n2.csv`, `influenza_h1n1.csv` | Pauly et al. (2017), eLife 6:e26437. Reported in the **coding (+) sense**, so no strand flip is needed. | per-site probabilities (row sums < 1) |
| `sars_cov_2.csv` | De Maio et al. (2021), Genome Biol Evol 13:evab087, Table 1, scaled as in the preprint methods (derivation below; composition from `NC_045512.2`). Positive-sense reference genome, so coding sense. | per-site probabilities |
| `human.csv` | Francioli et al. (2015) | relative |
| `ecoli.csv`, `yeast.csv` | cellular spectra, built strand-symmetric (six free classes mirrored) | relative |
| `jc69.csv`, `k80_kappa2.csv` | Jukes-Cantor flat; Kimura 80 with kappa = 2 | relative |

"Relative" matrices have row sums above one; `as_probabilities(q, row_sum=...)` rescales them. Within a
nucleotide-step stratum the ranking does not depend on that scale.

## Scales used by the pipeline

* `human.csv` is relative. The pipeline multiplies it by **1.2e-8** (the human per-site per-generation
  mutation rate) before use; pass `load_spectrum(...) * 1.2e-8`. With that scale `mutacc.accessibility`
  reproduces the simulation's `mut_prob_Human_Germline` column to floating-point precision.
* `sars_cov_2.csv` and the influenza matrices are used as given. The absolute scale already in
  `sars_cov_2.csv` is derived below.
* `jc69.csv`, `k80_kappa2.csv`, `ecoli.csv` and `yeast.csv` have no absolute scale in the archive's
  pipeline; use `row_sum=` and prefer within-stratum comparisons, which do not depend on it.

## The SARS-CoV-2 scaling factor

`sars_cov_2.csv` is De Maio et al. (2021) Table 1 — relative rates — multiplied by a single
factor so the matrix carries an absolute per-site per-replication scale. The Methods quote that
factor as **8.186e-7**, from a mutation rate of **1.3e-6** per site per replication (Amicone et
al. 2022) and the SARS-CoV-2 genome composition.

The factor `f` is the one making the composition-weighted mean rate equal 1.3e-6:

```
f = 1.3e-6 / sum_i pi_i * sum_j q_ij
```

with `q` the published relative rates and `pi` the base frequencies of **`NC_045512.2`**
(Wuhan-Hu-1), which now ships at `data/reference_genomes/NC_045512.2.fasta`. Its composition is
29.94% A, 18.37% C, 19.61% G, 32.08% T — the Methods' "29.9 / 18.4 / 19.6 / 32.1", rounded.

Equivalently, and the form the tests use because it needs no un-scaled values: the
composition-weighted mean of the **shipped** matrix must already be 1.3e-6. It is 1.3010e-6,
**0.076% high**.

That residual is stored precision, not a different scaling. The matrix is written to three
significant figures; perturbing every entry by half of its last digit moves the weighted mean
over `[1.2983e-6, 1.3037e-6]`, which contains 1.3e-6. Recovering the relative rates by dividing
the shipped matrix by 8.186e-7 and re-deriving the factor gives 8.1798e-7 for the same reason.

**It does not reach any result.** Accessibility is read as a ranking within a nucleotide-step
stratum, where a common positive scale cancels — the same reason the relative matrices can be
used with an arbitrary `row_sum`. The factor fixes the units the Methods report and nothing else.

Pinned by `tests/test_sars_cov_2_scaling.py`.
