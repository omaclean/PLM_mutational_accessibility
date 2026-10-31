**Figure 2D. Model agreement with mutational accessibility, within as well as across the
codon-distance strata.** We then checked whether the mutational signature was being learned
by the PLMs *within* the nt = 1, 2 and 3 strata of codon space as well as across them. The
correlations are notably weaker within strata, although still non-zero, suggesting the
largest component of accessibility learned by the PLM is the distance between codons.

Each dot is a Spearman rank correlation between the model's score and mutational
accessibility, computed over all replacements ("all") or only within replacements needing a
given number of nucleotide changes (1, 2 or 3). Hollow circles are the raw correlation;
filled circles are the same correlation with Grantham similarity partialled out, and carry a
95% percentile bootstrap interval that resamples **sites** (whole positions, with all of that
position's replacements), because replacements at one site share a backbone and are not
independent. The dash is Grantham similarity's own correlation with accessibility. One
lineage (J.2_int) is scored, not a mean over the set. BLOSUM62 is in the values table but is
not drawn: it is a comparator, not a control, since an empirical substitution matrix already
carries mutational accessibility.

### Plotted values

Spearman r. "Raw" = hollow circle; "Grantham removed" = filled circle with 95% interval.

| Stratum | n | Series | Raw r | Grantham removed r | 95% interval | % of raw kept |
|---|---|---|---|---|---|---|
| all | 10,754 | Base model (ESM-2) | 0.2750 | 0.2044 | 0.1845 to 0.2243 | 74% |
| all | 10,754 | Fine-tuned on influenza HA | 0.6549 | 0.6218 | 0.6030 to 0.6431 | 95% |
| 1 | 3,334 | Base model (ESM-2) | 0.0705 | 0.0523 | 0.0244 to 0.0772 | 74% |
| 1 | 3,334 | Fine-tuned on influenza HA | 0.2918 | 0.2914 | 0.2638 to 0.3218 | 100% |
| 2 | 5,487 | Base model (ESM-2) | 0.0705 | 0.0587 | 0.0301 to 0.0898 | 83% |
| 2 | 5,487 | Fine-tuned on influenza HA | 0.2496 | 0.2436 | 0.2189 to 0.2707 | 98% |
| 3 | 1,933 | Base model (ESM-2) | -0.0031 | 0.0319 | -0.0239 to 0.0844 | sign flips |
| 3 | 1,933 | Fine-tuned on influenza HA | 0.1390 | 0.1841 | 0.1392 to 0.2238 | 132% |

Grantham similarity alone (the dash), with its own 95% interval:

| Stratum | Spearman r | 95% interval |
|---|---|---|
| all | 0.2658 | 0.2465 to 0.2851 |
| 1 | 0.0602 | 0.0195 to 0.1061 |
| 2 | 0.0564 | 0.0270 to 0.0830 |
| 3 | -0.1314 | -0.1675 to -0.0966 |

BLOSUM62, table only, not drawn: all 0.3770 (0.3544 to 0.3973); nt = 1 0.0999 (0.0610 to
0.1362); nt = 2 0.1248 (0.0947 to 0.1565); nt = 3 -0.0741 (-0.1072 to -0.0373).

### Reading the numbers

The fine-tuned model's correlation with accessibility falls from 0.655 across all
replacements to 0.292, 0.250 and 0.139 within the one-, two- and three-step strata. The base
model falls from 0.275 to 0.071, 0.070 and -0.003. In both cases most of the across-the-board
agreement is carried by codon distance itself; what is left within a stratum is small but,
for the fine-tuned model, clearly above zero at every distance (all three intervals exclude
0). For the base model only the two-step stratum excludes 0.

Removing Grantham similarity barely touches the fine-tuned model (95% of the correlation
kept at "all", and essentially all of it within strata) but removes about a quarter of the
base model's. At nt = 3 both models' correlations rise once Grantham is removed, because
Grantham similarity is itself anti-correlated with accessibility there (-0.131).

Full precision is in `fig02d_strata_panel_values.tsv`.
