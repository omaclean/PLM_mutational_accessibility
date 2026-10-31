# Figure 3 under a masked-marginal estimator

Thirteen H3N2 lineage roots, guide `h3n2_13lineage`, 10,754 candidate replacements per
lineage per model (566 sites x 19 non-reference amino acids). The published panels read the
model's probability from a single unmasked forward pass over each lineage root, so the model
sees the residue it is scoring. Here each position is masked in turn (566 passes per lineage
per model) and everything else is held fixed: the same rows in the same order, the same
101-point alpha grid from the run manifest, the same `pseudocount=1e-16`, the same
accessibility column, the same bootstrap seed, and `figures/make_fig03.py` itself for the
drawing.

Built by `dev/masked_fig03/run_all_lineages.sh` then `dev/masked_fig03/build_masked_fig03.py`.
Both arms are in `fig03_alpha_sweep_masked_values.tsv.gz` under a `scoring` column.

## Gates

| gate | result |
| --- | --- |
| recomputed unmasked sweep reproduces the shipped `alpha_sweep_by_lineage`, 2,626 curve rows, 3 plotted columns | max abs diff 7.25e-15 |
| the same, over all 31 shared numeric columns | max abs diff 1.82e-02 (`site_rank_spearman_p`) |
| recomputed unmasked arm reproduces the shipped `fig03_alpha_sweep_values.tsv` (mean, lo, hi) | max abs diff 5.0e-07 |
| recomputed accessibility-only baseline reproduces the shipped baseline rows, 26 rows | max abs diff 3.31e-05 (tolerance 1e-04, see below) |
| unmasked profile reproduces the archive's `plm_prob` for `ESM2_650M_HA80_raw` | max abs diff 8.67e-06 |
| unmasked profile reproduces the archive's `plm_prob` for `ESM2_650M_HA80` | max abs diff 4.98e-06 |
| accessibility-only baseline identical across the two arms | exact, 26 rows |

The three columns the figure plots are reproduced on the curve rows to 7e-15, which is the
float round trip of the shipped table and not a difference in the sweep: the curves are the
published computation, not a lookalike. 5.0e-07 against the shipped values file is exactly
half its last printed decimal.

Two gates are looser and both are stated rather than waved through. The profile gates are
float32 round trips of the published `plm_cache` (the GPU scoring gated against that cache
itself, and passed on all 26 lineage-model pairs at 8.7e-06 or better). The baseline gate is
looser for a different reason: those rows score `mut_prob` with no model at all, and
`mut_prob` takes 861 distinct values over a lineage's 10,754 replacements because sites
sharing a codon share their accessibility exactly. A rank statistic on a score that tied is
decided in its fifth decimal by which ties survive the last bit of a `log` — on C.1 the
published Spearman is 0.3346284, scoring `mut_prob` directly gives 0.3346301 and scoring
`log(mut_prob)`, the same ordering in exact arithmetic, gives 0.3346235. The curve rows are
not like this because `plm_prob` breaks every tie. The orange square moves by 1.0e-05 between
the shipped figure and this one, which is below the width of the marker, and it is identical
between the two arms here because it does not depend on the model.

## Result

Mean over the 13 lineages. The alpha\* columns are each curve's own best weight on the
published grid; the gain is that curve's value at alpha\* minus its value at alpha = 0.

| panel | model | at alpha = 0, unmasked | masked | delta | alpha\* | at alpha\*, unmasked | masked | delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A Spearman | Base (ESM-2) | 0.1907 | 0.2054 | +0.0146 | +1.00 -> +1.00 | 0.3014 | 0.3037 | +0.0023 |
| A Spearman | Fine-tuned (HA-80) | 0.2943 | 0.3014 | +0.0071 | +0.66 -> +0.58 | 0.3143 | 0.3185 | +0.0042 |
| B AUROC | Base (ESM-2) | 0.7721 | 0.7927 | +0.0205 | +1.00 -> +1.00 | 0.9273 | 0.9305 | +0.0032 |
| B AUROC | Fine-tuned (HA-80) | 0.9195 | 0.9294 | +0.0099 | +0.62 -> +0.58 | 0.9464 | 0.9523 | +0.0059 |
| C Pearson | Base (ESM-2) | 0.0419 | 0.0455 | +0.0036 | +0.72 -> +0.54 | 0.0463 | 0.0493 | +0.0030 |
| C Pearson | Fine-tuned (HA-80) | 0.1154 | 0.1253 | +0.0099 | +0.02 -> -0.04 | 0.1154 | 0.1253 | +0.0099 |

Gain from weighting accessibility in, which is what the figure is about:

| panel | model | gain unmasked | gain masked | delta |
| --- | --- | --- | --- | --- |
| A Spearman | Base (ESM-2) | +0.1107 | +0.0983 | -0.0124 |
| A Spearman | Fine-tuned (HA-80) | +0.0200 | +0.0171 | -0.0029 |
| B AUROC | Base (ESM-2) | +0.1552 | +0.1378 | -0.0173 |
| B AUROC | Fine-tuned (HA-80) | +0.0269 | +0.0229 | -0.0040 |
| C Pearson | Base (ESM-2) | +0.0044 | +0.0038 | -0.0006 |
| C Pearson | Fine-tuned (HA-80) | +0.0000 | +0.0000 | +0.0000 |

The masked alpha = 0 value against the published panel's own 95% bootstrap interval. `lo`/`hi`
are the published interval; the masked point is not a draw from it, so "outside" means the
shift is large next to the spread between lineages, not that it is significant.

| panel | model | published 95% at alpha = 0 | masked | outside? |
| --- | --- | --- | --- | --- |
| A Spearman | Base (ESM-2) | [0.168, 0.215] | 0.205 | no |
| A Spearman | Fine-tuned (HA-80) | [0.258, 0.334] | 0.301 | no |
| B AUROC | Base (ESM-2) | [0.765, 0.778] | 0.793 | **yes** |
| B AUROC | Fine-tuned (HA-80) | [0.914, 0.925] | 0.929 | **yes** |
| C Pearson | Base (ESM-2) | [0.007, 0.078] | 0.045 | no |
| C Pearson | Fine-tuned (HA-80) | [0.076, 0.160] | 0.125 | no |

Accessibility alone, the orange square, is 0.2885 (A), 0.9089 (B) and 0.0212 (C) in both arms.

## What it says

Masking lifts every curve in every panel, at every weight. Both models are better predictors
of which replacements are observed than the published estimator makes them look, and the
unmasked wild-type leak was costing them most where the model is weakest: the base model
gains twice as much as the fine-tune at alpha = 0 in panels A and B (+0.015 against +0.007
Spearman, +0.021 against +0.010 AUROC). Panel C runs the other way, where the fine-tune gains
more (+0.0099 against +0.0036).

The figure's own claim survives, with a tenth shaved off it. Weighting accessibility in still
helps both models and still helps the base model many times more than the fine-tune: the base
model's Spearman still climbs 0.098 from alpha = 0 to its best weight against the fine-tune's
0.017, and its AUROC 0.138 against 0.023. Those gains are smaller than the published ones by
11% on the base model and 15% on the fine-tune, so the unmasked estimator was overstating how
much accessibility adds, but only by that much. Accessibility alone (0.289 Spearman, 0.909
AUROC) still beats the base
model scored at alpha = 0 (0.205, 0.793) after masking, which is the panel's headline and is
not an artefact of the leak.

The best weight barely moves and stays positive where the figure says it is. In panels A and B
the base model's optimum sits on the right-hand edge of the grid under both estimators, so
alpha\* = +1.00 is a grid boundary and not a measured optimum — that was already true of the
published panel. The fine-tune's interior optima shift a little towards zero (+0.66 -> +0.58
in A, +0.62 -> +0.58 in B). Panel C's fine-tuned optimum is at alpha = 0 under both estimators
(+0.02 -> -0.04) on a curve that is flat to four decimals across that span, so "the fine-tune
needs no accessibility weighting for the replacements that were actually seen" reads the same
way after masking.

What masking does not do is change the ordering of the two models. The fine-tune's lead at
alpha = 0 narrows slightly in A and B (+0.104 -> +0.096, +0.147 -> +0.137) and widens in C
(+0.074 -> +0.080); at each model's own optimum it grows in all three (+0.013 -> +0.015,
+0.019 -> +0.022, +0.069 -> +0.076). Two of the six alpha = 0 points — both in panel B — fall
outside the published bootstrap interval. That interval resamples lineages, and masking moves
every lineage in the same direction, so a shift this size clears it exactly where the lineages
agree most; it is not evidence that the published number was wrong.

Nothing here is a published number. Figure 3 as drawn in `output/fig03_alpha_sweep.*` is
unchanged.
