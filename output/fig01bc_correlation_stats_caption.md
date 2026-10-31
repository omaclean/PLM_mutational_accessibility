**Figure 1. Supporting table: correlation of model probability with mutational
accessibility, on every scale.** Each row is one coefficient for one model, one lineage
and one group of replacements. `scale` says which of the two variables was logged -- `raw`
neither, `log10_plm` the model probability only, `log10_mut` the accessibility only,
`log10_both` both -- and `plm_transform` and `mut_transform` repeat that one variable at a
time. `rank` is the Spearman correlation, written once because no monotone transform of
either variable can change it. The replacements are every single-residue replacement away
from a lineage's reference protein (10,754 of them per lineage), the same set panels B and
C rank and Figure 2A and 2B plot.

On the lineage the single-lineage figures are drawn from (`J.2_int`), over all
replacements, for the model labelled "Base model (ESM-2)" and for "Fine-tuned on influenza
HA" in turn: 0.054 and 0.045 with neither variable logged, 0.165 and 0.415 with the model
probability logged, 0.077 and 0.065 with the accessibility logged, 0.283 and 0.627 with
both, and rank correlations of 0.275 and 0.655.

The first pair is the coefficient printed under Figure 2A and 2B. Both probabilities span
about nine orders of magnitude, so a correlation taken on either untransformed is set by a
handful of large values -- but the two semi-log rows show the two variables are not
symmetric in this. Logging the model probability alone recovers a large part of the
distance to the joint log scale -- about half of it for the base model and more for the
fine-tuned one -- while logging the accessibility alone recovers little of it for either,
and for the fine-tuned model leaves the coefficient near raw. The raw coefficient is
therefore held down mainly by the skew of the model's own probabilities rather than by the
accessibility it is being compared with. A fraction of variance explained is only
interpretable beside the scale it was taken on, which is why `r_squared` never appears
without `scale`.

`ci_lo` and `ci_hi` are a 95% percentile bootstrap interval over sites, 1000 draws: the 19
replacements at a site share a codon and a model context, so resampling replacements
individually would report a precision the data do not have. No p-value is written, for the
same reason. Lineages are listed separately and never pooled, because the guide's lineages
are scored at the same sites and pooling them stacks near-copies of one cloud.

Written by `tools/export_fig01bc_correlations.py` from the guide's `replacement_scores`
table, the shipped per-replacement table that `tools/export_fig01bc_values.py` also reads.
