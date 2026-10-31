**Figure SO. What the wild-type (unmasked) estimator would have given.** Model
probabilities in the main figures are masked marginals: the site being scored is replaced
by `<mask>` and predicted from its context, so the residue actually there cannot inform
its own prediction. The alternative, used in an earlier version of this analysis, is the
wild-type (unmasked) marginal, read from a single forward pass of the intact reference in
which the model does see that residue. This figure shows what the second estimator does to
the results. (A--D) The Influenza H3N2 panels of Figure 4 under both: Spearman rho between
`log(model probability) + alpha * log(accessibility)` and the measured phenotype, swept
over alpha. Solid lines are the masked marginals the main figures report, dashed lines the
wild-type (unmasked) estimator, and the shaded band is the masked arm's 95% site
bootstrap. Filled and open circles mark each curve's optimum where it is interpretable.
(E) The same comparison without a phenotype: the Spearman correlation of model probability
with mutational accessibility over all replacements of one lineage, and within each
codon-distance class. Filled markers are masked marginals, open markers the wild-type
(unmasked) estimator, bars are 95% site bootstrap intervals. 3 of 8 masked estimates fall
outside the unmasked interval. No Grantham or BLOSUM62 control is applied here; the
question is only what the estimator does. Figure 2D itself is drawn from a single unmasked
pass and stays that way: this panel is the record of what that costs, which is what the
DMS figures needed the masked estimator for and this quantity does not.

Dropping the mask does not move the untuned model much (change in peak rho from -0.011 to
+0.016 across the 4 panels) but it lowers the fine-tuned model in 4 of 4 (by 0.016 to
0.079). That asymmetry is why the masked estimator is the one reported: a model fine-tuned
on a single protein family can reproduce a residue it is shown very closely, which
compresses the probability it has left for the alternatives -- and it is the ranking of
those alternatives that is being scored. An unmasked pass therefore penalises the
fine-tuned model for having learned the family, and the penalty is an artefact of the
estimator rather than a property of the model.

The choice does not change the direction of this manuscript's claims: the optimum sits at
a negative alpha in all 16 panel-model-estimator combinations (most positive -0.01), so
accessibility is still being divided out rather than added in under either estimator. What
it changes is the size of the gap between the two models. The two arms are computed by the
same code over the same arms in the same order with the same bootstrap seed, swapping
nothing but the probability column; the unmasked arm recomputed alongside reproduces the
originally published scores to 3.1e-05, which is what makes masking the only difference
between the two.
