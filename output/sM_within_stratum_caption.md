**Figure SM. Model score against measured fitness, within each nucleotide-step stratum.**
Mutational accessibility and the model score are both shaped by the genetic code, so a
correlation with measured fitness could be that shared dependence. In each panel the
replacements are split by the minimum number of nucleotide changes needed to make them (1,
2 or 3), so within a group every replacement is equally easy to reach. Bars are the
Spearman rank correlation between the model score (the model alone, no accessibility
weighting) and the effect measured by deep mutational scanning (DMS), for a base model and
the same model fine-tuned on that virus. The model probability is the masked marginal: the
scored site is replaced by the mask token and predicted from its context, so the residue
actually there cannot inform its own prediction. Each group holds 304 to 10,754
replacements at 160 to 1,240 sites. Error bars are 95% percentile intervals from a
bootstrap that resamples DMS sites, not replacements, because replacements at one site
share a backbone and are not independent.

In the assays that measure fitness, the interval lies wholly above zero for 22 of 28
base-model bars and 28 of 28 fine-tuned-model bars. Overlapping error bars do not show
that two models are alike: across the 28 (assay, stratum) cells, the difference between
fine-tuned and base correlation (resampling the same sites for both models, 400 resamples)
excludes zero in 18, with the fine-tuned model higher in 14 and lower in 4. Those
differences are in the accompanying table. Panels marked as receptor binding measure
binding rather than fitness, and many replacements raise binding, so the correlation there
is a property of the assay and not a model failure; they are excluded from these counts.
The overlaid purple curve and markers show the mean experimental phenotype (normalized as
a z-score relative to all assayed variants in each dataset, ± SEM) across distance groups
(right axis). In cell entry and growth preference assays, functional viability decays
sharply as codon distance increases, matching the drop in PLM prediction accuracy; in
immune escape assays, the phenotype remains flat. The purple stars test that curve itself:
three means with SEM whiskers do not show whether the strata are drawn from the same
distribution. A Kruskal-Wallis test asks whether the measured effect differs across the
three nucleotide-step strata. It is a rank test because DMS scales differ between assays
and several are bounded, so a test on the raw values would partly be testing the scale;
and it is run over the strata only, the pooled group being the union of those same
replacements. Each replacement is counted once, not once per model. The test is
significant in 6 of 9 panels, marked "KW" at the top right of the panel. Where it is,
three pairwise Mann-Whitney tests, Holm-corrected within the panel, say which stratum
carries it, and a star is drawn over a stratum only where it separates from both of the
others -- which is what a single odd group out can mean when there are three (3 panels
have one; a panel whose strata simply rank in order keeps its "KW" mark and gets no star).
Stars are * p < 0.05, ** p < 0.01, *** p < 0.001, and the star over a stratum carries the
weaker of its two corrected p-values, since the claim is only as strong as its lesser
half. Every statistic and p-value, corrected and raw, is in
sM_dms_stratum_tests_values.tsv.

(**J**) The accessibility weight refitted inside one stratum. The score of figure 4,
log(model probability) + α log(accessibility), was swept over α and rank-correlated with
the measured effect again, separately within each group of replacements. Each point is one
assay-and-model arm, the violin its distribution, the bar its median. Fitted across all
replacements at once, a negative α is favoured in 11 of 14 arms. Fitted inside a single
stratum it is still favoured in 33 of 42 (14 of 14 at 1 nt; 12 of 14 at 2 nt; 7 of 14 at 3
nt), and the interval excludes zero in 16. So down-weighting accessible replacements still
helps once the number of nucleotide changes is held fixed, and the weighting is not purely
a restatement of that count — but only at one and two changes. At three it is a coin flip
(7 of 14, median α* +0.01), where the replacements are rarest and the measured effects
weakest. The help is small throughout: the median gain in ρ over α = 0 is 0.008 (largest
0.091).

Two cautions. α is not comparable across the groups on the axis as drawn. Fixing the
number of changes removes most of the spread in log(accessibility) — pooled it is about 4
times wider than within one stratum (the SD under each tick) — so the same real amount of
down-weighting needs a correspondingly larger α. On a scale-matched footing, α* × SD, the
median weight is -0.36 pooled against -0.29 within strata, so most of the pooled effect is
the step count after all. Second, the 16 receptor-binding arms (BA.2 ACE2 binding, XBB.1.5
ACE2 binding) are left out of this panel and of every count above. Those assays measure a
phenotype that runs *against* fitness, so their correlations are negative across the whole
sweep and the largest one falls wherever the model is diluted hardest towards zero: 14 of
them put the optimum hard against the end of the swept range, which is where α runs when
there is no peak to find. No optimum is defined for them, and reading one would be
meaningless. Every arm, binding included, is in the accompanying table, and Figure SP
draws every curve.

Nothing else is set aside, which is why every group above has the same denominator: an arm
counted in one stratum is counted in all of them. An optimum at the end of the swept range
is kept (3 of the arms counted above) — where a curve runs plainly to that edge it is a
result, though a bound rather than a turning point. So are the 2 arms whose sweep found
nothing at all, no correlation anywhere and no gain from moving α; their α* sits beside
zero and counts as no preference, which can only weaken the tallies above, never
strengthen them. Figure SP draws every curve behind these counts.

Pooled, the two models separate completely: all 7 fine-tuned arms prefer a negative α,
between -0.24 and -0.13, while the 7 base arms fall between -0.06 and +0.04, only 4 of
them negative. The ranges do not overlap, so an arm's model can be read off its preferred
weight alone. Dividing accessibility back out helps where the model absorbed it, and the
fine-tuned model is the one that did.
