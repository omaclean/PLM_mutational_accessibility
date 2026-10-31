**Figure 3. Prediction of observed amino-acid replacements as the weight on mutational
accessibility changes.** Each panel scores how well the product of the model's
pseudo-probability and accessibility^α predicts replacements seen in sequence data, as the
weight α on accessibility runs from -1 to 1. α = 0 is the model alone: it is marked by the
dashed vertical rule, and each curve is read off where it crosses that rule. The orange
square is accessibility with no model at all. The unit throughout is one candidate
replacement -- each of the 19 non-reference amino acids at every site, scored within a
lineage. "PLM score" in the panel subtitles is the quantity being swept: the protein
language model's **pseudo-probability** for that amino acid at that site -- the
probability the masked model assigns it, which is a score over the 20 amino acids and not
a calibrated probability that the replacement will happen -- multiplied by
accessibility^α. At α = 0 it is the pseudo-probability alone. (A) Spearman rank
correlation between the score and the observed frequency of every possible replacement,
including the ones never seen (frequency 0). (B) AUROC over all possible replacements
(area under the receiver-operating-characteristic curve): the chance that a replacement
that occurs scores higher than one that does not; 0.5 is chance. (C) AUROC among observed
replacements: distinguishing the most frequent 5% of observed replacements from the
remaining observed replacements ("picking the winners"). While mutational accessibility
strongly predicts which replacements occur at all ((B), accessibility alone = 0.909), it
provides almost no signal for distinguishing high-frequency from low-frequency circulating
variants ((C), accessibility alone = 0.524, near chance). Weighting in mutational
accessibility provides essentially no additional predictive gain for the fine-tuned model
in (C) (0.7467 at α = 0 to 0.7468 at best α), indicating that mutational accessibility
governs evolutionary reachability rather than post-emergence fitness or fixation.

Each vertical axis stops at the no-relationship level (0, or 0.5 for AUROC). Below it the
score is anti-predictive. In (A) and (B) the curves fall under that line for some weights
and are not drawn there. In (C) no curve falls below the line. All plotted values,
including those, are in the accompanying values table.

Lines are means over 13 lineages, each scored separately. Bands are 95% percentile
bootstrap intervals of that mean, resampling lineages. They show the precision of the mean
over these 13 lineages. Lineages are disjoint sets of sequences but can share sites, so
the bands do not show how well a model would do on a new, unrelated lineage.
