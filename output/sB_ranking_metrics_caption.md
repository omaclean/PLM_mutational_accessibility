**Figure SB. Six ranking metrics across the accessibility weight.** The main figure
reports AUROC over every possible replacement. These are the metrics it displaced,
together with two that ask a narrower question, drawn on the same sweep (weight α from -1
to 1; 0 is the model alone). Lines are means over the 13 lineages and the orange square is
accessibility with no model.

**The top row scores every possible replacement**, observed against unobserved, which is
the main figure's question. (A) PR-AUC, the area under the precision-recall curve, and the
metric first reported. A random ranker scores the prevalence -- the fraction of possible
replacements that were observed -- and prevalence runs from 1.6% to 10.1% between
lineages. There is therefore no single chance line; each lineage's own chance level is a
grey tick in the right margin, and the mean curve is shown because it was reported, not
because the quantity it averages is comparable between lineages. (B) The same number
divided by each lineage's prevalence: fold enrichment over chance, where chance is 1 for
every lineage and the mean is meaningful. (C) Partial AUROC: the ROC area over
false-positive rates up to 5%, standardised so that chance is 0.5 and a perfect ranking is
1. The metric cannot fall below 0.4872, so everything worse than chance is compressed into
the gap beneath 0.5.

**The bottom row is put only to the replacements that were observed at all**, between 172
and 1,090 of them per lineage. (D) Spearman rank correlation between the score and the
observed frequency. (E) The same correlation as a linear one, computed on the logarithm of
the frequency rather than on the frequency itself, as asked in review; observed
frequencies span four orders of magnitude, so on the raw scale the few replacements that
went nearly to fixation set the answer, and the rank correlation in D and the log scale in
E are the two ways of not letting them. (F) AUROC again, with the negative class cut down
to replacements that occurred: positives are the most frequent 5% of a lineage's observed
replacements and negatives are the rest of them, so the zero-frequency replacements take
no part. Panel F asks whether a score can tell a replacement that swept from one that
merely appeared.

**What the bottom row shows.** The fine-tuned model gains essentially nothing from the
explicit accessibility term on any of the three: 0.3701 to 0.3711 at α = +0.06 in D,
0.3550 to 0.3551 at α = +0.04 in E, and 0.7467 to 0.7468 at α = -0.02 in F, every one of
them inside the width of its own band. The base model does gain materially on D and E --
0.1315 to 0.1668 at α = +1 and 0.1227 to 0.1459 at α = +0.92, still rising at the end of
the sweep -- and much less on F, 0.5765 to 0.5858 at α = +0.88. Accessibility on its own
scores 0.533 at picking the winners in F, against 0.909 at saying which replacements occur
at all, the quantity Figure 3 reports. That contrast is the point of this row: mutational
accessibility is informative about what is reachable and close to uninformative about what
succeeds once it is reached, and a model that has learned the reachable part has nothing
left to gain from being told it again.

Bands are 95% percentile bootstrap intervals of the mean, resampling lineages (2,000
resamples). They show the precision of the mean over these 13 lineages. Lineages are
disjoint sets of sequences but can share sites, so the bands do not show how well a model
would do on a new lineage. Every panel but A stops its vertical axis just under chance;
below it the score is anti-predictive. Every plotted value, including those, is in the
values table. Panels D and E are the sweep table's own columns; they, the raw-scale
Pearson correlation that is checked but not drawn, and the scores behind the other four
panels are checked against each other before the figure is drawn, and a disagreement stops
the run.
