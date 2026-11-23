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
and 1,090 of them per lineage. (D) Pearson regression on all non-zero allele frequencies,
weighting the highest-frequency alleles most strongly; increased α weighting on mutational
accessibility shows little benefit to the base model (0.0354 to 0.0391 at α = +0.68), and
no benefit to the fine-tuned model (0.1239 to 0.1242 at α = -0.1). (E) Pearson correlation
on logged non-zero frequencies which improves with higher α values (0.1227 to 0.1459 at α
= +0.92), as the mutational accessibility alone outperforms the base model (0.085 vs
0.123), perhaps because of low-frequency homoplasies, or because amino acid
interchangeability correlates with accessibility; the fine-tuned model gains essentially
nothing (0.3550 to 0.3551 at α = +0.04). (F) Spearman correlation across non-zero
frequencies, showing a similar trend to the logged Pearson score (0.1315 to 0.1668 at α =
+1 for the base model; 0.3701 to 0.3711 at α = +0.06 for fine-tuned).

**What the bottom row shows.** The fine-tuned model gains essentially nothing from the
explicit accessibility term on any of the three: 0.1239 to 0.1242 at α = -0.1 in D, 0.3550
to 0.3551 at α = +0.04 in E, and 0.3701 to 0.3711 at α = +0.06 in F, every one of them
inside the width of its own band. The base model does gain materially on E and F -- 0.1227
to 0.1459 at α = +0.92 and 0.1315 to 0.1668 at α = +1, still rising at the end of the
sweep -- and gains little on D, 0.0354 to 0.0391 at α = +0.68. Accessibility on its own
scores 0.020 in D, 0.085 in E, and 0.101 in F.

Bands are 95% percentile bootstrap intervals of the mean, resampling lineages (2,000
resamples). They show the precision of the mean over these 13 lineages. Lineages are
disjoint sets of sequences but can share sites, so the bands do not show how well a model
would do on a new lineage. Every panel but A stops its vertical axis just under chance;
below it the score is anti-predictive. Every plotted value, including those, is in the
values table. Panels D, E and F are read from the shipped sweep table and cross-checked
against the per-replacement tables, and any disagreement stops the run.
