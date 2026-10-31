**Table SW. The weight on mutational accessibility chosen without the lineage it is scored
on, and how tightly the deep mutational scans pin that weight down.** Figure 3 reads its
result off the peak of the curve averaged over all 13 influenza A/H3N2 haemagglutinin (HA)
lineages, and reports what that weight scores on those same lineages. This table removes
that circularity. Each lineage is held out in turn; the weight alpha is chosen at the peak
of the mean curve over the other 12 lineages, with the held-out lineage taking no part in
the choice; and that weight is then scored on the held-out lineage alone. The gain is its
score there minus its score at alpha = 0, which is the model with no accessibility
weighting at all. Every lineage is held out exactly once, giving 13 separate choices of
weight and 13 gains per model and metric. The folds are separate, not independent: any two
of them train on 11 of the same 12 lineages, and the lineages themselves are not
independent of one another either (see below).

The fine-tuned model gains +0.023 [+0.019, +0.027] in AUROC, with every fold's weight
landing at +0.56 to +0.58 and 13 of 13 folds above zero; and +0.071 [+0.053, +0.090] in
PR-AUC, with every fold's weight landing at +0.56 to +0.58 and 13 of 13 folds above zero.
The base model gains +0.140 [+0.133, +0.147] in AUROC, with every fold's weight landing at
+1.00 and 13 of 13 folds above zero; and +0.181 [+0.142, +0.223] in PR-AUC, with every
fold's weight landing at +1.00 and 13 of 13 folds above zero. In 52 of the 104 folds the
held-out weight sits at an edge of the searched range (-1.00 to +1.00), all of them Base
model (ESM-2): the training curve never turns over there, so that weight is a bound rather
than an estimate. The masked-marginal rows are quoted above because they are the arm
Figure 3 draws; the wild-type (unmasked) rows are in the table and agree in sign
throughout.

**The deep mutational scans, where no scan can be held out.** The 18 rows with `scope` =
`dms_alpha_star` are the 9 panels of Figure 4 and Figure SL, each scored by a base and a
virus-specific fine-tuned model. Each panel is a single experiment, so there is no second
scan to choose the weight on; what can be asked instead is how well the measured sites pin
the chosen weight down. Sites are resampled with replacement -- replacements at one site
share a backbone and are not independent -- the whole correlation-versus-alpha curve is
recomputed on the resampled replacements, and **that replicate's own best weight is
taken**, 1,000 times per row. `alpha_star` is the weight the figure marks, chosen on all
the sites; `estimate` is the median of the resampled weights and `ci_lo` to `ci_hi` their
2.5th and 97.5th percentiles. Over the 4 interpretable rows for the fine-tuned models on
the Influenza H3N2 panels the selected weight is -0.24 to -0.18, with 95% intervals
running from -0.28 to -0.14 and spanning 0.03 to 0.13. Over the 4 interpretable rows for
the base models on the Influenza H3N2 panels the selected weight is -0.06 to -0.03, with
95% intervals running from -0.20 to +0.17 and spanning 0.03 to 0.37. Over the 5
interpretable rows for the fine-tuned models on the SARS-CoV-2 panels the selected weight
is -0.32 to -0.13, with 95% intervals running from -0.47 to -0.06 and spanning 0.05 to
0.23. Over the 2 interpretable rows for the base models on the SARS-CoV-2 panels the
selected weight is +0.03 to +0.04, with 95% intervals running from -0.02 to +0.10 and
spanning 0.10 to 0.11.

What those intervals are not. The best weight is an **argmax**, which jumps: a replicate
in which one grid point tips above its neighbour moves the answer by a whole step, so the
resampled distribution is lumpy on the 0.01 grid and can have more than one mode. What is
quoted is a percentile interval of that distribution and nothing more -- not a confidence
interval with a coverage guarantee, which would need the smooth, asymptotically normal
statistic an argmax is not. Nothing is smoothed, trimmed or re-centred to tighten it:
where a curve is a broad plateau the interval comes back wide, and here they span 0.03 to
2.00 in alpha. A wide one is the finding that those sites do not locate the weight, not a
defect of the estimator. In 3 of the 18 rows more than 5% of the replicates put the
optimum at an end of the searched range (up to 34%, on BA.2 ACE2 binding, Base (ESM-C),
XBB.1.5 ACE2 binding, Base (ESM-C) and XBB.1.5 sera escape, Base (ESM-C)). Those
replicates found the end of the range rather than a peak, and an interval that includes
them is not an interval on an interior optimum. 3 of the 18 rows are not interpretable at
all and their `alpha_star` must not be quoted: 1 put the best weight at the end of the
searched range and 2 more have too little correlation at alpha = 0 for a weight to improve
on.

Columns. `scoring` is the model probability used -- the masked marginal the manuscript
reports, or the wild-type marginal used before 19 Oct 2026; the DMS rows are the masked
arm only, which is the one Figure 4 draws. `metric` is the ranking metric, or
`dms_spearman_rho` for a DMS row, whose curve is a Spearman rank correlation with the
laboratory measurement. `model`, `model_label` and `model_role` name the scoring model.
`scope` is `fold` for one held-out lineage, `mean` for the summary over them, and
`dms_alpha_star` for a DMS panel. `held_out_lineage` names the lineage taking no part in
the choice; `alpha_star` is the weight chosen without it and `n_tied_alpha` how many
weights shared that training peak; `metric_at_alpha_star` and `metric_at_reference` are
the held-out lineage's score at that weight and at alpha = 0. `estimate` is the difference
between them -- the gain -- on a fold row, and the mean of those gains on the summary row,
where `ci_lo` and `ci_hi` bracket it. `n_folds_gain_positive` counts the folds whose gain
is above zero.

On a DMS row the same three weight columns hold the selected weight, the correlation at it
and the correlation at alpha = 0, and `estimate`, `ci_lo` and `ci_hi` hold the resampled
median and interval described above. `panel_label`, `organism` and `trait` name the scan
and the phenotype; `n` and `n_sites` count the replacements measured and the protein sites
they sit at; `n_boot` is the number of resamples that produced a usable curve.
`frac_boundary_low` and `frac_boundary_high` are the fractions of those resamples whose
best weight landed at the negative and positive ends of the searched range. `boundary`,
`weak_signal` and `interpretable` are the flags of the point estimate, on the same rule
the figure marks its optima with: **read `alpha_star` only where `interpretable` is
True.** Columns that belong to the other kind of row are left empty rather than filled
with a stand-in.

Intervals on the lineage rows are 95% percentile bootstrap intervals of the mean over
folds, resampling the 13 held-out gains. They are the precision of that mean. The
selection is not re-run inside a resample, because it has already been made out of sample
in every fold; and, as for every interval in this project, the lineages share their sites,
so the interval does not say how a new, unrelated lineage would behave. Intervals on the
DMS rows are 95% percentile intervals over resampled sites, and there the selection **is**
re-run inside every resample, because that is the quantity being bracketed. Neither kind
says how an unrelated lineage or an unmeasured protein would behave.

**No p-value is given, and none should be computed from these rows.** The lineages are not
independent replicates -- their reference proteins are 96.6 to 99.7% identical and the
effective number of independent lineages is between 1.01 and 1.12 -- so a sign test or a
paired t-test over the folds would be anti-conservative by about that factor. A sign test
has a second problem on top: 13 of 13 is its floor whatever the effect size, so it would
report the fold count and call it evidence. The effect size, its interval and the fold
count are the result. The DMS rows carry none either, and for a reason of their own: the
question there is where one weight sits, which the interval answers directly, and the two
models of a panel score the very same replacements, so any set of such p-values would be
correlated by construction.

**Source.** The shipped per-lineage alpha sweeps, both scoring arms: 8 combinations of
arm, metric and model, each resolved into 13 folds. Computed by `mutacc.loo_alpha`;
nothing here is read from an earlier table of results. The DMS rows come from the same
shipped scan scores and masked-marginal probabilities Figure 4 is drawn from, searched
over the same weight grid, and are computed by `mutacc.dms_alpha_ci`; their `alpha_star`
is the optimum `sY_dms_alpha_optima.tsv` reports for the same panel and model, recomputed
rather than copied.
