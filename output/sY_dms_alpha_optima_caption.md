**Table SY. Best weight on mutational accessibility for each deep mutational scan.** One
row for each measured phenotype and each model, summarising the 9 panels of the two DMS
sweep figures (Dadonaite et al. 2024, Lee et al. 2018, Yu et al. 2026).

Every single amino-acid replacement in a scan is scored as log(model probability) + alpha
x log(accessibility). Alpha is the weight on mutational accessibility: 0 is the model on
its own, negative values divide accessibility out, positive values add more of it.
Accessibility is the probability, under the nucleotide change spectrum measured for that
virus, that the reference codon mutates to a codon for the replacement's amino acid. `rho`
is the Spearman rank correlation between that score and the laboratory measurement. The
model is run on each scan's own reference sequence, and its probability for a replacement
is the masked marginal: the scored site is replaced by the mask token and predicted from
its context, so the residue actually there cannot inform its own prediction. The figures'
own supplement shows what the wild-type (unmasked) estimator gives instead.

Columns. `dataset`, `trait` and `panel_label` name the scan and the phenotype measured,
`panel_label` being the name the figures use; `organism` and `citation` give the virus and
the source publication; `model_name`, `model_label` and `model_role` name the scoring
model, each panel pairing a base (untuned) model with the same architecture fine-tuned on
that virus; `n` and `n_sites` count the replacements scored and the protein sites they sit
at; `alpha_best` is the alpha giving the highest `rho`, searched over -1.00 to +1.00 in
steps of 0.01; `rho_best` and `rho_alpha0`, each with a 95% interval, are the correlation
at `alpha_best` and for the model alone; `delta_rho` is the difference between them, which
is what reweighting accessibility buys, and `delta_rho_in_size` is that difference taken
in size. The two differ only where a correlation is negative: the sweep places every
optimum by the size of the correlation, because a receptor-binding curve that grows more
negative has a stronger relationship, not a weaker one, and a signed difference reads
backwards there.

Three flags. `accessibility_helps` is True where `rho_best` is larger in size than either
end of the alpha = 0 interval -- a weak screen, not a paired test, since the two values
come from the same resamples. `boundary` is True where the best alpha sits at the edge of
the search range, so the curve has no interior optimum and `alpha_best` is not an
estimate. `weak_signal` is True where the scan carries too little signal to locate an
optimum at all. `interpretable` is False where either of the last two fires: **read
`alpha_best` only on rows where it is True.**

Intervals are 95% percentile intervals from a cluster bootstrap over DMS sites, because
replacements at one site share a backbone and are not independent. They are bootstrap
percentiles, not t-intervals.

**What the table shows.** On the 13 interpretable fitness rows the fine-tuned models put
their best weight at -0.24 to -0.13 and the base models at -0.06 to +0.04. Dividing
accessibility out improves the fine-tuned models by 0.015 to 0.064 in rho and the base
models by 0.001 to 0.007. The 4 receptor-binding rows are left out of those counts:
binding is not fitness, and many replacements raise it, so the sign of a correlation there
is a property of the assay rather than of the model.

**3 of 18 rows are not interpretable** and `alpha_best` must not be quoted for them. 3
carry too little signal at alpha = 0 for a weight to improve on, and 1 put the best alpha
at the edge of the search range, where the curve rises without ever turning over. The two
flags are not exclusive, so those counts may overlap.

**Source.** The shipped values tables of the two DMS sweep figures, which hold the full
201-point sweep: this table takes the row at each curve's best alpha and the row at alpha
= 0, and checks both against the `best_alpha`, `rho_best` and `rho_alpha0` columns those
files carry (it stops if they disagree beyond 1e-9).
