**Figure 4. Measured fitness effects against a language model's score, as the weight on
mutational accessibility changes.** Deep mutational scanning (DMS) builds nearly every
single amino-acid replacement in a protein and measures its effect in the laboratory, so
the measurement does not depend on how easily a replacement arises by mutation. Each panel
is one measured phenotype: MA22 (Influenza H3N2, Yu et al. 2026); Perth09 (Influenza H3N2,
Lee et al. 2018). Every replacement is scored by a protein language model (PLM) run on the
scan's own reference sequence, with the scored site replaced by the mask token so that the
residue actually there cannot inform its own prediction, and combined with accessibility
as log(model probability) + α × log(accessibility), where α is the weight on mutational
accessibility: 0 is the model alone and negative values divide accessibility out.
Accessibility is the probability, under the nucleotide change spectrum measured for that
virus, that the reference codon mutates to a codon for the replacement's amino acid. The
vertical axis is the Spearman rank correlation between that score and the measured effect
over all replacements scored (5,472 to 10,754 per curve, at 469 to 566 sites; 1 = same
ranking, 0 = no relationship).

Colour marks the model's role. Each panel compares a base (untuned) model with the same
architecture fine-tuned on that virus, so absolute correlations are not comparable between
model families. Dots mark each curve's best weight -- the α at which the correlation is
largest in size, so that a panel whose phenotype runs opposite to fitness is read at its
strongest relationship rather than where it is diluted closest to zero -- with its value
in the label. A hollow marker means something weaker: the curve has a real interior
optimum, at least half again as strong as the model alone, but the model alone is below
the 0.05 floor, so the weight is shown without being claimed as a result. Bands are 95%
percentile intervals from a cluster bootstrap over DMS sites (200 resamples), because
replacements at one site share a backbone and are not independent; the band is computed on
every fifth weight and interpolated between. The best weight is chosen on the data it is
scored on, so the height at the dot is optimistic by an unknown amount. In the 4 panels
where both curves have a marked optimum, the base model correlates more strongly than the
fine-tuned model at α = 0 in 2, and moving to the best weight raises the correlation in
size by a median of 0.001 for base models and 0.038 for fine-tuned models (in sample).

What is not shown. No optimum is marked on 0 of 8 curves: 0 because the largest
correlation lies at the end of the swept range and 0 more because the correlation at α = 0
is below 0.05 in size, leaving no signal for the weight to improve. The curves are still
drawn. Where optima are marked, the best weight ranges from −0.06 to −0.03 for base models
and from −0.24 to −0.18 for virus-specific fine-tuned models.
