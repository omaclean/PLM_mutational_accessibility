**Figure SD. Removing amino-acid chemistry from the model-accessibility link, three
ways.** One lineage (J.2_int) from the set h3n2_5lineage; one column per model. The
horizontal axis is the minimum number of nucleotide changes a replacement needs, with
"any" pooling all of them. Vertical lines are 95% percentile bootstrap intervals that
resample **sites** -- whole positions, carrying all of their replacements -- because
replacements at one site share a backbone and are not independent. All uncertainty shown
is over sites.

**Top row (panels A and B): rank-based, in correlation units.** The coloured circles
repeat Figure 2D, the plain Spearman correlation between the model's probability of a
replacement and that replacement's mutational accessibility. The two pink series control
for chemistry, in two different ways, and the point of showing both is that they can
disagree.

*Partial Spearman (squares).* Take three quantities over the same replacements: the
model's probability, the accessibility, and the Grantham score. Replace each by its rank.
Regress the ranked model probability on the ranked Grantham score and keep the residual --
the part of the model's ranking that chemistry cannot linearly account for. Do the same
for ranked accessibility. The partial Spearman r is the ordinary correlation between those
two residuals; equivalently, and as computed here, it is (r_ma - r_mc * r_ac) / sqrt((1 -
r_mc^2)(1 - r_ac^2)) from the three pairwise Spearman correlations between model (m),
accessibility (a) and control (c). The two routes are algebraically identical.

*Grantham-matched Spearman (hollow triangles).* The known weakness of a partial
correlation is that it removes only the *monotone* part of the control's effect, so any
curved dependence on chemistry survives into the residual and is still counted as model
signal. This series avoids that by conditioning instead of regressing: replacements are
sorted into 5 bins of Grantham distance, the correlation is computed within each bin among
replacements of similar chemistry only, and the bin correlations are pooled on Fisher's z
scale weighted by n - 3. Nothing is subtracted, so nothing non-monotone can leak through.
It is on the same scale as the other two and can be read directly against them.

**Bottom row (panels C and D): likelihood-based, in exponent units.** A rank correlation
counts every replacement equally, so a swap between two the model has all but ruled out
weighs as much as a swap between its top two -- and the model's probabilities span eight
orders of magnitude at a site. This row removes that objection by fitting the model's own
output as what it is, a softmax over amino acids at a site: q(site, aa) proportional to
exp(beta * ln accessibility + f(Grantham)). Chemistry enters as a restricted cubic spline,
so its effect is not forced to be monotone or even smooth in any particular direction. The
fit is within each site, so the site intercepts cancel and only the ordering and spacing
of replacements at a site informs it; each codon-distance group is normalised within
itself, so every site contributes equal weight to its own group and the three-change
estimate does not hang on a handful of high-probability sites. The "any" fit carries no
codon-distance offsets, because across all nineteen replacements the step between the
groups *is* most of the variation in log accessibility, and offsets plus a slope would be
estimating the same quantity.

*What beta is, and what it is not.* Beta is a power-law exponent, d ln q / d ln A: the
model behaves as though it had multiplied its probabilities by accessibility raised to
beta. It is therefore in the same units as the exponent the alpha sweep tunes from
outside, and the comparison is valid in either log base, because an exponent is invariant
to the base as long as one base is used on both sides. It is **not the same quantity** as
that alpha and should not be quoted as if it were. The sweep's alpha is an external weight
chosen to maximise agreement with a measured phenotype; beta is how much of that behaviour
the model's own output already shows, with no phenotype involved. A test generates data
from a known exponent and confirms the fit recovers it, so the scale is pinned rather than
assumed.

**What the three estimators agree on.** For "Fine-tuned on influenza HA" the chemistry
control changes almost nothing, by any route: with one nucleotide change the correlation
is 0.29 [0.26, 0.32] alone and 0.29 [0.26, 0.32] with Grantham's rank removed, and the
exponent is 0.33 [0.30, 0.37] alone against 0.33 [0.29, 0.37] with the spline. Its link to
accessibility is not chemistry wearing another name. At 3 nucleotide changes the control
does not merely fail to remove the signal, it uncovers more of it: the correlation rises
from 0.14 [0.09, 0.18] alone to 0.18 [0.14, 0.22] with Grantham's rank removed, and the
matched estimator agrees (0.17 [0.12, 0.21]). Chemistry was masking part of this model's
preference for accessible replacements rather than supplying it. For "Base model (ESM-2)"
at two nucleotide changes all three estimators keep the residual signal clear of zero.

**Where they disagree, and why that is in the figure.** Two cells do not reproduce across
estimators, and both involve the untuned model. With one nucleotide change its
chemistry-controlled signal is above zero on both rank estimators (0.05 [0.02, 0.08]
partial, 0.04 [0.01, 0.07] matched) but not distinguishable from zero once replacements
are weighted by the probability the model assigns them (0.01 [-0.02, 0.05]). With 3
nucleotide changes the two rows take opposite signs: the rank estimators sit at or just
above zero (0.03 [-0.02, 0.08] partial) while the exponent is negative (-0.10 [-0.13,
-0.07]), meaning that among the replacements it considers at all plausible, the untuned
model mildly *disprefers* the more accessible ones. Both readings are defensible and they
answer slightly different questions -- all replacements equally, or weighted by the
model's own confidence -- so the honest report is that the untuned model's residual link
to accessibility is weak and depends on the weighting, while the fine-tuned model's does
not depend on it at all. The claim this figure supports is the second one.

The control score is Grantham chemical similarity, a distance built from composition,
polarity and volume, sign flipped; it contains no model and no codon information. An
earlier version also plotted BLOSUM62 (the BLOcks SUbstitution Matrix) beside it. That has
been removed: BLOSUM62 summarises substitutions observed in nature, which already reflect
which changes mutation makes easy, so holding it constant removes genuine mutational
signal along with the chemistry, and the residual cannot be read as "chemistry held
constant".

**Why one lineage and not the set.** Earlier versions reported the mean over all 5
lineages. Those lineages are scored at the same sites and are pseudo-replicates for this
measurement: across the set the per-lineage estimates differ by a standard deviation of
0.001 to 0.013, against a site-bootstrap interval 0.04 to 0.11 wide, so the
between-lineage spread is three to ninety-six times narrower than the uncertainty already
shown. Averaging them moved no estimate by more than about 0.015 and added nothing to the
interval, while reporting "5 lineages" invited the reader to hear replication that the
shared sites do not provide. The lineage is chosen by a stated rule (provenance
established rather than unverified, then the most scored replacements, then sorted
identifier) rather than by hand. Every interval here is a statement about these sites, not
about a new lineage or a new protein.
