**Figure SJ. Agreement between deep mutational scans of one protein, against how far apart
the scanned strains are.** Deep mutational scanning (DMS) measures the effect of nearly
every single amino-acid change in a protein. Each point is a pair of DMS datasets of the
same protein measured on two strains (26 pairs: Spike RBD Yeast Display (20 pairs); Full
Spike Pseudovirus (2 pairs); HA Entry vs Growth Preference (1 pair); HA Cross-Subtype
Entry (3 pairs)). Both panels show the Spearman rank correlation of the measured effects
of the mutations the two scans share. (A) against the number of amino-acid substitutions
between the two parental strains; (B) against that number as a percentage of the protein
(log scales; the same points, re-spaced). Marker shape is the viral species and colour is
the measured trait, as in the legend; marker faces are semi-transparent so that
overlapping points remain readable. The assay each pair comes from (the systems listed
above) is not drawn on the panels; it is recorded for every point in the accompanying
values table. No model is plotted here, and the colours are drawn from the house
categorical palette rather than the blue and green that mean base and fine-tuned elsewhere
in this set. Points at the same x are offset by up to 7% so they do not overprint. Rank
agreement ranges from 0.39 to 0.96.

What is not shown. For 2 pairs, more than 5% of variants sit at the assay floor in both
scans (rank agreement 0.53, 0.55 as plotted, falling to 0.34, 0.34 with those variants
dropped). The pile-up inflates the plotted value and dropping the floor variants removes
the most deleterious mutations and over-corrects, so the true value lies between. That
range is not drawn: it is a sensitivity to a filtering choice, not sampling uncertainty,
and an interval on two points would be read as both. It is recorded per pair in the values
table. HA Cross-Subtype Entry pairs are joined by a structural rather than a sequence
alignment and, per the analysis notes, cross cell line as well as subtype, so they are not
like-for-like with the other systems. Binding traits measure affinity for a receptor, not
fitness, so their agreement is not a fitness agreement. Rank agreement is the only measure
plotted. The source table also holds the Pearson correlation of the same effects and the
mean per-site Jensen-Shannon divergence between the two scans' amino-acid preference
profiles; both tell the same story as the rank correlation and are left out to keep the
figure to one question. The synonymous-variant filter is not the same across systems (3
filter modes, recorded per pair in the source table), which moves the coefficients
slightly. Substitution counts exclude insertions and deletions, which are recorded
separately (5 pairs have at least one). The agreement of each pair is computed offline
from the raw scan tables; the figure shows no per-site or regional breakdown, and no mark
on either panel carries a vertical extent -- there are no uncertainty intervals anywhere
in it.
