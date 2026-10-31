**Figure SF. How strongly each model's score tracks mutational accessibility, spectrum by
spectrum.** Mutational accessibility of a replacement is the probability that mutation
turns the reference codon into any codon of the new amino acid, given a 4x4 table of
nucleotide change rates (a spectrum). Each row is a different spectrum (9 alternatives to
the reference, "Influenza H3N2 (Pauly et al.)"): the experimental spectra of supplementary
figure S5, the non-directional symmetrised version of the reference, the phylogenetic
fourfold GTR spectrum, and theoretical baselines (K80, JC69). Rates are rescaled to the
same mean so pooled values are comparable. The protein, the reference codons and the
replacements are identical in every row, so the nucleotide-step strata are too; only the
accessibility changes. **A star marks the spectrum each series correlates with most
strongly in that panel.**

**Natural phylogenetic spectrum vs experimental rates.** The fourfold GTR condition
estimates substitution rates directly from ancestral sequence reconstructions across the
global phylogenetic tree (>140,000 branches). Because natural sequence evolution causes
codons to transition between amino acids over time, only two codons remain strictly
fourfold degenerate across every branch of the entire tree; thus, rather than filtering
for a static alignment of globally conserved positions (which would discard >95% of
synonymous variation), rates are measured branch by branch. At each reconstructed
parent-child branch, every codon whose ancestral state possesses a fourfold-degenerate
prefix (Ala, Pro, Thr, Val, Gly, Ser, Arg, Leu) provides an opportunity for neutral
substitution at the third nucleotide position. Reversible GTR exchangeabilities are fitted
from the resulting synonymous substitution counts (79,228 substitutions) by enforcing
detailed balance against equilibrium base frequencies. The non-directional condition
averages forward and reverse fluxes of the reference experimental spectrum to enforce
detailed balance with respect to stationary base composition ($Q_\text{sym} = (Q +
\text{diag}(\pi)^{-1} Q^T \text{diag}(\pi))/2$). Both alternative parameterisations
correlate very strongly with each other and with the reference (Pearson r > 0.997),
confirming that the strong model-accessibility correlation is robust whether using
laboratory-measured error rates or natural phylogenetic estimates.

**Dots and whiskers.** A dot is the mean over lineages of the within-lineage Spearman
correlation between the score and accessibility. Whiskers are 95% **percentile bootstrap**
intervals, built as follows. Sites are resampled with replacement -- whole positions,
carrying all of their replacements, and the same positions in every lineage -- 300 times.
Each resample gives one value of the same lineage-averaged statistic, and the whisker runs
from the 2.5th to the 97.5th percentile of those 300 values. Sites are the resampling unit
because the up-to-nineteen replacements at one position share a backbone and are not
independent; resampling rows would treat them as independent and give intervals that are
far too narrow. Lineages are not the resampling unit either: they share their sites, so
the spread between them understates the uncertainty (`LINEAGE_VALIDITY.md` section 4, Q2).
One draw is reused across every spectrum, series and stratum, so differences between rows
are paired.

**The bottom row is the control.** Grantham similarity is not a model and holds no
nucleotide information, so it shows what the genetic code does by itself. It takes a row
of its own rather than repeating on all 10 spectrum rows. Its square is the value under
the reference spectrum, "Influenza H3N2 (Pauly et al.)", with the usual interval; the pale
bar behind it spans the lowest to the highest value the control reaches across all 10
spectra, because that range is wide enough that a single number would misrepresent it. No
star is placed on this row: the stars mark each model's best spectrum, and the control is
not a model.

**Why "Influenza H1N1" scores lowest.** Its matrix is the closest to the reference of any
in the panel by variance -- supplementary figure S5 puts the two at 83-85% of each other's
-- and its accessibility ordering agrees with the reference's more closely than any other
spectrum's here (Spearman 0.82 within one nucleotide step). It still scores lowest of the
measured spectra. The two facts are compatible because this panel does not measure
agreement with the reference spectrum; it measures agreement with the model's score.
Inside one nucleotide step every replacement needs one specific base change, so a
spectrum's ordering there is essentially its ordering of the twelve rates. Ranking the
model's score by which base change each replacement needs gives the model a preference
over those twelve: A>G (mean percentile 0.73) and G>A (0.66) at the top, G>C (0.36) and
T>G (0.37) at the bottom. How closely a spectrum's rate order matches that preference
predicts this panel almost exactly -- Spearman 0.96 across the eight spectra -- and this
one's match is 0.06, against 0.56 for the reference. The two agree on the dominant
transitions and disagree on the minor cells that decide the rest: the reference makes G>A
its third-largest rate, the model's second-most-favoured change, while the other demotes
G>A to sixth and promotes G>C, the model's least-favoured, to third. Swapping that one G>A
rate back recovers 35% of the gap, G>C a further 21% and C>T 17%. This is a property of
the two matrices' minor cells, not a parsing fault: transposing, complementing or
relabelling the bases of the second file all fit the reference worse than reading it as it
stands (`dev/spectrum_swap_anomaly/`).

BLOSUM62 is not drawn: an empirical substitution matrix already contains mutational
accessibility, which makes it a comparator rather than a biochemistry control, and showing
it beside Grantham invites that confusion.

A model that had learned the reference spectrum's direction would be expected to correlate
best with it. Where the control moves with the models, the pattern reflects the genetic
code and not learning (see SYMMETRY_TESTS.md). Only the spectrum varies, and only for one
protein: nothing in the shipped tables scores a second protein with a fine-tuned model.
Conditions in the exploration archive that redrew synonymous codons from codon-usage
tables are not reproduced. Accessibilities equal up to floating-point noise are treated as
ties, which matters for the flat spectrum; the reference agrees with the guide's own
accessibility to 5.7e-04.
