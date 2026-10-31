Figure SN. Reviewer 2 simulation control across four evolutionary arms.

A synthetic 700-amino-acid founder sequence was evolved under a shared site-wise fitness
landscape (beta = 2.0) across four distinct nucleotide substitution regimes: flat JC69
(neutral negative control), human germline, SARS-CoV-2, and influenza A H3N2. ESM-C 600M
was scored on the founder background prior to fine-tuning and after fine-tuning on 5,000
evolved sequences from each arm.

The flat/JC69 regime implements the Jukes-Cantor 1969 neutral substitution model, where
all twelve nucleotide substitution rates between distinct bases are exactly equal (each
proposing at a uniform 8.33% rate, 1/12). Because all base changes are equally probable,
the flat/JC69 arm lacks any directional mutation bias, transition/transversion skew, or
GC-content drive. Sequences in this arm evolve under selection on the shared fitness
landscape in the complete absence of mutational directionality. The flat/JC69 fine-tune
(Flat FT) therefore provides a strict negative control: because no directional bias exists
in its training data, any detected correlation with biological mutation spectra measures
the benchmark's false-positive rate.

(A) Empirical substitution spectra across 12 nucleotide mutation classes. Each biological
regime exhibits its characteristic evolutionary transitions (e.g. C->T transitions in
SARS-CoV-2 and A->G transitions in H3N2). Note that these are the substitutions selection
*retained*, not the rates at which they were proposed, and the two differ: the flat/JC69
arm proposes all twelve classes at a uniform 8.33% by construction, but its retained
spectrum is transition-enriched (53.9% transitions against the 33.3% expected from a flat
proposal over the founder's codons). This is the structure of the genetic code acting
through purifying selection, not a property of the proposal. Enumerating every
single-nucleotide change from the founder's codons, transitions are 52.4% of the
synonymous changes but only 28.8% of the nonsynonymous ones, because a transition at a
degenerate third position is usually silent. Purifying selection therefore removes
transversions preferentially, and retaining only the synonymous changes gives a 52.4%
transition share against the 53.9% observed. This mechanism is long established: Xia,
Hafner and Sudman (1996) On transition bias in mitochondrial genes of pocket gophers,
Journal of Molecular Evolution 43:32-40, doi:10.1007/BF02352297, showed that at twofold
degenerate sites, where a transition is synonymous and a transversion is not, purifying
selection rather than mutation accounts for most of the observed transition bias. The
separate route, in which transitions are milder when they *are* nonsynonymous, contributes
little here (mean ground-truth fitness 0.0264 for nonsynonymous transitions against 0.0210
for transversions) and is in any case disputed: Stoltzfus and Norris (2016, Mol Biol Evol
33:595-602) found no significant fitness difference between nonsynonymous transitions and
transversions, while Lyons and Lauring (2017, Mol Biol Evol 34:3205-3215) found selection
to be a major contributor. Nothing in this panel rests on that disputed route. The dashed
line marks the flat proposal rate, not an expectation for the plotted bars. The flat arm
remains a valid negative control: it carries no directional mutation bias, and the
structure visible in its bars is code-and-selection structure that all four arms share.

(B) Cross-correlation between model predictions and mutation spectra across the full
accessibility matrix (N = 13,300). Spearman rank correlation across all single-amino-acid
replacements is elevated for all models (rho ~ 0.25 to 0.50). This high correlation does
not reflect spectrum-specific learning, but is driven primarily by the shared geometry of
the genetic code and codon distance (1-step vs 2-step vs 3-step mutations) that globally
separates accessible from inaccessible replacements in both PLM probabilities and
mutational proposal rates. Crucially, the neutral Flat FT control also correlates at rho ~
0.25 to 0.31 across all spectra, proving that codon distance structure alone creates a
spurious baseline correlation across all replacements regardless of whether directional
bias was learned.

(C) Ground-truth fitness estimation across mutational accessibility weight alpha.
Combining PLM log-probabilities with mutational accessibility log P_mut reveals that
subtracting proposal bias (alpha < 0) systematically improves fitness landscape recovery
for models fine-tuned on biased evolutionary data. The untuned base model is optimal at
alpha = 0.

(D) Cross-correlation restricted to the N = 1 nucleotide step band (N = 4,174).
Conditioning on single-nucleotide changes controls for codon distance, eliminating the
large baseline correlation inflation driven by genetic code architecture. Consequently,
overall correlation magnitudes decrease relative to the full matrix (rho ~ 0.00 to 0.16).
Rather than running counter to the hypothesis, this lower correlation directly
demonstrates that the apparent signal in (B) was confounded: with codon distance held
constant, the neutral Flat/JC69 control drops to near-zero correlation across all
biological spectra (rho = -0.007 to +0.044), confirming zero false-positive bias.
Simultaneously, genuine learned proposal bias is unmasked in the biological arms: each
fine-tuned model specifically and uniquely peaks on its generating mutation spectrum
(SARS-CoV-2 FT rho = +0.162 on SARS-CoV-2 vs +0.049 on H3N2; H3N2 FT rho = +0.145 on H3N2
vs +0.075 on Flat; Human FT rho = +0.109 on Human vs +0.004 on Flat). Boxed cells
highlight the matching generating arm for each fine-tuned model.

Note that the untuned base model carries a *higher* absolute correlation than any
fine-tune on three of the four spectra (rho = +0.146, +0.189, +0.224 on Flat, Human and
SARS-CoV-2). This is not a failure of fine-tuning and it is not an artefact of codon
distance, which is already held fixed here: it is a row effect. A model trained on natural
sequence has absorbed the average mutational accessibility of life, so the untuned base
scores highly against every spectrum without being specific to any of them, for the same
reason an empirical substitution matrix cannot serve as a biochemistry control.
Fine-tuning on 5,000 sequences from a single arm replaces that broad prior with
site-specific fitness structure (Spearman rho with the true landscape rises from +0.17 for
the base model to +0.50 to +0.60 for the fine-tunes), trading generic accessibility for a
smaller arm-specific signal. Two checks confirm the diagnosis: holding the true fitness
landscape fixed by partial correlation leaves the base row essentially unchanged (+0.224
to +0.223 on SARS-CoV-2), and holding the number of single-nucleotide paths to the target
codon fixed also leaves the base model ahead. Absolute correlation in (D) therefore
measures how much generic accessibility a model carries, not what it learned; panel (E)
separates the two.

(E) The model x spectrum interaction in the N = 1 band, obtained by removing the row and
column means from the fine-tuned block of (D). Row means differ because models carry
different amounts of generic accessibility, and column means differ because the spectra
retain different amounts of structure once codon distance is fixed; neither bears on
specificity. What remains is whether each fine-tune favours the spectrum it was trained
on. It does: every fine-tune ranks its own generating arm first of four (Flat +0.041,
Human +0.025, SARS-CoV-2 +0.028, H3N2 +0.045; ringed in the panel). Enumerating all 4! =
24 one-to-one matchings of fine-tunes to arms, the true matching gives the largest matched
diagonal sum (+0.140) of all 24, an exact p = 0.042. With four arms, 1/24 = 0.042 is the
smallest attainable p-value, so this is the strongest statement the design supports; it is
not evidence of a large effect, and the interaction terms themselves are small (0.025 to
0.045).

One caveat on the flat arm. Within the N = 1 band the flat/JC69 accessibility takes only
13 distinct values spanning exactly threefold, and they encode nothing but how many
single-nucleotide paths reach the target codon: the flat arm has no direction to learn.
Flat FT matching its own arm in (E) therefore means it learned path multiplicity, not a
directional bias, and is not a false positive. The negative-control claim rests on the raw
values in (D), where Flat FT sits between -0.007 and +0.044 against every spectrum, and
not on its diagonal in (E).
