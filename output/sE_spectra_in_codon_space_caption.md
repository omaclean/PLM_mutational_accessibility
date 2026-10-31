**Figure SE. Mutation spectra compared in codon space.** Each nucleotide spectrum is
pushed through the standard genetic code into a 64 x 20 table giving the chance that a
codon changes into each amino acid, after scaling every spectrum to the same total
mutation rate so that only the shape of the spectrum counts. The cell at row *i*, column
*j* is the percent of variance in spectrum *i*'s table that spectrum *j*'s table accounts
for, 100 x (1 - var(residual) / var(target)). Before being used as a predictor every table
is averaged over the codons of each amino acid and rebuilt to 64 rows, which removes
everything it knew about which codon is in use. The scale is centred on zero because zero
is the only interpretable point on it: a predictor scoring zero is exactly as good as the
target's own average, and the negative cells are predictors that are worse than that. The
diagonal is each spectrum against itself after the same compression, so it measures how
much of that spectrum's codon-space structure survives being reduced to amino acids; for
Influenza H3N2 it is 84.8%. The manuscript's two published numbers are both cells of this
matrix: the Influenza H3N2 diagonal, and K80 (kappa=2) predicting Influenza H3N2 at 43.6%.
Without the compression step that second comparison is 51.0%, a different quantity kept in
the values table alongside. Expressivity alone does not explain that gap: SARS-CoV-2 has
the same 12 free nucleotide parameters as Influenza H3N2, against 1 for K80 (kappa=2), yet
scores -60.8% against Influenza H3N2's table — far worse than the much simpler matrix the
manuscript used. What a matrix can express is therefore not what decides whether it can
stand in for this one. Spectra are ordered by how well they correlate with Influenza H3N2,
which is the only reason the matrix is not symmetric in appearance; it is not symmetric in
value either, because retained variance is scaled by the target's own variance and so
depends on which of the pair is being predicted.

(**B**) The same comparisons asked as a ranking: if one matrix had to stand in for a
spectrum nobody had measured, which should it be? Each spectrum is scored against the
other 6 as a mean Pearson correlation; Influenza H1N1 is counted with its subtype rather
than separately, so a pair of close relatives cannot prop each other up. Each bar is
labelled with the spectrum's free nucleotide rate parameters, which is how much shape it
can express at all. E. coli ranks first at 0.812. Influenza H3N2, the spectrum this paper
is about, comes 7 of 7 on correlation (0.494), below matrices with far fewer free
parameters: JC69 (flat) has 0 free nucleotide parameters against Influenza H3N2's 12 and
still ranks 5. A spectrum fitted to one virus is a poor general stand-in, and being able
to express more is not what makes a matrix transferable. Bars are grey except the focus;
these are means over a fixed set of 6 comparisons, not estimates with a sampling
distribution, so no interval is drawn.
