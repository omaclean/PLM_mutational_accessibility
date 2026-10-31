**Figure 2A, B. Model probability against mutational accessibility, base model and
fine-tuned model.** Each dot is one amino-acid replacement at one site of the influenza
A/H3N2 haemagglutinin (HA) reference root sequence (10,754 replacements), plotted at the
model's probability for it (horizontal) and its mutational accessibility (vertical), both
on log axes. Colour is `Mut_num`, the fewest nucleotide changes that make the replacement
(3,334, 5,487 and 1,933 replacements need 1, 2 and 3). (A) Base model (ESM-2). (B) The
same replacements scored by the model: fine-tuned on influenza ha.

One lineage only: the root sequence `J.2_int` (`data/roots/h3n2_5lineage/J.2_int.nt.fa`),
which is an inferred ancestral sequence, not an observed one. Its FASTA header carries a
sequence accession, but the sequence itself is inferred, so the accession should not be
read as its source. Its header reads
`EPI2990337|HA|A/District_Of_Columbia/27/2023|EPI_ISL_18862356|J.2`. The other lineages of
the set are scored at the same sites, so they would add near-copies of these points, not
independent ones, and are not drawn.

Under each panel title: the Spearman rank correlation of the model's probability with
mutational accessibility over all replacements (0.275 base, 0.655 fine-tuned), and the
Pearson correlation of the two probabilities (0.054 and 0.045). Pearson is taken on the
probabilities themselves, not their logarithms, so a few large values dominate it and it
says little about the bulk of the points. Mutation probability takes only a few hundred
distinct values within a group, which is why the points fall in horizontal stripes.
