**Table SX. The lineage-defining replacements against the corpus the model was fine-tuned
on.** The corpus holds 8,099 sequences collected up to 2015-02-27, and the question is how
much of the new lineage it already contained. *Sequences with it* counts corpus sequences
carrying the **new** amino acid of that replacement; *Share of training data* is that
count over the whole corpus, so every row sits on the same denominator. A sequence
truncated past the site, or gapped at it, cannot carry the residue but is still in the
denominator, so at a site not every sequence reaches the share is a floor: HA1:K2N, near
the signal peptide boundary, is read in only 6,757 of the 8,099 sequences, giving 372 /
8,099 = 4.6% quoted here against 372 / 6,757 = 5.5% among those readable; every other site
is read in at least 8,097. The next two columns repeat count and share within the H3
sequences, whatever the host, and *In the human subset* repeats them again within the
human H3N2 sequences nested inside those, which is the narrowest group and the only one
whose share reads as a circulating frequency; the training data as a whole spans every
subtype. Residues are read at the ancestral root's own precursor coordinates, each corpus
sequence having been aligned pairwise to the translated root, and every ancestral residue
was confirmed on that backbone before counting. **All 9 of the new amino acids are already
present in the training data**, 9 of them within the H3 sequences and 8 within the human
H3N2 subset. HA1:T328A is present in the H3 sequences but in none of the human H3N2 ones,
so it was read by the model in a H3 background without ever having circulated in the human
population the lineage emerged from. At HA1:S144N, HA1:I160K, HA2:S49N the new residue is
not merely present but is the **majority** state of the human H3N2 subset, so at those
sites the lineage returns to a historically common residue rather than reaching a novel
one. Read it beside the corpus split control, which shows the fine-tuned advantage holding
on replacements absent from the corpus as well as on those present in it.
