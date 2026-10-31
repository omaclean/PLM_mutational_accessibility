**Figure 5. Does a model that was never shown codons tell synonymous codons apart?** The
models read amino acids, not codons. Serine has six codons, and which one a site uses
decides whether a change to asparagine needs one nucleotide change or two or three. If a
model had picked up nothing about codons, where the reference codon sits should not change
how it scores the replacement.

**(A) What one number means.** The amino-acid change is held fixed, here S→N in the
reference sequence, so chemistry, substitution matrices and the identity of the new
residue are all constant and only the reference codon differs. Each dot is one site,
placed by the probability the fine-tuned model gives the replacement there. Sites are
split by whether the reference codon reaches the new amino acid in one nucleotide change
(23 sites) or in two or three (24 sites). The statistic is the AUROC: the share of
easy-site/hard-site pairs in which the easy site scores higher. For this replacement it is
0.871.

**(B) How much data this rests on.** 929 easy and 1,079 hard site evaluations across the
reference sequence, summed over the 54 ordered amino-acid pairs. A pair is scored only
where both groups are non-empty.

**(C) The same number pooled, against its null.** Pooling over the 54 scored amino-acid
pairs gives 0.608 (95% interval 0.572 to 0.643) for the fine-tuned model and 0.509 (0.482
to 0.539) for the base model; the paired difference is 0.099 (0.052 to 0.137). The
histograms are the null: 1,000 shuffles that swap reference codons between positions
carrying the same amino acid, leaving everything else as it is. One shuffle gives one
pooled AUROC, and the same shuffles are used for both models. The fine-tuned model falls
outside its null (0.465 to 0.536, p = 0.001); the base model falls inside its own (0.471
to 0.527, p = 0.532).

Intervals resample sites: evaluated on a single reference sequence to prevent
cross-lineage pseudo-replication. The figure recomputes the null from the scores and
checks percentiles against the derivation; the point estimates agree with the scores to
0e+00.
