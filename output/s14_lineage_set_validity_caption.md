**Does the lineage set support the conclusions drawn from it?** Four checks on the lineage
sets (5 lineages and 13 lineages) behind the main figures. Lineages here are disjoint sets
of sequences, but they share most of their sites, so a result seen once per lineage is not
seen independently. AUROC is the area under the receiver- operating-characteristic curve:
the chance that a case of interest ranks above one that is not; 0.5 is chance.

(A) With the amino-acid replacement held fixed (for example alanine to valine), does the
model rank the replacement higher at sites where the reference codon is one nucleotide
step away than where it is two or three? Inside one replacement every amino-acid quantity,
including chemistry and substitution matrices, is constant, so only codon information is
left. Values are weighted means of per-replacement AUROCs; bars are 95% intervals from
resampling sites, because a site recurs in every lineage. 5 lineages: fine-tuned 0.609
[0.574, 0.641], base 0.511 [0.484, 0.541]; 13 lineages: fine-tuned 0.610 [0.576, 0.641],
base 0.510 [0.485, 0.542]. The fine-tuned model's interval excludes chance on 2 of 2 sets
and the base model's on 0 of 2.

(B) On the 13 lineages set, the fine-tuned model's AUROC advantage over the base model for
predicting which replacements occur, split by whether each replacement is present in the
fine-tuning corpus. Cells are positions by amino acids; a cell counts as observed if any
lineage observed it. Advantage on replacements absent from the corpus: +0.143 [+0.109,
+0.178] (160 observed); present: +0.181 [+0.162, +0.199] (1555 observed). The absent-group
estimate rests on fewer observed replacements and is the noisier one. Intervals resample
sites.
