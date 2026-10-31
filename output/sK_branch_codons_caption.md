**Figure SK. Nucleotide changes per codon on individual internal branches of the tree.**
Each branch of the UShER tree (a tree of sequences, built by placing each one on its
nearest relative) carries the nucleotide changes that separate a node from its parent.
Only branches with fewer than 10 nucleotide changes are used, because long branches are
mostly misplaced or poor-quality sequences, and only internal branches unless the table
says otherwise, because tips carry the most sequencing error. For each codon changed on
such a branch we count how many of its three positions differ from the parent codon.
**(A)** Every count is raw; none is averaged. Each row sorts amino-acid changes by the
fewest nucleotide changes that would have produced the new amino acid from the codon it
started from. *Individual internal branches* is every amino-acid change on every scored
internal branch, taken one branch at a time. The 50,094 branches hold 50,872 synonymous
and 30,028 amino-acid-changing codon changes; 277 of the amino-acid-changing ones (0.9%)
changed 2 or 3 positions. In 131 of those the new amino acid was reachable from the parent
codon by fewer changes. That comparison uses the parent codon actually seen on the tree.
It is not the minimum over all codons of the old amino acid, which often differs only
because the parent carried a synonymous codon far from the target. **(B)** The chart plots
that share against branch length, binned 1 / 2 / 3 / 4+, against the share expected if the
branch's changes fell at random. A rise with branch length on its own means nothing: a
branch carrying one nucleotide change cannot move two positions of a codon, and the more
changes a branch carries, the likelier two land in one codon. The chance line removes both
effects. It places each branch's changes at random, without replacement, over the 535
scored codons, and converts codons to amino-acid changes with P(amino acid changes | 1
position moved) = 0.37, taken from these data, and P(... | 2 or 3 positions moved) = 0.99
and 0.99, taken from the genetic code (the data's own 2-position split, 264 of 279,
agrees). Observed exceeds chance by 4.0x in the bin of 2, 3.8x in the bin of 3, 3.5x in
the bin of 4+: the excess is largest on the shortest branches, which is the opposite of
what the mechanical effect would produce. Bars are 95% Wilson intervals; a bin with fewer
than 10 amino-acid changes is not drawn. The table beside the chart gives the branches and
amino-acid changes behind each point. **(C)** Every tip of the tree by collection year,
1968 to 2026, on a log scale: the earliest years hold tens of sequences against tens of
thousands recently, and a linear axis hides them. The last ten years carry 77% of the
224,731 tips, so the scored branches sit overwhelmingly in recent diversity. The panel
beside it counts the tree and the branches drawn from it. The table's second row pools the
amino-acid differences between KY348531 (a dated sequence in the tree) and each of the
44,965 newest internal nodes (internal nodes whose children are all tips): 3,563,404
differences, of which 82% could be reached by one nucleotide change from the baseline
codon, 15% needed two and 3% needed three. Over a long path such changes are made in steps
through intermediate codons, which a single branch cannot do, and differences are counted
per node, so changes shared by most nodes are counted for each. The baseline is an
observed sequence, not a reconstruction. The walk's own start node is not used for it: the
tree reads sites that were never sequenced as the reference base, so near the root the
reconstruction is largely the reference record, and that record is a different subtype.
Comparing against it measures the distance between subtypes rather than change within one.
The null row (A) is what the rows around it are read against: from each of the 566 codons
of the J.2 root, the fewest nucleotide changes that reach each of the other 19 amino
acids. The lookup behind it is 64 x 20, not 20 x 20 or 64 x 64: the source is the codon
actually at the site, one of 64, and the target is an amino acid, taking its nearest
codon. A 20 x 20 version would have to pick one codon per starting amino acid and a 64 x
64 one would count synonymous targets, and both give a different answer. It is 10,754
codon-to-amino-acid pairs, 31% of which are one nucleotide change away, 51% two and 18%
three. It uses the genetic code and that sequence's own codons: no tree, no branches and
no counts, so it is what the columns would look like if an amino-acid change were equally
likely to go anywhere. It is not code-only -- every codon contributes its own 19
distances, so the row carries the sequence's codon composition, which is why it is taken
from the same root the model rows score rather than from the tree's reference record. All
table rows above the second hard rule are raw counts; none is averaged. The last two rows
(A), below the second hard rule, are a different kind of object and are not counts of
anything observed: they ask where a protein language model puts its probability. For each
of the 566 sites of the J.2 root the residue is masked, the model's probabilities for the
19 other amino acids are renormalised to sum to one, and each amino acid's weight is
assigned to the fewest nucleotide changes that reach it from the J.2 codon actually at
that site. Every site therefore contributes exactly one expected amino-acid change, so the
total is the number of sites and a site the model is confident about cannot drown out the
rest. Read against a flat distribution over the same 19-way choices, which would give
31.0% / 51.0% / 18.0% (the genetic-code null for these sites, close to the row above
because the split depends on the code rather than the sequence), base model (esm-2)
already concentrates on one-nucleotide neighbours at 64.4% and ha80 fine-tune at 87.7%,
with three-nucleotide changes falling from 18.0% to 4.8% and 0.6%. Both models thus carry
a mutational bias neither was given a nucleotide sequence to learn, and the fine-tuned one
carries much more of it; but both still fall short of the 99.5% of real single-branch
changes that are one nucleotide away, so the models interpolate between the genetic code
and what a single branch can actually do. Both rows use the masked marginals shipped in
the replacement-score table, and the checkpoints behind the two labels are given in the
Methods. The tree's reference record is a different protein, and 18 of its 553 codons have
no residue in the protein studied (alignment gaps). Those columns are left out of every
count, because a gap read as a base looks like a run of multi-position changes. Codon
numbers count along the reference record, not alignment columns, so gaps elsewhere in the
alignment cannot shift them. The tree reads sites that were never sequenced as the
reference base unless it was built with faToVcf -includeNoAltN; every number here inherits
that convention.
