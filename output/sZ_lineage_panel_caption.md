**Table SZ. The 13 influenza A/H3N2 haemagglutinin (HA) lineages, their sampling periods
and their variation.** Each lineage is a set of sequences assigned to one
A/Darwin/6/2021-referenced clade root; the sets are disjoint and hold 86,661 sequences in
all. *Sequences* is the number assigned. 5,404 of them carry no collection date, only a
year or a month, or a date that pre-dates their own lineage; these are retained in every
replacement, origin and frequency count and are left out only of the collection date
columns. *First* and *Last* are the extreme collection dates and *Middle 98%* is the 1st
to the 99th percentile of them. **The percentile span is the one to read.** Eight records
carried a collection date a decade or more before the lineage they belong to — four of
them in 1905, which is what a year becomes when it is read as a spreadsheet serial number
— and are excluded as implausible, since the corpus the model was pre-trained on ends in
2015 and nothing here can pre-date 2016. The percentile span is quoted all the same,
because it cannot be moved by one bad record that slips through. *Sites differing* counts
the sites of the aligned coding region at which any sequence in the lineage differs from
its root. *Replacements observed* counts the distinct amino-acid replacements seen, which
is the unit every model is scored on, so it is also each lineage's weight in the sweeps.
*Homoplasies* and *Multiple origins* ask how many of those replacements arose more than
once. An *origin* is a branch of the segment 4 UShER tree whose parent codon encodes a
different residue and whose child codon encodes the replacement's; synonymous changes, and
codons holding a stop or an ambiguous base, are not origins. Two origins therefore always
sit on two different branches, neither ancestral to the other with the residue unbroken. A
replacement with two or more origins is a *homoplasy*: 2,966 of the 5,982 replacements the
tree places, 50% overall and 22% to 66% across lineages. Two processes produce them and
both are counted. In 94% of cases the origins are the same nucleotide change in different
parts of the tree, so their independence rests on the topology; in the remaining 6% the
same residue is reached by two or more different codon changes, which cannot be one event
whatever the topology. **Multiple origins rises with sampling depth** (Spearman 0.99
against *Sequences* across the 13 lineages): every extra sequence is another chance to
catch a recurrence, so the column compares sampling effort as much as it compares biology,
and the counts are a floor on recurrence, never an estimate of its rate. Two gaps shrink
the denominator and are reported rather than absorbed: the UShER reference is not an H3,
so 31 of the 566 positions (1-26, the signal peptide and the first ten residues of mature
HA1, and 562-566) have no reference codon and no replacement there can be placed; and 51
further replacements are observed in a sequence the tree does not hold. Neither is counted
as a single origin. *Top 5% freq threshold* is the sequence frequency defining the 5% most
prevalent circulating replacements in each lineage (the positive class scored in Figure 3C
and Supplementary Figure B), ranging from 0.24% to 6.62%. Lineage size spans a 34-fold
range, from 776 (J.2.4) to 26,703 (J.2); means taken over lineages therefore weight these
sets equally despite that. **The combined row holds three different kinds of number and
the asterisk separates them.** *Sequences* is a sum, which is right because the 13 sets
are disjoint, and *First* and *Last* are the extremes. *Sites differing* and *Replacements
observed* are counted once over the panel rather than summed, because the same replacement
is reached in several lineages: 479 distinct sites and 1,639 distinct replacements,
against sums of 3,522 and 6,392. *Homoplasies* is a sum, and its unit is the
lineage-replacement pair, since one replacement recurring in several lineages is that many
separate recurrences. *Middle 98%* runs from the earliest percentile start to the latest
percentile end, a combined range, so it is not starred. The two starred cells are means of
the 13 rows rather than figures pooled over the sequences: *Multiple origins* is the mean
of the per-lineage percentages, 42%, where pooling them gives 50% — pooling weights a
lineage by its size and the deep lineages are the recurrent ones, so the pooled figure is
always the larger and the two must not be quoted interchangeably. Dates are collection
dates held in the sequence database. No per-sequence data is shown or shipped.
