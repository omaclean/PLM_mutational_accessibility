**Figure SC. How well two untuned protein language models recover a masked amino acid.**
Protein language models are trained to fill in a masked position in a protein sequence.
For each of 41 proteins (25,541 positions in all) we masked one amino acid at a time and
asked ESM-2 650M base and ESM-C 600M base what it expected there. Each dot is the
probability the model put on the true amino acid, averaged over every position of the
protein; 100% would mean the model always knew. Proteins are grouped (Influenza viruses,
Coronaviruses, Other viruses, Eukaryotes, Bacteria, Phages) and ordered within a group by
the mean of the two models, so neither model sets the order. Neither model has been
fine-tuned. Sequences are the reviewed UniProt entries listed in the sequence table, with
their accessions.

Two floors are marked, and they are not the same quantity. The dashed line at 5% is one
amino acid in twenty: what a model scores knowing nothing at all, and the same number for
every protein. The grey tick on each row is that protein's own composition baseline, the
score for guessing amino acids in the proportions that protein contains while ignoring the
sequence around them; it varies by protein (5.7% to 7.8% here), so it is drawn per row
rather than averaged into a line that would imply a constant.

A high score means the sequence is easy to predict from its context: often because it is
conserved, and in some cases because near-identical copies sit in the training data. A low
score is not proof that a model "does not know" the protein. Positions are not
independent, so no interval is drawn.
