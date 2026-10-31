**Figure SH. Does the model's coupling signal recover real 3D contacts, and where?**
Substituting each residue and reading how the model's predictions move at every other
position gives a coupling score for each pair of positions (the categorical Jacobian of
Zhang and Ovchinnikov 2024, on the zero-sum scale, average-product corrected and
symmetrised). Pairs at least 6 positions apart are ranked by that score and compared with
real contacts, a C-beta distance under 8 Angstrom in a reference structure (minimum over
the three protomers). AUROC is the chance that a true contact outranks a non-contact: 0.5
is chance. Two reference structures are shown because the answer differs between them:
6WXB resolves 485 residues; 7ZJ7 resolves 486 residues.

Panels A and B: AUROC in each structural stratum for Base model (ESM-2) and Fine-tuned on
influenza HA; the grey segment joins them. Strata are by burial (core and surface are the
lower and upper thirds of neighbour counts in that structure) and by domain (head versus
stalk, using the boundary the pipeline hard-codes). Panel C: the change on fine-tuning
with two 95% intervals. The thin interval resamples pairs, as the original analysis did.
The wide grey one resamples residues, which is the right unit because one residue sits in
hundreds of pairs and its pairs share its fate; it is new, and on average 2.2 times as
wide. Panel D: the same per-residue gain at the published antigenic residues against the
rest of the same domain, on the zero-sum gauge. The 5 named sites are pooled into one
class, so this is a single comparison rather than 5 uncorrected looks. Each bar is a 95%
bootstrap interval over residues (2,000 resamples) for that group's own mean gain; the
difference between the groups carries its own 95% bootstrap interval and a p-value from
permuting the antigenic and rest labels (10,000 shuffles), because two overlapping
intervals are not a test of a difference. Panel E: AUROC for each residue's own partners,
as a 25-residue moving average along the sequence, blank where fewer than 8 residues are
resolved. Panel F: the same difference. Shading alternates by domain; ticks mark the
clade-defining sites.

6WXB: over all long-range pairs the AUROC goes from 0.715 to 0.715 (-0.000); of 11 strata
the fine-tuned model is clearly ahead in 2 and clearly behind in 3 by the pair interval,
but ahead in 0 and behind in 1 once residues are resampled; 7ZJ7: over all long-range
pairs the AUROC goes from 0.736 to 0.753 (+0.017); of 11 strata the fine-tuned model is
clearly ahead in 4 and clearly behind in 0 by the pair interval, but ahead in 0 and behind
in 0 once residues are resampled. 4 strata hold fewer than 30 true contacts and should not
be read individually. The head and stalk boundary is hard-coded and known to be unstable,
so the head-stalk stratum carries that caveat. Fine-tuning is not specialised to the
antigenic sites: the gain there is about the gain elsewhere in the same domain — 6WXB:
+0.032 at 126 antigenic residues against +0.035 at 185 others (difference -0.003, 95%
bootstrap [-0.033, +0.030], permutation p = 0.85); 7ZJ7: +0.040 at 125 antigenic residues
against +0.034 at 183 others (difference +0.006, 95% bootstrap [-0.024, +0.039],
permutation p = 0.71). Antigenic positions are joined on the mature-chain numbering only;
neighbouring residues are not independent, so these intervals are if anything too narrow.
Positions are mature numbering throughout; the first residue of the HA2 region is position
330.
