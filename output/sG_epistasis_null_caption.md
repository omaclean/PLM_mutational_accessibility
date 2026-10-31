**Figure SG. Are the shifts between the clade-defining substitutions larger than what any
substitution does when its background changes?** The model gives a probability for every
residue at every position. Place one substitution on the sequence first and the score of a
second substitution moves. Here the 9 substitutions that define the clade are taken in
every ordered pair (72 pairs). To say how large a shift is, each substitution being scored
is also measured on every background that carries one change at a site outside the clade
set, 10,298 backgrounds per substitution (92,682 draws in all). Those changes are not
clade-defining, so they show what a background change does to that substitution by chance.

The shift is drawn on the zero-sum scale: log probability minus its average over the 20
residues. This is the scale of the categorical Jacobian of Zhang and Ovchinnikov (2024),
and the only one measured on here. The choice does not rest on which scale gives the most
significant pairs. It rests on room to move: a plain change in probability is limited
where the model is already sure, and at the most conserved site, T135K, the model gives
the residue already there a baseline probability of 0.998, leaving a plain change in
probability almost nowhere to go.

Panel A: the shift for every pair, rows the substitution scored and columns the one
already on the sequence; outlined cells have q below 0.05. Panel B: all 72 pairs (lines)
against the pooled null (grey), log counts. Panel C: each substitution against its own
null; grey bands hold 50, 90 and 98% of that null. Positions are mature numbering: HA1
residues keep their position, HA2 residues count from the HA2 start; add 16 for numbering
that includes the signal peptide.

3 of the 72 pairs have a Benjamini-Hochberg q below 0.05 against the pooled null: N158D
then Q173R (-0.73, q = 0.014); K189R then Q173R (+0.54, q = 0.014); S144N then I160K
(+0.61, q = 0.019). The largest shift is N158D then Q173R (-0.73). Of the 72 pairs, 3 have
q below 0.05 against all null draws and 6 against draws close to the pair in sequence
separation and 3D distance. A p-value is the share of null draws at least as extreme (with
a +1 correction), doubled for two sides; q adjusts for the 72 pairs tested. Clade
substitutions can sit on one branch of the tree, so the pairs are not independent
evidence, and the "close" test uses structure information that is not shipped.
