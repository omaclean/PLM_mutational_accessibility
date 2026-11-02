**Figure 2C. Correlation with mutational accessibility across fine-tuning epochs.**
Spearman rank correlation between the model's probability for each amino-acid replacement
and its mutational accessibility, for the model as released ("raw", 0.275) and after each
of 10 epochs of fine-tuning. It rises to 0.632 after the first epoch and then stays
between 0.624 and 0.657 (epoch 10: 0.641): almost all of the change happens in the first
epoch.

The values are from a per-epoch fine-tuning run of ESM-2 (650M) on the HA-80 corpus,
following the published methodology (seed 13, lr 5e-5, AdamW) and evaluated on the
reference lineage `J.2_int`, the same lineage panels A, B, and D use. This replaces the
digitised three-lineage mean from the earlier exploration archive.
