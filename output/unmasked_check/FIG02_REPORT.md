# Figure 2B and 2D under a masked-marginal estimator

Lineage `J.2_int`, guide `h3n2_5lineage`, 10754 replacements per model.
Published panels read the model's probability from a single unmasked forward pass over
the lineage root, so the model sees the residue it is scoring. Here each position is
masked in turn and everything else is held fixed.

## Gates

| gate | result |
| --- | --- |
| published `plm_prob` reproduced for `ESM2_650M_HA80_raw` | max abs diff 6.67e-06 |
| published `plm_prob` reproduced for `ESM2_650M_HA80` | max abs diff 3.53e-06 |
| unmasked arm reproduces the shipped `fig02ab` values | max abs diff 4.65e-07 |
| unmasked arm reproduces the shipped `fig02d` values | max abs diff 4.65e-07 |
| non-model columns unchanged | identical |

## Panel B (and A), correlation of PLM probability with accessibility

| panel | model | Spearman unmasked | Spearman masked | delta | Pearson unmasked | Pearson masked | delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | `ESM2_650M_HA80_raw` | 0.275 | 0.304 | +0.029 | 0.054 | 0.116 | +0.062 |
| B | `ESM2_650M_HA80` | 0.655 | 0.642 | -0.013 | 0.045 | 0.095 | +0.050 |

## Panel D, within-stratum correlation with accessibility

Bootstrap as the panel draws it: 200 site resamples, seed 20261009. `lo`/`hi` are the published panel's interval.

| series | control | stratum | unmasked | masked | delta | published 95% | outside? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ESM2_650M_HA80_raw` | none | all | 0.275 | 0.304 | +0.029 | [0.256, 0.296] | **yes** |
| `ESM2_650M_HA80` | none | all | 0.655 | 0.642 | -0.013 | [0.637, 0.675] | no |
| `ESM2_650M_HA80_raw` | neg_grantham | all | 0.204 | 0.227 | +0.023 | [0.184, 0.224] | **yes** |
| `ESM2_650M_HA80_raw` | blosum62 | all | 0.143 | 0.161 | +0.017 | [0.125, 0.161] | no |
| `ESM2_650M_HA80` | neg_grantham | all | 0.622 | 0.607 | -0.015 | [0.603, 0.643] | no |
| `ESM2_650M_HA80` | blosum62 | all | 0.584 | 0.566 | -0.017 | [0.564, 0.607] | no |
| `ESM2_650M_HA80_raw` | none | 1 | 0.071 | 0.065 | -0.005 | [0.041, 0.098] | no |
| `ESM2_650M_HA80` | none | 1 | 0.292 | 0.257 | -0.035 | [0.261, 0.324] | **yes** |
| `ESM2_650M_HA80_raw` | neg_grantham | 1 | 0.052 | 0.044 | -0.008 | [0.024, 0.077] | no |
| `ESM2_650M_HA80_raw` | blosum62 | 1 | 0.029 | 0.017 | -0.012 | [-0.002, 0.052] | no |
| `ESM2_650M_HA80` | neg_grantham | 1 | 0.291 | 0.255 | -0.036 | [0.264, 0.322] | **yes** |
| `ESM2_650M_HA80` | blosum62 | 1 | 0.278 | 0.239 | -0.039 | [0.252, 0.306] | **yes** |
| `ESM2_650M_HA80_raw` | none | 2 | 0.070 | 0.071 | +0.000 | [0.045, 0.103] | no |
| `ESM2_650M_HA80` | none | 2 | 0.250 | 0.231 | -0.018 | [0.223, 0.276] | no |
| `ESM2_650M_HA80_raw` | neg_grantham | 2 | 0.059 | 0.057 | -0.001 | [0.030, 0.090] | no |
| `ESM2_650M_HA80_raw` | blosum62 | 2 | 0.034 | 0.030 | -0.004 | [0.005, 0.066] | no |
| `ESM2_650M_HA80` | neg_grantham | 2 | 0.244 | 0.225 | -0.018 | [0.219, 0.271] | no |
| `ESM2_650M_HA80` | blosum62 | 2 | 0.224 | 0.206 | -0.019 | [0.200, 0.250] | no |
| `ESM2_650M_HA80_raw` | none | 3 | -0.003 | -0.008 | -0.005 | [-0.058, 0.046] | no |
| `ESM2_650M_HA80` | none | 3 | 0.139 | 0.075 | -0.064 | [0.094, 0.182] | **yes** |
| `ESM2_650M_HA80_raw` | neg_grantham | 3 | 0.032 | 0.034 | +0.002 | [-0.024, 0.084] | no |
| `ESM2_650M_HA80_raw` | blosum62 | 3 | 0.018 | 0.016 | -0.002 | [-0.040, 0.067] | no |
| `ESM2_650M_HA80` | neg_grantham | 3 | 0.184 | 0.123 | -0.061 | [0.139, 0.224] | **yes** |
| `ESM2_650M_HA80` | blosum62 | 3 | 0.172 | 0.111 | -0.061 | [0.125, 0.212] | **yes** |

## Summary

* Panel B's headline Spearman moves 0.655 -> 0.642 (-0.013).
* Panel D's fine-tuned, uncontrolled estimates move by 0.013 to 0.064 across strata.
* 8 of 24 panel-D estimates fall outside the published bootstrap interval under masking.
