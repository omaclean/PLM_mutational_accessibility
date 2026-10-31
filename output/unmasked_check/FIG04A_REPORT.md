# Figure 4 panel A under a masked-marginal estimator

MA22 cell entry (A/Massachusetts/18/2022 H3N2 HA, Yu et al. 2026), 9,234 scored
replacements over 504 sites, both ESM-2 arms. The published panel reads the model
probability from a single unmasked forward pass of the scanned backbone, so the model sees
the wild-type residue at the site it is scoring. Here each position is masked in turn
(504 passes per model) and everything else — accessibility column, phenotype, alpha grid,
bootstrap seed and arm order — is held fixed.

Built by `dev/masked_dms/embed_masked_profile.py` then `dev/masked_dms/make_fig04a_masked.py`.

## Gates

| gate | result |
| --- | --- |
| unmasked profile reproduces `plm_cache` for `ESM2_650M_base` | max abs diff 9.77e-06 |
| unmasked profile reproduces `plm_cache` for `ESM2_650M_HA80` | max abs diff 2.59e-06 |
| recomputed unmasked score reproduces shipped `log_plm` | max abs diff 3.13e-05 |
| published arm reproduces the shipped panel-A optimum | exact (`best_alpha`, `rho_best`, `rho_alpha0`, `n`, `n_sites`) |

Both gaps are float32 round-trips of the shipped cache, not differences in the sweep. The
left panel of the figure is the published computation, not a lookalike.

## Result

| scoring | model | alpha\* | rho at alpha\* | rho at alpha = 0 | gain |
| --- | --- | --- | --- | --- | --- |
| unmasked (published) | Base (ESM-2) | −0.05 | 0.4194 | 0.4172 | +0.0022 |
| unmasked (published) | Fine-tuned (HA-80) | −0.17 | 0.3232 | 0.2897 | +0.0335 |
| masked | Base (ESM-2) | −0.06 | 0.4306 | 0.4271 | +0.0035 |
| masked | Fine-tuned (HA-80) | −0.21 | 0.4018 | 0.3509 | +0.0509 |

Every curve is interpretable under the figure's own rule (interior optimum, signal at
alpha = 0).

## What it says

The negative-alpha optimum is not an artefact of the wild-type leak. Masking moves it
further from zero on both arms (−0.05 → −0.06 base, −0.17 → −0.21 fine-tuned) and the gain
from weighting grows rather than shrinks, most of all on the fine-tune (+0.034 → +0.051).
If the leak were doing the work, down-weighting accessible replacements would have stopped
helping once the model could no longer read the wild type; it helps more.

The other change is to the gap between the two models. Masking lifts the fine-tune far more
than the base (+0.061 at alpha = 0 against +0.010), so the fine-tune's deficit at its own
optimum narrows from 0.096 to 0.029. The unmasked estimator was penalising the fine-tune:
it is the arm whose training has sharpened its wild-type preference, so it is the arm that
loses most when that preference is handed to it for free. That is worth knowing before the
panel is read as "fine-tuning costs you accuracy against DMS".

Nothing here is a published number. Panel A as drawn in `output/fig04_alpha_vs_dms.*` is
unchanged.
