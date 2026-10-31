# Masked marginals for the DMS figures — progress

Working note for the 19–20 Oct 2026 change: every DMS comparison in the manuscript, main and
supplementary, now scores the model with a **masked marginal** instead of the wild-type
(unmasked) marginal. Status, what is gated, and what is left.

## Why

The site being scored is replaced by the mask token and predicted from its context, so the
residue actually there cannot inform its own prediction. The unmasked estimator is **not
neutral between the models being compared**: a model fine-tuned on one protein family can
largely copy a residue it is shown, which squeezes the probability left for the alternatives —
and it is the ranking of those alternatives that every one of these figures scores. Scoring both
arms unmasked therefore flatters the untuned model.

The size of it, Figure 4 panel A (MA22 cell entry): masking lifts the fine-tune by **+0.061** in
rho at α = 0 against **+0.010** for the base, cutting the fine-tune's apparent deficit at its own
optimum from 0.096 to 0.029. That deficit was mostly an artefact of the estimator.

It is not a claim about how far the individual probabilities move: those shift by about the
same amount for both arms (median 0.40 in log10). It is a claim about the ranking, which is
what the figures plot, and it is tested that way — across the seven fitness panels of both
viruses, masking lifts the fine-tuned arm more than the base arm in **every one**.

Rationale in full: `mutacc/masked_dms.py`.

## The gate, everywhere

The archive's own curves, flags and within-stratum correlations were all computed unmasked, so
the unmasked arm is the only one with a fixed point to check against. **Every switched figure
still recomputes its unmasked arm and checks it against the archive before drawing the masked
one.** That is what makes "masking is the only thing that changed" a measured statement rather
than an assumption — without it, a difference could be a tokeniser, a checkpoint or an
off-by-one.

Measured gates:

| gate | result | tolerance |
| --- | --- | --- |
| unmasked profiles reproduce the shipped `log_plm` (all 4 backbones) | 3.13e-05 | 1e-4 |
| ESM-C unmasked profiles vs published `plm_cache` (BA2/XBB15, both arms) | 2.98e-08 – 7.29e-08 | 1e-4 |
| Fig 4 / SL recomputed unmasked curves vs the archive's | 3.54e-08 | 1e-6 |
| Fig SM recomputed unmasked correlations vs `dms_within_stratum` | 9.7e-17 | 1e-9 |
| Fig SM refitted pooled α vs `dms_peak_summary` | 0.000 | one grid step |
| Fig 3 recomputed unmasked sweep vs shipped `alpha_sweep_by_lineage` | 7.25e-15 | 1e-9 |
| Fig 3 unmasked profiles vs archive `plm_prob` | 8.67e-06 | 1e-5 |

## Done

- **Masked profiles, all four DMS backbones.** MA22 and Perth09 (ESM-2, via `transformers`),
  BA2 and XBB15 (ESM-C, via EvolutionaryScale's `esm`). One forward pass per site per model.
  `dev/masked_dms/embed_masked_profile.py` and `embed_masked_profile_esmc.py`.
- **`dms_scores_masked`**, the shipped table carrying the masked probability for every row of
  `dms_scores`, so nothing in `figures/` needs a GPU or the 20 × L profiles.
  `tools/build_masked_dms_scores.py`; registered in `guides/h3n2_13lineage.yaml`, with a
  MANIFEST row and a provenance JSON of its own.
- **`mutacc/masked_dms.py`**, the shared reader/applier, promoted out of `dev/` now that the
  main figures depend on it. Asserts one-to-one keying and row order — the site bootstrap
  consumes draws in table order, so a reordered join would move every band while leaving the
  curves correct, which would look like a result. It also owns `--scoring`, so the switched
  figures word that option identically.
- **Figure 4 and Figure SL switched and installed.** `--scoring` defaults to `masked`,
  `unmasked` redraws the old version. Caption states the estimator. Gate runs on every build.
- **Figure SM (within-stratum) switched and installed.** Harder than Figure 4: the shipped
  `dms_within_stratum` holds the unmasked arm *and its intervals*, so the masked bars carry
  intervals recomputed in `recompute_with_band` — the same site bootstrap, not the archive's
  draw, and the caption says so. Both gates (the table, and the pooled α against
  `dms_peak_summary`) run on the unmasked arm first, whichever arm is drawn; the gate's α fit
  is pooled-only and unbootstrapped so the masked arm does not pay for the full refit twice.
- **`masked_dms_curves`** rebuilt over both viruses and both estimators (7,236 rows), with
  MANIFEST rows for it and `masked_strata`.
- **Figure SO** draws the main figure's virus; `--virus SC2` repeats the same check on the
  SARS-CoV-2 panels into `output/unmasked_check/`, where it gets a note rather than a caption
  (it is not the numbered figure and must not claim its label).
- **Table SY (`sY_dms_alpha_optima`) regenerated, and its caption now generated with
  it**, from the installed masked figures. This closes `docs/notes/extended_red_team_tests.md` A1 and A13:
  the duplicate `table_s15_*` files are gone, nothing is labelled S15 or S16 any more, the sign
  handling for the receptor-binding rows is explicit (`delta_rho_in_size`), and defect 5's
  `xfail(strict=True)` is off. The headline survives masking: fine-tuned optima −0.24 to −0.13
  against base −0.06 to +0.04.
- **Masked Figure 3.** First a sanity check in `output/unmasked_check/fig03_13lineage/`
  (`FIG03_REPORT.md`): masking lifts every curve in every panel, the figure's claim survives with
  about an eighth shaved off the gain, and model ordering is unchanged. On that result Figure 3
  was switched as well — it now reads `alpha_sweep_by_lineage_masked` and
  `replacement_scores_masked` by default, with `--scoring unmasked` for the old one, so
  `FIGURE_LABELS.md` is right that masked marginals run through Figures 3, 4, SB, SL and SM.

- **The archive summary is labelled as unmasked.** `dms_peak_summary` keeps its
  `rho_alpha0` / `peak_alpha` / `peak_rho` / `gain` columns — they are the fixed point the gate
  checks against and must not move — but the MANIFEST row and `DATA.md` now say in as many words
  that they are the wild-type estimator and are not what the manuscript reports. They cannot be
  quoted beside the figures: on **MA22 pH stability the two estimators disagree in sign** at
  α = 0. Unmasked puts the fine-tune below the base (0.127 vs 0.138); masked, which is what
  Figure 4 draws and Table SY carries, puts it above (0.138 vs 0.121). The Results sentence
  about a modest pH-stability gain is therefore the masked reading, and is correct as written
  provided the estimator is named. Four tests in `tests/test_masked_dms.py` pin it: Table SY
  equals the masked arm on all 18 arms and *differs* from the unmasked one, the archive table
  stays unmasked, and the sign disagreement itself is asserted in both directions so it cannot
  be quietly reconciled away.

## The Results sentence (manuscript edit, not in this repo)

The manuscript is edited in Word outside the repository, so this is the wording rather than a
diff. In MS (21), Results, the DMS paragraph reads:

> The impact of fine tuning on prediction accuracy of DMS experiments is mixed. For predicting
> cell entry, and cell line reproduction (preference ratio) the fine tuning reduces the Spearman
> ρ coefficient, but it is notably raised in sera escape, with modest gains in pH stability
> prediction also.

Every direction in it is **correct under the masked estimator**, which is what Figure 4 draws and
what the Methods already declare ("We used masked token embedding for each amino acid"). It is
the unmasked estimator that would make the pH clause false. What the sentence lacks is the
numbers and the estimator, which is what makes the claim checkable against Table SY:

> The impact of fine tuning on prediction accuracy of DMS experiments is mixed. Correlations are
> quoted at α = 0 under the masked-marginal estimator used throughout Figure 4 (Methods); the
> wild-type marginal reverses the pH stability ordering, and Figure SO15 shows both. For
> predicting cell entry (ρ 0.43 to 0.35) and cell line reproduction (preference ratio, 0.46 to
> 0.30) the fine tuning reduces the Spearman ρ coefficient, but it is notably raised in sera
> escape (0.08 to 0.14), with a modest gain in pH stability prediction also (0.12 to 0.14,
> though the site-bootstrap intervals overlap).

The overlap clause is worth keeping: the pH stability intervals at α = 0 are base
[0.061, 0.174] against fine-tuned [0.101, 0.189]. It is the one panel in the paragraph whose
direction rests on the estimator, so it should not be stated more firmly than the interval
supports. Numbers from `output/sY_dms_alpha_optima.tsv`.
- **Tests.** `tests/test_masked_dms.py` (23 cases): the shipped table's keying, the
  order-preserving join and its guards, the profile reader's bounds, the gate in both
  directions, and the ranking claim above. The switched figures' own tests run their masked
  default.
- **Documents.** `README.md`, `figures/README.md`, `DATA.md`, `RUNS.md` (the GPU commands and
  their gates) and `FIGURE_LABELS.md` carry the new tables, the new figure row and the
  estimator.
- **Stage 3 of the stem rename landed under this work** (`s12_within_stratum` ->
  `sM_within_stratum` and so on). Everything above is written against the new names; the one
  stem it left behind, the alpha-optima table, is now `sY_dms_alpha_optima`, written by
  `tools/export_sY_dms_alpha_optima.py`.

## Settled: Figure 2D stays unmasked

**Decision, 20 Oct 2026 (Oscar).** Figure 2D keeps its single unmasked pass. The supplement
carries the comparison — panel E of Figure SO is that quantity under both estimators — and that
is enough: the estimator was changed where a figure *ranks the alternatives at a site against
each other*, which is what the unmasked pass flatters, and 2D correlates one score with
accessibility rather than ranking alternatives. `masked_strata` exists and is drawn, so the
cost of the choice is on the page rather than assumed. The panel's caption says so, and names
2D through the numbering map rather than typing a figure number.

## Still to do

- The provenance JSONs under `output/unmasked_check/` written before 20 Oct carry absolute
  paths (`/home3/...`) for the reference FASTA and the checkpoint. The embedders now write
  those as `$PLM_REPO/...`, but the shipped files were produced before that and only a GPU
  re-run replaces them honestly. Not caught by any test — the no-absolute-paths rule reads
  `.py`, `.yaml`, `.sh` and `MANIFEST.tsv`, not JSON.

## Not ours, found on the way

`tests/test_panelise.py` has two failures that predate this work and belong to Figure 1's
layout, not to masking: `test_fig01_tree_panel_carries_a_header` (the 1A header is inset, so
`header_height` is 0) and `test_fig01_tree_crop_stays_clear_of_the_k_clade` (the layout crops
9% off the top, the test's ceiling is 6%). Either the layout or the tests are out of date;
which one is a decision about how Figure 1 should look.

## Environment note

ESM-C will not construct in `plm_entropy` without `Functions_HuggingFace.patch_esm_sequence_tokenizer()`
(transformers 4.41.2 predates the `_special_tokens_map` refactor `esm>=3` is written against),
and an ESM-C **fine-tune cannot be loaded by path** — it is a state dict loaded into the base
architecture. Both are handled by calling `_load_plm_runtime`, which is the function that built
the published cache. The conda *base* interpreter loads ESM-C bare but on a different torch and
esm, so it is a cross-check only.

The GPU scripts under `dev/masked_dms/` take their paths from the environment (`PLM_REPO`,
`PLM_MODEL_DIR`, `SC2_FINETUNE`; see `dev/masked_dms/_env.py`), because the repo forbids
absolute path literals. They exit with what to set if one is missing.
