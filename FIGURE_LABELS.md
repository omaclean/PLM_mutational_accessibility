# Supplementary figure labels

**Supplementary figures and tables carry letters, not numbers** (20 Oct 2026). A
supplementary figure is `Figure SA` … `Figure SO`; the one supplementary table is
`Table SZ`. Main-text figures keep their digits (`Figure 1` … `Figure 5`).

## Why a letter

The draft cites the supplement in an order that is still moving. A letter is a *name*, not
a position: a figure can be added, dropped or cited earlier without every caption, document
and test following it around, and the only edit a re-order needs is to the manuscript text.

The manuscript's final `S1 … SN` run is assigned **once, at submission**, by walking the
citations in order of first mention and substituting digits for letters. That substitution
is mechanical — each letter appears exactly once in this table — and nothing in the
repository has to change when it happens.

The letters are not a sequence anyone should read meaning into beyond the table below. In
particular `Table SZ` is parked at the end of the alphabet on purpose, so that adding a
supplementary figure never moves it.

## Where the labels live

One file: [`guides/manuscript/figure_numbers.yaml`](guides/manuscript/figure_numbers.yaml),
read through `mutacc.numbering`. Nothing else may hard-code a label — `mutacc.io.write_caption`
refuses a caption whose text names its own figure, and `tests/test_numbering.py` checks that
every shipped caption opens with the label the map gives it. To re-label a figure, edit that
file and re-run the figure (or `tools/build_figures.py`).

Scripts, output filenames and tests still name a figure by its **stem** (`s13_contact_recovery`).
The digits inside a stem are historical and mean nothing; stage 3 of
[`docs/notes/extended_red_team_tests.md`](docs/notes/extended_red_team_tests.md) §A1b replaces them with content names.

## The set

The *old draft number* column is the numbering the 15 Oct 2026 draft used, kept here so a
reference in an existing document or review reply can be translated.

| Label | Old draft number | Stem | Script | Content |
|---|---|---|---|---|
| Figure SA | S1 | `sA_spike_model_vs_accessibility` | `make_sA_spike.py` | The Figure 2 test on a second virus: SARS-CoV-2 spike. |
| Figure SB | S2 | `sB_ranking_metrics` | `make_sB_ranking_metrics.py` | Six ranking metrics across the accessibility weight (using masked marginals). |
| Figure SC | S3 | `sC_viral_panel_recovery` | `make_sC_viral_panel.py` | How well two untuned protein language models recover a hidden amino acid, across viral proteins and a cellular/phage comparison panel. |
| Figure SD | S4 | `sD_partial_correlation` | `make_sD_partial_correlation.py` | Removing amino-acid chemistry from the model–accessibility link, three ways. |
| Figure SE | S5 | `sE_spectra_in_codon_space` | `make_sE_spectra_codon_space.py` | Mutation spectra compared in codon space. |
| Figure SF | S6 | `sF_spectrum_swap` | `make_sF_spectrum_swap.py` | How strongly each model's score tracks mutational accessibility, spectrum by spectrum. |
| Figure SG | S7 | `sG_epistasis_null` | `make_sG_epistasis_null.py` | Are the shifts between the clade-defining substitutions larger than what any substitution does when its background changes? |
| Figure SH | S8 | `sH_contact_recovery` | `make_sH_contact_recovery.py` | Does the model's coupling signal recover real 3D contacts, and where? |
| Figure SI | S9 | `sI_codon_pair_tracking` | `make_codon_learning.py` | Every scored replacement (within-replacement AUROC for all 54). |
| Figure SJ | S10 | `sJ_dms_concordance` | `make_sJ_dms_concordance.py` | Agreement between deep mutational scans of one protein, against how far apart the scanned strains are. |
| Figure SK | S11 | `sK_branch_codons` | `make_sK_branch_codons.py` | Nucleotide changes per codon on individual internal branches of the tree. |
| Figure SL | S12 | `sL_dms_sc2` | `make_fig04.py` | Measured fitness effects against a language model's score, as the weight on mutational accessibility changes (SARS-CoV-2; masked marginals). |
| Figure SM | S13 | `sM_within_stratum` | `make_sM_within_stratum.py` | Model score against measured fitness, within each nucleotide-step stratum. |
| Figure SN | S14 | `sN_simulation_control` | `make_sN_simulation_control.py` | Reviewer 2 simulation control across four evolutionary arms. |
| Figure SO | S15 | `sO_masked_marginals` | `make_sO_masked_marginals.py` | What the wild-type (unmasked) estimator would have given (directly comparing masked marginals against unmasked marginals). |
| Figure SP | — (new) | `sP_alpha_curves` | `make_sP_alpha_curves.py` | Every deep mutational scan's full accessibility-weight curve, behind Figure SM's optimum counts. |
| Table SW | — (new) | `sW_loo_alpha_selection` | `tools/export_sW_loo_alpha.py` | The accessibility weight chosen on twelve lineages and scored on the thirteenth: held-out weight and gain for every fold, model and metric. |
| Table SY | Table S15 / S16 | `sY_dms_alpha_optima` | `tools/export_sY_dms_alpha_optima.py` | The best accessibility weight for each deep mutational scan, one row per panel and model. |
| Table SZ | Table S1 | `sZ_lineage_panel` | `make_sZ_lineage_panel.py` | The 13 influenza A/H3N2 HA lineages, their sampling periods and their variation. |
| Table SX | Table S2 | `sX_corpus_leakage` | `make_sX_corpus_leakage.py` | The nine subclade K replacements against the pre-2015 fine-tuning corpus: how many corpus sequences already carried each new residue. |

## Estimator usage: Masked vs. unmasked marginals

- **Masked marginals** are used throughout **Figure 3, Figure 4, and Supplementary Figure SB** (as well as Figure SL and Figure SM). The focal position is masked during model scoring (`<cls> ... <mask> ... <eos>`) so wild-type sequence identity cannot inflate the untuned model's probabilities.
- **Unmasked marginals** are restricted to **Figure 2** (preserving the published unmasked single-pass baseline) and **Figure SO** (the supplementary figure explicitly comparing the masked and unmasked estimators).

Dropped from the numbered set: `s14_lineage_set_validity` (retired 15 Oct 2026; its findings
stay in [`LINEAGE_VALIDITY.md`](LINEAGE_VALIDITY.md) §4), `fig02_model_vs_accessibility` and
`s06_spectrum_delta`. A retired figure has no label and asking for one raises.

## The table that was numbered twice (closed 20 Oct 2026)

The alpha-optima table shipped twice, as `table_s15_dms_alpha_optima.tsv` and
`sY_dms_alpha_optima.tsv`, byte-identical and both stale against the figure values
they are derived from. Only the second had a generator; the first had a hand-written caption
calling it "Supplementary Table S15", while the live one shared the number S16 with
`table_s16_simulation_control.tsv`. The `s15` copy and its caption are gone,
`tools/export_sY_dms_alpha_optima.py` now writes the caption with the table, and the table is
**Table SY** — a letter of its own, which is what [`docs/notes/extended_red_team_tests.md`](docs/notes/extended_red_team_tests.md)
§A1 was waiting for. The stem keeps its `s16`, which means nothing, until §A1b renames stems.

`table_s16_simulation_control.tsv` is the companion table of Figure SN and takes that
figure's label rather than one of its own.
