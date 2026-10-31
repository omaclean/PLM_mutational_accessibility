# Frozen panels for the composite figures

> ## ⚠️ `N145S` was a phantom mutation in Figure 1B/C — FIXED 20 Oct 2026
>
> **The J.2 → K transition is NINE replacements, not ten:**
> `K2N`, `T135K`, `S144N`, `N158D`, `I160K`, `Q173R`, `K189R`, `T328A`, `HA2:S49N`.
> Residue 145 is Serine in J.2, J.2.4 and K alike, and in the A/Aichi/2/1968 H3-numbering
> reference. There is no change at 145 on that path, in nucleotide or amino acid.
>
> **Root cause.** The panel diffed a *stored* results table's `ref_aa` against a *live*
> reference FASTA. The `J.2_int` root was corrected at precursor codon 161 (`aat` → `agt`)
> on 15 Jun 2026 and the May tables were never rebuilt, so the 2 Jul 2026 render compared a
> stale J.2 against a current K and emitted both `S144N` and `N145S`. Nothing warned.
>
> **Fix.** `guides/composites/fig01.yaml` places the fine-tuned model (`ESM2_650M_HA80`)
> render (superseded 23 Oct 2026 by `figures/make_fig01bc.py`; it was
> `data/frozen_panels/fig1bc_ranked_mutations_finetuned_esm2_ha80.pdf`), matching the
> Results text (HA2:S49N highest at rank 7, T328A lowest at rank 578); `output/fig01_composite.pdf`
> and `fig01_composite.png` carry exactly the canonical nine. The 2 Jul file is kept as
> `data/frozen_panels/fig1bc_ranked_mutations.pdf` **for provenance only — do not place it.**
>
> **Standing risk this exposed:** re-plotting an old table against newer roots can silently
> invent or erase replacements. Full account in `LINEAGE_DEFINITIONS.md` section 6.


Finished panel files that `figures/panelise.py` stitches into Figures 1 and 2
(layouts in `guides/composites/`). Total about 3.6 MB. Original paths are relative to
the old repository root (`Mut_acc_PLM_MS/`).

| File | What it is | Status | Original path |
|---|---|---|---|
| `fig1a_clade_tree.pdf` | Fig 1A: H3N2 clade tree (G.1, J, J.2, J.2.4, K) with mutation labels | **Frozen. Cannot be regenerated.** Built from GISAID sequences that cannot be redistributed; no generator exists. Vector, no text layer. | `results/final_figures/clade_viz_px445235_relabelled_evenbiggerfont.pdf` |
| `fig1a_clade_tree_large_labels.pdf` | Fig 1A as used in the composite: the same tree with every label scaled x1.65 and the legend x2.5 | **Derived**, not a new tree. Produced from `fig1a_clade_tree.pdf` by `tools/enlarge_tree_labels.py --k 1.65 --legend-k 2.5`, which scales only the label glyphs (about 14 pt tall at native size, about 4 pt once the tree is shrunk to column width), nudges each to keep clear of the tree, and adds a 30 pt blank margin on the left. Every other path is copied unchanged. The tool refuses any scale at which a label would touch the tree, another label, or the page edge. The legend carries its own, larger factor because the two constraints differ: the labels sit hard against their branches and cap `--k` at 1.65, while the legend sits in empty page at the bottom right and clears the tree to about 4.6. At 1.65 it printed at about 6 pt on the column. | derived from the row above |
| `fig1bc_ranked_mutations_finetuned_esm2_ha80.pdf` | Fig 1B and 1C: ranked PLM probabilities and ranked mutation probabilities for the fine-tuned model (`ESM2_650M_HA80`). | **No longer placed** (replaced 23 Oct 2026 by `figures/make_fig01bc.py`, drawn from `output/fig01bc_ranked_mutations_values.tsv`). It was drawn with an unstable sort, and accessibility ties run about thirty deep, so **all nine of its panel C ranks are a draw from inside a tie block rather than the block's own rank** — Q173R 250 against a competition rank of 241, I160K 3300 against 3294, and the tied S144N and HA2:S49N (both rank 928) split 13 places apart. Panel B's ranks were correct and are unchanged. Kept for provenance only. | `results/Lytras_OG/plots/per_model/ESM2_650M_HA80/ranked_mutations_j2_int_vs_k_lineage.pdf` |
| `fig1bc_ranked_mutations_corrected_13oct2026.pdf` | Fig 1B and 1C: base-model regeneration (`ESM2_650M_HA80_raw`), labelling exactly the nine. | Kept for comparison / provenance. Base unmasked model (I160K rank 24, S144N rank 31, T135K rank 5768). | `results/Lytras_OG/plots/per_model/ESM2_650M_HA80_raw/ranked_mutations_j2_int_vs_k_lineage.pdf` |
| `fig1bc_ranked_mutations.pdf` | Fig 1B and 1C: ranked PLM probabilities and ranked mutation probabilities, with the labelled replacements | **🔴 DEFECTIVE — do not ship.** This 2 Jul 2026 published copy labels **ten** replacements; the tenth, `N145S`, does not exist (residue 145 is Serine in J.2, J.2.4 and K alike — see `LINEAGE_DEFINITIONS.md`). Regenerable by `scripts/run_mutational_accessibility.py`. Kept for provenance only. | `results/final_figures/ranked_mutations_j2_int_vs_k_lineage.pdf` |
| `fig1d_codon_to_aa_heatmap.pdf` | **No longer used.** Replaced 14 Oct 2026 by `figures/make_fig01d.py`, which recomputes the matrix from `data/spectra/` with `mutacc.accessibility.codon_aa_table` and draws it on a log scale. Kept for comparison only. Fig 1D: H3N2 codon to amino-acid probability matrix (64 x 21) | **Frozen. Original not in this repo.** The notebook read it from a different repository (`.../Sequences/IAV_lineage_files/IAV_model_comparison_aa/codon_to_amino_acid_matrix_heatmap.pdf`) and no generator was found. This copy was cut out of the published composite, vector intact, with the panel letter removed. | `results/final_figures/panels/Figure_1_Composite.pdf` (bottom panel; letter streams blanked, clipped to y 1422.3-2133.5 pt) |
| `fig2a_scatter_esm2_raw.pdf` | **No longer used** (replaced 16 Oct 2026 by `figures/make_fig02ab.py`; kept for comparison). Drew all five lineages pooled, which is why its printed Spearman is 0.274. Fig 2A: PLM against mutation probability, raw ESM2 650M, coloured by number of nucleotide changes | Regenerable by `scripts/run_mutational_accessibility.py` (line writing `plm_vs_mut_prob_scatter_by_mutation_distance`). Copied so the layout is self-contained. 21 Sep render. | `results/Lytras_OG/plots/per_model/ESM2_650M_HA80_raw/plm_vs_mut_prob_scatter_by_mutation_distance.pdf` |
| `fig2b_scatter_esm2_ha80.pdf` | **No longer used**, as above (pooled, Spearman 0.652). Fig 2B: the same for the fine-tuned model (ESM2_650M_HA80) | As above. | `results/Lytras_OG/plots/per_model/ESM2_650M_HA80/plm_vs_mut_prob_scatter_by_mutation_distance.pdf` |
| `fig2c_epoch_values.tsv` | The 21 values of Fig 2C (raw and epochs 1 to 20), read back off `fig2c_plm_epoch.png` | **Derived** by `tools/digitise_epoch_curve.py`, which is what `figures/make_fig02c.py` redraws the panel from. Good to a few 1e-4. Means over three lineages, as published. | derived from the row below |
| `fig2c_plm_epoch.png` | Fig 2C: mean per-lineage Spearman correlation by training epoch | **No longer placed** (replaced 16 Oct 2026 by `figures/make_fig02c.py`, drawn from the values digitised from this file). **Frozen. No generator anywhere.** Raster (1440 x 1125 px). Its title is cut off in the image itself ("mutation rows/linea"). Uses 3 lineages, not 13. | `results/final_figures/plot_PLM_epoch.png` |

## Flags

- `fig1a_clade_tree.pdf` shows a GISAID-derived tree. It carries no strain names or
  accessions (the PDF has no text layer and the picture shows only mutation labels and
  clade names), but the tree structure is derived from GISAID data. Check the GISAID
  terms and acknowledgement requirements before redistributing this file.
- Panel 2C and the 1D heatmap have no generator, so neither can be *rebuilt*. Their plot
  areas keep their original fonts and colours and will not match the house style.
- Their **titles** are a separate matter, and are now handled. `figures/panelise.py` takes
  `crop_top_frac` and `title`/`tagline` per panel, so a layout can cut a frozen panel's own
  title off and have a house-style header drawn in its place. Figure 2 does this for 2A, 2B
  and 2C, which is what finally fixed 2C's title being truncated mid-word in the source
  image. The crop also removes 2C's floating one-entry legend box. Nothing inside the plot
  area is altered.
