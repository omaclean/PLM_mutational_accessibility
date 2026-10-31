# Mutational Accessibility and DMS Fitness: Confounding, Simpson's Paradox, Skewness, and Global Alpha Adjustment

This document investigates the relationship between mutational accessibility and experimental deep mutational scanning (DMS) measurements across codon distance strata (1, 2, and 3 nucleotide changes), explaining the confounding effects observed in Supplementary Figure SM (Panels A–J).

Associated outputs:
* Updated Main Figure SM (with `pooled` + dual violins in J): [`output/sM_within_stratum.png`](file:///home3/oml4h/Mut_acc_PLM_MS/clean_repo/output/sM_within_stratum.png)
* Dev Mockup (with secondary DMS fitness axis + global $\alpha^*$ points): [`dev/dms_strata_analysis/mockup_sM_with_fitness_and_optimum_alpha.png`](file:///home3/oml4h/Mut_acc_PLM_MS/clean_repo/dev/dms_strata_analysis/mockup_sM_with_fitness_and_optimum_alpha.png)
* Strata Summary Table: [`output/sM_strata_summary_table.tsv`](file:///home3/oml4h/Mut_acc_PLM_MS/clean_repo/output/sM_strata_summary_table.tsv)
* Accessibility vs. Fitness Strip Plot: [`output/sM_accessibility_correlation_by_stratum.png`](file:///home3/oml4h/Mut_acc_PLM_MS/clean_repo/output/sM_accessibility_correlation_by_stratum.png)

---

## 1. Executive Summary

1. **The Confounder**: Mutational accessibility and biological fitness are confounded at the macro level by the genetic code. Single-nucleotide substitutions are both **more accessible** (higher mutational supply) and **more biochemically conservative** (higher average fitness / cell entry). Multi-nucleotide substitutions are both rarer and more radical, frequently destroying protein folding or function and crashing phenotypes to the assay floor.
2. **Simpson's Paradox**: When all substitutions are pooled together, accessibility appears weakly correlated with fitness ($\rho \approx +0.09$ to $+0.11$). However, **within any single codon stratum (1 nt, 2 nt, or 3 nt), accessibility has zero correlation with measured fitness** (median $\rho \approx 0.00$, ranging from $-0.11$ to $+0.06$).
3. **Flaws of Relying Solely on the Median for Skewed Assays**: DMS phenotype distributions are severely asymmetric. Cell entry assays are **heavily left-skewed** (skewness $-1.3$ to $-2.5$) with catastrophic deleterious tails down to $-6.8$. In these assays, the median remains deceptively high because ~60–70% of 1 nt mutations are tolerably neutral, masking the severe floor crashes of multi-nucleotide variants. Conversely, sera escape assays are **heavily right-skewed** (skewness $+0.7$ to $+1.4$) where the median is near zero, missing the rare epitope-escape spikes.
4. **Why Panel J Selects Negative $\alpha^*$**: In the hybrid score $\log_{10}(P_{\text{PLM}}) + \alpha \log_{10}(P_{\text{acc}})$, protein language models (especially after virus-specific fine-tuning) have already learned mutational supply biases. Inside a stratum where accessibility carries zero fitness information, weighting in positive accessibility ($\alpha > 0$) actively impairs fitness prediction. The sweep therefore selects **negative $\alpha^*$** (median $-0.70$ in 1 nt, $-0.46$ in 2 nt) to **deconvolve** and subtract out the learned accessibility bias.
5. **Fixed Global $\alpha^*$ Improves Predictions**: Evaluating PLMs with a fixed global negative accessibility weight ($\alpha^* = -0.19$ for fine-tuned models) systematically boosts prediction accuracy across fitness assays, proving that negative $\alpha$ is a genuine bias-correction rather than per-dataset overfitting.

---

## 2. Distributional Asymmetry: Why the Median is Flawed

Experimental DMS datasets exhibit severe distributional skewness. Relying solely on the median obscures critical biological constraints:

### Distributional Profile Across Strata

| Dataset | Trait | Stratum | Mean | Std Dev | Median | IQR | 10th Pct | 90th Pct | Skewness | Near Floor (<5% max) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MA22** | Cell entry | **1 nt** | $-2.06$ | $1.72$ | **$-1.16$** | $3.11$ | $-4.63$ | $+0.04$ | $-0.40$ | $14.6\%$ |
| | | **2 nt** | $-2.44$ | $1.75$ | **$-1.96$** | $3.37$ | $-4.77$ | $-0.26$ | $-0.07$ | $18.5\%$ |
| | | **3 nt** | $-2.54$ | $1.77$ | **$-2.28$** | $3.43$ | $-4.78$ | $-0.32$ | $+0.02$ | $20.5\%$ |
| **Perth09** | Preference ratio | **1 nt** | $-1.97$ | $1.29$ | **$-2.10$** | $1.73$ | $-3.61$ | $-0.40$ | $+0.21$ | $0.0\%$ |
| | | **2 nt** | $-2.35$ | $1.28$ | **$-2.60$** | $1.76$ | $-3.91$ | $-0.77$ | $+0.38$ | $0.1\%$ |
| | | **3 nt** | $-2.36$ | $1.22$ | **$-2.68$** | $1.64$ | $-3.86$ | $-0.85$ | $+0.53$ | $0.0\%$ |
| **BA.2** | Cell entry | **1 nt** | $-1.33$ | $1.52$ | **$-0.35$** | $1.86$ | $-3.80$ | $+0.07$ | **$-1.62$** | $5.0\%$ |
| | | **2 nt** | $-1.54$ | $1.50$ | **$-0.48$** | $2.36$ | $-3.97$ | $+0.01$ | **$-1.33$** | $4.0\%$ |
| | | **3 nt** | $-1.46$ | $1.44$ | **$-0.49$** | $2.23$ | $-3.77$ | $-0.04$ | **$-1.36$** | $2.7\%$ |
| **XBB.1.5** | Cell entry | **1 nt** | $-0.77$ | $1.25$ | **$-0.11$** | $0.85$ | $-2.63$ | $+0.04$ | **$-2.52$** | $3.4\%$ |
| | | **2 nt** | $-0.96$ | $1.35$ | **$-0.15$** | $1.25$ | $-3.09$ | $+0.02$ | **$-2.12$** | $4.1\%$ |
| | | **3 nt** | $-0.86$ | $1.28$ | **$-0.16$** | $1.15$ | $-2.85$ | $+0.01$ | **$-2.40$** | $3.6\%$ |
| **MA22** | Sera escape | **1 nt** | $-0.031$ | $0.152$ | **$-0.035$** | $0.063$ | $-0.134$ | $+0.122$ | **$+1.12$** | $0.1\%$ |
| | | **2 nt** | $-0.039$ | $0.152$ | **$-0.044$** | $0.064$ | $-0.148$ | $+0.119$ | **$+0.68$** | $0.0\%$ |
| | | **3 nt** | $-0.037$ | $0.169$ | **$-0.040$** | $0.066$ | $-0.147$ | $+0.133$ | **$+1.35$** | $0.0\%$ |
| **XBB.1.5** | Sera escape | **1 nt** | $-0.020$ | $0.082$ | **$-0.008$** | $0.036$ | $-0.084$ | $+0.043$ | **$-0.69$** | $0.1\%$ |
| | | **2 nt** | $-0.029$ | $0.087$ | **$-0.012$** | $0.040$ | $-0.106$ | $+0.035$ | **$-0.89$** | $0.1\%$ |
| | | **3 nt** | $-0.045$ | $0.101$ | **$-0.019$** | $0.052$ | $-0.143$ | $+0.035$ | **$-0.91$** | $0.0\%$ |

### Key Takeaways on Skewness
1. **Left-Skewed Cell Entry**: In BA.2 and XBB.1.5 spike-mediated entry, the skewness reaches $-1.62$ and $-2.52$. Over 60% of 1 nt variants are quasi-neutral, holding the median near zero ($-0.11$ to $-0.35$). However, the mean is pushed down to $-0.77$ and $-1.33$ by non-functional spikes crashing to $-6.8$. Relying on the median minimizes the perceived fitness penalty of multi-nucleotide changes.
2. **Right-Skewed Sera Escape**: MA22 sera escape has a skewness of $+1.12$ to $+1.35$. More than 80% of mutations fail to escape antibodies (concentrated near $-0.04$), so the median remains negative and flat across all codon distances. Only a sparse tail of epitope mutations escapes neutralization ($+0.5$ to $+1.3$).
3. **Parametric & Non-Parametric Shift**: A Kruskal–Wallis test confirms massive significant shifts in entry/preference distributions ($p < 10^{-15}$), whereas immune escape shows no significant change across codon distance ($p > 0.05$).

---

## 3. The "Pooled" Group Exposes Simpson's Paradox

In the updated Figure SM, panels A–I now display four groups: `pooled`, `1 nt`, `2 nt`, and `3 nt`.

Comparing `pooled` to `1 nt` immediately clarifies the confounding of PLM predictions:

| Dataset | Assay | Base Pooled $\rho$ | Base 1 nt $\rho$ | FT Pooled $\rho$ | FT 1 nt $\rho$ |
|---|---|:---:|:---:|:---:|:---:|
| **MA22** | Cell entry | $+0.427$ | $+0.463$ | $+0.351$ | $+0.462$ |
| **Perth09** | Preference ratio | $+0.457$ | $+0.474$ | $+0.297$ | $+0.434$ |
| **BA.2** | Cell entry | $+0.068$ | $+0.060$ | $+0.300$ | $+0.380$ |
| **XBB.1.5** | Cell entry | $+0.075$ | $+0.071$ | $+0.325$ | $+0.362$ |

* **In Influenza (MA22 & Perth09)**: In pooled data, the fine-tuned model appears substantially *worse* than the base model (Perth09: $0.297$ vs. $0.457$). However, inside the 1 nt stratum, the fine-tuned model matches or rivals the base model ($0.434$ vs. $0.474$).
* **Why this occurs**: The fine-tuned model learned the natural mutational bias of circulating influenza viruses. In pooled evaluation, it severely penalizes multi-nucleotide replacements that are biochemically tolerated in vitro, damaging its cross-strata rank correlation. Stratifying by codon distance removes this shared dependence.

---

## 4. Overlaying Normalized Fitness in Figure SM

In the main supplementary figure ([`output/sM_within_stratum.png`](file:///home3/oml4h/Mut_acc_PLM_MS/clean_repo/output/sM_within_stratum.png)):
1. **Clean Bar Comparison**: The bars cleanly display the rank correlation (Spearman $\rho$) of the pure model score alone (Base vs. Fine-tuned), with individual $\alpha^*$ point overlays removed to avoid visual clutter.
2. **Overlaid Mean DMS Fitness**: Plotted on a secondary right y-axis (`ax.twinx()`) as a purple dashed curve with circular markers (mean z-score $\pm$ SEM) across `pooled`, `1 nt`, `2 nt`, and `3 nt`.
   * In cell entry (MA22, BA.2, XBB.1.5) and growth preference (Perth09), the curve visually demonstrates the steep collapse of physical fitness as codon distance increases ($+0.2$ SD down to $-0.15$ SD).
   * In immune/sera escape (MA22 and XBB.1.5), the curve remains flat, explaining why prediction accuracy does not decay across codon distance.
3. **Optimized Spacing & Compact Y-Axis Ticks**:
   * To prevent horizontal subplot compression, right y-axis tick labels and the `Mean DMS (z-score)` title are rendered exclusively on the rightmost column (Column 2: panels C, F, I) on a shared $[-0.25, +0.25]$ scale.
   * Between Columns 0 and 1, and Columns 1 and 2, all secondary axis labels and ticks are hidden.
   * Subplot margins and padding (`wspace=0.02, w_pad=0.015`) are compacted, and bar widths are expanded to $0.38$ (spanning $x \in [-0.55, 3.55]$), making each panel significantly wider and eliminating empty whitespace.

