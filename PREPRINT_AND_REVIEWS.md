# What the preprint says, and what the reviewers asked

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
> render (`data/frozen_panels/fig1bc_ranked_mutations_finetuned_esm2_ha80.pdf`), matching the
> Results text (HA2:S49N highest at rank 7, T328A lowest at rank 578); `output/fig01_composite.pdf`
> and `fig01_composite.png` carry exactly the canonical nine. The 2 Jul file is kept as
> `data/frozen_panels/fig1bc_ranked_mutations.pdf` **for provenance only — do not place it.**
>
> **Standing risk this exposed:** re-plotting an old table against newer roots can silently
> invent or erase replacements. Full account in `LINEAGE_DEFINITIONS.md` section 6.


Everything here comes from the two files in `MS_stuff/` (read 9 Oct 2026):

* `2026.07.06.736753v1.full.pdf` — the bioRxiv preprint (doi 10.64898/2026.07.06.736753, posted
  9 July 2026). MacLean, Lamb, Lytras, Mojsiejczuk, Yuan, Hughes, Robertson.
* `Email - Oscar MacLean - Outlook.pdf` — the MBE editorial decision of 3 Sep 2026 with both
  reviews.

**Why this file exists.** Earlier notes in this repo called several figures "published" that are
**not in the preprint**. The within-stratum correlations, the BLOSUM62 and Grantham comparators,
and the amino-acid direction analyses are all *revision-stage* work, written in answer to these
reviews. Read the section "Preprint vs revision figures" before trusting any "published" in
another file.

## Timeline

| Date | Event |
|---|---|
| 9 Jul 2026 | Preprint posted. Submitted to MBE as MBE-26-0726. |
| 3 Sep 2026 | Decision: **major revision**, 60 days (so about **2 Nov 2026**). Medium priority. The editors warn that a manuscript needing more than one major revision is often rejected. |

Editors' summary: both reviewers see a valuable, timely contribution but flag clarity and
**reproducibility**, "a lot of additional data analysis", and uncertainty missing throughout.

## What the preprint contains

Title: *Protein language models learn underlying mutation biases alongside fitness landscapes.*
Claim in the abstract: base PLMs (ESM-2, ESM-C) "implicitly learn generic nucleotide-level
mutational constraints, an effect strongly amplified by virus-specific fine-tuning"; adding explicit
accessibility "significantly improves" prediction of which replacements are observed; it hurts
prediction of the frequency of replacements that already exist; PLM probabilities are "a composite
metric that conflates protein structural fitness with historical biases in mutational supply".

### Figures

| Preprint figure | Content | Numbers quoted |
|---|---|---|
| **1** | (A) H3N2 K-lineage tree; (B) all 19 x 567 HA-80 replacement probabilities, ranked, with the J.2 to K replacements marked; (C) the same for mutation probabilities; (D) 64 x 20 codon-to-amino-acid heatmap. | Highest-probability K replacement HA2:S49N, p = 0.910, rank 7. Lowest T328A, p = 1.79e-5, rank 572 of 10,754. I160K ranks 3,296 in accessibility; Q173R 258. |
| **2** | (A) ESM-2 650M probability against accessibility; (B) the same for fine-tuned HA-80; (C) Spearman by fine-tuning epoch. Scatter only: **no strata, no BLOSUM62, no Grantham, no direction analysis.** | Spearman 0.274 (ESM-2), 0.652 (HA-80). Pearson R-squared 0.025 and 0.043. ESM-C on SARS-CoV-2 spike: 0.298 before, 0.682 after fine-tuning (Supplementary Fig 1). Learning plateaus after epoch 2. |
| **3** | Alpha sweep on **five** lineages (G.1, J, J.2, J.2.4, K): (A) Spearman on all replacements, zeros included; (B) PR-AUC of a logistic regression for which replacements are observed; (C) Pearson on observed replacements only. Score = PLM_prob x mutation_prob^alpha. | (A) HA-80 0.298 at alpha 0 to 0.311 at 1; accessibility alone 0.289; raw ESM-2 0.195 rising to 0.300. (B) HA-80 0.27, accessibility 0.13, ESM-2 0.23; with accessibility 0.37 and 0.32. (C) HA-80 0.228, ESM-2 0.107, accessibility alone 0.012; adding accessibility generally hurt. Negative alpha gave no real gain. |
| **4** | Epistasis scan of the nine J.2 to K replacements: shift in each replacement's raw probability when the others are present. | I160K shifts most; S144N and N158D drive it. I160K and N158D fall on the same branch. |

### Methods that matter

* **Sequences and lineages.** GenBank via the V-gTK toolkit plus GISAID up to 12 Feb 2026, H3 HA
  coding sequences. A sequence belongs to a lineage if it has all the lineage-defining mutations
  and **fewer than five additional amino-acid replacements** from that lineage's root; sequences
  with more than two of the *next* lineage's defining nucleotide mutations are dropped. Resulting
  counts: **G.1 229, J 4,132, J.2.4 877, J.2 27,452, K 17,898.**
* **Accessibility.** Follows Gunnarsson and Babu (2023) using the Pauly et al. (2017) mutation
  probabilities; "only the relative probabilities matter". Two-step changes are products of
  per-step probabilities. SARS-CoV-2 uses the De Maio et al. (2021) 4x4 matrix scaled by
  1.3e-6 per replication (Amicone 2022) and the genome composition (scaling 8.186e-7).
  ✅ **Checked 22 Oct 2026 and consistent.** The composition is `NC_045512.2` (Wuhan-Hu-1),
  which the manuscript did not cite and the repo did not hold; it now ships at
  `data/reference_genomes/NC_045512.2.fasta` and its 29.94/18.37/19.61/32.08 rounds to the
  stated percentages exactly. The factor does not recompute to the digit — the shipped
  matrix's composition-weighted mean is 1.3010e-6, 0.076% off 1.3e-6 — but that residual is
  inside the rounding interval of the three-significant-figure matrix, and the scale cancels
  in every within-stratum comparison. **Manuscript action: cite the accession in Methods.**
  Derivation in `data/spectra/README.md`; pinned by `tests/test_sars_cov_2_scaling.py`.
* **Compression analysis.** The 64 x 64 codon matrix is summed to 64 x 20, averaged over
  synonymous source codons (equal codon usage assumed) to 20 x 20, then rebuilt to 64 x 20.
  **84.8%** of the H3N2 matrix's variation survives, against **43.6%** under a Kimura 80
  model (kappa 2). The text reads this as evidence that fine-tuning "increases the understanding of
  mutational biases". Both reviewers object (below).
  ⚠️ **The statistic is not Pearson r**, despite an earlier note here saying so. The archive
  computes `100 x (1 - var(target - predictor) / var(target))` — a variance-retained fraction,
  which is unbounded below and reaches **-61%** for one of the spectra in S5. Pearson r and r²
  cannot go negative, so it is neither. It coincides with R² only when the predictor is a
  least-squares fit of the target, which these matrices are not. This answers Reviewer 2's
  "is it r or r-squared" directly; see S4 (withdrawn), S5 and `tools/spectra_compare.py`.
* **Models.** ESM-2 650M, and HA-80 from Lytras et al. (2026), fine-tuned on influenza A HA up to
  February 2015 (99% identity clusters, chosen so that the 2015 cutoff precedes J and K). ESM-C 600M
  fine-tuned on SARS-CoV-2 spike **with MuonW and MAGMA**, which HA-80 did not use.
* **Sweep.** Alpha is applied per lineage and the metrics are averaged across lineages. Three
  metrics: Spearman (all replacements), PR-AUC (which replacements are observed), Pearson
  (frequency of observed replacements).
* **Phylogeny.** As Dee et al. (2026): MMseqs2 at 0.988 identity, MAFFT, IQ-TREE v2.4.0, treetime
  v0.10.0 for ancestral changes.
* **Numbering.** The text uses mature H3 numbering (K2N, T135K, S144N, N158D, I160K, Q173R, K189R,
  T328A). Figures 1B and 1C use numbering offset by **16** (K18N, T151K, S160N, N174D, I176K,
  Q189R, K205R, T344A).

## The reviews

### Reviewer 1 (identifies themselves as Erick Matsen)

| # | Point | Where it stands in this repo |
|---|---|---|
| 1 | The K80 comparison conflates model expressivity with species: K80 has uniform base frequencies, which no pan-species data has. A fair comparison needs a model of equal capacity fitted to pan-species data. | **Done**: S4 (withdrawn) and S5 (`make_sD_partial_correlation.py`, `make_sF_spectrum_swap.py`). Measured spectra from six other organisms are added, several with the same twelve free parameters as influenza's. Expressivity does not order them: two twelve-parameter spectra score **+83%** and **-61%** against H3N2, while one-parameter K80 scores **+44%**. The gap is about identity, not capacity. |
| 2 | Describe the fine-tuning objective (masked language modelling); 84% is a ceiling, but "not a very tight ceiling". The wording that fine-tuning "increases the understanding" implies it is good, which sits oddly with a paper about conflation. | Text change. |
| 3 | Novelty: the conclusion and Fig 2A are Matsen et al. 2026 (eLife; their Fig 1A, with an ESM comparison in a supplement and a DMS comparison). The alpha-weighting hypothesis is Matsen et al. 2025 (PLOS Comp Biol), **not cited**. | Text and citations. |
| 4 | Show regressions **per codon stratum** (1, 2, 3 nucleotide changes): is fine-tuning learning more than the stratification? | **Done in the revision**: Fig 2 panel A, `make_fig02.py`. |
| 5 | Define Score mathematically; is it normalised; can it be read as a probability? The caption says PLM_prob x mut_prob^alpha but the axis says mut_acc. | Text. Our captions state score = log10 p + alpha log10 accessibility. |
| 6 | Fig 4 does not fit the narrative: epistasis is not mutational supply, and reading it as epistasis accepts that PLMs are *not* polluted. It is the "categorical Jacobian" of Zhang et al. (2024), uncited. | Epistasis moved to supplementary; **S7 is now ported** (`make_sG_epistasis_null.py`). The new Fig 4 is DMS. Zhang et al. (2024) is the source of the zero-sum gauge S7 uses by default. |
| 7 | Separate mutation from selection using Lee et al. (PNAS 2018) DMS. | **Done**: Fig 4 includes Perth09 (Lee 2018). |
| 8 | More references (CovFit, DeepCoV, two 2026 preprints, the Wilke-lab refutation of Hie et al.). | Text. |
| R | **Reproducibility**: absolute `/home3` paths; GISAID-free inputs missing (`IAV_lineage_guide.csv`, `SC2_month_snapshots/`); fine-tuned checkpoints not linked; repo repurposed from an unrelated 2024 paper; `PLM_entropy_play` not public. | This repo's reason for existing. Paths, guides and tables are fixed; **checkpoints are still not hosted**. |
| d | Figure 1 pixelated; Figures 2A, 2B heavily overplotted; 2C title cut off; "I160K" vs "I176K" numbering. | Fig 1 composite rebuilt (vector); the 2C title is truncated in the source image and has no generator; the numbering offset remains to fix in text. |

### Reviewer 2

| Point | Where it stands |
|---|---|
| **Confounder**: biochemical similarity and the genetic code. The three bands in Fig 2A,B are 1, 2, 3 nucleotide changes, about four orders of magnitude apart, so any score that ranks conservative above radical replacements will correlate. Compute within strata. | **Done**: Fig 2A. |
| Benchmark against matrices with no PLM: BLOSUM62, and the FLU matrix of Dang et al. (2010). "If BLOSUM62 reaches the correlation reported for ESM-2, and FLU that for HA-80, the finding is that PLMs recover what exchangeability matrices already encode." | BLOSUM62 and Grantham done. **FLU (Dang 2010) is not in `a34`'s score list**; the 9Octy README quotes r = 0.41 for it, which I have not verified. |
| **"Perhaps the PLM encodes non-reversible information, which would support the paper's title directly."** | **This is the symmetric-versus-directional question.** See `SYMMETRY_TESTS.md`. |
| Control separating mutation from selection: does the influenza fine-tune correlate better with the influenza matrix than with the SARS-CoV-2 matrix, and the ESM-C spike model the reverse? A crossover is what a virus-specific process predicts. | **Partly done**: S6 (`make_sG_epistasis_null.py`) ports this as ten spectrum conditions within strata. ⚠️ It varies the *spectrum* for one protein, not the protein: nothing shipped scores a second protein with a fine-tuned model, so the crossover the reviewer asks for is still not tested. The SARS-CoV-2 side needs the same treatment. |
| **Simulation control**: fine-tune on sequences evolved under the influenza spectrum and, separately, under a neutral spectrum with the same site-wise constraints. | Done for human and SARS-CoV-2 spectra (`dev/simulated_phylogenies/`, S28). **No neutral (JC69) arm was fine-tuned.** |
| The compression result contains no PLM: reframe as an **upper bound**, report where the PLM-implied matrix falls against it, and use empirical H3N2 codon usage instead of uniform. | Not done. |
| Uncertainty: no confidence interval or test anywhere, yet the abstract says "significantly". Fig 3B is titled "Logistic regression AUROC", the legend says PR-AUC, and the text quotes PR-AUC 0.13 to 0.37. Alpha is chosen on the data it is scored on: use leave-one-lineage-out or temporal cross-validation. Sequence counts differ hugely between lineages. Filtering may remove multi-replacement variants. | Intervals and per-lineage values: Figs 3, S2. Metric naming resolved (AUROC main, PR-AUC in S2). **Leave-one-lineage-out alpha is now computed here** (Table SW, `tools/export_sW_loo_alpha.py`, `mutacc.loo_alpha`): alpha is chosen on 12 lineages and scored on the 13th, 13 folds, both scoring arms. On the masked arm Figure 3 draws, the fine-tuned model's held-out gain is +0.023 [+0.019, +0.027] AUROC and +0.073 [+0.055, +0.092] PR-AUC, 13/13 folds, with every fold's weight at +0.52 to +0.58; the base model's alpha* is pegged at the +1.00 grid edge in every fold, so its larger gain is a bound, not an optimum. No p-value is quoted: the lineages are pseudo-replicates (see below), and 13/13 is a sign test's floor whatever the effect size. Temporal cross-validation and the depth control are still not re-run here. Multi-replacement filtering: not checked. |
| Fig 4 has no null; I160K and N158D on one branch are not independent evidence; "improbable I160K depends on S144N and N158D" is causal; test the negative-alpha claim on DMS. | DMS test done (Fig 4). **Epistasis null now ported** as S7: 72 clade pairs against a pooled null of 92,682 draws. Only 3 pairs pass pooled BH q<0.05 under the zero-sum gauge, and 0 under the wild-type gauge — the gauge changes the answer, so S7 states which one it draws. |
| Minor: harmonise numbering; use percentiles rather than ranks (the text says all K replacements rank high, yet I160K ranks 3,296 of 10,754); say whether 84.8% and 43.6% are r or r-squared (**answered: neither** — it is a variance-retained fraction that can go negative; see §What the preprint contains); Pearson on raw probabilities spanning sixteen orders is uninformative, use logs; report Spearman beside Pearson for frequencies; count independent origins on the tree; archive V-gTK with a DOI; name the IQ-TREE model; assess ancestral-reconstruction uncertainty; state whether the nine K replacements occur in the pre-2015 training data; match optimisers between HA-80 and the SARS-CoV-2 model; standardise model names. | The pre-2015 corpus question is answered in `LINEAGE_VALIDITY.md` (95.6% of observed replacements appear in it; the advantage holds on those absent). Others open. |

## Errors in the preprint found by the archive's re-computation

From `tmp_explore/out/CONSOLIDATED.md` (the archive's control ladder; read it before quoting any
headline number):

* **Two PR-AUC values in Fig 3B are transposed in the text.** Recomputed: HA-80 alone 0.272 (text 0.27,
  agrees); **accessibility alone 0.235 (text says 0.13)**; **base ESM-2 alone 0.130 (text says 0.23)**;
  HA-80 x accessibility 0.372 (0.37); base x accessibility 0.320 (0.32). The transposition reverses the
  sentence: accessibility on its own *beats* the base PLM at the binary task, where the text says the
  opposite. Fig 3C Pearson values agree (HA-80 0.232, base 0.105).
* **Fig 3C is scored on few effectively independent points** (Kish effective sample size of the frequency
  weights is far below the nominal count; archive analysis a10).
* **The lineages are not independent**: reference proteins are 96.6 to 99.7% identical, per-lineage PLM
  vectors correlate at 0.985 (fine-tuned) and 0.995 (base), and the effective number of independent
  lineages is **1.01 (design effect) to 1.12 (Cheverud-Nyholt): one, not five.**
* **About three quarters of the headline 0.652 is codon-distance stratification**, which itself rests on
  modelling a two-step change as the square of a one-step change. The base ESM-2 has no
  nucleotide-spectrum signal and prefers the SARS-CoV-2 matrix to influenza's.
* **Fine-tuning effect is mostly not virus-specific mutation.** Scoring five SARS-CoV-2 spike lineages on
  their real codons, 89% of the advantage disappears off influenza (reversing to -0.044 within one-step
  replacements). A separate, much smaller component (+0.043 [0.019, 0.071]) is influenza's spectrum.
* **The negative-alpha "isolates pure protein fitness" recommendation fails**: its top 100 candidates are
  100% three-nucleotide changes.
* **Directionality (archive test a3).** Splitting ordered amino-acid pairs into symmetric and
  antisymmetric parts, both models carry directional information and the fine-tune more
  (antisymmetric r 0.556 vs 0.339, paired difference 0.217 [0.035, 0.400]), but with codon geometry
  removed and only the spectrum left the fine-tune's edge is not distinguishable from zero
  (0.097 [-0.046, 0.246]). On the ADAR signature in single-step changes, the base model points the wrong way
  (-0.051) and the fine-tune is weakly right (+0.024) against a truth of +0.197. **This session's tests
  (`SYMMETRY_TESTS.md`) agree and add a calibrated crossover for SARS-CoV-2.**
* **Apparent arithmetic error in Supplementary Table SX2 (Page 55):** A reviewer noted that out of
  8,099 corpus sequences, HA1:K2N (372 sequences) was listed as 5.5% while HA1:N158D (368 sequences)
  was listed as 4.5%, noting $372 / 8,099 = 4.59\% \approx 4.6\%$. The root cause is denominator
  variation across sites: K2N sits at mature position 2 (precursor site 18), immediately adjacent to
  the 16-residue signal peptide boundary where 1,342 corpus sequences are truncated/unaligned, leaving
  6,757 readable sequences ($372 / 6,757 = 5.505\% \approx 5.5\%$). In contrast, at N158D 8,098 sequences
  are readable ($368 / 8,098 = 4.54\% \approx 4.5\%$). The table footer and caption have been updated to
  explicitly define the site-specific denominators alongside the full-corpus percentages (4.6% of all
  8,099 sequences for K2N).

Claims the archive judges to survive: PLM pseudo-probabilities do contain mutational-supply information
and fine-tuning increases it; within single-step replacements the fine-tune tracks a nucleotide-biased
accessibility better than a flat one by 0.158 and the base model does not; fine-tuning flips the ADAR
signal from significantly wrong to weakly correct (paired shift +0.074).

## Preprint vs revision figures

| Item | In the preprint? | Notes |
|---|---|---|
| Scatter of PLM probability vs accessibility, base and fine-tuned (Fig 2A, B) | **Yes** | Frozen panels in `data/frozen_panels/` (21 Sep render). |
| Epoch curve (Fig 2C) | **Yes** | No generator survives; title truncated. |
| Within-stratum correlations, Grantham/BLOSUM62, partial correlations (S4) | **No** | Revision-stage; `make_fig02.py` panel A, `make_sE_spectra_codon_space.py`. |
| Amino-acid direction analysis, pair-specific direction | **No** | Revision-stage; `make_fig02.py` panels B and C. |
| Alpha sweep on five lineages | **Yes** | PR-AUC and Pearson as main panels. The clean repo's Fig 3 uses **13** lineages and AUROC. |
| Alpha sweep on thirteen lineages, AUROC, PR-AUC in S2 | **No** | Revision-stage. |
| Epistasis scan (Fig 4) | **Yes** | Now supplementary (S7). |
| DMS fitness against the weighted score (new Fig 4), S10, S12 | **No** | Revision-stage; answers R1 point 7 and R2. |
| Codon-learning / synonymous-codon analyses | **No** | Revision-stage; see `LINEAGE_VALIDITY.md`. |
| Compression analysis: 84.8% and 43.6% | **Yes** (as two numbers in the text, no figure) | Both are cells of S5's matrix and are reproduced to the digit printed. |
| Cross-species spectra, expressivity vs identity (S4 (withdrawn), S5) | **No** | Revision-stage; answers R1 point 1. The preprint compared one virus against K80 only. |
| Spectrum swap within strata (S6) | **No** | Revision-stage; partial answer to R2's crossover request. |
| Contact recovery by region and at antigenic sites (S13) | **No** | Revision-stage. The antigenic comparison is a null; it was S8 and is now S13 panel D, sites pooled, zero-sum gauge only. |
| Lineage-set validity audit (S14) | **No** | Revision-stage; built to answer whether the thirteen-lineage set supports what is drawn from it. |

Corrections this forces on earlier notes:

* "Published Fig 2b had no intervals" (progress log 9 Oct) is wrong: Fig 2b was never in the
  preprint. It is a revision-stage figure whose default drew no intervals.
* "Published strata correlations" in `figures/README.md`: these are archive (revision-stage)
  numbers, not preprint numbers.
* The five-lineage guide corresponds to the preprint's lineage set (G.1, J, J.2, J.2.4, K). The
  preprint's own counts are above; the file names `J_int` and `J.2_int` are the internal-node roots
  the archive used for J and J.2.
* The preprint's Fig 3 Pearson values (HA-80 0.228) are on five lineages and need not equal this
  repo's thirteen-lineage Fig 3C (about 0.115). They have not been reconciled.
