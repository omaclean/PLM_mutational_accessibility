# Lineage Definitions: H3N2 J.2 to K Lineage Transition and Mutation Set

**Document status:** Authoritative lineage definition reference (opened 20 Oct 2026).  
**Scope:** Reconciles the J.2 parental lineage root (`J.2_int`), the emergent K lineage (`K`), the Nextstrain/Nextclade phylogenetic hierarchy, the Pangolin nomenclature analogy, the 9 true clade-defining mutations, and resolves the `N145S` reference artifact.

---

## 1. Executive Summary

1. **The J.2 to K divergence is defined by exactly 9 amino acid substitutions:**
   * **8 in HA1:** `K2N`, `T135K`, `S144N`, `N158D`, `I160K`, `Q173R`, `K189R`, `T328A`
   * **1 in HA2:** `HA2:S49N` (precursor codon 394; mature precursor residue 378; mature HA2 residue 49)
2. **`N145S` was a phantom 10th mutation — FIXED:**
   * Residue 145 is **Serine** in `J.2`, `J.2.4` and `K` alike (precursor codon 161 = `agt`),
     and in the A/Aichi/2/1968 H3-numbering reference. There is **zero** change at 145 on the
     J.2 → K path, in nucleotide or amino acid.
   * The label existed only in the frozen 2 Jul 2026 render
     `data/frozen_panels/fig1bc_ranked_mutations.pdf`. The data, the source run's own table
     and the current code all give nine. See section 6.
   * Site 145 is heavily homoplasic (73 branches in the Nextclade tree), so the label looked
     credible; the mechanism that injected it has **not** yet been identified.

3. **`J.2_int` is the canonical ancestral sequence:**
   * `data/roots/h3n2_5lineage/J.2_int.nt.fa` and `data/roots/h3n2_13lineage/J.2.nt.fa` translate to **100% identical amino acid sequences** across all 566 translated codons.
   * They differ by only a single synonymous nucleotide change at precursor codon 36 (`GTG` in 5-lineage vs `GTA` in 13-lineage, both encoding Valine).

---

## 2. What is the "Pangolin Definition"? (Lineage Nomenclature Explained)

In viral genomics, the term **Pangolin** refers to two distinct but connected concepts:

1. **SARS-CoV-2 Pangolin Nomenclature:**
   * Developed by Rambaut et al. (*Nature Microbiology* 2020), the **Phylogenetic Assignment of Named Global Outbreak LINeages (Pango / Pangolin)** system assigns hierarchical, dynamic labels to SARS-CoV-2 lineages (e.g., A, B, B.1.1.7, BA.1, JN.1, LP.8.1, XEC).
   * In `data/roots/sars_cov_2_spike/`, Pangolin lineages are tracked directly (`LP.8.1`, `JN.1`, `XEC`, `NB.1.8.1`, `Wuhan-Hu-1`).

2. **Influenza A/H3N2 "Pango-Style" Subclade Nomenclature:**
   * Historically, WHO and CDC classified influenza using clade designations based on phylogeny and antigenic characterization (e.g., `3C.2a`, `3C.2a1b.2a.2`).
   * To provide fine-grained, dynamic tracking during rapid evolution, Nextstrain (Bedford Lab, Neher Lab) introduced a **Pango-style hierarchical subclade system** for seasonal H3N2.
   * Under this system:
     - Major subclades of `3C.2a1b.2a.2` receive letter designations: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`.
     - Sub-lineages append numerical tiers hierarchically: `J` $\to$ `J.2` $\to$ `J.2.4` $\to$ `J.2.4.1`.
     - Alias rules prevent runaway naming depth: `J.2.4.1` is aliased directly to **`K`** (also designated `2a.2b`).
   * When researchers ask for the "Pangolin definition" of an H3N2 lineage, they are referring to this **Nextstrain Pango-style hierarchical definition**.

---

## 3. Nextstrain / Nextclade Lineage Hierarchy for J.2 to K

In the seasonal influenza A/H3N2 Nextclade dataset (`EPI1857216`, `tree.json`):

```
Root (A/Darwin/6/2021 backbone, 3C.2a1b.2a.2)
 │
 └──> G.1.3.1
       │
       └──> J (J_internal, 2a.2a)
             │   [Branch mutation: HA1:K276E]
             │
             └──> J.2 (J.2_internal / J.2_int)
                   │
                   ├──> NODE_0000825 [HA1:K189R]
                   │     │
                   │     └──> J.2.4 (J.2.4_internal) [HA1:T135K]
                   │           │
                   │           └──> NODE_0000842 [HA1:N158D, HA1:I160K, HA1:T328A]
                   │                 │
                   │                 └──> NODE_0000843 [HA1:S144N]
                   │                       │
                   │                       └──> K (K_internal / J.2.4.1 / 2a.2b) [HA1:K2N, HA1:Q173R, HA2:S49N]
```

Every one of the 9 substitutions maps to a documented internal branch in this phylogenetic progression.

---

## 4. The 9 True Clade-Defining Substitutions

All 9 replacements are single-nucleotide steps ($nt = 1$) under the standard genetic code.

| # | Region | Mature H3 Number | Precursor / Alignment Site | J.2 Codon | K Codon | AA Change | Genetic Code Steps | Phylogenetic Branch |
|---|---|---|---|---|---|---|---|---|
| 1 | **HA1** | **K2N** | 18 | `aaa` | `aac` | K $\to$ N | 1 nt | Node 843 $\to$ K_internal |
| 2 | **HA1** | **T135K** | 151 | `aca` | `aaa` | T $\to$ K | 1 nt | Node 825 $\to$ J.2.4_internal |
| 3 | **HA1** | **S144N** | 160 | `agt` | `aat` | S $\to$ N | 1 nt | Node 842 $\to$ Node 843 |
| 4 | **HA1** | **N158D** | 174 | `aac` | `gac` | N $\to$ D | 1 nt | J.2.4 $\to$ Node 842 |
| 5 | **HA1** | **I160K** | 176 | `ata` | `aaa` | I $\to$ K | 1 nt | J.2.4 $\to$ Node 842 |
| 6 | **HA1** | **Q173R** | 189 | `caa` | `cga` | Q $\to$ R | 1 nt | Node 843 $\to$ K_internal |
| 7 | **HA1** | **K189R** | 205 | `aag` | `agg` | K $\to$ R | 1 nt | J.2 $\to$ Node 825 |
| 8 | **HA1** | **T328A** | 344 | `acc` | `gcc` | T $\to$ A | 1 nt | J.2.4 $\to$ Node 842 |
| 9 | **HA2** | **HA2:S49N** | 394 | `agt` | `aat` | S $\to$ N | 1 nt | Node 843 $\to$ K_internal |

---

## 5. Numbering Conventions

Influenza Haemagglutinin (H3 subtype) carries a **16-amino acid signal peptide** that is post-translationally cleaved:
* **Precursor Sites 1–16:** Signal peptide (`MKTIIALSYILCLVFA`).
* **Precursor Sites 17–345 (329 residues):** Mature **HA1** domain.
  $$\text{Mature HA1 Position} = \text{Precursor Position} - 16$$
* **Precursor Sites 346–566 (221 residues):** Mature **HA2** domain.
  $$\text{Mature HA2 Position} = \text{Precursor Position} - 16 - 329 = \text{Precursor Position} - 345$$
  * For example, precursor site 394:
    $$\text{Mature HA2 Position} = 394 - 345 = 49 \implies \mathbf{HA2:S49N}$$
  * In unsegmented mature precursor numbering (where HA1 is 1–329 and HA2 is 330–550):
    $$\text{Mature Residue Number} = 394 - 16 = 378 \implies \mathbf{S378N}$$

---

## 6. Post-Mortem on the `N145S` Phantom Mutation

**Status: RESOLVED.** `output/fig01_composite.pdf` ships exactly the canonical nine.

### 6.1 What is established (verified 20 Oct 2026)

1. **The data has always been right.** Residue 145 (precursor codon 161) is `agt` = **Serine**
   in `J.2`, `J.2.4` and `K`, in both `data/roots/h3n2_5lineage/` and
   `data/roots/h3n2_13lineage/`. It is also Serine in the A/Aichi/2/1968 canonical
   reference used for H3 numbering (`Sequences/H3N2_canonical.fa`, repo root).
   No reference file in either this repo or the predecessor
   `PLM_SARS-CoV-2/Sequences/IAV_lineage_files/` carries `N` at 145.
2. **The source run's own table is right.** Rebuilding `focal_ref_seq` from
   `results/Lytras_OG/tables/combined_long_table.csv` (repo root) and diffing it against that
   run's `K` reference yields **exactly the nine** replacements — not ten.
3. **The current code is right.** Replaying the live path
   (`get_mutations` → `create_h3_numbering_map` → `mutations_to_canonical`,
   `scripts/run_mutational_accessibility.py`) reproduces the nine with correct mature-H3
   numbering and no site-145 entry.
4. **The frozen 2 Jul render had two discrepancies.** First, `data/frozen_panels/fig1bc_ranked_mutations.pdf`
   (2 Jul 2026) marked ten points, `N145S` among them at rank 669 (PLM panel) and 551
   (accessibility panel). Second, the 2 Jul and 13 Oct renders plotted the **base unmasked** model
   (`ESM2_650M_HA80_raw`), whereas the manuscript text (Page 9) described the **fine-tuned** model
   (`ESM2_650M_HA80`), where HA2:S49N ranks 7th ($p \approx 0.904$) and T328A ranks 578th ($p \approx 4.7 \times 10^{-4}$).
   `guides/composites/fig01.yaml` now places the fine-tuned model panel
   `data/frozen_panels/fig1bc_ranked_mutations_finetuned_esm2_ha80.pdf` (derived from
   `results/Lytras_OG/plots/per_model/ESM2_650M_HA80/ranked_mutations_j2_int_vs_k_lineage.pdf`),
   which marks exactly the nine and aligns with the manuscript Results text.

### 6.2 Why it was plausible enough to survive review

Site 145 is among the most homoplasic positions in H3 HA. In
`data/nextclade_h3n2_ha/tree.json` **73 branches** carry a mutation at 145, and the
state flips S↔N↔G↔K repeatedly. `N145S` is a genuine label on several *ancient* branches
(e.g. `NODE_0000051`, `NODE_0000059`, A/Victoria/512/2005) — just nowhere on the J.2→K path.
A spurious 145 label therefore reads as entirely credible to a reader who knows the site.

### 6.3 Root cause (established 20 Oct 2026)

**A stale focal reference was diffed against a live comparison reference.**

`_write_lineage_comparison_ranked_plot` builds the two sides of the diff from sources of
different provenance and different vintage, and never checks they agree:

```python
focal_ref_seq      = "".join(focal_positions_df["ref_aa"].astype(str).tolist())   # from the STORED results table
comparison_ref_seq = _load_reference_protein_from_path(...["reference_path"])     # read LIVE from the FASTA
```

The `J.2_int` root FASTA was corrected at **precursor codon 161** on 15 Jun 2026 — `aat`
(N) became `agt` (S) — and the stored tables were not rebuilt. The file is untracked in
the predecessor repo, so the pre-correction content survives only inside those tables:

| Source | Vintage | Precursor 160 | Precursor 161 |
|---|---|---|---|
| `esm2_flu_full_AdamW/tables/combined_long_table.csv` | 1 May 2026 | S | **N** |
| `esmc_flu_full_job4154992_epochX/.../combined_long_table.csv` | 8 May 2026 | S | **N** |
| `J.2_int.nt.fa` after the 15 Jun correction, and `data/roots/h3n2_5lineage/J.2_int.nt.fa` | 15 Jun 2026 | S | **S** |
| `results/Lytras_OG/tables/combined_long_table.csv` | 3 Jul 2026 | S | **S** |

`K` carries `aat agt` at codons 160–161 = **N160, S161**. So:

* stale focal `S160 N161` vs K `N160 S161` → **two** calls: `S160N` *and* `N161S` → `S144N` + **`N145S`** = **ten**
* corrected focal `S160 S161` vs K `N160 S161` → **one** call: `S160N` → `S144N` = **nine**

The 2 Jul 2026 render read a 1 May table against the then-current K FASTA, and so drew ten
points. Nobody fixed it: `results/Lytras_OG/` was re-run on 3 Jul from the corrected FASTA,
and its panel has had nine ever since. The defect survived only in the frozen copy.

**Verified independently of the above table:** the published panel's PLM ranks
(23/43/153/242/669/968/1358/1503/2291/5746) are those of the **base** ESM2, and the
corrected 13 Oct render gives 24/31/152/243/958/1350/1531/2301/5768 — the same computation
perturbed by a single residue. No model or code difference is involved.

**The structural defect is the missing check, not the sequence edit.** Nothing warns when a
stored table's `ref_aa` no longer matches the reference FASTA on disk. Any re-plot of an
older table against newer roots can silently invent or erase replacements.

### 6.4 Required actions

| # | Action | State |
|---|---|---|
| 1 | Stop shipping the defective panel in Figure 1 | **done 20 Oct 2026, updated 21 Oct 2026** — `guides/composites/fig01.yaml` now places `fig1bc_ranked_mutations_finetuned_esm2_ha80.pdf` (the fine-tuned model regeneration `ESM2_650M_HA80`, aligning with the Results text, with baked-in letters B/C redacted so `panelise.py` draws house headers). Crops retuned: `crop_top_frac` 0.094 → 0.108, gutter 0.537 → 0.4958. |
| 2 | Correct every "ten"/"10 replacements" count in captions, taglines and prose | done for `guides/composites/fig01.yaml` and `data/frozen_panels/README.md` |
| 3 | Regression test over *rendered* figures, not just tables | **open** |
| 4 | Find the mechanism that injected the point | **done 20 Oct 2026** — see 6.3: stale table `ref_aa` vs live FASTA. |
| 5 | Guard the stale-reference mismatch in code, with a test | **done 20 Oct 2026** |

**Note for the revision.** The corrected panel's ranks differ from the preprint's by a few
places (e.g. I160K 24 vs 23, S144N 31 vs 43, T135K 5768 vs 5746). This is expected and is
not a separate error: the published ranks were computed over a candidate set built from a
J.2 reference with the wrong residue at 145. The rank change should be stated in the
response to reviewers alongside the corrected count of nine.

## 7. Automated Regression Tests

The integrity of these definitions is enforced in `tests/test_lineage_definitions.py`:
* `test_j2_int_vs_k_exact_nine_mutations`: Asserts that `J.2_int` and `K` differ by exactly 9 amino acid substitutions (8 in HA1, 1 in HA2).
* `test_site_145_is_identically_serine`: Asserts that codon 161 (mature 145) is `S` in both J.2 and K, and that no `N145S` mutation exists.
* `test_j2_int_5lineage_and_13lineage_protein_identity`: Asserts 100% protein identity between the 5-lineage and 13-lineage J.2 roots.
* `test_all_nine_mutations_are_single_nucleotide_steps`: Asserts every mutation is a 1-step change under the genetic code.
* `test_mature_numbering_conversion`: Asserts mathematical consistency of mature HA1 and HA2 coordinates.
* `test_epistasis_focal_sites_exact_membership`: Asserts `epistasis_focal_sites.tsv.gz` contains exactly the 9 mutations.
