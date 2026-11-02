# Lineage-set validity and the codon-learning contradiction

**A separate workstream. Its findings must not change the design of `clean_repo`.**

Per the governing principle in [`docs/notes/PLAN.md`](docs/notes/PLAN.md) §1, the clean repo is a
question-agnostic figure engine: it runs off any guide file pointing at anything. This
document holds the *scientific* questions that the engine must stay neutral about.

**Status:** Q1–Q3 answered 9 Oct 2026 — see §4. Report at
`dev/lineage_validity/report.html`. Three follow-ups still open.
**Opened:** 2026-10-08.

**Reference methodology:**
The preprint of this manuscript (MacLean, Lamb, Lytras, Mojsiejczuk, Yuan, Hughes & Robertson,
bioRxiv `10.64898/2026.07.06.736753`, posted 9 July 2026). Anything done here must be
standardised against the methods already published there, not invented afresh.

---

## 1. The three questions

### Q1 — Does the synonymous-codon result survive?

Two analyses of closely related hypotheses reach opposite conclusions.

| | Strand B | Strand C |
|---|---|---|
| Code | `tmp_explore/analyses/a20_codon_randomisation.py` → `build_s4_figures.py` | `results/mutation_codon_learning/synonymous_permutation/run_test.py` |
| Lineages | **5** | **13** |
| Run | 8 Sep 2026 | 28 Sep 2026, seed 20260928 |
| Design | conservation-stratified null | pre-specified, permutation null |
| Result | **Positive.** AUC 0.608 [0.572, 0.645] vs null 0.520; 46/54 pairs > 0.5; sign test p = 1.4e-7. Base model sits on its null (0.511 vs 0.511, p = 0.47). | **Null.** *"The omnibus test finds nothing, the fine-tuned model is no better than the untuned base."* Pre-specified amino acid not significant on the choice-free statistic (p = 0.13). |

The difference is confounded: **lineage set and statistic changed together.** The
first job is to separate them.

- [x] **Tiebreak A** — apply strand C's statistic to strand B's 5-lineage data.
- [x] **Tiebreak B** — apply strand B's statistic to strand C's 13-lineage data.
- [x] Conclude: is the disagreement driven by the lineage set, or by the test?

Note in strand B's favour: its design argument is genuinely strong. Holding the amino
acid fixed is a *better* control than partialling Grantham out, because within a
synonymous stratum BLOSUM62, Grantham and every other 20×20 exchangeability matrix
have variance **exactly zero** — there is nothing left to partial. That argument is
sound regardless of which way the result lands.

Note against strand B: its pipeline is broken in two places. `build_s4_figures.py`
calls an `insert_s4_figures.py` that does not exist, and `codon_or_amino_acid.html` is
hand-authored with no builder. Its `_s4_*.svg` outputs are hand-copied and go stale
silently — the extracted SVGs are dated 5 Oct while the underlying `a20_s4_*.tsv` are
8 Sep. **Any re-run must rebuild the tables, not trust the SVGs.**

### Q2 — Is the 13-lineage set actually independent?

The working assumption is that 13 lineages licence an across-lineage confidence
interval. That rests on independence, and the evidence is mixed.

**What is established:**
- Each sequence is assigned to at most one lineage, by nearest root within a hard
  distance cut. The 13 sets are disjoint — **78,659 sequences, 78,659 distinct
  accessions, zero shared.** Verified.
- Clade C was deliberately dropped because its Nextclade tips start 2013-12-31, inside
  the pre-2015 corpus HA-80 was fine-tuned on. Step 5.1 of `run_all_figures.sh`
  re-checks this.

**What is not:**
- 🔴 **Residual outcome co-occurrence rose from 0.328 to 0.391** going from 5 to 13
  lineages, and is unchanged by controlling for depth (0.391 [0.369, 0.412]).
  *Disjoint sequences do not imply independent outcomes.* The lineages are
  near-duplicate backbones, so the same replacement can be "independently" observed in
  several of them. This directly limits how an across-lineage band may be described.
- **Prevalence ranges 2.0% to 10.9%** between lineages, which is why raw PR-AUC is not
  comparable across them and why the metric choice matters.

- [x] Quantify the effective number of independent lineages, not the nominal 13.
- [x] Decide what the across-lineage interval may legitimately be *called* in the text.
- [ ] Re-check the depth/downsampling control (a6) at 13 lineages. Known: at a common
      depth of 229 sequences, 27% of the fine-tuned PR-AUC gain survives, against 26%
      in the published five-lineage version.

### Q3 — Is there training-data leakage?

- 🔴 **The HA-80 pre-2015 fine-tuning corpus is public, and all nine K-lineage
  residues are in it.** `I160K` in particular is a *reversion to the historical
  majority*, not a novel escape. A model that "predicts" it may be recalling its
  training distribution rather than generalising.
- **Three of the five original lineage roots carry tip accessions but hold inferred
  ancestral sequence.** They are reconstructions, not observed sequences. Check
  whether the same is true across all 13, and that nothing downstream treats a root as
  an observed tip.

- [x] Audit every lineage root: accession vs reconstructed — **done, see §4 Q3b**.
      *(Date-vs-cutoff half still outstanding.)*
- [~] For each headline replacement, state whether it appears in the HA-80 corpus.
      *(Aggregate done — 95.6% of rows, 90.7% distinct. Per-replacement list not yet made.)*
- [x] Re-run the key comparison on the subset of replacements absent from the corpus.

---

## 2. Data-integrity tests to build

The user's instruction was *"the input data and the code get tested like crazy."*
These belong in `clean_repo/tests/test_data_integrity.py` even though the *questions*
above do not belong in the repo — the engine must be able to prove its inputs are sane
whatever guide it is pointed at.

- [ ] Every sequence appears in exactly one lineage; accessions unique; count matches
      the guide.
- [ ] **Guide carries `root_type: reconstructed | observed`**, and a test fails when
      a root header carries an accession while the guide calls it reconstructed.
      (Added 9 Oct from the Q3b audit.)
- [ ] **Guide-internal label agreement:** the lineage named in a root's FASTA header
      must match the lineage key the guide files it under. `K.nt.fa` currently fails
      this — its header says `J.2.4.1`.
- [ ] No lineage's tip dates precede the model's training-corpus cutoff, where the
      guide declares one.
- [ ] Reading frame intact: every CDS length divisible by 3, no internal stops.
- [ ] Lineage roots: flag reconstructed vs observed, and fail if a guide calls a
      reconstruction an accession.
- [ ] Prevalence reported per lineage, never pooled silently.
- [ ] Alignment columns map to the declared reference without an off-by-one; assert
      the frame explicitly rather than checking total width modulo 3.
- [ ] HA1/HA0 numbering: the `+16` offset is applied exactly once. (`dev/fig4_gauge`
      already has a test pinning this; port it.)
- [ ] Golden-value test: a fixed small guide reproduces known table values.

---

## 3. Agent brief

When this is picked up, the agent should:

1. Read the preprint PDF methods section first and record the published definitions of
   lineage assignment, the accessibility score, and the alpha sweep. **Standardise to
   those**; do not re-derive.
2. Run tiebreaks A and B (§1 Q1) and report which factor drives the disagreement.
3. Quantify effective independence (§1 Q2) and propose wording for the interval.
4. Audit leakage (§1 Q3).
5. Report findings here, in §4. **Do not modify anything under `clean_repo/`.**

Useful starting files:

| Path | What it is |
|---|---|
| `results/mutation_codon_learning/synonymous_permutation/README.md` | Strand C's own write-up, including why one hand-pooled p = 0.037 is not independent support |
| `tmp_explore/analyses/a20_codon_randomisation.py` | Strand B, 113 KB, `--draws 200 --perm 500 --boot 200 --seed 20260904` |
| `run_all_figures.sh` §5 | How the 13-lineage set was built, and the independence check |
| `scripts/check_lineage_temporal_overlap.py` | Existing temporal-overlap checker |
| `Sequences/IAV_lineage_guide_nextstrain.csv` | The 13-lineage guide |

⚠️ `grep` at the repo root is gitignore-aware and **silently skips `tmp_explore/`**.
Use `command grep`.

---

## 4. Findings — 9 Oct 2026

Full write-up and figure: **`dev/lineage_validity/report.html`** (old repo, as
instructed). Scripts and tables in `dev/lineage_validity/`. Emailed 9 Oct.

> ⚠️ **The figure was S14; it is being dropped from the supplementary set** (decision
> 15 Oct 2026, Oscar). The manuscript draft has no slot for it, and draft S14 is the
> simulation control. The generator and its tests are retained and retire to
> `dev/retired_figures/`; the findings in this section stand and are unaffected. See
> [`docs/notes/extended_red_team_tests.md`](docs/notes/extended_red_team_tests.md) §A1b.

### Q1 — the two codon analyses never disagreed ✅

They test different hypotheses. Running **each statistic on both lineage sets**
removes the confound, and neither flips.

| Statistic | 5-lineage | 13-lineage |
|---|---|---|
| **Strand B** — within-replacement AUROC, fine-tune | 0.6088 [0.5710, 0.6387] | **0.6104 [0.5733, 0.6414]** |
| **Strand B** — base model | 0.5106 [0.4852, 0.5407] | 0.5105 [0.4875, 0.5406] |
| **Strand C** — mean within-residue rho, fine-tune | −0.030, p = 0.27 | −0.037, p = 0.22 |
| **Strand C** — base model | −0.026, p = 0.30 | −0.046, p = 0.16 |

- Strand B **replicates at 13 lineages**, essentially unchanged, with a
  site-clustered interval clear of chance. The base model includes 0.5 in both.
- Strand C is **null on five lineages too**, where B is strongly positive. On 13
  lineages the *base* model is more negative than the fine-tune, so there is no
  fine-tuning effect in either direction.
- The reimplementation reproduces the archive's 5-lineage number (0.6079 → 0.6088; revision-stage, not in the preprint)
  over the same **54 scored pairs**, which validates it.

**Conclusion: strand C's null does not refute strand B.** The earlier reading in
`docs/notes/PLAN.md` — that the 28 Sep null killed codon learning — was wrong.

⚠️ Strand B's *figure chain* is still broken: missing `insert_s4_figures.py`,
hand-copied SVGs that go stale silently, tables dated 8 Sep against SVGs extracted
5 Oct. **Rebuild the tables from source before drawing anything.**

### Q2 — the lineages are not independent for this purpose ⚠️

Disjoint in sequences (78,659 / 78,659 distinct accessions) but they share sites.

| Measure | 5-lineage | 13-lineage |
|---|---|---|
| Identical reference codon in every lineage | 520 (91.9%) | 451 (79.7%) |
| Identical reference amino acid in every lineage | 546 (96.5%) | 515 (91.0%) |
| Mean between-lineage Spearman of model probability | 0.985 | 0.968 |
| Range of per-lineage codon AUROC (fine-tune) | 0.0055 | 0.0167 |

Effect on the interval — resampling **lineages** (what Fig 3 does) vs **sites**:

| Model | Mean AUROC | Resampling lineages | Resampling sites | Ratio |
|---|---|---|---|---|
| Fine-tune | 0.9195 | [0.9139, 0.9249] | [0.9047, 0.9165] | 1.1× |
| Base | 0.7721 | [0.7658, 0.7777] | [0.7454, 0.7808] | **~2.5×** (2,000 draws; 3.0× from the original 500-draw run was partly noise, 2.3–2.4× at 10,000) |

Not uniform, which is the interesting part: the band is about right for the
fine-tune and **3× too narrow for the base model**, so the band on the *difference*
between models is the one to distrust. For the codon statistic the gap is ~13×.

**Action:** report site-clustered intervals where a generalisation claim is made, or
say plainly that the band is the precision of the mean over thirteen near-duplicate
backbones — not generalisation to an unseen lineage. Consistent with the existing
a18 result (residual co-occurrence 0.391 [0.369, 0.412]).

### Q3 — no meaningful training-corpus leakage ✅

HA0→mature mapping confirmed empirically: parquet position *p* ↔ backbone residue
*p−16*, at **100% identity over 550 positions**.

95.6% of observed replacement rows (90.7% of distinct) do appear in the
8,099-sequence HA-80 corpus. But the advantage does not depend on that:

| Replacement | cells | observed | Fine-tune | Base | Advantage [95% CI] |
|---|---|---|---|---|---|
| **Absent from corpus** | 5,129 | 160 | 0.8889 | 0.7460 | **+0.143 [+0.107, +0.182]** |
| Present in corpus | 5,369 | 1,555 | 0.8655 | 0.6843 | +0.181 [+0.163, +0.201] |

The fine-tune reaches **0.889 AUROC on replacements it has never seen**, with the
advantage interval comfortably clear of zero. That is generalisation, not recall,
and it answers a likely reviewer objection with a number.

Caveat: only 160 observed replacements fall outside the corpus, at 0.4% prevalence
against 9.0% inside. AUROC is prevalence-invariant in expectation so the comparison
is fair, but the out-of-corpus estimate is the noisier one.

### Incidental finding — environment fragility

Only `/home3/oml4h/miniconda3/envs/PRESCOTT/bin/python` on this machine has both
`pyarrow` and `scipy`. The `plm_entropy`, `plm_sars` and `plm_esmc` envs cannot read
the parquets at all. Record this in `environment.yml`.

### Q3b — lineage-root provenance ✅ (13-lineage set is clean)

Audited 9 Oct by reading every root FASTA header.

**All thirteen nextstrain roots are honestly labelled as reconstructions:**

```
>C.1|HA|nextstrain_clade_root|ref=A/Darwin/6/2021|nuc_mutations_from_root=629
```

Every one carries `nextstrain_clade_root`, the same reference
(`A/Darwin/6/2021`) and an explicit `nuc_mutations_from_root` count (629–701,
rising monotonically down the clade ladder, as it should). **None claims a tip
accession.** The concern does not apply to the set actually in use.

**The original five are where the problem lives.** Their headers carry GISAID tip
accessions with no marker that the sequence is inferred:

| Root file | Header |
|---|---|
| `G.1.nt.fa` | `EPI4551091\|HA\|A/ENG/69/2025\|EPI_ISL_UNKNOWN\|G.1` |
| `J_int.nt.fa` | `EPI2178977\|HA\|A/Thailand/8/2022\|EPI_ISL_14991375\|J` |
| `J.2_int.nt.fa` | `EPI2990337\|HA\|A/District_Of_Columbia/27/2023\|EPI_ISL_18862356\|J.2` |
| `J.2.4.nt.fa` | `EPI4551140\|HA\|A/England/415/2024\|EPI_ISL_20080368\|J.2.4` |
| `K.nt.fa` | `EPI4748783\|HA\|A/England/01837755/2025\|EPI_ISL_20210731\|**J.2.4.1**` |

Two further problems in that set:

1. 🔴 **`K.nt.fa`'s header says lineage `J.2.4.1`, not `K`.** Either the file is
   mislabelled or the lineage was renamed and the header never followed. Resolve
   before any five-lineage number is quoted again.
2. 🔴 **The five-lineage guide points into the old repo** —
   `/home3/oml4h/PLM_SARS-CoV-2/Sequences/...` — so it cannot be run from this
   repository at all. The `_int` suffix on two entries is the only hint that they
   are internal-node reconstructions.

**Action for the clean repo:** the guide schema must carry an explicit
`root_type: reconstructed | observed` field, and a test must fail when a root
header carries an accession while the guide calls it reconstructed. This is
exactly the kind of thing §2 of this document is for.

### Preprint methods, recorded 9 Oct 2026

* Lineage assignment: a sequence belongs to a lineage if it has all the lineage-defining mutations and
  **fewer than five additional amino-acid replacements** from the lineage root; sequences with more than two of
  the next lineage's defining nucleotide mutations are dropped. Counts: G.1 229, J 4,132, J.2.4 877, J.2
  27,452, K 17,898 (five lineages). The thirteen-lineage set uses Nextstrain clades with a nearest-root
  distance cut, which is **not** the preprint's rule; the two have not been reconciled.
* Accessibility: Gunnarsson and Babu (2023) with Pauly et al. (2017); score = PLM_prob x accessibility^alpha;
  alpha applied per lineage, metrics averaged across lineages.
* Corpus: HA-80 fine-tuned on influenza A HA up to February 2015, 99% identity clusters.
* Reviewer 2 asked whether any of the nine K replacements are in the pre-2015 training data: answered in Q3 above.

### Still open
- [ ] Standardisation against the preprint methods section (partly done: see the block above). These analyses follow
      the code as written, not the published description.
- [ ] Clade C was dropped because its tips start 2013-12-31, inside the pre-2015
      corpus. The equivalent date check has not been repeated for the other twelve.
