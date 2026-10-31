# Lineage reassignment: applying the published rule to all 13 lineages

**Status:** S1–S6 and S8 complete; **D-REASSIGN-1 resolved 22 Oct 2026 (§4.2); S7 and S9 open.**
Opened 22 Oct 2026.
**Owner question:** [`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md) §4 Q2, and the Methods
discrepancy recorded at `LINEAGE_VALIDITY.md:293-297`.
**Governing constraint:** [`docs/notes/PLAN.md`](PLAN.md) §1 — the clean repo stays a
question-agnostic figure engine. Nothing here changes `mutacc/` or `figures/` behaviour;
it changes two *input tables*.

---

## 1. Why

The thirteen-lineage set shipped in `guides/h3n2_13lineage.yaml` was **not** built with the
published rule. It used Nextstrain clade labels with a nearest-root distance cut. The
preprint's rule is a mutation-defined ancestral-proximity rule. The two are not the same
construction, and on the one lineage where they can be compared cleanly the difference is
large and in a direction that is a genuine misassignment, not a nuance:

| Lineage | Published rule (5-set) | Nearest-root (13-set) | Change |
|---|---|---|---|
| G.1 | 229 | 779 | ×3.4 — root also redefined, not comparable |
| **J** | **4,132** | **2,489** | **−40%** |
| J.2 | 27,452 | 25,841 | −6% |
| J.2.4 | 877 | 964 | +10% |
| K | 17,898 | 19,492 | +9% |
| Total (shared five) | 50,588 | 49,565 | −2% |

Under nearest-root, sequences sitting between J and J.2 are pulled into J.2 because J.2's
root is nearer. The published rule screens them against J.2's *defining mutations* instead,
and keeps them in J. J is the lineage that loses, because J has the most descendants.

**Decision: re-run with the published rule, generalised to the 13-lineage tree.**

---

## 2. What is already confirmed

| Item | Finding |
|---|---|
| Assignment code | `/home3/oml4h/PLM_SARS-CoV-2/Notebooks/OM_influenza/build_lineage_subalignments.py` (806 lines) |
| Reproduces published counts | Yes — `lineage_subalignment_summary_hard_nextle2.tsv` gives 229 / 4,132 / 27,452 / 877 / 17,898 exactly |
| Required setting | `MAX_MUTATIONS = 5` (file currently reads 10; the published run used an uncommitted local edit) |
| Other settings | `ASSIGNMENT_MODE = "hard"`, `HARD_MAX_NEXT_MUTATIONS = 2`, `FINAL_LINEAGE_MAX_MISSING_CURRENT_BRANCH = 1` |
| Query alignment | `/home4/lm305z/IAV_DB/flu_vgtk_integrations/tmp/Protein-alignment/sgt_4_HA_AA.fasta` (233 MB, AA, anchor `PV511679`) — readable |
| Defining mutations | Not stored. Derived at runtime as pairwise diffs between consecutive roots in an ordered ladder FASTA |
| **GPU needed** | **No** — see §6 |

### 2.1 The rule as actually implemented

`passes_mode_criteria()` plus a Hamming cap in `main()`:

1. carries **all** of its own lineage's defining mutations (terminal lineage may miss ≤1);
2. carries all *cumulative* root→lineage states;
3. Hamming distance to the lineage root `<= MAX_MUTATIONS`;
4. carries `<=` `HARD_MAX_NEXT_MUTATIONS` of the **next** lineage's defining mutations;
5. among all references passing, assigned to the one with **minimum** Hamming distance.

### 2.2 Two things the Methods text gets wrong, independent of this rebuild

- The code tests `best_distance > MAX_MUTATIONS`, i.e. **no more than five** additional
  replacements. The manuscript says "fewer than five".
- The next-lineage test runs on **amino acids**. The manuscript says "defining
  **nucleotide** mutations".

Both need correcting in the Methods whichever rule ends up being used.

### 2.3 `FINAL_LINEAGE_MAX_MISSING_CURRENT_BRANCH` is narrower than its name

The terminal lineage is allowed to miss ≤1 of its own defining mutations. But
`passes_mode_criteria` then applies `has_all_mutations(query, cumulative)`
**unconditionally**, after the leniency. For a site that changes once down the path, the
net head→lineage state equals the branch state, so the cumulative check immediately
re-imposes what the leniency waived.

The leniency therefore only bites where a site changes **more than once** down the path,
so that the net state differs from the branch state — a reversion, in practice. Verified
against the original source and covered by two tests.

Consequence for the rebuild: "terminal" generalises from *one* lineage (K) to the six
childless lineages (C.1, D, E.2, G.1.1, G.2.2, K). Because the cumulative check nullifies
the leniency for ordinary sites, this should change very little — but it is a behaviour
change and belongs in the Methods.

---

## 3. The topology

The published rule assumes a single linear ladder: `next = order_index + 1`. The 13-lineage
set branches, so the ladder has to become a tree. Derived from three independent sources,
which agree.

### 3.1 Sources

1. **Root sequences** — `data/roots/h3n2_13lineage/*.nt.fa`, translated, pairwise AA
   distances. Additive-path test `d(O,P) + d(P,X) == d(O,X)`.
2. **UShER segment-4 tree** — `trees` table, `usher_refset_4_aln_merged_MSA_dedup`
   (9.9 MB newick, 572,119 tips) from
   `/home3/oml4h/RABV-gTK/test_out/IAV_DB_genbank_n_gisaid_rebuild/db_build/iav-db.db`,
   tips labelled from `nextclade_id_clade.tsv` (145,485 calls, 141,897 matched).
   Clade stem = deepest node holding ≥99% of that clade's tips; nesting read off a
   containment matrix. Stable at 90%, 95% and 99% thresholds.
3. **Published nomenclature** — `LINEAGE_DEFINITIONS.md` §3 (`J = G.1.3.1`, `K = J.2.4.1`).

### 3.2 Result

| lineage | parent | evidence |
|---|---|---|
| C.1 | — | outgroup branch; no ancestor among the 13 |
| D | — | outgroup branch; no ancestor among the 13 |
| E | — | head of the backbone within the 13 |
| E.2 | E | tree |
| G.1 | E | tree, via G (G is not one of the 13) |
| G.2 | E | tree, via G (G is not one of the 13) |
| G.2.2 | G.2 | tree |
| G.1.1 | G.1 | tree |
| G.1.3 | G.1 | tree |
| J | G.1.3 | nomenclature + root additivity (3 + 5 = 8) |
| J.2 | J | tree |
| J.2.4 | J.2 | nomenclature + root additivity (2 + 2 = 4) |
| K | J.2.4 | tree + the 7 documented defining mutations |

Children, which drive the next-test:

```
E      -> E.2, G.1, G.2
G.1    -> G.1.1, G.1.3
G.1.3  -> J
G.2    -> G.2.2
J      -> J.2
J.2    -> J.2.4
J.2.4  -> K
C.1, D -> (none)
```

### 3.3 Why the tree alone was not trusted

Two methods were tried and rejected before the one above:

- **Strict MRCA** misplaces J.2 (depth 129, swallowing C and D) and K. A handful of
  low-coverage tips placed near the root collapse the MRCA. Consistent with the known
  UShER defect where uncovered sites are read as the reference base.
- **Best-F1 clade node** returns the paraphyletic *remainder* of each clade, because
  Nextclade labels are mutually exclusive — a J sequence is labelled `J`, never `G.1.3`.
  It therefore finds no nesting at all.

The containment matrix appears to make J a sibling of G.1.3, and J.2 a sibling of J.2.4,
for that same exclusive-label reason. Root additivity is exact in both cases
(3 + 5 = 8; 2 + 2 = 4), so the nomenclature nesting is correct and the sibling reading is
the artefact.

---

## 4. The generalisation

Confirmed with the owner, 22 Oct 2026:

> **At a branch point, test `<=2` against *every* child's defining set, requiring all to pass.**

So for lineage *X* with children *c₁…cₙ*, a sequence is eligible for *X* when:

```
has_all(seq, defining[X])                      # X's own defining mutations
AND has_all(seq, cumulative[X])                # root -> X net states
AND hamming(seq, root[X]) <= 5
AND for every child c of X:  present(seq, defining[c]) <= 2
```

and is then assigned to whichever eligible lineage has the smallest Hamming distance.

`defining[X] = branch_mutations(root[parent[X]], root[X])`.

### 4.1 Known consequence to watch

C.1, D and E have no parent among the 13, so `defining[X]` is empty for them and the
first check is vacuous. C.1 and D additionally have no children, so the `<=2` test is
vacuous too — they are selected on `hamming <= 5` alone.

This is the same weakness the published G.1 bucket had: G.1 was the ladder root with no
defining mutations, and the purity check
(`lineage_summary_report.csv`) shows its 229 sequences were a mixed bag —
`G (106), G.1 (86), G.1.1 (36), G.3 (1)`. Expect C.1 and D to behave the same way.

**Open decision (D-REASSIGN-1):** whether to give C.1 / D / E defining mutations against
an outgroup root (A/Darwin/6/2021, or clade C) rather than leaving them empty. Not
blocking — the run can proceed and the purity be measured first.

**Measured, 22 Oct 2026 (S5).** The expectation above was half right. C.1 is *clean*
(0.0% contaminant, 7.5% ancestral C) — an empty defining set costs it nothing because
nothing else sits near it. **D (49.2%) and E (46.4%) are the problem**, both absorbing
clade B.1.2.x. So D-REASSIGN-1 narrows to D and E, and the options are:

1. **Give D and E defining mutations against an outgroup root.** Keeps all thirteen.
   Needs a defensible outgroup; the natural one is the clade the contamination comes
   from, which makes the choice circular unless it is fixed on phylogeny first.
2. **Drop D and E from the set**, reporting eleven lineages. Clean, costs two lineages,
   and changes the lineage count in the Methods and Fig 3 again.
3. **Keep them and declare the impurity.** Cheapest, but D and E would carry ~half their
   observed replacements from sequences that are not in the lineage, and both gain
   substantially in S6 (+40, +87) partly for that reason.

**Option 1 chosen, 22 Oct 2026.** See §4.2 — and note that the 49.2% / 46.4% figures
above were themselves partly an artefact; §4.2 corrects them.

### 4.2 D-REASSIGN-1 resolved — outgroup roots for D and E

**Half the problem was a measurement bug.** `summarize_purity.py` unaliased clade names
with a hand-maintained map holding only `J` and `K`. But the Pango-style names carry no
nesting of their own, and **`D` and `E` are aliases too**: read off the Nextclade
reference tree, `D`'s parent clade is **`B.1.2.1`** and `E`'s is **`B.1.2`** — precisely
the clades that had been counted as their top contaminants. Their own backbone was being
scored as foreign.

Nesting is now derived from `data/nextclade_h3n2_ha/tree.json` rather than from the
printed names (`dev/lineage_reassign/clade_nesting.py`). The derivation rediscovers both
documented aliases — `J → G.1.3.1`, `K → J.2.4` — without being told them, which is the
check that it is right. Rescoring the S5 run:

| lineage | contaminant (S5, alias map) | contaminant (tree nesting) | ancestral |
|---|---|---|---|
| D | 49.2% | **7.1%** | 42.0% |
| E | 46.4% | **24.0%** | 22.4% |

So D was mostly reaching back up its own backbone — exactly what an empty defining set
predicts — and only E had substantial foreign material.

**The fix, and why the outgroup is not circular.** The outgroup for each is its parent
clade *in the reference tree*, fixed on topology before looking at where the
contamination sat. Roots for `B.1.2` and `B.1.2.1` are reconstructed by the same
construction as the shipped thirteen (`dev/lineage_reassign/build_clade_roots.py`:
the tree's root sequence with every nucleotide mutation on the path to the clade's stem
applied, trimmed to the 1701 nt CDS). **Gate: it reproduces all thirteen shipped roots
exactly.**

The topology gains a `reference_only` column. Such a root supplies its child's
defining-mutation set and nothing else: it is never an assignment candidate, and the
cumulative head walk stops below it, so **every other lineage's cumulative set is
unchanged** and the five-lineage regression gate still passes exactly.

One further correction was needed. D is childless, so the terminal `≤1 missing`
leniency applies — and §2.3's point is that the leniency is only safe because the
cumulative check re-imposes what it waives. With the parent reference-only, `cumulative[D]`
came out empty and the leniency bit uncontrolled, admitting **`B.1.2.1.1`**, D's sibling,
whose root sits exactly 5 AA from D's root (the Hamming cap) and shares 3 of D's 5
defining mutations. Anchoring the cumulative set on the outgroup root when the parent is
reference-only restores the §2.3 invariant and removes them.

**Result — all thirteen lineages are clean:**

| lineage | n | correct | ancestral | contaminant |
|---|---|---|---|---|
| J.2 | 26,703 | 100.0% | — | — |
| K | 17,718 | 100.0% | — | — |
| G.1.1 | 12,597 | 100.0% | — | — |
| G.2 | 6,969 | 100.0% | — | — |
| C.1 | 5,821 | 92.5% | 7.5% | — |
| G.1.3 | 4,182 | 100.0% | — | — |
| J | 4,059 | 99.8% | 0.2% | — |
| G.2.2 | 2,460 | 99.9% | — | 0.1% |
| E.2 | 1,588 | 100.0% | — | — |
| **E** | **1,511** | **100.0%** | — | — |
| **D** | **1,139** | **100.0%** | — | — |
| G.1 | 1,138 | 100.0% | — | — |
| J.2.4 | 776 | 97.9% | 2.1% | — |

Only D (2,256 → 1,139) and E (2,834 → 1,511) moved; the other eleven are identical to
the S4 run. Total assigned 89,101 → 86,661.

**C.1 was left alone.** It has the same empty defining set and 7.5% ancestral (clade C),
and a `C` root was built and is available in
`dev/lineage_reassign/roots_13lineage_outgroup/`. It is clean on the contaminant measure,
so giving it a parent was not part of the agreed change — but it is a one-line topology
edit if the same treatment is wanted for consistency in the Methods.

---

## 5. Plan

- [x] **S1. Port the script.** `dev/lineage_reassign/build_lineage_subalignments_tree.py`
      (the old repo is untouched). The linear `branch_by_ref[i-1] -> [i]` ladder is replaced
      by a parent/children map; every threshold is a CLI flag. Two input modes: `--refs`
      (ordered ladder FASTA, linear topology, original behaviour) and
      `--roots-dir` + `--topology`.
- [x] **S2. Regression gate — PASSED 22 Oct 2026.** Five-lineage ladder,
      `hard` / `nextle2` / `max5`, run against the 396,392-record AA alignment.
      Counts **and** unique counts reproduce exactly:

      | lineage | published | ported |
      |---|---|---|
      | G.1 | 229 (62 unique) | 229 (62) |
      | J_int | 4,132 (937) | 4,132 (937) |
      | J.2_int | 27,452 (5,264) | 27,452 (5,264) |
      | J.2.4 | 877 (266) | 877 (266) |
      | K | 17,898 (2,747) | 17,898 (2,747) |

      Derived defining-mutation counts also match (J_int 9, J.2_int 2, J.2.4 2, K 7), and a
      200-record spot check against the published sub-alignments agreed 200/200.
      Environment: `plm_entropy` (Biopython 1.81).
- [x] **S3. Build the 13-lineage ladder inputs.** `dev/lineage_reassign/topology_13lineage.tsv`
      from §3.2; all 13 roots present in `data/roots/h3n2_13lineage/`.
- [x] **S4. Run the 13-lineage assignment — DONE 22 Oct 2026.**
      Output `dev/lineage_reassign/out_13lineage_named/` (sub-alignments named
      `subalignment_<lineage>.fasta`, which is what `scripts/build_lineage_guide.py:135`
      expects). 89,101 of 396,392 records assigned.

      | lineage | published 5-set | nearest-root 13-set | **this run** | vs published |
      |---|---|---|---|---|
      | J.2 | 27,452 | 25,841 | **26,703** | −2.7% |
      | K | 17,898 | 19,492 | **17,718** | −1.0% |
      | **J** | **4,132** | **2,489** | **4,059** | **−1.8%** |
      | J.2.4 | 877 | 964 | **776** | −11.5% |
      | G.1 | 229 | 779 | **1,138** | root redefined, not comparable |

      The other eight: C.1 5,821 · D 2,256 · E 2,834 · E.2 1,588 · G.1.1 12,597 ·
      G.1.3 4,182 · G.2 6,969 · G.2.2 2,460.

      **The §1 motivation is confirmed.** J comes back to 4,059 — within 1.8% of the
      published 4,132 — against 2,489 under nearest-root. The −40% was the
      misassignment the published rule exists to prevent, and the generalised rule
      prevents it on the branching tree too.

      **Run inputs changed from the plan:** `data/recovered_seg4_HA_AA_594.fasta`, the
      in-repo copy, not the external `sgt_4_HA_AA.fasta` — see R6 below.

- [x] **S5. Purity and sanity — DONE 22 Oct 2026.**
      `dev/lineage_reassign/summarize_purity.py` against `nextclade_id_clade.tsv`;
      report at `out_13lineage/purity_report.tsv`.

      Raw agreement with the Nextclade call understates purity badly, because Nextclade
      labels are *mutually exclusive*: a sequence inside lineage J is called `J.1`, never
      `G.1.3.1.1`. Only thirteen clades have a root to be assigned to, so a child clade
      with no root of its own **must** land in its nearest represented ancestor — that is
      the rule working, not failing. The report therefore unaliases both names
      (`K` → `J.2.4.1` → `G.1.3.1.2.4.1`) and splits the calls into
      self / descendant / ancestral / other. Only **other** is a misassignment.

      | lineage | n | correct (self+desc.) | ancestral | contaminant | top contaminants |
      |---|---|---|---|---|---|
      | J.2 | 26,703 | 100.0% | — | — | — |
      | K | 17,718 | 100.0% | — | — | — |
      | G.1.1 | 12,597 | 100.0% | — | — | — |
      | G.2 | 6,969 | 100.0% | — | — | — |
      | C.1 | 5,821 | 92.5% | 7.5% | 0.0% | — |
      | G.1.3 | 4,182 | 100.0% | — | — | — |
      | J | 4,059 | 100.0% | — | — | — |
      | **E** | 2,834 | **53.6%** | — | **46.4%** | B.1.2.1 (681), B.1 (345), B.1.2 (290) |
      | G.2.2 | 2,460 | 99.9% | — | 0.1% | G.2.1 (2) |
      | **D** | 2,256 | **50.8%** | — | **49.2%** | B.1.2.1 (948), B.1.2.1.1 (161) |
      | E.2 | 1,588 | 100.0% | — | — | — |
      | G.1 | 1,138 | 100.0% | — | — | — |
      | J.2.4 | 776 | 97.9% | 2.1% | — | — |

      Eleven of thirteen are clean. **R2 has landed, but only on D and E, not C.1.**
      §4.1 expected C.1 and D to behave alike; they do not. C.1 has an empty defining
      set and no children yet is 0% contaminated — it simply sits far from everything
      else. D and E sit next to clade B.1.2.x, and with nothing to screen against they
      absorb roughly half a bucket of it. **D-REASSIGN-1 is now a live decision, and it
      is about D and E specifically** (see §4.1).

- [x] **S6. Rebuild the two derived tables — DONE 22 Oct 2026.**
      Output `dev/lineage_reassign/out_tables_13lineage_reassigned/`. The shipped
      `results/h3n2_13lineage/` tables are untouched.

      §6's conclusion held but its premise did not: `obs_freq` / `obs_present` / `depth`
      genuinely do not depend on the model, but as the pipeline stood they were only ever
      *emitted* from inside `scripts/run_mutational_accessibility.py`, bolted to the GPU
      scoring loop, and both exporters in `clean_repo/tools/` read only the archive
      `combined_long_table.csv`. There was no FASTA → table path. **New tool:**
      `dev/lineage_reassign/build_observed_tables.py`, CPU-only, which reuses the
      pipeline's own `build_reference_to_alignment_column_map` and
      `compute_observed_diversity_profile_fast` rather than reimplementing them, so the
      column mapping and depth conventions cannot drift.

      **Regression gate — PASSED.** Run against the *existing* sub-alignments and roots
      it reproduces both shipped tables exactly: 139,802 rows, 6,546 present, every
      value equal.

      Rebuilt vs shipped, observed replacements: **6,546 → 6,571** (net +25), but with
      real churn underneath — 685 gained, 660 lost, 1,345 of 139,802 rows changed
      (0.96%).

      | lineage | old | new | gained | lost | net |
      |---|---|---|---|---|---|
      | C.1 | 628 | 595 | 1 | 34 | −33 |
      | D | 259 | 299 | 56 | 16 | +40 |
      | E | 329 | 416 | 110 | 23 | +87 |
      | E.2 | 308 | 297 | 0 | 11 | −11 |
      | G.1 | 214 | 215 | 37 | 36 | +1 |
      | G.1.1 | 557 | 704 | 184 | 37 | +147 |
      | G.1.3 | 391 | 390 | 43 | 44 | −1 |
      | G.2 | 562 | 609 | 75 | 28 | +47 |
      | G.2.2 | 397 | 369 | 2 | 30 | −28 |
      | J | 445 | 488 | 89 | 46 | +43 |
      | J.2 | 1,172 | 1,090 | 88 | 170 | −82 |
      | J.2.4 | 231 | 172 | 0 | 59 | −59 |
      | K | 1,053 | 927 | 0 | 126 | −126 |

      Note D (+40) and E (+87) gain most in relative terms — those gains are partly the
      clade-B contamination of S5, so D-REASSIGN-1 should be settled before these two
      lineages are believed.

      **Superseded 22 Oct 2026 by the outgroup run (§4.2).** The table above is the
      pre-decision rebuild and is kept only for comparison; it is not the number to
      quote. Rebuilt from `out_13lineage_outgroup/` into
      `dev/lineage_reassign/out_tables_13lineage_outgroup/`, against the shipped tables:
      **6,546 → 6,392 observed replacements (net −154)**, 526 gained, 680 lost,
      1,206 of 139,802 rows changed (0.86%).

      | lineage | shipped | new | gained | lost | net |
      |---|---|---|---|---|---|
      | C.1 | 628 | 595 | 1 | 34 | −33 |
      | D | 259 | 231 | 1 | 29 | −28 |
      | E | 329 | 305 | 6 | 30 | −24 |
      | E.2 | 308 | 297 | 0 | 11 | −11 |
      | G.1 | 214 | 215 | 37 | 36 | +1 |
      | G.1.1 | 557 | 704 | 184 | 37 | +147 |
      | G.1.3 | 391 | 390 | 43 | 44 | −1 |
      | G.2 | 562 | 609 | 75 | 28 | +47 |
      | G.2.2 | 397 | 369 | 2 | 30 | −28 |
      | J | 445 | 488 | 89 | 46 | +43 |
      | J.2 | 1,172 | 1,090 | 88 | 170 | −82 |
      | J.2.4 | 231 | 172 | 0 | 59 | −59 |
      | K | 1,053 | 927 | 0 | 126 | −126 |

      D and E now come **down** (−28, −24) rather than up (+40, +87): the earlier gains
      were largely the material the outgroup screen removes.

- [x] **S6b. Rebuild the alpha sweeps — IN PROGRESS 22 Oct 2026.**
      §6's cost estimate missed this, the same way it missed S6. Figure 3 panels A and B
      and all of Figure SB do **not** read the scores: they read the stored
      `alpha_sweep_by_lineage[_masked]` tables, whose metrics are computed from the
      scores *crossed with the observed outcomes*. No probability changes, but every
      metric does. **Table SW** (held-out alpha) reads the same tables and was missed in
      the first scan of affected outputs.

      New tool `dev/lineage_reassign/build_alpha_sweep.py`, CPU-only, reusing the
      pipeline's own `_build_alpha_tables_from_combined` so the alpha grid, pseudocount,
      logistic fit and column set cannot drift. Gate: rebuild from the *shipped* observed
      tables must reproduce the shipped sweep exactly. Both scoring arms are needed —
      Table SW writes masked and unmasked by design.

      *Operational note:* each arm is ~2,626 logistic fits. Run multi-threaded they
      thrash (load 65 on 12 cores) and take over an hour per model; pinned to one BLAS
      thread each they are roughly twice as fast. Pin them.

- [x] **S6c. Rebuild Table SZ's un-shipped inputs — DONE 22 Oct 2026.**
      `tools/export_lineage_panel.py` needs an accession→lineage list and the pipeline's
      `panel_metadata.tsv`; neither ships, and the archived `panel_metadata.tsv`
      (`results/nextstrain_13lin_dense/tables/`) describes the *original nearest-root*
      assignment (C.1 6,082, D 1,217, J 2,489), so both must be rebuilt.

      New tool `dev/lineage_reassign/build_panel_inputs.py`. No new estimator was needed:
      **`differing_sites_vs_reference_non_gap_non_stop` is exactly the number of distinct
      positions carrying an observed replacement**, which holds 13/13 against the archive.

      **Gates — ALL PASSED.** From the original sub-alignments
      (`results/lineage_subalignments_nextstrain_fa/`) it reproduces `clade_accessions.tsv`
      (78,659 rows) and the archived `panel_metadata.tsv` (13/13) exactly; feeding those
      to `export_lineage_panel.py` reproduces the shipped `lineage_panel.tsv.gz`
      **byte for byte**, and `tools/export_homoplasy.py` reproduces the shipped
      `homoplasy_by_lineage.tsv.gz` **byte for byte**.

      **The segment-4 tree was found**, which closes the one open question of the day:
      `/home3/oml4h/RABV-gTK/treepatch/work/seg4/repair/usher/usher.pb` — 780,727 nodes
      and 572,108 samples by `matUtils summary`, exactly as RUNS.md records. (The sibling
      `work/seg4/usher/usher.pb` is the *pre-repair* tree: 496,726 / 371,658.) The DB's
      `trees` table holds Newick only, so it cannot serve — but the extracted `seg4.nwk`
      is 9,921,098 bytes against the DB's 9,921,315-character record, confirming the same
      tree from a second direction. The three `matUtils extract` outputs are
      lineage-independent, so only `--accessions` changes between runs.

- [ ] **S7. Re-run figures** off the rebuilt tables, Fig 3 first, then Fig SB.
      **Unblocked 22 Oct 2026** — D-REASSIGN-1 is resolved and all thirteen lineages are
      clean (§4.2). Inputs: `dev/lineage_reassign/out_tables_13lineage_outgroup/`.
      Open sub-decision: whether those tables replace
      `clean_repo/results/h3n2_13lineage/` or the figure run points elsewhere.
- [x] **S8. Tests — DONE 22 Oct 2026.** 77 passing in `dev/lineage_reassign/`
      (`python -m pytest`, env `plm_entropy`):
      `test_build_lineage_subalignments_tree.py` (33) covers the topology parser, the
      all-children next-test, the Hamming cap boundary (`==5` kept, `==6` dropped) and
      the linear-ladder equivalence; `test_build_observed_tables.py` (17) covers the row
      universe, `obs_present ⇔ obs_freq > 0`, the ignore-character set, uncovered
      positions, and that the S6 gate fails on a mismatch rather than passing vacuously;
      `test_summarize_purity.py` (27) covers alias expansion, the nesting relation and
      the sibling trap (`G.1.10` is not a child of `G.1.1`).
- [ ] **S9. Methods rewrite** — the generalised rule, the real thresholds, the §2.2
      corrections, the reconciliation table, and now also the outgroup roots for D and E
      and the terminal-leniency anchoring (§4.2).

---

## 6. Cost

Much lower than feared. `replacement_scores` is keyed on the lineage **root** and the
model, and **the roots do not change** — only which sequences sit in each lineage does.
Per `DATA.md`, `replacement_observed` and `replacement_frequency` "do not depend on the
model". So:

- **No PLM re-scoring. No GPU. No fine-tuning.**
- Rebuild is: assignment → sub-alignments → `replacement_observed` + `replacement_frequency`
  → figures.

Dominant cost is one pass over the 233 MB AA alignment per run.

**Confirmed, 22 Oct 2026 — with one correction.** The conclusion is right and the whole
rebuild ran on CPU in minutes. But the step "sub-alignments → the two tables" did not
exist: those three columns were only ever emitted from inside
`scripts/run_mutational_accessibility.py`, alongside the GPU scoring, and the two
exporters in `clean_repo/tools/` read only the archived `combined_long_table.csv`.
`dev/lineage_reassign/build_observed_tables.py` is that missing step, and it reproduces
the shipped tables exactly from the existing sub-alignments (S6).

Measured: assignment ≈ 4 min, tables ≈ 2 min, both single-core.

---

## 7. Risks

| # | Risk | Handling |
|---|---|---|
| R1 | Port silently diverges from the published rule | S2 regression gate on exact counts |
| R2 | C.1 / D are low-purity buckets (§4.1) | ~~Measure in S5; D-REASSIGN-1 if bad~~ **Closed 22 Oct 2026** — §4.2. Landed on D and E, not C.1; roughly half of it was an alias-map defect in the measurement, the rest fixed by outgroup roots. All thirteen now 0.0% contaminant |
| R7 | Purity is scored with a hand-maintained alias map, which can silently call a clade's own ancestor foreign | **Realised, then closed 22 Oct 2026.** Nesting is now derived from the reference tree and checked by rediscovering the two documented aliases; two tests that asserted the defective behaviour were replaced |
| R3 | Topology wrong for C.1 / D, which have no ancestor in the set | They get no next-test either way, so Fig 3 is insensitive to their exact placement |
| R4 | Lineage sizes shift enough to change the bootstrap band | Expected and legitimate; report old vs new side by side |
| R5 | Headline Fig 3 result moves | **Updated (§4.2 run):** 6,546 → 6,392, 0.86% of rows changed, 526 gained against 680 lost |
| ~~R5 (pre-decision)~~ | ~~Headline Fig 3 result moves~~ | Report whatever it does. The −2% total across the shared five suggests it will not move much, but J is the lineage most affected and that is not knowable in advance. **Measured (S6):** the observed-replacement count moves 6,546 → 6,571, under 1% of rows changed — but with 685 gains against 660 losses, so the churn is far larger than the net |
| R6 | ~~`sgt_4_HA_AA.fasta` is owned by another user and could change~~ | **Closed 22 Oct 2026** — see below |

---

### 7.1 R6, and which alignment the run actually used

The external `sgt_4_HA_AA.fasta` and the in-repo `data/recovered_seg4_HA_AA_594.fasta`
are **not the same file**. They have identical sizes (243,784,608 B) and identical record
counts (396,392), which is why the duplication went unnoticed, but they differ from byte
150,350,922 on and hold different accessions in different order (the external has
`PX902199` where the in-repo copy has `OP510191`). The in-repo file is a rebuild, not a
copy — hence "recovered".

Both were tested, and it does not matter:

- the **five-lineage regression gate reproduces the published counts exactly on both**
  (229 / 4,132 / 27,452 / 877 / 17,898, unique counts too);
- the **13-lineage run gives byte-identical counts on both**, lineage by lineage.

The canonical run therefore uses the **in-repo copy**, because it is versioned here,
cannot be changed by another user, and is the file the existing `results/h3n2_13lineage/`
tables were built from. R6 is closed rather than mitigated. The external path is retained
in §2 only as the provenance of the in-repo copy.

---

## 8. Log

| Date | Entry |
|---|---|
| 2026-10-22 | Opened. Assignment code located and confirmed to reproduce published counts. External alignment confirmed readable. Established that no GPU re-scoring is needed. Topology derived from roots + UShER segment-4 tree + nomenclature; two naive estimators tried and rejected (§3.3). Branch rule set to "all children". |
| 2026-10-22 | **S1–S3 done. Regression gate passed** — the port reproduces all five published counts and unique counts exactly. Test suite written (`test_build_lineage_subalignments_tree.py`, 31 tests); first run 23 passed / 8 failed, all 8 traced to two defects in the *test fixture*, not the port: the back-translation helper mapped `B` to `GGA`, which translates to glycine, so every expected mutation position was wrong; and `_roots_tree` called `mkdir()` without `exist_ok`, so any test building profiles twice raised `FileExistsError`. Fixing the fixture, not the script. |
| 2026-10-22 | **S4–S6 and S8 done.** Fixture defects fixed; 33 tests green. 13-lineage assignment run: **J recovers to 4,059 against nearest-root's 2,489**, within 1.8% of the published 4,132 — §1's motivation confirmed on the branching tree. Purity (S5) needed an alias-aware nesting test before it meant anything, since Nextclade labels are mutually exclusive; with that, eleven of thirteen lineages are ~100% correct and **R2 lands on D (49.2% clade-B) and E (46.4%) only — not C.1, which is clean.** D-REASSIGN-1 narrowed to D and E and written up with three options. S6 needed a new CPU-only tool (`build_observed_tables.py`) because no FASTA → table path existed; it reproduces both shipped tables exactly as its gate, then rebuilds them: 6,546 → 6,571 observed replacements, 0.96% of rows changed, 685 gained / 660 lost. R6 closed — the external and in-repo alignments differ but both reproduce the published counts, so the run uses the in-repo copy. 77 tests green. S7 held pending D-REASSIGN-1. |
| 2026-10-22 | **D-REASSIGN-1 resolved: option 1, outgroup roots for D and E (§4.2).** First finding: roughly half the reported contamination was a defect in the *measurement*. The purity script unaliased names with a hand-maintained map holding only J and K, but D and E are aliases too — read off the Nextclade reference tree, D's parent clade is B.1.2.1 and E's is B.1.2, the very clades counted as their contaminants. Nesting now comes from the tree (`clade_nesting.py`) and the derivation rediscovers both documented aliases unprompted as its own check; rescoring the S5 run drops D from 49.2% to 7.1% contaminant and E from 46.4% to 24.0%. Second: outgroup roots for B.1.2 and B.1.2.1 reconstructed by the same construction as the shipped thirteen (`build_clade_roots.py`), gated on reproducing all thirteen exactly. The topology gained a `reference_only` column so an outgroup root supplies a defining set without becoming an assignment candidate and without changing any other lineage's cumulative set. Third: D is childless, so the terminal leniency applied with an empty cumulative set and admitted B.1.2.1.1 — D's sibling, root 5 AA away at the Hamming cap, sharing 3 of D's 5 defining mutations; anchoring cumulative on the outgroup root restores the §2.3 invariant and removes them. **All thirteen lineages are now 0.0% contaminant**, D and E both 100.0% correct. Only D (2,256 → 1,139) and E (2,834 → 1,511) moved. Five-lineage gate still exact. Tables rebuilt: 6,546 → 6,392 observed replacements, 0.86% of rows changed. 122 tests green. S7 and S9 unblocked. |
