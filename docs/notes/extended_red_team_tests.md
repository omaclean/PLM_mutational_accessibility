# Extended red team: a plan to confirm the tests test what they claim

**Opened:** 2026-10-15. **Target:** before resubmission (~2 Nov 2026).

This is the next pass after the 15 Oct pre-resubmission audit. That audit asked *"is the
code right?"* and found three defects (`tests/test_known_defects.py`). This pass asks a
different question:

> **Is the test suite actually capable of catching a wrong number, or does it only
> confirm that the code agrees with itself?**

A suite of 1,758 tests that all pass is evidence of nothing until you know what it would
take to make one fail. Everything below is organised around that.

Scope is `clean_repo` only. Per [`docs/notes/PLAN.md`](PLAN.md) §1 the engine stays
question-agnostic: these tests must pass whichever way the scientific disputes in
[`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md) and [`SYMMETRY_TESTS.md`](../../SYMMETRY_TESTS.md)
land.

---

## 0. Baseline, measured 15 Oct 2026

```
nice -n 10 ionice -c3 /home3/oml4h/miniconda3/bin/python -m pytest tests/ -q
1,758 tests collected. 1 failed, the rest passed or skipped.
(~4 min, single process, no GPU, no network)
```

The one failure is a **true positive** and is item A1 below. Everything else is green,
which is why this document exists: green tells us nothing on its own.

Suite shape, for reference:

| | |
|---|---|
| Test files | 50, about 14,000 lines |
| `def test` functions | ~950, expanding to 1,758 cases via parametrisation |
| Modules under test | 16 in `mutacc/`, ~3,700 lines |
| Marked `slow` | 12 tests, in 8 files — **no skip gate, they run by default** |
| Heaviest single test | `test_branch_codons.py::test_random_trees_match_the_independent_oracle`, 180 parametrised cases against a second implementation |

The suite's real strengths, which this plan must not disturb: `test_branch_codons.py` and
`test_codon_learning.py` both check against an **independent oracle**, and the
`test_redteam_*.py` files already pin a dozen genuine bugs (ranking under ties, partial
AUROC interpolation, `reference_codons` accepting `-1`, `direction_contrast` dividing by
the wrong row count). Section 3 lists what they cover so we do not re-test it.

---

## A. Confirmed problems — fix these before adding a single new test

These were found while planning. They are not hypotheses.

- [x] **A1. `output/` ships two orphan files** — investigated 15 Oct, and it is not an
      orphan problem. `table_s15_dms_alpha_optima.tsv` is byte-identical to
      `sY_dms_alpha_optima.tsv`, which `tools/export_sY_dms_alpha_optima.py` now writes, so
      **two different supplementary tables are both labelled S16** (the other is
      `table_s16_simulation_control.tsv` from `figures/make_s16.py`). The caption is
      hand-written, has no generator, and says "Supplementary Table S15".
      `docs/notes/PLAN.md:885` contradicts itself on the same line, and names a
      `tools/export_table_s16.py` that does not exist.
      **This opened into the renumbering work in §A1b.** *Closed 20 Oct 2026 with A13:* the
      `s15` copy and its hand-written caption are gone, the surviving table is lettered
      **Table SY** in the numbering map, and `table_s16_simulation_control.tsv` takes Figure
      SN's label as the companion table it is. Nothing is now labelled S16, or S15, at all.

### A1b. Supplementary numbering: match the draft, then decouple it

Chasing A1 showed the repo's supplementary numbering has drifted from the manuscript
draft. Oscar supplied the draft's current ordering on 15 Oct; the mapping below was
confirmed by reading each caption against the draft's description.

> ⚠️ **This ordering is provisional.** Oscar's note: *"they're all going to need
> re-ordering once the MS is fully written and some are likely to be dropped."* That is
> the whole reason for the map file in stage 1 — the next re-order must not cost another
> repo-wide rename.

| Draft | Repo stem today | Content | Move |
|---|---|---|---|
| S1–S7 | `s01`–`s07` | — | none, already correct |
| **S8** | `s13_contact_recovery` | epistatic coupling by region, two pH isoforms | 13 → 8 |
| S9 | `s09_codon_pair_tracking` | asymmetric codon pairs forest plot | none |
| S10 | `s10_dms_concordance` | DMS concordance vs distance | none |
| S11 | `s11_branch_codons` | branches × changes per codon | none |
| **S12** | `s18_dms_sc2` | SARS-CoV-2 DMS alpha sweeps | 18 → 12 |
| **S13** | `s12_within_stratum` | PLM vs DMS within nt strata | 12 → 13 |
| **S14** | `s16_simulation_control` | fine-tuned ESM-C on simulated data | 16 → 14 |
| — | `s14_lineage_set_validity` | lineage-set validity audit | **dropped** |

Two decisions taken 15 Oct:

* **The lineage-set validity figure is dropped** from the numbered set and retired to
  `dev/retired_figures/`. Its code and tests stay — only its manuscript slot goes. The
  findings it draws remain in [`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md) §4.
* **Numbers come out of filenames.** Stems become content names
  (`within_stratum`, `dms_sc2`, `contact_recovery`, `simulation_control`, …) and a single
  map file carries the manuscript number. Re-ordering then edits one file instead of ~20
  scripts, 80 output files, 50 test files and five documents.

Note that **S12 and S13 swap**, so any rename must go through temporary names or the two
clobber each other.

**Staged, with the suite run between each stage.** The suite is green apart from the one
true positive, and that signal is worth keeping while this much moves.

- [x] **Stage 1 — the map, reading nothing.** Landed 15 Oct 2026.
      `guides/manuscript/figure_numbers.yaml` (content stem → manuscript label) plus
      `mutacc/numbering.py`, and `tests/test_numbering.py` (70 cases, all green).
      **Path changed from the plan:** the map cannot live at `guides/*.yaml`, because
      seven test files and `load_guide` glob that directory and parse everything in it as
      a *run* guide — it broke collection in two files immediately. It sits under
      `guides/manuscript/`, beside the `guides/composites/` precedent.
      The map carries 27 entries: 19 numbered (5 main, 14 supplementary), 5 panels or
      side tables that take a parent's number, 3 retired. The loader rejects a duplicate
      label, a gap in the numbering, an entry that is both numbered and retired, a parent
      that is not in the map, a supplementary label in the main section, and an unknown
      key; each has a test. Suite after: green apart from the A1 true positive.
- [x] **Stage 2 — captions take their number from the map.** Landed 15 Oct 2026.
      `mutacc.io.write_caption` now prepends the label itself (`caption_label(stem, panel)`),
      **raises** if the caption text names its own number, and gives a retired figure no
      number at all. Every caption template in `figures/` had its self-naming removed, and
      `make_fig04.py`'s hard-coded `FIGURE_LABELS = {STEM: "Figure 4", STEM_SC2: "Figure
      S18"}` is gone — that dict was the drift, in the one place it could not be seen.
      The 19 shipped captions were rewritten from the map, re-wrapping the first paragraph
      exactly as `write_caption` does, so the next figure build reproduces them byte for
      byte. **The swap and the moves have now reached the text:** `s18_dms_sc2` opens
      "Figure S12", `s12_within_stratum` "Figure S13", `s13_contact_recovery` "Figure S8",
      `s16_simulation_control` "Figure S14". Ten captions that said only "Supplementary
      figure." now carry a number. Tests: 11 in `tests/test_numbering.py`, including one
      per shipped caption.
      **What stage 2 broke, and what that taught:** six tests failed on the first full
      run, all from one wrong assumption — that a figure's output stem is fixed. It is
      not. `make_s11.py` builds `s11_branch_codons_SYNTHETIC_FIXTURE` when the input is a
      planted tree, and a gauge or condition can be appended elsewhere; those are
      *conditions of the same figure* and take its number. `Numbering.__getitem__` now
      resolves a derived stem to the longest registered stem it extends — unambiguous
      only because `test_output_provenance.py` already forbids one declared stem from
      being a prefix of another. A genuinely new stem still raises.
      Second lesson: `make_s11.py`'s caption opens with a "SYNTHETIC FIXTURE — do not
      quote them" banner, and putting "Figure S11." in front of it buried the loudest
      thing in the file. `write_caption` grew a `banner=` argument written *ahead* of the
      label. A figure number must not outrank a do-not-quote warning.
- [x] **Stage 2b — the supplementary labels became letters.** Landed 20 Oct 2026.
      `Figure SA` … `Figure SO` and `Table SZ` replace `S1` … `S15` and `Table S1`; main-text
      figures keep their digits. Stage 2 fixed the *mechanism* by which a number reached a
      caption, but a number is still a position, so the 15 Oct re-order moved four figures and
      every document that named them had to follow. A letter is a name. It is assigned once,
      survives any re-order, and the manuscript's final `S1`…`SN` run is applied in a single
      pass at submission from the order of first citation. The loader now matches
      `^Figure S([A-Z])$` and `^Table S([A-Z])$`, reduces a letter to an ordinal so the
      no-gaps check still holds the figures to A..O, and **exempts supplementary tables from
      the run** so `Table SZ` can sit at the end of the alphabet and not move when a figure is
      added. A *numbered* supplementary label is now rejected, with a test. New document
      `FIGURE_LABELS.md` carries letter → old number → stem → script → content, held to the map
      by three tests in `test_document_numbers.py`. The 16 shipped captions were re-labelled in
      place; `write_caption` takes the label from the map, so a rebuild writes the same bytes.
- [ ] **Stage 3 — rename the stems to content names.** Two-phase via temporary names.
      Touches `figures/`, `output/`, `tests/`, `README.md`, `DATA.md`,
      `figures/README.md`, `docs/notes/PLAN.md`.
- [ ] **Stage 4 — retire lineage-set validity**, and collapse the triplication below.

**Found while mapping, and approved for collapse:** `figures/make_s16.py` writes the same
values frame to **three** filenames — `s16_simulation_control_values.tsv`,
`supl_table_16.tsv` and `table_s16_simulation_control.tsv`. Keep the `_values.tsv` the
house convention expects, drop the other two, and add a test that no script writes one
frame under two names. That also dissolves the S16 table collision in A1 without a
separate decision.

- [x] **A2. The defect ratchet is not wired up — resolved 15 Oct 2026: the fixes
      landed, the prose was stale.** Read against the modules, not inferred:
      `strata._additive_residual` now divides by the total about zero (`strata.py:323`,
      with a comment citing the test file), and `metrics._check` now rejects a non-finite
      or non-binary *outcome* (`metrics.py:88-91`). Both defects are fixed; the tests
      that describe them are regression guards, and no `xfail` is wanted — a strict
      xfail on a passing test fails the suite. Defect 3 (the alphabetical primary-lineage
      tie-break) is not a code fault and stays open as a decision for Oscar.
      The docstring and both comment blocks now say this.
      *The "arguably incompatible" pair is not incompatible, and the reason matters:* a
      least-squares fit is an orthogonal projection, so the residual sum of squares cannot
      exceed the total **about the same origin**. Against the uncentred total the share is
      in `[0,1]` by construction; against the centred total it was not. The two tests pass
      together only *because* of the fix, and would not have before it.
      *What the fix moved:* nothing shipped. `share_by_amino_acid` is written out only by
      `figures/make_fig02.py`, retired on 14 Oct; the pre-fix 0.886896 survives in
      `dev/retired_figures/fig02_model_vs_accessibility_values.tsv`. No shipped table,
      caption or document quotes the additive share — checked by grep across `output/`,
      every `.md`, and `SYMMETRY_TESTS.md`.

- [x] **A3. `figures/README.md` has drifted from the code — fixed 15 Oct 2026, and the
      drift found was not quite the one reported.** The two defects named in the audit had
      already been repaired: the README documents `make_s16.py`, and the duplicate
      `make_s04.py` row is struck through with its withdrawal explained. What was still
      wrong:
      * **two scripts had no row at all** — `make_fig02d.py` (panel D of Figure 2) and
        `make_s01.py` (the SARS-CoV-2 spike figure). Both now have one.
      * the `make_fig02.py` row still claimed to be "Figure 2". It was retired as a figure
        on 14 Oct and is kept only as the home of `PROPERTIES`, `TABLE`, `stratum_x` and
        `spread`, which `make_s04.py` and `make_fig02d.py` import. Struck through, with
        that reason in the row.
      * the warning box now points at the map file rather than at a prose list of moves.
      **So it cannot recur:** `tests/test_document_numbers.py` asserts every `make_*.py`
      has a row, and that the README names no script that does not exist. That is the test
      A3 needed and never had — the reason the drift was found by a human twice.

- [~] **A4. One test reads outside the repo — contained 15 Oct 2026, not yet decided.**
      `test_sim_table_exporter_function` now carries a `needs_exploration_tree` skip
      marker that states the missing path and why, so a clean checkout **skips with a
      reason** instead of failing. The path is a named constant, so a second test reaching
      outside is one marker away rather than a new string literal.
      This contains the symptom; it does not answer the question. **Ship or drop is still
      Oscar's call,** and the cost is worth knowing first: the test reads a 1.65 MB table,
      and `sim_table` then reads four more arm tables and a founder FASTA from the same
      tree. Shipping it honestly means several MB of fixture, or a trimmed one — at which
      point the test no longer checks the 13,300-row shape it asserts.
      Still open from the original item: Batch 6's `--strict-offline` conftest flag, which
      is what would stop a *new* outside read rather than documenting this one.

- [ ] **A5. Nothing pins a library version.** `environment.yml` pins none,
      `requirements.txt` uses lower bounds only. We assert goldens to `1e-9` and `1e-12`
      against scipy and numpy. A point release that changes a tie-break or a summation
      order turns a published number into a test failure, or worse, doesn't. Add an exact
      lockfile used by CI, keep the loose file for humans.

- [ ] **A6. `MANIFEST.tsv` hashes the wrong file.** `source_sha256_16` is the hash of the
      source in the exploration archive, which is not shipped and cannot be checked here.
      Nothing detects a shipped `.tsv.gz` being edited in place. Add a `sha256` of the
      shipped file and a test that recomputes it.

- [ ] **A7. Shipped direction tables may predate their own bug fix.**
      `tests/test_redteam_accessibility.py` carries an in-file note that the
      `symmetry_*_direction` tables were not regenerated after the `direction_contrast`
      denominator fix. If true, numbers in `SYMMETRY_TESTS.md` §4 come from pre-fix tables.
      Regenerate and diff, or document why the fix does not move them.

- [x] **A9. `fig02d_strata_panel_caption.md` has no generator, and the provenance test
      cannot see that.** Found 15 Oct while wiring stage 2. `make_fig02d.py` draws the
      panel and writes its values table but never calls `write_caption`; the shipped
      caption is hand-written and opens "**Figure 2D.**" with a number typed by hand.
      `test_every_output_file_has_a_producer` credits a script with every file matching
      `{stem}_*`, so a hand-written caption beside a figure that *is* produced is counted
      as produced. The same hole hides A1's `table_s15_dms_alpha_optima_caption.md`.
      **Pinned as defect 4 in `tests/test_known_defects.py`, `xfail(strict=True)`** — the
      first real use of the ratchet A2 found missing. Whoever gives those two captions a
      generator, or retires them, must take the marker off.
      *Open question for Oscar:* `make_fig01d.py` writes a panel with no caption at all,
      so the precedent may be that 2D's caption belongs in the composite's caption rather
      than in a file of its own.

- [ ] **A10. `s15_lineage_panel` is not in the §A1b ordering table above.** The script
      arrived mid-session on 15 Oct already carrying a `figures/README.md` row, a map
      entry (Figure S15) and `tests/test_s15_lineage_panel.py`, so it is **not** an
      inventory gap — I claimed twice that it was, and was wrong both times for not
      checking first. The one thing that is true: the draft ordering recorded in §A1b
      predates it and still stops at S14. Its caption had a hand-typed "Figure S15", now
      taken from the map like the rest. **Resolved 19 Oct 2026:** it is not a figure at all
      — Oscar moved it to **Table S1**, and the S15 figure slot went to the new
      `masked_marginals`. The numbering map now carries figure and table sequences
      separately (`mutacc/numbering.py`, `Table SN` labels), so a stem crossing between them
      closes its own hole. §A1b's table is still the pre-S15 draft ordering and still wants
      re-stating from the current draft.

- [ ] **A8. One bare seed literal.** `figures/make_codon_learning.py:189` uses
      `np.random.default_rng(7)` where every sibling uses a named module constant. Cosmetic,
      but `test_redteam_figures.py` scans for seeded RNGs and a literal is easy to miss.

### Found 16 Oct 2026, by rebuilding every figure into a scratch directory

`tools/build_figures.py` now exists (see Batch 6 and §D) and its first real build turned up
five things no test had caught. Each is pinned in `tests/test_build_figures.py`.

- [x] **A11. `results/` was git-ignored from the first commit.** The top-level `.gitignore`
      has a bare `results` rule meant for the exploration archive; it also matched
      `clean_repo/results/`, so **none of the 110 derived tables (18 MB) were ever
      committed** and a fresh clone could not build a single figure. `git ls-files
      clean_repo` showed `output/`, `data/`, `figures/` and no `results/`. Fixed with a
      narrow negation (`!clean_repo/results/` and `/**`), which only re-includes that
      directory; `clean_repo/build/` is ignored beside it. Pinned by
      `test_no_file_a_figure_reads_or_ships_is_git_ignored`, which fails on the old rule
      (checked by reverting it). **Nothing is committed yet:** `git status` now shows
      `clean_repo/results/` as untracked, ready to add.
- [x] **A12. `export_sY_dms_alpha_optima.py` silently dropped SARS-CoV-2.** `make_fig04.py` writes the
      influenza panels to `fig04_alpha_vs_dms_values.tsv` and the SARS-CoV-2 panels to
      `s18_dms_sc2_values.tsv`; the exporter read only the first and wrote **8 rows where the
      shipped table has 18**, with no error. It now reads both and refuses a source that is
      empty or lacks the columns (it used to write a 0-row table from a header-less file).
- [x] **A13. The shipped `sY_dms_alpha_optima.tsv` was stale, and so was its
      hand-written caption.** *Closed 20 Oct 2026.* `tools/export_sY_dms_alpha_optima.py` now writes the
      caption with the table, from the table, so the two cannot drift again; every number the
      caption quotes is read off the rows. The sign decision below was taken the way the sweep
      itself takes it: optima are placed by the **size** of the correlation, so the table gained
      `delta_rho_in_size` beside the signed `delta_rho`, and `accessibility_helps` now compares
      sizes against whichever end of the alpha = 0 interval is further from zero. The caption's
      claims are restricted to the fitness panels and say so. The duplicate
      `table_s15_dms_alpha_optima.tsv` and its caption are deleted, the table is **Table SY**,
      and defect 5's `xfail(strict=True)` is off. The original finding, for the record: Regenerated from the shipped values it differs in the ACE2-binding rows: the
      table says those four curves peak at the +1.00 boundary (so 13 rows interpretable), while
      Figure S18 now reads binding panels *in size* and finds interior optima for the BA.2
      fine-tune (-0.18) and XBB.1.5 fine-tune (-0.33), so 14 are interpretable. The model
      labels also changed ("Fine-tuned (influenza)" is now "(IAV)", "(SARS-CoV-2)" is "(SC2)").
      **Not regenerated, because the right fix is a decision:** the table's `delta_rho` and
      `accessibility_helps` assume a positive correlation, which is false for the binding
      panels, so regenerating alone would print `delta_rho = -0.039` and `False` for a panel
      whose correlation grew in size. Pinned as **defect 5, `xfail(strict=True)`**; whoever
      regenerates it must fix the sign handling, rewrite the caption's "thirteen / five rows"
      paragraph, and take the marker off. Same decision as A1 (which table number it carries).
- [ ] **A14. Two shipped captions were never regenerated after `write_caption` changed.**
      `s01` and `s02` wrap "fine-tune" and "anti-predictive" at the hyphen; `io.py` gained
      `break_on_hyphens=False` at 18:06 on 15 Oct, 17 minutes after those files were written.
      The Stage 2 claim above that "the next figure build reproduces them byte for byte" was
      therefore false for these two. Cosmetic, and cleared by any `--out output` build, but it
      is exactly what `build_figures.py`'s comparison exists to catch.
- [x] **A15. `dev/mutation/run_all.sh` hard-coded `/home3/...`**, failing
      `test_no_absolute_paths_in_code_or_config`. Now relocatable (`cd "$(dirname "$0")/../.."`,
      `$PYTHON`).

---

## B. The principles the new tests follow

Written down because the weaknesses in §4 all come from skipping one of these.

1. **An expected value is never computed by the code under test.** Use an independent
   implementation, a hand-worked case, scipy/sklearn, or a published number.
2. **Where a recomputation test is unavoidable** (checking a shipped table is not stale),
   it ships with a paired `test_a_tampered_X_is_caught`. `test_s12.py` and `test_s02.py`
   already do this; it is the pattern to copy.
3. **Goldens are pinned to digits, not to signs or bands.** `value < 0` passes for a value
   that has drifted by 400%.
4. **A test is named after what it asserts.** If it only checks that something runs, it is
   called `..._runs`, not `..._is_correct`.
5. **Mutation testing is the arbiter**, not coverage. Coverage says a line ran; mutation
   says the suite would have noticed if the line were wrong.

---

## C. The barrage

Eight batches. Batch 1 is the one that decides how much of the rest is needed, so it goes
first. Everything runs `nice -n 10 ionice -c3`, max 14 threads.

### Batch 1 — Mutation testing: the test that tests the tests

The whole point of this pass. Introduce small faults into `mutacc/` and count how many the
suite catches.

- [x] **Neither was installed, and neither was added.** This is a shared environment, and
      the hot spots this section predicts are specific and few. `dev/mutation/run_mutation.py`
      (~200 lines of `ast`) does it instead: parse the module, change one node, unparse it
      over the file, run the covering tests, restore in a `finally`. Operators: comparison
      swaps (`<=`↔`<`, `==`↔`!=`, …), arithmetic swaps (`+`↔`-`, `*`↔`/`), `and`↔`or`,
      dropped negation, and off-by-one / zero-one changes to numeric constants.
      Auditable — every mutant is named in the report — and no new dependency.
      **It runs against `dev/mutation/sandbox`, not the live tree.** The module is
      rewritten on disk for the duration of each mutant, so anyone else running the suite
      meanwhile would see failures that are not theirs. The sandbox is the code copied
      (~2 MB) with `results/`, `data/`, `output/` and `reports/` symlinked; a full `git
      worktree` would have written ~300 MB. The harness checks it restored the file and
      says so if it did not.
      *The sandbox had to be built twice, and the first failure was a good sign:*
      symlinking the data directories made every baseline fail, because
      `mutacc.guide._relative` rejects a path that resolves outside the repository — the
      check that exists because the old guides pointed into another repository. It did
      exactly its job. The sandbox now hardlinks those directories (`cp -al`), which
      costs no disk; `output/` is checksummed before and after, since a hardlink means a
      stray write would reach the real file.
- [x] Running 15 Oct 2026 in priority order. Counts: `metrics` 104 mutants,
      `accessibility` 72, `strata` 124 — about 35 minutes for the three at ~5 s a run.
      Full order remains:
      `accessibility` → `strata` → `metrics` → `dms_sweep` → `symmetry` → `audit` →
      `codon_learning` → `linkfit` → `spectrum_swap` → `antigenic`.
- [x] **Round 1 recorded, 16 Oct 2026.** Scoped to the covering tests (`dev/mutation/report_*.md`):

      | Module | Mutants | Killed | Survived |
      |---|---|---|---|
      | `metrics` | 104 | 88 | 16 |
      | `accessibility` | 72 | 66 | 6 |
      | `strata` | 124 | 95 | 29 |

      **The survivors were not scattered; they were five gaps.** (1) Nothing checked the
      interval *level*: `(1 - level) / 2 * 100` could become `/ 2.2` or `* 110` in both
      `metrics` and `strata` and every figure's band would still pass. (2) The published
      defaults (2000 draws / 200 draws, seed 20261009) were unpinned. (3) The
      primary-lineage rule behind Fig 2D and S4 was tested only for its alphabetical tie-break,
      not for "unverified roots are never chosen" or "most rows wins". (4) The `MIN_ROWS = 20`
      boundary and the bootstrap's "fewer than half defined" rule had no test at 19/20 or at
      half. (5) The 190-pair count, the `1e-32` log floor and `relative_asymmetry`'s value had
      no hand-worked case.
- [x] **Round 2: `tests/test_mutation_survivors.py` (45 tests), 16 Oct 2026.** Every expected
      value is hand-worked, a loop, or a spy on the call. Re-running only that file against
      round 1's survivors (`dev/mutation/compare_rounds.py`): `metrics` 9 of 16 killed,
      `accessibility` 5 of 6, `strata` 27 of 29 after one more test. **Every one left is an
      equivalent mutant, each argued in the test file's docstring:** `outcome > 0.5` on an
      outcome `_check` has already forced to 0 or 1 (three lines), a redundant
      `sort_index(axis=1)` after `pivot`, a digit in an error message, a `1e-6` tolerance
      against `1.1e-6`, and `triu_indices(k=0)`, whose extra diagonal pairs are NaN and
      filtered. None needed a code change; **the modules were right and the tests were thin**.
      Not yet run: the remaining seven modules in the order above.
- [ ] Still open from this line: `dms_sweep`, `symmetry`, `audit`, `codon_learning`,
      `linkfit`, `spectrum_swap`, `antigenic`.
- [ ] **A surviving mutant in arithmetic that feeds a figure is a finding, not a metric.**
      Each one gets either a new test or a written reason why it cannot matter.
- [ ] Expected hot spots, from reading the code: the `alpha` exponent sign in
      `dms_sweep`, the stratum boundary in `nucleotide_steps` (`<=` vs `<`), the direction
      convention in `pair_asymmetry`, the `+16` offset in `antigenic.load_sites`, and the
      bootstrap resampling unit in `strata` (sites vs rows).

### Batch 2 — Remove the self-referential expectations

Six places compute the expected value with the code under test. Each is a genuine
staleness guard and should stay — but each needs an independent check beside it.

| Where | What it actually proves today | What to add |
|---|---|---|
| `test_spectra.py::test_the_engine_recomputes_the_shipped_table` | the table is not stale | a second, hand-written compression on a 2×2 toy spectrum |
| `test_s11.py::test_shipped_fixture_is_exactly_what_the_generator_makes` | the generator is deterministic | the fixture's expected counts written out by hand, once |
| `test_s11.py::test_null_probabilities_take_j1_from_the_data_and_j23_from_the_code` | `q[2]`, `q[3]` agree with a sibling helper | enumerate the 64 codons directly for `q[2]` and `q[3]` |
| `test_s11.py::test_chart_numbers_come_from_the_tables` | bookkeeping is consistent | assert the quantity is the right one, on the planted fixture |
| `test_fig02.py::test_intervals_bracket_the_estimate_and_are_reproducible` | the call is reproducible | keep the bracketing half; move reproducibility to its own test |
| `test_simulation_pipeline.py` (whole file) | the shipped tables have not been swapped | see Batch 7 — these assert a *finding*, not the code |

### Batch 3 — Upgrade the tests that under-claim

- [x] **Done 15 Oct 2026.** The five α optima are now pinned to the grid they are
      reported on (Base 0.00, flat JC69 FT -0.16, Human FT -0.09, SC2 FT -0.13, H3N2 FT
      -0.22), and so is the panel-D diagonal — each fine-tune's correlation with **its
      own** spectrum (H3N2 0.1451, Human 0.1089, SC2 0.1618, to 5e-4). Added with it: a
      test that each fine-tune's strongest one-nucleotide-band correlation *is* its own
      spectrum, which is the claim the figure is in the manuscript to make and which
      nothing asserted. Note it has to be read in the 1-nt band: in panel B, on the full
      matrix, every arm correlates with every spectrum at 0.25–0.50 and the diagonal is
      invisible. The old test is left in place; these sit beside it.
- [ ] `test_s16.py::test_make_s16_execution` is `returncode == 0` plus six `.exists()`.
      Check the content of at least the values table.
- [x] **Done 15 Oct 2026.** Both shapes pinned: the battery frame is 139,802 rows over
      13 lineages with an nt-step split of 43,391 / 71,270 / 25,141, and the DMS table is
      33,862 rows split BA2 7,062 / MA22 9,238 / Perth09 10,754 / XBB15 6,808.
      `len(frame) > 0` passes for a frame that has lost 99% of its rows, and every
      battery result is a mean over them.
- [x] **Renamed 15 Oct 2026** to `test_exporter_modules_import_and_define_a_callable_main`,
      the honest option of the two. The check is worth keeping — a syntax error or a
      missing entry point fails here instead of at the end of a figure build — but
      "valid" promised something it never did. Calling each `main` on a fixture is still
      open; each needs real inputs.
- [ ] `test_report_and_simulation_scripts.py` — checks the HTML report exists and is over
      1 kB. Check its numbers against the tables it reports.
- [ ] `test_enlarge_tree_labels.py::test_finds_the_labels_and_the_legend` — replace the
      magic `len(plan.legend) > 6` with the expected label set.

### Batch 4 — Modules with no adversarial file at all

`test_redteam_*.py` covers `accessibility`, `symmetry`, `metrics`, `strata`, `dms_sweep`,
the guide loader, the shipped data and the figure captions. These have nothing:

- [ ] **`linkfit`** — the softmax exponent is quoted as comparable to the sweep α. Test:
      log-base confusion, knot placement at the data edges, BLAS thread count changing the
      fit, a degenerate single-bin input, and that `threadpoolctl` being absent changes
      only speed (as `requirements.txt` claims) and not the answer.
- [ ] **`audit`** — the S14 interval-width ratios are a headline. Test the permutation null
      on planted and null data, the corpus-membership split on a planted leak, and that the
      lineage-vs-site interval ratio is computed on the same draws.
- [ ] **`antigenic`** — a declared **null** result, which is the easiest kind to break
      silently. Test that a planted antigenic effect *is* detected (otherwise the null is
      just low power), and that the `+16` offset is caught when applied twice or zero times.
- [ ] **`codon_learning`** — `test_codon_learning.py` is already oracle-grade; the gap is
      adversarial input, not arithmetic. Fuzz malformed codon tables and root mismatches.
- [ ] **`spectrum_swap`** — `merge_rounding_ties` is a correctness hazard given `DATA.md`'s
      note that rounding `mut_prob` moves correlations in the third decimal. Test it.
- [ ] **`io`** and **`placement`** — currently only indirect. Small, cheap, do them.
- [ ] **`tools/exports/*`** — schema-level only today.

### Batch 5 — Do the documents agree with the tables?

Every `.md` in this repo quotes numbers. `test_redteam_figures.py` already checks that
*captions* match their values tables. The prose does not get the same treatment, and the
prose is what goes into the manuscript.

- [x] **Started 15 Oct 2026.** `tests/test_document_numbers.py` exists (42 cases). It
      covers `figures/README.md` so far: script coverage both ways (A3), the Figure column
      against the numbering map, and the Fig 3 means. `README.md`, `DATA.md`,
      `SYMMETRY_TESTS.md` and `docs/notes/S3_S11_PROGRESS.md` are **not** covered yet.
- [x] The Fig 3 lineage means (0.9195, 0.7721) are pinned — and pinned *by cell*
      (`site_logistic_auroc`, fine-tuned and base, at alpha = 0), not as "some number in
      the table rounds to this", which would have passed with the two models swapped.
- [ ] Still to do from this line: **every row count in `DATA.md`** against `MANIFEST.tsv`
      and the file on disk. The three spectra percentages (84.8%, 43.6%, 51.0%) turn out
      to be **unverifiable from the shipped set** — they are computed from the
      archive-only spectrum comparison passed with `--results`. Listed as such in
      `UNVERIFIABLE` in the new file rather than skipped quietly; `tests/test_spectra.py`
      checks the figure's own values table instead.
- [ ] Then the harder table-shaped claims: `SYMMETRY_TESTS.md` §3 and §4 are tables of
      numbers with no test behind them at all.
- [x] The `UNVERIFIABLE` dict in `tests/test_document_numbers.py` does this, with a test
      that fails if it is emptied without the claims being checked somewhere. Two entries
      so far: the spectra percentages, and the retired fig02 asymmetry Spearman 0.539.

### Batch 6 — The clean-clone run

Still open from `docs/notes/PLAN.md` Phase 6, and the only real answer to Reviewer 1's
reproducibility complaint.

- [~] **Half of this now exists: `tools/build_figures.py`, 16 Oct 2026.** One command
      rebuilds all 21 steps (no GPU, no network) into a scratch directory and diffs every
      `_values.tsv` and `_caption.md` against `output/` (values to one unit in the sixth
      significant digit, captions as text). Steps run four at a time on one BLAS thread each;
      the whole build is about 9 minutes, almost all of it S12. A composite placed from a scratch
      build uses that build's panels, never the shipped ones (`panelise.py --root`).
      **First real build: 47 of 53 files identical, the other 6 all explained** (A13, A14 and
      the A1/A9 orphans). `tests/test_build_figures.py` (53 tests) pins the step list against
      `figures/` and against the shipped PDFs both ways, the order from the composite layouts,
      the comparison with paired tampered inputs, failure propagation on throwaway scripts, and
      a real partial build.
      **Still not done:** the clone to a machine with no `/home3`. Its first finding is A11:
      before the `.gitignore` fix a clone had no `results/` at all.
- [ ] A clean clone, once `results/` is committed, running `pytest` and `build_figures.py`.
- [ ] Record what breaks. A4 is already known to break it.
- [ ] Add a `--strict-offline` conftest flag that fails any test touching a path outside
      the repo, so A4 cannot recur.

### Batch 7 — Separate "the code is right" from "the finding is real"

`test_simulation_pipeline.py` asserts things like `pct_flip <= 0.032` and
`rho_ft > rho_base + 0.08` by reading shipped result tables. These **cannot fail from a
code bug** — only from someone replacing a table. That is a useful thing to check, but it
is not a test of the pipeline and the file name says it is.

- [ ] Split into `test_simulation_tables.py` (shipped tables unchanged, with the manifest
      hash from A6) and `test_simulation_pipeline.py` (the code, on synthetic input with a
      planted answer).
- [ ] Apply the same split wherever a test's pass/fail depends on a scientific result
      rather than on the arithmetic.

### Batch 8 — Numerical and determinism hygiene

- [x] **Tolerance audit — done 15 Oct 2026.** 180 tolerances across `tests/`; the bulk
      are 1e-12 (42), 1e-6 (17), 5e-5 (14) and 1e-9 (14) and need no defence. Sixteen are
      looser than 1e-3; each was read against the claim it guards and measured.

      **The premise of this item was wrong.** `test_spectra.py`'s `abs=0.05` is not loose:
      the values are percentages (84.8, 43.6, 51.0), so 0.05 is exactly the half-width of
      the one-decimal rounding they are published at. It is the correct tolerance and
      stays. Worth keeping as a reminder that "looks loose" is not a finding until the
      units are checked.

      Three that were genuinely wrong, all now fixed and re-run:

      | Where | Was | Now | Why |
      |---|---|---|---|
      | `test_s06.py::test_golden_against_the_archive` | `abs=0.004` | `abs=5e-4` | Goldens are written to **five** decimals and checked to three: 0.29858 → 0.3025 passed. Measured worst gap between the shipped table and the archive is 3.3e-4, so 5e-4 is the tightest bound the reproduction meets. |
      | `test_dms_sweep.py` planted optimum | `abs=0.06` | `abs=0.01` | Six grid steps of slack for a recovery that is **exact**: `best_alpha` is -0.8 on seeds 0-5. One step kept only against an argmax tie-break change. |
      | `test_s04.py` chemistry control | `controlled == approx(plain, abs=0.08)` | both exponents pinned to 5e-5 | Not merely loose, **misleading**: it reads as "the control barely moves it", and at two nucleotide changes the exponent *rises 26%* under the control (0.2029 → 0.2548). The band hid the single most interesting number in the panel. The claim S4 actually supports is the `lo > 0` assertion beside it, which stays. |

      Still to judge, all bands rather than goldens and all plausibly right:
      `test_metrics.py`'s `abs=0.03`/`0.06` (chance level on random data — a band is
      correct), `test_s16.py`'s `atol=0.2` on a sum of rounded percentages, and
      `test_s13.py:161`'s `approx(0.9, abs=0.05)`.
- [ ] **Thread determinism.** Run the suite at 1, 4 and 14 BLAS threads and diff. Any test
      that moves is a test whose golden is not really pinned.
- [ ] **Re-run determinism.** Run the suite twice in one session and diff — catches state
      leaking between tests through module-scoped fixtures (several files hold the 4.4 MB
      13-lineage table at module scope).
- [ ] **Seed sensitivity.** For each seeded bootstrap, confirm a different seed moves the
      answer by less than the interval it reports. If it moves more, the draw count is too
      low — this already bit the S14 interval ratios once (`docs/notes/PLAN.md`).

---

## D. Suite infrastructure

- [ ] **Add a `--runslow` gate.** `pytest.ini` declares a `slow` marker but nothing skips
      it, so all 12 run by default. A fast default makes the suite usable mid-edit.
- [ ] **Decide about `pytest-xdist`.** Four files hold the 4.4 MB table at module scope; per
      worker that is 4 copies. Measure before adopting, and cap at 14 workers.
- [ ] **Register the `dev/fig4_gauge*` suites or retire them.** `docs/notes/PLAN.md` notes
      `testpaths = tests` hides 150 passing tests from a bare `pytest`. They are either
      part of the deliverable or they are not.
- [~] **One CI entry point** that runs: suite, figure build, clean-clone diff. The figure build
      is now `tools/build_figures.py` (exit status 1 on any failed step or any drift from
      `output/`); the suite is `pytest`; the clone is still to do (Batch 6).

---

## E. Order of work

0. **A1b, the renumbering** — now the largest single item, and it blocks a green suite.
   Four stages, suite run between each. It displaced the rest of this plan on 15 Oct;
   that was the right call only because the manuscript ordering has to be right before
   any of the numbers are checked against it.
1. **A2–A4** — half a day, independent of the renumbering.
2. **Batch 1 on `accessibility`, `strata`, `metrics`** — the three modules under the most
   published numbers. The survivor count here decides how much of Batches 2–4 is urgent.
3. **Batch 5's cheap half** (row counts, the three spectra numbers, the Fig 3 means) — a
   few hours, and it protects the manuscript text directly.
4. **Batch 6** — the clean clone, because it is what a reviewer will do.
5. Everything else, by what Batch 1 turned up.

Record results in §F as they land, newest last, same style as `docs/notes/S3_S11_PROGRESS.md`.

## F. Results

Newest last.

| Date | What | Outcome |
|---|---|---|
| 15 Oct | Baseline suite run (§0) | 1,758 collected, 1 failed — `test_every_output_file_has_a_producer`, a true positive |
| 15 Oct | Coverage survey of all 16 modules and 50 test files | Fed §3 and §4: six self-referential expectations, eight tests weaker than their names, five modules with no adversarial file |
| 15 Oct | A1 investigated | Not an orphan: a supplementary **table** numbering collision, two tables both labelled S16. Opened §A1b |
| 15 Oct | A2 investigated | Confirmed: **no `xfail` marker exists anywhere in `tests/`**, though `test_known_defects.py`'s docstring describes a strict-xfail ratchet. All six tests pass today |
| 15 Oct | A3 investigated | Confirmed: `figures/README.md` documents `make_s15.py` (file is `make_s16.py`) and lists `make_s04.py` twice for two different figures |
| 15 Oct | Draft supplementary ordering supplied by Oscar | Mapped to repo stems in §A1b; four figures move, S12↔S13 swap, lineage-set validity dropped |
| 15 Oct | Numbering decisions taken | Lineage-set validity retired; numbers decoupled from filenames via a map file |
| 15 Oct | **A1b stage 1 implemented** | `guides/manuscript/figure_numbers.yaml` + `mutacc/numbering.py` + `tests/test_numbering.py` (70 cases). Map lives under `guides/manuscript/`, not `guides/`: anything directly in `guides/` is parsed as a run guide and broke collection. Full suite after: green apart from the A1 true positive |
| 15 Oct | **A2 resolved** | Not an unpinned ratchet. Both code defects are **already fixed** in `mutacc/`; only the docstring was stale, and no `xfail` is wanted on a passing test. The "incompatible" pair is compatible precisely because of the fix (an orthogonal projection cannot overshoot the total about the same origin). The fix moved no shipped number: the only consumer, `make_fig02.py`, is retired |
| 15 Oct | **A3 fixed** | The two drifts the audit named were already repaired; the live ones were two scripts with **no row at all** (`make_fig02d.py`, `make_s01.py`) and a retired `make_fig02.py` still billed as "Figure 2". Now pinned by `tests/test_document_numbers.py` in both directions |
| 15 Oct | **Batch 5 opened** | `tests/test_document_numbers.py` (42 cases): README coverage, README numbers vs the map, Fig 3 means pinned by cell. The three spectra percentages are archive-only and are now *listed* as unverifiable instead of absent |

| 15 Oct | **A1b stage 2 implemented** | `write_caption` numbers from the map, refuses a caption that names itself, and gives a retired figure none. `make_fig04.py`'s hard-coded `FIGURE_LABELS` dict deleted. 19 shipped captions rewritten; the S12↔S13 swap and the S8/S14 moves have reached the text |
| 15 Oct | Stage 2 broke six tests, all one assumption | A figure's output stem is **not** fixed: `make_s11.py` builds a `_SYNTHETIC_FIXTURE` variant. Derived stems now resolve to the figure they extend. Separately, the number was being printed in front of a "do not quote" banner — `write_caption` grew `banner=` to keep the warning first |
| 15 Oct | **A9 found** | `fig02d_strata_panel_caption.md` has no generator, and the producer test structurally cannot see it. Pinned `xfail(strict=True)` — the first real use of the ratchet A2 found missing |
| 15 Oct | **A4 contained** | The out-of-repo read now skips with its reason on a clean checkout instead of failing. Ship-or-drop still open; the input is 1.65 MB plus four arm tables and a FASTA |
| 15 Oct | Two things the plan did not account for | `clean_repo` is **untracked by git** (0 files in `git ls-files`), so none of this work is recoverable; and the tree is being edited concurrently — `make_s15_lineage_panel.py` and an S15 map entry appeared mid-session |

| 15 Oct | Stage 2 verified | `test_s12`, `test_s13`, `test_s16`, `test_s11`, `test_fig04`, `test_output_provenance`, `test_known_defects`, `test_document_numbers`, `test_numbering` all green bar the A1 true positive. All 18 shipped captions round-trip byte-identically through `write_caption`, so the next figure build reproduces them exactly |
| 15 Oct | **Batch 8 tolerance audit done** | 180 tolerances read. Three wrong and fixed (`test_s06` 4e-3 → 5e-4, `test_dms_sweep` 0.06 → 0.01, `test_s04`'s misleading band → pinned exponents). The item's own example was wrong: `test_spectra`'s `abs=0.05` is the correct rounding half-width for a percentage quoted to one decimal |
| 15 Oct | A finding the loose band was hiding | Under the chemistry control the fine-tuned exponent at two nucleotide changes **rises 26%** (0.2029 → 0.2548). `approx(plain, abs=0.08)` read as "barely moves it". Not a code defect — a claim the test was mis-stating |
| 15 Oct | **Batch 3 part done** | S16's "golden values" test now has goldens (5 α optima, the panel-D diagonal) plus the claim nothing asserted: each fine-tune's strongest 1-nt-band correlation is with its own spectrum. The two symmetry-runner shapes pinned (139,802 and 33,862 rows, broken down). One over-claiming test renamed |

| 16 Oct | Suite re-run, as inherited | 2 failures: the A1 true positive, and a **new** one, `dev/mutation/run_all.sh` carrying `/home3` (A15). 460 s of the run is one test, `test_s12.py::test_script_renders_with_its_tables` |
| 16 Oct | `tools/build_figures.py` written | Answers "is there a runner": there was not. Its first build also found a panelise limitation (panel paths must be repo-relative), fixed with `panelise.py --root` rather than by loosening the rule |
| 16 Oct | First full rebuild | 21 steps, 53 values/caption files compared. Found A12 (exporter dropping SARS-CoV-2), A13 (stale alpha-optima table), A14 (two un-regenerated captions); the S11 step first lacked two table arguments the shipped caption used, which is how the README's own S11 command was found to be incomplete |
| 16 Oct | **A11 found while explaining why a clone would fail** | `results/` was never committed |
| 16 Oct | Mutation round 2 | See Batch 1: 41 of 51 round-1 survivors killed, the other 10 argued equivalent. 45 new tests |
| 16 Oct | README | Gained the rebuild section and the five commands it was missing (S1, lineage panel, simulation control, the two panel generators); `test_every_figure_command_in_the_readme_is_exactly_a_runner_step` pins it both ways |

**Suite now:** ~1,920 collected. One failure —
`test_every_output_file_has_a_producer`, still correct — plus one `xfail(strict=True)`
(A9).

**Not started:** Batches 1, 2, 4, 6, 7, the rest of 3, and §D.

**Next, and why in this order:**

1. **Batch 1 on `accessibility`, `strata`, `metrics`.** Still the item that decides how
   much of the rest matters. **Not started deliberately:** the box was carrying three
   concurrent pytest runs all afternoon, and a per-module mutation run is the heaviest
   thing in this plan. It needs a quiet machine.
2. **Stage 3 of the renumbering — hold.** It renames ~20 scripts and ~80 output files,
   and two conditions make that reckless right now: `clean_repo` is **untracked by git**,
   so there is no undo, and the tree is being edited by another session. Put it under
   version control first. Stages 1 and 2 already deliver the benefit that mattered — the
   number is in one file and the captions read it — so stage 3 is now cosmetic and can
   wait for a quiet tree.
3. **The rest of Batch 3** — cheap, file-at-a-time, no coordination needed. Left:
   `test_s16.py::test_make_s16_execution` (returncode plus six `.exists()`),
   `test_report_and_simulation_scripts.py` (an HTML report checked only for being over
   1 kB) and `test_enlarge_tree_labels.py`'s magic `len(plan.legend) > 6`.
