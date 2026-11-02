# One source: the influenza database

> **Status: plan. Nothing in §5 onwards is implemented.** Opened 15 Oct 2026.
> Tracked as [`PLAN.md`](PLAN.md) §5 Phase 7; risks 23–25; decision D10.
>
> **This is a working note, not a reviewer-facing document.** It was linked from the
> README until 2 Nov 2026, which put a to-do list with risk numbers and an absolute path
> in front of anyone checking the paper's data provenance. What a reader actually needs —
> which database build the shipped tables came from, the toolkit that built it, and the
> metadata defects that shape the figures — is now [`../../DB_SOURCING.md`](../../DB_SOURCING.md),
> with the build stamp itself in [`../../data/db_build/`](../../data/db_build/).
>
> **Two things in §1 have since been fixed and are no longer true as written.** The rename
> (§1.1) is handled: `tools/h3_numbering_from_db.py` takes `--db` with no default and
> documents `$IAV_DB`, so no code in the repository names `iav-db.rebuilt.db` any more, and
> "nothing that reads the database can currently run as documented" is stale. The S11 tree
> disagreement (§1.3) is annotated in `S3_S11_PROGRESS.md`, though the test that would catch
> a recurrence (§7.7) is still unwritten. §1.2 stands: the per-sequence inputs are still not
> in this repository, and cannot be — they are GISAID.

Everything downstream of a sequence — collection dates, subtype, host, segment,
clade label, the tip set a tree is pruned to — should come from one place:

```
/home3/oml4h/RABV-gTK/test_out/IAV_DB_genbank_n_gisaid_rebuild/db_build/iav-db.db
```

33,248,444,416 bytes, mtime 2026-10-15 13:45. Read-only, always.

---

## 1. Why now

Three things are true at once:

1. **The database the repo names no longer exists.** `iav-db.rebuilt.db` is gone —
   it was renamed into place as `iav-db.db` on 15 Oct. Three documents and one tool
   still name the old file: [`PLAN.md`](PLAN.md) §3.2 and the 15 Oct progress entry,
   [`S3_S11_PROGRESS.md`](S3_S11_PROGRESS.md) "Inputs", and the usage example in
   `tools/h3_numbering_from_db.py`. **Nothing in the repo that reads the database
   can currently run as documented.**
2. **The per-sequence inputs that built S15 are not in this repo.** They were staged
   in the *archive's* gitignored `dev/lineage_panel/`, and `clean_repo/dev/` holds
   only `retired_figures/` and `spectrum_swap_anomaly/`. The lineage panel cannot be
   rebuilt from a clean checkout today — which is the reviewer's reproducibility
   complaint, in miniature.
3. **Two documents disagree about which tree S11 used.**
   `S3_S11_PROGRESS.md` describes `h3_seg4_tree.pb`, 67,879 tips, 22 Sep. The shipped
   `results/s11_h3n2_ha_tree/MANIFEST.tsv` says `h3n2_1968.pb`, 224,731 tips, pruned
   from the treepatch repair `usher.pb` of 24 Sep. The manifest is right and the
   progress document is describing a superseded run. Nothing caught this because
   no test compares a run log to a manifest.

## 2. Inventory: where metadata comes from today

| What | Current source | Ships? | Reproducible from a clean checkout? |
|---|---|---|---|
| Per-sequence collection dates (13 lineages) | `--dates` dump of `meta_data`, 573k records, staged in the *archive's* `dev/lineage_panel/` | no | **no — the file is not here** |
| Lineage membership per accession | `--accessions`, built from the sub-alignments | no | **no** |
| Per-lineage sequence counts, sites-vs-root | `panel_metadata.tsv` from the pipeline run | no | no (archive) |
| S11 tip set: H3N2 only, dated, 1968 onward | baked into `h3n2_1968.pb`, pruned outside this repo | no | **no** |
| S11 per-year tip counts (`year_counts`, 59 rows) | the same pruning step | derived table only | **no** |
| UShER reference record `DQ864721` | `sequence_alignment` of `iav-db.rebuilt.db` | no | **no — DB renamed** |
| H3 mature numbering | 300 sampled GenBank tips, `tools/h3_numbering_from_db.py --db iav-db.rebuilt.db` | derived table only | **no — DB renamed** |
| Lineage root sequences | `data/roots/` | yes | yes |
| Spectra | `data/spectra/` | yes | yes |
| Viral panel | UniProt, `data/viral_panel/` | yes | yes |

The bottom three rows are fine and are out of scope. Everything above them is a
different path to the same underlying facts, and no two of them were checked
against each other.

## 3. What the database holds

Verified against the live schema, 15 Oct 2026.

| Need | Table.column | Notes |
|---|---|---|
| Collection date | `meta_data.collection_date` | plus `collection_day` / `collection_mon` / `collection_year`, and `collection_date_validated` |
| Strain / accession | `strain`, `Parsed_strain`, `primary_accession`, `accession_version`, `Isolate_Id` | `Isolate_Id` is empty on GenBank rows |
| Subtype | `serotype_validated` | also `nearest_reference_subtype` |
| Segment | `segment_validated`, `segment_name`, `segment_declared`, `segment_submitted` | validated vs declared can disagree — that is the point of having both |
| Host | `host_scientific_name`, `host_validated`, `host_taxa_id` | |
| Clade | `Clade`, `Lineage` | **GISAID-only; empty in every GenBank row sampled — see D10** |
| QC | `exclusion_status`, `exclusion`, `exclusion_criteria` | `0` = retained |
| Provenance | `dataset_source` | `genbank` / GISAID |
| Nucleotide sequence | `sequences.sequence` on `sequences.header` | indexed |
| Aligned row | `sequence_alignment.alignment` on `(primary_accession, alignment_name, segment)` | unique index |
| CDS coordinates | `features.cds_start_OG_seq`, `cds_end_OG_seq`, `product` | indexed on `accession` |
| Trees | `trees.newick` | **newick only, no `.pb`** — see §6 |

Scale, from `db_build/db_summary.txt` (generated 15 Oct 13:12, after curation):
3,947,146 sequences loaded, 3,908,177 passing QC; segment 4 is 572,005 passed and
1,302 failed.

## 4. Three properties of this file that shape the design

**(a) `meta_data` has no index.** Indexes exist only on `sequences(header)`,
`features(accession)` and `sequence_alignment(…)`. Any `WHERE strain = …` or
`WHERE serotype_validated = 'H3N2'` is a full scan of a 33 GB row-store. That is
minutes of sustained I/O on a shared server, per query. We must not add an index:
it is not our database and it is live.

**(b) The file is being written to.** `iav-db.db-shm` and `iav-db.db-wal` are both
present and the shm was touched at 18:05 today. Results could shift under a
long-running analysis, and two stages of one run could see different data.

**(c) `info` is not a version stamp.** `SELECT * FROM info` returns
`new db | 2026-08-26 14:19:29`, but the file is dated 15 Oct and `db_summary.txt`
says it was generated that day from `iav-db.rebuilt.db` after "part 1–3 curation".
**Do not use `info.date` to decide whether the database has changed.** Use the file
size, the mtime, and content hashes of the extracted rows.

Together these mean: **one guarded extraction pass, then a local snapshot**, not
live queries scattered through the figure code. "Query the database directly" is
satisfied by having exactly one piece of code that touches it, with its output
pinned, rather than by every tool opening the file.

## 5. Proposed shape

```
mutacc/iavdb.py              the only module that opens the big file. Read-only URI,
                             named queries, no SQL anywhere else in the repo.
tools/export_db_snapshot.py  one pass -> dev/db_snapshot/
dev/db_snapshot/seg4.sqlite  human H3N2 segment 4: accession, dates, host, clade,
                             subtype, QC flag. Indexed. Gitignored.
dev/db_snapshot/MANIFEST.json  source path, size, mtime, row counts, per-query
                             SHA-256, extraction timestamp, tool version.
```

Rules:

- Every shipped table's `MANIFEST.tsv` row gains the snapshot hash, so any number
  in the paper can be traced to one state of one file.
- `mutacc/iavdb.py` refuses to open the database in anything but read-only URI mode.
- The existing "no absolute paths" test (`PLAN.md` §5 Phase 6) must not be weakened:
  the database path is configuration, read from an environment variable or a
  command-line argument with no default pointing at `/home3`.
- Nothing heavy without `nice -n 10 ionice -c3`; never more than 14 threads.

## 6. What the database cannot give us

Say these plainly rather than discovering them mid-migration.

1. **No `.pb`.** The `trees` table holds newick strings only. S11 needs per-branch
   mutations, which is a mutation-annotated tree. The 224,731-tip `h3n2_1968.pb`
   therefore stays an external input. What *can* move into the database is
   everything that defines its tip set — H3N2, dated, 1968 onward — so the pruning
   becomes a reproducible query instead of a step someone remembers doing.
2. **No protein sequences.** `sequences` is nucleotide. Translation via `features`
   coordinates becomes our code, and needs its own tests (§7.5).
3. **Possibly no clade labels.** See D10.

## 7. Consistency tests at import

New file `tests/test_iavdb_import.py`. It runs against `dev/db_snapshot/`, not the
33 GB source, so the suite stays at its current ~1.5 minutes; the two tests that
must touch the source are marked `slow`. If the snapshot is absent the module
skips, so a clean clone with no `/home3` access still runs the rest of the suite —
the same convention the repo already uses for the archive-dependent tools.

**7.1 Provenance** *(slow)*
- Source file exists, is readable, and its size and mtime match `MANIFEST.json`.
  A mismatch **fails** with "snapshot is stale, re-extract" — this is the test that
  stops us quietly analysing a database that was rebuilt under us, which is exactly
  what happened between 8 and 15 Oct.
- Every table and column `mutacc/iavdb.py` names is present. Catches a rename
  upstream in RABV-gTK as an error rather than as a wrong number.
- `info` is recorded but explicitly **not** trusted as a version (§4c). A test
  asserts no code branches on it.

**7.2 Row identity**
- Every requested accession resolves to exactly one `meta_data` row and one
  `sequences` row for its segment.
- No duplicate `(accession, segment)`.
- A requested accession that is missing fails loudly. The repo already has
  `test_data_integrity.py` guarding silent empty joins; same failure mode, new source.
- Snapshot row counts match `MANIFEST.json` and match `db_summary.txt`'s segment-4
  figure (572,005 passing) to within a stated tolerance, or the difference is
  explained by the filters.

**7.3 Filters hold**
For every row in the snapshot: `segment_validated == 4` **and**
`segment_name == 'HA'` agree; `serotype_validated == 'H3N2'`;
`host_scientific_name == 'Homo sapiens'`; `exclusion_status == 0`;
`dataset_source` is one of the known values and is kept per row.

The segment agreement check is not pedantry — a mislabelled reference in this
database has already cost a stage re-run elsewhere.

**7.4 Dates** — the main reason for doing this
- Every `collection_date` either parses or is flagged partial. **No silent
  coercion**: a bare `2024` must stay visibly month-unknown, never become 1 January.
- `collection_year == year(collection_date)` wherever both exist;
  `collection_day` / `collection_mon` consistent with `collection_date`.
- Date within 1968 … today; no future dates.
- **The three defects `tools/export_lineage_panel.py` already documents become
  tests, counted rather than described**: mixed formats (ISO, `12-Dec-2024`,
  `2024-12`, bare `2024`); partial dates; and records dated long before their own
  lineage, including the two 1905 rows that are a year read as a spreadsheet serial.
  The test pins the *count* of each class. If the count moves, the database changed
  and we are told, instead of a percentile span quietly shifting.
- The percentile-span logic stays. It is the right answer to dirty dates and it
  survives this migration unchanged.

**7.5 CDS and translation**
- A `features` row exists with an HA `product`; coordinates lie inside the
  nucleotide sequence; CDS length is a multiple of 3; no internal stop codon.
- Round-trip on a sample: nucleotide → CDS → translate, compared against a
  known-good protein.

**7.6 Determinism** *(slow)*
- Two extractions against an unchanged database give a byte-identical snapshot and
  identical manifest hashes. Needs an explicit `ORDER BY` on every query and a fixed
  column order.

**7.7 Cross-document agreement** — the test that would have caught §1.3
- The tip count, tree file and date in every run log match the corresponding
  `MANIFEST.tsv`. `tests/test_document_numbers.py` already checks numbers across
  documents; extend it to the run logs rather than writing a second mechanism.

**7.8 Migration gate**
Every number the database currently feeds must come out the same, or be listed in
advance in §9. Specifically: the thirteen `lineage_panel` rows (counts, first/last,
percentile span, `n_outside_span`), the 78,659 total in
[`LINEAGE_VALIDITY.md`](../../LINEAGE_VALIDITY.md), the H3 numbering map for all 553
reference codons with landmarks Y98 / W153 / H183 / Y195 passing, and the
`DQ864721` aligned row (1,659 bases, 553 codons).

This gate is temporary and gets deleted once the snapshot is the only source.

## 8. Bottom-to-top re-run

The ordering matters: nothing re-runs until the baseline is frozen, and the GPU
stages do not re-run at all unless a sequence changed.

| Stage | What | Where |
|---|---|---|
| **0** | **Freeze the baseline.** Clean tree, record `git rev-parse HEAD`, run the full figure set and suite, copy `results/` + `output/` to `dev/baseline/2026-10-15/`, `sha256sum` everything. | clean_repo |
| 1 | Build the snapshot and manifest. One pass, `nice -n 10 ionice -c3`. Report clade coverage (decides D10). | clean_repo |
| 2 | Port `tools/h3_numbering_from_db.py` and `tools/export_lineage_panel.py` to `mutacc/iavdb.py`. Re-derive the numbering map and the thirteen-row panel. | clean_repo |
| 3 | Re-derive the S11 tip set as a query; check it reproduces the 224,731 tips of `h3n2_1968.pb`. **The `.pb` itself is not rebuilt** (§6.1). | clean_repo |
| 4 | Re-run every export in `tools/`, regenerating `MANIFEST.tsv` with snapshot hashes. | clean_repo |
| 5 | Re-run the figure set and the suite. | clean_repo |
| 6 | Compare against the baseline; write the result into §9 and the PLAN progress log. | clean_repo |

No GPU stage is in this list. Model scores do not depend on metadata, so if a
score changes, something is wrong — that is itself a check (§9).

## 9. Confirming values do not change

`tools/compare_baseline.py` takes two output directories and prints one pass/fail
table, exiting non-zero on any FAIL.

**Must be bit-identical.** Every shipped table not derived from the database:
`replacement_scores`, `replacement_observed`, `alpha_sweep_by_lineage`, `dms_*`,
`spectra_*`, `viral_panel_*`, the simulation tables. These never touched the
database, so a single changed byte means the migration leaked somewhere it should
not have. Compared by SHA-256.

**Must be identical in value.** The database-derived tables: `lineage_panel`
(13 rows), the H3 numbering map, `column_codons`, `baseline_codons`. Compared
field by field, not by hash, because `MANIFEST.tsv` will legitimately gain a
snapshot-hash column.

**Figures.** The existing `*_values.tsv` beside every panel is the comparison
surface — no image diffing. Each must match to the precision the suite already
asserts (1e-12 on goldens; note risk 24 — no library version is pinned, so pin the
environment before the baseline run or that tolerance is unenforceable).

**Documented exceptions, written down now so they cannot be rationalised later:**

1. `MANIFEST.tsv` gains a snapshot-hash column in every run. Expected.
2. If the database has genuinely been re-curated since the 15 Oct rebuild, the
   lineage panel's dates may move. This must be reported as *the database changed*,
   with the before/after on all thirteen rows and the `n_outside_span` counts — not
   folded silently into a re-run.
3. If D10 resolves to Nextclade labels, each relabelled lineage is listed
   individually with both labels.

Anything else that moves is a defect until shown otherwise. The accepted
comparison table gets pasted into the PLAN progress log as the record that the
migration was verified.

## 10. Rollback

Each stage is its own commit. `dev/baseline/2026-10-15/SHA256SUMS` is the fallback:
if the comparison fails and the cause is not quickly explicable, restore the
baseline tables and diagnose from the snapshot rather than pushing on. The
snapshot is small, so a failed migration leaves a diagnosable artefact behind
instead of a 33 GB file that has since moved on.
