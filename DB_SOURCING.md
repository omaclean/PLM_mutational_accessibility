# The sequence database behind the analysis

Every sequence-derived number in this repository — collection dates, subtype, host,
segment, the tip set a tree was pruned to — comes from one influenza sequence database.
This document says which one, what it supplied, which of its defects reach the figures,
and what a reader can and cannot rebuild from a clean checkout.

None of it is needed to redraw a figure. `tools/build_figures.py` reads only the shipped
tables under `results/`, and no figure script opens a database. This is provenance for the
tables those figures read.

## 1. Where the sequences came from

Sequences for influenza A were obtained from GenBank and GISAID EpiFlu using the Centre for
Virus Research viral genome toolkit, **V-gTK**:

* code — <https://github.com/centre-for-virus-research/V-gTK>
* archived release — DOI [10.5281/zenodo.23504636](https://doi.org/10.5281/zenodo.23504636)

V-gTK downloads, curates and aligns the records and assembles them into a single SQLite
database with per-sequence metadata, per-segment alignments and CDS feature coordinates.
The build used throughout this repository is `IAV_DB_genbank_n_gisaid_rebuild`, a 33 GB
file dated **15 October 2026**, holding 3,947,146 loaded sequences of which 3,908,177 pass
quality control; segment 4 (HA), the only segment used here, contributes 572,005 passing
and 1,302 failing.

**The build stamp ships: [`data/db_build/`](data/db_build/).** `BUILD.tsv` carries the
toolkit DOI and the file's identity — basename, 33,248,444,416 bytes, mtime
2026-10-15 13:45 — and `db_summary_head.txt` carries the per-segment QC table above,
verbatim from the build's own summary. That is the record against which a future rebuild
can be compared, rather than silently replacing this one.

The database itself is **not redistributable**: it contains GISAID records, whose terms
forbid it. [`LICENSE-DATA.md`](LICENSE-DATA.md) lists every input in that position.

Three properties of the file are worth stating, because they shape what the repository does
with it rather than being incidental:

* **`meta_data` carries no index.** Indexes exist on `sequences(header)`,
  `features(accession)` and `sequence_alignment(…)` only, so any query filtering on strain,
  subtype or host is a full scan of a 33 GB row-store. Reads from this repository are
  therefore single-row lookups on indexed keys, or one batched pass — never a scan per tool.
* **The file is live.** It has a `-wal`/`-shm` pair and is written to by the upstream
  pipeline. Every read is opened read-only.
* **The `info` table is not a version stamp.** It reads `new db | 2026-08-26` on a file
  rebuilt on 15 October. Nothing may branch on it; size and mtime are the identity, which is
  why `BUILD.tsv` records those.

## 2. What the database supplied

| What it supplied | Table and column | Where it ends up |
|---|---|---|
| Per-sequence collection dates, 13 lineages | `meta_data.collection_date`, with `collection_day` / `_mon` / `_year` | the thirteen-row `lineage_panel` aggregate (S15) |
| Subtype, host, segment | `serotype_validated`, `host_scientific_name`, `segment_validated` / `segment_name` | the filters defining the H3N2 human HA set |
| QC flag and provenance | `exclusion_status` (`0` = retained), `dataset_source` | which records enter a lineage at all |
| UShER reference record `DQ864721` | `sequence_alignment.alignment` | the reference row the S11 codon tables are indexed against |
| Mature H3 numbering | `sequence_alignment`, 300 sampled GenBank tips | `column_codons` in `results/s11_h3n2_ha_tree/`, 553 codons |
| CDS coordinates | `features.cds_start_OG_seq`, `cds_end_OG_seq`, `product` | translation of nucleotide rows to protein |

Two tools in this repository read it, both taking the path as an argument with **no
default**, so neither depends on a path that exists on one machine:

```bash
export IAV_DB=<your build>/db_build/iav-db.db
python tools/h3_numbering_from_db.py --db "$IAV_DB" --ref-fasta … --tips … --out …
python tools/export_lineage_panel.py --dates … --accessions … --panel-metadata … --out-dir …
```

`h3_numbering_from_db.py` refuses to write unless the sampled consensus carries the
conserved receptor-binding residues **Y98, W153, H183, Y195** at the numbers its counting
gives them — the check that stops a silent off-by-one in every H3 site number downstream.

## 3. Defects in the source metadata, and what they cost

These are properties of the deposited records, not of the parsing here. They are stated
because two of them visibly shape a published figure.

**Collection dates are not clean.** `tools/export_lineage_panel.py` handles three kinds of
defect explicitly rather than silently:

* **mixed formats** — ISO (`2024-12-12`), `12-Dec-2024`, `2024-12`, and bare `2024`;
* **partial dates** — counted, but excluded from first/last, since a bare year would
  otherwise drag a lineage's first date back to 1 January;
* **eight records dated a decade or more before their own lineage** — two in G.1.1, four in
  J.2, two in J. The strain name carries the true year in all eight. Four land in 1905,
  where the collection *year* has been read as an Excel 1900 day-serial: serial 2021 is
  1905-07-13 and serial 2025 is 1905-07-17, which is exactly what `A/Lisboa/12/2021` and
  `A/Baltimore/JH-1150/2025` carry. 277 records across the whole database share that
  artefact. Of the remaining four, three are GISAID and one is GenBank (`PV150037`, dated
  2012 but a J.2 sequence, with day, month, year and strain name all consistently wrong
  upstream).

**This is why `lineage_panel` carries both an absolute first/last and a 1st-to-99th
percentile span, and why the caption says to quote the percentile span.** It needs no
judgement call about which records to delete, and one bad date cannot move it.
`n_outside_span` reports how many records fall outside it, so nothing is hidden.

**Clade labels are GISAID-only.** `meta_data.Clade` and `Lineage` were populated on GISAID
rows and empty in every GenBank row sampled, so the database cannot supply a clade label
across the whole set. Lineage membership here comes from the sub-alignments, not from this
column.

**`segment_validated` and `segment_declared` can disagree**, which is the point of the
database carrying both: the validated call is a BLAST result, the declared one is the
submitter's. A mislabelled reference elsewhere in this project has already cost a stage
re-run, so segment agreement is checked rather than assumed.

## 4. What ships, and what cannot

What ships is aggregate only. No accession, no per-sequence date, nothing that identifies
a GISAID record:

| Shipped | Rows | Built from |
|---|---|---|
| `results/h3n2_13lineage/lineage_panel.tsv.gz` | 13 | per-sequence accessions and dates, plus the pipeline's `panel_metadata.tsv` |
| `results/s11_h3n2_ha_tree/column_codons.tsv.gz` | 553 | the `DQ864721` aligned row and 300 sampled GenBank tips |
| `results/s11_h3n2_ha_tree/year_counts.tsv.gz` | 59 | collection years over the tree's tips |

Each names its source in its directory's `MANIFEST.tsv`.

**Three things cannot be rebuilt from a clean checkout, and saying so is the honest
position:**

1. **The per-sequence inputs to `lineage_panel`** — accessions and collection dates for
   78,659 sequences across thirteen lineages. They are GISAID identifiers and dates and
   will never ship. The thirteen-row aggregate is the redistributable form.
2. **The mutation-annotated tree.** S11 needs per-branch mutations, which means a `.pb`.
   The database's `trees` table holds **newick strings only**, so `h3n2_1968.pb` —
   224,731 tips, H3N2, dated, 1968 onward, pruned from the treepatch repair `usher.pb` of
   24 September 2026 — is an external input. What the database *could* supply is everything
   defining its tip set, which would make the pruning a reproducible query; that has not
   been done.
3. **Protein sequences.** `sequences` holds nucleotide only; translation via `features`
   coordinates is this project's code.

Consolidating all of the above onto one guarded extraction pass with a pinned local
snapshot, consistency tests at import, and a bottom-to-top re-run confirming no value
changes, is planned but **not implemented**. The specification is a working note:
[`docs/notes/db_sourcing_plan.md`](docs/notes/db_sourcing_plan.md). Nothing in this
document depends on it.
