# Which influenza database build the derived tables came from

Every sequence-derived number in this repository — collection dates, subtype, host,
segment, the S11 tip set — traces back to one influenza sequence database, assembled from
GenBank and GISAID EpiFlu with **V-gTK**
([code](https://github.com/centre-for-virus-research/V-gTK),
DOI [10.5281/zenodo.23504636](https://doi.org/10.5281/zenodo.23504636)). The database
itself is 33 GB and is not redistributable: it holds GISAID records.

What ships here is the build's identity and its quality-control counts, so that a reviewer
can say *which* build produced the shipped tables, and so a future rebuild can be compared
against it rather than silently replacing it.

| File | What it is |
|---|---|
| `BUILD.tsv` | The build stamp: toolkit and DOI, file identity (basename, bytes, mtime), and the sequence counts below. One `key`/`value`/`note` row each. |
| `db_summary_head.txt` | The header and per-segment QC table of the build's own summary, verbatim except that the absolute source path is redacted to a basename. |

`db_summary_head.txt` is the first 35 lines of the build's `db_summary.txt`. **The rest of
that file does not ship**: its 6,300-line exclusion-reason table names individual GenBank
accessions and GISAID `EPI` identifiers, which [`../../LICENSE-DATA.md`](../../LICENSE-DATA.md)
keeps out of this repository. Nothing in the shipped excerpt names a sequence.

Two things the excerpt will look odd about unless you know them, both explained in
[`../../DB_SOURCING.md`](../../DB_SOURCING.md):

* it names `iav-db.rebuilt.db`, because the summary was generated at 13:12 on 15 Oct 2026
  and the file was renamed to `iav-db.db` at 13:45 the same day — one file, two names;
* the database's own `info` table reads `2026-08-26` on a file rebuilt on 15 Oct, so it is
  not a version stamp and no code may branch on it. Size and mtime are the identity.

No code in this repository reads these files; they are provenance for a reader. The two
tools that read the database take its path as an argument (`--db`, or `$IAV_DB`) with no
default, so nothing here depends on a path that exists only on one machine.
