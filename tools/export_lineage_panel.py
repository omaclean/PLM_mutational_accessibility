"""Build the per-lineage panel table: how big each lineage is, when it was sampled,
and how much diversity it carries.

    python tools/export_lineage_panel.py \
        --accessions dev/lineage_panel/clade_accessions.tsv \
        --dates dev/lineage_panel/seg4_dates.tsv \
        --panel-metadata <archive>/results/nextstrain_13lin_dense/tables/panel_metadata.tsv \
        --observed results/h3n2_13lineage/replacement_observed.tsv.gz \
        --out-dir results/h3n2_13lineage

Four inputs, because no single existing table holds all of it:

``--accessions``      two columns, accession and lineage, one row per sequence. Built from
                      the lineage sub-alignments; the accessions are GISAID and GenBank ids
                      and **this file does not ship**.
``--dates``           three or more columns from the influenza database's ``meta_data``:
                      accession, isolate id, collection date. Also does not ship -- it is a
                      dump of per-sequence metadata for 573k records.
``--panel-metadata``  the pipeline's own per-lineage record: sequence count and the number
                      of sites that differ from the lineage root.
``--observed``        one row per (lineage, site, amino acid) with a 0/1 for whether that
                      replacement was seen, which gives the replacement count.

What ships is only the aggregate written here: thirteen rows of counts and dates. No
accession, no per-sequence date, nothing that identifies a GISAID record.

**Collection dates are not clean.** Three kinds of defect are handled explicitly rather
than silently:

* mixed formats -- ISO (``2024-12-12``), ``12-Dec-2024``, ``2024-12`` and bare ``2024``;
* partial dates, counted but excluded from first/last, since a bare year would otherwise
  drag a lineage's first date back to 1 January;
* eight records whose date pre-dates their own lineage by a decade or more -- two in
  G.1.1, four in J.2, two in J. Every one is an error in the source metadata, not in the
  parsing here: the strain name carries the true year in all eight cases. Four land on
  1905, where the collection *year* has been read as an Excel 1900 day-serial: serial 2021
  is 1905-07-13 and serial 2025 is 1905-07-17, which is exactly what
  ``A/Lisboa/12/2021`` and ``A/Baltimore/JH-1150/2025`` carry. 277 records across the
  whole influenza database share that artefact. The remaining four are plain wrong years,
  three in GISAID and one in GenBank (``PV150037``, dated 2012 but a J.2 sequence, with
  ``collection_day/mon/year`` and the strain name all consistently wrong upstream).

The outlier problem is why the table carries **both** an absolute first/last and a
1st-to-99th percentile span. The percentile span is the one to quote: it needs no
judgement call about which records to delete, and a single bad date cannot move it.
``n_outside_span`` says how many records fall outside it, so nothing is hidden.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_base import Writer, sha256

TABLE = "lineage_panel.tsv.gz"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

#: Sequences outside this window are counted as unusable rather than parsed. The floor is
#: 2016 because the pre-training corpus ends in 2015: no sequence this panel describes can
#: legitimately pre-date it, and the earliest real lineage in the set, C.1, starts
#: 2016-05-21, so nothing genuine is cut. The ceiling guards against a typo in the year.
FLOOR = datetime.date(2016, 1, 1)
CEILING = datetime.date(2027, 1, 1)

#: Accessions whose collection date is inside the window above but still contradicts the
#: lineage the sequence belongs to. The window catches a year read as an Excel serial
#: (those land in 1905) and a year typo far enough out to fall below 2016, but it cannot
#: catch a wrong year that happens to be plausible on its own.
#:
#: Added 23 Oct 2026, after the lineage reassignment. Found by the rule the panel is
#: judged on -- more than three years before the lineage's own 1st percentile -- applied
#: across all thirteen lineages, which returns exactly two records: ``EPI2115977``
#: (2011-03-30, G.1.1), already excluded by FLOOR, and the one below. J.2 is a 2023
#: clade and carries J.2's defining replacements, so a 2020 collection date cannot be
#: right; the sequence is kept and counted, only its date is set aside.
#: Re-dated 23 Oct 2026 (dev/lineage_reassign/panel_filter.py, redated.tsv): the collection
#: date is withheld and the record is counted as a year-only date. Its strain name, clade and
#: DB record are 2024 or later and agree with the lineage, so 2021-11-13 is a dating error.
#: Only the year is kept; the strain name is not changed.
YEAR_ONLY_REDATED = {
    "EPI3864458": 2024,
}

KNOWN_BAD_DATES = {
    "EPI4678029": "dated 2020-06-06 but a J.2 sequence; J.2 arose in 2023",
}


def parse_date(text: str) -> tuple[datetime.date | None, bool]:
    """Return (date, is_exact). A partial date resolves to its first day, is_exact False."""
    text = text.strip()
    if not text:
        return None, False
    try:
        return datetime.date.fromisoformat(text), True
    except ValueError:
        pass
    parts = text.split("-")
    if len(parts) == 3 and parts[1] in MONTHS:
        try:
            return datetime.date(int(parts[2]), MONTHS[parts[1]], int(parts[0])), True
        except ValueError:
            return None, False
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        try:
            return datetime.date(int(parts[0]), int(parts[1]), 1), False
        except ValueError:
            return None, False
    if len(parts) == 1 and parts[0].isdigit() and len(parts[0]) == 4:
        try:
            return datetime.date(int(parts[0]), 1, 1), False
        except ValueError:
            return None, False
    return None, False


def read_accessions(path: Path) -> dict[str, str]:
    with Path(path).open() as handle:
        rows = [row for row in csv.reader(handle, delimiter="\t") if len(row) >= 2]
    mapping = {row[0]: row[1] for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"{path}: an accession appears in more than one lineage")
    return mapping


def collect_dates(path: Path, lineage_of: dict[str, str]):
    """Exact dates per lineage, plus counts of the partial and the unusable."""
    exact: dict[str, list[datetime.date]] = collections.defaultdict(list)
    partial: collections.Counter = collections.Counter()
    unusable: collections.Counter = collections.Counter()
    with Path(path).open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            lineage = lineage_of.get(row[0])
            if lineage is None:
                continue
            date, is_exact = parse_date(row[2] if len(row) > 2 else "")
            if row[0] in KNOWN_BAD_DATES:
                unusable[lineage] += 1
            elif row[0] in YEAR_ONLY_REDATED:
                partial[lineage] += 1
            elif date is None or not (FLOOR <= date < CEILING):
                unusable[lineage] += 1
            elif is_exact:
                exact[lineage].append(date)
            else:
                partial[lineage] += 1
    return exact, partial, unusable


def quantile(values: list[datetime.date], fraction: float) -> datetime.date:
    """Nearest-rank quantile of a sorted date list. No interpolation: a date between two
    sampled dates is not a date anything was sampled on."""
    return values[min(len(values) - 1, int(fraction * len(values)))]


def build(accessions: Path, dates: Path, panel_metadata: Path, observed: Path) -> pd.DataFrame:
    lineage_of = read_accessions(accessions)
    exact, partial, unusable = collect_dates(dates, lineage_of)

    meta = pd.read_csv(panel_metadata, sep="\t")
    meta = meta[meta["model"] == meta["model"].iloc[0]]          # one row per lineage
    sizes = dict(zip(meta["lineage"], meta["n_sequences"]))
    differing = dict(zip(meta["lineage"],
                         meta["differing_sites_vs_reference_non_gap_non_stop"]))

    seen = pd.read_csv(observed, sep="\t")
    replacements = seen[seen["obs_present"] == 1].groupby("lineage").size().to_dict()

    assigned = collections.Counter(lineage_of.values())
    rows = []
    for lineage, n_sequences in sizes.items():
        if assigned.get(lineage) != n_sequences:
            raise ValueError(
                f"{lineage}: the accession list has {assigned.get(lineage)} sequences but "
                f"the pipeline metadata says {n_sequences}")
        values = sorted(exact.get(lineage, []))
        if not values:
            raise ValueError(f"{lineage}: no usable collection date")
        low, high = quantile(values, 0.01), quantile(values, 0.99)
        rows.append({
            "lineage": lineage,
            "n_sequences": int(n_sequences),
            "n_dated": len(values),
            "n_partial_date": int(partial.get(lineage, 0)),
            "n_implausible_date": int(unusable.get(lineage, 0)),
            "n_undated": int(n_sequences) - len(values) - int(partial.get(lineage, 0))
                         - int(unusable.get(lineage, 0)),
            "first_date": values[0].isoformat(),
            "last_date": values[-1].isoformat(),
            "span_start_p1": low.isoformat(),
            "span_end_p99": high.isoformat(),
            "median_date": values[len(values) // 2].isoformat(),
            "span_days": (high - low).days,
            "n_outside_span": sum(1 for v in values if v < low or v > high),
            "sites_differing": int(differing[lineage]),
            "replacements_observed": int(replacements[lineage]),
        })
    frame = pd.DataFrame(rows).sort_values("span_start_p1").reset_index(drop=True)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accessions", type=Path, required=True)
    parser.add_argument("--dates", type=Path, required=True)
    parser.add_argument("--panel-metadata", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    frame = build(args.accessions, args.dates, args.panel_metadata, args.observed)
    writer = Writer(args.out_dir)
    writer.add(
        TABLE, frame,
        source=(f"aggregated by tools/export_lineage_panel.py from the lineage sub-alignment "
                f"accessions, the influenza database collection dates, {args.panel_metadata.name} "
                f"and {args.observed.name}"),
        sha=sha256(args.panel_metadata),
        why="Lineage panel figure: size, sampling span and diversity of each lineage",
    )
    writer.write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
