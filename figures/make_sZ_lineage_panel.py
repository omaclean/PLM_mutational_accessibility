"""S15: the lineage panel -- what each lineage is made of.

    python figures/make_sZ_lineage_panel.py --guide guides/h3n2_13lineage.yaml \
        --results results/h3n2_13lineage --out output

One table, one row per lineage, ordered by when the lineage was sampled. It answers the
three questions a reader has about a lineage set and the manuscript currently answers for
none of them: how many sequences, over what period, and how much variation.

Columns:

  Sequences              sequences assigned to that lineage.
  First / Last           earliest and latest collection date, to the month.
  Middle 98%             the 1st to the 99th percentile of collection dates. Still the
                         column to quote: the eight records dated a decade before their
                         own lineage are now excluded by the exporter's 2016 floor, but a
                         percentile span cannot be moved by one bad record that survives.
  Sites differing        sites of the aligned coding region at which any sequence in the
                         lineage differs from the lineage root.
  Replacements           distinct amino-acid replacements observed, counting a site with
                         two alternatives twice. This is the thing the models are scored
                         on, so it is the lineage's weight in every sweep.
  Homoplasies            of those replacements, how many arose two or more times
                         independently on the UShER tree.
  Multiple origins       the same thing as a percentage of the replacements the tree can
                         place. Read it alongside Sequences: it rises with how densely a
                         lineage was sampled, because every extra sequence is another
                         chance to catch a replacement recurring.
  Top 5% threshold       the within-lineage sequence frequency threshold that marks the top
                         5% most frequent observed replacements (the positive class for
                         AUROC in Figure 3C and Supplementary Figure B).

Nothing here is modelled or fitted; it is a description of the input data. The two origin
columns are counted by ``tools/export_homoplasy.py``; ``mutacc/homoplasy.py`` defines what
an origin is and what the tree cannot see.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mutacc import style
from mutacc.guide import load_guide
from mutacc.io import write_caption, write_values

STEM = "sZ_lineage_panel"
TABLE = "lineage_panel.tsv.gz"

#: (column in the frame, header, how to format one cell, alignment)
COLUMNS = (
    ("lineage", "Lineage", lambda v: str(v), "left"),
    ("n_sequences", "Sequences", lambda v: f"{int(v):,}", "right"),
    ("first_month", "First", lambda v: str(v), "right"),
    ("last_month", "Last", lambda v: str(v), "right"),
    ("span_months", "Middle 98%", lambda v: str(v), "right"),
    ("sites_differing", "Sites\ndiffering", lambda v: f"{int(v):,}", "right"),
    ("replacements_observed", "Replacements\nobserved", lambda v: f"{int(v):,}", "right"),
    ("n_homoplasy", "Homoplasies", lambda v: f"{int(v):,}", "right"),
    ("pct_multi_origin", "Multiple\norigins", lambda v: f"{v:.0f}%", "right"),
    ("top5_freq_threshold", "Top 5% freq\nthreshold", lambda v: f"{v * 100:.2f}%", "right"),
)

HOMOPLASY = "homoplasy_by_lineage.tsv.gz"

ROW_HEIGHT = 0.185      # inches per data row
HEADER_HEIGHT = 0.42    # inches for the two-line header plus its rule
FONT = 7.0
GAP_IN = 0.14           # inches of air between columns


def load(results: Path) -> pd.DataFrame:
    frame = pd.read_csv(results / TABLE, sep="\t")
    frame["pct_dated"] = 100.0 * frame["n_dated"] / frame["n_sequences"]
    for column, target in (("first_date", "first_month"), ("last_date", "last_month")):
        frame[target] = frame[column].str.slice(0, 7)
    frame["span_months"] = (frame["span_start_p1"].str.slice(0, 7) + "  –  "
                            + frame["span_end_p99"].str.slice(0, 7))

    freq_path = results / "replacement_frequency.tsv.gz"
    if freq_path.exists():
        freq_df = pd.read_csv(freq_path, sep="\t")
        obs = freq_df[freq_df["obs_freq"] > 0]
        thresholds = {}
        for lineage, grp in obs.groupby("lineage"):
            thresholds[lineage] = float(np.quantile(grp["obs_freq"].values, 0.95))
        frame["top5_freq_threshold"] = frame["lineage"].map(thresholds)
    elif "top5_freq_threshold" not in frame.columns:
        frame["top5_freq_threshold"] = np.nan

    frame = frame.merge(load_homoplasy(results, frame), on="lineage", validate="one_to_one")
    return frame.sort_values("span_start_p1").reset_index(drop=True)


def load_homoplasy(results: Path, panel: pd.DataFrame) -> pd.DataFrame:
    """The origin counts, checked against the panel's own replacement count.

    Both tables count the same thing from different directions -- the panel from the
    pipeline's observed table, the origin table from the shipped observed table beside the
    tree walk -- so a disagreement means one of them was rebuilt and the other was not.
    That is the failure that put a phantom mutation in Figure 1B/C, so it raises here.
    """
    frame = pd.read_csv(results / HOMOPLASY, sep="\t")
    merged = frame.merge(panel[["lineage", "replacements_observed"]], on="lineage",
                         validate="one_to_one")
    if len(merged) != len(panel):
        raise ValueError(f"{HOMOPLASY} describes {len(merged)} of the panel's "
                         f"{len(panel)} lineages")
    disagree = merged[merged["n_observed"] != merged["replacements_observed"]]
    if not disagree.empty:
        raise ValueError(
            "the origin table and the panel disagree about how many replacements were "
            "observed: " + ", ".join(
                f"{row.lineage} {row.n_observed} vs {row.replacements_observed}"
                for row in disagree.itertuples()))
    return frame[["lineage", "n_covered_positions", "n_observed_covered", "n_placed",
                  "n_unplaced", "n_homoplasy", "n_homoplasy_multi_route",
                  "pct_multi_origin", "max_origins"]]


#: Summary-row keys whose value is a mean of the thirteen rows rather than a figure pooled
#: over the sequences. Only these two are starred: pooling weights a lineage by its size,
#: so the pooled percentage is the larger and the difference is real. The span is not
#: starred -- earliest percentile start to latest percentile end is a combined range, and
#: the dates it is built from are the same ones the rows show.
ACROSS_ROWS = frozenset({"pct_multi_origin", "top5_freq_threshold"})



def summary_row(frame: pd.DataFrame, results: Path) -> pd.Series:
    """One combined row for the foot of the table.

    Three kinds of number meet here and they must not be confused, which is what the
    asterisk is for.

    *Combined totals* are the quantity for the thirteen lineages taken together.
    ``n_sequences`` is their sum because the sets are disjoint. ``first``/``last`` are the
    extremes. ``n_homoplasy`` is a sum, and its unit is the lineage-replacement pair: one
    replacement recurring in three lineages counts three times, because it is three
    separate recurrences.

    *Unions* are where a sum would double-count. The same (position, amino acid) is
    reached in several lineages, so ``sites_differing`` and ``replacements_observed`` are
    counted distinctly across the panel -- 484 sites and 1,715 replacements, against
    sums of 3,628 and 6,546. The per-lineage ``sites_differing`` equals that lineage's
    distinct replaced positions, which is what makes the union well defined.

    *Means over rows* are the two percentages, and they carry the asterisk. The mean of
    the thirteen multiple-origin percentages is not the pooled percentage: pooling weights
    a lineage by its size, and the deep lineages are the recurrent ones, so the pooled
    figure is the higher of the two (49.6% against 42.4%). The span is a combined range --
    earliest percentile start to latest percentile end -- so it is not starred.
    """
    seen = pd.read_csv(results / "replacement_observed.tsv.gz", sep="	")
    seen = seen[seen["obs_present"] == 1]
    panel_total = int(frame["replacements_observed"].sum())
    if len(seen) != panel_total:
        raise ValueError(f"the observed table holds {len(seen)} present replacements but "
                         f"the panel's columns sum to {panel_total}")

    row = {
        "lineage": f"All {len(frame)}",
        "n_sequences": int(frame["n_sequences"].sum()),
        "first_month": min(frame["first_month"]),
        "last_month": max(frame["last_month"]),
        "span_months": (min(frame["span_start_p1"]).replace("-", "-")[:7] + "  –  "
                        + max(frame["span_end_p99"])[:7]),
        "sites_differing": seen["position"].nunique(),
        "replacements_observed": len(set(zip(seen["position"], seen["aa"]))),
        "n_homoplasy": int(frame["n_homoplasy"].sum()),
        "pct_multi_origin": float(frame["pct_multi_origin"].mean()),
        "top5_freq_threshold": float(frame["top5_freq_threshold"].mean()),
    }
    missing = {key for key, _, _, _ in COLUMNS} - set(row)
    if missing:
        raise ValueError(f"the summary row has no value for {sorted(missing)}")
    return pd.Series(row)


def check_against_guide(frame: pd.DataFrame, guide) -> None:
    """The table must describe exactly the lineages the guide declares."""
    declared = {lineage.id for lineage in guide.lineages}
    present = set(frame["lineage"])
    if declared != present:
        raise ValueError(
            f"the table and the guide disagree: only in the guide {sorted(declared - present)}, "
            f"only in the table {sorted(present - declared)}")


def text_inches(text: str, bold: bool = False) -> float:
    """Printed width of ``text`` at the figure's font size, from its glyph outlines."""
    prop = FontProperties(weight="bold" if bold else "normal", size=FONT)
    return TextPath((0, 0), text, prop=prop).get_extents().width / 72.0


def column_positions(frame: pd.DataFrame, starred: frozenset[str] = frozenset()) -> list[float]:
    """Right edges in axes fractions, from each column's widest printed cell or header.

    Widths are measured from the glyphs, not counted in characters: a proportional face
    prints wider than a character count suggests (bold headers, mixed digits). ``starred``
    names columns whose summary cell carries a trailing asterisk, which is included in the
    measurement so the table does not clip it.
    """
    widths = []
    for key, header, fmt, _ in COLUMNS:
        cells = [text_inches(fmt(value) + ("*" if key in starred else "")) for value in frame[key]]
        heads = [text_inches(line, bold=True) for line in header.split("\n")]
        widths.append(max(max(cells), max(heads)))
    total = sum(widths) + GAP_IN * (len(widths) - 1)
    edges, running = [], 0.0
    for width in widths:
        running += width
        edges.append(running / total)
        running += GAP_IN
    return edges


def draw(frame: pd.DataFrame, n_total: int, summary: pd.Series | None = None):
    rows = len(frame) + (1.6 if summary is not None else 0)
    height = HEADER_HEIGHT + ROW_HEIGHT * rows + 0.30
    # The house default is a constrained layout, which has nothing to manage here: the
    # single axes is the whole canvas and every cell is placed by hand.
    fig = style.figure(width=style.WIDTH_DOUBLE, height=height, constrained_layout=False)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    measured = frame if summary is None else pd.concat(
        [frame, summary.to_frame().T], ignore_index=True)
    edges = column_positions(measured, ACROSS_ROWS if summary is not None else frozenset())
    # The title block is drawn above the axes, so the header starts at the very top of
    # them. Reserving canvas for it here as well would leave a band of dead space.
    top = 1.0 - (0.11 / height)
    header_y = top
    body_top = top - HEADER_HEIGHT / height

    for (key, header, _fmt, align), right in zip(COLUMNS, edges):
        left = right - 0.0
        x = 0.0 if align == "left" else right
        ha = "left" if align == "left" else "right"
        ax.text(x, header_y, header, ha=ha, va="top", fontsize=FONT,
                color=style.INK, fontweight="bold", linespacing=1.25)

    rule_y = body_top + 0.012
    ax.plot([0, 1], [rule_y, rule_y], color=style.INK, linewidth=0.8,
            solid_capstyle="butt")

    for index, row in frame.iterrows():
        y = body_y = body_top - (index + 0.72) * ROW_HEIGHT / height
        if index % 2 == 1:                            # a very light band, for tracking across
            ax.axhspan(y - 0.30 * ROW_HEIGHT / height, y + 0.70 * ROW_HEIGHT / height,
                       color=style.GRID, alpha=0.35, linewidth=0)
        for (key, _header, fmt, align), right in zip(COLUMNS, edges):
            x = 0.0 if align == "left" else right
            ha = "left" if align == "left" else "right"
            ax.text(x, y, fmt(row[key]), ha=ha, va="baseline", fontsize=FONT,
                    color=style.INK)

    if summary is not None:
        # A rule separates the combined row from the lineages, because it is a different
        # kind of number and must not be read as a fourteenth lineage.
        sep_y = body_top - (len(frame) + 0.30) * ROW_HEIGHT / height
        ax.plot([0, 1], [sep_y, sep_y], color=style.INK, linewidth=0.6,
                solid_capstyle="butt")
        y = body_top - (len(frame) + 1.02) * ROW_HEIGHT / height
        for (key, _header, fmt, align), right in zip(COLUMNS, edges):
            x = 0.0 if align == "left" else right
            ha = "left" if align == "left" else "right"
            text = fmt(summary[key]) + ("*" if key in ACROSS_ROWS else "")
            ax.text(x, y, text, ha=ha, va="baseline", fontsize=FONT,
                    color=style.INK, fontweight="bold")

    foot_y = body_top - (rows + 0.35) * ROW_HEIGHT / height
    ax.plot([0, 1], [foot_y, foot_y], color=style.GRID, linewidth=0.8,
            solid_capstyle="butt")
    note = f"{n_total:,} sequences in all"
    if summary is not None:
        # Kept to one short line: this text sets the figure's width, because the canvas is
        # trimmed to its contents on save. The reasoning belongs in the caption.
        note += (f"   *  mean of the {len(frame)} lineages, "
                 f"not pooled over the sequences")
    ax.text(0.0, foot_y - 0.055, note, ha="left", va="top",
            fontsize=FONT - 0.5, color=style.INK_MUTED)

    style.panel_title(
        ax,
        "The lineages the models are scored on",
        "size, sampling period and amount of variation in each",
    )
    return fig


def caption(frame: pd.DataFrame, guide, row: pd.Series, pooled: float) -> str:
    total = int(frame["n_sequences"].sum())
    smallest = frame.loc[frame["n_sequences"].idxmin()]
    largest = frame.loc[frame["n_sequences"].idxmax()]
    undated = int((frame["n_partial_date"] + frame["n_implausible_date"]
                   + frame["n_undated"]).sum())
    min_cut = frame["top5_freq_threshold"].min() * 100
    max_cut = frame["top5_freq_threshold"].max() * 100

    n_placed = int(frame["n_placed"].sum())
    n_homoplasy = int(frame["n_homoplasy"].sum())
    multi_route = int(frame["n_homoplasy_multi_route"].sum())
    one_route = n_homoplasy - multi_route
    pct_overall = 100.0 * n_homoplasy / n_placed
    pct_lo = frame["pct_multi_origin"].min()
    pct_hi = frame["pct_multi_origin"].max()
    n_unplaced = int(frame["n_unplaced"].sum())
    # Spearman without scipy: the house environment does not guarantee it, and a rank
    # correlation over thirteen rows is three lines.
    ranks = (frame["n_sequences"].rank(), frame["pct_multi_origin"].rank())
    rho = float(np.corrcoef(*ranks)[0, 1])
    covered = set(frame["n_covered_positions"])
    if len(covered) != 1:
        raise ValueError(f"the lineages disagree about the covered positions: {sorted(covered)}")
    n_covered = covered.pop()
    return (
        f"**The {len(frame)} {guide.virus} lineages, their sampling periods and "
        f"their variation.** Each lineage is a set of sequences assigned to one "
        f"{guide.reference}-referenced clade root; the sets are disjoint and hold "
        f"{total:,} sequences in all. *Sequences* is the number assigned. "
        f"{undated:,} of them carry no collection date, only a year or a month, or a date "
        f"that pre-dates their own lineage; these are retained in every replacement, origin "
        f"and frequency count and are left out only of the collection date columns. "
        f"*First* and *Last* are the extreme collection dates and *Middle 98%* is the 1st to "
        f"the 99th percentile of them. **The percentile span is the one to read.** Eight "
        f"records carried a collection date a decade or more before the lineage they belong "
        f"to — four of them in 1905, which is what a year becomes when it is read as a "
        f"spreadsheet serial number — and are excluded as implausible, since the corpus the "
        f"model was pre-trained on ends in 2015 and nothing here can pre-date 2016. The "
        f"percentile span is quoted all the same, because it cannot be moved by one bad "
        f"record that slips through. *Sites differing* counts the sites of the "
        f"aligned coding region at which any sequence in the lineage differs from its root. "
        f"*Replacements observed* counts the distinct amino-acid replacements seen, which is "
        f"the unit every model is scored on, so it is also each lineage's weight in the "
        f"sweeps. *Homoplasies* and *Multiple origins* ask how many of those replacements "
        f"arose more than once. An *origin* is a branch of the segment 4 UShER tree whose "
        f"parent codon encodes a different residue and whose child codon encodes the "
        f"replacement's; synonymous changes, and codons holding a stop or an ambiguous base, "
        f"are not origins. Two origins therefore always sit on two different branches, "
        f"neither ancestral to the other with the residue unbroken. A replacement with two or "
        f"more origins is a *homoplasy*: {n_homoplasy:,} of the {n_placed:,} replacements the "
        f"tree places, {pct_overall:.0f}% overall and {pct_lo:.0f}% to {pct_hi:.0f}% across "
        f"lineages. Two processes produce them and both are counted. In "
        f"{100 * one_route / n_homoplasy:.0f}% of cases the origins are the same nucleotide "
        f"change in different parts of the tree, so their independence rests on the topology; "
        f"in the remaining {100 * multi_route / n_homoplasy:.0f}% the same residue is reached "
        f"by two or more different codon changes, which cannot be one event whatever the "
        f"topology. **Multiple origins rises with sampling depth** (Spearman "
        f"{rho:.2f} against *Sequences* across the {len(frame)} lineages): every extra "
        f"sequence is another chance to catch a recurrence, so the column compares sampling "
        f"effort as much as it compares biology, and the counts are a floor on recurrence, "
        f"never an estimate of its rate. Two gaps shrink the denominator and are reported "
        f"rather than absorbed: the UShER reference is not an H3, so {566 - n_covered} of the "
        f"566 positions (1-26, the signal peptide and the first ten residues of mature HA1, "
        f"and 562-566) have no reference codon and no replacement there can be placed; and "
        f"{n_unplaced:,} further replacements are observed in a sequence the tree does not "
        f"hold. Neither is counted as a single origin. *Top 5% freq threshold* is the "
        f"sequence frequency defining the 5% most "
        f"prevalent circulating replacements in each lineage (the positive class scored in "
        f"Figure 3C and Supplementary Figure B), ranging from {min_cut:.2f}% to {max_cut:.2f}%. "
        f"Lineage size spans a {largest['n_sequences'] / smallest['n_sequences']:.0f}-fold "
        f"range, from {int(smallest['n_sequences']):,} ({smallest['lineage']}) to "
        f"{int(largest['n_sequences']):,} ({largest['lineage']}); means taken over lineages "
        f"therefore weight these sets equally despite that. **The combined row holds three "
        f"different kinds of number and the asterisk separates them.** *Sequences* is a sum, "
        f"which is right because the {len(frame)} sets are disjoint, and *First* and *Last* are "
        f"the extremes. *Sites differing* and *Replacements observed* are counted once over "
        f"the panel rather than summed, because the same replacement is reached in several "
        f"lineages: {int(row['sites_differing']):,} distinct sites and "
        f"{int(row['replacements_observed']):,} distinct replacements, against sums of "
        f"{int(frame['sites_differing'].sum()):,} and "
        f"{int(frame['replacements_observed'].sum()):,}. *Homoplasies* is a sum, and its unit "
        f"is the lineage-replacement pair, since one replacement recurring in several "
        f"lineages is that many separate recurrences. *Middle 98%* runs from the earliest "
        f"percentile start to the latest percentile end, a combined range, so it is not "
        f"starred. The two starred cells are means of the {len(frame)} rows rather than "
        f"figures pooled over the sequences: *Multiple origins* is the mean of the "
        f"per-lineage percentages, {row['pct_multi_origin']:.0f}%, where pooling them gives "
        f"{pooled:.0f}% — pooling weights a lineage by its size and the deep lineages are the "
        f"recurrent ones, so the pooled figure is always the larger and the two must not be "
        f"quoted interchangeably. Dates are collection dates held "
        f"in the sequence database. No per-sequence data is shown or shipped."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args(argv)

    matplotlib.use("Agg")
    style.use_house_style()

    guide = load_guide(args.guide)
    frame = load(args.results)
    check_against_guide(frame, guide)

    summary = summary_row(frame, args.results)
    fig = draw(frame, int(frame["n_sequences"].sum()), summary)
    written = style.save_figure(fig, args.out / STEM)
    write_values(pd.concat([frame, summary.to_frame().T], ignore_index=True),
                 args.out, STEM)
    pooled = 100.0 * frame["n_homoplasy"].sum() / frame["n_placed"].sum()
    write_caption(caption(frame, guide, summary, pooled), args.out, STEM)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
