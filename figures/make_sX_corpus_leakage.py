"""Table SX: the lineage-defining replacements against the fine-tuning corpus.

    python figures/make_sX_corpus_leakage.py --results results/h3n2_13lineage --out output

One row per lineage-defining replacement, answering one question: before the model was
ever asked to predict the replacement, how many sequences in the corpus it was fine-tuned
on already carried the amino acid that replacement produces?

Columns:

  Change                 the amino acid the ancestral lineage carries at that site, and
                         the one the descendant lineage carries.
  Sequences with it      corpus sequences carrying the *new* amino acid at that site.
  Share of training data that count over the whole corpus, every row on the same
                         denominator. A sequence truncated past the site, or gapped at
                         it, cannot carry the residue and is still counted below the
                         line, so at a site not every sequence reaches the share is a
                         floor; the footnote gives the one site where that bites.
  In <subtype> /         the same count and share within the subtype the lineage belongs
  Share of <subtype>     to, whatever the host: the question of whether the model read
                         the residue in a comparable sequence background at all.
  In the human subset /  the same again within the human sequences of that subtype,
  Share of human subset  nested inside the subtype columns. This is the narrowest group
                         and the only one where a share means "what was circulating".
  First seen             collection year of the earliest corpus sequence carrying it.

Both subsets are named by the table rather than by this script, because nothing under
figures/ may name a virus (tests/test_figure_rules.py).

The co-occurrence table the exporter also writes -- whether one corpus sequence carries
two of these replacements at once -- is deliberately **not** drawn here. It is a
different question from this one and crowded the table; it is still exported, still
tested, and still available to quote.

Nothing here is modelled or fitted; it is a description of the training data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mutacc import style
from mutacc.io import write_caption, write_values

STEM = "sX_corpus_leakage"
REPLACEMENT_TABLE = "k_corpus_leakage.tsv.gz"


def _count(value) -> str:
    return f"{int(value):,}"


def _pct(value) -> str:
    if pd.isna(value):
        return "–"
    # A row is in this table because the thing happened, so "0.0%" would read as a
    # flat contradiction of its own count.
    if 0.0 < value < 0.1:
        return "<0.1%"
    return f"{value:.1f}%"


def _year(value) -> str:
    text = str(value)
    return "–" if pd.isna(value) or text in ("", "nan") else text[:4]


def columns(frame: pd.DataFrame):
    """(column in the frame, header, how to format one cell, alignment), per column.

    Built from the frame rather than fixed, because the middle pair is headed by the
    subtype the table itself names: nothing under figures/ may name a virus
    (tests/test_figure_rules.py), so the word arrives as data.
    """
    subtype = frame["h3_label"].iloc[0]
    return (
        ("replacement", "Replacement", str, "left"),
        ("change", "Change", str, "right"),
        ("corpus_derived", "Sequences\nwith it", _count, "right"),
        ("corpus_pct", "Share of\ntraining data", _pct, "right"),
        ("h3_derived", f"In {subtype}", _count, "right"),
        ("h3_pct", f"Share of\n{subtype}", _pct, "right"),
        ("subset_derived", "In the human\nsubset", _count, "right"),
        ("subset_pct", "Share of\nhuman subset", _pct, "right"),
        ("corpus_first_year", "First\nseen", _year, "right"),
    )

ROW_HEIGHT = 0.185      # inches per data row
HEADER_HEIGHT = 0.42    # inches for the two-line header plus its rule
FONT = 7.0


def load(results: Path) -> pd.DataFrame:
    frame = pd.read_csv(results / REPLACEMENT_TABLE, sep="\t")
    frame["change"] = frame["ancestral_aa"] + " → " + frame["derived_aa"]
    return frame


def check(frame: pd.DataFrame) -> None:
    """The table must describe the full replacement set, each row backed by sequences."""
    if frame.empty:
        raise ValueError("the replacement table is empty")
    if frame["replacement"].duplicated().any():
        raise ValueError("a replacement appears twice")
    if (frame["corpus_typed"] <= 0).any():
        raise ValueError("a row has no readable corpus sequences behind it")
    if (frame["corpus_derived"] > frame["corpus_typed"]).any():
        raise ValueError("a count exceeds the sequences readable at its site")
    if frame["subset"].nunique() != 1 or frame["h3_label"].nunique() != 1:
        raise ValueError("the rows disagree about which subset they report")
    # The three groups nest, so their counts must: a sequence in the narrowest is in all
    # three. A column swap or a mis-specified mask shows up here and nowhere else.
    if (frame["subset_derived"] > frame["h3_derived"]).any():
        raise ValueError("the human subset carries more than the subtype it sits in")
    if (frame["h3_derived"] > frame["corpus_derived"]).any():
        raise ValueError("the subtype carries more than the training data it sits in")


def column_positions(frame: pd.DataFrame) -> list[float]:
    """Right edges in axes fractions, from each column's widest cell."""
    widths = []
    for key, header, fmt, _align in columns(frame):
        cells = [len(fmt(value)) for value in frame[key]]
        widths.append(max(max(cells), max(len(line) for line in header.split("\n"))))
    gap = 2.2                                        # characters of air between columns
    total = sum(widths) + gap * (len(widths) - 1)
    edges, running = [], 0.0
    for width in widths:
        running += width
        edges.append(running / total)
        running += gap
    return edges


def draw(frame: pd.DataFrame):
    rows = len(frame)
    height = HEADER_HEIGHT + ROW_HEIGHT * rows + 0.40
    # The house default is a constrained layout, which has nothing to manage here: the
    # single axes is the whole canvas and every cell is placed by hand.
    fig = style.figure(width=style.WIDTH_DOUBLE, height=height, constrained_layout=False)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    spec = columns(frame)
    edges = column_positions(frame)
    # The title block is drawn above the axes, so the header starts at the very top of
    # them. Reserving canvas for it here as well would leave a band of dead space.
    top = 1.0 - (0.11 / height)
    header_y = top
    body_top = top - HEADER_HEIGHT / height

    for (_key, header, _fmt, align), right in zip(spec, edges):
        x = 0.0 if align == "left" else right
        ha = "left" if align == "left" else "right"
        ax.text(x, header_y, header, ha=ha, va="top", fontsize=FONT,
                color=style.INK, fontweight="bold", linespacing=1.25)

    rule_y = body_top + 0.012
    ax.plot([0, 1], [rule_y, rule_y], color=style.INK, linewidth=0.8,
            solid_capstyle="butt")

    for index, row in frame.iterrows():
        y = body_top - (index + 0.72) * ROW_HEIGHT / height
        if index % 2 == 1:                            # a very light band, for tracking across
            ax.axhspan(y - 0.30 * ROW_HEIGHT / height, y + 0.70 * ROW_HEIGHT / height,
                       color=style.GRID, alpha=0.35, linewidth=0)
        for (key, _header, fmt, align), right in zip(spec, edges):
            x = 0.0 if align == "left" else right
            ha = "left" if align == "left" else "right"
            ax.text(x, y, fmt(row[key]), ha=ha, va="baseline", fontsize=FONT,
                    color=style.INK)

    foot_y = body_top - (rows + 0.35) * ROW_HEIGHT / height
    ax.plot([0, 1], [foot_y, foot_y], color=style.GRID, linewidth=0.8,
            solid_capstyle="butt")
    shallow = frame.loc[frame["corpus_typed"].idxmin()]
    ax.text(0.0, foot_y - 0.040,
            f"{int(frame['corpus_sequences'].iloc[0]):,} training sequences collected up "
            f"to {frame['corpus_last_date'].iloc[0]}; {int(frame['h3_sequences'].iloc[0]):,} "
            f"of them {frame['h3_label'].iloc[0]}, of which "
            f"{int(frame['subset_sequences'].iloc[0]):,} are {frame['subset'].iloc[0]}, "
            f"the human subset.\nEvery share is over its whole group. "
            f"{shallow['replacement']} is the one site not every sequence reaches "
            f"({int(shallow['corpus_typed']):,} readable), so its share is a floor: "
            f"{int(shallow['corpus_derived']):,} / {int(shallow['corpus_typed']):,} = "
            f"{shallow['corpus_derived'] / shallow['corpus_typed']:.1%} of those readable.",
            ha="left", va="top", fontsize=FONT - 1.0, color=style.INK_MUTED, linespacing=1.25)

    style.panel_title(
        ax,
        "The lineage-defining replacements in the fine-tuning corpus",
        "how much of the new lineage the model had already read",
    )
    return fig


def caption(frame: pd.DataFrame) -> str:
    subset = frame["subset"].iloc[0]
    subtype = frame["h3_label"].iloc[0]
    present = int((frame["corpus_derived"] > 0).sum())
    in_subtype = int((frame["h3_derived"] > 0).sum())
    in_subset = int((frame["subset_derived"] > 0).sum())
    majority = frame[frame["subset_pct"] > 50.0]
    named = ", ".join(majority["replacement"])
    shallow = frame.loc[frame["corpus_typed"].idxmin()]
    # The rows that separate the two subsets: present in the subtype at large, absent
    # from the human sequences within it. Naming them is the point of the split.
    split = frame[(frame["h3_derived"] > 0) & (frame["subset_derived"] == 0)]
    split_note = (
        " " + ", ".join(split["replacement"]) + " is present in the "
        f"{subtype} sequences but in none of the {subset} ones, so it was read by the "
        f"model in a {subtype} background without ever having circulated in the human "
        f"population the lineage emerged from."
        if len(split) == 1 else
        (" " + ", ".join(split["replacement"]) + f" are present in the {subtype} "
         f"sequences but in none of the {subset} ones." if len(split) else "")
    )
    return (
        f"**The lineage-defining replacements against the corpus the model was "
        f"fine-tuned on.** The corpus holds "
        f"{int(frame['corpus_sequences'].iloc[0]):,} sequences collected up to "
        f"{frame['corpus_last_date'].iloc[0]}, and the question is how much of the new "
        f"lineage it already contained. *Sequences with it* counts corpus sequences "
        f"carrying the **new** amino acid of that replacement; *Share of training data* "
        f"is that count over the whole corpus, so every row sits on the same denominator. "
        f"A sequence truncated past the site, or gapped at it, cannot carry the residue "
        f"but is still in the denominator, so at a site not every sequence reaches the "
        f"share is a floor: {shallow['replacement']}, near the signal peptide boundary, "
        f"is read in only {int(shallow['corpus_typed']):,} of the "
        f"{int(shallow['corpus_sequences']):,} sequences, giving "
        f"{int(shallow['corpus_derived']):,} / {int(shallow['corpus_sequences']):,} "
        f"= {shallow['corpus_pct']:.1f}% quoted here against "
        f"{int(shallow['corpus_derived']):,} / {int(shallow['corpus_typed']):,} = "
        f"{100.0 * shallow['corpus_derived'] / shallow['corpus_typed']:.1f}% among those "
        f"readable; every other site is read in at least "
        f"{int(frame['corpus_typed'].drop(shallow.name).min()):,}. "
        f"The next two columns repeat count and share within the {subtype} sequences, "
        f"whatever the host, and *In the human subset* repeats them again within the "
        f"{subset} sequences nested inside those, which is the narrowest group and the "
        f"only one whose share reads as a circulating frequency; the training data as a "
        f"whole spans every subtype. Residues are read at the ancestral root's own "
        f"precursor coordinates, each corpus sequence having been aligned pairwise to "
        f"the translated root, and every ancestral residue was confirmed on that "
        f"backbone before counting. **All {present} of the new amino acids are already "
        f"present in the training data**, {in_subtype} of them within the {subtype} "
        f"sequences and {in_subset} within the {subset} subset.{split_note} At "
        f"{named} the new residue is not merely present but is the **majority** state of "
        f"the {subset} subset, so at those sites the lineage returns to a historically "
        f"common residue rather than reaching a novel one. Read it beside the corpus "
        f"split control, which shows the fine-tuned advantage holding on replacements "
        f"absent from the corpus as well as on those present in it."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args(argv)

    matplotlib.use("Agg")
    style.use_house_style()

    frame = load(args.results)
    check(frame)

    fig = draw(frame)
    written = style.save_figure(fig, args.out / STEM)
    write_values(frame, args.out, STEM)
    write_caption(caption(frame), args.out, STEM)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
