"""Figure 1, panels B and C: every replacement ranked, with the nine tracked ones marked.

    python figures/make_fig01bc.py

Panel B ranks the 10,754 possible replacements on the J.2 backbone by the fine-tuned model's
probability; panel C ranks the same replacements by their mutational accessibility. Both are
drawn from ``output/fig01bc_ranked_mutations_values.tsv``, which
``tools/export_fig01bc_values.py`` writes from the shipped replacement table.

**Why this script exists** (23 Oct 2026). B and C were previously placed as one frozen
render, cropped in two by ``guides/composites/fig01.yaml``. That render was drawn with an
unstable sort, and accessibility is heavily tied -- about 860 distinct values across 10,754
replacements, with blocks running thirty deep -- so every one of panel C's nine baked-in
ranks was a draw from inside a tie block rather than the block's own rank. All nine differed
from the exporter's competition ranks, by between 2 and 19 places:

    Q173R 250->241  T328A 278->276  N158D 526->513  K189R 627->618  S144N 933->928
    HA2:S49N 946->928  K2N 1349->1331  T135K 2583->2564  I160K 3300->3294

The two that reach the manuscript are Q173R and I160K. S144N and HA2:S49N are the clearest
symptom: their accessibilities are equal to five decimal places, so they share rank 928, and
the render had split them 13 places apart.

**Ties take the first rank of their block** (competition ranking: 1, 2, 2, 4). That is what
the values table reports in ``rank``, and ``rank_tie_first``/``rank_tie_last`` give the block
either side of it, so a label reading ``(N=928)`` on two points is the data and not a
collision. The caption carries the widest block so the reader can size the ambiguity.

Drawn in the shared panel box (``style.PANEL_AXES``) at the shared size, and saved uncropped,
so the two line up with each other and with panel D. These are panels, not figures: the
composite places them and draws their headers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

from mutacc import style
from mutacc.guide import PANELS_DIR, REPO_ROOT
# captions belong to the exporter; this script writes panels only

#: The two panel renders. Declared as ``STEM``-named constants, and not only as values
#: inside ``PANELS``, because tests/test_numbering.py reads the figure set out of the
#: source by scanning for them: a stem reachable only through a dict would leave its
#: numbering entry looking stale.
STEM_B = "fig01b_ranked_model"
STEM_C = "fig01c_ranked_accessibility"

#: The ranking this script draws, and an *input* to it: tools/export_fig01bc_values.py
#: writes that table and the caption that goes with it, so the pair keeps one producer
#: (tests/test_output_provenance.py). This script writes panels only.
VALUES = REPO_ROOT / "output" / "fig01bc_ranked_mutations_values.tsv"

#: panel key -> (output stem, axis label, caption noun). The y label says what the reader is
#: looking at rather than naming a column, which ``style.finalise_axes`` requires.
#: The axis titles carry no "(log scale)": at 11.5 pt in a 3.05 in panel that ran past the
#: canvas and lost its closing bracket. The decade ticks say it, and the caption says it.
PANELS = {
    "B": (STEM_B, "Model probability", "the model's probability"),
    "C": (STEM_C, "Mutational accessibility", "mutational accessibility"),
}
XLABEL = "Rank (1 = highest)"

CURVE_COLOUR = style.NULL_GREY
MARK_COLOUR = style.ACCESS
CURVE_WIDTH = 1.0
MARKER_SIZE = 4.0
LABEL_PT = 6.2
LEADER_WIDTH = 0.5
#: Where the label column sits, and the band it spans, both in axes fractions. The nine
#: marked replacements all sit in the top-left corner, so the labels go in a single column
#: to their right and fan out downwards.
LABEL_X = 0.38
LABEL_TOP = 0.955
LABEL_BOTTOM = 0.27


def load_values(path: Path = VALUES) -> pd.DataFrame:
    """The exported ranking, checked for the columns and invariants the panels rely on."""
    needed = {"panel", "mutation", "log10_value", "rank", "rank_tie_first", "rank_tie_last",
              "n_ranked", "observed"}
    frame = pd.read_csv(path, sep="\t")
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for key in PANELS:
        rows = frame[frame["panel"] == key]
        if rows.empty:
            raise ValueError(f"{path}: no rows for panel {key}")
        n_ranked = int(rows["n_ranked"].iloc[0])
        if len(rows) != n_ranked:
            raise ValueError(f"{path}: panel {key} has {len(rows)} rows, n_ranked says {n_ranked}")
        if int(rows["observed"].sum()) != 9:
            raise ValueError(f"{path}: panel {key} marks {int(rows['observed'].sum())} "
                             "replacements, expected the nine J.2 -> K changes")
        if not (rows["rank"] == rows["rank_tie_first"]).all():
            raise ValueError(f"{path}: panel {key} rank is not the first of its tie block, so "
                             "it is not a competition rank")
    return frame


def draw(frame: pd.DataFrame, key: str):
    """One panel: the ranked curve in grey, the nine marked, each labelled with its rank."""
    _stem, ylabel, _noun = PANELS[key]
    rows = frame[frame["panel"] == key].sort_values("rank")
    marked = rows[rows["observed"]].sort_values("rank")

    fig = style.figure(*style.PANEL_SIZE)
    ax = style.panel_axes(fig)
    ax.plot(rows["rank"], rows["log10_value"], color=CURVE_COLOUR, linewidth=CURVE_WIDTH,
            zorder=2, solid_capstyle="butt")
    ax.scatter(marked["rank"], marked["log10_value"], s=MARKER_SIZE ** 2, color=MARK_COLOUR,
               zorder=4, linewidths=0)

    # The label column, top to bottom in the order the points are ranked, so a leader line
    # never has to cross another.
    n = len(marked)
    step = (LABEL_TOP - LABEL_BOTTOM) / max(n - 1, 1)
    for i, (_idx, row) in enumerate(marked.iterrows()):
        ax.annotate(
            f"{row['mutation']} (N={int(row['rank'])})",
            xy=(row["rank"], row["log10_value"]), xycoords="data",
            xytext=(LABEL_X, LABEL_TOP - i * step), textcoords="axes fraction",
            fontsize=LABEL_PT, color=MARK_COLOUR, va="center", ha="left", zorder=5,
            # The labels sit over the ranked curve, so each gets a tight white backing to
            # stay readable without hiding more of the curve than its own glyphs do.
            bbox=dict(boxstyle="square,pad=0.12", facecolor=style.SURFACE, edgecolor="none",
                      alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=MARK_COLOUR, linewidth=LEADER_WIDTH,
                            shrinkA=0.0, shrinkB=1.5, alpha=0.75),
        )

    ax.set_xlim(-int(rows["n_ranked"].iloc[0]) * 0.02, int(rows["n_ranked"].iloc[0]) * 1.02)
    ax.grid(True, linestyle="-", alpha=0.6)
    ax.tick_params(labelsize=style.PANEL_TICK_PT)
    style.finalise_axes(ax, xlabel=XLABEL, ylabel=ylabel)
    style.place_panel_labels(ax)
    return fig



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--values", type=Path, default=VALUES)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--panels-out", type=Path, default=PANELS_DIR,
                        help=f"where the panel renders go (default: "
                             f"{PANELS_DIR.relative_to(REPO_ROOT).as_posix()}/)")
    args = parser.parse_args(argv)

    style.use_house_style()
    frame = load_values(args.values)
    written = []
    for key, (stem, _ylabel, _noun) in PANELS.items():
        written.extend(style.save_panel(draw(frame, key), args.panels_out / stem,
                                        suffixes=(".pdf", ".png")))
    print("wrote", ", ".join(p.name for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
