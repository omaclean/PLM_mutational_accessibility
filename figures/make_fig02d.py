"""Figure 2, panel D: model vs mutational accessibility, one nucleotide-step group at a time.

    python figures/make_fig02d.py --guide guides/h3n2_5lineage.yaml

A simpler redraw of the old panel A, made to sit in the Figure 2 composite. Same statistic
(``mutacc.strata.strata_table``), now on a single lineage rather than the mean over the set:
the lineages are scored at the same sites and agree three to ninety-six times more closely than
the site bootstrap is wide, so averaging them added nothing but the appearance of replication.
Which lineage is the guide's ``primary_lineage``, not a rule applied here.

The intervals are percentile bootstraps that resample **sites** -- whole positions, carrying all
of that position's replacements with them -- because replacements at one site share a backbone
and are not independent. The panel says so, since "95% interval" alone does not.
Fewer marks, too: one dot and one interval per score and group,
no connecting lines, no end labels, a single legend. Models and the Grantham control only;
BLOSUM62 is left out because it is a comparator, not a control (an empirical substitution
matrix already carries mutational accessibility). It stays in the values table and in S10.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd

from mutacc import style
from mutacc.guide import parse_figure_args
from mutacc.io import write_values
from mutacc.strata import ALL, primary_lineage, strata_table

STEM = "fig02d_strata_panel"
CONTROL = ("neg_grantham", "Grantham similarity (control)")
GROUP_LABELS = {ALL: "all", "1": "1", "2": "2", "3": "3"}
OFFSET = {0: -0.24, 1: 0.0, 2: 0.24}
#: Panel 2C's y-axis title is this same text (figures/make_fig02c.py imports it).
YLABEL = "Spearman ρ with accessibility"
GAP = 0.45     # extra space between "all replacements" and the codon-distance groups
#: Legend text. Five entries name the two models, the before/after markers and the control, so
#: this panel's legend carries the figure's whole key and is set larger than the tick labels.
LEGEND_PT = 9.5


def parse(argv=None):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--n-boot", type=int, default=200)
    pre.add_argument("--seed", type=int, default=20261009)
    known, rest = pre.parse_known_args(argv)
    args = parse_figure_args(__doc__.split("\n\n")[0], rest)
    args.n_boot, args.seed = known.n_boot, known.seed
    return args


def draw(table: pd.DataFrame, guide):
    from matplotlib.lines import Line2D

    strata = list(table["stratum"].unique())
    xs = {s: i + (GAP if i > 0 else 0.0) for i, s in enumerate(strata)}
    models = sorted(guide.models, key=lambda m: m.role != "base")
    model_offsets = {models[0].key: -0.16, models[1].key: 0.16}

    fig = style.figure(*style.PANEL_SIZE)
    ax = style.panel_axes(fig)

    for m in models:
        colour = style.colour_for_role(m.role)
        off = model_offsets[m.key]
        raw = table[(table["measure"] == "plm_vs_access") & (table["series"] == m.key)
                    & (table["control"] == "")].set_index("stratum").loc[strata]
        adj = table[(table["measure"] == "plm_vs_access") & (table["series"] == m.key)
                    & (table["control"] == CONTROL[0])].set_index("stratum").loc[strata]
        x = np.array([xs[s] for s in strata]) + off

        # Raw point (hollow circle)
        ax.plot(x, raw["estimate"], linestyle="none", marker="o", markersize=6,
                markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.4, zorder=4)
        # Adjusted point (solid circle) with bootstrap interval
        ax.vlines(x, adj["lo"], adj["hi"], color=colour, linewidth=1.6, zorder=3)
        ax.plot(x, adj["estimate"], linestyle="none", marker="o", markersize=6,
                color=colour, markeredgecolor=colour, zorder=5)

        # Downward (or upward) adjustment arrows
        for s, r0, r1 in zip(strata, raw["estimate"], adj["estimate"]):
            if abs(r0 - r1) > 0.01:
                ax.annotate("", xy=(xs[s] + off, r1), xytext=(xs[s] + off, r0),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.2,
                                            shrinkA=4.0, shrinkB=4.0, mutation_scale=8), zorder=4)

        # Percentage retained at "all"
        if ALL in raw.index and ALL in adj.index:
            kept = adj.loc[ALL, "estimate"] / raw.loc[ALL, "estimate"]
            ax.text(xs[ALL] + off + (-0.09 if off < 0 else 0.09),
                    (raw.loc[ALL, "estimate"] + adj.loc[ALL, "estimate"]) / 2,
                    f"{kept:.0%}\nkept", ha="right" if off < 0 else "left", va="center",
                    fontsize=7.5, color=colour)

    # Grantham control alone (comparator marker)
    sub_ctrl = table[(table["measure"] == "property_vs_access") & (table["series"] == CONTROL[0])
                     & (table["control"] == "")].set_index("stratum").loc[strata]
    ax.plot([xs[s] for s in strata], sub_ctrl["estimate"], linestyle="none", marker="_",
            markersize=12, markeredgewidth=1.6, color=style.colour_for_role("biochemistry"),
            alpha=0.8, zorder=2)

    # No rule in the gap between "all" and the codon-distance groups. ``GAP`` already separates
    # them, and a line drawn where there are no points reads as something plotted.
    style.reference_line(ax, 0.0, "no relationship = 0", label_loc="left")
    ax.texts[-1].set_fontsize(9)
    ax.set_xticks([xs[s] for s in strata])
    ax.set_xticklabels([GROUP_LABELS.get(s, s) for s in strata], fontsize=style.PANEL_TICK_PT)
    ax.tick_params(axis="y", labelsize=style.PANEL_TICK_PT)
    ax.set_xlim(-0.55, xs[strata[-1]] + 0.55)
    ax.set_ylim(-0.24, 0.95)

    legend_handles = [
        Line2D([], [], linestyle="none", marker="o", color=style.colour_for_role(m.role), label=m.label)
        for m in models
    ]
    legend_handles += [
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
               markeredgecolor=style.INK_MUTED, markeredgewidth=1.2, label="before: Spearman ρ"),
        Line2D([], [], linestyle="none", marker="o", color=style.INK_MUTED, label="after: Grantham removed"),
        Line2D([], [], linestyle="none", marker="_", markersize=10, markeredgewidth=1.6,
               color=style.colour_for_role("biochemistry"), label=CONTROL[1]),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=LEGEND_PT, handlelength=1.0,
              borderaxespad=0.3, labelspacing=0.3, framealpha=0.0)
    style.finalise_axes(ax, xlabel="Nucleotide changes needed", ylabel=YLABEL)
    style.place_panel_labels(ax)
    return fig


def main(argv=None) -> int:
    args = parse(argv)
    guide = args.guide
    frame = guide.read_table("replacement_scores", args.results)
    guide.check_lineages(frame["lineage"].unique(), f"{guide.name} replacement_scores")
    chosen = primary_lineage(frame, [(lin.id, lin.root_type) for lin in guide.lineages],
                             guide.primary_lineage)
    one = frame[frame["lineage"].astype(str) == chosen]
    table = strata_table(one, [m.key for m in guide.models], [CONTROL[0], "blosum62"],
                         n_boot=args.n_boot, seed=args.seed).assign(lineage=chosen)
    fig = draw(table, guide)
    paths = style.save_panel(fig, args.panels_out / STEM, suffixes=(".pdf", ".png"))
    keep = table[(table["control"] == "") | (table["control"] == CONTROL[0])]
    write_values(keep, args.out, STEM)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
