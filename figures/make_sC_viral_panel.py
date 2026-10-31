"""S3: how well do two untuned protein language models fill in a masked amino acid, across proteins?

    python figures/make_sC_viral_panel.py --results results/viral_panel

Every position of every protein is masked in turn. The model is asked for a probability over the 20
amino acids at that position, and we record the probability it gave the TRUE one. The dot for a protein
is that probability averaged over all its positions: "how much of its belief did the model put on the
right answer, when it could not see it?". 100% would be a model that always knew.

Two floors are drawn, and they are different things. The dashed line is 5%: one amino acid in twenty,
what a model scores knowing nothing whatever. It is the same number for every protein, which is why it
can be a line. The grey tick on each row is that protein's OWN composition baseline -- guessing amino
acids in the proportions that protein happens to contain, ignoring the sequence around them. That is a
per-protein quantity (it runs from about 5.5% to 7.5% across this panel) and drawing its average as a
single line would state a constant that does not exist.

The viral groups are the original panel; the eukaryotic, bacterial and phage groups are there to give
the viral numbers a scale. No model here has been fine-tuned. Proteins are grouped and, within a group,
ordered by the average of the two models so that neither model sets the order.

Reads ``viral_panel_summary.tsv.gz`` from the results directory. Group names, labels, model names, the
colour role of each model and the display order all come from that table; nothing is named here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from mutacc import style
from mutacc.io import write_caption, write_values

STEM = "sC_viral_panel_recovery"
MARKERS = {"base": "o", "base_second": "^"}     # shape as well as colour, so identity never rests on hue alone
GROUP_GAP = 0.35                                 # extra rows of space above each group's heading
ROW = 1.0

#: 41 proteins in 6 groups, sized for A4. The page scales the whole figure to fit, so what
#: a reader actually sees is point size RELATIVE to FIG_WIDTH -- growing the canvas and the
#: type together changes nothing on paper. The width below is therefore deliberately modest
#: and the type is set against it. Two things follow. The aspect is ~1.3, not square: that
#: is the shape of A4's printable area, so the panel fills the page instead of leaving
#: margins that the scaling would charge to the type. And the row pitch is the binding
#: constraint -- in one column, 41 rows plus 6 headings divide the height, and no label can
#: be taller than its own row, which caps the type at roughly what is set here. Bigger text
#: than this needs fewer rows per column, not a bigger figure.
FIG_WIDTH = 10.5       # inches
ROW_HEIGHT = 0.265     # inches per protein row -- about 19 pt of pitch once headings are in
GROUP_PAD = 0.32       # inches added per group heading, on top of the row it occupies
LABEL_SIZE = 15.0      # protein names
GROUP_SIZE = 16.5      # group headings
TICK_SIZE = 17.0
AXIS_LABEL_SIZE = 17.0
LEGEND_SIZE = 16.0
NOTE_SIZE = 13.5       # the "widest disagreement" block and the baseline label
MARKER_AREA = 110      # points squared: ~12 pt across, inside a 19 pt row
DUMBBELL_WIDTH = 2.8

#: One amino acid in twenty. The only floor on this panel that is genuinely the same for
#: every protein, so the only one that has any business being drawn as a single line.
BLIND_GUESS = 1.0 / 20.0
COMPOSITION_LABEL = "guess from other AA usage"


def load(results: Path) -> pd.DataFrame:
    path = results / "viral_panel_summary.tsv.gz"
    df = pd.read_csv(path, sep="\t")
    need = {"model", "role", "label", "display_label", "group", "group_order", "length",
            "mean_p_wt", "top1", "composition_baseline", "n_sites"}
    if need - set(df.columns):
        raise ValueError(f"{path.name} is missing columns {sorted(need - set(df.columns))}")
    if not df["mean_p_wt"].between(0, 1).all():
        raise ValueError("a mean probability lies outside [0, 1]")
    unknown = set(df["role"]) - set(MARKERS)
    if unknown:
        raise ValueError(f"no marker defined for role(s) {sorted(unknown)}")
    if df.duplicated(["model", "label"]).any():
        raise ValueError("more than one row for a (model, protein) pair")
    per_model = df.groupby("label")["model"].nunique()
    if per_model.nunique() != 1:
        raise ValueError("not every protein has been scored by every model")
    return df


def layout(df: pd.DataFrame) -> pd.DataFrame:
    """One row per protein with its vertical position. Order: group, then mean over models (high first)."""
    wide = df.pivot(index="label", columns="model", values="mean_p_wt")
    info = df.drop_duplicates("label").set_index("label")[["display_label", "group", "group_order", "length",
                                                           "composition_baseline"]]
    info["mean_of_models"] = wide.mean(axis=1)
    info = info.sort_values(["group_order", "mean_of_models"], ascending=[True, False])
    y, pos, last = [], 0.0, None
    for group in info["group"]:
        if group != last:
            pos += GROUP_GAP if last is not None else 0.0
            pos += ROW                       # the heading row
            last = group
        y.append(pos)
        pos += ROW
    info["y"] = y
    return info


def draw(df: pd.DataFrame):
    info = layout(df)
    models = df.drop_duplicates("model").set_index("model")["role"]
    fig = style.figure(FIG_WIDTH, ROW_HEIGHT * len(info) + 1.9 + GROUP_PAD * info["group"].nunique())
    ax = fig.add_subplot(111)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)

    for label, row in info.iterrows():
        vals = df[df["label"] == label].set_index("model")["mean_p_wt"]
        ax.plot([vals.min(), vals.max()], [row["y"]] * 2, color=style.GRID, linewidth=DUMBBELL_WIDTH,
                zorder=2, solid_capstyle="round")
    # Each protein's own composition floor, on its own row, because it is not a constant.
    ax.scatter(info["composition_baseline"], info["y"], s=MARKER_AREA, marker="|",
               color=style.NULL_GREY, linewidths=2.2, zorder=2.5, label=COMPOSITION_LABEL)
    for model, role in models.items():
        sub = df[df["model"] == model].set_index("label").reindex(info.index)
        ax.scatter(sub["mean_p_wt"], info["y"], s=MARKER_AREA, marker=MARKERS[role],
                   color=style.colour_for_role(role), edgecolors="white", linewidths=1.4,
                   zorder=3, label=model)

    for group, block in info.groupby("group", sort=False):
        # Offset in points, not data: at this size the heading ran into the axis spine.
        ax.annotate(group, xy=(0.0, block["y"].min() - ROW), xycoords=ax.get_yaxis_transform(),
                    xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    fontweight="bold", fontsize=GROUP_SIZE, color=style.INK, clip_on=False)
    ax.set_yticks(info["y"])
    ax.set_yticklabels(info["display_label"], fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_ylim(info["y"].max() + 1.2, -0.4)

    style.reference_line(ax, BLIND_GUESS, f"{BLIND_GUESS:.0%} = blind guessing, 1 amino acid in 20",
                         axis="x", label_side="above")
    ax.texts[-1].set_fontsize(NOTE_SIZE)   # the helper labels its line at 6 pt
    # Log x. Two thirds of the panel sits between 5% and 15%, so on a linear 0-100% axis the
    # two models overprint on most rows and the dumbbells collapse to a dot.
    ax.set_xscale("log")
    ax.set_xlim(0.04, 1.0)
    ax.minorticks_off()
    ax.set_xticks([0.05, 0.1, 0.2, 0.4, 0.8])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    # On the figure, not the axes. Protein names take a third of the width, so the axes sit
    # well right of centre; an axes xlabel is centred on them and this one-line sentence then
    # runs off the right edge. supxlabel centres on the canvas, and constrained_layout gives
    # it its own row rather than overlapping the tick labels.
    style.finalise_axes(ax)
    fig.supxlabel("Average probability the model gave the true amino acid at a masked position",
                  fontsize=AXIS_LABEL_SIZE, color=style.INK)

    # Models first, then the composition tick: the legend reads as "the data, then the floor".
    # It goes above the panel, not inside it: every corner of a 41-row dumbbell chart has either
    # a dot or a composition tick in it, and the inside-lower-right version sat on the phage rows.
    handles, names = ax.get_legend_handles_labels()
    order = sorted(range(len(names)), key=lambda i: names[i] == COMPOSITION_LABEL)
    style.legend_above(fig, [handles[i] for i in order], [names[i] for i in order],
                       ncol=len(order), frameon=False, handletextpad=0.4,
                       fontsize=LEGEND_SIZE, columnspacing=2.0)
    return fig, info


def caption(df: pd.DataFrame, info: pd.DataFrame) -> str:
    n_prot = len(info)
    residues = int(df.drop_duplicates("label")["n_sites"].sum())
    names = " and ".join(df.drop_duplicates("model")["model"])
    groups = ", ".join(info["group"].drop_duplicates())
    lo, hi = info["composition_baseline"].min(), info["composition_baseline"].max()
    return f"""
    **How well two untuned protein language models recover a masked amino acid.**
    Protein language models are trained to fill in a masked position in a protein sequence. For each of
    {n_prot} proteins ({residues:,} positions in all) we masked one amino acid at a time and asked {names}
    what it expected there. Each dot is the probability the model put on the true amino acid, averaged
    over every position of the protein; 100% would mean the model always knew. Proteins are grouped
    ({groups}) and ordered within a group by the mean of the two models, so neither model sets the order.
    Neither model has been fine-tuned. Sequences are the reviewed UniProt entries listed in the sequence
    table, with their accessions.

    Two floors are marked, and they are not the same quantity. The dashed line at {BLIND_GUESS:.0%} is
    one amino acid in twenty: what a model scores knowing nothing at all, and the same number for every
    protein. The grey tick on each row is that protein's own composition baseline, the score for guessing
    amino acids in the proportions that protein contains while ignoring the sequence around them; it
    varies by protein ({lo:.1%} to {hi:.1%} here), so it is drawn per row rather than averaged into a
    line that would imply a constant.

    A high score means the sequence is easy to predict from its context: often because it is conserved,
    and in some cases because near-identical copies sit in the training data. A low score is not proof
    that a model "does not know" the protein. Positions are not independent, so no interval is drawn.
    """


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, type=Path, help="directory holding viral_panel_summary.tsv.gz")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    args = ap.parse_args(argv)
    df = load(args.results)
    fig, info = draw(df)
    paths = style.save_figure(fig, args.out / STEM)
    values = df.merge(info[["y"]].reset_index(), on="label").sort_values(["y", "model"])
    write_values(values[["model", "label", "display_label", "group", "length", "n_sites", "mean_p_wt", "top1",
                         "composition_baseline"]], args.out, STEM)
    write_caption(caption(df, info), args.out, STEM)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
