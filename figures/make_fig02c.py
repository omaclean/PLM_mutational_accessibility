"""Figure 2, panel C: correlation with accessibility across fine-tuning epochs.

    python figures/make_fig02c.py

Mean Spearman correlation between the model's probability and mutational accessibility at the
raw model and after each of 20 fine-tuning epochs, drawn from
``data/frozen_panels/fig2c_epoch_values.tsv``.

**Where the numbers come from.** The panel was published as a picture with no table behind it
(every epoch table in the repository holds only the raw and final model, and the run that made
the 20-epoch series is gone). ``tools/digitise_epoch_curve.py`` reads the plotted values back
off that picture, to a few 1e-4. They are means over three lineages, as published; the lineages
are scored at the same sites, so they are pseudo-replicates and the mean is a smoothing of what
one lineage would give, not an independent estimate. Redrawing from a fresh per-epoch run on the
one lineage that panels A, B and D use would replace this table and nothing else.

Drawn in the shared panel box (``style.PANEL_AXES``), at the shared size and label sizes, and
saved uncropped, so its axes and text line up with panels A, B and D. The y title is panel D's
(``make_fig02d.YLABEL``): both plot the same statistic.

Writes PDF and PNG. It is a panel, not a figure: the composite
(``guides/composites/fig02.yaml``) places it and draws its header.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd

from make_fig02d import YLABEL
from mutacc import style
from mutacc.guide import PANELS_DIR, REPO_ROOT
from mutacc.io import write_caption, write_values

STEM = "fig02c_epoch_panel"
VALUES = REPO_ROOT / "data" / "frozen_panels" / "fig2c_epoch_values.tsv"
LABEL_EVERY = 5          # x ticks: "raw", then every fifth epoch
YLIM = (0.25, 0.68)
#: Spacing of the y ticks, pinned rather than left to the autolocator. The whole point of this
#: panel is how little moves after epoch 1, so the plateau has to be readable to a hundredth;
#: with the ticks automatic, the larger tick font quietly coarsened them to steps of 0.1.
YTICK_STEP = 0.05
XLIM = (-0.7, 20.7)
MARKER_SIZE = 5.5
LINE_WIDTH = 1.6
COLUMNS = ["epoch", "label", "spearman_r", "source"]


def load_values(path: Path = VALUES) -> pd.DataFrame:
    """The digitised values, checked: raw then epochs 1 to 20 in order, correlations in [-1, 1]."""
    frame = pd.read_csv(path, sep="\t")
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame["epoch"].tolist() != list(range(len(frame))) or len(frame) < 2:
        raise ValueError(f"{path}: epochs must run 0 (raw), 1, 2, ... in order")
    if not np.isfinite(frame["spearman_r"]).all() or not frame["spearman_r"].between(-1, 1).all():
        raise ValueError(f"{path}: correlations must be finite and within [-1, 1]")
    return frame


def tick_epochs(n_epochs: int, every: int = LABEL_EVERY) -> list[int]:
    """Epochs that get a tick: the raw model (0) and each multiple of ``every``."""
    if every < 1:
        raise ValueError("every must be a positive integer")
    return [0] + [e for e in range(1, n_epochs + 1) if e % every == 0]


def draw(frame: pd.DataFrame):
    from matplotlib.colors import to_hex
    from matplotlib.ticker import MultipleLocator

    colour = to_hex("C0")        # the original panel's blue, the matplotlib default
    fig = style.figure(*style.PANEL_SIZE)
    ax = style.panel_axes(fig)
    ax.plot(frame["epoch"], frame["spearman_r"], color=colour, linewidth=LINE_WIDTH, marker="o",
            markersize=MARKER_SIZE, zorder=3)
    ticks = tick_epochs(int(frame["epoch"].max()))
    ax.set_xticks(ticks)
    ax.set_xticklabels(["raw" if e == 0 else str(e) for e in ticks])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.yaxis.set_major_locator(MultipleLocator(YTICK_STEP))
    ax.grid(True, linestyle="-", alpha=0.6)
    ax.tick_params(labelsize=style.PANEL_TICK_PT)
    style.finalise_axes(ax, xlabel="Epoch", ylabel=YLABEL)
    style.place_panel_labels(ax)
    return fig


def caption(frame: pd.DataFrame) -> str:
    raw, last = frame["spearman_r"].iloc[0], frame["spearman_r"].iloc[-1]
    first = frame["spearman_r"].iloc[1]
    plateau = frame["spearman_r"].iloc[2:]
    return f"""
    **Correlation with mutational accessibility across fine-tuning epochs.** Spearman rank
    correlation between the model's probability for each amino-acid replacement and its
    mutational accessibility, for the model as released ("raw", {raw:.3f}) and after each of
    {int(frame['epoch'].max())} epochs of fine-tuning. It rises to {first:.3f} after the first
    epoch and then stays between {plateau.min():.3f} and {plateau.max():.3f} (epoch
    {int(frame['epoch'].iloc[-1])}: {last:.3f}): almost all of the change happens in the first epoch.

    The values are means over three lineages and were read back from the published version of this
    panel (to about 0.0005); the per-epoch table behind it no longer exists. The lineages are scored
    at the same sites, so they are pseudo-replicates of one another and the mean stands in for what
    a single lineage would give.
    """


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--values", type=Path, default=VALUES)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--panels-out", type=Path, default=PANELS_DIR,
                        help=f"where the panel render goes (default: "
                             f"{PANELS_DIR.relative_to(REPO_ROOT).as_posix()}/)")
    args = parser.parse_args(argv)
    frame = load_values(args.values)
    paths = style.save_panel(draw(frame), args.panels_out / STEM, suffixes=(".pdf", ".png"))
    write_values(frame[COLUMNS], args.out, STEM)
    write_caption(caption(frame), args.out, STEM, panel="C")
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
