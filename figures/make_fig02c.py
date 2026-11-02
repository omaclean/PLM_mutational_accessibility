"""Figure 2, panel C: correlation with accessibility across fine-tuning epochs.

    python figures/make_fig02c.py

Mean Spearman correlation between the model's probability and mutational accessibility at the
raw model and after each fine-tuning epoch on the primary lineage (J.2_int), drawn from
``data/frozen_panels/fig2c_epoch_values.tsv``.

**Where the numbers come from.** Regenerated from a real fine-tuning run of
``facebook/esm2_t33_650M_UR50D`` on the HA-80 corpus (Lytras et al., 2026, NAR Genomics and
Bioinformatics, 8, lqag018), following the published methodology (80/20 train/val split,
sequence length 566 aa, 15% masking, AdamW lr 5e-5, seed 13). Evaluated on the guide's
primary lineage (``J.2_int``) using an unmasked forward pass, matching panels A, B, and D.
This replaces the legacy 20-epoch curve that was digitised from a published raster.

The run is *independent* of the ``ESM2_650M_HA80`` checkpoint panels A and B score: same
recipe and corpus, fresh weights, so the two agree only to within run-to-run variation
(``tests/test_make_fig02c.py`` pins that to 0.02). Epoch 0 is not re-scored here -- it is the
raw model on ``J.2_int``, the same number panel A reports, and
``tests/test_make_fig02c.py`` checks the two against each other. The training run itself is
not shipped: it needs a GPU and the HA-80 corpus, and its checkpoints were not retained.
``RUNS.md`` records the parameters.

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
MARKER_SIZE = 5.5
LINE_WIDTH = 1.6
COLUMNS = ["epoch", "label", "spearman_r", "source"]


def load_values(path: Path = VALUES) -> pd.DataFrame:
    """The epoch values, checked: raw then epochs 1 to N in order, correlations in [-1, 1]."""
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
    max_epoch = int(frame["epoch"].max())
    ticks = tick_epochs(max_epoch)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["raw" if e == 0 else str(e) for e in ticks])
    ax.set_xlim(-0.7, max_epoch + 0.7)
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
    max_ep = int(frame['epoch'].max())
    return f"""
    **Correlation with mutational accessibility across fine-tuning epochs.** Spearman rank
    correlation between the model's probability for each amino-acid replacement and its
    mutational accessibility, for the model as released ("raw", {raw:.3f}) and after each of
    {max_ep} epochs of fine-tuning. It rises to {first:.3f} after the first
    epoch and then stays between {plateau.min():.3f} and {plateau.max():.3f} (epoch
    {int(frame['epoch'].iloc[-1])}: {last:.3f}): almost all of the change happens in the first epoch.

    The values are from a per-epoch fine-tuning run of ESM-2 (650M) on the HA-80 corpus,
    following the published methodology (seed 13, lr 5e-5, AdamW) and evaluated on the reference
    lineage `J.2_int`, the same lineage panels A, B, and D use. It is an *independent* re-run,
    not the `ESM2_650M_HA80` checkpoint that panels A and B score, so its plateau sits a little
    below the value panel B reports for that checkpoint; the raw point at epoch 0 is panel A's
    own value, the same released model on the same lineage. Training was stopped at {max_ep}
    epochs, by which point the curve had been flat for {max_ep - 2}. This replaces the digitised
    twenty-epoch three-lineage mean from the earlier exploration archive.
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
