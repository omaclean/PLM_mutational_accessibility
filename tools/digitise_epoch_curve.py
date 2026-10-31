"""Recover the plotted values of the frozen Figure 2C epoch curve from its raster.

Panel 2C was published as a picture: a mean Spearman correlation at "raw" and at each of 20
fine-tuning epochs. The table behind it does not exist anywhere (every epoch table in the
repository holds only the raw and final model, and the run that produced the 20-epoch series
is gone), and the panel cannot be redrawn without its numbers. So they are read back off the
picture, the way one digitises any published figure:

* the markers are the filled blue discs, found by opening the blue mask with a disc larger than
  the line is thick (the line vanishes, the markers stay) and taking each component's centroid;
* the y axis is calibrated from the tick marks on the left axis, whose labels are 0.30 to 0.65
  in steps of 0.05 (``Y_TICK_VALUES``; read from the image, and the tool stops unless it finds
  exactly that many ticks, evenly spaced);
* the x position of each marker is matched to the x tick marks, so a marker cannot be assigned
  to the wrong epoch.

A pixel is 0.0004 of a correlation here, so the values are good to a few 1e-4: the table carries
four decimals and says where it came from. They are **means over three lineages**, as the
original panel was; the lineages are pseudo-replicates (scored at the same sites), so the mean
is a smoothing of what one of them would give, not an independent estimate.

    python tools/digitise_epoch_curve.py \\
        --src data/frozen_panels/fig2c_plm_epoch.png \\
        --dst data/frozen_panels/fig2c_epoch_values.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]

Y_TICK_VALUES = np.round(np.arange(0.30, 0.6501, 0.05), 2)   # the labels on the left axis
N_X_TICKS = 22                  # "mut", "raw", then epochs 1 to 20; the curve starts on "raw"
N_POINTS = 21                   # "raw" and epochs 1 to 20
CURVE_BLUE = (31, 119, 180)
CURVE_TOLERANCE = 60            # colour distance for the blue mask (anti-aliased edges included)
DARK = 120
OPEN_RADIUS = 9                 # larger than the line's half thickness, smaller than a marker's radius
SOURCE = "digitised from data/frozen_panels/fig2c_plm_epoch.png"


def _runs(indices: np.ndarray, gap: int = 1) -> list[list[int]]:
    groups: list[list[int]] = []
    for i in indices:
        if groups and i - groups[-1][-1] <= gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def find_frame(rgb: np.ndarray) -> tuple[int, int, int]:
    """Row of the bottom axis line, row of the top, column of the left axis line."""
    dark = rgb.mean(axis=2) < DARK
    rows = np.where(dark.sum(axis=1) > 0.6 * rgb.shape[1])[0]
    if len(rows) < 2:
        raise ValueError("no plot frame found: expected long dark rows for the top and bottom axes")
    top, bottom = int(rows.min()), int(rows.max())
    cols = np.where(dark[top:bottom].sum(axis=0) > 0.8 * (bottom - top))[0]
    if not len(cols):
        raise ValueError("no left axis line found")
    return bottom, top, int(cols.min())


def x_ticks(rgb: np.ndarray, bottom: int) -> list[int]:
    band = (rgb[bottom + 3: bottom + 11].mean(axis=2) < DARK).all(axis=0)
    ticks = [int(np.mean(g)) for g in _runs(np.where(band)[0])]
    if len(ticks) != N_X_TICKS:
        raise ValueError(f"expected {N_X_TICKS} x tick marks, found {len(ticks)}")
    return ticks


def y_calibration(rgb: np.ndarray, top: int, bottom: int, left: int) -> tuple[float, float]:
    """(slope, intercept) of value = slope * row + intercept, from the left-axis tick marks."""
    band = (rgb[top:bottom, left - 11: left - 3].mean(axis=2) < DARK).all(axis=1)
    rows = [float(np.mean(g)) + top for g in _runs(np.where(band)[0])]
    if len(rows) != len(Y_TICK_VALUES):
        raise ValueError(f"expected {len(Y_TICK_VALUES)} y tick marks, found {len(rows)}")
    steps = np.diff(rows)
    if np.abs(steps - np.median(steps)).max() > 2:
        raise ValueError(f"y ticks are not evenly spaced: {steps.tolist()}")
    slope, intercept = np.polyfit(rows, Y_TICK_VALUES[::-1], 1)    # top tick is the largest value
    return float(slope), float(intercept)


def markers(rgb: np.ndarray, top: int, bottom: int, left: int) -> list[tuple[float, float]]:
    """Centroids (x, y) of the marker discs inside the plot, left to right."""
    inside = rgb[top + 3: bottom - 2, left + 3:].astype(int)
    mask = np.abs(inside - np.array(CURVE_BLUE)).sum(axis=2) < CURVE_TOLERANCE
    yy, xx = np.mgrid[-OPEN_RADIUS: OPEN_RADIUS + 1, -OPEN_RADIUS: OPEN_RADIUS + 1]
    kept = ndimage.binary_opening(mask, structure=(xx ** 2 + yy ** 2) <= OPEN_RADIUS ** 2)
    labels, n = ndimage.label(kept)
    if n != N_POINTS:
        raise ValueError(f"expected {N_POINTS} markers, found {n}")
    centres = ndimage.center_of_mass(kept, labels, range(1, n + 1))
    return sorted((float(cx) + left + 3, float(cy) + top + 3) for cy, cx in centres)


def digitise(src: Path) -> pd.DataFrame:
    rgb = np.array(Image.open(src).convert("RGB"))
    bottom, top, left = find_frame(rgb)
    ticks = x_ticks(rgb, bottom)
    slope, intercept = y_calibration(rgb, top, bottom, left)
    pts = markers(rgb, top, bottom, left)
    raw_index = 1
    rows = []
    for k, (x, y) in enumerate(pts):
        tick = ticks[raw_index + k]
        if abs(x - tick) > 4:
            raise ValueError(f"marker {k} is {x - tick:+.1f} px from its tick: assigned to the wrong epoch?")
        rows.append(dict(epoch=k, label="raw" if k == 0 else str(k), x_px=round(x, 1), y_px=round(y, 1),
                         spearman_r=round(slope * y + intercept, 4), source=SOURCE))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, default=REPO / "data/frozen_panels/fig2c_plm_epoch.png")
    parser.add_argument("--dst", type=Path, default=REPO / "data/frozen_panels/fig2c_epoch_values.tsv")
    args = parser.parse_args(argv)
    table = digitise(args.src)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.dst, sep="\t", index=False)
    print(f"wrote {args.dst} ({len(table)} points; raw {table.spearman_r.iloc[0]:.4f}, "
          f"epoch 20 {table.spearman_r.iloc[-1]:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
