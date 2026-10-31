"""Collision-aware placement of text labels on an axes.

Direct labels are mandatory in the house style (identity must never ride on colour alone),
but a fixed offset from a line end collides with other curves, reference-line labels and
neighbouring labels as soon as the data change. This places each label at the first of its
candidate positions that overlaps nothing, falling back to the least-bad one.

It measures real text extents, so call it after the figure layout is final
(``fig.canvas.draw()``). It does not move anything that is already drawn.

Used by figures whose panels have free-running curves; it is not a general solver, only a
greedy one, and every figure must still be rendered and looked at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Annotation

__all__ = ["Candidate", "curve_obstacles", "place_label", "place_existing", "expand"]


@dataclass(frozen=True)
class Candidate:
    """One place a label could go: an anchor in ``xycoords`` and an offset from it."""
    xy: tuple[float, float]
    offset: tuple[float, float]       # points
    ha: str
    va: str
    xycoords: object = "data"


def expand(box: tuple[float, float, float, float], pad: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def curve_obstacles(ax: Axes, curves, *, density: int = 4) -> np.ndarray:
    """Display-space points along each (x, y) curve, densified so steep segments are caught."""
    pts = []
    for x, y in curves:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 2:
            continue
        fine = np.linspace(x.min(), x.max(), len(x) * density)
        pts.append(np.column_stack([fine, np.interp(fine, x, y)]))
    if not pts:
        return np.empty((0, 2))
    return ax.transData.transform(np.vstack(pts))


def _extent(ann: Annotation, renderer) -> tuple[float, float, float, float]:
    b = ann.get_window_extent(renderer)
    return (b.x0, b.y0, b.x1, b.y1)


def _points_inside(box, points: np.ndarray) -> int:
    if not len(points):
        return 0
    x0, y0, x1, y1 = box
    inside = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    return int(inside.sum())


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _score(ax: Axes, box, obstacles, placed, pad_px: float) -> int:
    renderer = ax.figure.canvas.get_renderer()
    ab = ax.get_window_extent(renderer)
    padded = expand(box, pad_px)
    score = _points_inside(padded, obstacles)
    score += 1000 * sum(_boxes_overlap(padded, other) for other in placed)
    outside = box[0] < ab.x0 - 1 or box[2] > ab.x1 + 1 or box[1] < ab.y0 - 1 or box[3] > ab.y1 + 1
    return score + 500 * bool(outside)


def _apply(ann: Annotation, cand: Candidate) -> None:
    ann.xy = cand.xy
    ann.xycoords = cand.xycoords
    ann.xyann = cand.offset
    ann.set_ha(cand.ha)
    ann.set_va(cand.va)


def place_existing(
    ax: Axes,
    ann: Annotation,
    candidates: list[Candidate],
    *,
    obstacles: np.ndarray,
    placed: list,
    pad_px: float = 2.0,
) -> int:
    """Move an annotation that already exists (e.g. a reference-line label) to its best candidate.

    Returns the score of the chosen position (0 is clean) and appends its box to ``placed``.
    """
    if not candidates:
        raise ValueError("place_existing needs at least one candidate")
    renderer = ax.figure.canvas.get_renderer()
    best = None
    for cand in candidates:
        _apply(ann, cand)
        box = _extent(ann, renderer)
        score = _score(ax, box, obstacles, placed, pad_px)
        if best is None or score < best[0]:
            best = (score, cand, box)
            if score == 0:
                break
    score, cand, box = best
    _apply(ann, cand)
    placed.append(expand(box, pad_px))
    return score


def place_label(
    ax: Axes,
    text: str,
    candidates: list[Candidate],
    *,
    obstacles: np.ndarray,
    placed: list,
    pad_px: float = 2.0,
    **text_kwargs,
) -> tuple[Annotation, int]:
    """Add ``text`` at the best candidate; append its box to ``placed``. Returns (artist, score).

    Score = obstacle points inside the box + 1000 per overlap with an already placed label
    + 500 if the box leaves the axes; ties go to the earlier candidate. Score 0 is a clean
    placement.
    """
    first = candidates[0]
    ann = ax.annotate(text, xy=first.xy, xycoords=first.xycoords, xytext=first.offset,
                      textcoords="offset points", ha=first.ha, va=first.va, **text_kwargs)
    score = place_existing(ax, ann, candidates, obstacles=obstacles, placed=placed, pad_px=pad_px)
    return ann, score
