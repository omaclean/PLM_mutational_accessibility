"""Enlarge the text on the frozen Figure 1A tree without touching the tree.

The tree PDF cannot be regenerated (no generator survives, and it is built from
GISAID-derived data), and it has no text layer: every character is a filled vector
path. At the composite's double-column width the labels come out at about 4 pt.

So this reads the page's drawing paths, finds each label as a cluster of black
glyph paths, scales it about a fixed anchor, and writes a new PDF. Nothing else on
the page is edited. The original asset is kept; the derived one is a different file.

Every label stays attached to what it labels. Vertical mutation labels grow upward
from their bottom edge, so the gap to the branch they sit on is unchanged. Most sit
hard against their own vertical branch, so each is scaled about whichever of its left,
centre or right edge keeps it on the correct side of that branch, and may then be
nudged a few points to clear it. The scale-bar label grows upward from the bar. The
legend (box, swatches and text) is scaled as one unit about its bottom-right corner,
into empty page.

A scale is accepted only if every label can be placed with clearance from every tree
stroke (measured on the glyph ink, not its padded box), no two labels touch, and all
stay on the page. With ``--k auto`` the largest scale that passes is used, so
enlarging cannot introduce an overlap.

    python tools/enlarge_tree_labels.py \\
        --src data/frozen_panels/fig1a_clade_tree.pdf \\
        --dst data/frozen_panels/fig1a_clade_tree_large_labels.pdf
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

BLACK = (0.0, 0.0, 0.0)
GAP = 6.0          # glyphs closer than this (pt) belong to one label
INK_INSET = 1.0    # glyph path boxes include side bearings; the ink is about this much tighter
CLEARANCE = 0.75   # required space between label ink and any tree stroke (pt)
LABEL_GAP = 2.0    # required space between two labels (pt)
MAX_SHIFT_X, MAX_SHIFT_Y = 6, 16  # how far a label may be nudged to clear the tree (pt)
PAD_LEFT = 30.0    # blank margin added on the left: the first label sits at the page edge


@dataclass
class Label:
    kind: str                       # "vertical" | "horizontal"
    glyphs: list[int]               # indices into the drawing list
    rect: fitz.Rect


@dataclass
class Plan:
    labels: list[Label]
    legend: list[int] = field(default_factory=list)   # every drawing inside the legend box
    legend_rect: fitz.Rect | None = None


# ---------------------------------------------------------------------------
# Finding the labels
# ---------------------------------------------------------------------------

def _is_glyph(d: dict) -> bool:
    return d["type"] == "f" and tuple(d.get("fill") or ()) == BLACK


def _legend_box(drawings: list[dict]) -> fitz.Rect | None:
    """The one white-filled, stroked box on the page (the legend frame)."""
    boxes = [d for d in drawings if d["type"] == "fs" and tuple(d.get("fill") or ()) == (1.0, 1.0, 1.0)]
    return fitz.Rect(boxes[0]["rect"]) if len(boxes) == 1 else None


def find_plan(drawings: list[dict]) -> Plan:
    legend_rect = _legend_box(drawings)
    def in_legend(r) -> bool:
        return legend_rect is not None and (
            legend_rect.x0 - 1 <= r.x0 and r.x1 <= legend_rect.x1 + 1
            and legend_rect.y0 - 1 <= r.y0 and r.y1 <= legend_rect.y1 + 1)

    glyph_ids = [i for i, d in enumerate(drawings) if _is_glyph(d)]
    parent = {i: i for i in glyph_ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(glyph_ids, 2):
        a = fitz.Rect(drawings[i]["rect"])
        a.x0 -= GAP; a.y0 -= GAP; a.x1 += GAP; a.y1 += GAP
        if a.intersects(drawings[j]["rect"]):
            parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in glyph_ids:
        groups.setdefault(find(i), []).append(i)

    labels = []
    for ids in groups.values():
        rect = fitz.Rect(drawings[ids[0]]["rect"])
        for i in ids:
            rect |= drawings[i]["rect"]
        if in_legend(rect):
            continue  # scaled with the legend as one unit
        # Vertical text reads bottom-to-top; the horizontal one is the scale-bar label.
        labels.append(Label("vertical" if rect.height > rect.width else "horizontal", ids, rect))

    legend = [i for i, d in enumerate(drawings) if in_legend(d["rect"])] if legend_rect is not None else []
    return Plan(labels=labels, legend=legend, legend_rect=legend_rect)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _scale_pt(p, anchor, k):
    return fitz.Point(anchor[0] + k * (p[0] - anchor[0]), anchor[1] + k * (p[1] - anchor[1]))


def _scale_rect(r, anchor, k):
    a, b = _scale_pt((r.x0, r.y0), anchor, k), _scale_pt((r.x1, r.y1), anchor, k)
    return fitz.Rect(min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y))


def _segments(drawings: list[dict], skip: set[int]):
    """Line segments of every stroke not in ``skip`` (curves sampled at 5 points)."""
    for idx, d in enumerate(drawings):
        if idx in skip or d["type"] not in ("s", "fs"):
            continue
        half = (d.get("width") or 0) / 2
        for item in d["items"]:
            if item[0] == "l":
                yield item[1], item[2], half
            elif item[0] == "c":
                p0, c1, c2, p3 = item[1:5]
                pts = []
                for t in (0, .25, .5, .75, 1):
                    u = 1 - t
                    pts.append((u**3*p0[0] + 3*u*u*t*c1[0] + 3*u*t*t*c2[0] + t**3*p3[0],
                                u**3*p0[1] + 3*u*u*t*c1[1] + 3*u*t*t*c2[1] + t**3*p3[1]))
                for a, b in zip(pts, pts[1:]):
                    yield a, b, half
            elif item[0] == "re":
                r = item[1]
                for a, b in (((r.x0, r.y0), (r.x1, r.y0)), ((r.x1, r.y0), (r.x1, r.y1)),
                             ((r.x1, r.y1), (r.x0, r.y1)), ((r.x0, r.y1), (r.x0, r.y0))):
                    yield a, b, half


def _seg_hits_rect(a, b, r: fitz.Rect) -> bool:
    """Liang-Barsky: does segment a-b touch rectangle r?"""
    x0, y0, x1, y1 = a[0], a[1], b[0], b[1]
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - r.x0), (dx, r.x1 - x0), (-dy, y0 - r.y0), (dy, r.y1 - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return True


class _Grid:
    """Spatial hash of stroke segments, so each label is tested against few."""

    def __init__(self, segments, cell=64.0):
        self.cell, self.cells = cell, {}
        for a, b, half in segments:
            x0, x1 = sorted((a[0], b[0])); y0, y1 = sorted((a[1], b[1]))
            for cx in range(int((x0 - half) // cell), int((x1 + half) // cell) + 1):
                for cy in range(int((y0 - half) // cell), int((y1 + half) // cell) + 1):
                    self.cells.setdefault((cx, cy), []).append((a, b, half))

    def hits(self, rect: fitz.Rect) -> bool:
        c = self.cell
        seen = set()
        for cx in range(int(rect.x0 // c), int(rect.x1 // c) + 1):
            for cy in range(int(rect.y0 // c), int(rect.y1 // c) + 1):
                for a, b, half in self.cells.get((cx, cy), ()):
                    if id(a) in seen and id(b) in seen:
                        pass
                    grown = fitz.Rect(rect.x0 - half, rect.y0 - half, rect.x1 + half, rect.y1 + half)
                    if _seg_hits_rect(a, b, grown):
                        return True
        return False


Transform = tuple  # (anchor_x, anchor_y, k, dx, dy)


def _apply(r: fitz.Rect, t: Transform) -> fitz.Rect:
    ax, ay, k, dx, dy = t
    out = _scale_rect(r, (ax, ay), k)
    return fitz.Rect(out.x0 + dx, out.y0 + dy, out.x1 + dx, out.y1 + dy)


def _ink(r: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(r.x0 + INK_INSET, r.y0 + INK_INSET, r.x1 - INK_INSET, r.y1 - INK_INSET)


def _grown(r: fitz.Rect, m: float) -> fitz.Rect:
    return fitz.Rect(r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m)


def _offsets():
    """Nudges to try, smallest first. Upward only: growing a label downward would
    close the gap to the branch it sits on."""
    cand = [(dx, dy) for dx in range(-MAX_SHIFT_X, MAX_SHIFT_X + 1) for dy in range(0, -MAX_SHIFT_Y - 1, -1)]
    return sorted(cand, key=lambda o: abs(o[0]) + abs(o[1]))


_OFFSETS = _offsets()


def place_label(drawings, lab: Label, k: float, grid: _Grid, page: fitz.Rect,
                placed: list[fitz.Rect]) -> Transform | None:
    """Smallest-nudge placement of one label at scale k, or None if it cannot fit."""
    r = lab.rect
    anchors_x = [r.x0, (r.x0 + r.x1) / 2, r.x1] if lab.kind == "vertical" else [(r.x0 + r.x1) / 2]
    best = None
    for ax in anchors_x:
        for dx, dy in _OFFSETS:
            if best is not None and abs(dx) + abs(dy) >= best[0]:
                break
            t = (ax, r.y1, k, dx, dy)
            moved = _apply(r, t)
            if not page.contains(_grown(moved, 1.0)):
                continue
            if any(_grown(moved, LABEL_GAP).intersects(p) for p in placed):
                continue
            if all(not grid.hits(_grown(_ink(_apply(drawings[i]["rect"], t)), CLEARANCE)) for i in lab.glyphs):
                best = (abs(dx) + abs(dy), t)
                break
    return None if best is None else best[1]


def solve(drawings, plan: Plan, page: fitz.Rect, k: float,
          legend_k: float | None = None) -> tuple[dict[int, Transform], list[str]]:
    """Transforms for every label and the legend at scale k, plus anything that fails.

    ``legend_k`` scales the legend by its own factor; it defaults to ``k``. The two are
    separable because the binding constraint is not shared: the labels sit hard against
    their own branches and cap ``k`` at 1.65, while the legend sits in empty page at the
    bottom right and clears the tree until about 4.6. Holding the legend to the labels'
    factor left it at about 6 pt on the printed column.
    """
    if legend_k is None:
        legend_k = k
    skip = {i for lab in plan.labels for i in lab.glyphs} | set(plan.legend)
    grid = _Grid(_segments(drawings, skip))
    page = fitz.Rect(page.x0 - PAD_LEFT, page.y0, page.x1, page.y1)   # the new page is wider on the left
    transforms: dict[int, Transform] = {}
    problems, placed = [], []

    order = sorted(range(len(plan.labels)), key=lambda n: plan.labels[n].rect.x0)   # left to right
    for n in order:
        lab = plan.labels[n]
        t = place_label(drawings, lab, k, grid, page, [r for _, r in placed])
        if t is None:
            problems.append(f"label {n} ({lab.kind}, near x={lab.rect.x0:.0f} y={lab.rect.y0:.0f}) has no clear position")
            continue
        for i in lab.glyphs:
            transforms[i] = t
        placed.append((f"label {n}", _apply(lab.rect, t)))

    if plan.legend_rect is not None:
        t = (plan.legend_rect.x1, plan.legend_rect.y1, legend_k, 0.0, 0.0)
        box = _apply(plan.legend_rect, t)
        if not page.contains(_grown(box, 1.0)):
            problems.append(f"legend at x{legend_k:g} leaves the page")
        elif grid.hits(_grown(box, CLEARANCE)):
            problems.append(f"legend at x{legend_k:g} crosses the tree")
        for i in plan.legend:
            transforms[i] = t
        placed.append(("legend", box))

    for (na, a), (nb, b) in itertools.combinations(placed, 2):
        if _grown(a, LABEL_GAP).intersects(b):
            problems.append(f"{na} and {nb} touch")
    return transforms, problems


def largest_safe_scale(drawings, plan, page, k_max=2.0, step=0.05) -> float:
    k = k_max
    while k > 1.0:
        if not solve(drawings, plan, page, k)[1]:
            return round(k, 2)
        k -= step
    return 1.0


# ---------------------------------------------------------------------------
# Writing the new page
# ---------------------------------------------------------------------------

def _transform_items(items, t: Transform):
    ax, ay, k, dx, dy = t

    def pt(p):
        q = _scale_pt(p, (ax, ay), k)
        return fitz.Point(q.x + dx, q.y + dy)

    def rc(r):
        q = _apply(r, t)
        return q

    out = []
    for item in items:
        op = item[0]
        if op == "l":
            out.append(("l", pt(item[1]), pt(item[2])))
        elif op == "c":
            out.append(("c", *[pt(p) for p in item[1:5]]))
        elif op == "re":
            out.append(("re", rc(item[1]), *item[2:]))
        elif op == "qu":
            q = item[1]
            out.append(("qu", fitz.Quad(*[pt(p) for p in (q.ul, q.ur, q.ll, q.lr)])))
        else:
            raise ValueError(f"unsupported path item {op!r}")
    return out


def _emit(shape: fitz.utils.Shape, d: dict, items) -> None:
    for item in items:
        op = item[0]
        if op == "l":
            shape.draw_line(item[1], item[2])
        elif op == "c":
            shape.draw_bezier(item[1], item[2], item[3], item[4])
        elif op == "re":
            shape.draw_rect(item[1])
        elif op == "qu":
            shape.draw_quad(item[1])


def write_page(drawings, transforms: dict[int, Transform], page_rect: fitz.Rect, dst: Path) -> None:
    """Re-emit every path, applying ``transforms`` to the labels. All else is copied as is.

    One path per path, never merged: overlapping hairlines composite their anti-aliased
    edges one by one, and merging them into a single stroke renders measurably lighter.
    """
    doc = fitz.open()
    page = doc.new_page(width=page_rect.width + PAD_LEFT, height=page_rect.height)
    shape = page.new_shape()
    for idx, d in enumerate(drawings):
        ax, ay, k, dx, dy = transforms.get(idx, (0.0, 0.0, 1.0, 0.0, 0.0))
        _emit(shape, d, _transform_items(d["items"], (ax, ay, k, dx + PAD_LEFT, dy)))
        _finish(shape, d)
    shape.commit()
    doc.save(dst, garbage=3, deflate=True)
    doc.close()


def _finish(shape, d: dict) -> None:
    cap = d.get("lineCap") or (0, 0, 0)
    shape.finish(
        color=d.get("color"), fill=d.get("fill"), width=d.get("width") or 1.0,
        lineCap=cap[0] if cap else 0, lineJoin=d.get("lineJoin") or 0,
        dashes=d.get("dashes") or None, closePath=bool(d.get("closePath")),
        even_odd=bool(d.get("even_odd")),
        stroke_opacity=d.get("stroke_opacity") if d.get("stroke_opacity") is not None else 1,
        fill_opacity=d.get("fill_opacity") if d.get("fill_opacity") is not None else 1,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--k", default="auto", help="scale factor, or 'auto' for the largest safe one")
    ap.add_argument("--k-max", type=float, default=2.0)
    ap.add_argument("--legend-k", type=float, default=None,
                    help="scale the legend by its own factor instead of --k. The legend sits "
                         "in empty page and clears the tree to about 4.6, where the labels cap "
                         "--k at 1.65. Refused on the same checks as everything else.")
    args = ap.parse_args(argv)

    src = fitz.open(args.src)
    page = src[0]
    drawings = page.get_drawings()
    plan = find_plan(drawings)
    _, base = solve(drawings, plan, page.rect, 1.0)
    if base:
        print("the source already has overlaps at k=1:", *base, sep="\n  ")
        return 1
    k = largest_safe_scale(drawings, plan, page.rect, args.k_max) if args.k == "auto" else float(args.k)
    transforms, bad = solve(drawings, plan, page.rect, k, args.legend_k)
    if bad:
        print(f"refusing k={k}:", *bad, sep="\n  ")
        return 1
    write_page(drawings, transforms, page.rect, args.dst)
    legend_k = k if args.legend_k is None else args.legend_k
    print(f"{len(plan.labels)} labels scaled x{k} and the legend x{legend_k:g}; wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
