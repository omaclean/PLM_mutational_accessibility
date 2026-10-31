"""The tree-label enlarger must never trade bigger text for an overlap."""

from pathlib import Path

import fitz
import pytest

from mutacc.guide import REPO_ROOT

import sys
sys.path.insert(0, str(REPO_ROOT / "tools"))
import enlarge_tree_labels as E  # noqa: E402

SRC = REPO_ROOT / "data" / "frozen_panels" / "fig1a_clade_tree.pdf"
DST = REPO_ROOT / "data" / "frozen_panels" / "fig1a_clade_tree_large_labels.pdf"


@pytest.fixture(scope="module")
def original():
    page = fitz.open(SRC)[0]
    drawings = page.get_drawings()
    return page, drawings, E.find_plan(drawings)


@pytest.fixture(scope="module")
def derived():
    page = fitz.open(DST)[0]
    return page, page.get_drawings()


def black_glyphs(drawings):
    return [d for d in drawings if E._is_glyph(d)]


def test_finds_the_labels_and_the_legend(original):
    _, _, plan = original
    assert len(plan.labels) == 14
    assert plan.legend_rect is not None and len(plan.legend) > 6
    assert sum(len(l.glyphs) for l in plan.labels) + len(plan.legend) >= len(black_glyphs(original[1]))


def test_the_source_has_no_overlaps_to_begin_with(original):
    page, drawings, plan = original
    assert E.solve(drawings, plan, page.rect, 1.0)[1] == []


def test_the_shipped_scale_is_collision_free(original):
    page, drawings, plan = original
    assert E.solve(drawings, plan, page.rect, 1.65)[1] == []


def test_an_impossible_scale_is_refused_not_drawn(original):
    page, drawings, plan = original
    problems = E.solve(drawings, plan, page.rect, 4.0)[1]
    assert problems, "a 4x enlargement cannot fit; the guard must say so"


def test_nothing_but_the_labels_changed(original, derived):
    (_, before, plan), (_, after) = original, derived
    assert len(before) == len(after)
    moved = {i for l in plan.labels for i in l.glyphs} | set(plan.legend)
    for i, (a, b) in enumerate(zip(before, after)):
        assert a["type"] == b["type"] and len(a["items"]) == len(b["items"])
        if i in moved:
            continue
        ra, rb = fitz.Rect(a["rect"]), fitz.Rect(b["rect"])
        assert abs(rb.x0 - (ra.x0 + E.PAD_LEFT)) < 1e-3 and abs(rb.y0 - ra.y0) < 1e-3, f"path {i} moved"
        assert abs(rb.width - ra.width) < 1e-3 and abs(rb.height - ra.height) < 1e-3, f"path {i} resized"


def test_labels_are_actually_bigger(original, derived):
    (_, before, plan), (_, after) = original, derived
    ratios = []
    for lab in plan.labels:
        for i in lab.glyphs:
            ra, rb = fitz.Rect(before[i]["rect"]), fitz.Rect(after[i]["rect"])
            if ra.height > 3:
                ratios.append(max(rb.width / ra.width, rb.height / ra.height))
    assert min(ratios) > 1.4


def test_every_glyph_stays_on_the_page(derived):
    page, drawings = derived
    for d in black_glyphs(drawings):
        assert page.rect.contains(d["rect"]), d["rect"]


# --- the legend's own scale factor -----------------------------------------------
# The labels cap --k at 1.65 because they sit against their branches; the legend sits
# in empty page and clears the tree to about 4.6. --legend-k separates the two.

def test_legend_defaults_to_the_label_factor(original):
    """Omitting legend_k must behave exactly as the single-factor version did."""
    page, drawings, plan = original
    same = E.solve(drawings, plan, page.rect, 1.65)[0]
    explicit = E.solve(drawings, plan, page.rect, 1.65, None)[0]
    assert same == explicit
    for i in plan.legend:
        assert same[i][2] == 1.65


def test_legend_scales_past_the_label_cap(original):
    """2.5 is refused for the labels but fine for the legend alone."""
    page, drawings, plan = original
    assert E.solve(drawings, plan, page.rect, 2.5)[1], "2.5 should be too big for the labels"
    transforms, problems = E.solve(drawings, plan, page.rect, 1.65, 2.5)
    assert problems == []
    for i in plan.legend:
        assert transforms[i][2] == 2.5
    for lab in plan.labels:
        for i in lab.glyphs:
            assert transforms[i][2] == 1.65


def test_an_oversized_legend_is_refused(original):
    """The legend is checked against the tree and the page like everything else."""
    page, drawings, plan = original
    problems = E.solve(drawings, plan, page.rect, 1.65, 8.0)[1]
    assert problems, "an 8x legend cannot fit; the guard must say so"
    assert any("legend" in p for p in problems), problems


def test_legend_grows_about_its_bottom_right_corner(original):
    """It must expand into empty page, not drift over the tree."""
    page, drawings, plan = original
    transforms = E.solve(drawings, plan, page.rect, 1.65, 2.5)[0]
    box = E._apply(plan.legend_rect, transforms[plan.legend[0]])
    assert abs(box.x1 - plan.legend_rect.x1) < 1e-6
    assert abs(box.y1 - plan.legend_rect.y1) < 1e-6
    assert box.width > plan.legend_rect.width * 2.4


def test_the_shipped_legend_is_bigger_than_the_shipped_labels(original, derived):
    """The published panel is built with --k 1.65 --legend-k 2.5; check it stayed that way."""
    (_, before, plan), (_, after) = original, derived
    ratios = []
    for i in plan.legend:
        ra, rb = fitz.Rect(before[i]["rect"]), fitz.Rect(after[i]["rect"])
        if ra.width > 1 and ra.height > 1:
            ratios.append(max(rb.width / ra.width, rb.height / ra.height))
    assert ratios, "the legend should hold measurable paths"
    assert min(ratios) > 2.0, "the shipped legend should carry its own, larger factor"
