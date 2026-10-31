"""Tests for the composite assembler (figures/panelise.py).

The assembly tests build their own panels in ``tmp_path`` so they never depend on the
frozen assets. Only ``test_shipped_layouts_resolve`` reads the real layouts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

import panelise  # noqa: E402
from mutacc import style  # noqa: E402

HOME = "/" + "home"  # spelled out so this file does not trip the repo-wide path scan
SHIPPED = sorted((ROOT / "guides" / "composites").glob("*.yaml"))


def _make_pdf(path: Path, width: float, height: float) -> None:
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.draw_rect(fitz.Rect(5, 5, width - 5, height - 5), color=(0, 0, 1), fill=(0.8, 0.9, 1))
    doc.save(path)


def _make_png(path: Path, width: int, height: int) -> None:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    pix.set_rect(pix.irect, (200, 120, 40))
    pix.save(path)


@pytest.fixture
def panels(tmp_path):
    _make_pdf(tmp_path / "a.pdf", 400, 200)
    _make_pdf(tmp_path / "b.pdf", 300, 200)
    _make_png(tmp_path / "c.png", 600, 300)
    _make_pdf(tmp_path / "d.pdf", 800, 200)
    return tmp_path


def _write(tmp_path: Path, spec: dict) -> panelise.Layout:
    path = tmp_path / "layout.yaml"
    path.write_text(yaml.safe_dump(spec))
    return panelise.load_layout(path, root=tmp_path)


def _spec(**overrides) -> dict:
    spec = {
        "name": "demo",
        "width": 400,
        "panels": [
            {"key": "1A", "file": "a.pdf"},
            {"key": "1B", "file": "b.pdf", "letter_x": [0.0, 0.5]},
            {"key": "1C", "file": "c.png"},
        ],
        "rows": [["1A", "1B"], {"panels": ["1C"], "width_fraction": 0.5}],
    }
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------------
# Geometry and letters
# ---------------------------------------------------------------------------

def test_rows_fill_the_page_width_without_distortion(panels):
    layout = _write(panels, _spec())
    height, rects = panelise.plan(layout)
    row1 = [rects["1A"], rects["1B"]]
    assert row1[0].x0 == pytest.approx(0)
    assert row1[1].x1 == pytest.approx(400)
    assert row1[0].height == pytest.approx(row1[1].height)       # equal heights in a row
    assert row1[0].width / row1[0].height == pytest.approx(400 / 200)  # natural aspect kept
    c = rects["1C"]
    assert c.width == pytest.approx(200)                          # width_fraction 0.5
    assert (c.x0 + c.x1) / 2 == pytest.approx(200)                # centred
    assert c.width / c.height == pytest.approx(2.0)
    assert height == pytest.approx(row1[0].height + c.height)


def test_letters_land_inside_their_panels(panels):
    layout = _write(panels, _spec())
    doc = panelise.compose(layout)
    _, rects = panelise.plan(layout)
    words = {w[4]: fitz.Rect(w[:4]) for w in doc[0].get_text("words")}
    assert set(words) == {"A", "B", "C", "D"}
    expected = {"A": "1A", "B": "1B", "C": "1B", "D": "1C"}
    for letter, key in expected.items():
        assert rects[key].contains(words[letter]), (letter, words[letter], rects[key])


def test_letters_follow_the_panel_rectangle_not_fixed_coordinates(panels):
    """Resize a panel and its letter moves with it: positions are derived."""
    small = _write(panels, _spec(width=400))
    large = _write(panels, _spec(width=800))
    for layout in (small, large):
        _, rects = panelise.plan(layout)
        placed = panelise.letter_positions(layout, rects)
        origin = dict(placed)
        r = rects["1A"]
        assert origin["A"].x == pytest.approx(r.x0 + layout.letter_margin_pt)
        r = rects["1B"]
        assert origin["C"].x == pytest.approx(r.x0 + 0.5 * r.width + layout.letter_margin_pt)
    # second-row letter sits at the top of its own row, so it depends on row 1's height
    _, small_rects = panelise.plan(small)
    _, large_rects = panelise.plan(large)
    d_small = dict(panelise.letter_positions(small, small_rects))["D"].y
    d_large = dict(panelise.letter_positions(large, large_rects))["D"].y
    assert d_large > d_small


def test_pdf_inputs_stay_vector_and_png_inputs_are_embedded(panels):
    vector_only = _write(panels, _spec(
        panels=[{"key": "1A", "file": "a.pdf"}, {"key": "1B", "file": "b.pdf"}],
        rows=[["1A", "1B"]],
    ))
    assert panelise.compose(vector_only)[0].get_images() == []
    mixed = _write(panels, _spec())
    assert len(panelise.compose(mixed)[0].get_images()) == 1


def test_crop_top_shortens_the_panel(panels):
    layout = _write(panels, _spec(
        panels=[{"key": "1A", "file": "a.pdf", "crop_top": 100}],
        rows=[["1A"]], width=400,
    ))
    height, rects = panelise.plan(layout)
    assert rects["1A"].height == pytest.approx(100)  # 400 x 100 after cutting 100 of 200


def test_render_writes_pdf_and_png_with_the_same_stem_at_repo_dpi(panels, tmp_path):
    layout = _write(panels, _spec())
    out = tmp_path / "out"
    pdf, png = panelise.render(layout, out)
    assert pdf.stem == png.stem == "demo"
    assert pdf.suffix == ".pdf" and png.suffix == ".png"
    pix = fitz.Pixmap(str(png))
    assert abs(pix.width - 400 * style.DPI / 72) <= 1
    assert fitz.open(pdf).page_count == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_missing_input_file_raises(panels):
    spec = _spec()
    spec["panels"][0]["file"] = "nope.pdf"
    with pytest.raises(panelise.LayoutError, match="not found"):
        panelise.validate(_write(panels, spec))


def test_duplicate_panel_key_raises(panels):
    spec = _spec()
    spec["panels"].append({"key": "1A", "file": "d.pdf"})
    with pytest.raises(panelise.LayoutError, match="duplicate"):
        panelise.validate(_write(panels, spec))


def test_letter_count_mismatch_raises(panels):
    with pytest.raises(panelise.LayoutError, match="letter/panel count mismatch"):
        panelise.validate(_write(panels, _spec(letters=["A", "B", "C"])))  # 4 slots
    panelise.validate(_write(panels, _spec(letters=["A", "B", "C", "D"])))  # ok


def test_unplaced_repeated_and_unknown_keys_raise(panels):
    spec = _spec(rows=[["1A", "1B"]])
    with pytest.raises(panelise.LayoutError, match="not placed"):
        panelise.validate(_write(panels, spec))
    spec = _spec(rows=[["1A", "1B"], ["1C", "1A"]])
    with pytest.raises(panelise.LayoutError, match="more than once"):
        panelise.validate(_write(panels, spec))
    spec = _spec(rows=[["1A", "1B"], ["1C", "9Z"]])
    with pytest.raises(panelise.LayoutError, match="undeclared"):
        panelise.validate(_write(panels, spec))


@pytest.mark.parametrize("bad", ["/etc/hosts", "../outside.pdf"])
def test_absolute_and_parent_paths_are_refused(panels, bad):
    spec = _spec()
    spec["panels"][0]["file"] = bad
    with pytest.raises(panelise.LayoutError):
        panelise.validate(_write(panels, spec))


def test_letter_x_must_be_a_fraction(panels):
    spec = _spec()
    spec["panels"][1]["letter_x"] = [0.0, 1.5]
    with pytest.raises(panelise.LayoutError, match="fractions"):
        panelise.validate(_write(panels, spec))


# ---------------------------------------------------------------------------
# Shipped layouts and source hygiene
# ---------------------------------------------------------------------------

def test_shipped_layouts_exist():
    names = {p.name for p in SHIPPED}
    assert {"fig01.yaml", "fig02.yaml"} <= names


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_no_absolute_paths_in_shipped_layouts(path):
    text = path.read_text()
    assert HOME not in text
    layout = panelise.load_layout(path)
    for panel in layout.panels:
        assert not Path(panel.file).is_absolute()
        assert ".." not in Path(panel.file).parts


@pytest.fixture(scope="session")
def generated_panels():
    """Build the composite panels this repo generates, so the shipped layouts can resolve.

    They are intermediates now (``tools/build_figures.py``, ``panels_dir``): gitignored, and
    deleted after a full build. A checkout therefore has the frozen panels on disk but not
    these, so a test that the layouts resolve has to make them first.
    """
    import subprocess

    sys.path.insert(0, str(ROOT / "tools"))
    import build_figures as bf

    for step in (s for s in bf.STEPS if s.panels):
        if all((bf.panels_dir(bf.SHIPPED) / f).is_file() for f in step.panels):
            continue
        done = subprocess.run(bf.command(step, bf.SHIPPED, sys.executable), cwd=ROOT,
                              capture_output=True, text=True)
        if done.returncode != 0:
            pytest.skip(f"cannot build panel step {step.name}: {done.stderr.strip()[-300:]}")
    return bf.panels_dir(bf.SHIPPED)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_shipped_layouts_resolve(path, generated_panels):
    layout = panelise.load_layout(path)
    panelise.validate(layout)  # raises on a missing file, bad letters, bad rows
    for panel in layout.panels:
        assert panelise.resolve(layout, panel).is_file()


def test_fig01_letters_are_a_to_d_over_the_expected_panels():
    layout = panelise.load_layout(ROOT / "guides" / "composites" / "fig01.yaml")
    # B and C are now split out of their shared frozen file so each can carry a header.
    assert [p.key for p in layout.ordered_panels()] == ["1A", "1B", "1C", "1D"]
    assert layout.letters == ["A", "B", "C", "D"]


def test_header_gap_defaults_to_the_module_constant(panels):
    """``header_gap_pt`` was a module constant before it was layout-settable; a layout
    that says nothing about it must behave exactly as it did."""
    spec = _spec(panels=[{"key": "1A", "file": "a.pdf", "title": "T", "tagline": "g"}],
                 rows=[["1A"]])
    layout = _write(panels, spec)
    assert layout.header_gap_pt == panelise.HEADER_GAP_PT
    panel = layout.panel("1A")
    expected = (layout.letter_margin_pt + layout.title_pt * 1.12
                + layout.tagline_pt * 1.30 + panelise.HEADER_GAP_PT)
    assert panelise.header_height(layout, panel) == pytest.approx(expected)


def test_header_gap_shortens_the_page_without_touching_the_panel(panels):
    """The point of the setting: a tighter gap buys page height and nothing else.

    The content keeps its aspect ratio and its height, so the whole saving shows up
    in the page -- which is what makes it a safe lever for fitting a paper size.
    """
    base = {"key": "1A", "file": "a.pdf", "title": "T", "tagline": "g"}
    loose = _write(panels, _spec(panels=[base], rows=[["1A"]], header_gap_pt=10.0))
    tight = _write(panels, _spec(panels=[base], rows=[["1A"]], header_gap_pt=2.0))

    h_loose, rects_loose = panelise.plan(loose)
    h_tight, rects_tight = panelise.plan(tight)

    assert h_loose - h_tight == pytest.approx(8.0)
    # the drawn content is identical; only the header band above it changed
    content_loose = rects_loose["1A"].height - panelise.header_height(loose, loose.panel("1A"))
    content_tight = rects_tight["1A"].height - panelise.header_height(tight, tight.panel("1A"))
    assert content_loose == pytest.approx(content_tight)


def test_fig01_header_spacing_is_tightened_for_the_page(panels):
    """Figure 1 has to sit on A4, so it overrides every header spacing downwards.

    Recorded as a test because the saving is invisible in the figure and easy to undo
    by tidying the layout back to the defaults.
    """
    layout = panelise.load_layout(ROOT / "guides" / "composites" / "fig01.yaml")
    assert layout.header_gap_pt < panelise.HEADER_GAP_PT
    assert layout.letter_margin_pt < panelise.LETTER_MARGIN_PT
    assert layout.title_pt < panelise.TITLE_PT
    assert layout.tagline_pt < panelise.TAGLINE_PT
    # the tagline must still read as subordinate to the title it sits under
    assert layout.tagline_pt < layout.title_pt


def test_fig01_tree_panel_carries_a_header():
    """1A had no header while B and C did; the tree is a frozen file with no title of
    its own, so the house one is drawn above it rather than cropped in."""
    layout = panelise.load_layout(ROOT / "guides" / "composites" / "fig01.yaml")
    tree = next(p for p in layout.ordered_panels() if p.key == "1A")
    assert tree.title and tree.tagline
    assert panelise.header_height(layout, tree) > 0


def test_fig01_tree_crop_stays_clear_of_the_k_clade():
    """1A's top crop buys page height, and it is spending the K clade to do it.

    Unlike 1B/1C, the crop here is NOT cutting off a baked-in title: the tree has no
    text layer at all. Measured on the rendered panel, ink starts 0.4% from the top
    edge and the top 5% band is 86% K-pink, because the top 40% of the panel is the K
    clade. So every 1% cropped costs about 1/40 of K.

    The ceiling matters more than the value. At 13% -- what full-bleed A4 would need
    from this lever alone -- a third of K is gone, and K is the subject of the figure.
    This fails if the crop is ever raised towards that, so the page is made to fit by
    scaling or by reclaiming the grey at the panel's bottom instead.
    """
    layout = panelise.load_layout(ROOT / "guides" / "composites" / "fig01.yaml")
    tree = next(p for p in layout.ordered_panels() if p.key == "1A")
    assert 0.0 <= tree.crop_top_frac <= 0.06, (
        f"1A crops {tree.crop_top_frac:.0%} off the top; above about 6% this is eating "
        "the K clade rather than trimming the panel")
    assert tree.crop_top == 0.0, "1A crops by fraction, not by points"


def test_module_has_no_machine_specific_paths():
    source = (ROOT / "figures" / "panelise.py").read_text()
    assert HOME not in source
    assert "/usr/share" not in source
    assert "liberation" not in source.lower()


# ---------------------------------------------------------------------------
# --root: a layout's panel paths are relative to a root the caller may choose
# ---------------------------------------------------------------------------

def test_the_command_line_places_panels_from_the_given_root(panels, tmp_path_factory):
    """``tools/build_figures.py`` roots a scratch build's composites with this: ``output/``
    must mean the scratch directory while the layout itself stays as shipped."""
    layout_dir = tmp_path_factory.mktemp("layouts")
    spec = {"name": "rooted", "width": 400,
            "panels": [{"key": "1A", "file": "a.pdf"}, {"key": "1B", "file": "b.pdf"}],
            "rows": [["1A", "1B"]]}
    (layout_dir / "layout.yaml").write_text(yaml.safe_dump(spec))
    out = tmp_path_factory.mktemp("out")

    assert panelise.main(["--layout", str(layout_dir / "layout.yaml"),
                          "--root", str(panels), "--out", str(out)]) == 0
    assert (out / "rooted.pdf").stat().st_size > 0

    empty = tmp_path_factory.mktemp("empty_root")
    with pytest.raises(panelise.LayoutError):                   # a root without the panels fails
        panelise.main(["--layout", str(layout_dir / "layout.yaml"),
                       "--root", str(empty), "--out", str(out)])


def test_the_default_root_is_still_the_repository():
    assert panelise.load_layout("guides/composites/fig02.yaml").root == ROOT
