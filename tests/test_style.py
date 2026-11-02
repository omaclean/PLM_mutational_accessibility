"""Tests for the house style module.

These pin the two defects that produced wrong figures in the predecessor repo:

1. A seaborn fallback that silently recoloured a figure when the import failed.
   Guarded by :func:`test_mutacc_does_not_import_seaborn` and
   :func:`test_style_imports_with_seaborn_blocked`.
2. Colour constants drifting apart across scripts -- four competing base/fine-tune
   schemes existed at once, two of them mapping the hues to opposite roles.
   Guarded by :func:`test_palette_values_are_pinned`.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mutacc import style  # noqa: E402


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

SERIES_COLOURS = {
    "BASE": style.BASE,
    "FINETUNE": style.FINETUNE,
    "ACCESS": style.ACCESS,
    "BIOCHEM": style.BIOCHEM,
}


def _relative_luminance(hex_colour: str) -> float:
    """WCAG 2.x relative luminance."""
    raw = hex_colour.lstrip("#")
    channels = [int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(a: str, b: str = "#ffffff") -> float:
    lighter, darker = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _lab(hex_colour: str) -> tuple[float, float, float]:
    """CIE L*a*b* under D65, for perceptual distance between two colours."""
    raw = hex_colour.lstrip("#")
    lin = []
    for i in (0, 2, 4):
        c = int(raw[i:i + 2], 16) / 255
        lin.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin
    xyz = (0.4124 * r + 0.3576 * g + 0.1805 * b,
           0.2126 * r + 0.7152 * g + 0.0722 * b,
           0.0193 * r + 0.1192 * g + 0.9505 * b)
    white = (0.95047, 1.0, 1.08883)

    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29

    fx, fy, fz = (f(v / w) for v, w in zip(xyz, white))
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e(a: str, b: str) -> float:
    return sum((x - y) ** 2 for x, y in zip(_lab(a), _lab(b))) ** 0.5


# The two-palette rule: the model pair is reserved, every other category is
# drawn from CATEGORICAL, and nothing in CATEGORICAL may read as the model pair.
MODEL_PAIR = {"BASE": style.BASE, "FINETUNE": style.FINETUNE}
MIN_DELTA_E = 42.0
MIN_GREYSCALE_GAP = 0.045


def test_categorical_palette_is_pinned():
    assert style.CATEGORICAL == ("#d55e00", "#cc79a7", "#6a3d9a")


@pytest.mark.parametrize("colour", style.CATEGORICAL)
def test_categorical_clears_the_contrast_floor(colour):
    assert contrast_ratio(colour) >= 3.0, (
        f"{colour} is {contrast_ratio(colour):.2f}:1 against white, below the 3:1 floor"
    )


@pytest.mark.parametrize("colour", style.CATEGORICAL)
def test_categorical_cannot_be_read_as_the_model_pair(colour):
    """A category colour must not look like "base" or "fine-tuned"."""
    for name, reserved in MODEL_PAIR.items():
        d = delta_e(colour, reserved)
        assert d >= MIN_DELTA_E, (
            f"{colour} is only dE {d:.1f} from {name} ({reserved}); "
            "readers would take it for a model colour"
        )


def test_categorical_colours_are_mutually_separable():
    palette = style.CATEGORICAL
    for i, a in enumerate(palette):
        for b in palette[i + 1:]:
            assert delta_e(a, b) >= 35.0, f"{a} and {b} are only dE {delta_e(a, b):.1f} apart"


def test_categorical_survives_greyscale_printing():
    lums = sorted(_relative_luminance(c) for c in style.CATEGORICAL)
    gaps = [b - a for a, b in zip(lums, lums[1:])]
    assert min(gaps) >= MIN_GREYSCALE_GAP, (
        f"two category colours differ by only {min(gaps):.3f} in luminance; "
        "they would merge in a mono print"
    )


def test_dark2_would_have_collided_with_finetune():
    """Guards the decision recorded in style.py: Dark2 is not usable as-is."""
    assert delta_e("#1b9e77", style.FINETUNE) < 10.0


def test_colours_for_categories_is_order_independent():
    """Dropping a category must not repaint the ones that remain."""
    full = style.colours_for_categories(["cell entry", "ACE2 binding", "stability"])
    reordered = style.colours_for_categories(["stability", "cell entry", "ACE2 binding"])
    assert full == reordered
    subset = style.colours_for_categories(["ACE2 binding", "stability"])
    assert subset["ACE2 binding"] == full["ACE2 binding"]


def test_colours_for_categories_refuses_to_cycle():
    too_many = [f"cat{i}" for i in range(len(style.CATEGORICAL) + 1)]
    with pytest.raises(ValueError, match="categorical palette"):
        style.colours_for_categories(too_many)


def test_palette_values_are_pinned():
    """The Okabe-Ito values, spelled out.

    If someone changes a colour, this test names the change rather than letting
    it propagate silently into half the figure set.
    """
    assert style.BASE == "#0072b2"
    assert style.FINETUNE == "#009e73"
    assert style.ACCESS == "#d55e00"
    assert style.BIOCHEM == "#cc79a7"
    assert style.NULL_GREY == "#999999"


@pytest.mark.parametrize("name,colour", sorted(SERIES_COLOURS.items()))
def test_series_colours_clear_the_contrast_floor(name, colour):
    """Every series colour reaches 3:1 against the page.

    The previously published pair did not: #35b779 sat at 2.49:1, and the repo
    compensated by giving the two colours different alphas.
    """
    assert contrast_ratio(colour) >= 3.0, (
        f"{name} ({colour}) is {contrast_ratio(colour):.2f}:1 against white, "
        "below the 3:1 floor"
    )


def test_series_colours_are_distinct():
    assert len(set(SERIES_COLOURS.values())) == len(SERIES_COLOURS)


def test_base_and_finetune_are_separable_in_greyscale():
    """Print and photocopying collapse hue; lightness must still separate them."""
    gap = abs(_relative_luminance(style.BASE) - _relative_luminance(style.FINETUNE))
    assert gap > 0.05, f"base and fine-tune differ by only {gap:.3f} in luminance"


def test_colour_for_role_rejects_unknown_roles():
    """A typo must raise, not silently return a default colour."""
    assert style.colour_for_role("base") == style.BASE
    with pytest.raises(KeyError, match="unknown plot role"):
        style.colour_for_role("baseline")


def test_every_role_colour_is_a_hex_triplet():
    for role, colour in style.ROLE_COLOURS.items():
        assert colour.startswith("#") and len(colour) == 7, f"{role}: {colour}"


# ---------------------------------------------------------------------------
# The seaborn regression
# ---------------------------------------------------------------------------

def test_mutacc_does_not_import_seaborn():
    """No module under mutacc/ may mention seaborn.

    The predecessor pattern was a bare try/except around
    sns.color_palette("viridis", 2), whose hand-written fallback used the raw
    viridis endpoints (purple/yellow) instead of the trimmed blue/green the call
    actually returns. Depending on nothing removes the failure mode.
    """
    package_root = Path(style.__file__).parent
    # The module docstring discusses the bug by name, so match import statements
    # rather than the bare word.
    offenders = [
        f"{path.name}:{lineno}: {line.strip()}"
        for path in package_root.glob("*.py")
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip().startswith(("import seaborn", "from seaborn"))
    ]
    assert not offenders, f"seaborn imported in: {offenders}"


def test_style_imports_with_seaborn_blocked(monkeypatch):
    """The module imports and keeps its palette when seaborn is unavailable."""
    real_import = builtins.__import__

    def deny_seaborn(name, *args, **kwargs):
        if name == "seaborn" or name.startswith("seaborn."):
            raise ImportError("seaborn blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_seaborn)
    monkeypatch.delitem(sys.modules, "mutacc.style", raising=False)

    reloaded = importlib.import_module("mutacc.style")
    assert reloaded.BASE == "#0072b2"
    assert reloaded.FINETUNE == "#009e73"


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def test_reference_line_requires_a_label():
    """An unlabelled rule asks the reader to guess what it means."""
    fig = style.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError, match="requires a label"):
        style.reference_line(ax, 0.5, "")
    style.save_figure(fig, _tmp_path_for("unused"), close=True)


def test_reference_line_is_neutral_grey_by_default():
    """Status colours such as red are reserved; a chance line is not a warning."""
    fig = style.figure()
    ax = fig.add_subplot(111)
    style.reference_line(ax, 0.5, "chance = 0.5")
    drawn = [ln for ln in ax.get_lines()]
    assert drawn, "no line was drawn"
    assert drawn[-1].get_color() == style.NULL_GREY
    style.save_figure(fig, _tmp_path_for("refline"), close=True)


def test_reference_line_writes_its_label_onto_the_axes():
    fig = style.figure()
    ax = fig.add_subplot(111)
    style.reference_line(ax, 0.5, "chance = 0.5")
    texts = [t.get_text() for t in ax.texts]
    assert "chance = 0.5" in texts
    style.save_figure(fig, _tmp_path_for("reflabel"), close=True)


@pytest.mark.parametrize("loc,expected_ha", [("right", "right"), ("left", "left")])
def test_reference_line_label_can_move_ends(loc, expected_ha):
    """A data point can sit where the label lands; label_loc moves it.

    Added after a real collision in S14 panel A, where the base model's point at
    the right-hand x position sat directly under "chance = 0.5".
    """
    fig = style.figure()
    ax = fig.add_subplot(111)
    style.reference_line(ax, 0.5, "chance = 0.5", label_loc=loc)
    ann = [t for t in ax.texts if t.get_text() == "chance = 0.5"]
    assert len(ann) == 1
    assert ann[0].get_ha() == expected_ha
    style.save_figure(fig, _tmp_path_for("loc"), close=True)


def test_reference_line_rejects_a_bad_label_loc():
    fig = style.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError, match="label_loc must be"):
        style.reference_line(ax, 0.5, "chance = 0.5", label_loc="middle")
    style.save_figure(fig, _tmp_path_for("badloc"), close=True)


@pytest.mark.parametrize("side,expected_va", [("above", "bottom"), ("below", "top")])
def test_reference_line_label_can_cross_the_rule(side, expected_va):
    """Both ends of the band on one side can be occupied; label_side moves it across.

    Added after a real collision in S4 panel A, where two point labels sat just above
    zero at opposite ends, leaving nowhere on that side for the rule's own label.
    """
    fig = style.figure()
    ax = fig.add_subplot(111)
    style.reference_line(ax, 0.0, "no relationship = 0", label_side=side)
    ann = [t for t in ax.texts if t.get_text() == "no relationship = 0"]
    assert len(ann) == 1
    assert ann[0].get_va() == expected_va
    style.save_figure(fig, _tmp_path_for("side"), close=True)


def test_reference_line_rejects_a_bad_label_side():
    fig = style.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError, match="label_side must be"):
        style.reference_line(ax, 0.5, "chance = 0.5", label_side="beside")
    style.save_figure(fig, _tmp_path_for("badside"), close=True)


def test_panel_letters_are_sequential():
    fig = style.figure(width=style.WIDTH_DOUBLE)
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    style.panel_letters(axes)
    assert [ax.texts[0].get_text() for ax in axes] == ["A", "B", "C"]
    style.save_figure(fig, _tmp_path_for("panels"), close=True)


def test_panel_letter_pad_points_sits_on_the_title_baseline():
    """``pad_points`` exists so a letter lines up with its heading instead of floating.

    Figure 4's letters were left at the default ``dy``, which put them well above the
    one-line titles. Passing the title's own pad to both is what lines them up, so the
    two must land on the same baseline when given the same value.
    """
    import matplotlib.pyplot as plt

    pad = 3.0
    fig = style.figure(style.WIDTH_DOUBLE, 2.5)
    try:
        ax = fig.add_subplot(111)
        ax.set_title("A heading", loc="left", fontsize=7.5, pad=pad)
        style.panel_letter(ax, "A", pad_points=pad)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        letter = next(t for t in ax.texts if t.get_text() == "A")
        # Compare baselines, not boxes: the two differ in size, so their extents differ.
        letter_y = letter.get_window_extent(renderer).y0
        title_y = ax._left_title.get_window_extent(renderer).y0
        assert abs(letter_y - title_y) < 3.0, (letter_y, title_y)
    finally:
        plt.close(fig)


def test_figure_four_letters_and_titles_share_one_pad():
    """Two separate literals would let the letter and its heading drift apart."""
    source = (Path(__file__).resolve().parents[1] / "figures" / "make_fig04.py").read_text()
    assert "pad=PANEL_HEADING_PAD_PT" in source
    assert "pad_points=PANEL_HEADING_PAD_PT" in source


def test_panel_letters_are_bold():
    """Captions reference panels by letter; they must be findable."""
    fig = style.figure()
    ax = fig.add_subplot(111)
    style.panel_letter(ax, "A")
    assert ax.texts[0].get_fontweight() == "bold"
    style.save_figure(fig, _tmp_path_for("bold"), close=True)


@pytest.mark.parametrize("bad_label", ["plm_prob", "mut_acc", "site.logistic_auroc"])
def test_finalise_axes_rejects_column_names(bad_label):
    """Axis labels are for readers, not for pandas."""
    fig = style.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError, match="looks like a column name"):
        style.finalise_axes(ax, ylabel=bad_label)
    style.save_figure(fig, _tmp_path_for("badlabel"), close=True)


@pytest.mark.parametrize("good_label", [
    "Model probability of this replacement",
    "Weight on mutational accessibility (alpha)",
    "AUROC",
])
def test_finalise_axes_accepts_plain_language(good_label):
    fig = style.figure()
    ax = fig.add_subplot(111)
    style.finalise_axes(ax, ylabel=good_label)
    assert ax.get_ylabel() == good_label
    style.save_figure(fig, _tmp_path_for("goodlabel"), close=True)


# ---------------------------------------------------------------------------
# rcParams and output
# ---------------------------------------------------------------------------

def test_house_style_is_idempotent():
    style.use_house_style()
    first = dict(style.plt.rcParams)
    style.use_house_style()
    assert dict(style.plt.rcParams) == first


def test_house_style_hides_top_and_right_spines():
    style.use_house_style()
    assert style.plt.rcParams["axes.spines.top"] is False
    assert style.plt.rcParams["axes.spines.right"] is False


def test_house_style_sets_one_savefig_dpi():
    style.use_house_style()
    assert style.plt.rcParams["savefig.dpi"] == style.DPI


def test_column_widths_match_mbe():
    assert style.WIDTH_SINGLE == pytest.approx(3.46, abs=0.01)
    assert style.WIDTH_DOUBLE == pytest.approx(7.09, abs=0.01)


def test_save_figure_writes_both_vector_and_raster(tmp_path):
    """Vector for submission; raster for review. Neither may drift from the other."""
    fig = style.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1], color=style.BASE)
    written = style.save_figure(fig, tmp_path / "demo")
    assert {p.suffix for p in written} == {".pdf", ".png"}
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_save_figure_creates_missing_directories(tmp_path):
    fig = style.figure()
    fig.add_subplot(111)
    written = style.save_figure(fig, tmp_path / "nested" / "deeper" / "demo")
    assert all(p.exists() for p in written)


# ---------------------------------------------------------------------------
# The swept-weight axis label and the legend that sits above the panels
# ---------------------------------------------------------------------------

def test_alpha_axis_label_bolds_only_the_symbol():
    """The alpha in the brackets is bold; the words around it are not."""
    label = style.ALPHA_SWEEP_XLABEL
    assert r"($\mathbf{\alpha}$)" in label
    assert label.startswith("Weight on mutational accessibility")
    # One mathtext span, so nothing else in the sentence is set bold by accident.
    assert label.count("$") == 2
    # Upright, not the slanted \boldsymbol form, so it matches the alphas on the panels.
    assert "boldsymbol" not in label


def test_alpha_axis_label_renders_as_mathtext():
    """A malformed mathtext span raises at draw time, not at import, so draw it."""
    from matplotlib.textpath import TextToPath  # noqa: F401  (mathtext parser is exercised below)

    fig = style.figure()
    ax = fig.add_subplot(111)
    text = fig.supxlabel(style.ALPHA_SWEEP_XLABEL)
    fig.canvas.draw()
    assert text.get_window_extent(fig.canvas.get_renderer()).width > 0
    del ax
    style.save_figure(fig, _tmp_path_for("alphalabel"), close=True)


def test_figures_three_and_four_share_one_alpha_label():
    """Both sweeps label the same quantity; a second copy is a chance to drift."""
    root = Path(__file__).resolve().parents[1]
    for name in ("make_fig03.py", "make_fig04.py"):
        source = (root / "figures" / name).read_text()
        assert "style.ALPHA_SWEEP_XLABEL" in source, name
        assert "Weight on mutational accessibility" not in source, (
            f"{name} spells the label out instead of using style.ALPHA_SWEEP_XLABEL"
        )


def test_legend_above_tightens_the_band_under_the_legend():
    """The gap between the key and the panel headings is pad we chose, not the default."""
    fig = style.figure(style.WIDTH_DOUBLE, 2.5)
    axes = fig.subplots(1, 2)
    for ax in axes:
        ax.plot([0, 1], [0, 1], color=style.BASE, label="Base model")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = style.legend_above(fig, handles, labels, ncol=1)
    assert legend in fig.legends
    assert fig.get_layout_engine().get()["h_pad"] == pytest.approx(style.LEGEND_BAND_PAD)
    assert style.LEGEND_BAND_PAD < 0.04167, "the default pad is what we are closing"
    assert style.LEGEND_BAND_PAD >= 0.016, "below this the legend touches a two-line tagline"
    style.save_figure(fig, _tmp_path_for("legendabove"), close=True)


def test_legend_above_leaves_less_white_than_the_plain_outside_legend():
    """Measured on the drawn canvas: the heading sits closer to the key than before."""
    def gap(tighten: bool) -> float:
        fig = style.figure(style.WIDTH_DOUBLE, 2.5)
        ax = fig.subplots(1, 1)
        ax.plot([0, 1], [0, 1], color=style.BASE, label="Base model")
        style.panel_title(ax, "Spearman correlation", "a two-line\ntagline")
        handles, labels = ax.get_legend_handles_labels()
        if tighten:
            legend = style.legend_above(fig, handles, labels)
        else:
            legend = fig.legend(handles, labels, loc="outside upper center")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        below_legend = legend.get_window_extent(renderer).y0
        top_of_heading = max(a.get_tightbbox(renderer).y1 for a in fig.axes)
        style.plt.close(fig)
        return below_legend - top_of_heading

    tightened, default = gap(True), gap(False)
    assert tightened >= 0, "the legend must not sit on top of the panel heading"
    assert tightened < default


def test_legend_above_refuses_a_figure_without_the_constrained_layout():
    """It works by setting that engine's pad, so it must say so when the engine is gone."""
    fig = style.figure()
    fig.set_layout_engine("none")
    fig.add_subplot(111).plot([0, 1], [0, 1], label="Base model")
    with pytest.raises(ValueError, match="constrained layout"):
        style.legend_above(fig, *fig.axes[0].get_legend_handles_labels())
    assert not fig.legends, "a refused call must not leave a legend behind"
    style.plt.close(fig)


def test_alpha_zero_rule_is_heavier_than_an_ordinary_reference_line():
    """It marks the value the whole figure is read against, so it outweighs a hairline."""
    assert style.ALPHA_ZERO_RULE["linewidth"] > 0.8
    assert style.ALPHA_ZERO_RULE["linestyle"] == (0, (5, 2))


def test_alpha_zero_rule_cannot_be_mutated_by_a_caller():
    """Both sweep figures unpack it; one script editing it in place would move the other."""
    with pytest.raises(TypeError):
        style.ALPHA_ZERO_RULE["linewidth"] = 99


def test_both_sweep_figures_draw_the_alpha_zero_rule_from_the_shared_spec():
    root = Path(__file__).resolve().parents[1]
    for name in ("make_fig03.py", "make_fig04.py"):
        source = (root / "figures" / name).read_text()
        assert "style.ALPHA_ZERO_RULE" in source, name
        assert "(0, (5, 2))" not in source, f"{name} re-spells the dash pattern"


def test_diverging_map_is_not_a_rainbow():
    """Polarity needs two hues and a neutral midpoint, never a rainbow."""
    assert style.DIVERGING not in {"jet", "rainbow", "hsv", "turbo", "nipy_spectral"}
    assert style.SEQUENTIAL not in {"jet", "rainbow", "hsv", "turbo"}


# ---------------------------------------------------------------------------

def _tmp_path_for(name: str) -> Path:
    import tempfile
    return Path(tempfile.mkdtemp(prefix="mutacc_style_")) / name
