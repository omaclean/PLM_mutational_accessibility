"""mutacc.style helpers for composite panels: one shared box, uncropped saves, pinned titles."""

import numpy as np
import pytest
from PIL import Image

from mutacc import style


def test_the_panel_box_is_a_valid_figure_fraction_with_room_for_the_labels():
    left, bottom, width, height = style.PANEL_AXES
    assert 0 < left < 0.4 and 0 < bottom < 0.4
    assert left + width <= 1.0 and bottom + height <= 1.0
    assert width > 0.7 and height > 0.7, "the plot fills the page: the margins are the labels, nothing else"


def test_panel_axes_returns_one_axes_in_that_box_and_stops_layout_moving_it():
    import matplotlib.pyplot as plt

    fig = style.figure(*style.PANEL_SIZE)
    try:
        ax = style.panel_axes(fig)
        assert fig.axes == [ax]
        assert tuple(ax.get_position().bounds) == pytest.approx(style.PANEL_AXES)
        ax.set_xlabel("a long label " * 3)
        fig.canvas.draw()
        assert tuple(ax.get_position().bounds) == pytest.approx(style.PANEL_AXES)
    finally:
        plt.close(fig)


def test_place_panel_labels_pins_position_and_size():
    import matplotlib.pyplot as plt

    fig = style.figure(*style.PANEL_SIZE)
    try:
        ax = style.panel_axes(fig)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        style.place_panel_labels(ax)
        assert ax.yaxis.label.get_position()[0] == style.PANEL_YLABEL_X
        assert ax.xaxis.label.get_position()[1] == style.PANEL_XLABEL_Y
        assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == style.PANEL_LABEL_PT
    finally:
        plt.close(fig)


#: The three kinds of x tick label the Figure 2 panels actually draw. Log ticks (A and B)
#: are mathtext and sit lowest, so they are the ones the x-axis title can collide with.
_TICK_KINDS = {
    "log (panels A, B)": dict(scale="log", xlim=(1e-9, 1e1), label="PLM probability"),
    "numeric (panel C)": dict(scale="linear", xlim=(0, 20), label="Epoch"),
    "categorical (panel D)": dict(scale="category", xlim=(-0.55, 3.55),
                                  label="Nucleotide changes needed"),
}


def _panel_with_ticks(kind):
    spec = _TICK_KINDS[kind]
    fig = style.figure(*style.PANEL_SIZE)
    ax = style.panel_axes(fig)
    if spec["scale"] == "log":
        ax.set_xscale("log")
    elif spec["scale"] == "category":
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["all", "1", "2", "3"])
    ax.set_xlim(*spec["xlim"])
    ax.tick_params(labelsize=style.PANEL_TICK_PT)
    ax.set_xlabel(spec["label"])
    style.place_panel_labels(ax)
    fig.canvas.draw()
    return fig, ax


@pytest.mark.parametrize("kind", list(_TICK_KINDS))
def test_x_axis_title_clears_the_tick_labels(kind):
    """PANEL_XLABEL_Y is a fixed offset, so tightening it can run the title into the ticks.

    It did: at -0.071 panel D's "Nucleotide changes needed" was drawn straight through
    "all 1 2 3". The offset is shared by every panel, so the worst case -- the mathtext log
    ticks, which reach lowest -- is what it has to clear.
    """
    import matplotlib.pyplot as plt

    fig, ax = _panel_with_ticks(kind)
    try:
        renderer = fig.canvas.get_renderer()
        ticks_bottom = min(t.get_window_extent(renderer).y0
                           for t in ax.get_xticklabels() if t.get_text())
        title_top = ax.xaxis.label.get_window_extent(renderer).y1
        assert title_top < ticks_bottom, f"{kind}: the x-axis title overlaps its tick labels"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", list(_TICK_KINDS))
def test_x_axis_title_stays_inside_the_panel(kind):
    """Nothing is cropped on save, so a title pushed past the edge is simply lost."""
    import matplotlib.pyplot as plt

    fig, ax = _panel_with_ticks(kind)
    try:
        renderer = fig.canvas.get_renderer()
        assert ax.xaxis.label.get_window_extent(renderer).y0 > 0, (
            f"{kind}: the x-axis title runs off the bottom of the panel"
        )
    finally:
        plt.close(fig)


def test_panel_tick_furniture_is_tighter_than_the_house_default():
    """The panel is scaled down in the composite, so house-sized ticks eat visible space."""
    assert style.PANEL_TICK_LEN_PT < 3.5
    assert style.PANEL_TICK_PAD_PT < 3.5


def test_place_panel_labels_applies_the_tick_furniture():
    import matplotlib.pyplot as plt

    fig = style.figure(*style.PANEL_SIZE)
    try:
        ax = style.panel_axes(fig)
        style.place_panel_labels(ax)
        tick = ax.xaxis.get_major_ticks()[0]
        assert tick.get_pad() == pytest.approx(style.PANEL_TICK_PAD_PT)
        assert tick.tick1line.get_markersize() == pytest.approx(style.PANEL_TICK_LEN_PT)
    finally:
        plt.close(fig)


def test_title_positions_do_not_depend_on_tick_label_width():
    """Wide tick labels (10^-14) and narrow ones (0.65) leave the titles in the same place."""
    import matplotlib.pyplot as plt

    positions = []
    for labels in (["0.1", "0.2"], ["10^{-14}", "10^{-16}"]):
        fig = style.figure(*style.PANEL_SIZE)
        try:
            ax = style.panel_axes(fig)
            ax.set_yticks([0.25, 0.75])
            ax.set_yticklabels([f"${l}$" if "^" in l else l for l in labels])
            ax.set_ylabel("y")
            ax.set_xlabel("x")
            style.place_panel_labels(ax)
            fig.canvas.draw()
            positions.append((ax.yaxis.label.get_position(), ax.xaxis.label.get_position()))
        finally:
            plt.close(fig)
    assert positions[0] == positions[1]


def test_save_panel_keeps_the_whole_canvas_whatever_is_drawn(tmp_path):
    import matplotlib.pyplot as plt

    sizes = []
    for text in ("short", "a very long annotation that would widen a cropped save " * 2):
        fig = style.figure(*style.PANEL_SIZE)
        ax = style.panel_axes(fig)
        ax.text(0.5, 0.5, text, transform=ax.transAxes)
        written = style.save_panel(fig, tmp_path / f"p{len(sizes)}", suffixes=(".png",))
        sizes.append(Image.open(written[0]).size)
    expected = tuple(int(round(v * style.DPI)) for v in style.PANEL_SIZE)
    assert sizes[0] == sizes[1] == expected


def test_save_figure_still_crops_so_the_difference_is_real(tmp_path):
    import matplotlib.pyplot as plt

    fig = style.figure(*style.PANEL_SIZE)
    style.panel_axes(fig)
    (png,) = style.save_figure(fig, tmp_path / "cropped", suffixes=(".png",))
    assert Image.open(png).size != tuple(int(round(v * style.DPI)) for v in style.PANEL_SIZE)


def test_save_panel_writes_pdf_and_png_and_refuses_other_formats(tmp_path):
    fig = style.figure(*style.PANEL_SIZE)
    style.panel_axes(fig)
    written = style.save_panel(fig, tmp_path / "ok")
    assert sorted(p.suffix for p in written) == [".pdf", ".png"] and all(p.is_file() for p in written)
    fig = style.figure(*style.PANEL_SIZE)
    with pytest.raises(ValueError, match="writes .pdf and .png"):
        style.save_panel(fig, tmp_path / "bad", suffixes=(".svg",))
