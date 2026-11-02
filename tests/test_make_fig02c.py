"""Figure 2 panel C, redrawn from digitised values, and the digitiser that makes them."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import make_fig02c as c
import make_fig02d as d
from mutacc import style
from mutacc.guide import REPO_ROOT
import sys

sys.path.insert(0, str(REPO_ROOT / "tools"))
import digitise_epoch_curve as dg  # noqa: E402

RASTER = REPO_ROOT / "data" / "frozen_panels" / "fig2c_plm_epoch.png"


@pytest.fixture(scope="module")
def shipped():
    return c.load_values()


# --- the values --------------------------------------------------------------------------

def test_the_shipped_values_are_raw_plus_epochs_in_order(shipped):
    n = len(shipped)
    assert shipped["epoch"].tolist() == list(range(n))
    assert shipped["label"].tolist() == ["raw"] + [str(e) for e in range(1, n)]
    assert shipped["source"].str.contains("ESM2|esm2|facebook", case=False).all()


def test_golden_shape_of_the_curve(shipped):
    r = shipped["spearman_r"]
    assert r.iloc[0] == pytest.approx(0.275, abs=0.005)          # the raw model
    assert r.iloc[1] == pytest.approx(0.622, abs=0.04)           # the first epoch does the work
    assert r.iloc[2:].between(0.62, 0.67).all()                  # then a plateau
    assert r.iloc[1] - r.iloc[0] > 0.3


def test_the_raw_and_final_values_agree_with_the_one_lineage_panels(shipped):
    """Both evaluated on J.2_int, agreeing with panels A and B within stochastic run variation."""
    values = pd.read_csv(REPO_ROOT / "output" / "fig02ab_scatter_panels_values.tsv", sep="\t")
    rho = values[values["measure"] == "spearman_r"].set_index("panel")["value"]
    assert abs(shipped["spearman_r"].iloc[0] - rho["A"]) < 0.005
    assert abs(shipped["spearman_r"].iloc[-1] - rho["B"]) < 0.02


def test_the_digitiser_recovers_known_values_from_legacy_raster():
    fresh = dg.digitise(RASTER)
    assert len(fresh) == 21
    assert fresh["spearman_r"].iloc[0] == pytest.approx(0.274, abs=0.003)
    assert fresh["spearman_r"].iloc[-1] == pytest.approx(0.651, abs=0.003)


@pytest.mark.parametrize("edit, message", [
    (lambda f: f.drop(columns="source"), "missing columns"),
    (lambda f: f.assign(epoch=f["epoch"] + 1), "in order"),
    (lambda f: f.iloc[::-1].reset_index(drop=True), "in order"),
    (lambda f: f.assign(spearman_r=f["spearman_r"].where(f["epoch"] != 3, 1.5)), "within"),
    (lambda f: f.assign(spearman_r=f["spearman_r"].where(f["epoch"] != 3, np.nan)), "finite"),
])
def test_load_values_refuses_bad_tables(tmp_path, shipped, edit, message):
    bad = edit(shipped.copy())
    path = tmp_path / "v.tsv"
    bad.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match=message):
        c.load_values(path)


# --- the drawing -------------------------------------------------------------------------

def test_the_panel_is_drawn_in_the_shared_box_with_the_shared_text(shipped):
    import matplotlib.pyplot as plt

    fig = c.draw(shipped)
    try:
        (ax,) = fig.axes
        assert tuple(fig.get_size_inches()) == pytest.approx(style.PANEL_SIZE)
        assert tuple(ax.get_position().bounds) == pytest.approx(style.PANEL_AXES)
        assert ax.get_ylabel() == d.YLABEL == "Spearman ρ with accessibility"
        assert ax.get_xlabel() == "Epoch"
        assert ax.yaxis.label.get_size() == ax.xaxis.label.get_size() == style.PANEL_LABEL_PT
        assert ax.yaxis.get_label().get_position()[0] == style.PANEL_YLABEL_X
        assert ax.get_xticklabels()[0].get_size() == style.PANEL_TICK_PT
    finally:
        plt.close(fig)


def test_every_value_is_plotted_and_the_ticks_are_raw_and_every_fifth_epoch(shipped):
    import matplotlib.pyplot as plt

    fig = c.draw(shipped)
    try:
        (ax,) = fig.axes
        (line,) = ax.lines
        n = len(shipped)
        assert np.allclose(line.get_ydata(), shipped["spearman_r"]) and len(line.get_xdata()) == n
        expected_ticks = c.tick_epochs(int(shipped["epoch"].max()))
        assert [t.get_text() for t in ax.get_xticklabels()] == ["raw" if e == 0 else str(e) for e in expected_ticks]
        assert list(ax.get_xticks()) == expected_ticks
        lo, hi = ax.get_ylim()
        assert lo < shipped["spearman_r"].min() and hi > shipped["spearman_r"].max()
    finally:
        plt.close(fig)


def test_the_curve_keeps_the_original_blue():
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt

    fig = c.draw(c.load_values())
    try:
        assert to_hex(fig.axes[0].lines[0].get_color()) == "#1f77b4"
    finally:
        plt.close(fig)


def test_tick_epochs():
    assert c.tick_epochs(20) == [0, 5, 10, 15, 20]
    assert c.tick_epochs(20, every=10) == [0, 10, 20]
    assert c.tick_epochs(7) == [0, 5]
    with pytest.raises(ValueError):
        c.tick_epochs(20, every=0)


def test_the_composite_places_the_generated_panel_and_its_subheader_names_no_lineage():
    import yaml
    panels = {p["key"]: p for p in yaml.safe_load((REPO_ROOT / "guides/composites/fig02.yaml").read_text())["panels"]}
    assert panels["2C"]["file"] == "build/panels/fig02c_epoch_panel.pdf"
    assert "crop_top_frac" not in panels["2C"]
    assert panels["2C"]["tagline"] == "base model, then each epoch of fine-tuning"


# --- the caption -------------------------------------------------------------------------

def test_the_caption_reports_its_own_numbers_and_its_provenance(shipped):
    text = " ".join(c.caption(shipped).split())
    r = shipped["spearman_r"]
    assert f"{r.iloc[0]:.3f}" in text and f"{r.iloc[1]:.3f}" in text
    assert f"{r.iloc[2:].min():.3f}" in text and f"{r.iloc[2:].max():.3f}" in text
    assert "fine-tuning run" in text and "J.2_int" in text


# --- the digitiser -----------------------------------------------------------------------

#: The raster is a plain matplotlib default plot: a full black frame, black ticks, no grid. The house
#: style (open top and right) would not be that anatomy, so the synthetic panels are drawn plainly.
PLAIN = {"axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "black", "axes.linewidth": 0.8,
         "xtick.color": "black", "ytick.color": "black", "axes.grid": False, "figure.constrained_layout.use": False,
         "savefig.bbox": "standard", "font.size": 10}


def synthetic_raster(tmp_path, values):
    """A panel with the raster's own anatomy: 22 x ticks (mut, raw, 1..20), 8 y ticks, blue discs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(PLAIN):
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=300)
        ax.set_xticks(range(22))
        ax.set_yticks(dg.Y_TICK_VALUES)
        ax.set_ylim(0.25, 0.68)
        ax.set_xlim(-0.5, 21.7)
        ax.plot(range(1, 22), values, color="#1f77b4", marker="o", markersize=7, linewidth=1.5)
        path = tmp_path / "synthetic.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return path


def test_the_digitiser_recovers_known_values_from_a_synthetic_panel(tmp_path):
    rng = np.random.default_rng(3)
    truth = np.round(rng.uniform(0.3, 0.65, 21), 4)
    recovered = dg.digitise(synthetic_raster(tmp_path, truth))
    assert recovered["spearman_r"].to_numpy() == pytest.approx(truth, abs=0.002)
    assert recovered["epoch"].tolist() == list(range(21))


def test_the_y_calibration_runs_the_right_way_up(tmp_path):
    """The top tick is the largest value: a rising curve must digitise as rising."""
    truth = np.linspace(0.3, 0.6, 21)
    recovered = dg.digitise(synthetic_raster(tmp_path, truth))["spearman_r"].to_numpy()
    assert np.all(np.diff(recovered) > 0)


def test_the_digitiser_refuses_a_picture_that_is_not_the_expected_panel(tmp_path):
    Image.new("RGB", (400, 300), "white").save(tmp_path / "blank.png")
    with pytest.raises(ValueError, match="plot frame"):
        dg.digitise(tmp_path / "blank.png")


def test_the_digitiser_refuses_a_curve_with_a_missing_marker(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(PLAIN):
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=300)
        ax.set_xticks(range(22))
        ax.set_yticks(dg.Y_TICK_VALUES)
        ax.set_ylim(0.25, 0.68)
        ax.set_xlim(-0.5, 21.7)
        ax.plot(range(1, 21), np.linspace(0.3, 0.6, 20), color="#1f77b4", marker="o", markersize=7)   # one short
        fig.savefig(tmp_path / "short.png", dpi=300)
        plt.close(fig)
    with pytest.raises(ValueError, match="markers"):
        dg.digitise(tmp_path / "short.png")
