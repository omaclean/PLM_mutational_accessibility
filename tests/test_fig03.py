"""Figure 3 numbers are checked against an independent calculation, not the script's own."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

import make_fig03 as fig3
from mutacc import style
from mutacc.guide import REPO_ROOT, load_guide

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")


@pytest.fixture(scope="module")
def table():
    return fig3.ensure_winners_auroc(GUIDE.read_table("alpha_sweep_by_lineage_masked"), GUIDE)


@pytest.fixture(scope="module")
def bands(table):
    return fig3.band_table(table, GUIDE, n_boot=300)


#: The golden values below were repinned on 23 Oct 2026 for the lineage reassignment
#: (docs/notes/LINEAGE_REASSIGNMENT_PLAN.md): the same thirteen lineages, delimited by the published
#: assignment rule generalised to the branching tree rather than by nearest root, with
#: outgroup roots for D and E. Membership moved a great deal -- 78,659 sequences became
#: 86,661 -- but every number here moved in the third decimal and no ordering changed.
#: Masked AUROC at alpha=0 went 0.9294 -> 0.928436 (fine-tuned) and 0.7927 -> 0.789431
#: (base); winners 0.5740 -> 0.576470 and 0.7394 -> 0.746716.


def raw_mean(table, model, metric, alpha):
    s = table[(table.model == model) & (table.model_variant == "plm_alpha_sweep") & (np.isclose(table.alpha, alpha))]
    assert len(s) == GUIDE.n_lineages
    return s[metric].mean()


@pytest.mark.parametrize("metric", fig3.TABLE_METRICS)
@pytest.mark.parametrize("alpha", [-1.0, 0.0, 1.0])
def test_band_means_equal_a_direct_groupby(table, bands, metric, alpha):
    for m in GUIDE.models:
        got = bands[(bands.metric == metric) & (bands.model == m.key) & np.isclose(bands.alpha, alpha)]["mean"].iloc[0]
        assert got == pytest.approx(raw_mean(table, m.key, metric, alpha), abs=1e-12)


def test_golden_auroc_at_alpha_zero(table):
    """Under masked marginals: 0.9294 fine-tuned, 0.7927 base."""
    by_role = {m.role: m.key for m in GUIDE.models}
    assert raw_mean(table, by_role["finetune"], "site_logistic_auroc", 0.0) == pytest.approx(0.928436, abs=5e-5)
    assert raw_mean(table, by_role["base"], "site_logistic_auroc", 0.0) == pytest.approx(0.789431, abs=5e-5)


def test_golden_winners_auroc_at_alpha_zero(table):
    """Mean over lineages, with no accessibility term: base 0.5740, fine-tuned 0.7394 under masked marginals."""
    by_role = {m.role: m.key for m in GUIDE.models}
    assert raw_mean(table, by_role["base"], "winners_auroc", 0.0) == pytest.approx(0.576470, abs=5e-4)
    assert raw_mean(table, by_role["finetune"], "winners_auroc", 0.0) == pytest.approx(0.746716, abs=5e-4)


def test_golden_auroc_at_alpha_zero_unmasked():
    """Matches the unmasked means quoted in LINEAGE_VALIDITY.md section 4, Q2 (0.9195 / 0.7721)."""
    unmasked = GUIDE.read_table("alpha_sweep_by_lineage")
    by_role = {m.role: m.key for m in GUIDE.models}
    assert raw_mean(unmasked, by_role["finetune"], "site_logistic_auroc", 0.0) == pytest.approx(0.914828, abs=5e-5)
    assert raw_mean(unmasked, by_role["base"], "site_logistic_auroc", 0.0) == pytest.approx(0.767442, abs=5e-5)


def test_golden_accessibility_only_baseline(table):
    base = fig3.baseline_table(table, GUIDE).set_index("metric")["mean"]
    assert base["site_logistic_auroc"] == pytest.approx(0.909034, abs=1e-5)
    assert base["mut_flat_global_spearman_r"] == pytest.approx(0.284996, abs=2e-5)
    assert base["winners_auroc"] == pytest.approx(0.532684, abs=1e-4)


def test_bands_bracket_the_mean_and_have_width(bands):
    assert (bands.lo <= bands["mean"] + 1e-12).all() and (bands["mean"] <= bands.hi + 1e-12).all()
    assert (bands.hi - bands.lo > 0).all()


def test_auroc_curve_dips_below_chance_for_negative_alpha(bands):
    """If this fails, a clamped column is being plotted."""
    a = bands[(bands.metric == "site_logistic_auroc") & (bands.alpha < 0)]
    assert (a["mean"] < 0.5).any() and a["mean"].min() < 0.3


def test_lineage_count_mismatch_with_guide_raises(table):
    import dataclasses
    short = dataclasses.replace(GUIDE, lineages=GUIDE.lineages[:-1])
    with pytest.raises(ValueError, match="lineages"):
        fig3.band_table(table, short, n_boot=10)


def test_dropped_lineage_in_table_raises(table):
    with pytest.raises(ValueError):
        fig3.band_table(table[table.lineage != GUIDE.lineage_ids[0]], GUIDE, n_boot=10)


def test_nan_in_metric_raises_instead_of_drawing_a_hole(table):
    broken = table.copy()
    broken.loc[broken.index[0], "site_logistic_auroc"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fig3.band_table(broken, GUIDE, n_boot=10)


def test_baseline_disagreement_between_models_raises(table):
    broken = table.copy()
    m = GUIDE.models[0].key
    mask = (broken.model == m) & broken.is_mutation_only_baseline.astype(bool)
    broken.loc[mask, "site_logistic_auroc"] += 0.1
    with pytest.raises(ValueError, match="differs between models"):
        fig3.baseline_table(broken, GUIDE)


@pytest.mark.slow
def test_script_renders_pdf_png_values_and_caption(tmp_path):
    rc = fig3.main(["--guide", str(GUIDE.source), "--out", str(tmp_path)])
    assert rc == 0
    for ext in ("pdf", "png"):
        assert (tmp_path / f"fig03_alpha_sweep.{ext}").stat().st_size > 10_000
    values = pd.read_csv(tmp_path / "fig03_alpha_sweep_values.tsv", sep="\t")
    sweep = values.dropna(subset=["alpha"])
    assert values["mean"].notna().all() and len(sweep) == len(fig3.TABLE_METRICS) * len(GUIDE.models) * 101
    assert (values["model"] == "Mutational_accessibility_alone").sum() == len(fig3.TABLE_METRICS)
    caption = (tmp_path / "fig03_alpha_sweep_caption.md").read_text()
    assert "13" in caption and "not show how well a model would do" in " ".join(caption.split())


def test_rendering_is_deterministic(table):
    a = fig3.band_table(table, GUIDE, n_boot=100, seed=5)
    b = fig3.band_table(table, GUIDE, n_boot=100, seed=5)
    pd.testing.assert_frame_equal(a, b)


@pytest.fixture(scope="module")
def drawn(table):
    fig, bands, baseline = fig3.draw(table, GUIDE)
    yield fig
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_vertical_axes_stop_at_the_no_relationship_level(drawn):
    """Below it the score is anti-predictive and says nothing; those values are cut."""
    axes = drawn.axes[: len(fig3.PANELS)]
    for ax, (_, _, _, _, ref) in zip(axes, fig3.PANELS):
        low, high = ax.get_ylim()
        assert ref[0] - 1.5 * style.AXIS_FLOOR_PAD * (high - ref[0]) <= low <= ref[0], (ax.get_title(), low)


def test_cut_values_are_still_in_the_values_table(table):
    """Cutting the axis must not cut the data: the table keeps the negative-alpha values."""
    bands = fig3.band_table(table, GUIDE, n_boot=10)
    auroc = bands[(bands.metric == "site_logistic_auroc") & (bands.alpha < 0)]
    assert auroc["mean"].min() < 0.3


def test_titles_name_the_metric_and_taglines_are_small(drawn):
    axes = drawn.axes[: len(fig3.PANELS)]
    for ax, (_, metric_name, tagline, _, _) in zip(axes, fig3.PANELS):
        assert ax.get_title(loc="left") == metric_name
        tag = [t for t in ax.texts if t.get_text() == tagline]
        assert len(tag) == 1
        assert tag[0].get_fontsize() < ax.title.get_fontsize()


def test_no_point_marker_sits_on_the_curves_at_alpha_zero(drawn):
    """Figure 4 spends a filled dot on a curve's best weight, which is a result.

    Figure 3 used the same mark at alpha = 0, a weight chosen in advance, so the two
    figures made unlike claims look alike. Nothing is drawn on the curves there now; the
    dashed rule carries it. The only markers left on a panel are the orange baseline
    squares, which sit off to the right of the sweep.
    """
    axes = drawn.axes[: len(fig3.PANELS)]
    for ax in axes:
        # Only scatter makes a PathCollection; the bands are PolyCollections, and those
        # report a single placeholder offset at the origin that is not a drawn point.
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert scatters, "the baseline square should still be a scatter on every panel"
        for coll in scatters:
            xs = [float(x) for x, _ in coll.get_offsets()]
            assert not any(abs(x - fig3.ALPHA_REFERENCE) < 1e-9 for x in xs), (
                f"a marker is drawn at alpha = {fig3.ALPHA_REFERENCE} on {ax.get_title(loc='left')!r}"
            )
        for line in ax.lines:
            if line.get_marker() not in ("", "None", None):
                assert fig3.ALPHA_REFERENCE not in list(line.get_xdata()), line


def test_alpha_zero_rule_uses_the_shared_emphasised_spec(drawn):
    """The rule is what the reader reads the crossing off, so it is the heavier shared one."""
    axes = drawn.axes[: len(fig3.PANELS)]
    for ax in axes:
        rules = [
            ln for ln in ax.lines
            if len(ln.get_xdata()) == 2
            and all(abs(float(x) - fig3.ALPHA_REFERENCE) < 1e-9 for x in ln.get_xdata())
            and ln.get_linestyle() != "-"
        ]
        assert len(rules) == 1, f"{len(rules)} vertical rules at alpha = 0"
        rule = rules[0]
        assert rule.get_linewidth() == style.ALPHA_ZERO_RULE["linewidth"]
        # Line2D normalises an on/off tuple to "--" and keeps the lengths privately, so the
        # pattern is compared against a probe built from the same spec rather than read back.
        probe = Line2D([], [], **style.ALPHA_ZERO_RULE)
        assert rule._unscaled_dash_pattern == probe._unscaled_dash_pattern
        assert rule.is_dashed()


def test_caption_does_not_promise_a_marker_that_is_not_drawn(drawn):
    caption = fig3.caption(GUIDE, fig3.band_table(GUIDE.read_table("alpha_sweep_by_lineage"), GUIDE, n_boot=10))
    assert "circle" not in caption.lower()
    assert "dashed vertical rule" in caption


def test_caption_says_the_axis_is_cut():
    caption = fig3.caption(GUIDE, fig3.band_table(GUIDE.read_table("alpha_sweep_by_lineage"), GUIDE, n_boot=10))
    assert "anti-predictive" in caption and "not drawn" in caption
