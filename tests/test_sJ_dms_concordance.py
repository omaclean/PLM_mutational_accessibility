"""Figure S10: agreement between two DMS scans of one protein, against strain divergence."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import make_sJ_dms_concordance as sJ
from mutacc import style
from mutacc.guide import REPO_ROOT, load_guide

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
SCRIPT = REPO_ROOT / "figures" / "make_sJ_dms_concordance.py"


@pytest.fixture(scope="module")
def table():
    return GUIDE.read_table("dms_cross_lineage_pairs")


@pytest.fixture(scope="module")
def drawn(table):
    return sJ.draw(table)


def row(table, system, category=None, s1=None, s2=None):
    sub = table[table.system == system]
    if category is not None:
        sub = sub[sub.metric_category == category]
    if s1 is not None:
        sub = sub[sub.sample_1.str.contains(s1) & sub.sample_2.str.contains(s2)]
    assert len(sub) == 1, (system, category, len(sub))
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# The shipped table: golden values (read from the archive's own table) and internal consistency
# ---------------------------------------------------------------------------

def test_table_has_every_pair(table):
    assert len(table) == 26 and table["system"].nunique() == 4 and table["metric_category"].nunique() == 3


def test_golden_headline_pairs(table):
    entry = row(table, "Full Spike Pseudovirus", "Cell Entry / Viral Fitness")
    binding = row(table, "Full Spike Pseudovirus", "ACE2 Binding")
    ha = row(table, "HA Entry vs Growth Preference")
    assert (entry.pearson_r, entry.spearman_rho) == (pytest.approx(0.8233, abs=5e-5), pytest.approx(0.7318, abs=5e-5))
    assert (binding.pearson_r, binding.spearman_rho) == (pytest.approx(0.5258, abs=5e-5), pytest.approx(0.3910, abs=5e-5))
    assert ha.pearson_r == pytest.approx(0.6192, abs=5e-5) and ha.aa_distance == 39
    assert ha.pct_divergence == pytest.approx(7.738, abs=5e-4)


def test_golden_floor_censoring_sensitivity(table):
    cross = table[table.system == "HA Cross-Subtype Entry"].sort_values("frac_at_floor")
    assert list(cross.frac_at_floor.round(2)) == [0.02, 0.31, 0.31]
    worst = cross.iloc[-1]
    assert worst.pearson_r == pytest.approx(0.5632, abs=5e-5) and worst.pearson_r_no_floor == pytest.approx(0.3830, abs=5e-5)


def test_percentage_divergence_is_distance_over_length(table):
    assert ((table.aa_distance / table.focal_protein_length * 100) - table.pct_divergence).abs().max() < 1e-9


def test_floor_censoring_inflates_the_coefficient(table):
    """Why the sensitivity is worth recording: with the floor pile-up in, the coefficient is
    higher. It is reported in the caption and the values table, never drawn as an interval."""
    heavy = table[table.frac_at_floor > sJ.FLOOR_BRACKET]
    assert len(heavy) == 2
    assert (heavy.pearson_r > heavy.pearson_r_no_floor).all() and (heavy.spearman_rho > heavy.spearman_rho_no_floor).all()


def test_agreement_falls_with_divergence_across_systems(table):
    """The headline trend, as a rank correlation over the shipped pairs (negative, and clearly so)."""
    rho = table[["pct_divergence", "spearman_rho"]].corr(method="spearman").iloc[0, 1]
    assert rho < -0.5
    # the unplotted Pearson column tells the same story, which is why only one of them is drawn
    assert table[["pct_divergence", "pearson_r"]].corr(method="spearman").iloc[0, 1] < -0.5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_empty_table_raises(table):
    with pytest.raises(ValueError, match="empty"):
        sJ.draw(table.iloc[0:0])


@pytest.mark.parametrize("column", ["aa_distance", "pct_divergence", "spearman_rho"])
def test_nan_in_a_plotted_column_raises(table, column):
    broken = table.copy()
    broken.loc[broken.index[3], column] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sJ.validate(broken)


def test_zero_distance_raises_for_the_log_axis(table):
    broken = table.copy()
    broken.loc[broken.index[0], "aa_distance"] = 0
    broken.loc[broken.index[0], "pct_divergence"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        sJ.validate(broken)


def test_inconsistent_percentage_raises(table):
    broken = table.copy()
    broken.loc[broken.index[0], "pct_divergence"] += 0.5
    with pytest.raises(ValueError, match="pct_divergence"):
        sJ.validate(broken)


def test_correlation_outside_unit_interval_raises(table):
    broken = table.copy()
    broken.loc[broken.index[0], "spearman_rho_no_floor"] = 1.2
    with pytest.raises(ValueError, match=r"outside \[-1, 1\]"):
        sJ.validate(broken)


def test_missing_column_raises(table):
    with pytest.raises(ValueError, match="lacks column"):
        sJ.validate(table.drop(columns=["frac_at_floor"]))


def test_more_species_than_marker_shapes_raises(table):
    extra = table.copy()
    extra["virus"] = [f"Virus {i % 3} (strain)" for i in range(len(extra))]
    with pytest.raises(ValueError, match="marker shapes"):
        sJ.validate(extra)


def test_panel_with_no_rows_raises(table):
    nothing = table.copy()
    nothing["spearman_rho"] = np.nan
    with pytest.raises(ValueError, match="no rows"):
        sJ.panel_points(nothing, "spearman_rho", "spearman_rho_no_floor", "aa_distance")


# ---------------------------------------------------------------------------
# Panel points
# ---------------------------------------------------------------------------

def test_jitter_is_bounded_and_seeded(table):
    a = sJ.panel_points(table, "spearman_rho", "spearman_rho_no_floor", "aa_distance")
    b = sJ.panel_points(table, "spearman_rho", "spearman_rho_no_floor", "aa_distance")
    pd.testing.assert_frame_equal(a, b)
    ratio = a["x_plotted"] / a["x"]
    assert ratio.between(1 - sJ.JITTER, 1 + sJ.JITTER).all() and ratio.nunique() > 1
    c = sJ.panel_points(table, "spearman_rho", "spearman_rho_no_floor", "aa_distance", seed=1)
    assert not np.allclose(a["x_plotted"], c["x_plotted"])


def test_floor_sensitivity_recorded_only_where_the_floor_fraction_exceeds_the_threshold(table):
    """The flag and the no-floor value reach the values table; see the drawing test below for the
    guarantee that neither becomes a mark on a panel."""
    pts = sJ.panel_points(table, "spearman_rho", "spearman_rho_no_floor", "pct_divergence")
    assert int(pts.censored.sum()) == 2
    assert pts.loc[~pts.censored, "y_no_floor"].isna().all() and pts.loc[pts.censored, "y_no_floor"].notna().all()
    assert (pts.loc[pts.censored, "y_no_floor"] < pts.loc[pts.censored, "y"]).all()


def test_both_panels_draw_every_pair(table):
    """Nothing drops out: every pair has a rank correlation, so both panels hold all 26."""
    for x_col in ("aa_distance", "pct_divergence"):
        pts = sJ.panel_points(table, "spearman_rho", "spearman_rho_no_floor", x_col)
        assert len(pts) == len(table) == 26


# ---------------------------------------------------------------------------
# Encoding is stable: filtering rows must not restyle the rest
# ---------------------------------------------------------------------------

def test_marker_and_colour_assignments_come_from_the_full_table(table):
    """Shape is the species and colour the trait, both assigned by sorted name, so a panel that
    draws a subset of the rows cannot restyle the groups that remain."""
    shapes = sJ.species_markers(table)
    assert list(shapes) == sorted(shapes) == ["Influenza A", "SARS-CoV-2"]
    assert len(set(shapes.values())) == len(shapes) == 2
    colours = sJ.trait_colours(table)
    assert list(colours) == sorted(colours) and len(set(colours.values())) == 3
    # a panel that draws a subset still uses the full-table assignment
    sub = table[table.system != table.system.iloc[0]]
    assert sJ.species_markers(sub).items() <= shapes.items()
    assert sJ.trait_colours(sub).items() <= colours.items()


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def test_two_lettered_panels(drawn):
    axes = [ax for ax in drawn.axes if ax.lines]
    assert len(axes) == 2
    assert sorted(t.get_text() for ax in axes for t in ax.texts if t.get_fontweight() == "bold") == list("AB")


def test_the_question_is_not_repeated_as_a_panel_title(drawn):
    """Both panels answer the same question of the same points; the y label and caption carry it."""
    assert [ax.get_title() for ax in drawn.axes if ax.lines] == ["", ""]


def test_each_panel_draws_one_marker_per_pair(drawn, table):
    counts = []
    for ax in drawn.axes:
        points = [ln for ln in ax.lines if ln.get_linestyle() == "None" and len(ln.get_xdata()) == 1 and ln.get_marker() != "None"]
        counts.append(len(points))
    assert [c for c in counts if c] == [26, 26]


def test_no_mark_on_a_panel_has_a_vertical_extent(drawn):
    """No error bar, and no bar a reader could take for one.

    Two pairs carry a floor-censoring sensitivity range (the coefficient with and without the
    variants piled at the assay detection floor). It used to be drawn as an I-bar, which is the
    grammar of a confidence interval: a reader reads sampling uncertainty onto those two points
    and, by omission, precision onto the other twenty-four. The range is reported in the caption
    and the values table instead. This pins that no mark spans two y values, whatever its colour
    or source -- a bracket, an errorbar container, or a stray two-point segment.
    """
    reference = {"perfect agreement = 1"}
    for ax in [a for a in drawn.axes if a.lines]:
        assert not ax.containers, "an errorbar/bar container was added to a panel"
        for ln in ax.lines:
            ys = np.asarray(ln.get_ydata(), dtype=float)
            if ys.size < 2 or ln.get_linestyle() == "None":
                continue
            if ln in ax.get_lines() and np.allclose(ys, 1.0):
                continue  # the dashed "perfect agreement = 1" reference, which is horizontal
            assert np.allclose(ys, ys[0]), f"a mark spans y {ys.min():.3f}-{ys.max():.3f}"
    assert reference <= {t.get_text() for a in drawn.axes for t in a.texts}


def test_the_floor_sensitivity_is_reported_in_words_not_drawn(table):
    """The information survives the removal: both coefficients reach the caption and the table."""
    text = re.sub(r"\s+", " ", sJ.caption(table))
    censored = table[table.frac_at_floor > sJ.FLOOR_BRACKET]
    assert f"For {len(censored)} pairs, more than" in text
    for col in ("spearman_rho", "spearman_rho_no_floor"):
        for v in censored[col]:
            assert f"{v:.2f}" in text, f"{col}={v:.2f} missing from the caption"
    values = sJ.values_table(table)
    assert int(values.censored.sum()) == 2 * len(censored)
    assert values.loc[values.censored, "y_no_floor"].notna().all()


def test_axes_are_log_and_agreement_panels_have_a_labelled_reference(drawn):
    scales = [(ax.get_xscale(), ax.get_yscale()) for ax in drawn.axes if ax.lines]
    assert all(x == "log" and y == "linear" for x, y in scales)
    for ax in [ax for ax in drawn.axes if ax.lines]:
        assert "perfect agreement = 1" in [t.get_text() for t in ax.texts]


def test_no_label_overlaps_another_and_none_leaves_the_axes(drawn):
    assert max(drawn.label_scores) < 500


def test_series_colours_come_only_from_the_categorical_palette(drawn):
    """Traits are coloured from CATEGORICAL; furniture stays grey. Nothing else appears."""
    furniture = {style.INK, style.INK_MUTED, style.SURFACE, style.NULL_GREY}
    allowed = furniture | set(style.CATEGORICAL)
    for ax in drawn.axes:
        for ln in ax.lines:
            assert ln.get_color() in allowed or ln.get_marker() != "None", ln.get_color()
            if ln.get_marker() != "None":
                assert ln.get_markeredgecolor() in allowed, ln.get_markeredgecolor()


def test_no_model_colour_appears(drawn):
    """No model is plotted here, so base blue and fine-tune green must not show up.

    They mean "untuned" and "fine-tuned" in every other figure in the set; borrowing
    them for an assay type would carry that meaning across.
    """
    reserved = {style.BASE, style.FINETUNE}
    for ax in drawn.axes:
        for ln in ax.lines:
            assert ln.get_color() not in reserved, ln.get_color()
            if ln.get_marker() != "None":
                assert ln.get_markeredgecolor() not in reserved
                assert ln.get_markerfacecolor() not in reserved


def test_colour_is_the_only_trait_channel(table):
    """One channel per variable: colour is the trait, shape is the species, nothing doubles up."""
    colours = sJ.trait_colours(table)
    assert len(set(colours.values())) == len(colours), "two traits share a colour"
    assert set(colours) == set(table.metric_category)


def test_marker_faces_are_translucent_and_edges_are_not(drawn):
    """Opacity is what lets a stack of points read as a stack; the edge stays crisp."""
    faces = [ln.get_markerfacecolor() for ax in drawn.axes for ln in ax.lines
             if ln.get_marker() not in ("None", None) and len(ln.get_xdata()) == 1]
    assert faces, "no markers drawn"
    assert all(f[3] == sJ.FACE_ALPHA for f in faces)


def test_axis_labels_are_plain_words(drawn):
    labels = [ax.get_ylabel() for ax in drawn.axes if ax.lines] + [ax.get_xlabel() for ax in drawn.axes if ax.lines]
    assert any("Spearman" in label for label in labels)
    assert not any(re.fullmatch(r"[a-z_]+", label) for label in labels if label)


def test_main_writes_figure_values_and_caption(tmp_path, table):
    rc = sJ.main(["--guide", str(GUIDE.source), "--out", str(tmp_path)])
    assert rc == 0
    for ext in ("pdf", "png"):
        assert (tmp_path / f"{sJ.STEM}.{ext}").stat().st_size > 20_000
    values = pd.read_csv(tmp_path / f"{sJ.STEM}_values.tsv", sep="\t")
    assert sorted(values.panel.unique()) == list("AB")
    assert len(values) == 2 * 26
    text = " ".join((tmp_path / f"{sJ.STEM}_caption.md").read_text().split())
    for phrase in ("deep mutational scanning (DMS)", "Jensen-Shannon divergence",
                   "rank agreement is the only measure plotted", "the true value lies between",
                       "there are no uncertainty intervals anywhere in it",
                   "not like-for-like", "26 pairs"):
        assert phrase.lower() in text.lower(), phrase


def test_caption_pluralises_and_counts_from_the_data(table):
    text = " ".join(sJ.caption(table).split())
    assert "(1 pair)" in text and "(3 pairs)" in text and "(20 pairs)" in text
    trimmed = table[table.system != "Full Spike Pseudovirus"]
    assert "(24 pairs:" in " ".join(sJ.caption(trimmed).split())


# ---------------------------------------------------------------------------
# House rules specific to this script
# ---------------------------------------------------------------------------

def test_script_does_not_name_systems_traits_or_samples(table):
    source = SCRIPT.read_text()
    names = set(table.system) | set(table.metric_category) | set(table.sample_1) | set(table.sample_2) | set(table.virus)
    hits = [n for n in names if re.search(r'["\']' + re.escape(n) + r'["\']', source)]
    assert not hits, f"table values typed into the script: {hits}"
