"""S7: epistasis against a null. Golden values come from the archive's own tables and notes."""

from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd
import pytest

import make_sG_epistasis_null as sG
from mutacc.guide import REPO_ROOT, GuideError, load_guide

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
SCRIPT = REPO_ROOT / "figures" / "make_sG_epistasis_null.py"
GAUGES = ["wt", "logit", "prob", "zerosum"]


@pytest.fixture(scope="module")
def tables():
    return (GUIDE.read_table("epistasis_cells"), GUIDE.read_table("epistasis_null_draws"),
            GUIDE.read_table("epistasis_focal_sites"))


@pytest.fixture(scope="module")
def raw_cells(tables):
    """The shipped table, every gauge. Still four, because the archive exported four."""
    return tables[0]


@pytest.fixture(scope="module")
def raw_draws(tables):
    return tables[1]


@pytest.fixture(scope="module")
def cells(tables):
    """What the figure actually works on: zero-sum only, gauge column gone."""
    return sG.select_gauge(tables[0], "cells")


@pytest.fixture(scope="module")
def draws(tables):
    return sG.select_gauge(tables[1], "null draws")


@pytest.fixture(scope="module")
def sites(tables):
    return tables[2]


def cell(raw_cells, gauge, background, focal):
    sub = raw_cells[(raw_cells.gauge == gauge) & (raw_cells.background == background) & (raw_cells.focal == focal)]
    assert len(sub) == 1
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# Shape and golden values
# ---------------------------------------------------------------------------

def test_shape(raw_cells, raw_draws, cells, draws, sites):
    assert sorted(raw_cells.gauge.unique()) == sorted(GAUGES)
    assert (raw_cells.groupby("gauge").size() == 72).all()
    assert len(sites) == 9
    assert (raw_draws.groupby(["gauge", "focal"]).size() == 10298).all()
    assert len(raw_draws) == 4 * 9 * 10298
    assert (raw_cells.null_n == 10298).all()
    # the figure sees one gauge of that, with the gauge column dropped
    assert len(cells) == 72 and "gauge" not in cells.columns
    assert len(draws) == 9 * 10298 and "gauge" not in draws.columns
    sG.validate_inputs(cells, draws, sites)


@pytest.mark.parametrize("gauge,lo,hi", [("zerosum", -0.731, 0.612), ("wt", -0.649, 1.029), ("prob", -0.029, 0.135)])
def test_cell_ranges_match_archive_notes(raw_cells, gauge, lo, hi):
    d = raw_cells[raw_cells.gauge == gauge].delta
    assert d.min() == pytest.approx(lo, abs=6e-4) and d.max() == pytest.approx(hi, abs=6e-4)


def test_the_preprints_probability_cells_reproduce(raw_cells):
    """The archive notes: S144N->I160K 0.135, N158D->I160K 0.121, K189R->I160K -0.029."""
    assert cell(raw_cells, "prob", "S144N", "I160K").delta == pytest.approx(0.13537, abs=5e-5)
    assert cell(raw_cells, "prob", "N158D", "I160K").delta == pytest.approx(0.12146, abs=5e-5)
    assert cell(raw_cells, "prob", "K189R", "I160K").delta == pytest.approx(-0.02920, abs=5e-5)


def test_headline_cell_across_gauges(raw_cells):
    """S144N -> I160K: delta, and the q values the archive README tabulates."""
    z = cell(raw_cells, "zerosum", "S144N", "I160K")
    assert z.delta == pytest.approx(0.611751, abs=1e-6)
    assert z.q_bh_pooled == pytest.approx(0.0186, abs=5e-4) and z.q_bh_stud == pytest.approx(0.035, abs=5e-4)
    w = cell(raw_cells, "wt", "S144N", "I160K")
    assert w.delta == pytest.approx(0.5634, abs=5e-4) and w.q_bh_pooled == pytest.approx(0.084, abs=2e-3)
    assert w.q_bh_stud == pytest.approx(0.115, abs=2e-3)
    p = cell(raw_cells, "prob", "S144N", "I160K")
    assert p.q_bh_pooled == pytest.approx(0.120, abs=2e-3)


def test_n158d_changes_sign_between_probability_and_wild_type_scales(raw_cells):
    """Why the scale is not a free choice: this pair points the other way on two of them.

    The figure no longer draws or compares gauges, so this asserts the fact about the shipped
    table directly rather than through a helper that has been removed with that path.
    """
    p = cell(raw_cells, "prob", "N158D", "I160K").delta
    w = cell(raw_cells, "wt", "N158D", "I160K").delta
    assert p > 0 > w and w == pytest.approx(-0.2085, abs=5e-4)


def test_how_many_pairs_stand_out(cells):
    c = sG.how_many_stand_out(cells)
    assert c["n_cells"] == 72
    assert (c["n_q_pooled"], c["n_q_studentised"]) == (3, 6)


def test_clade_pairs_that_stand_out_under_zerosum(cells):
    sub = cells[cells.q_bh_pooled < 0.05]
    assert set(zip(sub.background, sub.focal)) == {("S144N", "I160K"), ("K189R", "Q173R"), ("N158D", "Q173R")}


# ---------------------------------------------------------------------------
# Recomputation from the draws
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recomputed(cells, draws):
    return sG.recompute_placement(cells, draws)


def test_pooled_p_and_q_values_recompute(recomputed):
    gaps = sG.check_placement(recomputed)
    assert gaps["p"] < 2.5 / 10299 and gaps["q_stud"] < 1e-9
    # almost always exact: ties are the only source of a one-draw difference
    off = (recomputed.p_two_re - recomputed.p_pooled_two).abs() > 1e-9
    assert off.sum() <= 3


def test_null_summaries_recompute(cells, draws):
    assert sG.check_null_summaries(cells, draws) < 1e-9


def test_a_tampered_p_value_is_caught(recomputed):
    bad = recomputed.copy()
    bad.loc[5, "p_pooled_two"] += 0.01
    with pytest.raises(ValueError, match="disagree"):
        sG.check_placement(bad)


def test_a_tampered_q_value_is_caught(recomputed):
    bad = recomputed.copy()
    bad.loc[7, "q_bh_stud"] = 0.001
    with pytest.raises(ValueError, match="BH"):
        sG.check_placement(bad)
    bad = recomputed.copy()
    bad["q_bh_pooled"] = bad["q_bh_pooled"] * 0.5
    with pytest.raises(ValueError, match="BH"):
        sG.check_placement(bad)


def test_a_tampered_draw_is_caught(cells, draws):
    bad = draws.copy()
    bad.loc[bad.index[:2000], "delta"] += 0.05
    with pytest.raises(ValueError):
        sG.check_null_summaries(cells, bad)


def test_empirical_p_edges():
    null = np.arange(10.0)
    up, dn, two = sG.empirical_p(null, 100.0)
    assert up == pytest.approx(1 / 11) and dn == 1.0 and two == pytest.approx(2 / 11)
    up, dn, two = sG.empirical_p(null, -100.0)
    assert dn == pytest.approx(1 / 11) and up == 1.0
    assert sG.empirical_p(null, 4.5)[2] == 1.0          # dead centre: two-sided p capped at 1
    assert sG.empirical_p(np.zeros(5), 0.0) == (1.0, 1.0, 1.0)  # all ties


def test_benjamini_hochberg_known_values_and_nan():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    q = sG.benjamini_hochberg(p)
    assert q[:4] == pytest.approx([0.02, 0.04, 0.04, 0.02]) and np.isnan(q[4])
    assert np.isnan(sG.benjamini_hochberg(np.array([np.nan, np.nan]))).all()
    assert sG.benjamini_hochberg(np.array([0.9, 0.99])).max() <= 1.0
    q = sG.benjamini_hochberg(np.random.default_rng(1).uniform(size=200))
    assert (q >= 0).all() and (q <= 1).all()


def test_bh_is_monotone_in_p():
    p = np.random.default_rng(2).uniform(size=300) ** 3
    q = sG.benjamini_hochberg(p)
    order = np.argsort(p)
    assert (np.diff(q[order]) >= -1e-12).all() and (q >= p - 1e-12).all()


# ---------------------------------------------------------------------------
# Numbering: HA1 / HA2 / signal peptide
# ---------------------------------------------------------------------------

def test_numbering_of_the_shipped_sites(sites):
    sG.check_numbering(sites)
    ha2 = sites[sites.region == "HA2"].iloc[0]
    assert (ha2.focal, ha2.index_label, ha2.seq_pos, ha2.region_pos) == ("HA2:S49N", "S378N", 378, 49)
    assert 378 - 330 + 1 == 49
    ha1 = sites[sites.region == "HA1"]
    assert (ha1.seq_pos == ha1.region_pos).all() and ha1.seq_pos.max() < 330


def test_clade_site_positions(sites):
    assert dict(zip(sites.index_label, sites.seq_pos)) == {
        "K2N": 2, "T135K": 135, "S144N": 144, "N158D": 158, "I160K": 160, "Q173R": 173,
        "K189R": 189, "T328A": 328, "S378N": 378}


def test_baseline_probabilities(sites):
    s = sites.set_index("focal")
    assert s.loc["N158D", "p_wt0"] == pytest.approx(0.985166, abs=1e-6)      # headroom_by_site.tsv
    assert s.loc["I160K", "p_wt0"] == pytest.approx(0.0606768, abs=1e-6)
    assert s.loc["S144N", "p_wt0"] == pytest.approx(0.718825, abs=1e-6)
    assert s.loc["I160K", "p_alt0"] == pytest.approx(0.520226, abs=1e-6)
    assert ((sites.p_wt0 + sites.p_alt0) <= 1.0 + 1e-9).all()


def test_probability_shift_is_bounded_by_baseline_probability(raw_cells, sites):
    """P(alt) lies in [0, 1], so a probability shift cannot be below -p_alt0 or above 1 - p_alt0."""
    p = raw_cells[raw_cells.gauge == "prob"].merge(sites[["focal", "p_alt0"]], on="focal")
    assert (p.delta >= -p.p_alt0 - 1e-9).all() and (p.delta <= 1 - p.p_alt0 + 1e-9).all()
    assert np.allclose(p.focal_p0, p.p_alt0, atol=1e-9)


@pytest.mark.parametrize("edit,match", [
    ({"focal": "HA2:S378N"}, "label"),                       # sequence index used as an HA2 label
    ({"seq_pos": 394}, "index label"),      # shifted by the signal peptide
    ({"region": "HA1"}, "HA1 site"),
    ({"region_pos": 50}, "HA2 sites imply|label"),
    ({"index_label": "S377N"}, "index label"),
    ({"p_wt0": 1.5}, "outside"),
    ({"region": "HC"}, "unknown region"),
])
def test_numbering_errors_are_caught(sites, edit, match):
    bad = sites.copy()
    i = bad.index[bad.region == "HA2"][0]
    for k, v in edit.items():
        bad.loc[i, k] = v
    with pytest.raises(ValueError, match=match):
        sG.check_numbering(bad)


def test_ha1_position_shifted_by_the_signal_peptide_is_caught(sites):
    bad = sites.copy()
    i = bad.index[bad.focal == "I160K"][0]
    bad.loc[i, ["focal", "region_pos"]] = ["I176K", 176]
    bad.loc[i, "seq_pos"] = 160
    with pytest.raises(ValueError):
        sG.check_numbering(bad)


def test_two_different_ha2_starts_are_caught(sites):
    bad = pd.concat([sites, sites[sites.region == "HA2"].assign(focal="HA2:S50N", region_pos=50, seq_pos=380,
                                                                index_label="S380N")], ignore_index=True)
    with pytest.raises(ValueError, match="start"):
        sG.check_numbering(bad)


# ---------------------------------------------------------------------------
# Malformed inputs
# ---------------------------------------------------------------------------

def test_missing_columns_rejected(tables):
    cells, draws, sites = (t.copy() for t in tables)
    with pytest.raises(ValueError, match="lacks columns"):
        sG.validate_inputs(cells.drop(columns=["q_bh_pooled"]), draws, sites)
    with pytest.raises(ValueError, match="lacks columns"):
        sG.validate_inputs(cells, draws.drop(columns=["delta"]), sites)
    with pytest.raises(ValueError, match="lacks columns"):
        sG.validate_inputs(cells, draws, sites.drop(columns=["region_pos"]))


def test_nan_and_infinite_values_rejected(tables):
    cells, draws, sites = (t.copy() for t in tables)
    cells.loc[0, "delta"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sG.validate_inputs(cells, draws, sites)
    cells, draws, sites = (t.copy() for t in tables)
    draws.loc[3, "delta"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        sG.validate_inputs(cells, draws, sites)


def test_duplicate_and_self_pairs_rejected(cells, draws, sites):
    cells, draws, sites = cells.copy(), draws.copy(), sites.copy()
    with pytest.raises(ValueError, match="duplicate"):
        sG.validate_inputs(pd.concat([cells, cells.iloc[:1]]), draws, sites)
    bad = cells.copy()
    bad.loc[0, "background"] = bad.loc[0, "focal"]
    with pytest.raises(ValueError, match="own background"):
        sG.validate_inputs(bad, draws, sites)


def test_mismatched_draws_rejected(cells, draws, sites):
    cells, draws, sites = cells.copy(), draws.copy(), sites.copy()
    with pytest.raises(ValueError, match="differ in size"):
        sG.validate_inputs(cells, draws.drop(draws.index[0]), sites)
    with pytest.raises(ValueError, match="null_n"):
        sG.validate_inputs(cells.assign(null_n=10000), draws, sites)
    with pytest.raises(ValueError, match="missing from the site table"):
        sG.validate_inputs(cells, draws, sites.iloc[1:])


def test_empty_table_is_a_guide_error_not_a_blank_panel(tmp_path):
    import shutil
    results = tmp_path / "r"
    shutil.copytree(GUIDE.results_dir, results, ignore=shutil.ignore_patterns("symmetry*", "dms*", "replacement*", "alpha*"))
    pd.read_csv(results / "epistasis_cells.tsv.gz", sep="\t").iloc[:0].to_csv(
        results / "epistasis_cells.tsv.gz", sep="\t", index=False)
    with pytest.raises(GuideError, match="empty"):
        sG.main(["--guide", str(GUIDE.source), "--results", str(results), "--out", str(tmp_path / "o")])


def test_all_nan_null_stats_do_not_slip_through(cells, draws):
    bad = cells.assign(p_stud_two=np.nan)
    with pytest.raises(ValueError):
        sG.check_placement(sG.recompute_placement(bad, draws))


def test_a_table_without_the_zerosum_gauge_is_refused(raw_cells):
    """The gauge is not switchable any more, so the failure to guard against is its absence."""
    with pytest.raises(ValueError, match="no 'zerosum' rows"):
        sG.select_gauge(raw_cells[raw_cells.gauge != "zerosum"], "cells")
    bare = raw_cells.drop(columns=["gauge"])
    assert sG.select_gauge(bare, "cells") is bare


def test_matrix_orientation_rows_scored_columns_present(cells, sites):
    order = list(sites.focal)
    m = sG.matrix_of(cells, order)
    assert m.loc["I160K", "S144N"] == pytest.approx(0.611751, abs=1e-6)      # row = scored, column = already present
    assert m.loc["S144N", "I160K"] != m.loc["I160K", "S144N"]
    assert np.isnan(np.diag(m.values)).all()
    assert int(m.notna().sum().sum()) == 72


def test_null_bands_are_ordered(draws, sites):
    b = sG.null_bands(draws, list(sites.focal))
    assert (b.p1 <= b.p5).all() and (b.p5 <= b.p25).all() and (b.p25 <= b.p50).all() and (b.p75 <= b.p95).all() and (b.p95 <= b.p99).all()
    assert (b.n_draws == 10298).all()


def test_nulls_exclude_clade_sites_so_pooled_size_is_stable(raw_draws):
    assert raw_draws.groupby("gauge").size().nunique() == 1


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

def test_the_only_gauge_is_zerosum_and_is_named_in_plain_words():
    assert sG.GAUGE == "zerosum"
    assert " " in sG.SCALE and "_" not in sG.SCALE and sG.DEFINITION
    for gone in ("GAUGE_INFO", "DEFAULT_GAUGE", "PUBLISHED_GAUGE", "gauge_summary", "sign_changes", "gauge_names"):
        assert not hasattr(sG, gone), f"{gone} is part of the retired gauge-switching path"


def test_no_absolute_paths_in_script():
    assert not any(n in SCRIPT.read_text() for n in ("/" + "home", "/" + "software", "~" + "/"))


def test_seed_free_figure_is_deterministic(cells, draws, sites):
    a = sG.null_bands(draws, list(sites.focal))
    b = sG.null_bands(draws, list(sites.focal))
    pd.testing.assert_frame_equal(a, b)


def test_caption_is_data_driven(cells, draws, sites):
    text = " ".join(sG.caption(cells, draws, sites).split())
    assert "3 of the 72 pairs" in text and "S144N then I160K" in text and "10,298" in text and "92,682" in text
    assert "mature numbering" in text and "add 16" in text
    assert "zero-sum scale" in text and "Zhang and Ovchinnikov" in text
    assert "Panel D" not in text, "the cross-gauge panel is gone; the caption must not promise it"


@pytest.mark.slow
def test_script_renders(tmp_path):
    assert sG.main(["--guide", str(GUIDE.source), "--out", str(tmp_path)]) == 0
    stem = "sG_epistasis_null"
    assert (tmp_path / f"{stem}.png").stat().st_size > 50_000 and (tmp_path / f"{stem}.pdf").exists()
    v = pd.read_csv(tmp_path / f"{stem}_values.tsv", sep="\t")
    assert len(v) == 72 and set(v.gauge) == {"zerosum"}
    assert not (tmp_path / f"{stem}_gauges_values.tsv").exists(), "the per-gauge table is retired"
    assert (tmp_path / f"{stem}_caption.md").read_text().count("\n") > 3


def test_draws_are_sorted_within_each_group_and_finite(raw_draws):
    for (g, f), grp in raw_draws.groupby(["gauge", "focal"]):
        v = grp["delta"].to_numpy()
        assert (np.diff(v) >= 0).all() and np.isfinite(v).all()


def test_site_order_matches_the_preprint_matrix(sites):
    assert list(sites.index_label) == ["K2N", "T135K", "S144N", "N158D", "I160K", "Q173R", "K189R", "T328A", "S378N"]
    assert list(sites.focal)[-1] == "HA2:S49N"
