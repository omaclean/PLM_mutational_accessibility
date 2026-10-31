"""The arithmetic behind Figure 4: ranks, curves, optimum flags, accessibility, label placement."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr

from mutacc import dms_sweep as D
from mutacc import style
from mutacc.placement import Candidate, curve_obstacles, place_existing, place_label


# ---------------------------------------------------------------------------
# Ranks and correlations
# ---------------------------------------------------------------------------

def test_col_ranks_match_scipy_including_ties():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 6, size=(80, 5)).astype(float)       # heavy ties
    X[:, 2] = rng.normal(size=80)                             # a tie-free column
    expected = np.column_stack([rankdata(X[:, j]) - 1 for j in range(X.shape[1])])   # 0-based
    np.testing.assert_allclose(D.col_ranks(X), expected, atol=1e-12)


def test_rho_cols_match_scipy():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 4))
    X[::3, 1] = 0.0
    y = rng.normal(size=120)
    yr = rankdata(y)
    got = D.rho_cols(X, yr - yr.mean())
    expected = [spearmanr(X[:, j], y).statistic for j in range(4)]
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_constant_column_gives_nan_not_zero():
    X = np.column_stack([np.ones(30), np.arange(30.0)])
    yr = rankdata(np.arange(30.0))
    out = D.rho_cols(X, yr - yr.mean())
    assert np.isnan(out[0]) and out[1] == pytest.approx(1.0)


def test_sweep_curve_alpha_zero_is_the_model_alone():
    rng = np.random.default_rng(3)
    lp, la, y = rng.normal(size=200), rng.normal(size=200), rng.normal(size=200)
    curve = D.sweep_curve(lp, la, y, np.array([-1.0, 0.0, 1.0]))
    assert curve[1] == pytest.approx(spearmanr(lp, y).statistic, abs=1e-12)
    assert curve[2] == pytest.approx(spearmanr(lp + la, y).statistic, abs=1e-12)


def test_sweep_recovers_a_planted_accessibility_contamination():
    """Fitness is the model's signal; accessibility is added to the score. The optimum is negative."""
    rng = np.random.default_rng(4)
    truth, acc = rng.normal(size=4000), rng.normal(size=4000)
    log_plm = truth + 0.8 * acc                    # the model is contaminated by accessibility
    alphas = np.round(np.arange(-1, 1.001, 0.01), 6)
    flags = D.curve_flags(D.sweep_curve(log_plm, acc, truth, alphas), alphas)
    # One grid step, not the six this used to allow. Measured 15 Oct 2026: the
    # recovered optimum is exactly -0.8 on seeds 0-5, so abs=0.06 was guarding a
    # result that does not wander. A step of slack is kept only against a
    # tie-break change in the argmax.
    assert flags["best_alpha"] == pytest.approx(-0.8, abs=0.01)
    assert flags["interpretable"]


# ---------------------------------------------------------------------------
# Optimum flags
# ---------------------------------------------------------------------------

ALPHAS = np.round(np.arange(-1, 1.001, 0.01), 6)


def test_flag_boundary_when_the_peak_is_at_the_end_of_the_range():
    rho = 0.2 + 0.1 * ALPHAS                       # rises to the right edge
    flags = D.curve_flags(rho, ALPHAS)
    assert flags["boundary"] and not flags["interpretable"] and flags["best_alpha"] == 1.0


def test_flag_weak_signal_when_rho_at_zero_is_tiny():
    rho = 0.01 - (ALPHAS + 0.3) ** 2 * 0.01
    flags = D.curve_flags(rho, ALPHAS)
    assert flags["weak_signal"] and not flags["boundary"] and not flags["interpretable"]


def test_flag_interpretable_for_an_interior_peak_with_signal():
    rho = 0.3 - (ALPHAS + 0.2) ** 2 * 0.3
    flags = D.curve_flags(rho, ALPHAS)
    assert flags["interpretable"] and flags["best_alpha"] == pytest.approx(-0.2)


def test_flag_rejects_a_curve_with_no_finite_values():
    with pytest.raises(ValueError, match="no finite"):
        D.curve_flags(np.full(len(ALPHAS), np.nan), ALPHAS)


# ---------------------------------------------------------------------------
# Site bootstrap
# ---------------------------------------------------------------------------

def _toy(n_sites=40, per_site=6, seed=5):
    rng = np.random.default_rng(seed)
    sites = np.repeat(np.arange(n_sites), per_site)
    site_effect = np.repeat(rng.normal(size=n_sites), per_site)
    lp = rng.normal(size=len(sites)) + site_effect
    la = rng.normal(size=len(sites))
    y = lp + rng.normal(size=len(sites))
    return lp, la, y, sites


def test_band_brackets_the_curve_and_has_width():
    lp, la, y, sites = _toy()
    alphas = np.round(np.arange(-1, 1.001, 0.05), 6)
    rho = D.sweep_curve(lp, la, y, alphas)
    lo, hi = D.site_bootstrap_band(lp, la, y, sites, alphas, np.random.default_rng(0), n_boot=60, band_every=2)
    assert (lo < hi).all()
    assert ((lo <= rho + 0.05) & (rho - 0.05 <= hi)).all()


def test_clustering_widens_the_band_relative_to_row_resampling():
    """Sites carry a shared effect, so resampling sites must give a wider band than rows would."""
    lp, la, y, sites = _toy(n_sites=30, per_site=20)
    alphas = np.array([0.0])
    lo_s, hi_s = D.site_bootstrap_band(lp, la, y, sites, alphas, np.random.default_rng(0), n_boot=300, band_every=1)
    lo_r, hi_r = D.site_bootstrap_band(lp, la, y, np.arange(len(y)), alphas, np.random.default_rng(0),
                                       n_boot=300, band_every=1)
    assert (hi_s - lo_s)[0] > (hi_r - lo_r)[0]


def test_band_knots_always_include_the_last_alpha():
    knots = D.band_knots(np.arange(0, 11.0), 4)
    assert knots[-1] == 10.0 and list(knots[:3]) == [0.0, 4.0, 8.0]


def test_zero_bootstrap_replicates_raise():
    lp, la, y, sites = _toy()
    with pytest.raises(ValueError, match="n_boot"):
        D.site_bootstrap_band(lp, la, y, sites, np.array([0.0]), np.random.default_rng(0), n_boot=0)


# ---------------------------------------------------------------------------
# Genetic code and accessibility
# ---------------------------------------------------------------------------

def test_standard_code_matches_biopython():
    """The code is a literal in the library; Biopython's table 1 is the authority."""
    codon_table = pytest.importorskip("Bio.Data.CodonTable")
    table = codon_table.unambiguous_dna_by_id[1]
    codons = ["".join(c) for c in itertools.product("TCAG", repeat=3)]
    expected = "".join(table.forward_table.get(c, "*") for c in codons)
    assert D.STANDARD_CODE == expected


def _matrix(scale=1e-5):
    rng = np.random.default_rng(6)
    m = pd.DataFrame(rng.uniform(0.2, 1.0, (4, 4)) * scale, index=list("ACGT"), columns=list("ACGT"))
    m.index.name = "from_base"
    return m


def test_accessibility_table_shape_and_synonymous_dominance():
    table = D.accessibility_table(_matrix(), reversible=False)
    assert table.shape == (64, 20)
    assert set(table.columns) == set("ACDEFGHIKLMNPQRSTVWY")
    # a codon "mutates" to its own amino acid with probability ~1 (the diagonal term)
    assert table.loc["GCT", "A"] > 0.99


def test_reversible_matrix_is_symmetric_in_effect():
    m = _matrix()
    forward = D.accessibility_table(m, reversible=True)
    flipped = D.accessibility_table(m.T, reversible=True)       # transpose changes direction only
    pd.testing.assert_frame_equal(forward, flipped)
    directional = D.accessibility_table(m, reversible=False)
    assert not np.allclose(directional.to_numpy(), forward.to_numpy())


def test_one_nucleotide_step_is_far_more_accessible_than_three():
    table = D.accessibility_table(_matrix(), reversible=False)
    one_step = table.loc["GCT", "S"]        # GCT (Ala) -> TCT (Ser): one change
    three_step = table.loc["GCT", "W"]      # Ala -> Trp needs three changes
    assert one_step > 1e3 * three_step > 0


def test_log_accessibility_floors_zeros_and_rejects_unknowns():
    table = pd.DataFrame([[0.0, 1e-3], [1e-2, 0.1]], index=["AAA", "AAC"], columns=["K", "N"])
    got = D.log_accessibility(["AAA", "AAC"], ["K", "N"], table)
    # the floor is a tenth of the smallest positive value among the pairs looked up (0.1 here)
    assert got[0] == pytest.approx(np.log10(0.1 / 10))
    assert got[1] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="not in the accessibility table"):
        D.log_accessibility(["TTT"], ["K"], table)


# ---------------------------------------------------------------------------
# Label placement
# ---------------------------------------------------------------------------

def _axes():
    fig = style.figure(style.WIDTH_SINGLE, 2.2)
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 100)
    ax.plot(x, 0.5 + 0 * x)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    return fig, ax, x


def test_place_label_avoids_a_curve_when_a_clear_spot_exists():
    fig, ax, x = _axes()
    obstacles = curve_obstacles(ax, [(x, 0.5 + 0 * x)])
    on_line = Candidate((0.5, 0.5), (0, -4), "center", "bottom")       # sits across the line
    clear = Candidate((0.5, 0.5), (0, 12), "center", "bottom")
    ann, score = place_label(ax, "label", [on_line, clear], obstacles=obstacles, placed=[], fontsize=6)
    assert score == 0 and ann.xyann == (0, 12)


def test_place_label_prefers_the_earlier_candidate_when_both_are_clear():
    fig, ax, x = _axes()
    first = Candidate((0.5, 0.5), (0, 12), "center", "bottom")
    second = Candidate((0.5, 0.5), (0, -12), "center", "top")
    ann, score = place_label(ax, "label", [first, second], obstacles=np.empty((0, 2)), placed=[], fontsize=6)
    assert score == 0 and ann.xyann == (0, 12)


def test_place_label_refuses_to_overlap_a_label_already_placed():
    fig, ax, x = _axes()
    placed: list = []
    place_label(ax, "first", [Candidate((0.5, 0.8), (0, 0), "center", "center")],
                obstacles=np.empty((0, 2)), placed=placed, fontsize=6)
    ann, score = place_label(
        ax, "second",
        [Candidate((0.5, 0.8), (0, 0), "center", "center"), Candidate((0.5, 0.2), (0, 0), "center", "center")],
        obstacles=np.empty((0, 2)), placed=placed, fontsize=6)
    assert score == 0 and ann.xy == (0.5, 0.2)


def test_place_label_penalises_leaving_the_axes():
    fig, ax, x = _axes()
    off = Candidate((1.0, 0.5), (60, 0), "left", "center")
    _, score = place_label(ax, "label", [off], obstacles=np.empty((0, 2)), placed=[], fontsize=6)
    assert score >= 500


def test_place_existing_moves_a_reference_label():
    fig, ax, x = _axes()
    style.reference_line(ax, 0.5, "chance = 0.5")
    label = next(t for t in ax.texts if t.get_text() == "chance = 0.5")
    obstacles = curve_obstacles(ax, [(x, 0.5 + 0 * x)])
    cands = [Candidate((1.0, 0.5), (-2, -1), "right", "center", ("axes fraction", "data")),   # on the line
             Candidate((0.0, 0.5), (2, 8), "left", "bottom", ("axes fraction", "data"))]
    score = place_existing(ax, label, cands, obstacles=obstacles, placed=[])
    assert score == 0 and label.get_ha() == "left"


def test_place_label_needs_a_candidate():
    fig, ax, x = _axes()
    with pytest.raises((ValueError, IndexError)):
        place_label(ax, "label", [], obstacles=np.empty((0, 2)), placed=[])
