"""S4's two extra estimators: scale, agreement with Figure 2D, and the guards.

The important test here is :func:`test_softmax_recovers_a_known_exponent`. The softmax slope
is quoted in the caption as a power-law exponent comparable with the alpha of the sweep
figure, and that claim is only true if the fit is in natural log on both sides. Generating
data from a known exponent and recovering it pins the scale, so a later change of log base
anywhere in the chain fails here instead of silently rescaling a published number by 2.3.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

import make_sD_partial_correlation as sD
from mutacc import linkfit
from mutacc.guide import REPO_ROOT, load_guide

G5 = load_guide(REPO_ROOT / "guides" / "h3n2_5lineage.yaml")
BASE = next(m.key for m in G5.models if m.role == "base")
FT = next(m.key for m in G5.models if m.role == "finetune")
CONTROL = "neg_grantham"
TOL = 6e-4


@pytest.fixture(scope="module")
def one_lineage():
    frame = G5.read_table("replacement_scores")
    return frame[frame["lineage"].astype(str) == G5.primary_lineage]


@pytest.fixture(scope="module")
def fits(one_lineage):
    return linkfit.link_table(one_lineage, [FT, BASE], CONTROL, n_boot=30)


def est(table, measure, series, stratum, control="", col="estimate"):
    row = table[(table.measure == measure) & (table.series == series)
                & (table.stratum == stratum) & (table.control == control)]
    assert len(row) == 1
    return float(row[col].iloc[0])


# ---------------------------------------------------------------------------
# The scale of the softmax exponent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.65, 1.0, -0.4])
def test_softmax_recovers_a_known_exponent(alpha):
    """Data generated as q proportional to A**alpha must come back with beta = alpha.

    Site intercepts are deliberately large and random: the fit has to cancel them, since a
    model that is simply more confident at conserved sites must not be read as one that
    weights accessibility more heavily there.
    """
    rng = np.random.default_rng(11)
    n_sites, n_aa = 300, 19
    gid = np.repeat(np.arange(n_sites), n_aa)
    log_acc = rng.normal(-20.0, 4.0, n_sites * n_aa)
    eta = alpha * log_acc + np.repeat(rng.normal(0.0, 3.0, n_sites), n_aa)
    q = np.exp(eta - eta.max())
    assert linkfit.softmax_beta(q, log_acc, gid) == pytest.approx(alpha, abs=1e-6)


def test_softmax_is_invariant_to_the_log_base_of_the_probabilities():
    """Rescaling every probability at a site cannot move the exponent.

    The fit conditions on the site, so a site-wise constant factor is exactly what it is
    built to ignore. If this ever fails, the site intercepts are not cancelling.
    """
    rng = np.random.default_rng(3)
    n_sites, n_aa = 120, 19
    gid = np.repeat(np.arange(n_sites), n_aa)
    log_acc = rng.normal(-18.0, 3.0, n_sites * n_aa)
    q = np.exp(0.5 * log_acc + rng.normal(0.0, 1.0, n_sites * n_aa))
    scaled = q * np.repeat(rng.uniform(1e-6, 1e3, n_sites), n_aa)
    assert linkfit.softmax_beta(scaled, log_acc, gid) == pytest.approx(
        linkfit.softmax_beta(q, log_acc, gid), abs=1e-6)


def test_a_flat_model_has_no_exponent():
    """A model with the same probability everywhere must return beta = 0, not a sign."""
    rng = np.random.default_rng(5)
    gid = np.repeat(np.arange(100), 19)
    log_acc = rng.normal(-20.0, 4.0, 100 * 19)
    assert linkfit.softmax_beta(np.ones(100 * 19), log_acc, gid) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The spline basis
# ---------------------------------------------------------------------------

def test_spline_is_linear_outside_the_outer_knots():
    """"Restricted" means linear beyond the end knots; that is what stops the control
    inventing a steep chemistry effect where the data are thin."""
    knots = np.array([10.0, 60.0, 120.0, 190.0])
    x = np.array([-80.0, -40.0, 0.0, 240.0, 300.0, 360.0])
    basis = linkfit.rcs_basis(x, knots)
    assert np.allclose(basis[:, 0], x)
    for col in range(1, basis.shape[1]):
        lo = np.diff(basis[:3, col])
        hi = np.diff(basis[3:, col])
        assert np.allclose(lo, lo[0]) and np.allclose(hi, hi[0])


def test_spline_rejects_unsorted_or_too_few_knots():
    with pytest.raises(ValueError, match="increasing"):
        linkfit.rcs_basis(np.arange(10.0), np.array([5.0, 1.0, 9.0]))
    with pytest.raises(ValueError, match="increasing"):
        linkfit.rcs_basis(np.arange(10.0), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# The matched-rank estimator
# ---------------------------------------------------------------------------

def test_matched_rank_equals_plain_spearman_when_every_row_shares_one_bin():
    """With a single bin there is nothing to match on, so it must reduce to the plain r."""
    rng = np.random.default_rng(7)
    q = rng.random(400)
    acc = q + rng.normal(0, 0.4, 400)
    edges = np.array([-np.inf, np.inf])
    assert linkfit.matched_rank(q, acc, np.zeros(400), edges) == pytest.approx(
        spearmanr(q, acc).statistic, abs=1e-9)


def test_matched_rank_removes_a_link_that_is_entirely_between_bins():
    """A correlation created only by the control must vanish once bins are matched.

    Accessibility and the model are both pure functions of Grantham here plus noise that is
    independent within a bin, so there is real correlation overall and none within a bin.
    """
    rng = np.random.default_rng(13)
    grantham = rng.integers(0, 5, 2000).astype(float) * 50.0
    q = grantham + rng.normal(0, 1.0, 2000)
    acc = grantham + rng.normal(0, 1.0, 2000)
    edges = np.array([-np.inf, 25.0, 75.0, 125.0, 175.0, np.inf])
    assert spearmanr(q, acc).statistic > 0.8
    assert abs(linkfit.matched_rank(q, acc, grantham, edges)) < 0.1


# ---------------------------------------------------------------------------
# Against the real table
# ---------------------------------------------------------------------------

def test_the_two_modules_compute_the_same_plain_correlation(one_lineage, fits):
    """linkfit rebuilds the Figure 2D correlation from its own arrays. The two routes
    through the data must land on the same number, or the strata do not line up."""
    from mutacc import strata
    ranks = strata.strata_table(one_lineage, [FT, BASE], [CONTROL], n_boot=0)
    for key in (FT, BASE):
        for stratum in ("all", "1", "2", "3"):
            mine = est(fits, "spearman", key, stratum)
            theirs = ranks[(ranks.measure == "plm_vs_access") & (ranks.series == key)
                           & (ranks.stratum == stratum) & (ranks.control == "")]
            assert mine == pytest.approx(float(theirs.estimate.iloc[0]), abs=TOL)


def test_every_stratum_including_three_changes_is_estimated(fits):
    """The three-change group is reported, not dropped. Normalising within the group is
    what makes that possible: a single softmax over all nineteen replacements would starve
    it, because three-change options carry almost no probability at a site."""
    for stratum in ("all", "1", "2", "3"):
        for key in (FT, BASE):
            for control in ("", CONTROL):
                value = est(fits, "softmax_beta", key, stratum, control)
                assert np.isfinite(value)
                assert np.isfinite(est(fits, "softmax_beta", key, stratum, control, "lo"))


def test_the_fine_tuned_model_keeps_its_exponent_under_the_chemistry_control(fits):
    """The claim S4 exists to support: the fine-tuned link survives chemistry on the
    likelihood route as well as the rank route, at one and two nucleotide changes."""
    # The exponents, pinned to digits (deterministic: identical across repeat
    # fits, 15 Oct 2026). This replaces ``controlled == approx(plain, abs=0.08)``,
    # which read as "the control barely moves it" and was not true: at two
    # nucleotide changes the exponent *rises* by 26% under the chemistry control
    # (0.2029 -> 0.2548), and a band of 0.08 hid that. The claim the figure
    # actually supports is the one below it -- the controlled interval excludes
    # zero -- and it is now stated without the misleading companion.
    expected = {"1": (0.33495, 0.33074), "2": (0.20292, 0.25481)}
    for stratum in ("1", "2"):
        plain = est(fits, "softmax_beta", FT, stratum)
        controlled = est(fits, "softmax_beta", FT, stratum, CONTROL)
        assert est(fits, "softmax_beta", FT, stratum, CONTROL, "lo") > 0
        assert plain == pytest.approx(expected[stratum][0], abs=5e-5)
        assert controlled == pytest.approx(expected[stratum][1], abs=5e-5)


def test_link_table_refuses_more_than_one_lineage():
    frame = G5.read_table("replacement_scores")
    with pytest.raises(ValueError, match="one lineage"):
        linkfit.link_table(frame, [FT, BASE], CONTROL, n_boot=0)


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

def test_sD_partial_correlation_renders_both_rows_and_says_where_the_estimators_disagree(tmp_path):
    out = tmp_path / "o"
    assert sD.main(["--guide", str(G5.source), "--out", str(out), "--n-boot", "10"]) == 0
    assert (out / f"{sD.STEM}.png").stat().st_size > 10_000
    values = pd.read_csv(out / f"{sD.STEM}_values.tsv", sep="\t")
    assert set(values.control.fillna("")) == {"", CONTROL}
    assert set(values.measure) == {"plm_vs_access", "matched_rank", "softmax_beta"}
    assert set(values.n_lineages) == {1}
    # The published partial correlation must not have moved.
    assert est(values, "plm_vs_access", FT, "3", CONTROL) == pytest.approx(0.184127, abs=TOL)
    text = " ".join((out / f"{sD.STEM}_caption.md").read_text().split())
    assert "Grantham" in text and "has been removed" in text
    # The caption must keep saying the two things it would be most tempting to drop.
    assert "not the same quantity" in text
    assert "opposite signs" in text
