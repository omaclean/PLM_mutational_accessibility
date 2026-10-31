"""Figure SP draws the curves behind Figure SM's optimum counts.

The figure exists because the counts moved: Figure SM dropped arms per stratum, so its
denominators differed between strata for reasons no reader could see. These tests pin the
two rules that changed with it -- receptor-binding assays out, sweep-end optima kept -- and
the agreement between the two figures' optima, which is the whole point of drawing one
against the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mutacc.dms_sweep import WEAK_SIGNAL_RHO, curve_flags, curve_has_signal
from mutacc.guide import REPO_ROOT

make_sP = pytest.importorskip("make_sP_alpha_curves")

OUTPUT = REPO_ROOT / "output"
STEM = make_sP.STEM


@pytest.fixture(scope="module")
def values() -> pd.DataFrame:
    path = OUTPUT / f"{STEM}_values.tsv"
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    return pd.read_csv(path, sep="\t")


# ---------------------------------------------------------------------------
# curve_has_signal: the rule that replaced "|rho| at alpha = 0"
# ---------------------------------------------------------------------------

def test_a_curve_that_peaks_away_from_zero_is_not_called_flat():
    """The case that forced the rule: BA.2 cell entry at 3 nt reads 0.042 at alpha = 0 and
    0.084 at its optimum. ``curve_flags`` calls that weak; it plainly is not."""
    alphas = np.linspace(-1.0, 1.0, 201)
    rho = 0.042 + 0.042 * (1.0 - (alphas + 0.9) ** 2 / 0.81).clip(0.0)
    assert curve_flags(rho, alphas)["weak_signal"] is True
    assert curve_has_signal(rho) is True


def test_a_curve_that_is_flat_everywhere_is_still_called_flat():
    """The weaker rule must not rescue a curve with nothing anywhere in the sweep."""
    rho = np.full(201, 0.01)
    assert curve_has_signal(rho) is False
    assert curve_has_signal(np.full(201, -0.01)) is False


def test_the_threshold_is_the_shared_one_and_is_inclusive():
    assert curve_has_signal(np.array([0.0, WEAK_SIGNAL_RHO])) is True
    assert curve_has_signal(np.array([0.0, WEAK_SIGNAL_RHO - 1e-9])) is False
    assert curve_has_signal(np.array([0.0, -WEAK_SIGNAL_RHO])) is True, "sign must not matter"


def test_an_all_nan_curve_raises_rather_than_passing():
    with pytest.raises(ValueError):
        curve_has_signal(np.full(5, np.nan))


# ---------------------------------------------------------------------------
# The two rules that distinguish this figure from the one it explains
# ---------------------------------------------------------------------------

def test_receptor_binding_assays_are_drawn_but_not_counted(values):
    """Their curves are the evidence for excluding them, so they are plotted; a signed-rho
    optimum is not defined for a phenotype measured against fitness, so they are not tallied."""
    binding = values.loc[
        values["trait"].str.contains(make_sP.RECEPTOR_TRAIT, case=False, na=False)]
    assert not binding.empty, "the receptor-binding panels should be drawn, not dropped"
    assert not binding["counted"].any(), (
        "receptor-binding assays must stay out of the tallies: "
        f"{sorted(set(binding.loc[binding['counted'], 'panel_label']))}")
    rest = values.loc[~values.index.isin(binding.index)]
    assert rest["counted"].all(), "a non-binding assay was excluded from the tallies"


def test_the_tallies_cover_only_the_counted_arms(values):
    """counts() must filter to `counted`, or Figure SM panel J and this figure disagree."""
    tally = make_sP.counts(values)
    counted = values.loc[values["counted"]]
    for group in (make_sP.POOLED, *(str(s) for s in make_sP.STRATA)):
        assert tally[group]["arms"] == int((counted["group"] == group).sum())


def test_a_sweep_end_optimum_is_marked_and_sits_at_the_edge(values):
    """A bound is still a result when the curve runs to it, so it is marked, not dropped."""
    at_end = values.loc[values["at_sweep_end"]]
    assert not at_end.empty, "no sweep-end optima left to check; has the grid widened?"
    assert at_end["peak_alpha"].abs().eq(1.0).all()


def test_nothing_is_excluded_from_the_tallies(values):
    """The figure counts every arm. An earlier version dropped weak and sweep-end fits, which
    made the denominator move between strata for a reason no reader of the count could see."""
    tally = make_sP.counts(values)
    counted = values.loc[values["counted"]]
    for group in (make_sP.POOLED, *(str(s) for s in make_sP.STRATA)):
        drawn = int((counted["group"] == group).sum())
        assert tally[group]["arms"] == drawn, (
            f"{group}: {tally[group]['arms']} counted of {drawn} drawn -- something is "
            "being filtered out of the tally again")
    assert len({tally[g]["arms"] for g in
                (make_sP.POOLED, *(str(s) for s in make_sP.STRATA))}) == 1, \
        "the per-group denominators differ, so some arm is missing a fit"


def test_the_weak_flag_is_set_on_exactly_the_curves_that_never_reach_the_threshold(values):
    """``has_signal`` is information for the reader, not a filter -- but it must be right."""
    for _, row in values.iterrows():
        expected = row["max_abs_rho"] >= WEAK_SIGNAL_RHO
        assert bool(row["has_signal"]) == expected, (
            f"{row['panel_label']} / {row['role']} / {row['group']}: has_signal="
            f"{row['has_signal']} with |rho| peaking at {row['max_abs_rho']:.3f}")


def test_a_marker_is_suppressed_only_when_the_sweep_found_nothing(values):
    """The rule the user caught: of the one arm that is weak throughout, the 1 nt curve falls
    steadily to the far edge and must keep its marker, while the pooled and 2 nt curves are
    flat and must not get one. Weakness alone never suppresses a marker."""
    for _, row in values.iterrows():
        expected = bool(row["counted"]
                        and (row["has_signal"] or row["gain"] >= make_sP.MARKED_GAIN))
        assert bool(row["has_optimum"]) == expected, (
            f"{row['panel_label']} / {row['role']} / {row['group']}: has_optimum="
            f"{row['has_optimum']} with has_signal={row['has_signal']} gain={row['gain']:.5f}")
    lost = values.loc[~values["has_optimum"] & values["counted"]]
    assert (~lost["has_signal"]).all(), "a counted curve with real signal lost its marker"
    assert (lost["gain"] < make_sP.MARKED_GAIN).all()
    assert not values.loc[~values["counted"], "has_optimum"].any(), \
        "a receptor-binding arm was given an optimum; none is defined for those assays"


def test_an_unmarked_fit_is_still_counted(values):
    """Suppressing a marker must not quietly shrink a denominator -- that is the failure this
    whole figure was built to expose."""
    tally = make_sP.counts(values)
    counted = values.loc[values["counted"]]
    for group in (make_sP.POOLED, *(str(s) for s in make_sP.STRATA)):
        assert tally[group]["arms"] == int((counted["group"] == group).sum())


def test_a_weak_curve_still_contributes_its_optimum(values):
    """The case the user caught: the arm flagged weak has perfectly locatable optima
    (0.00 pooled, -1.00 at 1 nt, +0.02 at 2 nt) and must not be silently discarded."""
    weak = values.loc[~values["has_signal"]]
    if weak.empty:
        pytest.skip("no weak curves in the shipped set")
    assert weak["peak_alpha"].notna().all()
    tally = make_sP.counts(values)
    for group in weak["group"].unique():
        assert tally[group]["weak"] >= 1
        assert tally[group]["arms"] > tally[group]["weak"], \
            "a whole group is weak; the tally would be meaningless"


# ---------------------------------------------------------------------------
# Agreement with the figure it explains
# ---------------------------------------------------------------------------

def test_the_optima_match_the_within_stratum_table(values):
    """Both figures sweep the same grid over the same cells, so an arm they share must get
    the same optimum. If this fails the two figures are telling different stories."""
    path = OUTPUT / "sM_peak_alpha_by_stratum_values.tsv"
    if not path.exists():
        pytest.skip("sM_peak_alpha_by_stratum_values.tsv not built")
    other = pd.read_csv(path, sep="\t")
    other["group"] = other["group"].astype(str)
    key = ["panel_label", "model_name", "group"]
    merged = values.merge(other[key + ["peak_alpha"]], on=key, how="inner",
                          suffixes=("_here", "_sM"))
    assert len(merged) == len(values), (
        f"{len(values) - len(merged)} arm-groups drawn here have no row in Figure SM")
    gap = (merged["peak_alpha_here"] - merged["peak_alpha_sM"]).abs().max()
    assert gap <= make_sP.ALPHA_STEP / 2, f"optima disagree by up to {gap}"


def test_every_panel_carries_both_models(values):
    """A row of the figure is a paired comparison; a panel with one model is a broken join."""
    per_panel = values.groupby("panel_label")["role"].nunique()
    assert (per_panel == 2).all(), f"panels missing a model: {per_panel[per_panel != 2].index.tolist()}"


def test_the_pooled_fit_and_three_strata_are_all_present(values):
    groups = values.groupby(["panel_label", "role"])["group"].apply(set)
    expected = {make_sP.POOLED, *(str(s) for s in make_sP.STRATA)}
    short = {k: sorted(expected - v) for k, v in groups.items() if expected - v}
    assert not short, f"arms missing a fit: {short}"


def test_the_caption_counts_are_the_tallies_the_table_supports(values):
    """The caption quotes 'N of M' per stratum and per model; recompute them from the table."""
    tally = make_sP.counts(values)
    counted = values.loc[values["counted"]]
    for key, row in tally.items():
        if key == "within":
            subset = counted.loc[counted["group"] != make_sP.POOLED]
        elif ":" in key:
            group, role = key.split(":")
            subset = counted.loc[(counted["group"] == group) & (counted["role"] == role)]
        else:
            subset = counted.loc[counted["group"] == key]
        assert row["arms"] == len(subset), key
        assert row["negative"] == int((subset["peak_alpha"] < 0).sum()), key
        assert row["negative"] <= row["arms"], key


def test_the_two_models_are_tallied_separately(values):
    """The pooled fit separates the models completely, so a median over both lands in the gap
    between them. The caption reports them apart; this pins that the split exists."""
    tally = make_sP.counts(values)
    for role in values["role"].unique():
        assert f"{make_sP.POOLED}:{role}" in tally, f"no pooled tally for {role}"
