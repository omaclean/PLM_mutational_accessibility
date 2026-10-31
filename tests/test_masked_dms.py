"""``mutacc.masked_dms``: the masked marginal every DMS figure now scores with.

Since 19 Oct 2026 Figure 4, Figure SL and Figure SM read the model's probability for a
replacement from ``dms_scores_masked`` rather than from ``dms_scores.log_plm``. Two things
have to hold for that to mean anything, and both are tested here:

* the join is **one-to-one and order-preserving**. The site bootstrap in those figures
  consumes its random draws arm by arm in table order, so a reordered or duplicated join
  would move every band while leaving the curves correct -- a difference that would look
  like a result and would never be caught by eye.
* the unmasked arm still reproduces the shipped ``log_plm``. That is the gate the whole
  switch rests on: it is what makes "masking is the only thing that changed" a measured
  statement rather than an assumption.

The profile CSVs the table is built from need a GPU to produce but not to read, so the
reader is tested here against the real shipped ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mutacc.guide import REPO_ROOT, load_guide
from mutacc.masked_dms import (
    AA20, MASKED_COLUMN, PROFILE_LABEL, UNMASKED_TOL, apply_masked, build_masked_column,
    profile_log_plm,
)

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
PROFILES = REPO_ROOT / "output" / "unmasked_check"


@pytest.fixture(scope="module")
def scores():
    return GUIDE.read_table("dms_scores")


@pytest.fixture(scope="module")
def masked():
    return GUIDE.read_table("dms_scores_masked")


# ---------------------------------------------------------------------------
# The shipped table
# ---------------------------------------------------------------------------

def test_the_masked_table_keys_one_to_one_against_the_scores(scores, masked):
    key = ["dataset", "model_name", "site", "aa"]
    assert not masked.duplicated(key).any()
    assert len(masked) == len(scores)
    assert set(map(tuple, masked[key].to_numpy())) == set(map(tuple, scores[key].to_numpy()))


def test_every_masked_probability_is_a_log_of_a_probability(masked):
    values = masked[MASKED_COLUMN].to_numpy(float)
    assert np.isfinite(values).all()
    assert values.max() <= 0.0, "log10 of a probability cannot be positive"


def test_the_masked_column_is_not_the_unmasked_one(scores, masked):
    """A table accidentally built from the unmasked profiles would pass every other check."""
    swapped = apply_masked(scores, masked)
    moved = (swapped["log_plm"].to_numpy(float) - scores["log_plm"].to_numpy(float))
    assert np.abs(moved).max() > UNMASKED_TOL * 100


def test_masking_lifts_the_fine_tuned_arm_more_in_every_fitness_panel():
    """The reason for the switch, as a number, on the quantity the figures actually plot.

    Shown the residue it is scoring, a model fine-tuned on one family can largely copy it,
    which squeezes the probability left for the alternatives -- and the ranking of those
    alternatives is what every one of these figures scores. Masking should therefore help the
    fine-tuned arm more than the untuned one. It is not a claim about how far the individual
    probabilities move: those shift by about the same amount for both (median 0.40 in log10),
    which is why this is checked on rho.

    Receptor binding is left out: those correlations are negative, so "higher rho" is not
    "better" there and the comparison would not mean the same thing.

    The gain is read at alpha = 0, the model alone, because that is the arm an estimator can
    flatter; with accessibility weighted in, part of the score is not the model's at all.
    """
    curves = GUIDE.read_table("masked_dms_curves")
    at_zero = curves.loc[curves["alpha"].abs() < 1e-9]
    wide = at_zero.pivot_table(index=["panel_label", "model_role"], columns="scoring",
                               values="rho")
    gain = (wide["masked"] - wide["published"]).unstack("model_role")
    fitness = [p for p in gain.index if "binding" not in p.lower()]
    assert len(fitness) >= 7
    losers = [p for p in fitness if not gain.loc[p, "finetune"] > gain.loc[p, "base"]]
    assert not losers, f"masking did not favour the fine-tuned arm in {losers}"


# ---------------------------------------------------------------------------
# apply_masked: the guards that protect the bootstrap
# ---------------------------------------------------------------------------

def test_apply_masked_preserves_row_order(scores, masked):
    shuffled = masked.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = apply_masked(scores, shuffled)
    assert out.index.equals(scores.index)
    for column in ("dataset", "model_name", "site", "aa"):
        assert out[column].equals(scores[column])
    # and the values follow their own keys, not the shuffled order
    assert out["log_plm"].equals(apply_masked(scores, masked)["log_plm"])


def test_a_duplicated_key_is_refused(scores, masked):
    doubled = pd.concat([masked, masked.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        apply_masked(scores, doubled)


def test_a_table_missing_rows_is_refused(scores, masked):
    with pytest.raises(ValueError, match="stale"):
        apply_masked(scores, masked.iloc[1:])


def test_a_table_missing_a_column_is_refused(scores, masked):
    with pytest.raises(ValueError, match="lacks column"):
        apply_masked(scores, masked.drop(columns=[MASKED_COLUMN]))


# ---------------------------------------------------------------------------
# The profiles and the gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(PROFILE_LABEL), ids=lambda k: f"{k[0]}:{k[1]}")
def test_each_arm_has_both_profiles(key):
    dataset, label = key[0], PROFILE_LABEL[key]
    for kind in ("masked", "unmasked"):
        assert (PROFILES / f"{dataset}__{label}__{kind}_profile.csv").exists()


def test_profile_rows_are_the_twenty_residues_and_columns_are_one_indexed_sites(scores):
    dataset, model_name = next(iter(PROFILE_LABEL))
    label = PROFILE_LABEL[(dataset, model_name)]
    frame = pd.read_csv(PROFILES / f"{dataset}__{label}__masked_profile.csv", index_col=0)
    assert sorted(frame.index) == sorted(AA20)
    assert list(frame.columns.astype(int)) == list(range(1, frame.shape[1] + 1))


def test_a_site_outside_the_protein_is_refused():
    dataset, model_name = next(iter(PROFILE_LABEL))
    label = PROFILE_LABEL[(dataset, model_name)]
    with pytest.raises(ValueError, match="outside"):
        profile_log_plm(PROFILES, dataset, label, "masked",
                        np.array([10_000]), np.array(["A"]))


def test_an_unknown_replacement_is_refused():
    dataset, model_name = next(iter(PROFILE_LABEL))
    label = PROFILE_LABEL[(dataset, model_name)]
    with pytest.raises(ValueError, match="not rows of"):
        profile_log_plm(PROFILES, dataset, label, "masked", np.array([1]), np.array(["X"]))


def test_a_missing_profile_says_how_to_build_it():
    with pytest.raises(FileNotFoundError, match="embed_masked_profile"):
        profile_log_plm(PROFILES, "NoSuchBackbone", "NoSuchModel", "masked",
                        np.array([1]), np.array(["A"]))


def test_the_gate_reproduces_the_shipped_log_plm(scores):
    """The whole switch rests on this: recomputed unmasked == published."""
    _, worst = build_masked_column(scores, PROFILES)
    assert worst < UNMASKED_TOL


def test_the_gate_fails_on_a_tampered_score(scores):
    bad = scores.copy()
    bad.loc[bad.index[0], "log_plm"] -= 1.0
    with pytest.raises(ValueError, match="not be like for like"):
        build_masked_column(bad, PROFILES)


def test_the_rebuilt_column_equals_the_shipped_table(scores, masked):
    """The shipped table is not stale against the profiles it was built from."""
    rebuilt, _ = build_masked_column(scores, PROFILES)
    key = ["dataset", "model_name", "site", "aa"]
    merged = rebuilt.merge(masked, on=key, how="outer", suffixes=("_now", "_shipped"))
    assert len(merged) == len(masked)
    gap = (merged[f"{MASKED_COLUMN}_now"] - merged[f"{MASKED_COLUMN}_shipped"]).abs().max()
    assert gap < 1e-9


# --------------------------------------------------------------------------------------
# Which estimator the reported numbers come from.
#
# ``dms_peak_summary`` still ships, and must: it is the unmasked fixed point every switched
# figure's gate checks against. But its ``rho_alpha0`` / ``peak_alpha`` / ``peak_rho``
# columns are that same unmasked estimator, and on MA22 pH stability the two estimators
# disagree *in sign* about which model leads at alpha = 0. Quoting the archive table for a
# number the manuscript takes off Figure 4 therefore reverses a Results sentence. These
# tests pin the disagreement -- so it cannot be "fixed" by a silent edit to one table --
# and pin that Table SY, which is what the manuscript quotes, is the masked arm.
# --------------------------------------------------------------------------------------

#: Rounding in the shipped TSVs, not analysis error: Table SY is written to 6 decimals.
REPORTED_TOL = 1e-5

#: The panel the two estimators disagree in sign about. Values at alpha = 0.
PH_STABILITY = "MA22 pH stability"
PH_UNMASKED = {"base": 0.137807, "finetune": 0.126771}
PH_MASKED = {"base": 0.121007, "finetune": 0.138381}

ARM_KEY = ["panel_label", "model_role"]


@pytest.fixture(scope="module")
def curves():
    """``masked_dms_curves``: the Figure 4 sweep under both estimators, every arm."""
    return GUIDE.read_table("masked_dms_curves")


@pytest.fixture(scope="module")
def arms(curves):
    cols = ARM_KEY + ["best_alpha", "rho_best", "rho_alpha0"]
    return curves.drop_duplicates(["scoring"] + ARM_KEY)[["scoring"] + cols]


@pytest.fixture(scope="module")
def table_sy():
    """Table SY, written off the installed Figure 4 and Figure SL values tables."""
    path = REPO_ROOT / "output" / "sY_dms_alpha_optima.tsv"
    if not path.exists():
        pytest.skip(f"{path.name} is built by tools/export_sY_dms_alpha_optima.py")
    return pd.read_csv(path, sep="\t")


def test_the_reported_table_is_the_masked_arm(table_sy, arms):
    """Every row of Table SY equals the masked sweep, for all 18 arms of both viruses.

    Both the optimum and the height at alpha = 0, because the Results text quotes both.
    """
    masked = arms[arms["scoring"] == "masked"]
    merged = table_sy.merge(masked, on=ARM_KEY, validate="one_to_one")
    assert len(merged) == len(table_sy) == 18
    assert (merged["alpha_best"] == merged["best_alpha"]).all()
    for reported, swept in (("rho_alpha0_x", "rho_alpha0_y"), ("rho_best_x", "rho_best_y")):
        assert (merged[reported] - merged[swept]).abs().max() < REPORTED_TOL


def test_the_reported_table_is_not_the_unmasked_arm(table_sy, arms):
    """The converse, so the test above cannot pass on two tables that have converged.

    If masking ever stopped mattering this fails, which is the right way round: it would be
    a finding, not a green suite.
    """
    unmasked = arms[arms["scoring"] == "published"]
    merged = table_sy.merge(unmasked, on=ARM_KEY, validate="one_to_one")
    assert (merged["rho_alpha0_x"] - merged["rho_alpha0_y"]).abs().max() > 0.01


def test_the_archive_summary_holds_the_unmasked_estimator():
    """``dms_peak_summary`` is the gate's fixed point, so it stays unmasked."""
    summary = GUIDE.read_table("dms_peak_summary")
    ph = summary[(summary["panel_label"] == PH_STABILITY)
                 & (summary["spectrum"] == "published")]
    got = dict(zip(ph["model_role"], ph["rho_alpha0"]))
    assert set(got) == set(PH_UNMASKED)
    for role, expected in PH_UNMASKED.items():
        assert abs(got[role] - expected) < REPORTED_TOL


def test_the_two_estimators_disagree_in_sign_on_ph_stability(arms, table_sy):
    """Pinned, not reconciled: it is the estimator change, and it reverses the ordering.

    Unmasked puts the fine-tune below the base at alpha = 0; masked puts it above. The
    manuscript reports the masked ordering, so Table SY must carry that one.
    """
    unmasked = arms[(arms["scoring"] == "published") & (arms["panel_label"] == PH_STABILITY)]
    masked = arms[(arms["scoring"] == "masked") & (arms["panel_label"] == PH_STABILITY)]
    u = dict(zip(unmasked["model_role"], unmasked["rho_alpha0"]))
    m = dict(zip(masked["model_role"], masked["rho_alpha0"]))
    assert u["finetune"] < u["base"], "the unmasked arm is the one the archive shipped"
    assert m["finetune"] > m["base"], "the masked arm is the one Figure 4 draws"

    reported = table_sy[table_sy["panel_label"] == PH_STABILITY]
    r = dict(zip(reported["model_role"], reported["rho_alpha0"]))
    assert r["finetune"] > r["base"]
    for role, expected in PH_MASKED.items():
        assert abs(r[role] - expected) < REPORTED_TOL
