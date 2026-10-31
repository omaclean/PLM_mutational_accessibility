"""Leave-one-lineage-out selection of the accessibility weight (``mutacc.loo_alpha``).

The module exists to answer one objection -- that Figure 3's alpha is chosen on the data it
is scored on -- and it answers it only if the held-out lineage really takes no part in the
choice. That is a property no amount of plausible-looking output can demonstrate, so it is
asserted directly here, on data built so that a leak would move ``alpha_star`` to a weight
that can only have come from the held-out fold.

The rest pins the conventions a reader would otherwise have to infer from the numbers: the
reference weight is exactly alpha = 0 and gains measured there are exactly zero, ties are
broken toward zero rather than by ``argmax``'s first index, the accessibility-only baseline
rows (which carry no alpha at all) never reach the arithmetic, and every lineage is held out
once and only once.

There is deliberately no p-value anywhere in the module, the exporter or its caption, and
the last test in this file fails if one appears: the thirteen lineages are not independent
replicates, so a sign test or a paired t-test over the folds is anti-conservative, and 13 of
13 is a sign test's floor whatever the effect size.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tokenize
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mutacc.guide import REPO_ROOT, load_guide
from mutacc.loo_alpha import (
    ALPHA_REFERENCE, FOLD, MEAN, fold_summary, fold_table, loo_alpha_table, select_alpha,
)
from mutacc.metrics import lineage_matrix

sys.path.insert(0, str(REPO_ROOT / "tools"))
import export_sW_loo_alpha as sW  # noqa: E402

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
OUTPUT = REPO_ROOT / "output"


def sweep(values, alphas, *, model="m", variant="plm_alpha_sweep", lineages=None):
    """A sweep table in the shape the shipped one has: one row per (lineage, alpha)."""
    lineages = lineages or [f"L{i}" for i in range(len(values))]
    rows = [
        {"lineage": lin, "alpha": a, "model": model, "model_variant": variant,
         "is_mutation_only_baseline": variant != "plm_alpha_sweep", "x": v}
        for lin, row in zip(lineages, values) for a, v in zip(alphas, row)
    ]
    return pd.DataFrame(rows)


def wide(values, alphas, **kwargs):
    return lineage_matrix(sweep(values, alphas, **kwargs), "x", model="m")


# ---------------------------------------------------------------------------
# The property the whole module exists for
# ---------------------------------------------------------------------------

def test_alpha_star_is_chosen_without_the_held_out_lineage():
    """Three lineages peak at alpha = 0, one peaks hugely at alpha = +1.

    The mean over all four peaks at +1, because the outlier is large enough to carry it.
    The mean over the other three -- which is what each fold must see when that outlier is
    the lineage held out -- peaks at 0. So the outlier's own fold selects +1 if and only if
    its curve leaked into the selection, and 0 if it did not.
    """
    alphas = [-1.0, 0.0, 1.0]
    quiet = [[0.0, 1.0, 0.0]] * 3
    outlier = [[0.0, 1.0, 99.0]]
    frame = wide(quiet + outlier, alphas, lineages=["a", "b", "c", "loud"])

    # The in-sample choice -- what Figure 3 makes -- is +1.
    assert select_alpha(alphas, frame.to_numpy(float).mean(axis=0)) == 1.0

    folds = fold_table(frame).set_index("held_out_lineage")
    assert folds.loc["loud", "alpha_star"] == 0.0, (
        "the held-out lineage's own curve chose its weight: the selection leaked"
    )
    # ...and the quiet folds, whose training set does contain the outlier, still choose +1.
    assert set(folds.loc[["a", "b", "c"], "alpha_star"]) == {1.0}


def test_the_training_curve_is_the_mean_of_exactly_the_other_lineages():
    """The stronger version: recompute each fold's selection by hand, from the rows the
    fold is allowed to see, and require the same weight."""
    rng = np.random.default_rng(11)
    alphas = np.round(np.linspace(-1.0, 1.0, 21), 2)
    values = rng.normal(size=(6, len(alphas)))
    frame = wide(values, alphas)
    folds = fold_table(frame)
    for i, row in enumerate(folds.itertuples(index=False)):
        others = np.delete(values, i, axis=0).mean(axis=0)
        assert row.alpha_star == select_alpha(alphas, others)
        assert row.held_out_lineage == frame.index[i]


def test_every_lineage_is_held_out_exactly_once_on_the_shipped_sweep():
    frame = GUIDE.read_table("alpha_sweep_by_lineage", GUIDE.results_dir)
    folds = fold_table(lineage_matrix(frame, "site_logistic_auroc", model=GUIDE.models[0].key))
    assert len(folds) == GUIDE.n_lineages == 13
    assert sorted(folds["held_out_lineage"]) == sorted(GUIDE.lineage_ids)


# ---------------------------------------------------------------------------
# The reference weight
# ---------------------------------------------------------------------------

def test_a_fold_that_selects_alpha_zero_gains_exactly_zero():
    """Not "close to zero": the gain is one table cell minus itself, so any drift here
    would mean the reference column and the selected column are not the same column."""
    alphas = [-0.5, 0.0, 0.5]
    frame = wide([[0.0, 1.0, 0.0]] * 4, alphas)
    folds = fold_table(frame)
    assert (folds["alpha_star"] == ALPHA_REFERENCE).all()
    assert (folds["gain"] == 0.0).all()
    assert (folds["metric_at_alpha_star"] == folds["metric_at_reference"]).all()


def test_the_gain_is_always_the_difference_of_the_two_columns_beside_it():
    rng = np.random.default_rng(3)
    alphas = np.round(np.linspace(-1.0, 1.0, 11), 2)
    folds = fold_table(wide(rng.normal(size=(5, len(alphas))), alphas))
    assert np.array_equal(folds["gain"],
                          folds["metric_at_alpha_star"] - folds["metric_at_reference"])


def test_a_grid_that_does_not_carry_alpha_zero_is_refused():
    """The gain is defined against the unweighted model. A grid that steps over zero has
    no such column, and silently taking the nearest one would redefine every gain."""
    with pytest.raises(ValueError, match="expected exactly 1"):
        fold_table(wide([[1.0, 2.0], [3.0, 4.0]], [-0.5, 0.5]))


def test_one_lineage_cannot_be_left_out_of_itself():
    with pytest.raises(ValueError, match="nothing to select on"):
        fold_table(wide([[0.0, 1.0, 0.0]], [-1.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# Ties
# ---------------------------------------------------------------------------

def test_a_tied_peak_resolves_to_the_weight_nearest_zero():
    """Adjacent weights that leave every ranking unchanged give identical AUROCs, so this
    is a real case, not a contrived one. ``np.argmax`` would answer -0.5 here."""
    alphas = np.array([-0.5, 0.0, 0.25, 1.0])
    curve = np.array([7.0, 3.0, 7.0, 7.0])
    assert select_alpha(alphas, curve) == 0.25
    assert np.argmax(curve) == 0            # what the rule is chosen over


def test_two_weights_equally_far_from_zero_resolve_downward():
    assert select_alpha(np.array([-0.4, 0.0, 0.4]), np.array([5.0, 1.0, 5.0])) == -0.4


def test_the_tie_rule_does_not_depend_on_the_order_the_grid_is_given_in():
    alphas = np.array([1.0, -0.5, 0.25, 0.0])
    curve = np.array([7.0, 7.0, 7.0, 3.0])
    assert select_alpha(alphas, curve) == 0.25


def test_the_number_of_tied_weights_is_reported():
    alphas = [-1.0, 0.0, 1.0]
    folds = fold_table(wide([[4.0, 1.0, 4.0]] * 3, alphas))
    assert (folds["n_tied_alpha"] == 2).all()
    assert (folds["alpha_star"] == -1.0).all()      # equally distant, so downward


def test_a_hole_in_the_curve_is_refused_rather_than_stepped_over():
    with pytest.raises(ValueError, match="non-finite"):
        select_alpha(np.array([-1.0, 0.0, 1.0]), np.array([1.0, np.nan, 2.0]))


# ---------------------------------------------------------------------------
# The baseline rows, which carry no alpha
# ---------------------------------------------------------------------------

def test_the_accessibility_only_rows_never_reach_the_arithmetic():
    """``is_mutation_only_baseline`` rows have ``alpha`` NaN and a variant of their own.
    Pivoting them in would put a NaN column on the grid and poison every mean."""
    alphas = [-1.0, 0.0, 1.0]
    frame = sweep([[0.0, 1.0, 2.0], [0.0, 1.0, 3.0]], alphas)
    baseline = sweep([[9.0], [9.0]], [np.nan], variant="mutation_accessibility_only")
    folds = fold_table(lineage_matrix(pd.concat([frame, baseline]), "x", model="m"))
    assert len(folds) == 2
    assert np.isfinite(folds[["alpha_star", "gain"]].to_numpy(float)).all()
    assert (folds["alpha_star"] == 1.0).all()


def test_the_shipped_sweep_has_baseline_rows_for_this_to_exclude():
    """Guard the test above against the day the shipped table stops carrying them, which
    would make it pass by having nothing to exclude."""
    frame = GUIDE.read_table("alpha_sweep_by_lineage", GUIDE.results_dir)
    baseline = frame[frame["is_mutation_only_baseline"].astype(bool)]
    assert len(baseline) == GUIDE.n_lineages * len(GUIDE.models)
    assert baseline["alpha"].isna().all()
    assert lineage_matrix(frame, "site_logistic_auroc",
                          model=GUIDE.models[0].key).notna().all().all()


# ---------------------------------------------------------------------------
# The summary over folds
# ---------------------------------------------------------------------------

def test_the_summary_counts_and_brackets_what_the_folds_hold():
    rng = np.random.default_rng(5)
    alphas = np.round(np.linspace(-1.0, 1.0, 11), 2)
    folds = fold_table(wide(rng.normal(size=(7, len(alphas))), alphas))
    summary = fold_summary(folds, n_boot=400, seed=1)
    assert summary["n_folds"] == len(folds)
    assert summary["n_folds_gain_positive"] == int((folds["gain"] > 0).sum())
    assert summary["mean_gain"] == pytest.approx(folds["gain"].mean())
    assert summary["ci_lo"] <= summary["mean_gain"] <= summary["ci_hi"]


def test_the_summary_is_seeded():
    alphas = np.round(np.linspace(-1.0, 1.0, 11), 2)
    rng = np.random.default_rng(6)
    folds = fold_table(wide(rng.normal(size=(5, len(alphas))), alphas))
    assert fold_summary(folds, n_boot=200, seed=2) == fold_summary(folds, n_boot=200, seed=2)


def test_the_long_table_carries_one_summary_row_per_metric_and_model():
    alphas = np.round(np.linspace(-1.0, 1.0, 11), 2)
    rng = np.random.default_rng(7)
    frame = sweep(rng.normal(size=(4, len(alphas))), alphas)
    frame["y"] = frame["x"] * 2.0
    table = loo_alpha_table(frame, ["m"], ["x", "y"], n_boot=200)
    assert len(table) == 2 * (4 + 1)
    assert (table[table["scope"] == MEAN].groupby(["metric", "model"]).size() == 1).all()
    assert table[table["scope"] == FOLD]["ci_lo"].isna().all()
    assert table[table["scope"] == MEAN]["ci_lo"].notna().all()
    assert table[table["scope"] == MEAN]["alpha_star"].isna().all()


# ---------------------------------------------------------------------------
# The exporter and the table it ships
# ---------------------------------------------------------------------------

VALUES = OUTPUT / f"{sW.STEM}_values.tsv"


@pytest.fixture(scope="module")
def shipped():
    return pd.read_csv(VALUES, sep="\t")


def test_the_shipped_table_covers_both_arms_both_metrics_and_both_models(shipped):
    assert set(shipped["scoring"]) == set(sW.ARMS)
    assert set(shipped["metric"]) == set(sW.METRICS)
    assert set(shipped["model"]) == {m.key for m in GUIDE.models}
    assert set(shipped["model_role"]) == {"base", "finetune"}
    folds = shipped[shipped["scope"] == FOLD]
    assert len(folds) == len(sW.ARMS) * len(sW.METRICS) * len(GUIDE.models) * GUIDE.n_lineages
    for key, group in folds.groupby(["scoring", "metric", "model"]):
        assert sorted(group["held_out_lineage"]) == sorted(GUIDE.lineage_ids), key


def test_the_shipped_summary_rows_agree_with_their_own_folds(shipped):
    """A summary joined onto the wrong fold group is the failure this catches."""
    folds = shipped[shipped["scope"] == FOLD]
    means = shipped[shipped["scope"] == MEAN]
    assert len(means) == len(sW.ARMS) * len(sW.METRICS) * len(GUIDE.models)
    for row in means.itertuples(index=False):
        mine = folds[(folds["scoring"] == row.scoring) & (folds["metric"] == row.metric)
                     & (folds["model"] == row.model)]["estimate"]
        # ``write_values`` writes %.6g, so a mean of written-out folds and the written-out
        # mean agree only to a unit in the sixth significant digit (build_figures.RTOL).
        assert row.estimate == pytest.approx(mine.mean(), rel=2e-5, abs=1e-9)
        assert row.n_folds == len(mine) == GUIDE.n_lineages
        assert row.n_folds_gain_positive == int((mine > 0).sum())
        assert row.ci_lo <= row.estimate <= row.ci_hi


def test_a_rebuild_reproduces_the_shipped_table(tmp_path):
    """The exporter is deterministic -- the one bootstrap in it is seeded -- so the
    shipped file must come back identical."""
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "tools/export_sW_loo_alpha.py"),
                           str(REPO_ROOT), str(tmp_path)],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    rebuilt = pd.read_csv(tmp_path / VALUES.name, sep="\t")
    pd.testing.assert_frame_equal(rebuilt, pd.read_csv(VALUES, sep="\t"))
    assert (tmp_path / f"{sW.STEM}_caption.md").read_text() == \
        (OUTPUT / f"{sW.STEM}_caption.md").read_text()


def test_the_table_is_the_held_out_result_and_not_the_in_sample_one(shipped):
    """The in-sample peak of the mean curve is one number for all thirteen lineages. If the
    shipped ``alpha_star`` were that peak everywhere, the selection would never have been
    held out -- so at least one arm has to disagree with it."""
    frame = GUIDE.read_table("alpha_sweep_by_lineage", GUIDE.results_dir)
    model = GUIDE.models_with_role("finetune")[0].key
    matrix = lineage_matrix(frame, "site_logistic_auroc", model=model)
    in_sample = select_alpha(matrix.columns.to_numpy(float), matrix.to_numpy(float).mean(axis=0))
    held_out = shipped[(shipped["scope"] == FOLD) & (shipped["scoring"] == "unmasked")
                       & (shipped["metric"] == "site_logistic_auroc")
                       & (shipped["model"] == model)]["alpha_star"]
    assert len(held_out) == GUIDE.n_lineages
    assert (held_out != in_sample).any(), (
        "every fold reproduced the in-sample peak; the folds are not being held out"
    )


# ---------------------------------------------------------------------------
# The p-value that must not appear
# ---------------------------------------------------------------------------

#: Names a test statistic would be computed under. ``binomtest`` and ``ttest_rel`` are the
#: two scipy calls this would most naturally be written with; the rest would be the column
#: or variable the result were put in.
_FORBIDDEN = re.compile(r"binomtest|ttest|pvalue|p_value|p_sign|sign_test", re.I)


def _code_only(path: Path) -> str:
    """The file with every comment and string literal removed.

    The prose *must* be free to say why there is no p-value -- that is the point of writing
    it down -- so the ban applies to the code, where a number would actually be produced.
    Checking line by line and allow-listing the explanatory sentences was the first attempt;
    it failed on the explanation wrapping onto a second line, which is precisely the kind of
    accident an allow-list of phrases cannot survive.
    """
    kept = []
    with tokenize.open(path) as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type not in (tokenize.COMMENT, tokenize.STRING):
                kept.append(token.string)
    return " ".join(kept)


@pytest.mark.parametrize("path", [
    REPO_ROOT / "mutacc" / "loo_alpha.py",
    REPO_ROOT / "tools" / "export_sW_loo_alpha.py",
])
def test_no_test_statistic_is_computed_over_the_folds(path):
    """The lineages are pseudo-replicates (residual outcome co-occurrence about 0.391), so
    a sign test or a paired t-test over the folds is anti-conservative -- and 13 of 13 is a
    sign test's floor of 2 / 2^13 whatever the effect size. Mentions in prose are allowed
    and are the point; a computation is not."""
    found = _FORBIDDEN.findall(_code_only(path))
    assert not found, f"{path.name} computes a test statistic over the folds: {found}"
    assert "scipy" not in path.read_text(), f"{path.name} imports scipy; nothing here needs it"


def test_the_ban_would_catch_a_p_value_that_was_added():
    """Guard the guard: a regex that matches nothing would pass on any file."""
    assert _FORBIDDEN.search("p_value = binomtest(k, n).pvalue")


def test_the_shipped_caption_and_table_carry_no_p_value(shipped):
    assert not {c for c in shipped.columns if c.startswith(("p_", "q_")) or "pval" in c}
    caption = (OUTPUT / f"{sW.STEM}_caption.md").read_text()
    # The caption must *say* there is none, and must not then print one.
    assert "No p-value is given" in caption
    assert not re.search(r"\bp\s*[=<>]\s*0?\.\d", caption), "the caption quotes a p-value"
