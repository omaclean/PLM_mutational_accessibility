import numpy as np
import pandas as pd
import pytest

from mutacc.metrics import bootstrap_mean_ci, lineage_matrix, mean_over_lineages


def frame(values, model="m"):
    rows = [
        {"lineage": f"L{i}", "alpha": a, "model": model, "model_variant": "plm_alpha_sweep", "x": v}
        for i, row in enumerate(values) for a, v in zip((0.0, 1.0), row)
    ]
    return pd.DataFrame(rows)


def test_matrix_shape_and_mean():
    w = lineage_matrix(frame([[1, 2], [3, 4]]), "x", model="m")
    assert w.shape == (2, 2)
    assert np.allclose(mean_over_lineages(w), [2.0, 3.0])


def test_missing_cell_raises_not_averages():
    f = frame([[1, 2], [3, 4]])
    f.loc[0, "x"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        lineage_matrix(f, "x", model="m")


def test_duplicates_and_unknown_model_and_metric():
    f = frame([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="no rows"):
        lineage_matrix(f, "x", model="other")
    with pytest.raises(KeyError):
        lineage_matrix(f, "nope", model="m")
    with pytest.raises(ValueError, match="duplicate"):
        lineage_matrix(pd.concat([f, f]), "x", model="m")


def test_bootstrap_is_seeded_and_brackets_the_mean():
    w = lineage_matrix(frame([[1, 10], [2, 11], [3, 12], [4, 13]]), "x", model="m")
    lo, hi = bootstrap_mean_ci(w, n_boot=500, seed=1)
    lo2, hi2 = bootstrap_mean_ci(w, n_boot=500, seed=1)
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)
    m = mean_over_lineages(w)
    assert (lo <= m).all() and (m <= hi).all() and (lo < hi).all()


def test_bootstrap_needs_two_lineages():
    w = lineage_matrix(frame([[1, 2]]), "x", model="m")
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_mean_ci(w)


def test_identical_lineages_give_zero_width_band():
    w = lineage_matrix(frame([[1, 2], [1, 2], [1, 2]]), "x", model="m")
    lo, hi = bootstrap_mean_ci(w, n_boot=200)
    assert np.allclose(lo, hi)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
from mutacc.metrics import auroc, mcclish_floor, partial_auroc, pr_auc, roc_points  # noqa: E402


def test_perfect_and_reversed_rankings():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert auroc(s, y) == pytest.approx(1.0)
    assert auroc(-s, y) == pytest.approx(0.0)
    assert pr_auc(s, y) == pytest.approx(1.0)


def test_auroc_matches_the_pairwise_definition():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.2).astype(int)
    s = rng.normal(size=200) + y
    pos, neg = s[y == 1], s[y == 0]
    expected = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert auroc(s, y) == pytest.approx(expected, abs=1e-12)


def test_ties_collapse_to_one_roc_point():
    fpr, tpr = roc_points(np.array([1.0, 1.0, 1.0, 1.0]), np.array([1, 0, 1, 0]))
    assert list(fpr) == [0.0, 1.0] and list(tpr) == [0.0, 1.0]
    assert auroc(np.zeros(4), np.array([1, 0, 1, 0])) == pytest.approx(0.5)


def test_random_ranking_is_chance_for_any_prevalence():
    rng = np.random.default_rng(1)
    for prevalence in (0.02, 0.1, 0.4):
        y = (rng.random(20000) < prevalence).astype(int)
        s = rng.normal(size=20000)
        assert auroc(s, y) == pytest.approx(0.5, abs=0.03)
        assert pr_auc(s, y) == pytest.approx(y.mean(), abs=0.03)   # PR-AUC chance IS the prevalence


def test_partial_auroc_floor_chance_and_ceiling():
    cap = 0.05
    y = np.array([1] * 50 + [0] * 950)
    perfect = np.r_[np.ones(50), np.zeros(950)]
    fpr, tpr = roc_points(perfect, y)
    assert partial_auroc(fpr, tpr, cap)[1] == pytest.approx(1.0)
    worst = np.r_[np.zeros(50), np.ones(950)]
    fpr, tpr = roc_points(worst, y)
    raw, std = partial_auroc(fpr, tpr, cap)
    assert raw == pytest.approx(0.0, abs=1e-12)
    assert std == pytest.approx(mcclish_floor(cap)) and std == pytest.approx(0.4872, abs=1e-4)
    rng = np.random.default_rng(2)
    yy = (rng.random(40000) < 0.1).astype(int)
    fpr, tpr = roc_points(rng.normal(size=40000), yy)
    assert partial_auroc(fpr, tpr, cap)[1] == pytest.approx(0.5, abs=0.06)


def test_partial_auroc_rejects_a_bad_cap():
    with pytest.raises(ValueError):
        partial_auroc(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.0)


@pytest.mark.parametrize("y", [np.zeros(5), np.ones(5)])
def test_a_single_class_cannot_be_ranked(y):
    with pytest.raises(ValueError, match="both"):
        auroc(np.arange(5.0), y)


def test_bad_scores_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        auroc(np.array([1.0, np.nan, 3.0]), np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="same length"):
        auroc(np.arange(3.0), np.array([1, 0]))


def test_metrics_are_unchanged_by_a_positive_rescaling_of_the_score():
    """The pipeline scored in natural log and the figure in log10: they must agree."""
    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.1).astype(int)
    s = rng.normal(size=500) + y
    for k in (0.4343, 7.0):
        assert auroc(k * s, y) == pytest.approx(auroc(s, y), abs=1e-12)
        assert pr_auc(k * s, y) == pytest.approx(pr_auc(s, y), abs=1e-12)
