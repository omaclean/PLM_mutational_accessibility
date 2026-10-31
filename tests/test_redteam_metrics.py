"""Red team: ranking metrics, rank statistics with ties, bootstraps, silent empty joins.

Everything is checked against an independent brute-force version on tiny cases, or against
scipy. Fuzzing is seeded numpy (hypothesis is not installed).

Bugs these tests pinned (both fixed in ``mutacc/metrics.py``):

* ``pr_auc`` took tied scores "in input order", so the same data in a different row order
  gave a different number. The accessibility-only baseline of S2 is almost all ties.
* ``partial_auroc`` interpolated with ``np.interp`` at a cap that equalled an fpr grid value
  sitting on a vertical ROC segment, which returns the top of the segment: the standardised
  value jumped from 0.487 to 0.744 on a ranker that puts every hit last.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr

from mutacc import dms_sweep as D
from mutacc import metrics as M
from mutacc import strata as S


# ---------------------------------------------------------------------------
# Brute-force references
# ---------------------------------------------------------------------------

def brute_auroc(score, y):
    pos = score[y > 0.5]
    neg = score[y <= 0.5]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brute_average_precision(score, y):
    """Precision at every distinct threshold, weighted by the recall gained there."""
    total = (y > 0.5).sum()
    prev_recall, ap = 0.0, 0.0
    for thr in sorted(set(score), reverse=True):
        sel = score >= thr
        tp = (sel & (y > 0.5)).sum()
        recall = tp / total
        ap += (recall - prev_recall) * tp / sel.sum()
        prev_recall = recall
    return ap


def brute_partial_roc_area(score, y, cap):
    """Area under the ROC path (diagonal across ties) for fpr <= cap, by dense sampling."""
    thr = sorted(set(score), reverse=True)
    n_pos, n_neg = (y > 0.5).sum(), (y <= 0.5).sum()
    pts = [(0.0, 0.0)]
    for t in thr:
        sel = score >= t
        pts.append((((sel) & (y <= 0.5)).sum() / n_neg, (sel & (y > 0.5)).sum() / n_pos))
    xs, ys = zip(*pts)
    grid = np.linspace(0, cap, 200001)
    # tpr as a function of fpr: for a vertical segment take the top, but the AREA is what matters
    return np.trapezoid(np.interp(grid, xs, ys), grid), pts


def tied_case(rng, n=60, levels=5, prevalence=0.25):
    s = rng.integers(0, levels, n).astype(float)
    y = (rng.random(n) < prevalence).astype(float)
    if y.sum() in (0, n):
        y[0], y[1] = 1.0, 0.0
    return s, y


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def test_auroc_matches_mann_whitney_with_heavy_ties():
    rng = np.random.default_rng(101)
    for _ in range(150):
        s, y = tied_case(rng, n=int(rng.integers(8, 70)), levels=int(rng.integers(1, 8)))
        assert M.auroc(s, y) == pytest.approx(brute_auroc(s, y), abs=1e-12)


def test_auroc_is_not_floored_at_chance_and_is_complementary():
    rng = np.random.default_rng(102)
    for _ in range(50):
        s = rng.normal(size=80)
        y = (rng.random(80) < 0.2).astype(float)
        y[0], y[1] = 1, 0
        assert M.auroc(-s, y) == pytest.approx(1 - M.auroc(s, y), abs=1e-12)
    # an anti-predictive score must read below 0.5 (the old clamp read exactly 0.5)
    s = np.arange(40.0)
    y = np.r_[np.ones(5), np.zeros(35)]
    assert M.auroc(s, y) < 0.05


def test_auroc_ignores_row_order_and_monotone_transforms():
    rng = np.random.default_rng(103)
    s, y = tied_case(rng, n=80, levels=6)
    base = M.auroc(s, y)
    for _ in range(10):
        p = rng.permutation(len(s))
        assert M.auroc(s[p], y[p]) == pytest.approx(base, abs=1e-12)
    assert M.auroc(np.exp(s), y) == pytest.approx(base, abs=1e-12)
    assert M.auroc(10 * s - 3, y) == pytest.approx(base, abs=1e-12)


def test_roc_points_are_a_monotone_path_from_origin_to_corner():
    rng = np.random.default_rng(104)
    for _ in range(40):
        s, y = tied_case(rng, n=50, levels=int(rng.integers(1, 10)))
        fpr, tpr = M.roc_points(s, y)
        assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
        assert (np.diff(fpr) >= 0).all() and (np.diff(tpr) >= 0).all()
        assert len(fpr) == len(set(s)) + 1


@pytest.mark.parametrize("bad", ["nan", "inf", "shape", "2d", "no_pos", "no_neg"])
def test_metric_inputs_are_refused(bad):
    s, y = np.array([0.1, 0.5, 0.2, 0.9]), np.array([1.0, 0, 1, 0])
    if bad == "nan":
        s[1] = np.nan
    elif bad == "inf":
        s[1] = np.inf
    elif bad == "shape":
        y = y[:3]
    elif bad == "2d":
        s, y = s.reshape(2, 2), y.reshape(2, 2)
    elif bad == "no_pos":
        y = np.zeros(4)
    elif bad == "no_neg":
        y = np.ones(4)
    for fn in (M.auroc, M.pr_auc, M.roc_points):
        with pytest.raises(ValueError):
            fn(s, y)


# ---------------------------------------------------------------------------
# PR-AUC
# ---------------------------------------------------------------------------

def test_pr_auc_does_not_depend_on_the_order_of_tied_rows():
    rng = np.random.default_rng(105)
    for _ in range(60):
        s, y = tied_case(rng, n=int(rng.integers(10, 120)), levels=int(rng.integers(1, 6)))
        ref = M.pr_auc(s, y)
        for _ in range(5):
            p = rng.permutation(len(s))
            assert M.pr_auc(s[p], y[p]) == pytest.approx(ref, abs=1e-12)


def test_pr_auc_matches_threshold_by_threshold_average_precision():
    rng = np.random.default_rng(106)
    for _ in range(80):
        s, y = tied_case(rng, n=int(rng.integers(8, 90)), levels=int(rng.integers(1, 9)))
        assert M.pr_auc(s, y) == pytest.approx(brute_average_precision(s, y), abs=1e-12)


def test_pr_auc_agrees_with_scikit_learn_when_available():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(107)
    for _ in range(30):
        s, y = tied_case(rng, n=100, levels=4)
        assert M.pr_auc(s, y) == pytest.approx(skm.average_precision_score(y, s), abs=1e-12)


def test_constant_score_pr_auc_is_exactly_the_prevalence():
    y = np.r_[np.ones(7), np.zeros(93)]
    for p in (np.arange(100), np.random.default_rng(1).permutation(100)):
        assert M.pr_auc(np.zeros(100), y[p]) == pytest.approx(0.07, abs=1e-12)


# ---------------------------------------------------------------------------
# Partial AUROC
# ---------------------------------------------------------------------------

def test_partial_auroc_is_continuous_in_the_cap_at_a_grid_point():
    """20 negatives, cap 0.05 = exactly one negative. A hit-last ranker is at the floor either way."""
    s = np.array([10] + [9, 8.5, 8.4] + [-i for i in range(19)], float)
    y = np.array([0, 1, 1, 1] + [0] * 19, float)
    fpr, tpr = M.roc_points(s, y)
    assert 0.05 in fpr                                   # the cap sits exactly on the grid
    at_grid = M.partial_auroc(fpr, tpr, 0.05)
    just_above = M.partial_auroc(fpr, tpr, 0.05 * (1 + 1e-9))
    just_below = M.partial_auroc(fpr, tpr, 0.05 * (1 - 1e-9))
    assert at_grid[0] == pytest.approx(just_above[0], abs=1e-8)
    assert at_grid[1] == pytest.approx(just_below[1], abs=1e-6)
    assert at_grid[1] == pytest.approx(M.mcclish_floor(0.05), abs=1e-6)


def test_partial_auroc_matches_dense_integration_for_caps_on_and_off_the_grid():
    rng = np.random.default_rng(108)
    for _ in range(60):
        s, y = tied_case(rng, n=int(rng.integers(20, 80)), levels=int(rng.integers(2, 30)), prevalence=0.3)
        fpr, tpr = M.roc_points(s, y)
        n_neg = int((y <= 0.5).sum())
        caps = [k / n_neg for k in (1, 2, max(3, n_neg // 4))] + [float(rng.uniform(0.02, 0.9))]
        for cap in caps:
            raw, _ = M.partial_auroc(fpr, tpr, min(cap, 1.0))
            ref, _ = brute_partial_roc_area(s, y, min(cap, 1.0))
            assert raw == pytest.approx(ref, abs=2e-5), cap


def test_partial_auroc_endpoints_for_perfect_and_worst_rankers_at_any_cap():
    for n_neg in (20, 40, 100):
        n_pos = 7
        y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        perfect = np.r_[np.arange(n_pos, 0, -1) + 100, -np.arange(n_neg)].astype(float)
        worst = -perfect
        for cap in (0.05, 0.1, 0.25, 1.0):
            _, hi = M.partial_auroc(*M.roc_points(perfect, y), cap)
            _, lo = M.partial_auroc(*M.roc_points(worst, y), cap)
            assert hi == pytest.approx(1.0, abs=1e-9)
            assert lo == pytest.approx(M.mcclish_floor(cap), abs=1e-9), (n_neg, cap)


def test_partial_auroc_cap_one_is_the_full_auroc():
    rng = np.random.default_rng(109)
    s, y = tied_case(rng, n=70, levels=7)
    fpr, tpr = M.roc_points(s, y)
    raw, std = M.partial_auroc(fpr, tpr, 1.0)
    assert raw == pytest.approx(M.auroc(s, y), abs=1e-12)
    assert std == pytest.approx(M.auroc(s, y), abs=1e-12)


@pytest.mark.parametrize("cap", [0.0, -0.1, 1.5, float("nan")])
def test_partial_auroc_rejects_bad_caps(cap):
    fpr, tpr = M.roc_points(np.array([3.0, 2, 1, 0]), np.array([1.0, 0, 1, 0]))
    with pytest.raises(ValueError):
        M.partial_auroc(fpr, tpr, cap)


# ---------------------------------------------------------------------------
# Rank statistics with ties
# ---------------------------------------------------------------------------

def test_col_ranks_equal_scipy_average_ranks():
    rng = np.random.default_rng(110)
    for _ in range(40):
        X = rng.integers(0, int(rng.integers(1, 12)), size=(int(rng.integers(2, 60)), 4)).astype(float)
        want = np.column_stack([rankdata(X[:, j]) - 1 for j in range(4)])
        assert np.allclose(D.col_ranks(X), want)


def test_rho_cols_and_sweep_curve_equal_scipy_spearman_with_ties():
    rng = np.random.default_rng(111)
    for _ in range(30):
        n = int(rng.integers(30, 80))
        y = rng.integers(0, 6, n).astype(float)
        la = rng.integers(0, 4, n).astype(float)
        lp = rng.integers(0, 5, n).astype(float)
        alphas = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        got = D.sweep_curve(lp, la, y, alphas)
        want = [spearmanr(lp + a * la, y).statistic for a in alphas]
        assert np.allclose(got, want, atol=1e-12, equal_nan=True)


def test_spearman_helper_matches_scipy_and_is_nan_when_undefined():
    rng = np.random.default_rng(112)
    for _ in range(30):
        x, y = rng.integers(0, 5, 40).astype(float), rng.integers(0, 5, 40).astype(float)
        if x.std() and y.std():
            assert D.spearman(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)
    assert np.isnan(D.spearman(np.ones(10), np.arange(10.0)))
    assert np.isnan(D.spearman(np.arange(2.0), np.arange(2.0)))


def test_partial_from_ranks_equals_the_residual_method():
    rng = np.random.default_rng(113)
    for _ in range(40):
        z = rng.normal(size=200)
        x = 0.6 * z + rng.normal(size=200)
        y = -0.4 * z + 0.3 * x + rng.normal(size=200)
        c = np.corrcoef([x, y, z])
        got = S.partial_from_ranks(c[0, 1], c[0, 2], c[1, 2])

        def resid(a):
            slope = np.polyfit(z, a, 1)
            return a - np.polyval(slope, z)

        assert got == pytest.approx(np.corrcoef(resid(x), resid(y))[0, 1], abs=1e-10)


# ---------------------------------------------------------------------------
# strata_table against brute force
# ---------------------------------------------------------------------------

def make_replacement_frame(rng, lineages=("L1", "L2", "L3"), n_sites=70, aas="ACDEF", drop_row=None):
    rows = []
    ref = {p: "GAVLI"[p % 5] for p in range(1, n_sites + 1)}
    for lin in lineages:
        for p in range(1, n_sites + 1):
            for a in aas:
                if a == ref[p]:
                    continue
                rows.append(dict(lineage=lin, position=p, ref_aa=ref[p], aa=a,
                                 mut_prob=float(rng.integers(1, 9)) / 100, nt_steps=int(rng.integers(1, 4)),
                                 neg_grantham=float(rng.integers(-9, 0)), blosum62=float(rng.integers(-3, 4))))
    base = pd.DataFrame(rows)
    out = []
    for key in ("fine", "base"):
        f = base.copy()
        f["model"] = key
        f["plm_prob"] = rng.integers(1, 30, len(f)) / 1000.0      # heavy ties
        out.append(f)
    return pd.concat(out, ignore_index=True)


def test_strata_estimates_equal_per_lineage_scipy_spearman_means():
    rng = np.random.default_rng(114)
    frame = make_replacement_frame(rng)
    table = S.strata_table(frame, ["fine", "base"], ["neg_grantham", "blosum62"], n_boot=0)
    fine = frame[frame.model == "fine"]
    for stratum in (S.ALL, 1, 2, 3):
        vals = []
        for _, g in fine.groupby("lineage"):
            g = g if stratum == S.ALL else g[g.nt_steps == stratum]
            vals.append(spearmanr(g["plm_prob"], g["mut_prob"]).statistic)
        row = table[(table.series == "fine") & (table.stratum == str(stratum)) & (table.control == "")]
        assert row["estimate"].iloc[0] == pytest.approx(np.mean(vals), abs=1e-12)
        assert row["lineage_min"].iloc[0] == pytest.approx(min(vals), abs=1e-12)
        assert row["lineage_max"].iloc[0] == pytest.approx(max(vals), abs=1e-12)
        g = fine if stratum == S.ALL else fine[fine.nt_steps == stratum]
        assert row["pooled"].iloc[0] == pytest.approx(spearmanr(g["plm_prob"], g["mut_prob"]).statistic, abs=1e-12)
        prop = table[(table.series == "neg_grantham") & (table.stratum == str(stratum))]
        want = np.mean([spearmanr(h["neg_grantham"], h["mut_prob"]).statistic
                        for _, h in fine.groupby("lineage") if True]) if stratum == S.ALL else None
        if want is not None:
            assert prop["estimate"].iloc[0] == pytest.approx(want, abs=1e-12)


def test_strata_partial_correlation_equals_residual_method_on_ranks():
    rng = np.random.default_rng(115)
    frame = make_replacement_frame(rng)
    table = S.strata_table(frame, ["fine", "base"], ["neg_grantham", "blosum62"], n_boot=0)
    fine = frame[frame.model == "fine"]
    vals = []
    for _, g in fine.groupby("lineage"):
        g = g[g.nt_steps == 2]
        x, y, z = (rankdata(g[c]) for c in ("plm_prob", "mut_prob", "neg_grantham"))

        def resid(a):
            return a - np.polyval(np.polyfit(z, a, 1), z)

        vals.append(np.corrcoef(resid(x), resid(y))[0, 1])
    row = table[(table.series == "fine") & (table.stratum == "2") & (table.control == "neg_grantham")]
    assert row["estimate"].iloc[0] == pytest.approx(np.mean(vals), abs=1e-10)


def test_strata_bootstrap_is_seeded_and_the_interval_brackets_the_estimate():
    rng = np.random.default_rng(116)
    frame = make_replacement_frame(rng)
    a = S.strata_table(frame, ["fine", "base"], ["neg_grantham"], n_boot=30, seed=5)
    b = S.strata_table(frame, ["fine", "base"], ["neg_grantham"], n_boot=30, seed=5)
    c = S.strata_table(frame, ["fine", "base"], ["neg_grantham"], n_boot=30, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["lo"], c["lo"])
    assert (a["lo"] <= a["hi"]).all()
    assert ((a["lo"] - 0.15 <= a["estimate"]) & (a["estimate"] <= a["hi"] + 0.15)).all()
    zero = S.strata_table(frame, ["fine", "base"], ["neg_grantham"], n_boot=0)
    assert zero["lo"].isna().all() and zero["hi"].isna().all()


def test_strata_resamples_whole_sites_not_rows():
    """A site draw must keep all of a site's replacements together, in every lineage."""
    rng = np.random.default_rng(117)
    frame = make_replacement_frame(rng, lineages=("L1", "L2"), n_sites=12)
    wide = S.wide_scores(frame, ["fine"], ["neg_grantham"])
    blocks = S._blocks(wide, ["plm::fine", "mut_prob", "neg_grantham"])
    universe = np.unique(np.concatenate([b.pos for b in blocks]))
    picked = np.array([0, 0, 3, 5])
    for blk, idx in zip(blocks, S._resample_indices(blocks, universe, picked)):
        sites = blk.pos[idx]
        want = np.concatenate([blk.pos[blk.pos == universe[i]] for i in picked])
        assert (sites == want).all()


def test_wide_scores_refuses_silent_inner_joins():
    rng = np.random.default_rng(118)
    frame = make_replacement_frame(rng, lineages=("L1",), n_sites=30)
    ok = S.wide_scores(frame, ["fine", "base"], ["neg_grantham"])
    assert len(ok) == (frame.model == "fine").sum()
    dropped = frame.drop(frame[(frame.model == "base")].index[:3])
    with pytest.raises(ValueError, match="same replacements"):
        S.wide_scores(dropped, ["fine", "base"], ["neg_grantham"])
    with pytest.raises(ValueError, match="no rows for model"):
        S.wide_scores(frame, ["fine", "ghost"], ["neg_grantham"])
    dup = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        S.wide_scores(dup, ["fine"], ["neg_grantham"])
    skew = frame.copy()
    skew.loc[(skew.model == "base") & (skew.index == skew.index[skew.model == "base"][0]), "nt_steps"] += 1
    with pytest.raises(ValueError, match="differs between models"):
        S.wide_scores(skew, ["fine", "base"], ["neg_grantham"])
    nan = frame.copy()
    nan.loc[0, "plm_prob"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        S.wide_scores(nan, ["fine", "base"], ["neg_grantham"])


def test_a_stratum_with_too_few_rows_raises_not_nan():
    rng = np.random.default_rng(119)
    frame = make_replacement_frame(rng, lineages=("L1", "L2"), n_sites=8, aas="ACD")
    frame.loc[frame.nt_steps == 3, "nt_steps"] = 2
    frame.loc[frame.position == 1, "nt_steps"] = 3      # stratum 3 now has under 20 rows per lineage
    with pytest.raises(ValueError, match="fewer than"):
        S.strata_table(frame, ["fine", "base"], ["neg_grantham"], n_boot=0)


# ---------------------------------------------------------------------------
# pair_asymmetry
# ---------------------------------------------------------------------------

def make_pair_frame(rng):
    aas = list(S.AA_ORDER)
    rows = []
    for lin in ("L1", "L2"):
        for p in range(1, 41):
            ref = aas[p % 20]
            for a in aas:
                if a != ref:
                    rows.append(dict(lineage=lin, position=p, ref_aa=ref, aa=a, model="m",
                                     plm_prob=float(rng.uniform(1e-6, 1e-1)), mut_prob=float(rng.uniform(1e-9, 1e-3))))
    return pd.DataFrame(rows)


def test_pair_asymmetry_matches_a_direct_loop_and_flips_sign_with_direction():
    rng = np.random.default_rng(120)
    frame = make_pair_frame(rng)
    pairs = S.pair_asymmetry(frame, "m")
    logp = np.log10(frame["plm_prob"])
    for r in pairs.head(25).itertuples():
        ab = logp[(frame.ref_aa == r.aa_from) & (frame.aa == r.aa_to)].mean()
        ba = logp[(frame.ref_aa == r.aa_to) & (frame.aa == r.aa_from)].mean()
        assert r.plm_asym == pytest.approx(ab - ba, abs=1e-12)
    # swapping every pair's direction negates both asymmetries and cannot change agreement
    flipped = frame.rename(columns={"ref_aa": "aa", "aa": "ref_aa"})
    other = S.pair_asymmetry(flipped, "m")
    s1, s2 = S.asymmetry_summary(pairs), S.asymmetry_summary(other)
    assert s1["n_agree"] == s2["n_agree"] and s1["rho_overall"] == pytest.approx(s2["rho_overall"], abs=1e-9)


def test_pair_asymmetry_refuses_synonymous_rows_and_bad_symbols():
    rng = np.random.default_rng(121)
    frame = make_pair_frame(rng)
    bad = frame.copy()
    bad.loc[0, "aa"] = bad.loc[0, "ref_aa"]
    with pytest.raises(ValueError, match="synonymous"):
        S.pair_asymmetry(bad, "m")
    bad = frame.copy()
    bad.loc[0, "aa"] = "*"
    with pytest.raises(ValueError, match="outside the 20"):
        S.pair_asymmetry(bad, "m")


def test_agreement_count_treats_a_zero_asymmetry_as_agreeing_only_with_zero():
    """Open edge: sign(0) == sign(0), so a pair that is symmetric in both counts as agreeing.
    Pinned so that a change in the convention is deliberate."""
    pairs = pd.DataFrame({"aa_from": list("GAVL"), "aa_to": list("AVLI"), "mut_asym": [0.0, 0.0, 0.5, -0.5],
                          "plm_asym": [0.0, 0.2, 0.5, 0.5], "mut_resid": [0.0, 0.0, 0.5, -0.5],
                          "plm_resid": [0.0, 0.2, 0.5, 0.5]})
    assert S.asymmetry_summary(pairs)["n_agree"] == 2


# ---------------------------------------------------------------------------
# Bootstraps over lineages
# ---------------------------------------------------------------------------

def sweep_frame(rng, lineages=5, model="m", alphas=(-1.0, 0.0, 1.0), skip=None):
    rows = []
    for li in range(lineages):
        for a in alphas:
            if skip == (li, a):
                continue
            rows.append(dict(lineage=f"L{li}", alpha=a, model=model, model_variant="plm_alpha_sweep",
                             metric=float(rng.normal())))
    return pd.DataFrame(rows)


def test_lineage_bootstrap_is_reproducible_and_seed_sensitive():
    rng = np.random.default_rng(122)
    wide = M.lineage_matrix(sweep_frame(rng), "metric", model="m")
    a = M.bootstrap_mean_ci(wide, n_boot=300, seed=1)
    b = M.bootstrap_mean_ci(wide, n_boot=300, seed=1)
    c = M.bootstrap_mean_ci(wide, n_boot=300, seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.allclose(a[0], c[0])
    default_1 = M.bootstrap_mean_ci(wide, n_boot=50)
    default_2 = M.bootstrap_mean_ci(wide, n_boot=50)
    assert all(np.array_equal(x, y) for x, y in zip(default_1, default_2))


def test_lineage_bootstrap_interval_properties():
    rng = np.random.default_rng(123)
    wide = M.lineage_matrix(sweep_frame(rng, lineages=8), "metric", model="m")
    lo, hi = M.bootstrap_mean_ci(wide, n_boot=500, seed=3)
    mean = M.mean_over_lineages(wide)
    assert (lo <= hi).all()
    assert (lo >= wide.min().to_numpy() - 1e-12).all() and (hi <= wide.max().to_numpy() + 1e-12).all()
    assert ((lo <= mean + 1e-9) & (mean <= hi + 1e-9)).all()
    const = pd.DataFrame(np.full((4, 3), 0.7), columns=[0.0, 1.0, 2.0])
    lo, hi = M.bootstrap_mean_ci(const, n_boot=50)
    assert np.allclose(lo, 0.7) and np.allclose(hi, 0.7)
    with pytest.raises(ValueError, match="at least 2"):
        M.bootstrap_mean_ci(const.iloc[:1])


def test_lineage_matrix_refuses_a_hole_a_duplicate_and_a_wrong_model():
    rng = np.random.default_rng(124)
    with pytest.raises(ValueError, match="missing or non-finite"):
        M.lineage_matrix(sweep_frame(rng, skip=(2, 0.0)), "metric", model="m")
    f = sweep_frame(rng)
    with pytest.raises(ValueError, match="duplicate"):
        M.lineage_matrix(pd.concat([f, f.iloc[:1]]), "metric", model="m")
    with pytest.raises(ValueError, match="no rows for model"):
        M.lineage_matrix(f, "metric", model="other")
    with pytest.raises(KeyError):
        M.lineage_matrix(f, "nonexistent", model="m")
    f2 = f.copy()
    f2.loc[3, "metric"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        M.lineage_matrix(f2, "metric", model="m")


def test_site_bootstrap_band_is_reproducible_and_ordered():
    rng = np.random.default_rng(125)
    n = 300
    sites = np.repeat(np.arange(60), 5)
    lp, la, y = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    alphas = np.round(np.linspace(-1, 1, 21), 6)
    a = D.site_bootstrap_band(lp, la, y, sites, alphas, np.random.default_rng(9), n_boot=40)
    b = D.site_bootstrap_band(lp, la, y, sites, alphas, np.random.default_rng(9), n_boot=40)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert (a[0] <= a[1]).all()
    with pytest.raises(ValueError):
        D.site_bootstrap_band(lp, la, y, sites, alphas, np.random.default_rng(9), n_boot=0)


def test_curve_flags_boundary_weak_signal_and_ties():
    alphas = np.round(np.linspace(-1, 1, 21), 6)
    rho_peak = 0.5 - (alphas - 0.3) ** 2
    f = D.curve_flags(rho_peak, alphas)
    assert f["best_alpha"] == pytest.approx(0.3) and f["interpretable"]
    edge = D.curve_flags(alphas * 0.4 + 0.2, alphas)
    assert edge["boundary"] and not edge["interpretable"]
    weak = D.curve_flags(0.01 - 0.0005 * (alphas - 0.2) ** 2, alphas)
    assert weak["weak_signal"] and not weak["interpretable"]
    with pytest.raises(ValueError):
        D.curve_flags(np.full(21, np.nan), alphas)
    # a perfectly flat curve: every alpha ties; the first is chosen, which is the left boundary
    flat = D.curve_flags(np.full(21, 0.3), alphas)
    assert flat["boundary"] and flat["best_alpha"] == -1.0


def test_paired_gap_is_zero_for_identical_scores_and_seeded():
    rng = np.random.default_rng(126)
    sites = np.repeat(np.arange(40), 4)
    x, y = rng.normal(size=160), rng.normal(size=160)
    same = D.paired_site_gap(x, x, y, sites, np.random.default_rng(1), n_boot=50)
    assert same["gap"] == 0.0 and same["lo"] == 0.0 and same["hi"] == 0.0 and same["p"] == 1.0
    a = D.paired_site_gap(x, y + 0.1 * x, y, sites, np.random.default_rng(1), n_boot=50)
    b = D.paired_site_gap(x, y + 0.1 * x, y, sites, np.random.default_rng(1), n_boot=50)
    assert a == b
