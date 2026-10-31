"""Two further estimators of the model-accessibility link, for S4.

No plotting, no paths. Used by S4 alongside ``mutacc.strata``.

The question is the same one Figure 2D asks: does a model's probability track mutational
accessibility for a reason other than amino-acid chemistry? The partial Spearman in
``mutacc.strata`` answers it one way. That answer has two known weak points, and each
estimator here is built to fail differently rather than to agree by construction:

* **Partial correlation removes only a monotone, linear-in-ranks effect of the control.**
  Any curved dependence on chemistry survives into the residual and is still counted as
  model signal. ``matched_rank`` instead *conditions*: it correlates only replacements of
  similar Grantham distance and pools the within-bin correlations. Nothing is regressed
  out, so nothing non-monotone can leak through.

* **A rank correlation weights every replacement equally.** The model's probability spans
  eight orders of magnitude at a site, so a swap between two replacements it has all but
  ruled out counts as much as a swap between its top two. ``softmax_beta`` weights by
  where the model actually puts its probability.

If the three disagree, the result depends on how the probability simplex is weighted and
should be reported as fragile. If they agree, it does not.

What the softmax fit is
-----------------------

A model's output at a site is a softmax over amino acids, so it is modelled as one::

    q(site, aa) proportional to exp(beta * ln accessibility + f(Grantham))

fitted within each site by cross-entropy against the model's own normalised probabilities.
The site intercepts cancel, which is the point: every site contributes its *ordering and
spacing* of replacements and nothing about its overall conservation.

``beta`` is a power-law exponent, ``d ln q / d ln A``. The model behaves as though it had
multiplied its probabilities by accessibility to the power beta, so it is the exponent the
alpha sweep tunes from outside, measured instead from the model's own output. It is
**natural log on both sides**, which makes it directly comparable to an exponent estimated
from ``log10`` on both sides, because a power-law exponent is invariant to the log base as
long as one base is used throughout. ``tests/test_sD_partial_correlation.py`` pins this by generating data
from a known exponent and recovering it.

This is *not* the alpha of the sweep figure and must not be captioned as the same number.
That alpha is an external weight tuned to maximise agreement with a measured phenotype;
beta is how much of that behaviour the model's own output already shows. Same units,
different questions.

Why each stratum is normalised within itself
--------------------------------------------

One softmax over all nineteen replacements at a site would let the three codon-distance
groups share a denominator. Replacements needing three nucleotide changes then carry almost
no probability, a handful of sites hold nearly all of what there is, and the bootstrap for
that group swings on whether those sites were drawn. Normalising within each (site, group)
cell gives every site equal weight in its own group, so the three-change estimate is as
stable as the others and does not have to be dropped.

The pooled "any" fit deliberately carries no codon-distance offsets. Most of the variation
in log accessibility across all nineteen replacements *is* the step between the groups, so
offsets and the slope would be estimating the same thing and neither would be identified.
"""

from __future__ import annotations

from dataclasses import dataclass

import contextlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from mutacc.strata import ALL, LOG_FLOOR, MIN_ROWS, wide_scores

try:                                            # pragma: no cover - import guard
    from threadpoolctl import threadpool_limits
except ImportError:                             # pragma: no cover - import guard
    def threadpool_limits(limits=None, user_api=None):
        """No-op stand-in: without threadpoolctl the fit is correct, only slower."""
        return contextlib.nullcontext()

__all__ = [
    "rcs_basis", "matched_rank", "softmax_beta", "link_table",
    "GRANTHAM_BINS", "MIN_PER_SITE", "SPLINE_KNOTS",
]

#: Grantham bins for the matched estimator: fifths of the distance, within each stratum.
GRANTHAM_BINS = 5

#: A site needs at least this many replacements in a stratum to inform a within-site fit.
MIN_PER_SITE = 2

#: Knot count for the Grantham spline, placed at equally spaced quantiles of the distance.
#: Four is the usual floor for a restricted cubic spline and leaves three fitted columns.
SPLINE_KNOTS = 4


# ---------------------------------------------------------------------------
# Spline basis
# ---------------------------------------------------------------------------

def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis for ``x`` (Harrell), ``len(knots) - 1`` columns.

    Column 0 is ``x`` itself; the rest are the cubic terms. "Restricted" means the fit is
    forced to be linear beyond the outer knots, which stops the control from inventing a
    steep chemistry effect out at the sparse ends of the Grantham range and charging the
    difference to the accessibility slope.
    """
    t = np.asarray(knots, dtype=float)
    if t.ndim != 1 or len(t) < 3 or not np.all(np.diff(t) > 0):
        raise ValueError("knots must be at least three strictly increasing values")
    x = np.asarray(x, dtype=float)
    k = len(t)
    scale = (t[-1] - t[0]) ** 2
    cube = lambda v: np.clip(v, 0.0, None) ** 3                      # noqa: E731
    cols = [x]
    for j in range(k - 2):
        term = (cube(x - t[j])
                - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(term / scale)
    return np.column_stack(cols)


def _knots(x: np.ndarray, n: int = SPLINE_KNOTS) -> np.ndarray:
    """Knot positions at equally spaced quantiles, de-duplicated."""
    return np.unique(np.quantile(np.asarray(x, dtype=float), np.linspace(0.05, 0.95, n)))


# ---------------------------------------------------------------------------
# Estimator 1: Grantham-matched rank correlation
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < MIN_ROWS:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def matched_rank(q: np.ndarray, acc: np.ndarray, grantham: np.ndarray,
                 edges: np.ndarray) -> float:
    """Spearman r between model and accessibility *within* bins of Grantham distance.

    The within-bin correlations are pooled on Fisher's z scale weighted by ``n - 3`` (the
    usual inverse-variance weight for a correlation) and transformed back, so the result is
    on the same scale as the Spearman r plotted beside it and can be read against it
    directly. Bins with too few replacements are skipped rather than guessed.
    """
    which = np.clip(np.searchsorted(edges[1:-1], grantham, side="right"), 0, len(edges) - 2)
    zs, ws = [], []
    for b in range(len(edges) - 1):
        sel = which == b
        r = _spearman(q[sel], acc[sel]) if sel.sum() >= MIN_ROWS else np.nan
        if np.isfinite(r) and abs(r) < 1.0:
            zs.append(np.arctanh(r))
            ws.append(int(sel.sum()) - 3)
    if not zs:
        return np.nan
    return float(np.tanh(np.average(zs, weights=ws)))


# ---------------------------------------------------------------------------
# Estimator 2: site-conditioned softmax (conditional logit)
# ---------------------------------------------------------------------------

def _group_bounds(gid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index and length of each run in a sorted group-id array."""
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    counts = np.diff(np.r_[starts, len(gid)])
    return starts, counts


def _nll_and_grad(theta: np.ndarray, X: np.ndarray, w: np.ndarray,
                  starts: np.ndarray, counts: np.ndarray):
    eta = X @ theta
    mx = np.maximum.reduceat(eta, starts)
    e = np.exp(eta - np.repeat(mx, counts))
    den = np.add.reduceat(e, starts)
    p = e / np.repeat(den, counts)
    logz = np.repeat(mx + np.log(den), counts)
    return float(-(w * (eta - logz)).sum()), X.T @ (p - w)


def softmax_beta(q: np.ndarray, log_acc: np.ndarray, gid: np.ndarray,
                 spline: np.ndarray | None = None) -> float:
    """Fit ``q(site, aa) ~ exp(beta * log_acc + f(Grantham))`` within sites; return beta.

    ``q`` is the model's probability (not logged), ``gid`` the site each row belongs to,
    already sorted and contiguous. The targets are ``q`` renormalised within each site, so
    every site contributes exactly one unit of weight however conserved it is. ``spline``,
    when given, is the Grantham basis from :func:`rcs_basis`.

    Returns NaN rather than a number when the fit is not identified (one group, or no
    variation in accessibility within any group), which is what the bootstrap expects.
    """
    starts, counts = _group_bounds(gid)
    keep = counts >= MIN_PER_SITE
    if keep.sum() < 2:
        return np.nan
    rows = np.concatenate([np.arange(s, s + c) for s, c in zip(starts[keep], counts[keep])])
    X = log_acc[rows][:, None] if spline is None else np.column_stack([log_acc[rows], spline[rows]])
    starts, counts = _group_bounds(gid[rows])
    totals = np.add.reduceat(q[rows], starts)
    if not np.all(totals > 0):
        return np.nan
    w = q[rows] / np.repeat(totals, counts)
    # No within-group variation in accessibility means beta is not identified at all.
    spread = X[:, 0] - np.repeat(np.add.reduceat(X[:, 0], starts) / counts, counts)
    if not np.any(np.abs(spread) > 1e-12):
        return np.nan
    fit = minimize(_nll_and_grad, np.zeros(X.shape[1]), args=(X, w, starts, counts),
                   jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12})
    return float(fit.x[0]) if fit.success else np.nan


# ---------------------------------------------------------------------------
# The table S4 draws
# ---------------------------------------------------------------------------

@dataclass
class _Data:
    pos: np.ndarray
    steps: np.ndarray
    q: dict[str, np.ndarray]
    acc: np.ndarray       # rank statistics use this, so they match Figure 2D exactly
    log_acc: np.ndarray   # the softmax uses this, because its slope is an exponent
    grantham: np.ndarray


def _prepare(frame: pd.DataFrame, model_keys: list[str], control: str) -> _Data:
    wide = wide_scores(frame, model_keys, [control])
    if wide["lineage"].nunique() != 1:
        raise ValueError("link_table analyses one lineage; select it before calling")
    wide = wide.sort_values(["position", "aa"], kind="stable").reset_index(drop=True)
    acc = wide["mut_prob"].to_numpy(float)
    if not np.all(acc > 0):
        raise ValueError("accessibility must be positive before taking a logarithm")
    return _Data(
        pos=wide["position"].to_numpy(),
        steps=wide["nt_steps"].to_numpy(),
        q={k: np.clip(wide[f"plm::{k}"].to_numpy(float), LOG_FLOOR, None) for k in model_keys},
        # The rank estimators take accessibility as it comes. Taking its logarithm first
        # would be harmless in theory -- a logarithm cannot reorder anything -- but in
        # floating point it fuses neighbouring values into ties and moves a Spearman r in
        # the fifth decimal, which is enough to stop these numbers matching Figure 2D.
        acc=acc,
        # Natural log, so the fitted slope is a power-law exponent and nothing has to be
        # rescaled to compare it with an exponent estimated in any other base.
        log_acc=np.log(acc),
        grantham=-wide[control].to_numpy(float),
    )


def _series():
    """Ordered (measure, control) pairs; the layout of every statistic vector.

    ``spearman`` with no control is the plain correlation Figure 2D plots. It is recomputed
    here, from this module's own arrays, purely so a test can check the two routes land on
    the same number; it is not drawn twice.
    """
    return [("spearman", ""), ("matched_rank", "grantham"),
            ("softmax_beta", ""), ("softmax_beta", "grantham")]


def _vector(d: _Data, rows: np.ndarray, cluster: np.ndarray,
            strata, model_keys, bins, knots) -> np.ndarray:
    """Every statistic for one row selection, laid out (stratum, model, series).

    ``cluster`` labels the site each row was drawn as. It is not simply the position:
    a site drawn twice by the bootstrap has to act as two independent sites rather than
    one site with thirty-eight replacements, or the within-site fit would see a cluster
    that never existed.
    """
    out = np.full((len(strata), len(model_keys), len(_series())), np.nan)
    for si, s in enumerate(strata):
        keep = slice(None) if s == ALL else (d.steps[rows] == s)
        sel, gid = rows[keep], cluster[keep]
        if len(sel) < MIN_ROWS:
            continue
        order = np.argsort(gid, kind="stable")
        sel, gid = sel[order], gid[order]
        basis = rcs_basis(d.grantham[sel], knots[s])
        for mi, key in enumerate(model_keys):
            q, acc, log_acc, g = d.q[key][sel], d.acc[sel], d.log_acc[sel], d.grantham[sel]
            out[si, mi, 0] = _spearman(q, acc)
            out[si, mi, 1] = matched_rank(q, acc, g, bins[s])
            out[si, mi, 2] = softmax_beta(q, log_acc, gid)
            out[si, mi, 3] = softmax_beta(q, log_acc, gid, basis)
    return out


def link_table(
    frame: pd.DataFrame,
    model_keys: list[str],
    control: str,
    *,
    n_boot: int = 200,
    seed: int = 20261009,
    level: float = 0.95,
) -> pd.DataFrame:
    """Matched-rank and softmax estimates per stratum and model, with a site bootstrap.

    One row per (measure, series, control, stratum), the same shape
    ``mutacc.strata.strata_table`` returns, so S4 can concatenate the two and draw from one
    table. ``measure`` is ``matched_rank`` (a Spearman r, comparable with the correlations
    beside it) or ``softmax_beta`` (a power-law exponent, which is not). ``control`` is
    empty or the control column's name.

    The interval resamples **sites**, as everywhere else in this figure: replacements at one
    site share a backbone and are not independent, and for the softmax a site is the unit
    the likelihood conditions on, so it is also the unit that has to be resampled.

    BLAS is held to one thread for the duration, which is both neighbourly on a shared
    machine and about five times faster here. The design matrix has at most four columns, so
    a threaded matrix multiply spends all its time in synchronisation: left alone, this call
    took eleven cores and ninety-seven seconds where one core takes twenty.
    """
    with threadpool_limits(limits=1, user_api="blas"):
        return _link_table(frame, model_keys, control, n_boot, seed, level)


def _link_table(frame, model_keys, control, n_boot, seed, level) -> pd.DataFrame:
    d = _prepare(frame, model_keys, control)
    strata = [ALL, *sorted(int(s) for s in np.unique(d.steps))]
    rows = np.arange(len(d.pos))

    # Bins and knots are fixed on the full data, not refitted inside each bootstrap draw:
    # a control whose shape moves with the resample would mix its own noise into the slope.
    bins, knots = {}, {}
    for s in strata:
        g = d.grantham[rows if s == ALL else rows[d.steps == s]]
        bins[s] = np.quantile(g, np.linspace(0, 1, GRANTHAM_BINS + 1))
        bins[s][0], bins[s][-1] = -np.inf, np.inf
        knots[s] = _knots(g)

    point = _vector(d, rows, d.pos, strata, model_keys, bins, knots)

    lo = np.full_like(point, np.nan)
    hi = np.full_like(point, np.nan)
    if n_boot > 0:
        sites = np.unique(d.pos)
        starts = np.searchsorted(d.pos, sites, side="left")
        ends = np.searchsorted(d.pos, sites, side="right")
        widths = ends - starts
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot,) + point.shape)
        for r in range(n_boot):
            picked = rng.integers(0, len(sites), len(sites))
            sel = np.concatenate([np.arange(starts[i], ends[i]) for i in picked])
            # One fresh cluster label per draw, so a site picked twice counts twice.
            cluster = np.repeat(np.arange(len(picked)), widths[picked])
            draws[r] = _vector(d, sel, cluster, strata, model_keys, bins, knots)
        tail = (1.0 - level) / 2.0 * 100.0
        with np.errstate(invalid="ignore"):
            lo, hi = np.nanpercentile(draws, [tail, 100.0 - tail], axis=0)
        enough = np.isfinite(draws).sum(axis=0) >= max(1, n_boot // 2)
        lo, hi = np.where(enough, lo, np.nan), np.where(enough, hi, np.nan)

    rows_out = []
    for si, s in enumerate(strata):
        for mi, key in enumerate(model_keys):
            for ci, (measure, ctrl) in enumerate(_series()):
                n = int(len(rows) if s == ALL else (d.steps == s).sum())
                rows_out.append({
                    "measure": measure, "series": key,
                    "control": control if ctrl else "", "stratum": str(s),
                    "estimate": point[si, mi, ci], "lo": lo[si, mi, ci], "hi": hi[si, mi, ci],
                    "n_rows": n,
                })
    return pd.DataFrame(rows_out)
