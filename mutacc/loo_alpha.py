"""Leave-one-lineage-out selection of the accessibility weight.

No plotting, no paths. Used by ``tools/export_sW_loo_alpha.py``.

Figure 3 sweeps the weight alpha, averages each curve over the lineages, and reads its
result off the peak of that average. The peak and the number reported at it come from the
same thirteen lineages, so whatever part of the peak is a property of *these* lineages
rather than of the weighting is counted as a gain. Reviewer 2 put it in one sentence:
"alpha is chosen on the data it is scored on" (``PREPRINT_AND_REVIEWS.md``, Reviewer 2,
uncertainty row).

The interval already beside that figure does not answer it. ``metrics.bootstrap_mean_ci``
resamples lineages at a **fixed** alpha and reports the precision of the mean at each
weight; it never re-selects the weight inside a replicate, so a weight chosen on the full
set stays chosen in every draw. The circularity is in the selection step, and only a
selection step that cannot see the data it is scored on removes it.

What this module does
---------------------

For one model and one metric, the sweep is the matrix ``metrics.lineage_matrix`` returns:
thirteen lineages by 101 weights. Hold lineage *L* out, average the other twelve curves,
take the weight at that average's peak -- ``alpha*_L``, chosen without *L* ever being
looked at -- and score *L* at it. The gain is that score minus *L*'s score at alpha = 0,
which is the model with no accessibility weighting at all and the reference the whole
figure is read against.

Thirteen folds give thirteen held-out weights and thirteen gains. What is reported is the
mean gain, a percentile interval for it, and how many of the thirteen folds gained at all.
The spread of ``alpha*`` across folds is itself a result: a weight that moves from fold to
fold is a weight the data does not pin down, whatever the mean curve's peak looks like.

**The interval is over folds, not over the selection.** ``bootstrap_mean_ci`` is reused so
that it is the same estimator the figures use, but what it resamples here are the thirteen
*already held-out* gains. It is the precision of their mean. It is not a second validation
of the selection, and -- as everywhere else in this project -- it is not a generalisation
interval for an unseen lineage, because the lineages share their sites
(``LINEAGE_VALIDITY.md`` section 4).

Ties
----

The peak of a mean curve can be a plateau: two neighbouring weights can leave the ranking
of every replacement unchanged, and then AUROC and PR-AUC are equal at both to the last
bit. ``select_alpha`` breaks such a tie toward the weight **nearest zero**, and, between
two equally distant, toward the lower (more negative) one. Nearest zero because the
question being asked is how much reweighting the data actually buys: when several weights
are tied, the honest answer is the one that reweights least, which is also the one that
makes the reported gain smallest. ``numpy.argmax``'s own rule -- the first index -- would
instead always take the left end of a plateau, which is an arbitrary direction on an axis
that runs from -1 to +1 through the quantity of interest.

Why there is no p-value here
----------------------------

Deliberately absent, and not an oversight. A sign test or a paired t-test over the folds
would treat the thirteen lineages as independent replicates. They are not: their reference
proteins are 96.6 to 99.7% identical, their per-lineage score vectors correlate at 0.985
and above, residual outcome co-occurrence between lineages is about 0.391, and the
effective number of independent lineages is between 1.01 and 1.12 -- one, not thirteen
(``PREPRINT_AND_REVIEWS.md``, "The lineages are not independent"). Any such p-value is
anti-conservative by roughly that factor. A sign test has a second problem on top: 13 of 13
is its floor of 2 / 2^13 whatever the effect size, so it would report the fold count and
call it evidence. The effect size, its interval and the fold count are reported instead,
and they are what a reader should quote.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from mutacc.metrics import bootstrap_mean_ci, lineage_matrix

__all__ = [
    "ALPHA_REFERENCE", "FOLD", "MEAN", "select_alpha", "fold_table", "fold_summary",
    "loo_alpha_table",
]

#: The weight every gain is measured against: the model with no accessibility weighting.
#: Figure 3 reads its curves off the same rule (``make_fig03.ALPHA_REFERENCE``).
ALPHA_REFERENCE = 0.0

#: Values of the ``scope`` column: one row per held-out lineage, plus one summary row.
FOLD, MEAN = "fold", "mean"

#: How close a grid weight must be to ``reference`` to count as it. The sweep grid is in
#: steps of 0.02 and carries an exact 0.0, so this only absorbs the representation error of
#: a grid written out and read back as text.
_REFERENCE_TOL = 1e-9


def select_alpha(alphas: np.ndarray, curve: np.ndarray) -> float:
    """The weight at the peak of ``curve``, ties broken toward zero then downward.

    ``alphas`` and ``curve`` are the grid and the values on it. Ties are real -- adjacent
    weights that leave every ranking unchanged give identical AUROCs -- so the rule is
    stated rather than left to ``argmax``; see the module docstring for why it points at
    zero. Raises on a non-finite value rather than letting ``nanargmax``-like behaviour
    pick a peak out of a curve with a hole in it.
    """
    alphas = np.asarray(alphas, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if alphas.ndim != 1 or alphas.shape != curve.shape:
        raise ValueError("alphas and curve must be 1-D and the same length")
    if not np.isfinite(curve).all():
        raise ValueError("curve contains non-finite values; a peak cannot be located")
    tied = np.flatnonzero(curve == curve.max())
    # lexsort takes the LAST key as primary: distance from zero first, then the weight
    # itself, so two equally distant weights resolve to the lower one.
    return float(alphas[tied[np.lexsort((alphas[tied], np.abs(alphas[tied])))[0]]])


def fold_table(wide: pd.DataFrame, *, reference: float = ALPHA_REFERENCE) -> pd.DataFrame:
    """One row per held-out lineage: the weight chosen without it, and what it bought.

    ``wide`` is lineages by weights, as ``metrics.lineage_matrix`` returns it -- that
    function is also what keeps the non-sweep rows out, since the accessibility-only
    baseline carries no alpha at all and is a different ``model_variant``.

    Columns: ``held_out_lineage``, ``alpha_star`` (chosen on the other lineages),
    ``n_tied_alpha`` (how many weights shared that peak), ``metric_at_alpha_star``,
    ``metric_at_reference`` and ``gain``, the difference between the last two.

    Every lineage is held out exactly once, so the fold count is the lineage count.
    """
    alphas = wide.columns.to_numpy(dtype=float)
    values = wide.to_numpy(dtype=float)
    if values.shape[0] < 2:
        raise ValueError("leaving one lineage out of one lineage leaves nothing to select on")
    at_reference = np.flatnonzero(np.abs(alphas - reference) <= _REFERENCE_TOL)
    if len(at_reference) != 1:
        raise ValueError(
            f"{len(at_reference)} columns at alpha = {reference}, expected exactly 1; "
            f"the gain is measured against that column"
        )
    ref = int(at_reference[0])

    rows = []
    for i, lineage in enumerate(wide.index):
        others = np.delete(values, i, axis=0).mean(axis=0)
        star = select_alpha(alphas, others)
        best = int(np.flatnonzero(alphas == star)[0])
        rows.append({
            "held_out_lineage": str(lineage),
            "alpha_star": star,
            "n_tied_alpha": int((others == others.max()).sum()),
            "metric_at_alpha_star": float(values[i, best]),
            "metric_at_reference": float(values[i, ref]),
            "gain": float(values[i, best] - values[i, ref]),
        })
    return pd.DataFrame(rows)


def fold_summary(folds: pd.DataFrame, *, n_boot: int = 2000, seed: int = 20261009,
                 level: float = 0.95) -> dict[str, float]:
    """Mean gain over the folds, its percentile interval, and the sign consistency count.

    The bootstrap is ``metrics.bootstrap_mean_ci`` over the folds' gains, for the same
    reason every other interval in this project is: one estimator, so two intervals printed
    beside each other mean the same thing. It resamples folds whose weights were already
    chosen out of sample; it does not re-run the selection, and it is not asked to.

    No p-value is returned. The lineages are not independent replicates and the fold count
    is not a test statistic -- see the module docstring.
    """
    gains = folds.set_index("held_out_lineage")[["gain"]]
    lo, hi = bootstrap_mean_ci(gains, n_boot=n_boot, seed=seed, level=level)
    return {
        "n_folds": int(len(folds)),
        "n_folds_gain_positive": int((folds["gain"] > 0).sum()),
        "mean_gain": float(folds["gain"].mean()),
        "ci_lo": float(lo[0]),
        "ci_hi": float(hi[0]),
    }


def loo_alpha_table(
    frame: pd.DataFrame,
    model_keys: Sequence[str],
    metrics: Sequence[str],
    *,
    reference: float = ALPHA_REFERENCE,
    n_boot: int = 2000,
    seed: int = 20261009,
    level: float = 0.95,
) -> pd.DataFrame:
    """Held-out weights and gains for every (metric, model), folds and summary in one frame.

    One row per fold with ``scope = "fold"``, then one row with ``scope = "mean"`` carrying
    the mean gain in ``estimate`` with ``ci_lo`` and ``ci_hi`` around it. The two scopes
    share the ``estimate`` column on purpose: it is the same quantity, a gain in the metric,
    for one held-out lineage or averaged over all of them, and a reader plotting the column
    gets the folds and their mean on one scale. The columns that belong to a single fold --
    the chosen weight and the two scores differenced to make the gain -- are empty on the
    summary row, and the two counts that belong to the summary are empty on a fold row.
    """
    rows = []
    for metric in metrics:
        for model in model_keys:
            wide = lineage_matrix(frame, metric, model=model)
            folds = fold_table(wide, reference=reference)
            summary = fold_summary(folds, n_boot=n_boot, seed=seed, level=level)
            for fold in folds.itertuples(index=False):
                rows.append({
                    "metric": metric, "model": model, "scope": FOLD,
                    "held_out_lineage": fold.held_out_lineage,
                    "alpha_star": fold.alpha_star,
                    "n_tied_alpha": fold.n_tied_alpha,
                    "metric_at_alpha_star": fold.metric_at_alpha_star,
                    "metric_at_reference": fold.metric_at_reference,
                    "estimate": fold.gain,
                    "ci_lo": np.nan, "ci_hi": np.nan,
                    "n_folds": summary["n_folds"],
                    "n_folds_gain_positive": np.nan,
                })
            rows.append({
                "metric": metric, "model": model, "scope": MEAN,
                "held_out_lineage": "",
                "alpha_star": np.nan, "n_tied_alpha": np.nan,
                "metric_at_alpha_star": np.nan, "metric_at_reference": np.nan,
                "estimate": summary["mean_gain"],
                "ci_lo": summary["ci_lo"], "ci_hi": summary["ci_hi"],
                "n_folds": summary["n_folds"],
                "n_folds_gain_positive": summary["n_folds_gain_positive"],
            })
    return pd.DataFrame(rows)
