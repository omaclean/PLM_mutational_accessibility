"""Small statistics shared by figure scripts. No plotting, no paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "lineage_matrix", "bootstrap_mean_ci", "mean_over_lineages",
    "roc_points", "auroc", "partial_auroc", "mcclish_floor", "pr_auc",
    "winners_labels",
]


def lineage_matrix(
    frame: pd.DataFrame,
    metric: str,
    *,
    model: str,
    lineage_col: str = "lineage",
    alpha_col: str = "alpha",
    model_col: str = "model",
    variant_col: str = "model_variant",
    variant: str = "plm_alpha_sweep",
) -> pd.DataFrame:
    """Lineages (rows) by alpha (columns) for one model and one metric.

    Raises on missing or non-finite cells rather than averaging over them. A
    lineage that silently drops out of one alpha would change what the band
    means from one end of the axis to the other.
    """
    sub = frame.loc[(frame[model_col] == model) & (frame[variant_col] == variant)]
    if sub.empty:
        raise ValueError(f"no rows for model {model!r}, variant {variant!r}")
    if metric not in sub.columns:
        raise KeyError(f"metric column {metric!r} not in table")
    dup = sub.duplicated([lineage_col, alpha_col])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (lineage, alpha) rows for model {model!r}")
    wide = sub.pivot(index=lineage_col, columns=alpha_col, values=metric).sort_index(axis=1)
    if not np.isfinite(wide.to_numpy(dtype=float)).all():
        bad = int((~np.isfinite(wide.to_numpy(dtype=float))).sum())
        raise ValueError(f"{bad} missing or non-finite cells in {metric!r} for model {model!r}")
    return wide


def mean_over_lineages(wide: pd.DataFrame) -> np.ndarray:
    return wide.to_numpy(dtype=float).mean(axis=0)


def bootstrap_mean_ci(
    wide: pd.DataFrame,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 20261009,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval for the mean over rows, at every column.

    Resamples *lineages* (rows) with replacement. This is the precision of the
    mean over the lineages in hand. It is **not** a generalisation interval for
    an unseen lineage: the lineages share their sites, so the band understates
    uncertainty, most for the weaker model (LINEAGE_VALIDITY.md section 4, Q2).
    """
    values = wide.to_numpy(dtype=float)
    n = values.shape[0]
    if n < 2:
        raise ValueError("a bootstrap over a single lineage has no spread; need at least 2")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)  # (n_boot, n_alpha)
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(means, [tail, 100.0 - tail], axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# Ranking metrics. Scores are any real numbers (higher = more likely to occur);
# outcomes are 0/1 with 1 = the replacement was observed.
# ---------------------------------------------------------------------------

def _check(score: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if score.shape != outcome.shape or score.ndim != 1:
        raise ValueError("score and outcome must be 1-D and the same length")
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")
    if not np.isfinite(outcome).all():
        raise ValueError("outcome contains non-finite values")
    if not np.isin(outcome, (0.0, 1.0)).all():
        raise ValueError("outcome must be 0 or 1")
    positives = int((outcome > 0.5).sum())
    if positives == 0 or positives == len(outcome):
        raise ValueError("need both observed and unobserved replacements to rank")
    return score, outcome


def roc_points(score: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False- and true-positive rates at each distinct score.

    Tied scores collapse to a single point, so the curve runs straight across a tie
    block instead of stepping through it in input order.
    """
    score, outcome = _check(score, outcome)
    order = np.argsort(-score, kind="mergesort")
    hit = outcome[order] > 0.5
    ordered = score[order]
    tp = np.cumsum(hit, dtype=float)
    fp = np.cumsum(~hit, dtype=float)
    keep = np.r_[np.diff(ordered) != 0, True]     # last element of each tie block
    tp, fp = tp[keep], fp[keep]
    return np.r_[0.0, fp / fp[-1]], np.r_[0.0, tp / tp[-1]]


def auroc(score: np.ndarray, outcome: np.ndarray) -> float:
    """Area under the ROC curve. 0.5 is chance for every data set, whatever its prevalence."""
    fpr, tpr = roc_points(score, outcome)
    return float(np.trapezoid(tpr, fpr))


def mcclish_floor(cap: float) -> float:
    """Standardised partial AUROC of a ranker that puts every observed replacement last.

    The standardisation maps the random floor (cap^2 / 2) to 0.5 and the perfect
    ceiling (cap) to 1.0, so a raw area of zero lands at 0.5 * (1 - cap / (2 - cap)),
    not at zero: 0.4872 for cap = 0.05. The metric cannot go lower, so everything
    worse than chance is compressed into the gap between that and 0.5.
    """
    return 0.5 * (1.0 - cap / (2.0 - cap))


def partial_auroc(fpr: np.ndarray, tpr: np.ndarray, cap: float) -> tuple[float, float]:
    """ROC area over false-positive rate <= cap: (raw, McClish-standardised).

    Ordinary AUROC credits ranking anywhere in the list, including a tail no
    experiment would reach. Restricting to a small false-positive rate keeps only the
    range where the number of false alarms is small enough to act on. Chance is 0.5
    and perfect is 1.0 whatever cap is chosen.
    """
    if not 0.0 < cap <= 1.0:
        raise ValueError(f"cap must be in (0, 1], got {cap}")
    # First point at or beyond the cap. The ROC path can run straight up at a fixed fpr (a
    # block of observed replacements), so interpolating with np.interp at an fpr that is
    # *equal* to such a block returns the top of the block and overstates the area. The curve
    # reaches ``cap`` at the first point with fpr >= cap, so use that point's tpr.
    idx = int(np.searchsorted(fpr, cap, side="left"))
    if fpr[idx] == cap:
        t_cap = tpr[idx]
    else:
        f0, f1, t0, t1 = fpr[idx - 1], fpr[idx], tpr[idx - 1], tpr[idx]
        t_cap = t0 + (t1 - t0) * (cap - f0) / (f1 - f0)
    f = np.r_[fpr[:idx], cap]
    t = np.r_[tpr[:idx], t_cap]
    raw = float(np.trapezoid(t, f))
    floor, ceiling = cap * cap / 2.0, cap
    return raw, 0.5 * (1.0 + (raw - floor) / (ceiling - floor))


def pr_auc(score: np.ndarray, outcome: np.ndarray) -> float:
    """Area under the precision-recall curve, as a step sum of precision at each hit.

    A random ranker scores the prevalence, not a fixed number, so raw values are not
    comparable between data sets with different prevalence; divide by prevalence for
    fold enrichment over chance (chance = 1). Tied scores form one threshold (as in scikit-learn's
    average precision), so the value does not depend on the order of tied rows.
    """
    score, outcome = _check(score, outcome)
    order = np.argsort(-score, kind="mergesort")
    hit = outcome[order] > 0.5
    ordered = score[order]
    tp = np.cumsum(hit, dtype=float)
    fp = np.cumsum(~hit, dtype=float)
    # Evaluate only at the end of each block of tied scores, so the result does not depend on the
    # order in which tied rows happen to be listed.
    keep = np.r_[np.diff(ordered) != 0, True]
    tp, fp = tp[keep], fp[keep]
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def winners_labels(freq: np.ndarray, top: float = 0.05) -> np.ndarray:
    """1 for the most frequent ``top`` share of a lineage's observed replacements, 0 for the rest.

    ``freq`` holds only the replacements observed at all; leaving the zero-frequency ones out
    is the whole point of the metric, and passing one in is an error rather than a row that
    quietly joins the negatives. The cut is this lineage's own ``1 - top`` quantile, so a
    lineage with more observed replacements gets a larger positive class rather than a
    different threshold.

    Equality joins the positives, which keeps the cut from splitting a block of tied
    frequencies by position in the table. It also means a lineage whose frequencies are all
    equal has no negatives; that, and a positive class that comes out empty, raise. A
    degenerate lineage is not dropped: the band is a bootstrap over lineages, and one that
    silently held a different set at one end of the axis than the other would not mean
    anything.
    """
    freq = np.asarray(freq, dtype=float)
    if freq.ndim != 1 or freq.size == 0:
        raise ValueError("winners_labels needs a non-empty 1-D array of frequencies")
    if not np.isfinite(freq).all() or (freq <= 0).any():
        raise ValueError("winners_labels is defined on observed replacements only: every "
                         "frequency must be finite and positive")
    if not 0.0 < top < 1.0:
        raise ValueError(f"top must be in (0, 1), got {top}")
    label = (freq >= np.quantile(freq, 1.0 - top)).astype(float)
    n_pos = int(label.sum())
    if n_pos == 0 or n_pos == freq.size:
        raise ValueError(
            f"the top {top:.0%} cut leaves {n_pos} of {freq.size} observed replacements as "
            "winners; this lineage has no usable contrast and cannot be scored"
        )
    return label
