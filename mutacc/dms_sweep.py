"""The alpha sweep against measured deep-mutational-scanning phenotypes.

Score = log(model probability) + alpha * log(accessibility). For each alpha the score
is rank-correlated (Spearman) with a measured phenotype over every replacement the scan
measured. This module holds the arithmetic; ``figures/make_fig04.py`` draws it.

Ported from the exploration archive's ``a37_dms_step_collapse.py`` (rank machinery) and
``a31_alpha_vs_dms_plot.py`` (curve, site bootstrap, optimum flags, reversible spectra).
The ranks are tie-averaged exactly as ``scipy.stats.spearmanr`` does: at alpha = 0 the
score is the model probability alone and its 20 x L profile repeats values, and breaking
those ties arbitrarily moves rho by ~4e-6, enough to fail the 1e-6 self-check.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

__all__ = [
    "col_ranks", "rho_cols", "sweep_curve", "band_knots", "site_blocks", "site_bootstrap_band",
    "curve_flags", "STANDARD_CODE", "BASES", "accessibility_table", "log_accessibility",
    "WEAK_SIGNAL_RHO", "MARKED_GAIN", "curve_has_signal", "spearman", "site_cluster_ci", "paired_site_gap",
]

BASES = "ACGT"

#: Spearman |rho| at alpha = 0 below which a curve has no signal for the sweep to improve.
#: a31's own threshold (``abs(rho_alpha0) < 0.05``).
WEAK_SIGNAL_RHO = 0.05

#: Improvement in rho over alpha = 0 below which the sweep has found nothing, so the argmax is
#: wherever numerical noise happens to peak rather than an optimum. Paired with the weak test
#: rather than used alone: several base-model curves are strong (rho ~ 0.45) but flat in alpha,
#: and their optimum near zero is a real finding ("no weighting helps"), not an absent one.
MARKED_GAIN = 0.0005

#: Standard genetic code, first base slowest, bases in T, C, A, G order. Pinned against
#: Biopython's table 1 in tests/test_dms_sweep.py; it is not typed from memory in use.
STANDARD_CODE = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_CODE_BASES = "TCAG"


# ---------------------------------------------------------------------------
# Ranks and curves
# ---------------------------------------------------------------------------

def col_ranks(X: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks down each column of X, counted from 0 (scipy's ``rankdata`` minus 1).

    A constant offset does not matter to a correlation, which centres the ranks anyway.
    """
    n = X.shape[0]
    order = np.argsort(X, axis=0, kind="stable")
    Xs = np.take_along_axis(X, order, axis=0)
    pos = np.arange(n, dtype=float)[:, None]
    new = np.empty(Xs.shape, bool)
    new[0] = True
    new[1:] = Xs[1:] != Xs[:-1]
    first = np.maximum.accumulate(np.where(new, pos, -1.0), axis=0)
    nxt = np.where(new, pos, float(n))
    after = np.full(Xs.shape, float(n))
    after[:-1] = np.minimum.accumulate(nxt[::-1], axis=0)[::-1][1:]
    out = np.empty(X.shape)
    np.put_along_axis(out, order, (first + after - 1.0) / 2.0, axis=0)
    return out


def rho_cols(X: np.ndarray, y_rank_centred: np.ndarray) -> np.ndarray:
    """Spearman correlation of every column of X with an already ranked, centred phenotype."""
    ranks = col_ranks(X)
    ranks -= ranks.mean(axis=0)
    denom = np.sqrt((ranks * ranks).sum(axis=0) * (y_rank_centred @ y_rank_centred))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (ranks.T @ y_rank_centred) / denom, np.nan)


def sweep_curve(log_plm: np.ndarray, log_acc: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Spearman rho between ``log_plm + alpha * log_acc`` and ``y`` at every alpha."""
    ranked = rankdata(y)
    ranked = ranked - ranked.mean()
    return rho_cols(log_plm[:, None] + alphas[None, :] * log_acc[:, None], ranked)


def band_knots(alphas: np.ndarray, every: int) -> np.ndarray:
    """Every ``every``-th alpha, always including the last."""
    knots = alphas[::every]
    if knots[-1] != alphas[-1]:
        knots = np.append(knots, alphas[-1])
    return knots


def site_blocks(sites: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """The distinct sites and, for each, the row indices belonging to it.

    The one definition of what a site bootstrap resamples. Everything that draws sites with
    replacement -- the band below, the within-stratum intervals, the optimum bootstrap in
    ``mutacc.dms_alpha_ci`` -- draws indices into these blocks, so two intervals printed
    beside each other are resampling the same thing in the same way.
    """
    uniq, inverse = np.unique(sites, return_inverse=True)
    return uniq, [np.flatnonzero(inverse == i) for i in range(len(uniq))]


def site_bootstrap_band(
    log_plm: np.ndarray,
    log_acc: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    alphas: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int,
    band_every: int = 5,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile band by cluster bootstrap over SITES, interpolated between knots.

    Replacements at one site share a backbone and a local structural context, so the
    site, not the replacement, is the independent unit. The band is computed on every
    ``band_every``-th alpha and interpolated: a percentile envelope over a couple of
    hundred replicates is smooth on this scale and costs a fifth as much. Nothing
    quantitative should be read off the band between knots.

    The generator is advanced in place, one draw per replicate, so the band depends on
    the order arms are fed in; callers wanting reproducible bands keep a fixed order.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    knots = band_knots(alphas, band_every)
    uniq, by_site = site_blocks(sites)
    reps = np.empty((n_boot, len(knots)))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by_site[i] for i in pick])
        reps[b] = sweep_curve(log_plm[idx], log_acc[idx], y[idx], knots)
    tail = (1.0 - level) / 2.0 * 100.0
    lo = np.interp(alphas, knots, np.nanpercentile(reps, tail, axis=0))
    hi = np.interp(alphas, knots, np.nanpercentile(reps, 100.0 - tail, axis=0))
    return lo, hi


def curve_flags(rho: np.ndarray, alphas: np.ndarray, *, absolute: bool = False) -> dict:
    """The optimum of a curve and whether it can be read as one.

    ``boundary``: the best alpha sits at the end of the swept range, so the argmax is
    wherever dilution hurts least, not a peak. ``weak_signal``: |rho| at alpha = 0 is
    below :data:`WEAK_SIGNAL_RHO`, so there is nothing for the sweep to recover.
    ``interpretable`` is neither.

    ``absolute`` takes the optimum at the largest |rho| rather than the largest rho. The
    default (largest rho) is a37/a31's rule and is what the shipped summary was built with,
    but it only reads a curve correctly where the phenotype runs with fitness. On a curve
    that is negative throughout -- a trait measured in the opposite direction, such as
    receptor affinity -- maximising rho just dilutes the model towards zero and the argmax
    runs to whichever end of the range dilutes hardest, hiding a real interior optimum.
    Under ``absolute`` that optimum is found; ``rho_best`` keeps its sign either way, so a
    reader can still see which direction the curve runs in.

    Note that reading a curve on |rho| concedes the sign of the trait. That has to be
    justified by the assay in advance: a rule that lets every panel choose its own sign
    after the fact will find a peak on noise.
    """
    rho = np.asarray(rho, float)
    if not np.isfinite(rho).any():
        raise ValueError("curve has no finite values")
    k = int(np.nanargmax(np.abs(rho) if absolute else rho))
    # NOTE ``weak_signal`` below reads |rho| at alpha = 0 only. For a curve whose optimum
    # lies away from zero that is the wrong place to look -- see :func:`curve_has_signal`.
    i0 = int(np.argmin(np.abs(alphas)))
    boundary = bool(abs(alphas[k]) >= np.abs(alphas).max() - 1e-9)
    weak = bool(abs(rho[i0]) < WEAK_SIGNAL_RHO)
    return {
        "best_alpha": float(alphas[k]), "rho_best": float(rho[k]), "rho_alpha0": float(rho[i0]),
        "boundary": boundary, "weak_signal": weak, "interpretable": bool(not boundary and not weak),
    }


def curve_has_signal(rho: np.ndarray, *, threshold: float = WEAK_SIGNAL_RHO) -> bool:
    """Whether a curve has any signal for the sweep to work with, judged over the whole sweep.

    :func:`curve_flags` answers this with ``|rho|`` **at alpha = 0**, which is a31's rule.
    That is the right question for a curve whose optimum sits near zero and the wrong one
    for a curve that merely passes through a shallow point there and peaks elsewhere. The
    case that forced this: BA.2 cell entry, base model, within the 3-nucleotide stratum
    reads ``rho = 0.042`` at alpha = 0 and ``0.084`` at its optimum. Under the alpha = 0
    rule it is set aside as having nothing to recover, while the curve plainly has a peak
    and a real gain; under this one it is kept.

    Taking the maximum over the sweep rather than one point cannot make a flat curve look
    good -- a curve that is everywhere below the threshold still fails -- so this is strictly
    the weaker claim, and the one that matches what the flag is for.
    """
    rho = np.asarray(rho, float)
    if not np.isfinite(rho).any():
        raise ValueError("curve has no finite values")
    return bool(np.nanmax(np.abs(rho)) >= threshold)


# ---------------------------------------------------------------------------
# Within-stratum correlations and the paired gap between two models (S12)
# ---------------------------------------------------------------------------

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho; NaN for fewer than 3 points or a constant input (as the archive's helper)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = rankdata(x) - (len(x) + 1) / 2, rankdata(y) - (len(y) + 1) / 2
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def site_cluster_ci(
    x: np.ndarray, y: np.ndarray, sites: np.ndarray, rng: np.random.Generator, *, n_boot: int, level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval for Spearman(x, y), resampling SITES with replacement.

    Replacements at one site share a backbone and a measurement batch, so the site is the
    independent unit. Draws are ``rng.choice(unique sites, size=n_sites)``, one call per
    replicate, exactly as the archive's ``plot_dms_within_stratum.py`` makes them, so a
    generator seeded the same and fed the same cells in the same order reproduces its interval.
    Replicates with no defined correlation are dropped before the percentiles.
    """
    uniq, blocks = site_blocks(sites)
    if len(uniq) < 3:
        return float("nan"), float("nan")
    vals = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(np.arange(len(uniq)), size=len(uniq), replace=True)
        idx = np.concatenate([blocks[i] for i in drawn])
        vals[b] = spearman(x[idx], y[idx])
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(vals, tail)), float(np.percentile(vals, 100.0 - tail))


def paired_site_gap(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, sites: np.ndarray, rng: np.random.Generator,
    *, n_boot: int, level: float = 0.95,
) -> dict:
    """Spearman(x_b, y) - Spearman(x_a, y) with a PAIRED site bootstrap.

    Both scores are evaluated on the same resampled sites in every replicate, so what the
    interval measures is the gap, not the sum of two independent uncertainties. Two intervals
    that overlap can still hide a gap that excludes zero (they share their sampling noise), and
    two that do not overlap are not a test: this is the test. ``p`` is the two-sided bootstrap
    p-value, ``2 * min(P(gap <= 0), P(gap >= 0))``, floored at ``1 / n_boot`` and capped at 1; it carries no
    multiplicity correction.
    """
    uniq, blocks = site_blocks(sites)
    gap = spearman(x_b, y) - spearman(x_a, y)
    if len(uniq) < 3:
        return {"gap": gap, "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([blocks[i] for i in picked])
        draws[b] = spearman(x_b[idx], y[idx]) - spearman(x_a[idx], y[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size < 2:
        return {"gap": gap, "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    tail = (1.0 - level) / 2.0 * 100.0
    below, above = float((draws <= 0).mean()), float((draws >= 0).mean())
    return {
        "gap": gap, "lo": float(np.percentile(draws, tail)), "hi": float(np.percentile(draws, 100.0 - tail)),
        "p": float(min(1.0, max(2.0 * min(below, above), 1.0 / draws.size))),
    }


# ---------------------------------------------------------------------------
# Accessibility from a 4 x 4 nucleotide matrix
# ---------------------------------------------------------------------------

def _codon_aa_pairs() -> tuple[list[str], dict[str, list[str]]]:
    codons = ["".join(c) for c in itertools.product(_CODE_BASES, repeat=3)]
    aa_to_codons: dict[str, list[str]] = {}
    for codon, aa in zip(codons, STANDARD_CODE):
        if aa != "*":
            aa_to_codons.setdefault(aa, []).append(codon)
    return codons, aa_to_codons


def accessibility_table(matrix: pd.DataFrame, *, reversible: bool) -> pd.DataFrame:
    """64 codons x 20 amino acids: summed probability that the codon mutates to each amino acid.

    ``matrix`` is a 4 x 4 nucleotide change table (rows = from, columns = to, bases
    A, C, G, T); its diagonal is ignored. ``reversible=True`` symmetrises it so that
    rate(x -> y) = rate(y -> x), which removes all directional information from the
    spectrum. A synonymous codon's own probability stays on the diagonal as 1 - row sum.
    """
    m = matrix.loc[list(BASES), list(BASES)].to_numpy(dtype=float).copy()
    np.fill_diagonal(m, 0.0)
    if reversible:
        m = (m + m.T) / 2.0
    probs = m.copy()
    for i in range(4):
        probs[i, i] = 1.0 - m[i].sum()
    index = {b: i for i, b in enumerate(BASES)}

    codons, aa_to_codons = _codon_aa_pairs()
    codon_to_codon = np.empty((64, 64))
    for i, c_from in enumerate(codons):
        for j, c_to in enumerate(codons):
            p = 1.0
            for k in range(3):
                p *= probs[index[c_from[k]], index[c_to[k]]]
            codon_to_codon[i, j] = p
    position = {c: i for i, c in enumerate(codons)}
    out = {}
    for aa, targets in sorted(aa_to_codons.items()):
        out[aa] = codon_to_codon[:, [position[t] for t in targets]].sum(axis=1)
    return pd.DataFrame(out, index=codons)


def log_accessibility(codons, aas, table: pd.DataFrame) -> np.ndarray:
    """log10 accessibility per (reference codon, replacement amino acid).

    Zeros are floored at a tenth of the smallest positive value (as the archive's
    ``safe_log10`` does) so no step class collapses to ties. Raises on a codon or amino
    acid the table does not hold, rather than returning NaN into a correlation.
    """
    codons = np.asarray(codons)
    aas = np.asarray(aas)
    unknown_c = set(codons) - set(table.index)
    unknown_a = set(aas) - set(table.columns)
    if unknown_c or unknown_a:
        raise ValueError(f"not in the accessibility table: codons {sorted(unknown_c)}, amino acids {sorted(unknown_a)}")
    values = table.to_numpy()[table.index.get_indexer(codons), table.columns.get_indexer(aas)]
    positive = values[np.isfinite(values) & (values > 0)]
    floor = positive.min() / 10.0 if positive.size else 1e-300
    return np.log10(np.clip(values, floor, None))
