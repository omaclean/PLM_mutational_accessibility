"""Does a score track the *direction* of a mutation spectrum, or only its overall shape?

The question behind it. A language model's probabilities correlate with mutational accessibility.
Accessibility is directional (influenza's ADAR editing makes A>G far commoner than G>A), but the
symmetrised spectrum (Q + Q^T)/2, which has no direction at all, correlates with the model *better*
than the real one. Is that evidence that the model is not directional?

It is not, on its own, because the same ordering appears for scores that contain no nucleotide
information (BLOSUM62, Grantham, hydropathy, volume). Every test here is therefore run on the
language models **and** on those controls, and several are calibrated against a signal of known
direction planted into a control. A result about a model means something only relative to them.

Tests
-----
``orientation_battery``   correlation with the spectrum, its transpose, its complement and its
                          symmetrisation, with paired site-bootstrap intervals on the differences.
``null_battery``          where the real spectrum ranks among all 24 relabellings of the bases
                          and among random shuffles of its rates.
``flip_null``             where the real direction assignment ranks among all 64 ways of flipping the
                          six reciprocal base pairs. Every version has the same rates and the same
                          symmetric structure; only direction differs. This is the null that isolates
                          direction (the shuffle null also destroys transition/transversion structure).
``direction_contrast``    the direction signal itself: do scores favour changes in the spectrum's
                          favoured direction *after the unordered amino-acid pair is absorbed*?
                          A symmetric score must read exactly zero here.
``planted_signal``        dose-response: mix a known amount of directional accessibility into a
                          control and watch how each statistic responds.
``rank_robustness``       the same differences with random tie-breaking and on log values.

Units, sites and strata. Rows belong to *units* (lineages, or DMS datasets). Within a unit,
correlations are Spearman within a stratum of nucleotide steps (1, 2, 3) or pooled ("all"), averaged
over units. Intervals resample SITES with the same draw applied to every unit that shares them,
because rows at one site are not independent and lineages share sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mutacc import accessibility as acc

__all__ = [
    "STRATA", "Unit", "units_from_frame", "accessibility_for", "orientation_variants",
    "orientation_battery", "null_battery", "flip_null", "direction_contrast", "planted_signal",
    "rank_robustness", "favoured_classes",
]

STRATA = (1, 2, 3, "all")
_CONTRASTS = (("direct", "symmetrised"), ("direct", "transposed"), ("direct", "complemented"))


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """One set of rows that are analysed together (a lineage, or a DMS dataset)."""

    name: str
    site: np.ndarray             # integer site id per row
    steps: np.ndarray            # nucleotide steps per row
    ref_codon: np.ndarray        # codon index per row
    aa: np.ndarray               # target amino-acid index per row
    ref_aa: np.ndarray           # reference amino-acid index per row
    scores: dict[str, np.ndarray]
    _rows_by_site: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        n = len(self.site)
        for k in ("steps", "ref_codon", "aa", "ref_aa"):
            if len(getattr(self, k)) != n:
                raise ValueError(f"{self.name}: {k} has {len(getattr(self, k))} rows, expected {n}")
        for k, v in self.scores.items():
            if len(v) != n:
                raise ValueError(f"{self.name}: score {k!r} has {len(v)} rows, expected {n}")
            if not np.isfinite(v).all():
                raise ValueError(f"{self.name}: score {k!r} has non-finite values")
        order = np.argsort(self.site, kind="stable")
        sites, start = np.unique(self.site[order], return_index=True)
        self._rows_by_site = dict(zip(sites.tolist(), np.split(order, start[1:])))

    @property
    def n(self) -> int:
        return len(self.site)

    @property
    def sites(self) -> np.ndarray:
        return np.array(sorted(self._rows_by_site))

    def rows(self, site_draw: np.ndarray | None) -> np.ndarray:
        """Row indices for a resample of sites (``None`` = every row). Missing sites are skipped."""
        if site_draw is None:
            return np.arange(self.n)
        parts = [self._rows_by_site[s] for s in site_draw if s in self._rows_by_site]
        if not parts:
            raise ValueError(f"{self.name}: the site draw hit no rows")
        return np.concatenate(parts)


def units_from_frame(
    frame: pd.DataFrame,
    score_cols: Sequence[str],
    *,
    unit_col: str = "lineage",
    site_col: str = "position",
    codon_col: str = "ref_codon",
    aa_col: str = "aa",
    ref_aa_col: str = "ref_aa",
    steps_col: str = "nt_steps",
) -> list[Unit]:
    """Split a long table into units. Refuses missing columns, NaNs and unknown amino acids."""
    need = [unit_col, site_col, codon_col, aa_col, ref_aa_col, steps_col, *score_cols]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise KeyError(f"table lacks columns {missing}")
    if frame[need].isna().any().any():
        raise ValueError("table has missing values in the columns the battery reads")
    units = []
    for name, g in frame.groupby(unit_col, sort=False):
        units.append(Unit(
            name=str(name),
            site=g[site_col].to_numpy(int),
            steps=g[steps_col].to_numpy(int),
            ref_codon=np.array([acc.codon_index(c) for c in g[codon_col]]),
            aa=np.array([acc.aa_index(a) for a in g[aa_col]]),
            ref_aa=np.array([acc.aa_index(a) for a in g[ref_aa_col]]),
            scores={c: g[c].to_numpy(float) for c in score_cols},
        ))
    if not units:
        raise ValueError("no units")
    return units


def accessibility_for(unit: Unit, q: np.ndarray, *, row_sum: float | None = None) -> np.ndarray:
    return acc.codon_aa_table(q, row_sum=row_sum)[unit.ref_codon, unit.aa]


def orientation_variants(q: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "direct": np.asarray(q, float),
        "symmetrised": acc.symmetrise(q),
        "transposed": acc.transpose(q),
        "complemented": acc.complement(q),
    }


# ---------------------------------------------------------------------------
# Core statistic
# ---------------------------------------------------------------------------

def _mask(unit: Unit, rows: np.ndarray, stratum) -> np.ndarray:
    return rows if stratum == "all" else rows[unit.steps[rows] == stratum]


def _rho_block(columns: np.ndarray, n_scores: int) -> np.ndarray:
    """Spearman between the first ``n_scores`` columns and the rest. NaN when a column is constant."""
    ranks = rankdata(columns, axis=0)
    z = ranks - ranks.mean(axis=0)
    norm = np.sqrt((z * z).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = z / norm
    return z[:, :n_scores].T @ z[:, n_scores:]


def _rho_tensor(
    units: Sequence[Unit],
    score_names: Sequence[str],
    targets: Mapping[str, Sequence[np.ndarray]],
    target_names: Sequence[str],
    draw: np.ndarray | None,
    strata: Sequence,
) -> np.ndarray:
    """(score, target, stratum) tensor of mean-over-units Spearman correlations."""
    out = np.full((len(score_names), len(target_names), len(strata)), np.nan)
    acc_sum = np.zeros_like(out)
    count = np.zeros(len(strata))
    for ui, u in enumerate(units):
        base_rows = u.rows(draw)
        for si, s in enumerate(strata):
            r = _mask(u, base_rows, s)
            if len(r) < 10:
                continue
            cols = [u.scores[n][r] for n in score_names] + [targets[t][ui][r] for t in target_names]
            rho = _rho_block(np.column_stack(cols), len(score_names))
            if np.isnan(rho).any():
                continue
            acc_sum[:, :, si] += rho
            count[si] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = acc_sum / count[None, None, :]
    out[:, :, count == 0] = np.nan
    return out


def _draws(units: Sequence[Unit], n_boot: int, seed: int) -> list[np.ndarray]:
    sites = np.unique(np.concatenate([u.sites for u in units]))
    rng = np.random.default_rng(seed)
    return [rng.choice(sites, size=len(sites), replace=True) for _ in range(n_boot)]


def _ci(samples: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    tail = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(samples, [tail, 100 - tail], axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# 1. Orientation battery
# ---------------------------------------------------------------------------

def orientation_battery(
    units: Sequence[Unit],
    q: np.ndarray,
    score_names: Sequence[str],
    *,
    strata: Sequence = STRATA,
    n_boot: int = 200,
    seed: int = 20261009,
    row_sum: float | None = None,
) -> pd.DataFrame:
    """Correlation of each score with the spectrum and its variants, and the paired differences.

    Returns tidy rows with ``kind`` = ``"rho"`` (one per score, variant, stratum) or ``"delta"``
    (direct minus variant). Differences use the same site resample for both arms, which is what
    makes a small difference between two highly correlated targets detectable.
    """
    variants = orientation_variants(q)
    names = list(variants)
    targets = {n: [accessibility_for(u, variants[n], row_sum=row_sum) for u in units] for n in names}
    point = _rho_tensor(units, score_names, targets, names, None, strata)
    boots = np.stack([
        _rho_tensor(units, score_names, targets, names, d, strata) for d in _draws(units, n_boot, seed)
    ])                                                         # (B, score, target, stratum)
    rows = []
    for si, s in enumerate(score_names):
        for ti, t in enumerate(names):
            lo, hi = _ci(boots[:, si, ti, :])
            for ki, k in enumerate(strata):
                rows.append(dict(kind="rho", score=s, target=t, versus="", stratum=k,
                                 value=point[si, ti, ki], lo=lo[ki], hi=hi[ki]))
        for a, b in _CONTRASTS:
            ia, ib = names.index(a), names.index(b)
            diff = boots[:, si, ia, :] - boots[:, si, ib, :]
            lo, hi = _ci(diff)
            for ki, k in enumerate(strata):
                rows.append(dict(kind="delta", score=s, target=a, versus=b, stratum=k,
                                 value=point[si, ia, ki] - point[si, ib, ki], lo=lo[ki], hi=hi[ki]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Null battery
# ---------------------------------------------------------------------------

def null_battery(
    units: Sequence[Unit],
    q: np.ndarray,
    score_names: Sequence[str],
    *,
    strata: Sequence = (1, 2, 3),
    n_shuffle: int = 500,
    seed: int = 20261009,
    row_sum: float | None = None,
) -> pd.DataFrame:
    """Rank of the real spectrum among relabellings of its bases and shuffles of its rates.

    ``pct_relabel`` is the fraction of the 24 relabellings (the real one included) that correlate at
    least as well as the real spectrum; small means the real labelling is special. ``pct_shuffle`` is
    the same against random permutations of the twelve rates. A score that has learned *this*
    spectrum should have small values; a score that only likes any smooth-looking matrix will not.
    """
    rng = np.random.default_rng(seed)
    relabelled = [acc.relabel(q, p) for p in acc.all_relabellings()]
    shuffled = [acc.shuffle(q, rng) for _ in range(n_shuffle)]
    names = ["relabel_%d" % i for i in range(len(relabelled))] + ["shuffle_%d" % i for i in range(n_shuffle)]
    mats = relabelled + shuffled
    targets = {n: [accessibility_for(u, m, row_sum=row_sum) for u in units] for n, m in zip(names, mats)}
    rho = _rho_tensor(units, score_names, targets, names, None, strata)    # (score, cand, stratum)
    n_rel = len(relabelled)
    rows = []
    for si, s in enumerate(score_names):
        for ki, k in enumerate(strata):
            col = rho[si, :, ki]
            real = col[0]                                    # relabelling 0 is the identity
            rel, shuf = col[:n_rel], col[n_rel:]
            rows.append(dict(
                score=s, stratum=k, rho_real=real,
                pct_relabel=float((rel >= real - 1e-12).mean()),
                pct_shuffle=float((shuf >= real - 1e-12).mean()),
                shuffle_mean=float(np.nanmean(shuf)), shuffle_sd=float(np.nanstd(shuf)),
                z_shuffle=float((real - np.nanmean(shuf)) / np.nanstd(shuf)),
                best_relabel=int(np.nanargmax(rel)), rho_best_relabel=float(np.nanmax(rel)),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2b. Direction-flip null
# ---------------------------------------------------------------------------

_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def flip_variants(q: np.ndarray) -> list[np.ndarray]:
    """All 64 versions of ``q`` with each reciprocal pair's two rates independently swapped or not.

    Version 0 is the original. Version 63 is its transpose. Every version has the same multiset of
    rates, the same symmetrised matrix, and the same transition/transversion structure.
    """
    q = np.asarray(q, float)
    out = []
    for mask in range(64):
        v = q.copy()
        for k, (i, j) in enumerate(_PAIRS):
            if mask >> k & 1:
                v[i, j], v[j, i] = q[j, i], q[i, j]
        out.append(v)
    return out


def flip_null(
    units: Sequence[Unit],
    q: np.ndarray,
    score_names: Sequence[str],
    *,
    strata: Sequence = (1, 2, 3),
    row_sum: float | None = None,
) -> pd.DataFrame:
    """Rank of the real direction among the 64 direction-flipped versions of the spectrum.

    ``pct_flip`` is the fraction of the 64 versions (the real one included) that correlate at least as
    well as the real one: 1/64 = 0.016 if the real direction is the best of all. A score that has no
    direction information has no reason to prefer the real assignment, so ``pct_flip`` is spread
    over 0 to 1 and is typically large for symmetric scores.
    """
    variants = flip_variants(q)
    names = [f"flip_{m}" for m in range(64)]
    targets = {n: [accessibility_for(u, v, row_sum=row_sum) for u in units] for n, v in zip(names, variants)}
    rho = _rho_tensor(units, score_names, targets, names, None, strata)       # (score, 64, stratum)
    rows = []
    for si, s in enumerate(score_names):
        for ki, k in enumerate(strata):
            col = rho[si, :, ki]
            real = col[0]
            rows.append(dict(
                score=s, stratum=k, rho_real=real, pct_flip=float((col >= real - 1e-12).mean()),
                rho_flip_mean=float(np.nanmean(col[1:])), rho_flip_max=float(np.nanmax(col)),
                rho_flip_min=float(np.nanmin(col)), best_flip=int(np.nanargmax(col)),
                rho_transposed=float(col[63]),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Direction contrast
# ---------------------------------------------------------------------------

def favoured_classes(q: np.ndarray) -> tuple[set, set]:
    """Base changes in the favoured and the disfavoured direction of each reciprocal pair.

    For every unordered pair of bases, the direction with the higher rate is favoured. Pairs with
    equal rates are left out. A symmetric spectrum has no favoured direction and returns two empty sets.
    """
    fav, dis = set(), set()
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = q[i, j], q[j, i]
            if np.isclose(a, b, rtol=1e-9, atol=0.0):   # relative only: rates can be ~1e-8 or smaller
                continue
            hi, lo = ((i, j), (j, i)) if a > b else ((j, i), (i, j))
            fav.add((acc.BASES[hi[0]], acc.BASES[hi[1]]))
            dis.add((acc.BASES[lo[0]], acc.BASES[lo[1]]))
    return fav, dis


def _route_class(unit: Unit, rows: np.ndarray) -> list[tuple | None]:
    """The single (from, to) base change for one-step rows with exactly one route; else None."""
    out = []
    for r in rows:
        routes = acc.single_step_changes(acc.CODONS[unit.ref_codon[r]], acc.AA20[unit.aa[r]])
        out.append(routes[0] if len(routes) == 1 else None)
    return out


def direction_contrast(
    units: Sequence[Unit],
    q: np.ndarray,
    score_names: Sequence[str],
    *,
    n_boot: int = 200,
    seed: int = 20261009,
    include_truth: bool = True,
    row_sum: float | None = None,
    forward: Iterable[tuple] | None = None,
    reverse: Iterable[tuple] | None = None,
) -> pd.DataFrame:
    """Do scores favour the spectrum's favoured direction beyond what amino-acid similarity explains?

    Among one-step replacements with a single route, each row's score is converted to a within-unit
    rank in [0, 1]. The effect of the *unordered* amino-acid pair is then removed (mean rank of all
    rows with that pair, both directions and all units), so a symmetric score leaves exactly zero.
    The contrast is the mean residual of favoured-direction rows minus that of disfavoured rows.

    Ranks are fixed on the full data; the bootstrap resamples sites, recomputing pair means. With
    ``include_truth`` the real accessibility is scored too, as the scale against which a score's
    contrast is read (``fraction_of_truth``).
    """
    fav, dis = favoured_classes(q) if forward is None else (set(forward), set(reverse or ()))
    if not fav or not dis:
        raise ValueError("the spectrum has no favoured direction; there is nothing to contrast")
    names = list(score_names) + (["__truth__"] if include_truth else [])

    prepared = []
    for ui, u in enumerate(units):
        rows = np.where(u.steps == 1)[0]
        cls = _route_class(u, rows)
        keep = np.array([c is not None and (c in fav or c in dis) for c in cls])
        rows, cls = rows[keep], [c for c, k in zip(cls, keep) if k]
        is_fav = np.array([c in fav for c in cls])
        a, b = u.ref_aa[rows], u.aa[rows]
        pair = np.minimum(a, b) * 20 + np.maximum(a, b)
        ranks = {}
        for n in names:
            v = accessibility_for(u, q, row_sum=row_sum) if n == "__truth__" else u.scores[n]
            ranks[n] = (rankdata(v[rows]) - 0.5) / len(rows)      # rank among the kept rows, in (0, 1)
        prepared.append(dict(u=u, rows=rows, is_fav=is_fav, pair=pair, ranks=ranks))

    def site_weights(site_ids: np.ndarray, draw: np.ndarray | None) -> np.ndarray:
        """How many times each row's site appears in the resample (1 everywhere when ``draw`` is None)."""
        if draw is None:
            return np.ones(len(site_ids))
        uniq, cnt = np.unique(draw, return_counts=True)
        pos = np.clip(np.searchsorted(uniq, site_ids), 0, len(uniq) - 1)
        return np.where(uniq[pos] == site_ids, cnt[pos], 0).astype(float)

    for p in prepared:
        p["site"] = p["u"].site[p["rows"]]

    def contrast(draw: np.ndarray | None) -> dict[str, float]:
        weights = [site_weights(p["site"], draw) for p in prepared]
        out = {}
        for n in names:
            pair_sum = np.zeros(400)
            pair_w = np.zeros(400)
            for p, w in zip(prepared, weights):
                pair_sum += np.bincount(p["pair"], weights=w * p["ranks"][n], minlength=400)
                pair_w += np.bincount(p["pair"], weights=w, minlength=400)
            # A pair absent from this resample has no mean. Every row of such a pair has weight 0,
            # so any finite fill leaves the sums unchanged; NaN would turn 0 * NaN into a NaN
            # replicate that nanpercentile then silently drops (biased against rare pairs).
            pair_mean = np.divide(pair_sum, pair_w, out=np.zeros(400), where=pair_w > 0)
            num = {True: 0.0, False: 0.0}
            den = {True: 0.0, False: 0.0}
            for p, w in zip(prepared, weights):
                resid = p["ranks"][n] - pair_mean[p["pair"]]
                for flag in (True, False):
                    sel = p["is_fav"] == flag
                    num[flag] += float((w[sel] * resid[sel]).sum())
                    den[flag] += float(w[sel].sum())
            out[n] = num[True] / den[True] - num[False] / den[False]
        return out

    point = contrast(None)
    samples = {n: [] for n in names}
    for d in _draws(units, n_boot, seed):
        c = contrast(d)
        for n in names:
            samples[n].append(c[n])
    rows = []
    truth = point.get("__truth__", np.nan)
    for n in names:
        s = np.array(samples[n])
        lo, hi = np.nanpercentile(s, [2.5, 97.5])
        rows.append(dict(score=n.replace("__truth__", "accessibility (truth)"), contrast=point[n], lo=lo, hi=hi,
                         fraction_of_truth=(point[n] / truth) if include_truth else np.nan,
                         n_favoured=int(sum(p["is_fav"].sum() for p in prepared)),
                         n_disfavoured=int(sum((~p["is_fav"]).sum() for p in prepared))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Planted signal
# ---------------------------------------------------------------------------

def planted_signal(
    units: Sequence[Unit],
    q: np.ndarray,
    control_names: Sequence[str],
    *,
    lambdas: Sequence[float] = tuple(np.round(np.linspace(0, 1, 11), 2)),
    strata: Sequence = (1, 2, 3),
    row_sum: float | None = None,
) -> pd.DataFrame:
    """Dose-response of the orientation statistics to a known directional signal.

    For a control (no nucleotide content) the score is replaced by
    ``(1 - lam) * rank(control) + lam * rank(accessibility)`` within each unit and stratum, with the
    accessibility either the real spectrum (``planted="direct"``) or its symmetrisation
    (``planted="symmetrised"``). The statistic ``direct - symmetrised`` (and ``direct - transposed``)
    is then read off against ``lam``. This is the scale on which a model's value can be located:
    how much planted direction would produce the same difference?
    """
    variants = orientation_variants(q)
    rows = []
    for planted in ("direct", "symmetrised"):
        for s in strata:
            for ctl in control_names:
                for lam in lambdas:
                    vals = {n: [] for n in ("direct", "symmetrised", "transposed", "complemented")}
                    for u in units:
                        r = _mask(u, np.arange(u.n), s)
                        if len(r) < 10:
                            continue
                        rc = (rankdata(u.scores[ctl][r]) - 0.5) / len(r)
                        ra = (rankdata(accessibility_for(u, variants[planted], row_sum=row_sum)[r]) - 0.5) / len(r)
                        mixed = (1 - lam) * rc + lam * ra
                        cols = [mixed] + [accessibility_for(u, variants[n], row_sum=row_sum)[r] for n in vals]
                        rho = _rho_block(np.column_stack(cols), 1)[0]
                        for n, v in zip(vals, rho):
                            vals[n].append(v)
                    m = {n: float(np.nanmean(v)) for n, v in vals.items()}
                    rows.append(dict(planted=planted, stratum=s, control=ctl, lam=lam,
                                     rho_direct=m["direct"], rho_symmetrised=m["symmetrised"],
                                     rho_transposed=m["transposed"], rho_complemented=m["complemented"],
                                     delta_sym=m["direct"] - m["symmetrised"],
                                     delta_trans=m["direct"] - m["transposed"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Rank robustness
# ---------------------------------------------------------------------------

def rank_robustness(
    units: Sequence[Unit],
    q: np.ndarray,
    score_names: Sequence[str],
    *,
    strata: Sequence = (1, 2, 3),
    n_tiebreak: int = 20,
    seed: int = 20261009,
    row_sum: float | None = None,
) -> pd.DataFrame:
    """``direct - symmetrised`` under three ways of ranking, to test the ties/variance hypothesis.

    ``average``  average ranks for ties (the default everywhere else).
    ``tiebreak`` ties broken at random, averaged over ``n_tiebreak`` draws. A tie-driven effect
                 would change here.
    ``log``      Pearson on log10 values rather than ranks, which uses magnitude.
    The accessibility columns are also summarised by how many distinct values each has per stratum.
    """
    rng = np.random.default_rng(seed)
    dir_t = [accessibility_for(u, q, row_sum=row_sum) for u in units]
    sym_t = [accessibility_for(u, acc.symmetrise(q), row_sum=row_sum) for u in units]
    rows = []
    for s in strata:
        for sc in score_names:
            res = {"average": [], "tiebreak": [], "log": []}
            for ui, u in enumerate(units):
                r = _mask(u, np.arange(u.n), s)
                if len(r) < 10:
                    continue
                x, d, y = u.scores[sc][r], dir_t[ui][r], sym_t[ui][r]
                rd = _rho_block(np.column_stack([x, d, y]), 1)[0]
                res["average"].append(rd[0] - rd[1])
                diffs = []
                for _ in range(n_tiebreak):
                    jitter = lambda v: np.lexsort((rng.random(len(v)), v)).argsort()   # random tie order
                    a, b, c = jitter(x), jitter(d), jitter(y)
                    diffs.append(np.corrcoef(a, b)[0, 1] - np.corrcoef(a, c)[0, 1])
                res["tiebreak"].append(np.mean(diffs))
                lx = np.log10(np.clip(x, 1e-300, None)) if (x > 0).all() else x
                ld, ly = np.log10(np.clip(d, 1e-300, None)), np.log10(np.clip(y, 1e-300, None))
                res["log"].append(np.corrcoef(lx, ld)[0, 1] - np.corrcoef(lx, ly)[0, 1])
            for how, v in res.items():
                rows.append(dict(score=sc, stratum=s, method=how, delta_sym=float(np.nanmean(v)) if v else np.nan))
    out = pd.DataFrame(rows)
    ties = []
    for s in strata:
        dn = np.mean([len(np.unique(dir_t[i][_mask(u, np.arange(u.n), s)])) for i, u in enumerate(units)])
        sn = np.mean([len(np.unique(sym_t[i][_mask(u, np.arange(u.n), s)])) for i, u in enumerate(units)])
        n = np.mean([len(_mask(u, np.arange(u.n), s)) for u in units])
        ties.append(dict(stratum=s, rows=n, distinct_direct=dn, distinct_symmetrised=sn))
    out.attrs["distinct_values"] = pd.DataFrame(ties)
    return out
