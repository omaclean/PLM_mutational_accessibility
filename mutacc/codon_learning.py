"""Does a model that was never shown codons still tell synonymous codons apart?

The models read amino acids. They are never shown which codon encoded a residue, so any
sensitivity to the codon is learned indirectly. Two tests ask whether the fine-tuned model has
it anyway. This module holds the arithmetic for both; it plots nothing and reads no paths
except the root FASTA files it is handed.

**Strand B, the within-replacement test.** Fix the replacement, say serine to asparagine. Every
amino-acid quantity is now constant: biochemistry, Grantham distance, any exchangeability
matrix. What still varies between sites is the reference *codon*, because serine has six. A site
whose codon can reach asparagine in one nucleotide change is *exposed*; one that needs two or
three is not. The question is whether the model ranks asparagine higher at exposed sites.

* The outcome is the rank of the target among the nineteen alternatives **at its own site**
  (``plm_prob`` ranked within site), which removes the site's overall mutability.
* The statistic is the probability that an exposed site outranks an unexposed one (an AUROC,
  0.5 = no codon sensitivity), computed in each (lineage, replacement) block that has at
  least :data:`MIN_EXPOSED` exposed and :data:`MIN_UNEXPOSED` unexposed sites and
  :data:`MIN_BLOCK` sites in all, then pooled with van Elteren weights
  ``n1 * n2 / (n1 + n2 + 1)``. The same filter selects the 54 scored replacements in the
  shipped thirteen- and five-lineage tables: a block with no spread in codons cannot pass it.
* Ties are handled with mid-ranks, as the exploration code did (``a20``), which gives
  0.6079 for the five-lineage fine-tune. A later re-implementation broke ties in row order
  and got 0.6088; :func:`audit_pooled_auc` reproduces that number so the two can be compared.
* Intervals resample **sites** (whole positions, with the same positions drawn in every
  lineage and for both models). The lineages share their sites, so resampling lineages gives an
  interval about thirteen times too narrow for this statistic (``LINEAGE_VALIDITY.md`` Q2).
* The null permutes reference codons **between positions that carry the same amino acid**
  (and moves a position's codon in every lineage together), then recomputes which sites are
  exposed. The outcome ranks are untouched, so a model with no codon sensitivity sits at 0.5.
  The archive also reported a stricter null that matches on site conservation (0.520 for the
  five-lineage fine-tune, 0.511 for the base); conservation is not in the shipped tables, so
  it is not recomputed here.

**Strand C, the within-residue test.** For each amino acid separately, does a site's total
probability of *changing* the residue fall as the codon makes a change more likely to be
synonymous? ``persistence`` is the probability that a mutation of the codon is synonymous under a
given spectrum, and the statistic is the Spearman correlation between persistence and the
site's summed alternative-amino-acid probability, averaged over amino acids (predicted
negative). It is a different hypothesis from strand B, not a replication of it, and it is null
for both models on both lineage sets (``LINEAGE_VALIDITY.md`` section 4).

Reference codons come from the lineage roots: position *p* of the table is codon *p* of the
root. :func:`check_roots` proves that on the data (translation matches ``ref_aa`` and the
recomputed step count matches ``nt_steps`` in every row) before anything is scored.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

from mutacc import accessibility as acc

__all__ = [
    "EXPOSED_STEP", "MIN_EXPOSED", "MIN_UNEXPOSED", "MIN_BLOCK", "MIN_SITES_PER_AA",
    "ALTERNATIVES", "steps_table", "canonical_scores", "WithinPair", "BlockResult",
    "audit_pooled_auc", "bootstrap_sites", "reference_codon_matrix", "check_roots",
    "position_groups", "permutation_null", "pair_table", "per_lineage_auc", "lineage_resample_ci",
    "sign_test", "robustness", "six_codon_residues", "persistence", "strand_c_sites",
    "strand_c_per_aa", "strand_c_bootstrap", "strand_c_null", "unit_ranks", "read_derived",
]

EXPOSED_STEP = 1
MIN_EXPOSED = 5
MIN_UNEXPOSED = 5
MIN_BLOCK = 15
MIN_SITES_PER_AA = 6
#: Alternatives scored at every site: the other nineteen amino acids.
ALTERNATIVES = len(acc.AA20) - 1

KEY = ["lineage", "position", "ref_aa", "aa"]
REQUIRED = {"model", "lineage", "position", "ref_aa", "aa", "plm_prob", "nt_steps"}


# ---------------------------------------------------------------------------
# Genetic code geometry, from Biopython via mutacc.accessibility
# ---------------------------------------------------------------------------

_STEPS = None


def steps_table() -> np.ndarray:
    """64 x 20 table: fewest nucleotide changes from each codon to each amino acid.

    Rows follow ``acc.CODONS``, columns ``acc.AA20``. A stop codon as the *source* is given
    the step count it would have to each amino acid; such rows are never used because no
    reference codon is a stop.
    """
    global _STEPS
    if _STEPS is None:
        _STEPS = np.array([[acc.nucleotide_steps(c, a) for a in acc.AA20] for c in acc.CODONS], dtype=np.int8)
    return _STEPS


def six_codon_residues() -> list[str]:
    """Amino acids with six codons, read off the genetic code rather than typed."""
    counts = pd.Series(list(acc.GENETIC_CODE.values())).value_counts()
    return sorted(a for a, n in counts.items() if n == 6 and a != "*")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def canonical_scores(scores: pd.DataFrame, model: str) -> pd.DataFrame:
    """One model's rows, validated and in a fixed (lineage, position, aa) order.

    Raises on anything that would silently change what is scored: missing columns, an empty
    model, non-finite or negative probabilities, duplicated or synonymous rows, step counts
    outside 1 to 3, and any site that does not carry exactly nineteen alternatives. Adds
    ``site_rank``, the mid-rank of ``plm_prob`` among the alternatives at the same site.
    """
    missing = REQUIRED - set(scores.columns)
    if missing:
        raise KeyError(f"replacement table is missing columns {sorted(missing)}")
    sub = scores.loc[scores["model"] == model]
    if sub.empty:
        raise ValueError(f"no rows for model {model!r}")
    sub = sub.copy()
    sub["lineage"] = sub["lineage"].astype(str)
    p = sub["plm_prob"].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("plm_prob must be finite and non-negative")
    if sub.duplicated(KEY).any():
        raise ValueError(f"model {model!r}: duplicate (lineage, position, replacement) rows")
    if (sub["ref_aa"] == sub["aa"]).any():
        raise ValueError("replacement table holds synonymous rows (ref_aa == aa)")
    if not set(sub["nt_steps"].unique()) <= {1, 2, 3}:
        raise ValueError("nt_steps must be 1, 2 or 3")
    letters = set(acc.AA20)
    if not (set(sub["ref_aa"]) <= letters and set(sub["aa"]) <= letters):
        raise ValueError("replacement table holds symbols outside the 20 amino acids")
    sizes = sub.groupby(["lineage", "position"]).size()
    if (sizes != ALTERNATIVES).any():
        raise ValueError(f"every site must carry exactly {ALTERNATIVES} alternatives; "
                         f"{int((sizes != ALTERNATIVES).sum())} sites do not")
    sub = sub.sort_values(["lineage", "position", "aa"], kind="stable").reset_index(drop=True)
    sub["site_rank"] = sub.groupby(["lineage", "position"])["plm_prob"].rank(method="average")
    return sub


# ---------------------------------------------------------------------------
# Strand B
# ---------------------------------------------------------------------------

@dataclass
class BlockResult:
    """Per-block output of :meth:`WithinPair.blocks`, plus the pooled value."""
    pooled: float
    auc: np.ndarray       # per block; NaN where a block has no exposed or no unexposed row
    weight: np.ndarray    # van Elteren weight; zero where the block is filtered out
    ok: np.ndarray        # block passes the minimum-count filter
    n_exposed: np.ndarray
    n_unexposed: np.ndarray


class WithinPair:
    """One model's rows arranged for fast within-replacement AUROC under any weights or exposure.

    Built once from :func:`canonical_scores` output. Rows are sorted by (block, outcome) so the
    Mann-Whitney count needs only cumulative sums, which lets the bootstrap and the permutation
    null re-score 140,000 rows in about ten milliseconds. A *block* is one lineage and one
    replacement (an ordered amino-acid pair).
    """

    def __init__(self, canon: pd.DataFrame):
        if canon.empty:
            raise ValueError("no rows to score")
        self.lineages = sorted(canon["lineage"].unique())
        self.positions = np.sort(canon["position"].unique())
        lin = np.searchsorted(self.lineages, canon["lineage"].to_numpy())
        pos = np.searchsorted(self.positions, canon["position"].to_numpy())
        pair_label = (canon["ref_aa"] + ">" + canon["aa"]).to_numpy()
        self.pair_names, pair = np.unique(pair_label, return_inverse=True)
        key = lin.astype(np.int64) * len(self.pair_names) + pair
        ukey, block = np.unique(key, return_inverse=True)
        self.n_blocks = len(ukey)
        self.block_lineage = (ukey // len(self.pair_names)).astype(int)
        self.block_pair = (ukey % len(self.pair_names)).astype(int)

        # arrays in canonical (input) order, kept for permuting exposure
        self.lin_row = lin
        self.pos_row = pos
        self.aa_row = np.array([acc.aa_index(a) for a in canon["aa"]], dtype=int)
        self.ref_row = np.array([acc.aa_index(a) for a in canon["ref_aa"]], dtype=int)
        self.steps_row = canon["nt_steps"].to_numpy(int)
        self.n_rows = len(canon)

        value = canon["site_rank"].to_numpy(float)
        self.order = np.lexsort((value, block))
        self.block = block[self.order]
        self.pos_sorted = pos[self.order]
        v = value[self.order]
        self.value = v
        new = np.ones(self.n_rows, bool)
        new[1:] = (self.block[1:] != self.block[:-1]) | (v[1:] != v[:-1])
        self.group = np.cumsum(new) - 1
        self.n_groups = int(self.group[-1]) + 1
        self.exposed = (self.steps_row[self.order] == EXPOSED_STEP)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_names)

    def exposure_from_steps(self, steps_row: np.ndarray) -> np.ndarray:
        """Exposure flags in sorted order from step counts given in canonical row order."""
        return (np.asarray(steps_row)[self.order] == EXPOSED_STEP)

    def blocks(self, exposed: np.ndarray | None = None, weights: np.ndarray | None = None) -> BlockResult:
        """AUROC of exposed against unexposed sites in every block, and the pooled value.

        ``exposed`` (sorted order) and ``weights`` (sorted order, one per row; a site resampled
        twice carries weight two) default to the observed data. A weighted block is identical
        to a block in which the row had been duplicated.
        """
        ex = self.exposed if exposed is None else np.asarray(exposed, bool)
        w = np.ones(self.n_rows) if weights is None else np.asarray(weights, float)
        exf = ex.astype(float)
        w1, w0 = w * exf, w * (1.0 - exf)
        nb = self.n_blocks
        n1 = np.bincount(self.block, weights=w1, minlength=nb)
        n0 = np.bincount(self.block, weights=w0, minlength=nb)
        g_tot = np.bincount(self.group, weights=w0, minlength=self.n_groups)
        g_before = np.cumsum(g_tot) - g_tot
        b_before = np.cumsum(n0) - n0
        below = g_before[self.group] - b_before[self.block]
        u = np.bincount(self.block, weights=w1 * (below + 0.5 * g_tot[self.group]), minlength=nb)
        n = n1 + n0
        ok = (n1 >= MIN_EXPOSED) & (n0 >= MIN_UNEXPOSED) & (n >= MIN_BLOCK)
        with np.errstate(divide="ignore", invalid="ignore"):
            auc = u / (n1 * n0)
            wt = np.where(ok, n1 * n0 / (n1 + n0 + 1.0), 0.0)
        pooled = float(np.nansum(auc[ok] * wt[ok]) / wt[ok].sum()) if ok.any() else float("nan")
        return BlockResult(pooled, auc, wt, ok, n1, n0)

    def pooled(self, exposed=None, weights=None) -> float:
        return self.blocks(exposed, weights).pooled

    def pair_values(self, res: BlockResult) -> np.ndarray:
        """Weighted mean AUROC over the blocks of each replacement; NaN where none was scored."""
        num = np.bincount(self.block_pair, weights=np.nan_to_num(res.auc) * res.weight, minlength=self.n_pairs)
        den = np.bincount(self.block_pair, weights=res.weight, minlength=self.n_pairs)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def pooled_excluding(self, res: BlockResult, pair_ids) -> tuple[float, int]:
        """Pooled AUROC with some replacements removed, and how many scored replacements remain."""
        keep = res.ok & ~np.isin(self.block_pair, np.asarray(list(pair_ids), dtype=int))
        if not keep.any() or res.weight[keep].sum() <= 0:
            return float("nan"), 0
        value = float(np.sum(res.auc[keep] * res.weight[keep]) / res.weight[keep].sum())
        return value, int(len(np.unique(self.block_pair[keep])))


def audit_pooled_auc(canon: pd.DataFrame) -> float:
    """Pooled AUROC with ties broken in row order, as the 9 Oct audit re-implementation did.

    The outcome is an integer-valued within-site rank, so a block holds many ties. Breaking them
    by row order is arbitrary rather than wrong, but it differs from the mid-rank statistic by up
    to about 0.001 (fine-tune) and 0.001 (base): 0.6104 and 0.5105 here against 0.6105 and 0.5116
    with mid-ranks on the thirteen-lineage set. Kept so the audit's numbers can be reproduced.
    """
    wp = WithinPair(canon)
    block = wp.block
    # canonical order within a block is (lineage, position, aa); sort by outcome with row order breaking ties
    value = canon["site_rank"].to_numpy(float)[wp.order]
    rows = wp.order
    o = np.lexsort((rows, value, block))
    bs = block[o]
    counts = np.bincount(bs, minlength=wp.n_blocks)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ranks = np.empty(wp.n_rows)
    ranks[o] = np.arange(wp.n_rows) - starts[bs] + 1.0
    ex = wp.exposed.astype(float)
    n1 = np.bincount(block, weights=ex, minlength=wp.n_blocks)
    n = counts.astype(float)
    n0 = n - n1
    r1 = np.bincount(block, weights=ranks * ex, minlength=wp.n_blocks)
    ok = (n1 >= MIN_EXPOSED) & (n0 >= MIN_UNEXPOSED) & (n >= MIN_BLOCK)
    if not ok.any():
        return float("nan")
    auc = (r1[ok] - n1[ok] * (n1[ok] + 1) / 2.0) / (n1[ok] * n0[ok])
    w = n1[ok] * n0[ok] / (n1[ok] + n0[ok] + 1.0)
    return float(np.sum(auc * w) / w.sum())


def _check_same_sites(preps: Mapping[str, WithinPair]) -> WithinPair:
    first = next(iter(preps.values()))
    for name, p in preps.items():
        if (p.lineages != first.lineages or not np.array_equal(p.positions, first.positions)
                or not np.array_equal(p.pos_row, first.pos_row) or not np.array_equal(p.aa_row, first.aa_row)
                or not np.array_equal(p.lin_row, first.lin_row)):
            raise ValueError(f"model {name!r} was not scored on the same sites and replacements as the others")
    return first


def bootstrap_sites(preps: Mapping[str, WithinPair], *, n_boot: int, seed: int) -> dict:
    """Site-clustered bootstrap of the pooled and per-replacement AUROC.

    Each draw resamples whole positions with replacement and applies the draw to every lineage
    and every model, so differences between models keep their pairing. Returns
    ``{model: (pooled, per_pair)}`` with shapes ``(n_boot,)`` and ``(n_boot, n_pairs)``; draws
    in which nothing is scorable are NaN.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    first = _check_same_sites(preps)
    npos = len(first.positions)
    rng = np.random.default_rng(seed)
    out = {m: (np.full(n_boot, np.nan), np.full((n_boot, p.n_pairs), np.nan)) for m, p in preps.items()}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, npos, npos), minlength=npos).astype(float)
        for m, p in preps.items():
            res = p.blocks(weights=counts[p.pos_sorted])
            out[m][0][b] = res.pooled
            out[m][1][b] = p.pair_values(res)
    return out


# ---------------------------------------------------------------------------
# Reference codons and the permutation null
# ---------------------------------------------------------------------------

def reference_codon_matrix(prep: WithinPair, roots: Mapping[str, str | Path]) -> np.ndarray:
    """Codon index of every (lineage, position) from the lineage root; -1 where the lineage lacks the site."""
    mat = np.full((len(prep.lineages), len(prep.positions)), -1, dtype=int)
    for li, lin in enumerate(prep.lineages):
        if lin not in roots:
            raise KeyError(f"no root sequence for lineage {lin!r}")
        pos = np.unique(prep.positions[prep.pos_row[prep.lin_row == li]])
        for p, codon in zip(pos, acc.reference_codons(roots[lin], pos)):
            mat[li, np.searchsorted(prep.positions, p)] = acc.codon_index(codon)
    return mat


def check_roots(prep: WithinPair, codons: np.ndarray) -> None:
    """Raise unless the roots explain the table: every translation equals ``ref_aa`` and every step count matches."""
    ci = codons[prep.lin_row, prep.pos_row]
    if (ci < 0).any():
        raise ValueError("a scored site has no root codon")
    table = steps_table()
    translated = np.array([acc.aa_index(acc.GENETIC_CODE[acc.CODONS[i]]) if acc.GENETIC_CODE[acc.CODONS[i]] != "*" else -1
                           for i in range(len(acc.CODONS))])
    bad_aa = int((translated[ci] != prep.ref_row).sum())
    if bad_aa:
        raise ValueError(f"{bad_aa} rows: the root codon does not translate to the table's reference amino acid")
    bad_steps = int((table[ci, prep.aa_row] != prep.steps_row).sum())
    if bad_steps:
        raise ValueError(f"{bad_steps} rows: nt_steps does not match the root codon")


def position_groups(prep: WithinPair, codons: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, np.ndarray]]]:
    """Who may swap codons with whom.

    Returns ``(shared, own)``. ``shared`` are groups of positions that carry the same amino
    acid in every lineage; a position moves with all its lineage copies together, because the
    lineages are near-duplicates and shuffling their copies independently would treat one site
    as thirteen. ``own`` are ``(lineage index, positions)`` groups for the positions whose amino
    acid differs between lineages or that some lineage lacks: they swap only within their own
    lineage, among sites with the same amino acid. Leaving them in place instead would keep
    the real association at exactly the variable sites where it is strongest and bias the null
    upwards.
    """
    n_lin, n_pos = codons.shape
    ref = np.full((n_lin, n_pos), -2, dtype=int)
    ref[prep.lin_row, prep.pos_row] = prep.ref_row
    shared: dict[int, list[int]] = {}
    leftover = np.zeros(n_pos, bool)
    for p in range(n_pos):
        col = ref[:, p]
        if (col < 0).any() or (col != col[0]).any():
            leftover[p] = True
        else:
            shared.setdefault(int(col[0]), []).append(p)
    own = []
    for li in range(n_lin):
        by_aa: dict[int, list[int]] = {}
        for p in np.flatnonzero(leftover):
            if ref[li, p] >= 0:
                by_aa.setdefault(int(ref[li, p]), []).append(p)
        own += [(li, np.array(g)) for g in by_aa.values() if len(g) > 1]
    return [np.array(g) for g in shared.values() if len(g) > 1], own


def permutation_null(preps: Mapping[str, WithinPair], codons: np.ndarray, *, n_perm: int, seed: int) -> dict:
    """Pooled AUROC when reference codons are shuffled between positions with the same amino acid.

    The same shuffles are used for every model. Returns ``{model: array(n_perm)}``.
    """
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1")
    first = _check_same_sites(preps)
    shared, own = position_groups(first, codons)
    if not shared and not own:
        raise ValueError("no positions can swap codons; the permutation null is undefined")
    table = steps_table()
    rng = np.random.default_rng(seed)
    out = {m: np.full(n_perm, np.nan) for m in preps}
    for i in range(n_perm):
        cm = codons.copy()
        for g in shared:
            cm[:, g] = codons[:, rng.permutation(g)]
        for li, g in own:
            cm[li, g] = codons[li, rng.permutation(g)]
        steps = table[cm[first.lin_row, first.pos_row], first.aa_row]
        for m, p in preps.items():
            out[m][i] = p.pooled(exposed=p.exposure_from_steps(steps))
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def pair_table(prep: WithinPair, res: BlockResult, draws: np.ndarray | None = None) -> pd.DataFrame:
    """Per-replacement AUROC pooled over lineages, with site-bootstrap interval when draws are given.

    ``sites_one_step`` and ``sites_multi_step`` are the exposed and unexposed site counts per
    scored block (one lineage), averaged over the lineages in which the replacement was scored.
    """
    values = prep.pair_values(res)
    keep = np.isfinite(values)
    n1 = np.bincount(prep.block_pair, weights=np.nan_to_num(res.n_exposed) * res.ok, minlength=prep.n_pairs)
    n0 = np.bincount(prep.block_pair, weights=np.nan_to_num(res.n_unexposed) * res.ok, minlength=prep.n_pairs)
    den = np.bincount(prep.block_pair, weights=res.weight, minlength=prep.n_pairs)
    n_ok = np.maximum(np.bincount(prep.block_pair, weights=res.ok.astype(float), minlength=prep.n_pairs), 1.0)
    lo = hi = np.full(prep.n_pairs, np.nan)
    if draws is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # replacements never scored are all-NaN columns
            lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    names = prep.pair_names
    return pd.DataFrame({
        "pair": names[keep],
        "ref_aa": [n[0] for n in names[keep]], "aa": [n[2] for n in names[keep]],
        "auc": values[keep], "ci_lo": lo[keep], "ci_hi": hi[keep], "weight": den[keep],
        "sites_one_step": n1[keep] / n_ok[keep], "sites_multi_step": n0[keep] / n_ok[keep],
    }).sort_values("auc", ascending=False, kind="stable").reset_index(drop=True)


def per_lineage_auc(prep: WithinPair, res: BlockResult) -> pd.DataFrame:
    """Pooled AUROC computed inside each lineage on its own."""
    rows = []
    for li, lin in enumerate(prep.lineages):
        m = res.ok & (prep.block_lineage == li)
        if not m.any():
            continue
        rows.append({"lineage": lin, "auc": float(np.sum(res.auc[m] * res.weight[m]) / res.weight[m].sum()),
                     "n_pairs": int(len(np.unique(prep.block_pair[m])))})
    if not rows:
        raise ValueError("no lineage has a scorable block")
    return pd.DataFrame(rows)


def lineage_resample_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Interval for the mean over lineages from resampling lineages. NOT a generalisation interval:
    shown beside the site-clustered one to say how much narrower it is."""
    v = np.asarray(values, float)
    if len(v) < 2:
        raise ValueError("need at least two lineages")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def sign_test(pair_auc: Sequence[float]) -> dict:
    """How many replacements sit above 0.5, and the two-sided sign-test p (the archive's 1.4e-7 for 46 of 54).

    The replacements share their sites, so they are not independent observations; the p is
    descriptive. A replacement exactly at 0.5 counts as not above.
    """
    v = np.asarray(pair_auc, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise ValueError("no scored replacements")
    k = int((v > 0.5).sum())
    return {"n_pairs": len(v), "n_above_half": k, "p_sign": float(binomtest(k, len(v), 0.5).pvalue)}


def robustness(prep: WithinPair, res: BlockResult) -> pd.DataFrame:
    """Pooled AUROC after removing the strongest replacements, and after removing the six-codon residues.

    Dropping the strongest replacements picks them on the outcome, so this is a stress test of
    how much of the pooled value rests on a few replacements, not an independent estimate.
    """
    pairs = pair_table(prep, res)
    index = {n: i for i, n in enumerate(prep.pair_names)}
    rows = []
    v, k = prep.pooled_excluding(res, [])
    rows.append(("nothing dropped", k, v))
    for n in (1, 3, 5, 10):
        v, k = prep.pooled_excluding(res, [index[p] for p in pairs["pair"][:n]])
        rows.append((f"drop the {n} strongest replacements", k, v))
    six = set(six_codon_residues())
    drop = [i for name, i in index.items() if name[0] in six]
    v, k = prep.pooled_excluding(res, drop)
    rows.append(("drop replacements from six-codon amino acids", k, v))
    return pd.DataFrame(rows, columns=["check", "pairs_remaining", "auc"])


# ---------------------------------------------------------------------------
# Strand C
# ---------------------------------------------------------------------------

def persistence(codon: str, q: np.ndarray) -> float:
    """Probability that a mutation of ``codon`` is synonymous, given a mutation occurs, under spectrum ``q``.

    Each of the nine single-nucleotide changes is weighted by its rate in ``q`` (rows are the
    base changed from, columns the base changed to, order ACGT).
    """
    aa = acc.GENETIC_CODE[codon]
    total = syn = 0.0
    for p in range(3):
        for b in acc.BASES:
            if b == codon[p]:
                continue
            rate = q[acc.BASES.index(codon[p]), acc.BASES.index(b)]
            total += rate
            if acc.GENETIC_CODE[codon[:p] + b + codon[p + 1:]] == aa:
                syn += rate
    if total <= 0:
        raise ValueError(f"spectrum gives codon {codon} no mutation rate")
    return syn / total


def strand_c_sites(scores: pd.DataFrame, model: str, roots: Mapping[str, str | Path], q: np.ndarray) -> pd.DataFrame:
    """One row per position: mean over lineages of the summed alternative probability, reference amino acid, codon and persistence."""
    canon = canonical_scores(scores, model)
    esc = canon.groupby(["lineage", "position"]).agg(escape=("plm_prob", "sum"), ref_aa=("ref_aa", "first")).reset_index()
    codon = []
    for lin, sub in esc.groupby("lineage", sort=False):
        if lin not in roots:
            raise KeyError(f"no root sequence for lineage {lin!r}")
        codon += list(zip(sub.index, acc.reference_codons(roots[lin], sub["position"].to_numpy())))
    esc["ref_codon"] = pd.Series(dict(codon))
    bad = [(r.lineage, r.position) for r in esc.itertuples() if acc.GENETIC_CODE[r.ref_codon] != r.ref_aa]
    if bad:
        raise ValueError(f"{len(bad)} sites: the root codon does not translate to the reference amino acid, e.g. {bad[0]}")
    site = (esc.groupby("position")
               .agg(escape=("escape", "mean"), ref_aa=("ref_aa", lambda s: s.mode().iat[0]),
                    ref_codon=("ref_codon", lambda s: s.mode().iat[0])).reset_index())
    cache: dict[str, float] = {}
    site["persistence"] = [cache.setdefault(c, persistence(c, q)) for c in site["ref_codon"]]
    return site


def unit_ranks(x) -> np.ndarray:
    """Centred mid-ranks scaled to unit length, so a dot product of two is their Spearman correlation."""
    r = rankdata(x).astype(float)
    r = r - r.mean()
    norm = np.linalg.norm(r)
    return r / norm if norm > 0 else r


def _usable(site: pd.DataFrame):
    for aa, s in site.groupby("ref_aa", sort=True):
        if len(s) >= MIN_SITES_PER_AA and s["persistence"].nunique() >= 2:
            yield aa, s


def strand_c_per_aa(site: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between persistence and escape, within each reference amino acid.

    Amino acids with fewer than :data:`MIN_SITES_PER_AA` sites, or with no spread in persistence
    (a single codon, or no variation among the codons in use), are not scored.
    """
    rows = [{"aa": aa, "n_sites": len(s), "rho": float(unit_ranks(s["persistence"]) @ unit_ranks(s["escape"]))}
            for aa, s in _usable(site)]
    if not rows:
        raise ValueError("no amino acid has enough sites with varying persistence")
    return pd.DataFrame(rows)


def strand_c_bootstrap(site: pd.DataFrame, *, n_boot: int, seed: int) -> np.ndarray:
    """Mean-over-amino-acids correlation under resampling of sites within each amino acid."""
    groups = [(s["persistence"].to_numpy(), s["escape"].to_numpy()) for _, s in _usable(site)]
    if not groups:
        raise ValueError("no amino acid has enough sites with varying persistence")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        rhos = []
        for s, e in groups:
            i = rng.integers(0, len(s), len(s))
            if np.ptp(s[i]) > 0:
                rhos.append(unit_ranks(s[i]) @ unit_ranks(e[i]))
        out[b] = np.mean(rhos) if rhos else np.nan
    return out


def strand_c_null(site: pd.DataFrame, *, n_perm: int, seed: int) -> np.ndarray:
    """Mean-over-amino-acids correlation when persistence is shuffled between sites of the same amino acid."""
    cols = []
    rng = np.random.default_rng(seed)
    for _, s in _usable(site):
        rs, rv = unit_ranks(s["persistence"]), unit_ranks(s["escape"])
        perm = np.argsort(rng.random((n_perm, len(s))), axis=1)
        cols.append(rs[perm] @ rv)
    if not cols:
        raise ValueError("no amino acid has enough sites with varying persistence")
    return np.mean(np.column_stack(cols), axis=1)


# ---------------------------------------------------------------------------
# Shipped derived tables
# ---------------------------------------------------------------------------

def read_derived(guide, name: str, results: Path | None = None) -> pd.DataFrame:
    """Read a table written by ``tools/export_codon_s06.py`` from the guide's results directory.

    These tables are not declared in the guides, so a missing file names the tool that makes it.
    An empty table raises: an empty join must not become a blank panel.
    """
    path = (Path(results) if results is not None else guide.results_dir) / f"{name}.tsv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; make it with "
                                f"`python tools/export_codon_s06.py --set {guide.name} --tables codon sG`")
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        raise ValueError(f"table {path.name} is empty")
    return frame


def _interval(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise ValueError("too few finite bootstrap draws")
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def derive_codon_tables(guide, scores: pd.DataFrame, *, n_boot: int = 1000, n_perm: int = 1000,
                        n_perm_c: int = 10000, spectrum: Path, symmetrise: bool = True,
                        seed: int = 20261009, strand_c_seed: int = 20260928) -> dict[str, pd.DataFrame]:
    """Compute the within-pair and strand C codon learning tables from replacement_scores."""
    base_models = guide.models_with_role("base")
    ft_models = guide.models_with_role("finetune")
    if len(base_models) != 1 or len(ft_models) != 1:
        raise ValueError(f"{guide.name}: need exactly one base and one finetune model")
    base, ft = base_models[0].key, ft_models[0].key
    models = [base, ft]
    roots = {lin.id: lin.root for lin in guide.lineages}

    canon = {m: canonical_scores(scores, m) for m in models}
    preps = {m: WithinPair(canon[m]) for m in models}
    codons = reference_codon_matrix(preps[ft], roots)
    for m in models:
        check_roots(preps[m], codons)

    boots = bootstrap_sites(preps, n_boot=n_boot, seed=seed)
    nulls = permutation_null(preps, codons, n_perm=n_perm, seed=seed + 1)
    results = {m: preps[m].blocks() for m in models}

    summary, by_pair, by_lin, robust = [], [], [], []
    for m in models:
        p, r = preps[m], results[m]
        pooled_draws, pair_draws = boots[m]
        lo, hi = _interval(pooled_draws)
        pt = pair_table(p, r, pair_draws)
        lt = per_lineage_auc(p, r)
        llo, lhi = lineage_resample_ci(lt["auc"], seed=seed + 2) if len(lt) >= 2 else (float("nan"), float("nan"))
        null = nulls[m]
        nlo, nhi = np.percentile(null, [2.5, 97.5])
        p_perm = float((np.sum(np.abs(null - 0.5) >= abs(r.pooled - 0.5)) + 1) / (len(null) + 1))
        st = sign_test(pt["auc"])
        summary.append({
            "model": m, "role": guide.model(m).role, "auc": r.pooled, "ci_lo": lo, "ci_hi": hi,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "null_lo": float(nlo),
            "null_hi": float(nhi), "p_perm": p_perm,
            "n_pairs": st["n_pairs"], "n_pairs_above_half": st["n_above_half"], "p_sign": st["p_sign"],
            "n_blocks": int(r.ok.sum()), "n_rows": p.n_rows, "n_lineages": len(p.lineages), "n_sites": len(p.positions),
            "lineage_min": float(lt["auc"].min()), "lineage_max": float(lt["auc"].max()),
            "lineage_ci_lo": llo, "lineage_ci_hi": lhi,
            "audit_auc_ties_in_row_order": audit_pooled_auc(canon[m]),
            "n_boot": n_boot, "n_perm": n_perm, "seed": seed,
        })
        by_pair.append(pt.assign(model=m))
        by_lin.append(lt.assign(model=m))
        robust.append(robustness(p, r).assign(model=m))

    d = boots[ft][0] - boots[base][0]
    dlo, dhi = _interval(d)
    diff = pd.DataFrame([{"finetune": ft, "base": base, "difference": results[ft].pooled - results[base].pooled,
                          "ci_lo": dlo, "ci_hi": dhi, "n_boot": n_boot, "seed": seed}])

    # strand C
    q = acc.load_spectrum(spectrum)
    if symmetrise:
        q = acc.symmetrise(q)
    sc_summary, sc_per_aa = [], []
    for m in models:
        site = strand_c_sites(scores, m, roots, q)
        per = strand_c_per_aa(site)
        obs = float(per["rho"].mean())
        null = strand_c_null(site, n_perm=n_perm_c, seed=strand_c_seed)
        boot = strand_c_bootstrap(site, n_boot=n_boot, seed=seed + 3)
        blo, bhi = _interval(boot)
        nlo, nhi = np.percentile(null, [2.5, 97.5])
        sc_summary.append({
            "model": m, "role": guide.model(m).role, "mean_rho": obs, "ci_lo": blo, "ci_hi": bhi,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "null_lo": float(nlo), "null_hi": float(nhi),
            "p_perm_lower": float((np.sum(null <= obs) + 1) / (len(null) + 1)),
            "n_aa": len(per), "n_negative": int((per["rho"] < 0).sum()),
            "n_sites": len(site), "spectrum_file": f"data/spectra/{Path(spectrum).name}", "symmetrised": bool(symmetrise),
            "n_boot": n_boot, "n_perm": n_perm_c, "seed": strand_c_seed,
        })
        sc_per_aa.append(per.assign(model=m))

    return {
        "codon_within_pair_summary": pd.DataFrame(summary),
        "codon_within_pair_difference": diff,
        "codon_within_pair_by_pair": pd.concat(by_pair, ignore_index=True),
        "codon_within_pair_by_lineage": pd.concat(by_lin, ignore_index=True),
        "codon_within_pair_robustness": pd.concat(robust, ignore_index=True),
        "codon_strand_c_summary": pd.DataFrame(sc_summary),
        "codon_strand_c_per_aa": pd.concat(sc_per_aa, ignore_index=True),
    }
