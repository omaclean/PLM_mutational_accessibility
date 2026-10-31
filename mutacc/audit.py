"""Statistics for the lineage-set validity audit (figure S14). No plotting, no paths.

Four questions, each a function here so a figure only has to draw the answer:

``within_replacement``  Hold the amino-acid replacement fixed (say Ala to Val). Does the model put
    that target higher at sites where the reference codon is one nucleotide step away than where
    it is two or three? Inside one replacement every amino-acid quantity (chemistry, Grantham,
    BLOSUM62) is a constant, so anything the model tracks is codon-level information.
``persistence``  Does the model's total probability of leaving a residue track how likely a
    mutation at that codon is to be synonymous? Compared with a permutation null.
``interval_unit``  The same AUROC with an interval from resampling lineages and from resampling
    sites. The lineages share their sites, so the two can differ by a large factor.
``corpus_split``  Fine-tuned minus base AUROC, split by whether the replacement occurs in the
    fine-tuning corpus.

Every bootstrap resamples SITES, never rows, because a site recurs in every lineage and a
replacement recurs across sites' neighbours: rows are not independent observations.

Random-number use is fixed and documented, so the shipped tables can be rebuilt bit for bit: each
bootstrap makes one ``rng.choice(n_positions, size=n_positions)`` call per draw, from a generator
seeded once per call.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

from mutacc.accessibility import BASES, GENETIC_CODE, symmetrise

__all__ = [
    "SEED", "HA_PRECURSOR_OFFSET", "rank_auroc", "block_pooled_auc", "within_replacement",
    "persistence_site_table", "synonymous_fraction", "persistence", "interval_unit",
    "corpus_membership", "corpus_split", "check_columns",
]

SEED = 20261008
#: Position in the precursor (HA0) numbering minus position in the mature chain (HA1 + HA2).
#: Exported tables carry the precursor position, and anything joined on the mature numbering
#: must subtract this exactly once.
HA_PRECURSOR_OFFSET = 16
MIN_EXPOSED, MIN_UNEXPOSED, MIN_ROWS = 5, 5, 15
EXPECTED_ALTERNATIVES = 19
SKIPPED_RESIDUES = ("M", "W")  # one codon each: no synonymous choice to track
MIN_SITES_PER_RESIDUE = 6


def check_columns(frame: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{what}: missing column(s) {missing}; has {sorted(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{what}: empty table")


def _finite(values, what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{what} contains missing or non-finite values")
    return arr


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def rank_auroc(score, outcome) -> float:
    """Rank-sum AUROC (ties averaged). NaN when either class is empty."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(outcome).astype(bool)
    if score.shape != y.shape or score.ndim != 1:
        raise ValueError("score and outcome must be 1-D and the same length")
    n1 = int(y.sum())
    n2 = len(y) - n1
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def _percentile_ci(draws: np.ndarray) -> tuple[float, float]:
    if np.isnan(draws).all():
        raise ValueError("every bootstrap draw was undefined (one class empty); cannot form an interval")
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _rows_by_position(position: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sorted unique positions and, for each, the row indices holding it."""
    order = np.argsort(position, kind="stable")
    uniq, start = np.unique(position[order], return_index=True)
    return uniq, np.split(order, start[1:])


# ---------------------------------------------------------------------------
# Strand B: within-replacement AUROC of nucleotide distance
# ---------------------------------------------------------------------------

def block_pooled_auc(block: np.ndarray, value: np.ndarray, exposed: np.ndarray, n_blocks: int):
    """Van Elteren weighted mean of per-block AUROCs, and the mask of blocks that were scored.

    ``exposed`` is 1 where the replacement is one nucleotide step away. A block (one lineage and
    one replacement) is scored only when it has at least five of each class and fifteen rows.
    Ranks are taken within the block; ties are broken arbitrarily, which costs nothing for a
    continuous model score.
    """
    order = np.lexsort((value, block))
    b_sorted = block[order]
    counts = np.bincount(b_sorted, minlength=n_blocks)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ranks = np.empty(len(block), float)
    ranks[order] = np.arange(len(block)) - starts[b_sorted] + 1

    r1 = np.bincount(block, weights=ranks * exposed, minlength=n_blocks)
    n1 = np.bincount(block, weights=exposed, minlength=n_blocks)
    n = np.bincount(block, minlength=n_blocks).astype(float)
    n2 = n - n1
    ok = (n1 >= MIN_EXPOSED) & (n2 >= MIN_UNEXPOSED) & (n >= MIN_ROWS)
    if not ok.any():
        return float("nan"), ok
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)
    w = np.where(ok, n1 * n2 / (n1 + n2 + 1.0), 0.0)
    return float(np.nansum(auc[ok] * w[ok]) / w[ok].sum()), ok


def _prepare_within(frame: pd.DataFrame) -> dict:
    check_columns(frame, ["lineage", "position", "ref_aa", "aa", "plm_prob", "nt_steps"], "within_replacement")
    _finite(frame["plm_prob"], "plm_prob")
    d = frame.reset_index(drop=True)
    if d.duplicated(["lineage", "position", "aa"]).any():
        raise ValueError("within_replacement: a (lineage, position, aa) cell appears twice; pass one model")
    # The outcome is where this target ranks among the alternatives at its own site: a site's
    # overall mutability dominates the raw scale and has nothing to do with codons.
    site_rank = d.groupby(["lineage", "position"])["plm_prob"].rank().to_numpy()
    pair = d["ref_aa"].astype(str) + ">" + d["aa"].astype(str)
    block, _ = pd.factorize(d["lineage"].astype(str) + "|" + pair)
    block = block.astype(np.int64)
    pair_of_block = (pd.DataFrame({"b": block, "p": pair.to_numpy()}).drop_duplicates("b")
                     .sort_values("b")["p"].to_numpy())
    return {"block": block, "n_blocks": int(block.max()) + 1, "value": site_rank,
            "exposed": (d["nt_steps"].to_numpy() == 1).astype(float),
            "position": d["position"].to_numpy(), "lineage": d["lineage"].to_numpy(),
            "pair_of_block": pair_of_block, "n_rows": len(d)}


def within_replacement(frame: pd.DataFrame, *, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Within-replacement AUROC for one model, its site-clustered interval and per-lineage values.

    0.5 means one-step replacements rank no higher than two- and three-step ones once the
    replacement is held fixed. ``frame`` must hold a single model.
    """
    a = _prepare_within(frame)
    auc, ok = block_pooled_auc(a["block"], a["value"], a["exposed"], a["n_blocks"])
    if not ok.any():
        raise ValueError("within_replacement: no replacement block has enough one-step and multi-step sites")

    per_lineage = {}
    for lin in np.unique(a["lineage"]):
        m = a["lineage"] == lin
        b, _ = pd.factorize(a["block"][m])
        per_lineage[str(lin)], _ = block_pooled_auc(b.astype(np.int64), a["value"][m], a["exposed"][m],
                                                    int(b.max()) + 1)

    lo = hi = float("nan")
    if n_boot:
        rng = np.random.default_rng(seed)
        positions, groups = _rows_by_position(a["position"])
        draws = np.full(n_boot, np.nan)
        for i in range(n_boot):
            pick = rng.choice(len(positions), size=len(positions), replace=True)
            rows = np.concatenate([groups[p] for p in pick])
            bl, _ = pd.factorize(a["block"][rows])
            draws[i], _ = block_pooled_auc(bl.astype(np.int64), a["value"][rows], a["exposed"][rows],
                                           int(bl.max()) + 1)
        lo, hi = _percentile_ci(draws)
    return {"auc": auc, "lo": lo, "hi": hi, "n_rows": a["n_rows"], "n_boot": n_boot,
            "n_lineages": len(per_lineage), "n_blocks_scored": int(ok.sum()),
            "n_pairs_scored": int(pd.unique(a["pair_of_block"][ok]).size), "per_lineage": per_lineage}


# ---------------------------------------------------------------------------
# Strand C: does the model's escape probability track codon persistence?
# ---------------------------------------------------------------------------

def synonymous_fraction(codon: str, rates: Mapping[tuple[str, str], float]) -> float:
    """P(a mutation at this codon is synonymous | a mutation occurs), from nucleotide rates."""
    if len(codon) != 3 or set(codon) - set(BASES):
        raise ValueError(f"not a codon: {codon!r}")
    aa, total, syn = GENETIC_CODE[codon], 0.0, 0.0
    for p in range(3):
        for nb in BASES:
            if nb == codon[p]:
                continue
            r = rates[(codon[p], nb)]
            total += r
            if GENETIC_CODE[codon[:p] + nb + codon[p + 1:]] == aa:
                syn += r
    if total <= 0:
        raise ValueError(f"codon {codon}: no mutation has a positive rate")
    return syn / total


def reversible_rates(q: np.ndarray) -> dict[tuple[str, str], float]:
    """The symmetrised 4x4 spectrum as a ``{(from, to): rate}`` mapping."""
    s = symmetrise(np.asarray(q, dtype=float))
    return {(a, b): float(s[i, j]) for i, a in enumerate(BASES) for j, b in enumerate(BASES) if i != j}


def persistence_site_table(frame: pd.DataFrame, codons: pd.DataFrame, rates) -> pd.DataFrame:
    """One row per model and site: total escape probability, residue and synonymous fraction S.

    ``frame`` has ``model, lineage, position, ref_aa, plm_prob`` (19 alternatives per site);
    ``codons`` has ``lineage, position, ref_codon``. The codon is checked to translate to the
    table's own reference residue, so a root file that is not the sequence that was scored cannot
    pass quietly.
    """
    check_columns(frame, ["model", "lineage", "position", "ref_aa", "plm_prob"], "persistence")
    check_columns(codons, ["lineage", "position", "ref_codon"], "persistence codons")
    _finite(frame["plm_prob"], "plm_prob")
    esc = (frame.groupby(["model", "lineage", "position"])
           .agg(escape=("plm_prob", "sum"), ref_aa=("ref_aa", "first"), n_alt=("plm_prob", "size"))
           .reset_index())
    if (esc["n_alt"] != EXPECTED_ALTERNATIVES).any():
        raise ValueError(f"persistence: every site must have {EXPECTED_ALTERNATIVES} alternatives")
    esc = esc.merge(codons.drop_duplicates(["lineage", "position"]), on=["lineage", "position"], how="left",
                    validate="many_to_one")
    if esc["ref_codon"].isna().any():
        raise ValueError(f"persistence: {int(esc['ref_codon'].isna().sum())} sites have no reference codon")
    translated = esc["ref_codon"].map(GENETIC_CODE)
    if (translated != esc["ref_aa"]).any():
        bad = esc.loc[translated != esc["ref_aa"], ["lineage", "position", "ref_aa", "ref_codon"]].head(3)
        raise ValueError(f"persistence: reference codon does not translate to the scored residue:\n{bad}")
    site = (esc.groupby(["model", "position"])
            .agg(escape=("escape", "mean"),
                 ref_aa=("ref_aa", lambda s: s.mode().iat[0]),
                 ref_codon=("ref_codon", lambda s: s.mode().iat[0]))
            .reset_index())
    site["S"] = [synonymous_fraction(c, rates) for c in site["ref_codon"]]
    return site


def _unit_ranks(x) -> np.ndarray:
    r = rankdata(x).astype(float)
    r = r - r.mean()
    n = np.linalg.norm(r)
    return r / n if n > 0 else r


def persistence(site: pd.DataFrame, *, n_perm: int = 10000, seed: int = 20260928):
    """Mean within-residue Spearman of S with escape, against a permutation null.

    Within each reference residue (not Met or Trp, at least six sites, S not constant) the sites'
    S values are permuted; the null is the mean over residues. The generator is created once and
    consumed model by model, residue by residue, in sorted order.
    """
    check_columns(site, ["model", "ref_aa", "S", "escape"], "persistence")
    rng = np.random.default_rng(seed)
    rows, per_aa = [], []
    for model, sm in site.groupby("model"):
        rhos, null_cols = [], []
        for aa, s in sm.groupby("ref_aa"):
            if aa in SKIPPED_RESIDUES or len(s) < MIN_SITES_PER_RESIDUE or s["S"].nunique() < 2:
                continue
            rs, rv = _unit_ranks(s["S"].to_numpy()), _unit_ranks(s["escape"].to_numpy())
            perm = np.argsort(rng.random((n_perm, len(s))), axis=1)
            null_cols.append(rs[perm] @ rv)
            rhos.append(float(rs @ rv))
            per_aa.append({"model": model, "aa": aa, "n_sites": len(s), "rho": rhos[-1]})
        if not rhos:
            raise ValueError(f"persistence: model {model!r} has no residue with enough sites")
        rhos = np.array(rhos)
        null = np.mean(np.column_stack(null_cols), axis=1)
        obs, k_neg = float(rhos.mean()), int((rhos < 0).sum())
        lo, hi = np.percentile(null, [2.5, 97.5])
        rows.append({"model": model, "n_aa": len(rhos), "mean_rho": obs, "n_negative": k_neg,
                     "p_sign": float(binomtest(k_neg, len(rhos), 0.5, alternative="greater").pvalue),
                     "p_perm": float(((null <= obs).sum() + 1) / (n_perm + 1)),
                     "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                     "null_lo": float(lo), "null_hi": float(hi),
                     "z": float((obs - null.mean()) / null.std(ddof=1)), "n_perm": n_perm})
    return pd.DataFrame(rows), pd.DataFrame(per_aa)


# ---------------------------------------------------------------------------
# Resampling unit: lineages or sites
# ---------------------------------------------------------------------------

def _join_observed(scores: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    check_columns(scores, ["lineage", "position", "aa", "plm_prob"], "scores")
    check_columns(observed, ["lineage", "position", "aa", "obs_present"], "observed")
    _finite(scores["plm_prob"], "plm_prob")
    if scores.duplicated(["lineage", "position", "aa"]).any():
        raise ValueError("scores: a (lineage, position, aa) cell appears twice; pass one model")
    if observed.duplicated(["lineage", "position", "aa"]).any():
        raise ValueError("observed: a (lineage, position, aa) cell appears twice")
    joined = scores.merge(observed[["lineage", "position", "aa", "obs_present"]],
                          on=["lineage", "position", "aa"], how="left", validate="one_to_one")
    if joined["obs_present"].isna().any():
        raise ValueError(f"{int(joined['obs_present'].isna().sum())} scored replacements have no observed flag")
    if len(joined) != len(scores):
        raise ValueError("joining observed flags changed the row count")
    joined["obs_present"] = joined["obs_present"].astype(float) > 0.5
    return joined


def per_lineage_auroc(scores: pd.DataFrame, observed: pd.DataFrame) -> pd.Series:
    d = _join_observed(scores, observed)
    out = {lin: rank_auroc(g["plm_prob"].to_numpy(), g["obs_present"].to_numpy()) for lin, g in d.groupby("lineage")}
    return pd.Series(out, name="auroc")


def interval_unit(scores: pd.DataFrame, observed: pd.DataFrame, *, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Mean per-lineage AUROC with two intervals: resampling lineages, and resampling sites.

    Resampling lineages is what a band over lineages does. Resampling sites carries every
    lineage's copy of a site together, which is the honest unit when the lineages share sites.
    """
    d = _join_observed(scores, observed)
    per_lin = per_lineage_auroc(scores, observed)
    if len(per_lin) < 2:
        raise ValueError("interval_unit: need at least two lineages to resample")
    vals = per_lin.to_numpy(float)
    rng = np.random.default_rng(seed)
    by_lineage = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    lo_l, hi_l = _percentile_ci(by_lineage)

    score, y = d["plm_prob"].to_numpy(), d["obs_present"].to_numpy()
    positions, groups = _rows_by_position(d["position"].to_numpy())
    rng = np.random.default_rng(seed)
    by_site = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(len(positions), size=len(positions), replace=True)
        rows = np.concatenate([groups[p] for p in pick])
        by_site[i] = rank_auroc(score[rows], y[rows])
    lo_s, hi_s = _percentile_ci(by_site)
    return {"mean_auroc": float(vals.mean()), "per_lineage_sd": float(vals.std(ddof=1)),
            "per_lineage_min": float(vals.min()), "per_lineage_max": float(vals.max()),
            "n_lineages": len(vals), "n_sites": len(positions), "n_boot": n_boot,
            "lineage_boot_lo": lo_l, "lineage_boot_hi": hi_l, "lineage_boot_width": hi_l - lo_l,
            "site_boot_lo": lo_s, "site_boot_hi": hi_s, "site_boot_width": hi_s - lo_s,
            "width_ratio": (hi_s - lo_s) / (hi_l - lo_l)}


# ---------------------------------------------------------------------------
# Training-corpus leakage
# ---------------------------------------------------------------------------

def corpus_membership(backbone: str, training: Iterable[str], *, offset: int = HA_PRECURSOR_OFFSET) -> pd.DataFrame:
    """Per (precursor position, amino acid): how many training sequences carry it.

    ``backbone`` and ``training`` are aligned rows of one alignment; ``backbone`` is the first.
    The k-th ungapped backbone residue is mature residue k, which is precursor position k + offset.
    The offset is applied here and nowhere else.
    """
    cols, k = {}, 0
    for c, ch in enumerate(backbone):
        if ch != "-":
            k += 1
            cols[k] = c
    training = list(training)
    if not training:
        raise ValueError("corpus_membership: no training sequences")
    if any(len(s) != len(backbone) for s in training):
        raise ValueError("corpus_membership: sequences are not aligned to the backbone length")
    rows = []
    for k, c in cols.items():
        counts = Counter(s[c] for s in training)
        depth = sum(v for a, v in counts.items() if a.isalpha())
        for aa, v in counts.items():
            if aa.isalpha():
                rows.append({"position": k + offset, "residue": k, "backbone_aa": backbone[c], "aa": aa,
                             "corpus_count": v, "corpus_depth": depth})
    return pd.DataFrame(rows)


def _corpus_cells(scores: pd.DataFrame, observed: pd.DataFrame, membership: pd.DataFrame,
                  finetune: str, base: str) -> pd.DataFrame:
    check_columns(scores, ["model", "lineage", "position", "aa", "plm_prob"], "scores")
    check_columns(membership, ["position", "residue", "aa", "corpus_count"], "membership")
    if (membership["position"] - membership["residue"] != HA_PRECURSOR_OFFSET).any():
        raise ValueError("membership: position minus residue is not the precursor offset on every row")
    have = set(scores["model"].unique())
    if finetune not in have or base not in have:
        raise ValueError(f"corpus_split: need models {finetune!r} and {base!r}; scores hold {sorted(have)}")
    wide = (scores.loc[scores["model"].isin([finetune, base])]
            .pivot_table(index=["position", "aa"], columns="model", values="plm_prob").reset_index())
    obs = (observed.groupby(["position", "aa"])["obs_present"].max().astype(float) > 0.5).rename("obs").reset_index()
    cells = wide.merge(obs, on=["position", "aa"], how="left", validate="one_to_one")
    if cells["obs"].isna().any():
        raise ValueError("corpus_split: some scored cells have no observed flag")
    # Cells with no corpus column (the signal peptide) drop out here; every other cell is kept.
    full = membership.drop_duplicates(["position", "aa"])[["position", "aa", "corpus_count"]]
    cells = cells.merge(full, on=["position", "aa"], how="left")
    covered = set(membership["position"])
    cells = cells.loc[cells["position"].isin(covered)].copy()
    cells["corpus_count"] = cells["corpus_count"].fillna(0)
    cells["in_corpus"] = cells["corpus_count"] > 0
    return cells.reset_index(drop=True)


def corpus_split(scores: pd.DataFrame, observed: pd.DataFrame, membership: pd.DataFrame, *,
                 finetune: str, base: str, n_boot: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Fine-tuned minus base AUROC on replacements absent from, and present in, the corpus.

    A cell is a (position, amino acid); its model probability is the mean over lineages and it is
    observed if any lineage observed it. The interval resamples positions and re-splits each draw,
    so both groups move together; one generator is consumed absent-group first, then present.
    """
    cells = _corpus_cells(scores, observed, membership, finetune, base)
    positions, groups = _rows_by_position(cells["position"].to_numpy())
    ft, bs = cells[finetune].to_numpy(float), cells[base].to_numpy(float)
    y, inside = cells["obs"].to_numpy(bool), cells["in_corpus"].to_numpy(bool)
    rng = np.random.default_rng(seed)
    out = []
    for flag in (False, True):
        m = inside == flag
        if not m.any():
            raise ValueError(f"corpus_split: no replacements with in_corpus={flag}")
        a_ft, a_bs = rank_auroc(ft[m], y[m]), rank_auroc(bs[m], y[m])
        draws = np.full(n_boot, np.nan)
        for i in range(n_boot):
            pick = rng.choice(len(positions), size=len(positions), replace=True)
            rows = np.concatenate([groups[p] for p in pick])
            rows = rows[inside[rows] == flag]
            draws[i] = rank_auroc(ft[rows], y[rows]) - rank_auroc(bs[rows], y[rows])
        lo, hi = _percentile_ci(draws) if n_boot else (float("nan"), float("nan"))
        out.append({"in_corpus": flag, "n_cells": int(m.sum()), "n_observed": int(y[m].sum()),
                    "prevalence": float(y[m].mean()), "auroc_finetune": a_ft, "auroc_base": a_bs,
                    "advantage": a_ft - a_bs, "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot})
    return pd.DataFrame(out)
