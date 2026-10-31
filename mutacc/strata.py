"""Correlations within codon-distance strata, and amino-acid-pair asymmetry.

No plotting, no paths. Used by Figure 2 and S4.

The question behind the first half: a model that merely prefers chemically similar
replacements will correlate with mutational accessibility anyway, because the genetic code
puts similar amino acids one nucleotide change apart. So the correlation is recomputed
*within* each stratum of codon distance (replacements reachable by 1, 2 or 3 nucleotide
changes), where that shortcut is unavailable, and compared with amino-acid scores that
contain no model and no codon information at all.

Conventions, all inherited from the exploration code so published numbers can be reproduced:

* The estimate is the **mean over lineages** of the per-lineage Spearman correlation.
* The interval is a **percentile bootstrap that resamples sites** (whole positions, with all
  of their replacements, the same positions in every lineage). Resampling rows would treat the
  up-to-nineteen replacements at one site as independent. Resampling lineages would be wrong
  too: the lineages share their sites, so the spread between them is far narrower than the
  uncertainty (``LINEAGE_VALIDITY.md`` section 4, Q2).
* A partial correlation is the Pearson correlation of ranks after removing the linear effect of
  the control's rank. It is computed here from the three rank correlations, which is
  algebraically identical to the residual method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

__all__ = [
    "AA_ORDER", "LOG_FLOOR", "ALL", "MIN_ROWS",
    "wide_scores", "strata_table",
    "pair_asymmetry", "asymmetry_summary",
    "partial_from_ranks", "primary_lineage",
]


def primary_lineage(frame: pd.DataFrame, lineages, chosen: str | None = None) -> str:
    """Pick the one lineage to analyse, by a stated rule rather than by hand.

    ``chosen`` is the guide's own ``primary_lineage``, and it wins: naming the lineage in
    the guide keeps the choice out of the figure scripts, where a lineage name is not
    allowed, and makes it one visible decision rather than a rule the reader has to
    re-derive. It still has to be a lineage that is actually present and scored.

    Figure 2D and S4 used to report the mean over all five lineages. Measured on
    14 Oct 2026, those five agree to a between-lineage SD of 0.001 to 0.013 while
    the site bootstrap that accompanies them is 0.04 to 0.11 wide -- three to
    ninety-six times wider. The lineages share their sites, so they are
    pseudo-replicates for this measurement: averaging them changes no estimate by
    more than about 0.015 and contributes nothing to the uncertainty, while the
    phrase "five lineages" invites a reader to hear fivefold replication that is
    not there. One lineage with the site bootstrap says the same thing honestly.

    The rule, in order:

    1. Prefer a lineage whose root provenance is established, that is, whose
       ``root_type`` is not ``unverified``. In the five-lineage H3N2 set three of
       the five roots are unverified and one of those does not even agree with
       itself about which lineage it is.
    2. Among those, take the one with the most scored replacements.
    3. Break ties on the sorted identifier, so the choice is reproducible.

    ``lineages`` is a sequence of ``(id, root_type)`` pairs, which keeps this
    function independent of the guide dataclasses. Only lineages present in
    ``frame`` are considered.
    """
    counts = frame["lineage"].astype(str).value_counts()
    available = [(str(i), str(t)) for i, t in lineages if str(i) in counts.index]
    if not available:
        raise ValueError("none of the guide's lineages appear in the table")
    if chosen is not None:
        if chosen not in dict(available):
            raise ValueError(f"the guide names {chosen!r} as its primary lineage, but the scored "
                             f"table holds only {sorted(dict(available))}")
        return chosen
    trusted = [p for p in available if p[1] != "unverified"]
    pool = trusted or available
    return min(pool, key=lambda p: (-int(counts[p[0]]), p[0]))[0]

#: Amino acids grouped by side-chain character (small/hydrophobic, polar, acidic, basic), the
#: order the published asymmetry matrices used. The order fixes which direction of each pair is
#: called "A to B", which is arbitrary; nothing reported depends on it except the sign.
AA_ORDER = tuple("GAVLIMFWP" "STCYNQ" "DE" "KRH")

#: Probabilities are clipped here before taking log10, as in the exploration code.
LOG_FLOOR = 1e-32

#: Label of the stratum that pools every replacement.
ALL = "all"

#: A correlation over fewer rows than this is reported as missing rather than guessed.
MIN_ROWS = 20

KEY = ["lineage", "position", "ref_aa", "aa"]


# ---------------------------------------------------------------------------
# Correlations within strata
# ---------------------------------------------------------------------------

def partial_from_ranks(r_xy: float, r_xz: float, r_yz: float) -> float:
    """Partial correlation of x and y given z, from the three pairwise correlations."""
    denom = np.sqrt((1.0 - r_xz ** 2) * (1.0 - r_yz ** 2))
    return float((r_xy - r_xz * r_yz) / denom) if denom > 0 else float("nan")


def wide_scores(frame: pd.DataFrame, model_keys: list[str], properties: list[str]) -> pd.DataFrame:
    """One row per (lineage, site, replacement) with one probability column per model.

    The models must have been scored on exactly the same replacements and agree on
    accessibility and codon distance, which depend on the reference codon and not on the model.
    Anything else is an error: an inner join that quietly drops rows would change what each
    stratum contains.
    """
    needed = {"model", "plm_prob", "mut_prob", "nt_steps", *KEY, *properties}
    missing = needed - set(frame.columns)
    if missing:
        raise KeyError(f"replacement table is missing columns {sorted(missing)}")
    wide = None
    for key in model_keys:
        sub = frame.loc[frame["model"] == key]
        if sub.empty:
            raise ValueError(f"no rows for model {key!r}")
        if sub.duplicated(KEY).any():
            raise ValueError(f"model {key!r}: duplicate (lineage, position, replacement) rows")
        sub = sub.rename(columns={"plm_prob": f"plm::{key}"})
        if wide is None:
            wide = sub[KEY + ["mut_prob", "nt_steps", *properties, f"plm::{key}"]].reset_index(drop=True)
            continue
        merged = wide.merge(sub[KEY + ["mut_prob", "nt_steps", f"plm::{key}"]], on=KEY,
                            how="outer", suffixes=("", "_other"), indicator=True, validate="one_to_one")
        if (merged["_merge"] != "both").any():
            raise ValueError(f"model {key!r} was not scored on the same replacements as the others")
        if not (np.allclose(merged["mut_prob"], merged["mut_prob_other"], rtol=1e-6)
                and (merged["nt_steps"] == merged["nt_steps_other"]).all()):
            raise ValueError(f"model {key!r}: accessibility or codon distance differs between models")
        wide = merged.drop(columns=["mut_prob_other", "nt_steps_other", "_merge"])
    block = wide[[f"plm::{k}" for k in model_keys] + ["mut_prob", *properties]].to_numpy(dtype=float)
    if not np.isfinite(block).all():
        raise ValueError("non-finite probability or property in the replacement table")
    return wide


@dataclass
class _Block:
    lineage: str
    pos: np.ndarray      # sorted site numbers
    steps: np.ndarray    # codon distance per row
    cols: np.ndarray     # n x k: model probs..., accessibility, properties...


def _blocks(wide: pd.DataFrame, columns: list[str]) -> list[_Block]:
    out = []
    for lineage, sub in wide.groupby("lineage", sort=True):
        sub = sub.sort_values(["position", "aa"], kind="stable")
        out.append(_Block(str(lineage), sub["position"].to_numpy(), sub["nt_steps"].to_numpy(),
                          sub[columns].to_numpy(dtype=float)))
    return out


def _measures(model_keys, properties):
    """Ordered (measure, series, control) triples; the layout of every statistic vector."""
    k = len(model_keys)
    out = [("plm_vs_access", m, "") for m in model_keys]
    out += [("property_vs_access", p, "") for p in properties]
    out += [("plm_vs_access", m, p) for m in model_keys for p in properties]
    return out, k


def _stat_vector(cols: np.ndarray, model_keys, properties) -> np.ndarray:
    """Every statistic for one block of rows. NaN when there are too few rows."""
    k, j = len(model_keys), len(properties)
    n_stats = k + j + k * j
    if cols.shape[0] < MIN_ROWS:
        return np.full(n_stats, np.nan)
    ranks = rankdata(cols, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(ranks, rowvar=False)
    acc = k  # accessibility column
    vec = [c[i, acc] for i in range(k)]
    vec += [c[acc + 1 + q, acc] for q in range(j)]
    for i in range(k):
        for q in range(j):
            vec.append(partial_from_ranks(c[i, acc], c[i, acc + 1 + q], c[acc, acc + 1 + q]))
    return np.asarray(vec, dtype=float)


def _strata_vectors(blocks, strata, model_keys, properties, idx_per_block=None):
    """Mean over blocks of the statistic vector, for every stratum. Shape (n_strata, n_stats)."""
    result = np.empty((len(strata), len(model_keys) + len(properties) + len(model_keys) * len(properties)))
    per_block = []
    for b, blk in enumerate(blocks):
        idx = None if idx_per_block is None else idx_per_block[b]
        cols = blk.cols if idx is None else blk.cols[idx]
        steps = blk.steps if idx is None else blk.steps[idx]
        rows = []
        for s in strata:
            sel = cols if s == ALL else cols[steps == s]
            rows.append(_stat_vector(sel, model_keys, properties))
        per_block.append(np.vstack(rows))
    stack = np.stack(per_block)           # (blocks, strata, stats)
    result[:] = stack.mean(axis=0)        # NaN if any lineage is missing a cell
    return result, stack


def _resample_indices(blocks, universe, picked):
    """Row indices for the picked sites, per block, using the same sites in every lineage."""
    out = []
    for blk in blocks:
        starts = np.searchsorted(blk.pos, universe, side="left")[picked]
        ends = np.searchsorted(blk.pos, universe, side="right")[picked]
        lens = ends - starts
        total = int(lens.sum())
        offsets = np.repeat(starts - (np.cumsum(lens) - lens), lens)
        out.append(offsets + np.arange(total))
    return out


def strata_table(
    frame: pd.DataFrame,
    model_keys: list[str],
    properties: list[str],
    *,
    n_boot: int = 200,
    seed: int = 20261009,
    level: float = 0.95,
) -> pd.DataFrame:
    """Correlation with accessibility, for every stratum and every series.

    One row per (measure, series, control, stratum):

    ``estimate``   mean over lineages of the within-lineage Spearman correlation
    ``lo, hi``     site-resampling bootstrap interval of that mean (NaN if ``n_boot`` is 0)
    ``lineage_min, lineage_max``  the spread between lineages (for information; not an interval)
    ``pooled``     one Spearman correlation over all lineages' rows together
    ``n_rows``     rows per lineage in the stratum, averaged over lineages

    ``measure`` is ``plm_vs_access`` (a model's probability against accessibility, optionally
    ``control`` = an amino-acid property held constant) or ``property_vs_access`` (an
    amino-acid property against accessibility; it involves no model).
    """
    wide = wide_scores(frame, model_keys, properties)
    columns = [f"plm::{k}" for k in model_keys] + ["mut_prob", *properties]
    blocks = _blocks(wide, columns)
    steps = sorted(int(s) for s in np.unique(wide["nt_steps"]))
    strata = [ALL, *steps]
    measures, _ = _measures(model_keys, properties)

    point, stack = _strata_vectors(blocks, strata, model_keys, properties)
    if not np.isfinite(point).all():
        bad = np.argwhere(~np.isfinite(stack))[0]
        raise ValueError(
            f"lineage {blocks[bad[0]].lineage!r}, stratum {strata[bad[1]]!r}: fewer than {MIN_ROWS} rows "
            "or a constant column, so a correlation cannot be computed"
        )

    pooled_cols = np.vstack([b.cols for b in blocks])
    pooled_steps = np.concatenate([b.steps for b in blocks])
    pooled = np.vstack([
        _stat_vector(pooled_cols if s == ALL else pooled_cols[pooled_steps == s], model_keys, properties)
        for s in strata
    ])

    lo = np.full_like(point, np.nan)
    hi = np.full_like(point, np.nan)
    if n_boot > 0:
        universe = np.unique(np.concatenate([b.pos for b in blocks]))
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot,) + point.shape)
        for r in range(n_boot):
            picked = rng.integers(0, len(universe), len(universe))
            draws[r], _ = _strata_vectors(blocks, strata, model_keys, properties,
                                          _resample_indices(blocks, universe, picked))
        tail = (1.0 - level) / 2.0 * 100.0
        with np.errstate(invalid="ignore"):
            lo, hi = np.nanpercentile(draws, [tail, 100.0 - tail], axis=0)
        valid = np.isfinite(draws).sum(axis=0)
        if (valid < max(1, n_boot // 2)).any():
            raise ValueError("more than half the bootstrap replicates were undefined for some statistic")

    n_rows = {s: float(np.mean([len(b.steps) if s == ALL else (b.steps == s).sum() for b in blocks]))
              for s in strata}
    rows = []
    for si, s in enumerate(strata):
        for mi, (measure, series, control) in enumerate(measures):
            rows.append({
                "measure": measure, "series": series, "control": control, "stratum": str(s),
                "estimate": point[si, mi], "lo": lo[si, mi], "hi": hi[si, mi],
                "lineage_min": stack[:, si, mi].min(), "lineage_max": stack[:, si, mi].max(),
                "pooled": pooled[si, mi], "n_rows": n_rows[s],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Asymmetry of amino-acid pairs
# ---------------------------------------------------------------------------

def _pair_matrix(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (frame.groupby(["ref_aa", "aa"])[column].mean().unstack("aa")
            .reindex(index=list(AA_ORDER), columns=list(AA_ORDER)))


def _additive_residual(values: np.ndarray, first: np.ndarray, second: np.ndarray, n_aa: int):
    """Remove the part of a pair asymmetry that is a difference of per-amino-acid scores.

    ``asym(A->B)`` is fitted as ``s_A - s_B`` by least squares (one score per amino acid). What
    is left is specific to the pair. Returns (residual, share of variance the scores explained).
    """
    design = np.zeros((len(values), n_aa))
    design[np.arange(len(values)), first] = 1.0
    design[np.arange(len(values)), second] = -1.0
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    resid = values - design @ coef
    # The design has no intercept -- ``s_A - s_B`` cannot produce a constant -- so the
    # total is taken about zero, not about the mean. Dividing an uncentred fit by a
    # centred total is not a share of anything: on data with an offset and no additive
    # structure it goes large and negative (tests/test_known_defects.py).
    total = (values ** 2).sum()
    return resid, float(1.0 - (resid ** 2).sum() / total) if total > 0 else float("nan")


def pair_asymmetry(frame: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Direction of every amino-acid pair, for the model and for mutational accessibility.

    For each ordered pair the mean log10 value (over every site and lineage in the table) is
    taken, and the asymmetry of an unordered pair {A, B} is ``value(A->B) - value(B->A)``. A
    symmetric score such as Grantham distance or BLOSUM62 gives exactly zero for every pair, so
    it cannot express direction at all.

    Columns: ``aa_from, aa_to`` (in ``AA_ORDER``), ``mut_asym, plm_asym``, and ``mut_resid,
    plm_resid``: the same asymmetries after removing each amino acid's overall tendency
    (see :func:`_additive_residual`). The per-amino-acid tendency matters because a model
    that simply favours common amino acids, as the accessibility does, produces asymmetry
    with no knowledge of any particular pair.
    """
    sub = frame.loc[frame["model"] == model_key]
    if sub.empty:
        raise ValueError(f"no rows for model {model_key!r}")
    letters = set(AA_ORDER)
    if not (set(sub["ref_aa"]) <= letters and set(sub["aa"]) <= letters):
        raise ValueError("replacement table holds symbols outside the 20 amino acids")
    if (sub["ref_aa"] == sub["aa"]).any():
        raise ValueError("replacement table holds synonymous rows (ref_aa == aa)")
    sub = sub.assign(
        log_plm=np.log10(np.clip(sub["plm_prob"].to_numpy(float), LOG_FLOOR, None)),
        log_mut=np.log10(np.clip(sub["mut_prob"].to_numpy(float), LOG_FLOOR, None)),
    )
    plm, mut = _pair_matrix(sub, "log_plm"), _pair_matrix(sub, "log_mut")
    first, second = np.triu_indices(len(AA_ORDER), k=1)
    mut_asym = (mut - mut.T).to_numpy()[first, second]
    plm_asym = (plm - plm.T).to_numpy()[first, second]
    ok = np.isfinite(mut_asym) & np.isfinite(plm_asym)
    if ok.sum() < MIN_ROWS:
        raise ValueError(f"only {int(ok.sum())} amino-acid pairs observed in both directions")
    first, second, mut_asym, plm_asym = first[ok], second[ok], mut_asym[ok], plm_asym[ok]
    mut_resid, _ = _additive_residual(mut_asym, first, second, len(AA_ORDER))
    plm_resid, _ = _additive_residual(plm_asym, first, second, len(AA_ORDER))
    return pd.DataFrame({
        "aa_from": [AA_ORDER[i] for i in first], "aa_to": [AA_ORDER[j] for j in second],
        "mut_asym": mut_asym, "plm_asym": plm_asym, "mut_resid": mut_resid, "plm_resid": plm_resid,
    })


def asymmetry_summary(pairs: pd.DataFrame) -> dict:
    """Headline numbers for one model's :func:`pair_asymmetry` table."""
    from scipy.stats import spearmanr

    n = len(pairs)
    agree = int((np.sign(pairs["mut_asym"]) == np.sign(pairs["plm_asym"])).sum())
    index = {a: i for i, a in enumerate(AA_ORDER)}
    first = pairs["aa_from"].map(index).to_numpy()
    second = pairs["aa_to"].map(index).to_numpy()
    _, share_mut = _additive_residual(pairs["mut_asym"].to_numpy(), first, second, len(AA_ORDER))
    _, share_plm = _additive_residual(pairs["plm_asym"].to_numpy(), first, second, len(AA_ORDER))
    return {
        "n_pairs": n,
        "n_agree": agree,
        "rho_overall": float(spearmanr(pairs["mut_asym"], pairs["plm_asym"]).statistic),
        "rho_pair_specific": float(spearmanr(pairs["mut_resid"], pairs["plm_resid"]).statistic),
        "share_by_amino_acid_mut": share_mut,
        "share_by_amino_acid_plm": share_plm,
    }
