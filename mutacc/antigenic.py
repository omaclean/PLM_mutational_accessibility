"""Does fine-tuning help more at known antigenic sites? Statistics for figure S8. No plotting.

Antigenic sites are published annotation, so there is no region boundary to tune. The question
is whether the fine-tuned model's gain in recovering structural contacts is larger at those
residues than elsewhere in the same domain. The analysis is a **negative result** and the
figure says so; this module only measures.

Numbering
---------
Structure residues are in mature-chain numbering. The antigenic-site table carries both the
mature position (``ha1_position``) and the precursor position (``ha0_position``), which is the
mature position plus 16. Residues are joined on the mature position **only**; ``load_sites``
checks that the two columns differ by exactly the precursor offset on every row, so a table
already shifted once cannot be shifted again.

Gauge
-----
One scale only: the zero-sum gauge of Zhang and Ovchinnikov (2024), log probability minus its
average over the 20 residues. The shipped tables still carry four other gauges, because they
are what the archive exported, but nothing here reads them: ``select_gauge`` drops them on the
way in. The earlier code summarised all five and then discarded four, which bought nothing but
a matching random stream.

Random numbers
--------------
``group_summary`` and ``site_summary`` take a generator and consume it in a fixed order
(structure, then stratum or site, in sorted order; one ``choice`` call per interval with at
least five residues). Intervals therefore depend on the order of calls, which is fixed here
but is not the order the five-gauge code used: interval ends sit within about 0.002 of the
archive's, and no point estimate moves at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mutacc.audit import HA_PRECURSOR_OFFSET, check_columns

__all__ = [
    "SEED", "N_BOOT", "GAUGE", "GAUGE_LABEL", "STRATUM_ANTIGENIC", "STRATUM_REST", "load_sites",
    "select_gauge", "annotate_residues", "paired_bootstrap", "group_summary", "site_summary", "contrast",
]

#: The only scale measured on: Zhang and Ovchinnikov (2024)'s double-centred log-probability gauge.
GAUGE = "zerosum"
GAUGE_LABEL = "Zero-sum"

SEED = 20261008
N_BOOT = 2000
MIN_RESIDUES = 5
#: The archive stores every per-residue AUROC and its delta rounded to 4 decimals, so
#: delta can differ from (fine-tuned - base) by up to 1.5e-4. Anything larger is an error.
ROUNDING_TOLERANCE = 2e-4
STRATUM_ANTIGENIC = "antigenic"
STRATUM_REST = "rest"


def load_sites(sites: pd.DataFrame) -> dict[int, str]:
    """Validate an antigenic-site table and return ``{mature position: site letter}``."""
    check_columns(sites, ["site", "ha1_position", "ha0_position"], "antigenic sites")
    shift = (sites["ha0_position"] - sites["ha1_position"]).unique()
    if len(shift) != 1 or int(shift[0]) != HA_PRECURSOR_OFFSET:
        raise ValueError(f"antigenic sites: precursor minus mature position must be exactly "
                         f"{HA_PRECURSOR_OFFSET} on every row, found {sorted(int(s) for s in shift)}")
    if sites["ha1_position"].duplicated().any():
        dup = sites.loc[sites["ha1_position"].duplicated(keep=False), "ha1_position"].unique()[:5]
        raise ValueError(f"antigenic sites: position(s) {list(dup)} assigned more than once")
    if sites["site"].isna().any():
        raise ValueError("antigenic sites: a residue has no site letter")
    return dict(zip(sites["ha1_position"].astype(int), sites["site"].astype(str)))


def select_gauge(res: pd.DataFrame) -> pd.DataFrame:
    """Keep the zero-sum rows of a per-residue table and drop the gauge columns.

    The shipped table carries five gauges; only this one is ever measured on. Dropping the
    columns here is what stops a gauge choice reappearing further down as a loop variable.
    """
    if "gauge" not in res.columns:
        return res
    out = res.loc[res["gauge"] == GAUGE]
    if out.empty:
        raise ValueError(f"per-residue accuracy: no {GAUGE!r} rows; it carries "
                         f"{sorted(set(res['gauge']))}")
    return out.drop(columns=[c for c in ("gauge", "gauge_label", "headline") if c in out.columns])


def annotate_residues(res: pd.DataFrame, site_of: dict[int, str]) -> pd.DataFrame:
    """Add ``site`` and ``antigenic`` to a per-residue table (mature numbering) and keep the
    residues that are scored and inside the domain being compared."""
    check_columns(res, ["structure", "res_id", "auroc_finetune", "auroc_base", "delta", "in_domain"],
                  "per-residue accuracy")
    d = select_gauge(res).copy()
    d["site"] = d["res_id"].map(site_of)
    d["antigenic"] = d["site"].notna()
    d = d.loc[d["in_domain"].astype(bool) & d["delta"].notna()].reset_index(drop=True)
    if d.empty:
        raise ValueError("per-residue accuracy: no scored residues inside the domain")
    if not np.allclose(d["delta"], d["auroc_finetune"] - d["auroc_base"], atol=ROUNDING_TOLERANCE):
        raise ValueError("per-residue accuracy: delta is not fine-tuned minus base")
    return d


def paired_bootstrap(delta, rng: np.random.Generator, n_boot: int = N_BOOT):
    """Mean, 95% percentile interval and count of a per-residue delta. NaN below five residues."""
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < MIN_RESIDUES:
        return np.nan, np.nan, np.nan, len(d)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(d)


def _row(mean, lo, hi, n):
    return {"n_residues": n, "mean_delta": mean, "ci_lo": lo, "ci_hi": hi,
            "crosses_zero": not (lo > 0 or hi < 0) if np.isfinite(lo) else True}


def group_summary(d: pd.DataFrame, rng: np.random.Generator, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Per structure and stratum (antigenic, rest): mean gain and its interval."""
    rows = []
    for struct in sorted(d["structure"].unique()):
        sub = d.loc[d["structure"] == struct]
        for label, part in ((STRATUM_ANTIGENIC, sub[sub["antigenic"]]), (STRATUM_REST, sub[~sub["antigenic"]])):
            mean, lo, hi, n = paired_bootstrap(part["delta"].to_numpy(), rng, n_boot)
            rows.append({"structure": struct, "stratum": label, **_row(mean, lo, hi, n),
                         "auroc_finetune": part["auroc_finetune"].mean(),
                         "auroc_base": part["auroc_base"].mean()})
    return pd.DataFrame(rows)


def site_summary(d: pd.DataFrame, sites: list[str], rng: np.random.Generator,
                 n_boot: int = N_BOOT) -> pd.DataFrame:
    """Per structure and named site: mean gain and its interval.

    Not drawn: S13 pools the sites into one class so the comparison is a single test rather
    than one uncorrected look per site. Kept because the per-site spread is worth being able
    to check -- site C stands well above the others in both structures.
    """
    rows = []
    for struct in sorted(d["structure"].unique()):
        for site in sorted(sites):
            sub = d.loc[(d["structure"] == struct) & (d["site"] == site)]
            mean, lo, hi, n = paired_bootstrap(sub["delta"].dropna().to_numpy(), rng, n_boot)
            rows.append({"structure": struct, "site": site, **_row(mean, lo, hi, n)})
    return pd.DataFrame(rows)


def contrast(a, b, *, n_boot: int = N_BOOT, n_perm: int = 10000, seed: int = SEED) -> dict:
    """Mean of ``a`` minus mean of ``b``: percentile interval from resampling each group, and a
    two-sided permutation p-value from shuffling group labels.

    Two overlapping intervals are not a test of whether two means differ; this is. Residues are not
    independent (neighbours share structure), so the interval is if anything too narrow.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_RESIDUES or len(b) < MIN_RESIDUES:
        raise ValueError(f"contrast: need at least {MIN_RESIDUES} residues in each group, have {len(a)} and {len(b)}")
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    boot = (rng.choice(a, size=(n_boot, len(a))).mean(axis=1) - rng.choice(b, size=(n_boot, len(b))).mean(axis=1))
    pooled = np.concatenate([a, b])
    shuffled = np.argsort(rng.random((n_perm, len(pooled))), axis=1)
    perm = pooled[shuffled[:, :len(a)]].mean(axis=1) - pooled[shuffled[:, len(a):]].mean(axis=1)
    p = float(((np.abs(perm) >= abs(point) - 1e-15).sum() + 1) / (n_perm + 1))
    return {"difference": point, "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "p_perm": p, "n_a": len(a), "n_b": len(b), "n_boot": n_boot, "n_perm": n_perm}
