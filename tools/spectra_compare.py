"""Compare nucleotide mutation spectra in codon space (S5, and the non-exclusivity panel of S4).

Each spectrum is a 4x4 matrix (``data/spectra``). It is turned into a 64 x 20 table (chance that
a codon changes into any codon of each amino acid), then "compressed" to 20 x 20 by averaging
over the codons of each source amino acid and rebuilt. The statistic is the percent of variance
in a target table retained by a predictor table: 100 * (1 - var(residual) / var(target)).
Every matrix is first scaled to the same total rate, so only the shape of the spectrum matters.
Genetic code from Biopython (via mutacc.accessibility), never typed.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mutacc import accessibility as acc  # noqa: E402

TOTAL_RATE = 3.0e-4   # total of the twelve rates after scaling, as in the archive analysis


def codon_table(q: np.ndarray) -> np.ndarray:
    """64 x 20 change table at a common total rate; own-amino-acid column set to NaN."""
    q = np.array(q, dtype=float)
    np.fill_diagonal(q, 0.0)
    if not q.any():
        raise ValueError("spectrum has no non-zero rate")
    q = q * (TOTAL_RATE / q.sum())
    table = acc.codon_aa_table(q)
    for i, c in enumerate(acc.CODONS):
        aa = acc.GENETIC_CODE[c]
        if aa != "*":
            table[i, acc.aa_index(aa)] = np.nan
    return table


def compress(table: np.ndarray) -> np.ndarray:
    """Average over the codons of each source amino acid, then rebuild 64 rows. Stop codons are NaN."""
    out = np.full_like(table, np.nan)
    for aa in acc.AA20:
        rows = [i for i, c in enumerate(acc.CODONS) if acc.GENETIC_CODE[c] == aa]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # own-amino-acid column is all NaN by design
            avg = np.nanmean(table[rows], axis=0)
        out[rows] = avg
    return out


def retained_and_corr(target: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    a, b = target.ravel(), pred.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    a, b = a[ok], b[ok]
    tv = np.var(a)
    ret = 100.0 * (1.0 - np.var(a - b) / tv) if tv > 0 else float("nan")
    corr = float(np.corrcoef(a, b)[0, 1]) if np.var(b) > 0 else float("nan")
    return float(ret), corr


def load_sources(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path)
    need = {"key", "label", "kind", "free_nt_parameters"}
    if need - set(s.columns):
        raise ValueError(f"{path}: missing columns {sorted(need - set(s.columns))}")
    if s["key"].duplicated().any():
        raise ValueError(f"{path}: duplicate keys")
    return s


def pairwise(spectra_dir: Path, sources: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Long table: one row per (target, predictor)."""
    tabs = {}
    for k in order:
        f = Path(spectra_dir) / f"{k}.csv"
        if not f.is_file():
            raise FileNotFoundError(f)
        tabs[k] = codon_table(acc.load_spectrum(f))
    rec = {k: compress(t) for k, t in tabs.items()}
    rows = []
    for a in order:
        for b in order:
            r_c, _ = retained_and_corr(tabs[a], rec[b])
            r_u, corr = retained_and_corr(tabs[a], tabs[b])
            rows.append({"target": a, "predictor": b, "retained_compressed": r_c,
                         "retained_uncompressed": r_u, "corr": corr})
    return pd.DataFrame(rows)


def average_table(pw: pd.DataFrame, sources: pd.DataFrame, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """Mean correlation of each spectrum with every other, leaving out ``exclude`` entirely."""
    keep = [k for k in pw["target"].unique() if k not in exclude]
    rows = []
    for k in keep:
        others = [o for o in keep if o != k]
        sub = pw[(pw["target"] == k) & pw["predictor"].isin(others)]
        back = pw[(pw["predictor"] == k) & pw["target"].isin(others)]
        rows.append({"key": k, "mean_corr_to_others": sub["corr"].mean(), "min_corr_to_others": sub["corr"].min(),
                     "mean_retained_predicting_others": back["retained_compressed"].mean(), "n_others": len(others)})
    return pd.DataFrame(rows).merge(sources[["key", "label", "free_nt_parameters"]], on="key")
