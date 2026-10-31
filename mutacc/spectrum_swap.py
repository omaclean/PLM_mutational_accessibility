"""Correlation of a score with mutational accessibility when the mutation spectrum is swapped.

The question behind S6: the correlation between a model's probability and mutational
accessibility is computed with *one* spectrum, the influenza one. If the model had learned that
spectrum, other spectra should fit worse. This module recomputes the correlation under any
number of 4x4 spectra, on the **same protein, the same reference codons and the same strata**,
so only the accessibility column changes. Nothing here is a multi-protein comparison: the
protein is always the one in the table. Spectra come from other organisms, not other
proteins.

It holds the real codons fixed because nucleotide-step strata depend on the codon: swapping
the codons would move replacements between strata and the comparison would no longer be
like for like. The exploration code (``a35``) also ran conditions that redraw synonymous
codons from codon-usage tables typed into the script; those are not ported (no provenance
for the tables).

Conventions are those of :mod:`mutacc.strata` and the exploration code: Spearman within lineage
and stratum, mean over lineages, a percentile bootstrap that resamples **sites** with the same
draw for every condition, score and stratum. The difference between a condition and the
reference therefore gets its own interval from the *differenced* replicates, which keeps the
pairing (two correlations measured on the same replacements move together, so their
difference is far better determined than two overlapping intervals suggest).

Every spectrum is rescaled to the same mean row sum before accessibility is computed, as the
exploration code did, because pooled across strata the correlation depends on the scale
(within a stratum it does not).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from mutacc import accessibility as acc
from mutacc import strata as S

__all__ = ["Condition", "row_codons", "condition_accessibility", "merge_rounding_ties", "swap_table",
           "mean_row_sum", "TIE_DIGITS"]

#: Significant digits kept when deciding which accessibilities are equal. Accessibilities that are
#: equal in exact arithmetic differ by floating-point noise of about 1e-16 relative, which a rank
#: correlation would read as an ordering. Genuinely different values differ by 1e-4 or more at
#: the spectra's scale, so twelve digits separates the two.
TIE_DIGITS = 12


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    q: np.ndarray            # 4x4 rates, coding sense, order ACGT
    reference: bool = False


def mean_row_sum(q: np.ndarray) -> float:
    q = np.array(q, dtype=float)
    np.fill_diagonal(q, 0.0)
    return float(q.sum(axis=1).mean())


def row_codons(frame: pd.DataFrame, roots: Mapping[str, str | Path]) -> list[str]:
    """Reference codon of every row from its lineage root; raises if it does not translate to ``ref_aa``."""
    out = pd.Series(index=frame.index, dtype=object)
    for lin, sub in frame.groupby("lineage", sort=False):
        key = str(lin)
        if key not in roots:
            raise KeyError(f"no root sequence for lineage {key!r}")
        pos = np.sort(sub["position"].unique())
        lookup = dict(zip(pos, acc.reference_codons(roots[key], pos)))
        out.loc[sub.index] = sub["position"].map(lookup)
    codons = out.tolist()
    bad = sum(acc.GENETIC_CODE[c] != a for c, a in zip(codons, frame["ref_aa"]))
    if bad:
        raise ValueError(f"{bad} rows: the root codon does not translate to the table's reference amino acid")
    return codons


def condition_accessibility(codons: Sequence[str], aas: Sequence[str], q: np.ndarray, row_sum: float) -> np.ndarray:
    """Accessibility of every (codon, target amino acid) under spectrum ``q`` scaled to ``row_sum``."""
    if not row_sum > 0:
        raise ValueError("row_sum must be positive")
    return acc.accessibility(list(codons), list(aas), q, row_sum=row_sum)


def merge_rounding_ties(values: np.ndarray, digits: int = TIE_DIGITS) -> np.ndarray:
    """Round to ``digits`` significant digits so values equal up to floating-point noise become ties.

    Matters most for flat spectra, where most accessibilities are exactly tied: without this the
    JC69 correlation within one nucleotide step moves by about 0.015 depending on rounding noise.
    """
    v = np.asarray(values, float)
    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("accessibilities must be finite and non-negative")
    u, inv = np.unique(v, return_inverse=True)
    rounded = np.array([float(f"{x:.{digits - 1}e}") for x in u])
    return rounded[inv]


def swap_table(
    frame: pd.DataFrame,
    model_keys: list[str],
    properties: list[str],
    roots: Mapping[str, str | Path],
    conditions: Sequence[Condition],
    *,
    n_boot: int,
    seed: int,
    row_sum: float | None = None,
    check_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Correlation with accessibility for every condition, stratum and score, with paired differences.

    One row per (condition, series, stratum). ``estimate`` is the mean over lineages of the
    within-lineage Spearman correlation; ``lo, hi`` its site-bootstrap interval. ``delta`` is the
    reference condition's estimate minus this condition's, with an interval from the differenced
    bootstrap replicates (zero for the reference itself). ``kind`` is ``model`` or ``property``.

    Exactly one condition must be the reference, and its accessibility must reproduce the
    table's own ``mut_prob`` to ``check_tolerance`` (relative): otherwise the codons, roots or
    scale are wrong and nothing below can be trusted.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    refs = [c for c in conditions if c.reference]
    if len(refs) != 1:
        raise ValueError("exactly one condition must be the reference")
    if len({c.key for c in conditions}) != len(conditions):
        raise ValueError("duplicate condition keys")
    ref = refs[0]
    scale = mean_row_sum(ref.q) if row_sum is None else row_sum

    wide = S.wide_scores(frame, model_keys, properties)
    codons = row_codons(wide, roots)
    columns = [f"plm::{k}" for k in model_keys] + ["mut_prob", *properties]
    series = [(k, "model") for k in model_keys] + [(p, "property") for p in properties]
    n_series = len(series)
    strata = [S.ALL, *sorted(int(s) for s in np.unique(wide["nt_steps"]))]

    per_cond = {}
    for c in conditions:
        a = condition_accessibility(codons, wide["aa"], c.q, scale)
        if c.reference and not np.allclose(a, wide["mut_prob"].to_numpy(float), rtol=check_tolerance, atol=0):
            raise ValueError(f"the reference condition {c.key!r} does not reproduce the table's mut_prob; "
                             "check the spectrum, the roots and the scale")
        per_cond[c.key] = S._blocks(wide.assign(mut_prob=merge_rounding_ties(a)), columns)

    first = per_cond[ref.key]
    points, stacks = {}, {}
    for key, blocks in per_cond.items():
        point, stack = S._strata_vectors(blocks, strata, model_keys, properties)
        if not np.isfinite(point[:, :n_series]).all():
            raise ValueError(f"condition {key!r}: a correlation could not be computed (too few rows or a constant column)")
        points[key], stacks[key] = point[:, :n_series], stack[:, :, :n_series]

    universe = np.unique(np.concatenate([b.pos for b in first]))
    rng = np.random.default_rng(seed)
    draws = {k: np.empty((n_boot, len(strata), n_series)) for k in per_cond}
    for r in range(n_boot):
        picked = rng.integers(0, len(universe), len(universe))
        idx = S._resample_indices(first, universe, picked)
        for key, blocks in per_cond.items():
            vec, _ = S._strata_vectors(blocks, strata, model_keys, properties, idx)
            draws[key][r] = vec[:, :n_series]

    def band(x):
        return np.nanpercentile(x, [2.5, 97.5], axis=0)

    n_rows = {s: float(np.mean([len(b.steps) if s == S.ALL else (b.steps == s).sum() for b in first])) for s in strata}
    rows = []
    for c in conditions:
        lo, hi = band(draws[c.key])
        d = points[ref.key] - points[c.key]
        dlo, dhi = band(draws[ref.key] - draws[c.key])
        for si, s in enumerate(strata):
            for qi, (name, kind) in enumerate(series):
                rows.append({
                    "condition": c.key, "label": c.label, "reference": c.reference,
                    "series": name, "kind": kind, "stratum": str(s),
                    "estimate": points[c.key][si, qi], "lo": lo[si, qi], "hi": hi[si, qi],
                    "delta": d[si, qi], "delta_lo": dlo[si, qi], "delta_hi": dhi[si, qi],
                    "n_rows": n_rows[s], "n_boot": n_boot,
                })
    out = pd.DataFrame(rows)
    if not np.isfinite(out[["estimate", "lo", "hi", "delta", "delta_lo", "delta_hi"]].to_numpy()).all():
        raise ValueError("a bootstrap interval is undefined for some cell")
    return out
