"""Invariants over every shipped ``output/*_values.tsv``.

These are the numbers that go into the manuscript. A figure can be redrawn, but a
wrong number in a values file is quoted in the text and in the captions, so these
tests check the files directly rather than the code that wrote them.

Everything here is generic over the whole set unless a file is named, so a new
figure picks the checks up for free. The rules are deliberately shallow and
cheap — ordering, range, and derived columns agreeing with the columns they are
derived from — because the expensive end-to-end regeneration lives elsewhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mutacc.guide import REPO_ROOT

OUTPUT = REPO_ROOT / "output"
VALUE_FILES = sorted(OUTPUT.glob("*_values.tsv"))

#: Absolute slack for a float that was written out rounded.
ATOL = 1e-9

#: (point estimate, lower, upper) column names used across the shipped files.
INTERVAL_TRIPLES = [
    ("estimate", "lo", "hi"),
    ("estimate", "ci_lo", "ci_hi"),
    ("mean", "lo", "hi"),
    ("value", "lo", "hi"),
    ("rho", "ci_lo", "ci_hi"),
    ("auc", "ci_lo", "ci_hi"),
    ("delta", "delta_lo", "delta_hi"),
    ("difference", "ci_lo", "ci_hi"),
    ("mean_delta", "ci_lo", "ci_hi"),
    ("peak_alpha", "peak_alpha_ci_lo", "peak_alpha_ci_hi"),
    ("share_multi", "lo", "hi"),
    ("baseline_mean", "baseline_lo", "baseline_hi"),
]

#: Columns that are correlations or signed contrasts: [-1, 1].
CORRELATION_COLUMNS = {
    "rho", "corr", "rho_best", "rho_alpha0", "rho_peak", "rho_alpha",
    "auroc", "auroc_base", "auroc_ft", "auroc_finetune",
}

#: Columns that are proportions or rates on [0, 1]. ``weight`` (s09) and
#: ``scaled_weight`` (sM) are deliberately absent: they are unnormalised
#: weights, not proportions, and run to 11.5 and to -1.0 respectively.
UNIT_COLUMNS = {
    "prevalence", "pr_auc", "mean_p_wt", "top1", "composition_baseline",
    "p_perm", "p_sign",
}

#: Columns that count things: non-negative integers. ``n_rows`` is excluded
#: because in the stratified files it is a mean row count per lineage and is
#: legitimately fractional (see test_stratum_row_counts_sum_to_the_all_row).
FRACTIONAL_COUNTS = {"n_rows"}


def _ids(path):
    return path.name


def _read(path):
    return pd.read_csv(path, sep="\t")


def test_there_are_values_files_to_check():
    """Guard against the glob silently matching nothing and every test vacuously passing."""
    assert len(VALUE_FILES) >= 25, [p.name for p in VALUE_FILES]


@pytest.mark.parametrize("path", VALUE_FILES, ids=_ids)
def test_values_file_is_non_empty_with_unique_columns(path):
    frame = _read(path)
    assert not frame.empty, f"{path.name} has a header and no rows"
    assert len(set(frame.columns)) == len(frame.columns), f"{path.name} has duplicate column names"
    assert not any(c.strip() != c for c in frame.columns), f"{path.name} has padded column names"


@pytest.mark.parametrize("path", VALUE_FILES, ids=_ids)
def test_confidence_intervals_bracket_their_point_estimate(path):
    """``lo <= estimate <= hi`` wherever a file carries an interval.

    This is the check that catches an interval joined onto the wrong row, a
    lower and upper bound written out swapped, or a point estimate recomputed
    without its interval.
    """
    frame = _read(path)
    columns = set(frame.columns)
    checked = 0
    for point, lo, hi in INTERVAL_TRIPLES:
        if not {point, lo, hi} <= columns:
            continue
        sub = frame[[point, lo, hi]].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.empty:
            continue
        checked += len(sub)
        below = sub[sub[lo] > sub[point] + ATOL]
        above = sub[sub[point] > sub[hi] + ATOL]
        assert below.empty, f"{path.name}: {len(below)} rows with {lo} > {point}\n{below.head()}"
        assert above.empty, f"{path.name}: {len(above)} rows with {point} > {hi}\n{above.head()}"
    if not checked:
        # Either the file carries no intervals, or its interval columns are
        # empty; the emptiness is the business of test_no_all_nan_numeric_columns.
        pytest.skip(f"{path.name} has no populated interval to check")


@pytest.mark.parametrize("path", VALUE_FILES, ids=_ids)
def test_correlations_and_aurocs_are_in_range(path):
    frame = _read(path)
    for column in sorted(CORRELATION_COLUMNS & set(frame.columns)):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        bad = values[(values < -1 - ATOL) | (values > 1 + ATOL)]
        assert bad.empty, f"{path.name}: {column} outside [-1, 1]: {bad.head().tolist()}"


@pytest.mark.parametrize("path", VALUE_FILES, ids=_ids)
def test_proportions_and_p_values_are_in_the_unit_interval(path):
    frame = _read(path)
    columns = set(frame.columns)
    unit = UNIT_COLUMNS & columns
    unit |= {c for c in columns if c.startswith(("p_", "q_")) and not c.startswith("panel")}
    for column in sorted(unit):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        bad = values[(values < -ATOL) | (values > 1 + ATOL)]
        assert bad.empty, f"{path.name}: {column} outside [0, 1]: {bad.head().tolist()}"


@pytest.mark.parametrize("path", VALUE_FILES, ids=_ids)
def test_count_columns_are_non_negative_integers(path):
    frame = _read(path)
    counts = [
        c for c in frame.columns
        if (c == "n" or c.startswith("n_")) and c not in FRACTIONAL_COUNTS
    ]
    for column in counts:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        assert (values >= 0).all(), f"{path.name}: {column} has negative counts"
        assert np.allclose(values, values.round()), (
            f"{path.name}: {column} is a count but is fractional: "
            f"{values[~np.isclose(values, values.round())].head().tolist()}"
        )


#: Columns that ship empty on purpose, pinned exactly so that populating one
#: forces this list to be updated instead of the change going unnoticed.
#:
#: The entries here are NOT padding and are open questions, raised
#: 15 Oct 2026:
#:   * the simulation-control files ship ``lo``/``hi`` for all 616 rows and fill
#:     neither, so anything reading an interval off S16 silently gets NaN;
#:   * ``fig05``'s lineage-level interval is empty on the headline codon figure,
#:     which is the figure the pseudo-replication question bears on.
#: Either populate them or drop the columns; do not leave them half-present.
EMPTY_BY_DESIGN = {
    "s15_simulation_control_values.tsv": {"lo", "hi"},
    "sN_simulation_control_values.tsv": {"lo", "hi"},
    "fig05_codon_learning_values.tsv": {"lineage_ci_lo", "lineage_ci_hi"},
}


@pytest.mark.parametrize("path", VALUE_FILES, ids=_ids)
def test_no_unexpected_all_nan_columns(path):
    """An all-NaN column has already shipped once in this project and passed every
    comparison silently (docs/notes/PLAN.md, 9 Oct). A column that is entirely missing is a
    broken join, not a result, so every empty column must be on the record."""
    frame = _read(path)
    empty = {c for c in frame.columns if frame[c].isna().all()}
    assert empty == EMPTY_BY_DESIGN.get(path.name, set()), (
        f"{path.name}: empty columns {sorted(empty)} but "
        f"{sorted(EMPTY_BY_DESIGN.get(path.name, set()))} were expected"
    )


# ---------------------------------------------------------------------------
# Derived columns must agree with the columns they are derived from
# ---------------------------------------------------------------------------

def test_sH_contact_recovery_base_rate_is_the_contact_percentage():
    """``base_rate`` is a percentage of pairs that are contacts, not a proportion.

    It reads like a proportion and is above 1 for most categories, so this pins
    which it is and keeps the two from being mixed up in a caption.
    """
    frame = _read(OUTPUT / "sH_contact_recovery_values.tsv")
    expected = 100.0 * frame["n_contacts"] / frame["n_pairs"]
    assert np.allclose(frame["base_rate"], expected, atol=5e-4), (
        frame.loc[~np.isclose(frame["base_rate"], expected, atol=5e-4),
                  ["structure", "category", "n_contacts", "n_pairs", "base_rate"]]
    )
    assert (frame["n_contacts"] <= frame["n_pairs"]).all()


@pytest.mark.parametrize(
    "filename",
    ["sH_contact_recovery_values.tsv", "sH_contact_recovery_antigenic_values.tsv"],
)
def test_crosses_zero_flags_agree_with_their_intervals(filename):
    """A boolean that restates an interval must not drift from it."""
    frame = _read(OUTPUT / filename)
    pairs = [
        ("crosses_zero", "delta_lo", "delta_hi"),
        ("crosses_zero", "ci_lo", "ci_hi"),
        ("crosses_zero_resid", "delta_lo_resid", "delta_hi_resid"),
    ]
    checked = False
    for flag, lo, hi in pairs:
        if not {flag, lo, hi} <= set(frame.columns):
            continue
        sub = frame[[flag, lo, hi]].dropna()
        if sub.empty:
            continue
        checked = True
        derived = (sub[lo] <= 0) & (0 <= sub[hi])
        stated = sub[flag].astype(bool)
        mismatch = sub[derived != stated]
        assert mismatch.empty, f"{filename}: {flag} disagrees with [{lo}, {hi}]\n{mismatch}"
    assert checked, f"{filename}: no crosses_zero column was checked"


@pytest.mark.parametrize(
    "filename,keys",
    [
        ("sA_spike_model_vs_accessibility_values.tsv", ["measure", "series", "control"]),
        ("sD_partial_correlation_values.tsv", ["measure", "series", "control"]),
        ("fig02d_strata_panel_values.tsv", ["measure", "series", "control"]),
    ],
)
def test_stratum_row_counts_sum_to_the_all_row(filename, keys):
    """The nucleotide-step strata partition the replacements, so their row counts
    must add up to the ``all`` row's. A stratum selected with an overlapping or
    leaky rule shows up here and nowhere else."""
    frame = _read(OUTPUT / filename)
    keys = [k for k in keys if k in frame.columns]
    seen = 0
    for key, group in frame.groupby(keys, dropna=False):
        is_all = group["stratum"].astype(str) == "all"
        total, parts = group.loc[is_all, "n_rows"], group.loc[~is_all, "n_rows"]
        if total.empty or parts.empty:
            continue
        seen += 1
        assert np.isclose(total.iloc[0], parts.sum(), rtol=1e-6), (
            f"{filename} {key}: all={total.iloc[0]} but strata sum to {parts.sum()}"
        )
    assert seen, f"{filename}: no (all, strata) group was found to check"


@pytest.mark.parametrize(
    "filename",
    ["sA_spike_model_vs_accessibility_values.tsv", "fig02d_strata_panel_values.tsv"],
)
def test_per_lineage_range_contains_the_pooled_estimate(filename):
    """``lineage_min`` and ``lineage_max`` are the extremes over lineages of the
    quantity ``estimate`` summarises, so the estimate has to lie between them."""
    frame = _read(OUTPUT / filename)
    sub = frame[["lineage_min", "lineage_max", "estimate"]].dropna()
    assert not sub.empty
    assert (sub["lineage_min"] <= sub["lineage_max"] + ATOL).all(), "lineage_min above lineage_max"
    outside = sub[
        (sub["lineage_min"] > sub["estimate"] + ATOL)
        | (sub["estimate"] > sub["lineage_max"] + ATOL)
    ]
    assert outside.empty, f"{filename}: estimate outside the per-lineage range\n{outside}"


def test_alpha_sweeps_share_one_grid_and_bracket_zero():
    """Fig 3 and S2 are the same sweep drawn two ways; every model must be scored
    on the same alpha grid or the curves are not comparable."""
    for filename in ("fig03_alpha_sweep_values.tsv", "sB_ranking_metrics_values.tsv"):
        frame = _read(OUTPUT / filename)
        sweep = frame.dropna(subset=["alpha"])
        grids = {
            key: tuple(sorted(group["alpha"].round(6)))
            for key, group in sweep.groupby(["metric", "model"])
        }
        assert len(set(grids.values())) == 1, (
            f"{filename}: models are scored on different alpha grids: "
            f"{ {k: len(v) for k, v in grids.items()} }"
        )
        alphas = sweep["alpha"]
        assert alphas.min() < 0 < alphas.max(), f"{filename}: grid does not bracket alpha = 0"
