"""The SM strata summary table against the values files it is built from.

``dev/dms_strata_analysis/make_strata_summary.py`` broke silently on 21 Oct 2026: the
``stratum`` and ``group`` labels in both values files changed from integers to the strings
``"pooled"``, ``"1"``, ``"2"``, ``"3"`` when the pooled group was added. The script compared
them to ``1`` and looked up ``"all"``, so every lookup missed -- raising partway through in
one place and, where a guard turned the miss into ``np.nan``, writing an empty column in the
other. These tests pin the label contract and check the shipped table row by row, so neither
failure mode can return quietly.
"""

import pandas as pd
import pytest

from mutacc.guide import REPO_ROOT

OUTPUT = REPO_ROOT / "output"
SUMMARY = OUTPUT / "sM_strata_summary_table.tsv"
WITHIN = OUTPUT / "sM_within_stratum_values.tsv"
ALPHA = OUTPUT / "sM_peak_alpha_by_stratum_values.tsv"

#: The pooled row carries the column suffix ``_all`` for historical reasons: the group was
#: called "all" when these columns were named, and only the values file was relabelled.
GROUPS = {"pooled": "all", "1": "1nt", "2": "2nt", "3": "3nt"}


@pytest.fixture(scope="module")
def tables():
    return (
        pd.read_csv(SUMMARY, sep="\t"),
        pd.read_csv(WITHIN, sep="\t", dtype={"stratum": str}),
        pd.read_csv(ALPHA, sep="\t", dtype={"group": str}),
    )


def test_the_stratum_labels_are_the_strings_the_script_looks_up(tables):
    """The contract that broke. If these change again, every lookup below goes empty."""
    _, within, alpha = tables
    assert set(within["stratum"]) == set(GROUPS)
    assert set(alpha["group"]) == set(GROUPS)


def test_no_column_is_empty(tables):
    """A missed lookup used to land here as a NaN column rather than an error."""
    summary = tables[0]
    carried = [c for c in summary.columns if c.startswith(("plm_", "ft_", "base_", "acc_"))]
    assert carried
    empty = {c: int(summary[c].isna().sum()) for c in carried if summary[c].isna().any()}
    assert not empty, f"NaN means a lookup matched nothing: {empty}"


def test_rho_columns_match_the_within_stratum_values(tables):
    summary, within, _ = tables
    for row in summary.itertuples():
        sub = within[(within.dataset == row.dataset) & (within.trait == row.trait)]
        for role, prefix in (("finetune", "plm_ft_rho_"), ("base", "plm_base_rho_")):
            for group, suffix in GROUPS.items():
                if suffix == "all":
                    continue  # the table carries no pooled rho column
                hit = sub[(sub.role == role) & (sub.stratum == group)]
                assert len(hit) == 1, (row.panel, role, group)
                assert getattr(row, f"{prefix}{suffix}") == pytest.approx(
                    float(hit["rho"].iloc[0])
                ), (row.panel, role, group)


def test_peak_alpha_columns_match_the_alpha_values(tables):
    summary, _, alpha = tables
    for row in summary.itertuples():
        sub = alpha[(alpha.dataset == row.dataset) & (alpha.trait == row.trait)]
        for role, prefix in (("finetune", "ft_peak_alpha_"), ("base", "base_peak_alpha_")):
            for group, suffix in GROUPS.items():
                hit = sub[(sub.role == role) & (sub.group == group)]
                assert len(hit) == 1, (row.panel, role, group)
                assert getattr(row, f"{prefix}{suffix}") == pytest.approx(
                    float(hit["peak_alpha"].iloc[0])
                ), (row.panel, role, group)


def test_the_pooled_column_holds_the_pooled_row_not_a_stratum(tables):
    """``_all`` and ``"pooled"`` are the same row under two names; a lookup that fell back
    to a stratum would be invisible in every other check here."""
    summary, _, alpha = tables
    for row in summary.itertuples():
        sub = alpha[(alpha.dataset == row.dataset) & (alpha.trait == row.trait)
                    & (alpha.role == "finetune")]
        pooled = float(sub.loc[sub.group == "pooled", "peak_alpha"].iloc[0])
        assert row.ft_peak_alpha_all == pytest.approx(pooled)
        # the pooled fit sees every replacement, so it is the widest group
        assert float(sub.loc[sub.group == "pooled", "n"].iloc[0]) == pytest.approx(
            float(sub.loc[sub.group.isin({"1", "2", "3"}), "n"].sum())
        )


def test_the_summary_is_not_older_than_the_values_it_is_built_from():
    newest_input = max(WITHIN.stat().st_mtime, ALPHA.stat().st_mtime)
    assert SUMMARY.stat().st_mtime >= newest_input, (
        f"{SUMMARY.name} predates its inputs; re-run "
        "dev/dms_strata_analysis/make_strata_summary.py"
    )


def test_a_relabelled_values_file_raises_rather_than_writing_nan(tables):
    """The old guard was ``if "all" in index else np.nan``. Guard the replacement: a group
    that is not there must raise."""
    source = (REPO_ROOT / "dev" / "dms_strata_analysis" / "make_strata_summary.py").read_text()
    assert "else np.nan" not in source
    assert "raise KeyError" in source
