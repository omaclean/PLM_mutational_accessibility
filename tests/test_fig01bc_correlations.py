"""The correlation table that says which scale a quoted coefficient was taken on.

``tools/export_fig01bc_correlations.py`` writes one coefficient per (model, lineage,
nucleotide-change group, statistic). Three things have to hold for the table to be worth
quoting, and each is checked here against something outside the exporter:

* its raw-scale Pearson rows are the numbers Figure 2A and 2B print, computed by
  ``figures/make_fig02ab.correlations`` -- if the two ever drift, the table and the panel
  would be attributing different numbers to the same replacements;
* every coefficient recomputes from the shipped ``replacement_scores`` table through
  SciPy, not through the exporter's own arithmetic;
* the machinery underneath -- the site blocks the bootstrap resamples, the refusal of a
  probability with no logarithm, the scale-invariance that separates the rank row from
  the other two -- behaves on inputs built here, where the answer is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

from mutacc.guide import GuideError, REPO_ROOT, load_guide

import figures.make_fig02ab as fig02ab
import tools.export_fig01bc_correlations as exporter

VALUES = REPO_ROOT / "output" / f"{exporter.STEM}_values.tsv"
GUIDE = exporter.DEFAULT_GUIDE


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    if not VALUES.exists():
        pytest.skip(f"{VALUES.name} has not been built")
    return pd.read_csv(VALUES, sep="\t")


@pytest.fixture(scope="module")
def guide():
    return load_guide(GUIDE)


@pytest.fixture(scope="module")
def scores(guide) -> pd.DataFrame:
    return guide.read_table("replacement_scores")


def _cloud(scores: pd.DataFrame, model: str, lineage: str) -> pd.DataFrame:
    rows = scores[(scores["model"] == model) & (scores["lineage"] == lineage)]
    assert not rows.empty, f"no rows for {model} / {lineage}"
    return rows


# ---------------------------------------------------------------------------
# Shape: every cell the table claims to cover is there exactly once
# ---------------------------------------------------------------------------

def test_the_table_has_the_declared_columns(table):
    assert list(table.columns) == exporter.COLS


def test_every_scale_says_which_variable_it_logged(table):
    """``scale`` is a name; ``plm_transform``/``mut_transform`` are the thing itself. A row
    where they disagree would describe a transform the number was not taken under."""
    for scale, (plm, mut) in exporter.SCALES.items():
        rows = table[table["scale"] == scale]
        assert not rows.empty, f"no rows on scale {scale!r}"
        assert set(rows["plm_transform"]) == {plm}, scale
        assert set(rows["mut_transform"]) == {mut}, scale
    assert set(table["scale"]) == {sc for scales in exporter.STATISTICS.values() for sc in scales}


def test_the_declared_scales_are_exactly_the_reported_ones():
    assert set(exporter.SCALES) == {sc for scales in exporter.STATISTICS.values() for sc in scales}


def test_both_semi_log_scales_are_reported(table):
    """The point of the table: a reader must be able to tell a raw-scale coefficient from
    one taken with only the model probability logged, or only the accessibility."""
    assert exporter.SCALES["log10_plm"] == (exporter.LOG10, exporter.IDENTITY)
    assert exporter.SCALES["log10_mut"] == (exporter.IDENTITY, exporter.LOG10)
    assert {"log10_plm", "log10_mut"} <= set(table["scale"])


def test_every_model_lineage_group_carries_every_statistic(table, guide):
    expected = {(s, sc) for s, scales in exporter.STATISTICS.items() for sc in scales}
    groups = {exporter.ALL_GROUP, *(str(g) for g in exporter.GROUPS)}
    cells = 0
    for (model, lineage, group), part in table.groupby(["model", "lineage", "group"]):
        assert set(zip(part["statistic"], part["scale"])) == expected, (model, lineage, group)
        assert len(part) == len(expected), f"duplicate rows for {model}/{lineage}/{group}"
        cells += 1
    assert cells == len({m.key for m in guide.models}) * len(guide.lineages) * len(groups)
    assert set(table["model"]) == {m.key for m in guide.models}
    assert set(table["lineage"]) == {lin.id for lin in guide.lineages}
    assert set(table["group"].astype(str)) == groups


def test_one_lineage_is_marked_primary(table, guide):
    marked = set(table[table["is_primary"]]["lineage"])
    assert marked == {guide.primary_lineage}


def test_the_group_rows_partition_the_whole_cloud(table):
    """Each replacement is in exactly one nucleotide-change group, so the group row
    counts must sum to the ``all`` row rather than merely being no larger than it."""
    for (model, lineage, statistic, scale), part in table.groupby(
            ["model", "lineage", "statistic", "scale"]):
        whole = part[part["group"] == exporter.ALL_GROUP]["n"]
        pieces = part[part["group"] != exporter.ALL_GROUP]["n"]
        assert int(pieces.sum()) == int(whole.iloc[0]), (model, lineage, statistic)


def test_r_squared_is_the_square_of_the_coefficient_on_its_own_row(table):
    assert np.allclose(table["r_squared"], table["value"] ** 2, atol=1e-6)


def test_every_coefficient_is_a_correlation(table):
    assert table["value"].between(-1.0, 1.0).all()
    assert (table["ci_lo"] <= table["value"]).all() and (table["value"] <= table["ci_hi"]).all()


# ---------------------------------------------------------------------------
# Values: the table against the data, and against the panel that prints them
# ---------------------------------------------------------------------------

def test_the_raw_rows_are_what_figure_2ab_prints(table, scores, guide):
    """The panel subheaders and this table must quote one number, not two.

    ``make_fig02ab.correlations`` is the function whose output is drawn under Figure 2A
    and 2B. Its Pearson correlation is taken on the probabilities themselves, which is
    the ``raw`` row here; its Spearman is the ``rank`` row.
    """
    lineage = guide.primary_lineage
    for model in guide.models:
        panel = fig02ab.correlations(_cloud(scores, model.key, lineage))
        rows = table[(table["model"] == model.key) & (table["lineage"] == lineage)
                     & (table["group"] == exporter.ALL_GROUP)].set_index(["statistic", "scale"])
        assert rows.loc[("pearson_r", "raw"), "value"] == pytest.approx(
            panel["pearson_r"], abs=5e-7)
        assert rows.loc[("spearman_r", "rank"), "value"] == pytest.approx(
            panel["spearman_r"], abs=5e-7)


@pytest.mark.parametrize("group", [exporter.ALL_GROUP, "1", "2", "3"])
def test_every_coefficient_recomputes_from_the_shipped_table(table, scores, guide, group):
    """Through SciPy, on rows selected here, for every model and every lineage."""
    for model in guide.models:
        for lineage in (lin.id for lin in guide.lineages):
            rows = _cloud(scores, model.key, lineage)
            if group != exporter.ALL_GROUP:
                rows = rows[rows["nt_steps"] == int(group)]
            plm = rows[exporter.PLM_COL].to_numpy(float)
            mut = rows[exporter.MUT_COL].to_numpy(float)
            want = {
                ("pearson_r", "raw"): pearsonr(plm, mut)[0],
                ("pearson_r", "log10_plm"): pearsonr(np.log10(plm), mut)[0],
                ("pearson_r", "log10_mut"): pearsonr(plm, np.log10(mut))[0],
                ("pearson_r", "log10_both"): pearsonr(np.log10(plm), np.log10(mut))[0],
                ("spearman_r", "rank"): spearmanr(plm, mut)[0],
            }
            got = table[(table["model"] == model.key) & (table["lineage"] == lineage)
                        & (table["group"].astype(str) == group)].set_index(["statistic", "scale"])
            assert len(got) == len(want)
            for key, value in want.items():
                assert got.loc[key, "value"] == pytest.approx(value, abs=5e-7), (
                    f"{model.key} / {lineage} / group {group} / {key}")
                assert got.loc[key, "n"] == len(rows)


def test_the_two_scales_disagree_on_the_primary_lineage(table, guide):
    """The table exists because they do: a raw-scale Pearson over probabilities spanning
    nine orders of magnitude is not the correlation the log axes of the panels show. If
    this ever stopped being true the table would be redundant, and that should be noticed
    deliberately rather than by the numbers quietly converging."""
    rows = table[(table["lineage"] == guide.primary_lineage)
                 & (table["group"] == exporter.ALL_GROUP)
                 & (table["statistic"] == "pearson_r")]
    wide = rows.set_index(["model_role", "scale"])["value"]
    for role in ("base", "finetune"):
        assert wide[(role, "log10_both")] > wide[(role, "raw")] + 0.2, role


def test_the_caption_claim_about_the_two_semi_log_scales_holds(table, guide):
    """The caption says the raw coefficient is held down mainly by the skew of the model's
    own probabilities: logging those alone recovers a large part of the distance to the
    joint log scale -- about half for the base model, more for the fine-tuned one --
    while logging the accessibility alone recovers little of it. That is a claim about
    these numbers, so it is checked against them rather than left in prose."""
    rows = table[(table["lineage"] == guide.primary_lineage)
                 & (table["group"] == exporter.ALL_GROUP)
                 & (table["statistic"] == "pearson_r")]
    wide = rows.set_index(["model_role", "scale"])["value"]
    for role in ("base", "finetune"):
        raw, plm_only = wide[(role, "raw")], wide[(role, "log10_plm")]
        mut_only, both = wide[(role, "log10_mut")], wide[(role, "log10_both")]
        assert plm_only > mut_only, role
        assert plm_only - raw > 0.40 * (both - raw), role
        assert mut_only - raw < 0.20 * (both - raw), role
        assert both >= plm_only, role


# ---------------------------------------------------------------------------
# The machinery, on inputs built here
# ---------------------------------------------------------------------------

def test_site_blocks_partition_the_rows_whatever_order_they_arrive_in():
    position = np.array([7, 3, 7, 3, 5, 7])
    blocks = exporter.site_blocks(position)
    assert sorted(np.concatenate(blocks).tolist()) == list(range(len(position)))
    assert {tuple(sorted(b.tolist())) for b in blocks} == {(1, 3), (4,), (0, 2, 5)}
    for block in blocks:
        assert len(set(position[block])) == 1


def test_the_bootstrap_resamples_sites_and_not_replacements():
    """A draw must contain whole sites. Built so that every replacement at a site carries
    that site's own value: a draw of site blocks can then only ever hold a multiple of 19
    rows per site it picked, which resampling rows individually would not."""
    rng = np.random.default_rng(0)
    position = np.repeat(np.arange(40), 19)
    plm = np.repeat(rng.uniform(0.01, 1.0, 40), 19)
    mut = np.repeat(rng.uniform(0.01, 1.0, 40), 19)
    blocks = exporter.site_blocks(position)
    assert len(blocks) == 40 and all(len(b) == 19 for b in blocks)
    lo, hi = exporter.bootstrap_intervals(plm, mut, position, n_boot=200, seed=1)[
        ("pearson_r", "raw")]
    assert lo < pearsonr(plm, mut)[0] < hi


def test_the_interval_narrows_as_the_draws_agree():
    """A site bootstrap over a cloud with no spread between sites must return an interval
    on the point estimate, not a wide one -- the check that the draws are of sites and the
    percentiles are taken over the right axis."""
    position = np.repeat(np.arange(30), 19)
    value = np.tile(np.linspace(0.1, 1.0, 19), 30)
    point = exporter.coefficients(value, value * 2.0)
    out = exporter.bootstrap_intervals(value, value * 2.0, position, n_boot=100, seed=3)
    for key, (lo, hi) in out.items():
        assert lo == pytest.approx(point[key], abs=1e-9)
        assert hi == pytest.approx(point[key], abs=1e-9)


def test_a_bootstrap_over_one_site_is_refused():
    position = np.zeros(19, dtype=int)
    value = np.linspace(0.1, 1.0, 19)
    with pytest.raises(GuideError, match="at least two sites"):
        exporter.bootstrap_intervals(value, value, position, n_boot=10)


@pytest.mark.parametrize("bad", [0.0, -1e-9, np.nan, np.inf])
def test_a_probability_with_no_logarithm_is_refused_not_dropped(bad):
    frame = pd.DataFrame({
        exporter.PLM_COL: [0.5, 0.25, bad],
        exporter.MUT_COL: [0.1, 0.2, 0.3],
        "position": [1, 2, 3],
    })
    with pytest.raises(GuideError, match="non-positive"):
        exporter.paired_columns(frame)


def test_the_rank_row_is_the_only_one_a_monotone_rescaling_leaves_alone():
    """Squaring a positive variable is monotone, so it cannot change a rank correlation;
    it does change both Pearson rows. That difference is the whole content of the
    ``scale`` column, so it is pinned rather than assumed."""
    rng = np.random.default_rng(11)
    plm = 10.0 ** rng.uniform(-9, 0, 500)
    mut = 10.0 ** (np.log10(plm) * 0.5 + rng.normal(0, 1.0, 500))
    before = exporter.coefficients(plm, mut)
    after = exporter.coefficients(plm ** 2, mut)
    # Squaring is monotone, so ranks cannot move; it is linear in log10, so a scale that
    # logs the model probability cannot move either. The two that read it untransformed do.
    assert after[("spearman_r", "rank")] == pytest.approx(before[("spearman_r", "rank")])
    assert after[("pearson_r", "log10_plm")] == pytest.approx(before[("pearson_r", "log10_plm")])
    assert after[("pearson_r", "log10_both")] == pytest.approx(before[("pearson_r", "log10_both")])
    assert after[("pearson_r", "raw")] != pytest.approx(before[("pearson_r", "raw")])
    assert after[("pearson_r", "log10_mut")] != pytest.approx(before[("pearson_r", "log10_mut")])


def test_an_unknown_transform_is_refused():
    with pytest.raises(GuideError, match="unknown transform"):
        exporter.transformed(np.array([1.0, 2.0]), "sqrt")


def test_an_unknown_nucleotide_change_group_is_refused(guide):
    """``nt_steps`` outside 1-3 would be silently excluded from every group row while
    still counting in the ``all`` row, which the partition test would then fail far from
    its cause."""
    frame = pd.DataFrame({
        exporter.PLM_COL: [0.5, 0.25],
        exporter.MUT_COL: [0.1, 0.2],
        "position": [1, 2],
        "nt_steps": [1, 4],
    })
    with pytest.raises(GuideError, match="nt_steps"):
        exporter.rows_for(frame, model=guide.models[0], lineage="x", is_primary=False,
                          n_boot=10, seed=0)


# ---------------------------------------------------------------------------
# The caption describes the file beside it
# ---------------------------------------------------------------------------

def test_the_caption_quotes_the_table_it_sits_beside(table, guide):
    caption = (REPO_ROOT / "output" / f"{exporter.STEM}_caption.md")
    if not caption.exists():
        pytest.skip(f"{caption.name} has not been built")
    text = caption.read_text()
    rows = table[(table["lineage"] == guide.primary_lineage)
                 & (table["group"] == exporter.ALL_GROUP)]
    for _, row in rows.iterrows():
        assert f"{row['value']:.3f}" in text, (
            f"the caption does not quote the {row['model_role']} {row['scale']} "
            f"{row['statistic']} it describes")
    assert guide.primary_lineage in text
