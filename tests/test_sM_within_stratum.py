"""S12: within-stratum DMS correlations, recomputed from scores, plus a paired difference."""

import numpy as np
import pandas as pd
import pytest

import make_sM_within_stratum as sM
from mutacc.guide import REPO_ROOT, load_guide

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")


@pytest.fixture(scope="module")
def data():
    scores = GUIDE.read_table("dms_scores")
    table = GUIDE.read_table("dms_within_stratum")
    summary = GUIDE.read_table("dms_peak_summary")
    panel = sM.panels(summary)
    return scores, table, summary, panel, sM.role_of_model(summary)


def test_shipped_correlations_are_exactly_what_the_scores_give(data):
    scores, table, _, panel, _ = data
    assert sM.check_against_table(sM.recompute(scores, panel), table) < 1e-9


def test_a_tampered_table_is_caught(data):
    scores, table, _, panel, _ = data
    bad = table.copy()
    bad.loc[0, "rho"] += 0.01
    with pytest.raises(ValueError, match="disagrees"):
        sM.check_against_table(sM.recompute(scores, panel), bad)


def test_a_missing_cell_is_caught(data):
    scores, table, _, panel, _ = data
    with pytest.raises(ValueError, match="differ"):
        sM.check_against_table(sM.recompute(scores, panel), table.iloc[1:])


def test_binding_assays_are_flagged_by_what_they_measure_not_by_the_optimum_flag(data):
    """The summary's own `interpretable` column is about the optimum, not the assay.
    Using it once labelled a sera-escape panel as receptor binding."""
    panel = data[3]
    flagged = panel.loc[panel["binding"], "trait"].tolist()
    assert flagged and all("binding" in t.lower() for t in flagged)
    assert not panel.loc[~panel["binding"], "trait"].str.lower().str.contains("binding").any()
    assert panel["trait"].str.contains("sera escape").any()
    assert not panel.loc[panel["trait"].str.contains("sera escape"), "binding"].any()


def test_every_panel_has_one_base_and_one_finetuned_model(data):
    _, table, _, panel, roles = data
    for p in panel.itertuples():
        got = {roles[(p.dataset, m)] for m in table.loc[(table.dataset == p.dataset) & (table.trait == p.trait), "model_name"]}
        assert got == {"base", "finetune"}, p


@pytest.fixture(scope="module")
def diff(data):
    scores, _, _, panel, roles = data
    return sM.paired_difference(scores, panel, roles, n_boot=60, seed=1)


def test_paired_difference_covers_every_cell(diff, data):
    panel = data[3]
    assert len(diff) == len(panel) * len(sM.GROUPS)
    assert diff[["ci_lo", "difference", "ci_hi"]].notna().all().all()
    assert ((diff["ci_lo"] <= diff["ci_hi"])).all()


def test_paired_difference_is_the_difference_of_the_two_published_correlations(diff, data):
    _, table, _, panel, roles = data
    t = table.copy()
    t["role"] = [roles[(d, m)] for d, m in zip(t.dataset, t.model_name)]
    wide = t.pivot_table(index=["dataset", "trait", "stratum"], columns="role", values="rho")
    expected = (wide["finetune"] - wide["base"]).rename("expected").reset_index()
    got = diff.merge(expected, on=["dataset", "trait", "stratum"])
    assert len(got) == len(panel) * len(sM.STRATA)
    assert np.allclose(got["difference"], got["expected"], atol=1e-12)


def test_paired_bootstrap_is_seeded(data):
    scores, _, _, panel, roles = data
    one = panel.iloc[:1]
    a = sM.paired_difference(scores, one, roles, n_boot=15, seed=3)
    b = sM.paired_difference(scores, one, roles, n_boot=15, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_models_scoring_different_replacements_are_refused(data):
    scores, _, _, panel, roles = data
    one = panel.iloc[:1]
    broken = scores.drop(scores[(scores.dataset == one.iloc[0].dataset)
                                & (scores.model_name == "HA-80 fine-tuned")].index[:5])
    with pytest.raises(ValueError, match="do not score the same replacements"):
        sM.paired_difference(broken, one, roles, n_boot=5)


def test_a_dataset_with_one_model_is_refused(data):
    scores, _, _, panel, roles = data
    one = panel.iloc[:1]
    broken = scores[~((scores.dataset == one.iloc[0].dataset) & (scores.model_name == "HA-80 fine-tuned"))]
    with pytest.raises(ValueError, match="one base and one fine-tuned"):
        sM.paired_difference(broken, one, roles, n_boot=5)


def test_caption_counts_come_from_the_data_and_exclude_binding(data, diff):
    _, table, _, panel, roles = data
    # table with all groups
    table_all = sM.recompute_with_band(data[0], panel, n_boot=2)
    c = sM.summarise(diff, table_all, panel, roles)
    assert c["cells"] == len(diff.loc[~diff["binding"]])
    assert c["bars"] == {"base": c["cells"], "finetune": c["cells"]}
    text = sM.caption(panel, table_all, diff, roles, 60)
    assert f"{c['excluding_zero']}," in text and "not a model failure" in " ".join(text.split())


# ---------------------------------------------------------------------------
# The alpha refit along the bottom
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alpha(data):
    """Point estimates only; the bootstrap interval is covered separately and is slow."""
    scores, _, _, panel, roles = data
    return sM.alpha_by_stratum(scores, panel, roles, n_boot=0)


def test_pooled_refit_reproduces_the_shipped_peak_alpha(alpha, data):
    assert sM.check_pooled_alpha(alpha, data[2]) <= sM.ALPHA_STEP


def test_a_shifted_shipped_peak_alpha_is_caught(alpha, data):
    bad = data[2].copy()
    bad.loc[bad["spectrum"] == "published", "peak_alpha"] += 0.5
    with pytest.raises(ValueError, match="differs from the shipped peak_alpha"):
        sM.check_pooled_alpha(alpha, bad)


def test_every_arm_is_fitted_in_every_group(alpha, data):
    panel = data[3]
    assert len(alpha) == len(panel) * 2 * len(sM.GROUPS)
    assert set(alpha["group"]) == {str(g) for g in sM.GROUPS}


def test_binding_arms_are_kept_in_the_table_but_not_bootstrapped(alpha):
    binding = alpha.loc[alpha["binding"]]
    assert len(binding) and binding["peak_alpha_ci_lo"].isna().all()
    assert (binding["n_boot"] == 0).all()


def test_the_violin_drops_binding_and_nothing_else(alpha):
    """Receptor binding is the only exclusion. Until 22 Oct 2026 this panel also applied
    ``curve_flags``' ``interpretable``, which dropped sweep-end optima and curves judged weak
    at alpha = 0 -- and that made the denominator differ between strata for a reason no reader
    of the count could see, while removing the +1-pinned fits that argue *against* the result."""
    arms = sM.violin_arms(alpha)
    assert not arms["binding"].any()
    assert len(arms) == int((~alpha["binding"]).sum()), "something other than binding was dropped"
    assert 0 < len(arms) < len(alpha)


def test_every_stratum_counts_the_same_arms(alpha):
    """The uniform denominator is the point of the change: an arm counted in one stratum is
    counted in all of them, so the per-group counts can be read against each other."""
    arms = sM.violin_arms(alpha)
    per_group = arms.groupby("group").size()
    assert per_group.nunique() == 1, f"denominators differ between groups: {per_group.to_dict()}"


def test_the_kept_fits_include_ones_the_old_rule_dropped(alpha):
    """A guard against the filter creeping back: at least one arm that is counted now would
    have been thrown away by ``interpretable``, and the table still records why."""
    arms = sM.violin_arms(alpha)
    assert (~arms["interpretable"]).any(), "no arm survives that the old rule dropped"
    assert {"readable", "has_signal", "max_abs_rho"} <= set(alpha.columns)


def test_the_shipped_table_keeps_its_sweep_end_optima():
    """The caption counts sweep-end optima among the arms it tallies, so at least one must
    survive ``violin_arms`` in the arm that ships.

    This reads the written table rather than the ``alpha`` fixture because it is the one
    assertion here that depends on the scoring arm. The figure is drawn from the *masked*
    marginal; the fixture recomputes from the unmasked scores, where no counted arm reaches
    the end of the range. Asserting it on the fixture tested the wrong arm and passed only
    by accident of which scores were loaded.
    """
    shipped = pd.read_csv(REPO_ROOT / "output" / "sM_peak_alpha_by_stratum_values.tsv",
                          sep="\t")
    arms = sM.violin_arms(shipped)
    assert arms["boundary"].any(), "no sweep-end optimum is kept among the counted arms"
    assert (arms["peak_alpha"].abs() >= sM.ALPHA_LIMIT - 1e-9).sum() == arms["boundary"].sum()


def test_alpha_is_rescaled_by_the_spread_it_acts_on(alpha):
    """Raw alpha looks larger inside a stratum only because log accessibility is narrower
    there. If scaled_weight ever stops being alpha x SD, the caption's comparison is wrong."""
    assert np.allclose(alpha["scaled_weight"], alpha["peak_alpha"] * alpha["sd_log_acc"])
    arms = sM.violin_arms(alpha)
    pooled = arms.loc[arms["group"] == sM.POOLED, "sd_log_acc"].median()
    strata = arms.loc[arms["group"] != sM.POOLED, "sd_log_acc"].median()
    assert pooled > 3 * strata


def test_alpha_bootstrap_is_seeded(data):
    scores, _, _, panel, roles = data
    one = panel.loc[~panel["binding"]].iloc[:1]
    kw = dict(n_boot=6, seed=11)
    a = sM.alpha_by_stratum(scores, one, roles, **kw)
    b = sM.alpha_by_stratum(scores, one, roles, **kw)
    pd.testing.assert_frame_equal(a, b)
    assert a["peak_alpha_ci_lo"].notna().any()


def test_alpha_caption_counts_come_from_the_table(data, diff, alpha):
    _, table, _, panel, roles = data
    a = sM.summarise_alpha(alpha)
    arms = sM.violin_arms(alpha)
    strata = arms.loc[arms["group"] != sM.POOLED]
    assert a["n_strata"] == len(strata)
    assert a["neg_strata"] == int((strata["peak_alpha"] < 0).sum())
    assert a["binding_assays"] and all("binding" in s.lower() for s in a["binding_assays"])
    text = " ".join(sM.caption(panel, table, diff, roles, 60, alpha).split())
    assert f"favoured in {a['neg_strata']} of {a['n_strata']}" in text
    assert "receptor-binding arms" in text
    # The per-group breakdown explains the within-stratum count, so it must sum to it and must
    # not quietly include the pooled group, which is not a stratum.
    assert a["neg_strata"] == sum(a["by_group"][str(s)]["negative"] for s in sM.STRATA)
    assert f"at {sM.GROUP_LABEL[sM.POOLED]}" not in text.split("Two cautions")[0]
    # No alpha table, no alpha paragraph: the caption must not promise a panel it has not drawn.
    assert "refitted inside one stratum" not in " ".join(
        sM.caption(panel, table, diff, roles, 60).split())


@pytest.mark.slow
def test_script_renders_with_its_tables(tmp_path):
    assert sM.main(["--guide", str(GUIDE.source), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "sM_within_stratum.png").stat().st_size > 20_000
    values = pd.read_csv(tmp_path / "sM_within_stratum_values.tsv", sep="\t")
    assert len(values) == 72 and values["role"].isin(["base", "finetune"]).all()
    paired = pd.read_csv(tmp_path / "sM_paired_difference_values.tsv", sep="\t")
    assert len(paired) == 36
    peak = pd.read_csv(tmp_path / "sM_peak_alpha_by_stratum_values.tsv", sep="\t")
    assert len(peak) == 72 and peak["peak_alpha"].between(-1, 1).all()
    stratum_tests = pd.read_csv(tmp_path / "sM_dms_stratum_tests_values.tsv", sep="\t")
    assert len(stratum_tests) == 36 and set(stratum_tests["test"]) == {"kruskal", "mannwhitney"}


# ---------------------------------------------------------------------------
# The masked arm (19 Oct 2026)
# ---------------------------------------------------------------------------

def test_the_masked_arm_recomputes_its_own_bars_and_intervals(data):
    """``dms_within_stratum`` holds the unmasked arm, intervals included.

    There is no shipped masked counterpart, so the masked panel's bars *and* whiskers are
    both computed in the figure. The point estimates must still be the correlations the
    masked scores give, and the interval must bracket the bar: a cell whose bar sits outside
    its own whiskers would be a join or an ordering fault, which is the failure mode these
    tables have.
    """
    scores, _, _, panel, _ = data
    masked = sM.apply_masked(scores, GUIDE.read_table("dms_scores_masked"))
    one = panel.iloc[:1]
    band = sM.recompute_with_band(masked, one, n_boot=30, seed=5)
    models = masked.loc[masked["dataset"] == one["dataset"].iloc[0], "model_name"].nunique()
    assert len(band) == len(sM.GROUPS) * models
    assert (band["ci_lo"] <= band["rho"]).all() and (band["rho"] <= band["ci_hi"]).all()
    # The bars are the masked correlations, not the unmasked ones the shipped table holds.
    point = sM.recompute(masked, one)
    merged = band.merge(point, on=["dataset", "trait", "model_name", "stratum"])
    assert len(merged) == len(point)
    assert (merged["rho"] - merged["rho_recomputed"]).abs().max() < 1e-12


def test_the_masked_bars_differ_from_the_shipped_unmasked_ones(data):
    """If they did not, nothing would have been switched and no test above would notice."""
    scores, table, _, panel, _ = data
    masked = sM.apply_masked(scores, GUIDE.read_table("dms_scores_masked"))
    point = sM.recompute(masked, panel)
    merged = table.merge(point, on=["dataset", "trait", "model_name", "stratum"])
    assert len(merged) == len(table)
    assert (merged["rho"] - merged["rho_recomputed"]).abs().max() > 1e-3


def test_the_gate_still_runs_against_the_unmasked_scores(data):
    """The masked arm is drawn, but the arm that is *checked* is the unmasked one: it is the
    only one with a shipped fixed point. Swapping the columns must break the gate."""
    scores, table, _, panel, _ = data
    masked = sM.apply_masked(scores, GUIDE.read_table("dms_scores_masked"))
    with pytest.raises(ValueError, match="disagrees"):
        sM.check_against_table(sM.recompute(masked, panel), table)


def test_the_caption_states_which_estimator_it_drew(data, diff, alpha):
    _, table, _, panel, roles = data
    masked = " ".join(sM.caption(panel, table, diff, roles, 60, alpha, scoring="masked").split())
    unmasked = " ".join(sM.caption(panel, table, diff, roles, 60, alpha, scoring="unmasked").split())
    assert "masked marginal" in masked and "wild-type (unmasked) marginal" in unmasked
    assert masked != unmasked


def test_the_gate_refit_asks_for_the_pooled_group_only(data):
    """The gate reads the pooled optimum, so it must not pay for the three strata as well."""
    scores, _, summary, panel, roles = data
    pooled = sM.alpha_by_stratum(scores, panel.iloc[:1], roles, n_boot=0, groups=(sM.POOLED,))
    assert set(pooled["group"]) == {str(sM.POOLED)}


# ---------------------------------------------------------------------------
# Does the measured phenotype itself differ across strata? (21 Oct 2026)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tests_table(data):
    scores, _, _, panel, _ = data
    return sM.dms_stratum_tests(scores, panel)


def test_holm_matches_a_worked_example():
    got = sM._holm([0.01, 0.04, 0.03])
    # 3x0.01, then max(that, 2x0.03), then max(that, 1x0.04): Holm is monotone in rank order.
    assert np.allclose(got, [0.03, 0.06, 0.06])
    assert (sM._holm([0.5, 0.6, 0.9]) <= 1.0).all()


def test_stars_follow_the_stated_thresholds():
    assert sM._stars(0.0005) == "***" and sM._stars(0.005) == "**" and sM._stars(0.04) == "*"
    assert sM._stars(0.051) == "" and sM._stars(float("nan")) == ""


def test_every_panel_gets_one_omnibus_and_three_pairs(tests_table, data):
    _, _, _, panel, _ = data
    kw = tests_table.loc[tests_table["test"] == "kruskal"]
    assert len(kw) == len(panel)
    for (_, _), g in tests_table.groupby(["dataset", "trait"]):
        assert (g["test"] == "mannwhitney").sum() == 3
    pairs = set(zip(tests_table.loc[tests_table["test"] == "mannwhitney", "group_a"],
                    tests_table.loc[tests_table["test"] == "mannwhitney", "group_b"]))
    assert pairs == {("1", "2"), ("1", "3"), ("2", "3")}


def test_the_omnibus_uses_the_three_strata_and_not_the_pooled_group(tests_table, data):
    """The pooled group is the union of the strata, so testing it against them would compare
    a set with itself. The omnibus n must therefore be the three strata summed, nothing more."""
    scores, _, _, panel, _ = data
    kw = tests_table.loc[tests_table["test"] == "kruskal"].set_index(["dataset", "trait"])
    for p in panel.itertuples():
        pairs = tests_table.loc[(tests_table["dataset"] == p.dataset)
                                & (tests_table["trait"] == p.trait)
                                & (tests_table["test"] == "mannwhitney")]
        per_stratum = dict(zip(pairs["group_a"], pairs["n_a"])) | dict(zip(pairs["group_b"], pairs["n_b"]))
        assert kw.loc[(p.dataset, p.trait), "n_a"] == sum(per_stratum.values())
        assert set(per_stratum) == {"1", "2", "3"}


def test_the_omnibus_p_is_what_scipy_gives_on_the_deduplicated_values(tests_table, data):
    """Each replacement is measured once but scored once per model, so the test must run on the
    de-duplicated values -- the same ones the purple curve averages. Testing the un-deduplicated
    frame doubles every count and shrinks every p-value."""
    from scipy.stats import kruskal
    scores, _, _, panel, _ = data
    p = panel.iloc[0]
    sub = scores.loc[scores["dataset"] == p.dataset].drop_duplicates(["site", "aa"])
    sub = sub.loc[sub[p.trait].notna()]
    expect = kruskal(*(sub.loc[sub["nt_steps"] == s, p.trait].to_numpy(float) for s in sM.STRATA))
    row = tests_table.loc[(tests_table["dataset"] == p.dataset) & (tests_table["trait"] == p.trait)
                          & (tests_table["test"] == "kruskal")].iloc[0]
    assert row["p_value"] == pytest.approx(expect.pvalue)
    assert row["statistic"] == pytest.approx(expect.statistic)
    assert row["n_a"] < len(scores.loc[scores["dataset"] == p.dataset, p.trait].dropna())


def test_a_stratum_is_flagged_only_when_it_beats_both_of_the_others(tests_table):
    """With three groups, "the one that is different" can only mean separating from both."""
    for (_, _), g in tests_table.groupby(["dataset", "trait"]):
        kw = g.loc[g["test"] == "kruskal"].iloc[0]
        pairs = g.loc[g["test"] == "mannwhitney"]
        flagged = {s for s in kw["flagged_groups"].split("|") if s}
        if kw["p_value"] >= sM.TEST_ALPHA:
            assert not flagged and pairs["p_holm"].isna().all()
            continue
        for s in ("1", "2", "3"):
            involved = pairs.loc[(pairs["group_a"] == s) | (pairs["group_b"] == s)]
            assert len(involved) == 2
            assert ((involved["p_holm"] < sM.TEST_ALPHA).all()) == (s in flagged)


def test_the_corrected_p_is_never_smaller_than_the_raw_one(tests_table):
    pairs = tests_table.loc[(tests_table["test"] == "mannwhitney") & tests_table["p_holm"].notna()]
    assert len(pairs) and (pairs["p_holm"] >= pairs["p_value"] - 1e-12).all()
    assert (pairs["p_holm"] <= 1.0).all()


def test_a_thin_stratum_is_skipped_rather_than_tested(data):
    scores, _, _, panel, _ = data
    p = panel.iloc[:1]
    thin = scores.loc[scores["dataset"] == p["dataset"].iloc[0]].copy()
    keep = thin["nt_steps"] != 3
    thin = pd.concat([thin.loc[keep], thin.loc[~keep].head(sM.MIN_TEST_N - 1)])
    assert sM.dms_stratum_tests(thin, p).empty


def test_the_panel_draws_a_star_only_for_a_significant_omnibus(tests_table):
    """``stratum_stars`` is what the panel reads. A panel whose omnibus is not significant must
    give back nothing at all, stratum stars included -- the post hoc tests are not drawn alone."""
    kw = tests_table.loc[tests_table["test"] == "kruskal"]
    for row in kw.itertuples():
        omnibus, marks = sM.stratum_stars(tests_table, row.dataset, row.trait)
        if row.p_value >= sM.TEST_ALPHA:
            assert omnibus == "" and marks == {}
        else:
            assert omnibus == sM._stars(row.p_value)
            assert set(marks) == {int(s) for s in row.flagged_groups.split("|") if s}
            assert all(m for m in marks.values())
    assert sM.stratum_stars(tests_table, "no-such-dataset", "no-such-trait") == ("", {})


def test_the_star_over_a_stratum_takes_its_weaker_separation(tests_table):
    """It is flagged for differing from both others, so the claim is only as strong as the
    lesser of the two; drawing the stronger one would overstate it."""
    for row in tests_table.loc[tests_table["test"] == "kruskal"].itertuples():
        _, marks = sM.stratum_stars(tests_table, row.dataset, row.trait)
        pairs = tests_table.loc[(tests_table["dataset"] == row.dataset)
                                & (tests_table["trait"] == row.trait)
                                & (tests_table["test"] == "mannwhitney")]
        for s, mark in marks.items():
            involved = pairs.loc[(pairs["group_a"] == str(s)) | (pairs["group_b"] == str(s))]
            assert mark == sM._stars(float(involved["p_holm"].max()))


def test_the_caption_sentence_counts_come_from_the_test_table(tests_table, data, diff, alpha):
    _, table, _, panel, roles = data
    text = " ".join(sM.caption(panel, table, diff, roles, 60, alpha, dms_tests=tests_table).split())
    kw = tests_table.loc[tests_table["test"] == "kruskal"]
    n_sig = int((kw["p_value"] < sM.TEST_ALPHA).sum())
    n_flagged = int(((kw["p_value"] < sM.TEST_ALPHA) & (kw["flagged_groups"] != "")).sum())
    assert f"significant in {n_sig} of {len(kw)} panels" in text
    assert f"({n_flagged} panels have one;" in text
    # No test table, no sentence: the caption must not describe stars the panel has not drawn.
    assert "Purple stars" not in " ".join(sM.caption(panel, table, diff, roles, 60, alpha).split())
