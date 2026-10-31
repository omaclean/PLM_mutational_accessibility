"""Red team: figure scripts. Captions and legends against the numbers actually plotted, titles
against the data, seeds, hard-coded paths, and the failure modes that used to draw a quiet lie.

Bugs these tests pinned (fixed in the figure scripts):

* Figure 2's caption printed "-0.00", and said the lineage-to-lineage spread was "much narrower"
  than the bootstrap interval although for one cell it is 83% of its width. The caption now
  reads the ratio from the table.
* Figure 3's caption said that for negative weights "the curves fall under that line", but panel C
  (Pearson) never does; the sentence is now built per panel from the plotted means.
* Figures 2, 5 (S4) and 2 (S2) quoted the guide's lineage count in titles and captions without
  checking that the table holds those lineages.
* Figure 4's legend listed "Best weight" with no optimum marked and an interval band that was not
  drawn when the bootstrap was off.
"""

import ast
import re

import numpy as np
import pandas as pd
import pytest

import make_fig02 as fig2
import make_fig03 as fig3
import make_fig04 as fig4
import make_sB_ranking_metrics as sB
import make_sD_partial_correlation as sD
import make_sJ_dms_concordance as sJ
import make_sM_within_stratum as sM
from mutacc import strata
from mutacc import style
from mutacc.guide import REPO_ROOT, GuideError, load_guide
from mutacc.metrics import mcclish_floor

G13 = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
G5 = load_guide(REPO_ROOT / "guides" / "h3n2_5lineage.yaml")


def title_of(ax):
    return " ".join(ax.get_title(loc=loc) for loc in ("left", "center", "right"))


def flat(text):
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Figure 2 and S4
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frames():
    return G5.read_table("replacement_scores"), G13.read_table("replacement_scores")


@pytest.fixture(scope="module")
def fig2_parts(frames):
    f5, f13 = frames
    table, pairs, summaries = fig2.compute(f5, f13, G5, G13, n_boot=12, seed=3)
    return table, pairs, summaries


def test_fig2_titles_claim_the_lineage_counts_the_tables_hold(fig2_parts, frames):
    f5, f13 = frames
    table, pairs, summaries = fig2_parts
    fig = fig2.draw(table, pairs, summaries, G5, G13)
    titles = [title_of(ax) for ax in fig.axes]
    assert f"({f5['lineage'].nunique()} lineages)" in titles[0]
    assert f"({f13['lineage'].nunique()} lineages)" in titles[1]
    assert f"({f13['lineage'].nunique()} lineages)" in titles[2]


def test_fig2_panel_b_legend_counts_match_the_plotted_points(fig2_parts):
    table, pairs, summaries = fig2_parts
    fig = fig2.draw(table, pairs, summaries, G5, G13)
    ax_b = fig.axes[1]
    labels = [t.get_text() for t in ax_b.get_legend().get_texts()]
    for model in G13.models:
        want = [x for x in labels if x.startswith(model.label + ":")]
        assert len(want) == 1, labels
        n_agree = int((np.sign(pairs[model.key]["mut_asym"]) == np.sign(pairs[model.key]["plm_asym"])).sum())
        assert want[0] == f"{model.label}: {n_agree} of {len(pairs[model.key])} pairs agree"
    drawn = sum(len(c.get_offsets()) for c in ax_b.collections)
    assert drawn == sum(len(p) for p in pairs.values())


def test_fig2_caption_numbers_equal_the_values_table(fig2_parts):
    table, pairs, summaries = fig2_parts
    text = flat(fig2.caption(table, summaries, G5, G13))
    values = fig2.build_values(table, pairs, summaries, G5, G13)
    ft = next(m for m in G5.models if m.role == "finetune")
    base = next(m for m in G5.models if m.role == "base")
    last = [s for s in table["stratum"].unique() if s != fig2.ALL][-1]
    a = values[(values.panel == "A") & (values.group == last)].set_index("series")
    for key, label in ((ft.key, "fine-tuned"), (base.key, "base")):
        assert fig2.fmt2(a.loc[key, "value"]) in text
    assert fig2.fmt2(a.loc["neg_grantham", "value"]) in text and fig2.fmt2(a.loc["blosum62", "value"]) in text
    n_pairs = int(values[(values.panel == "C") & (values.series == ft.key)]["n"].iloc[0])
    assert f"{n_pairs} pairs" in text and len(pairs[ft.key]) == n_pairs
    c = values[values.panel == "C"].set_index(["series", "measure"])["value"]
    assert fig2.fmt2(c.loc[(ft.key, "rho_overall")]) in text and fig2.fmt2(c.loc[(ft.key, "rho_pair_specific")]) in text
    assert "-0.00" not in text


@pytest.mark.parametrize("value, text", [(-0.001, "0.00"), (-0.004999, "0.00"), (0.0, "0.00"), (-0.006, "-0.01"),
                                         (0.1449, "0.14"), (-0.13, "-0.13")])
def test_two_decimal_formatting_never_prints_negative_zero(value, text):
    assert fig2.fmt2(value) == text


def test_fig2_lineage_spread_sentence_is_read_from_the_numbers(fig2_parts):
    table, _, _ = fig2_parts
    rows = table[table["control"] == ""]
    worst = ((rows["lineage_max"] - rows["lineage_min"]) / (rows["hi"] - rows["lo"])).max()
    sentence = fig2.lineage_spread_text(table)
    assert (f"at most {worst:.0%}" in sentence) == (worst < 1)
    wide = table.copy()
    wide["lineage_max"] = wide["lineage_min"] + 10 * (wide["hi"] - wide["lo"]).clip(lower=1e-3)
    assert fig2.lineage_spread_text(wide).startswith("not narrower")
    assert "much narrower" not in sentence


def test_fig2_and_friends_refuse_a_table_whose_lineages_differ_from_the_guide(frames):
    f5, f13 = frames
    five_as_thirteen = f5.copy()
    with pytest.raises(GuideError, match="lineages differ"):
        fig2.compute(five_as_thirteen, f13, G13, G13, n_boot=0, seed=1)          # 5 lineages, guide says 13
    with pytest.raises(GuideError, match="lineages differ"):
        fig2.compute(f5, f5, G5, G13, n_boot=0, seed=1)                          # panel B/C table is the wrong set
    with pytest.raises(GuideError, match="lineages differ"):
        sD.compute(f13, G5, n_boot=0, seed=1)
    with pytest.raises(GuideError, match="lineages differ"):
        G5.check_lineages(["G.1", "J_int"], "t")
    G5.check_lineages(G5.lineage_ids, "t")


def test_fig2_bootstrap_is_seeded_and_the_default_seed_is_a_constant(frames):
    f5, f13 = frames
    a = fig2.compute(f5, f13, G5, G13, n_boot=4, seed=11)[0]
    b = fig2.compute(f5, f13, G5, G13, n_boot=4, seed=11)[0]
    c = fig2.compute(f5, f13, G5, G13, n_boot=4, seed=12)[0]
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["lo"], c["lo"])
    seeds = {fig2.parse(["--guide", str(G5.source)]).seed for _ in range(3)}
    assert len(seeds) == 1


def test_sD_partial_correlation_titles_and_values_name_the_single_lineage(frames):
    """S4 is one lineage now, so the title must say which one, not a lineage count."""
    f5, _ = frames
    table = sD.compute(f5, G5, n_boot=6, seed=2)
    chosen = strata.primary_lineage(f5, [(lin.id, lin.root_type) for lin in G5.lineages],
                                    G5.primary_lineage)
    assert set(table["lineage"]) == {chosen}
    fig = sD.draw(table, G5)
    for ax in fig.axes:
        if ax.get_visible():
            assert f"lineage {chosen}" in title_of(ax)
            assert "lineages)" not in title_of(ax)
    text = flat(sD.caption(table, G5))
    ft = next(m for m in G5.models if m.role == "finetune")
    last = [s for s in table["stratum"].unique() if s != sD.ALL][-1]
    for control in ("", "neg_grantham"):
        row = table[(table.series == ft.key) & (table.stratum == last) & (table.control == control)]
        assert fig2.fmt2(float(row["estimate"].iloc[0])) in text or f"{float(row['estimate'].iloc[0]):.2f}" in text


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep():
    return G13.read_table("alpha_sweep_by_lineage")


def test_fig3_caption_alpha_range_lineage_count_and_hidden_panels_match_the_values(sweep):
    bands = fig3.band_table(sweep, G13, n_boot=30, seed=1)
    text = flat(fig3.caption(G13, bands))
    n = sweep.loc[sweep["model_variant"] == "plm_alpha_sweep", "lineage"].nunique()
    assert f"means over {n} lineages" in text
    assert f"from {bands['alpha'].min():g} to {bands['alpha'].max():g}" in text
    hidden = {letter for letter, (metric, *_rest, ref) in zip("ABC", fig3.PANELS)
              if (bands.loc[bands["metric"] == metric, "mean"] < ref[0]).any()}
    assert hidden == {"A", "B"}, "on the shipped sweep the Pearson panel never falls under its cut line"
    assert "In (A) and (B) the curves fall under that line" in text
    assert "In (C) no curve falls below the line" in text
    assert "for negative weights the curves fall under that line" not in text


def test_fig3_hidden_sentence_covers_all_cases():
    cols = ["metric", "mean"]
    metrics = [p[0] for p in fig3.PANELS]
    none = pd.DataFrame({"metric": metrics, "mean": [0.5, 0.9, 0.5]})
    every = pd.DataFrame({"metric": metrics, "mean": [-1.0, 0.1, -1.0]})
    assert fig3.hidden_text(none) == "No curve falls below the line in any panel."
    assert fig3.hidden_text(every).startswith("In (A) and (B) and (C)")
    assert list(none.columns) == cols


def test_fig3_panel_titles_name_the_column_they_plot():
    names = {"mut_flat_global_spearman_r": "Spearman", "site_logistic_auroc": "AUROC",
             "winners_auroc": "AUROC"}
    for column, title, tagline, ylabel, ref in fig3.PANELS:
        assert names[column] in title and names[column] in ylabel


def test_fig3_refuses_a_table_missing_a_lineage_or_a_cell(sweep):
    lost = sweep[sweep["lineage"] != "K"]
    with pytest.raises(ValueError, match="lineages"):
        fig3.band_table(lost, G13, n_boot=3)
    holed = sweep.drop(sweep[(sweep.lineage == "K") & np.isclose(sweep.alpha.fillna(9), 0.5)
                             & (sweep.model == G13.models[0].key)].index)
    with pytest.raises(ValueError):
        fig3.band_table(holed, G13, n_boot=3)


# ---------------------------------------------------------------------------
# S2 (three alphas only, to keep it quick)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s02_parts():
    frame = sB.load_scores(G13, G13.results_dir)
    alphas = np.array([-1.0, 0.0, 1.0])
    # The two correlation panels are read from the sweep table, so the quick grid has to be
    # cut to the same three alphas before the two routes can be joined.
    shipped = sB.sweep_grid(G13, G13.results_dir)
    shipped = shipped[shipped["alpha"].isin(np.round(alphas, 10))]
    grid = sB.join_grids(sB.metric_grid(frame, alphas, sB.FPR_CAP), shipped)
    base = sB.baseline_grid(frame, G13, G13.results_dir, sB.FPR_CAP)
    return frame, grid, base


def test_sB_ranking_metrics_caption_prevalence_and_floor_claims_match_an_independent_count(s02_parts):
    frame, grid, base = s02_parts
    observed = G13.read_table("replacement_observed")
    prev = observed.groupby("lineage")["obs_present"].mean()
    n_observed = G13.read_table("replacement_frequency").query("obs_freq > 0").groupby("lineage").size()
    band = sB.bands(grid, G13, n_boot=10, seed=1)
    text = flat(sB.caption(base, band, G13, sB.FPR_CAP, sB.TOP_FRACTION, 10, n_observed))
    assert f"prevalence runs from {prev.min():.1%} to {prev.max():.1%}" in text
    assert f"{mcclish_floor(sB.FPR_CAP):.4f}" in text
    # "between N and M of them per lineage": counted off the frequency table, not the prose
    assert f"between {n_observed.min():,} and {n_observed.max():,} of them per lineage" in text
    # "the metric cannot fall below" that value: true of every recomputed cell
    assert grid["pauroc_std"].min() >= mcclish_floor(sB.FPR_CAP) - 1e-9
    assert (base["prevalence"].to_numpy() > 0).all() and len(base) == G13.n_lineages


def test_sB_ranking_metrics_baseline_does_not_depend_on_row_order(s02_parts):
    """The accessibility-only score is almost all ties, so PR-AUC used to depend on row order.

    Panel F's AUROC is scored on the same tied accessibility values over a much smaller set,
    so it is held to the same standard.
    """
    frame, _, base = s02_parts
    shuffled = frame.sample(frac=1.0, random_state=5).reset_index(drop=True)
    again = sB.baseline_grid(shuffled, G13, G13.results_dir, sB.FPR_CAP).set_index("lineage").sort_index()
    first = base.set_index("lineage").sort_index()
    for col in ("pr_auc", "pauroc_std", "auroc", "prauc_fold", "winners_auroc"):
        assert np.allclose(first[col], again[col], atol=1e-12), col


def test_sB_ranking_metrics_chance_ticks_equal_the_number_of_lineages(s02_parts):
    _frame, grid, base = s02_parts
    band = sB.bands(grid, G13, n_boot=5, seed=1)
    fig = sB.draw(base, band, G13, sB.FPR_CAP, sB.TOP_FRACTION)
    ticks = [ln for ln in fig.axes[0].lines if ln.get_marker() == "_"]
    assert len(ticks) == 1 and len(ticks[0].get_xdata()) == G13.n_lineages
    # One axes per panel, and only panel A carries the per-lineage ticks.
    assert len(fig.axes) == len(sB.PANELS)
    assert not [ln for ax in fig.axes[1:] for ln in ax.lines if ln.get_marker() == "_"]


def test_sB_ranking_metrics_refuses_scores_for_a_different_lineage_set(monkeypatch):
    table_name = sB.scores_table_name()
    real = G13.read_table(table_name)
    cut = real[real["lineage"] != "K"]
    monkeypatch.setattr(type(G13), "read_table",
                        lambda self, name, results=None, **kw: cut if name == table_name
                        else pd.read_csv(self.table_path(name, results), sep="\t", **kw))
    with pytest.raises(GuideError, match="lineages differ"):
        sB.load_scores(G13, G13.results_dir)


# ---------------------------------------------------------------------------
# S10 and S12
# ---------------------------------------------------------------------------

def test_sJ_dms_concordance_caption_counts_equal_the_plotted_points():
    table = G13.read_table("dms_cross_lineage_pairs")
    values = sJ.values_table(table)
    text = flat(sJ.caption(table))
    assert f"({len(table)} pairs:" in text
    for system, n in table["system"].value_counts().items():
        assert f"{system} ({n} pair" in text
    spearman_a = values[(values.measure == "spearman_rho") & (values.x_axis == "aa_distance")]
    assert f"For {int(spearman_a['censored'].sum())} pairs, more than" in text
    assert f"ranges from {table['spearman_rho'].min():.2f} to {table['spearman_rho'].max():.2f}" in text
    # every plotted point is in the values table exactly once per panel
    assert len(spearman_a) == table["spearman_rho"].notna().sum()


def test_sJ_dms_concordance_jitter_is_seeded_and_never_moves_a_point_more_than_it_says():
    table = G13.read_table("dms_cross_lineage_pairs")
    a = sJ.panel_points(table, "spearman_rho", "spearman_rho_no_floor", "aa_distance")
    b = sJ.panel_points(table, "spearman_rho", "spearman_rho_no_floor", "aa_distance")
    pd.testing.assert_frame_equal(a, b)
    ratio = a["x_plotted"] / a["x"]
    assert ratio.between(1 - sJ.JITTER - 1e-12, 1 + sJ.JITTER + 1e-12).all()


def test_sJ_dms_concordance_refuses_inconsistent_divergence_and_out_of_range_coefficients():
    table = G13.read_table("dms_cross_lineage_pairs")
    bad = table.copy()
    bad.loc[0, "pct_divergence"] *= 1.5
    with pytest.raises(ValueError, match="pct_divergence"):
        sJ.validate(bad)
    bad = table.copy()
    bad.loc[0, "spearman_rho"] = 1.2
    with pytest.raises(ValueError, match="outside"):
        sJ.validate(bad)
    bad = table.copy()
    bad.loc[0, "spearman_rho"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sJ.validate(bad)


def test_sM_within_stratum_caption_counts_are_recomputed_from_the_values_not_copied():
    scores = G13.read_table("dms_scores")
    table = G13.read_table("dms_within_stratum")
    summary = G13.read_table("dms_peak_summary")
    panel, roles = sM.panels(summary), sM.role_of_model(summary)
    diff = sM.paired_difference(scores, panel, roles, n_boot=20, seed=1)
    text = flat(sM.caption(panel, table, diff, roles, 20))
    fit = diff[~diff["binding"]]
    assert f"across the {len(fit)} (assay, stratum) cells" in text
    excl = int(((fit.ci_lo > 0) | (fit.ci_hi < 0)).sum())
    assert f"excludes zero in {excl}, with the fine-tuned model higher in {int((fit.ci_lo > 0).sum())}" in text
    assert f"lower in {int((fit.ci_hi < 0).sum())}" in text


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fig4_curves():
    scores = G13.read_table("dms_scores")
    summary = G13.read_table("dms_peak_summary")
    return fig4.compute_curves(scores, summary, n_boot=0)


def test_fig4_caption_counts_equal_the_curves(fig4_curves):
    arms = fig4._arm_table(fig4_curves)
    text = flat(fig4.caption(fig4_curves, n_boot=0))
    flagged = arms[~arms["interpretable"]]
    n_edge = int(flagged["boundary"].sum())
    n_weak = int((flagged["weak_signal"] & ~flagged["boundary"]).sum())
    m = re.search(r"No optimum is marked on (\d+) of (\d+) curves: (\d+) because .*? (\d+) more because", text)
    assert m, text
    assert tuple(map(int, m.groups())) == (len(flagged), len(arms), n_edge, n_weak)
    assert len(flagged) == n_edge + n_weak
    assert f"{arms['n'].min():,} to {arms['n'].max():,} per curve" in text
    assert f"{arms['n_sites'].min():,} to {arms['n_sites'].max():,} sites" in text
    for dataset in arms["dataset"].unique():
        assert dataset in text


def test_fig4_legend_names_only_what_is_drawn(fig4_curves):
    fig = fig4.draw(fig4_curves)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "Best weight" in labels and "95% interval over DMS sites" not in labels      # n_boot = 0: no band
    dots = [ln for ax in fig.axes for ln in ax.lines if ln.get_marker() == "o"]
    assert dots, "legend names a best-weight dot and some are drawn"
    # With nothing interpretable, no filled dot and no "Best weight" entry. A hollow marker
    # may still be drawn for a real interior optimum under the signal floor, and when one is,
    # it has to be named in the legend too -- the rule is that the legend names what is drawn.
    none_marked = fig4_curves.assign(interpretable=False)
    fig = fig4.draw(none_marked)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "Best weight" not in labels
    hollow = [ln for ax in fig.axes for ln in ax.lines
              if ln.get_marker() == "o" and ln.get_markerfacecolor() == style.SURFACE]
    filled = [ln for ax in fig.axes for ln in ax.lines
              if ln.get_marker() == "o" and ln.get_markerfacecolor() != style.SURFACE]
    assert not filled
    assert ("Optimum below the signal floor" in labels) == bool(hollow)


def test_fig4_band_entry_appears_when_a_band_exists():
    scores = G13.read_table("dms_scores")
    summary = G13.read_table("dms_peak_summary")
    curves = fig4.compute_curves(scores, summary, n_boot=3, max_arms=2)
    fig = fig4.draw(curves)
    assert "95% interval over DMS sites" in [t.get_text() for t in fig.legends[0].get_texts()]


# ---------------------------------------------------------------------------
# Whole-repo properties of the figure code
# ---------------------------------------------------------------------------

PORTED = ["make_fig02.py", "make_fig03.py", "make_fig04.py", "make_sB_ranking_metrics.py", "make_sD_partial_correlation.py", "make_sJ_dms_concordance.py",
          "make_sM_within_stratum.py", "panelise.py"]
LIBRARY = ["accessibility.py", "dms_sweep.py", "guide.py", "io.py", "metrics.py", "placement.py",
           "strata.py", "style.py", "symmetry.py"]


def sources():
    for name in PORTED:
        yield REPO_ROOT / "figures" / name
    for name in LIBRARY:
        yield REPO_ROOT / "mutacc" / name
    for tool in ("run_symmetry_battery.py", "run_symmetry_dms.py", "export_tables.py"):
        yield REPO_ROOT / "tools" / tool


@pytest.mark.parametrize("path", list(sources()), ids=lambda p: p.name)
def test_every_random_generator_is_seeded_and_no_global_random_state_is_used(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name == "default_rng":
            assert node.args or node.keywords, f"{path.name}:{node.lineno} default_rng() without a seed"
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute) and fn.value.attr == "random" \
                and getattr(fn.value.value, "id", "") in ("np", "numpy") \
                and fn.attr not in ("default_rng", "Generator", "SeedSequence", "PCG64"):
            raise AssertionError(f"{path.name}:{node.lineno}: global numpy random state ({fn.attr})")
        if isinstance(fn, ast.Attribute) and getattr(fn.value, "id", "") == "random":
            raise AssertionError(f"{path.name}:{node.lineno}: python random module")
        assert name != "seed" or "default" in path.name, f"{path.name}:{node.lineno}: global seeding"


def test_every_figure_script_has_a_fixed_default_seed():
    for name in PORTED:
        text = (REPO_ROOT / "figures" / name).read_text()
        for m in re.finditer(r"(?:seed|SEED)\w*\s*[:=]\s*(?:int\s*=\s*)?(\d+)", text):
            assert int(m.group(1)) > 0
        assert "time.time" not in text and "datetime.now" not in text, f"{name} reads the clock"


def test_generated_text_never_contains_a_machine_specific_path(tmp_path, fig2_parts):
    table, pairs, summaries = fig2_parts
    outputs = [fig2.caption(table, summaries, G5, G13),
               fig2.build_values(table, pairs, summaries, G5, G13).to_csv(sep="\t")]
    for text in outputs:
        assert "/" + "home" not in text and "/" + "Users" not in text and str(REPO_ROOT) not in text


def test_figure_values_never_hold_non_finite_numbers_where_a_value_is_drawn(fig2_parts):
    table, pairs, summaries = fig2_parts
    values = fig2.build_values(table, pairs, summaries, G5, G13)
    drawn = values[values["measure"].isin(["plm_vs_access", "property_vs_access", "direction"])]
    assert np.isfinite(drawn["value"].to_numpy(float)).all()
    assert not np.isinf(values.select_dtypes("number").to_numpy(float)).any()
