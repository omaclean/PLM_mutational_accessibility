"""Figure 2 and S4: golden values, independent re-derivation, input guards and render smoke tests.

The golden numbers are the published ones (the Fig. 2 panel titles, the strata summary in the
exploration archive's ``strata_metrics_summary.tsv``, and the Fig. 5 asymmetry panel), reproduced
from the shipped tables. Tolerances are 6e-4: ``plm_prob`` is shipped to 7 significant digits.
"""

import re

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr

import make_fig02 as fig2
import make_sD_partial_correlation as sD
from mutacc import io, strata
from mutacc.guide import REPO_ROOT, load_guide, read_fasta_headers, root_header_problems

G5 = load_guide(REPO_ROOT / "guides" / "h3n2_5lineage.yaml")
G13 = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
BASE = next(m.key for m in G5.models if m.role == "base")
FT = next(m.key for m in G5.models if m.role == "finetune")
PROPS = [p[0] for p in fig2.PROPERTIES]
TOL = 6e-4


@pytest.fixture(scope="module")
def frame5():
    return G5.read_table("replacement_scores")


@pytest.fixture(scope="module")
def frame13():
    return G13.read_table("replacement_scores")


@pytest.fixture(scope="module")
def table5(frame5):
    return strata.strata_table(frame5, [FT, BASE], PROPS, n_boot=30)


def est(table, series, stratum, control="", col="estimate"):
    row = table[(table.series == series) & (table.stratum == stratum) & (table.control == control)]
    assert len(row) == 1
    return float(row[col].iloc[0])


# ---------------------------------------------------------------------------
# Golden values
# ---------------------------------------------------------------------------

GOLDEN_STRATA = [  # (series, control, {stratum: published mean over lineages})
    (FT, "", {"all": 0.6521, "1": 0.2883, "2": 0.2451, "3": 0.1398}),
    (BASE, "", {"all": 0.2744, "1": 0.0705, "2": 0.0701, "3": -0.0014}),
    ("blosum62", "", {"all": 0.3774, "1": 0.1023, "2": 0.1260, "3": -0.0750}),
    ("neg_grantham", "", {"all": 0.2660, "1": 0.0617, "2": 0.0575, "3": -0.1306}),
    (FT, "neg_grantham", {"all": 0.6188, "1": 0.2867, "2": 0.2390, "3": 0.1850}),
    (FT, "blosum62", {"all": 0.5802, "1": 0.2728, "2": 0.2194, "3": 0.1734}),
    (BASE, "neg_grantham", {"all": 0.2038, "1": 0.0518, "2": 0.0580, "3": 0.0338}),
    (BASE, "blosum62", {"all": 0.1427, "1": 0.0282, "2": 0.0329, "3": 0.0196}),
]


@pytest.mark.parametrize("series, control, golden", GOLDEN_STRATA)
def test_strata_reproduce_published_five_lineage_values(table5, series, control, golden):
    for stratum, value in golden.items():
        assert est(table5, series, stratum, control) == pytest.approx(value, abs=TOL), (series, control, stratum)


def test_strata_reproduce_the_published_pooled_titles(table5):
    """The Fig. 2 hexbin titles: pooled rho 0.288/0.245/0.139 fine-tuned, 0.070/0.070/-0.001 base."""
    for stratum, value in {"1": 0.288, "2": 0.245, "3": 0.139}.items():
        assert est(table5, FT, stratum, col="pooled") == pytest.approx(value, abs=1e-3)
    for stratum, value in {"1": 0.070, "2": 0.070, "3": -0.001}.items():
        assert est(table5, BASE, stratum, col="pooled") == pytest.approx(value, abs=1e-3)


def test_strata_partition_every_row(table5, frame5):
    """The three strata must together be every replacement: nothing silently dropped."""
    per_lineage = frame5[frame5.model == FT].groupby("lineage").size().mean()
    parts = sum(est(table5, FT, s, col="n_rows") for s in ("1", "2", "3"))
    assert parts == pytest.approx(per_lineage)
    assert est(table5, FT, "all", col="n_rows") == pytest.approx(per_lineage)


def test_thirteen_lineage_strata_agree_with_five_lineage_to_within_a_few_hundredths(frame13, table5):
    t13 = strata.strata_table(frame13, [FT, BASE], PROPS, n_boot=0)
    for s in ("1", "2", "3"):
        assert abs(est(t13, FT, s) - est(table5, FT, s)) < 0.02
        assert abs(est(t13, BASE, s) - est(table5, BASE, s)) < 0.02


GOLDEN_ASYM = {  # published Fig. 5 panel F: Spearman +0.539 over 190 pairs (fine-tuned, 13 lineages)
    FT: dict(n_pairs=190, n_agree=135, rho_overall=0.5394, rho_pair_specific=0.3492),
    BASE: dict(n_pairs=190, n_agree=117, rho_overall=0.2917, rho_pair_specific=-0.0094),
}


@pytest.mark.parametrize("model", [FT, BASE])
def test_asymmetry_reproduces_published_values(frame13, model):
    got = strata.asymmetry_summary(strata.pair_asymmetry(frame13, model))
    want = GOLDEN_ASYM[model]
    assert got["n_pairs"] == want["n_pairs"] and got["n_agree"] == want["n_agree"]
    assert got["rho_overall"] == pytest.approx(want["rho_overall"], abs=TOL)
    assert got["rho_pair_specific"] == pytest.approx(want["rho_pair_specific"], abs=TOL)


def test_most_of_the_asymmetry_is_a_per_amino_acid_tendency(frame13):
    """Why panel C exists: ~80% of both asymmetries is a difference of per-residue scores."""
    s = strata.asymmetry_summary(strata.pair_asymmetry(frame13, FT))
    assert 0.75 < s["share_by_amino_acid_mut"] < 0.9
    assert 0.7 < s["share_by_amino_acid_plm"] < 0.9


def test_symmetric_scores_have_exactly_zero_asymmetry(frame13):
    """Grantham and BLOSUM62 cannot express direction: the reference line in panel B is true."""
    for col in PROPS:
        m = frame13.drop_duplicates(["ref_aa", "aa"]).pivot(index="ref_aa", columns="aa", values=col)
        assert np.nanmax(np.abs(m.to_numpy() - m.to_numpy().T)) == 0.0, col


# ---------------------------------------------------------------------------
# Independent re-derivation (not through mutacc.strata)
# ---------------------------------------------------------------------------

def test_a_stratum_cell_equals_scipy_on_the_raw_rows(frame5, table5):
    rhos = []
    for lineage, g in frame5[(frame5.model == FT) & (frame5.nt_steps == 3)].groupby("lineage"):
        rhos.append(spearmanr(g.plm_prob, g.mut_prob).statistic)
    assert est(table5, FT, "3") == pytest.approx(np.mean(rhos), abs=1e-9)


def test_partial_equals_the_residual_method(frame5, table5):
    """Rank-residual partial correlation, computed the long way, for one lineage and stratum."""
    g = frame5[(frame5.model == FT) & (frame5.nt_steps == 2) & (frame5.lineage == G5.lineage_ids[0])]
    rx, ry, rz = (rankdata(g[c]) for c in ("plm_prob", "mut_prob", "neg_grantham"))
    design = np.column_stack([np.ones_like(rz), rz])
    res = [v - design @ np.linalg.lstsq(design, v, rcond=None)[0] for v in (rx, ry)]
    long_way = np.corrcoef(*res)[0, 1]
    short = strata.partial_from_ranks(spearmanr(rx, ry).statistic, spearmanr(rx, rz).statistic,
                                      spearmanr(ry, rz).statistic)
    assert short == pytest.approx(long_way, abs=1e-9)


def test_additive_residual_removes_a_pure_per_residue_asymmetry():
    n = 6
    first, second = np.triu_indices(n, k=1)
    score = np.array([0.0, 1.0, -2.0, 0.5, 3.0, -1.0])
    resid, share = strata._additive_residual(score[first] - score[second], first, second, n)
    assert np.abs(resid).max() < 1e-9 and share == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def test_intervals_bracket_the_estimate_and_are_reproducible(frame5, table5):
    again = strata.strata_table(frame5, [FT, BASE], PROPS, n_boot=30)
    pd.testing.assert_frame_equal(table5, again)
    assert (table5.lo <= table5.estimate + 1e-9).mean() > 0.9 and (table5.hi >= table5.estimate - 1e-9).mean() > 0.9
    assert (table5.hi > table5.lo).all()


def test_site_bootstrap_is_wider_than_the_spread_between_lineages(table5):
    """The lineages share their sites, so their spread understates the uncertainty (LINEAGE_VALIDITY Q2)."""
    for s in ("1", "2"):
        width = est(table5, BASE, s, col="hi") - est(table5, BASE, s, col="lo")
        spread = est(table5, BASE, s, col="lineage_max") - est(table5, BASE, s, col="lineage_min")
        assert width > 3 * spread


def test_no_bootstrap_gives_missing_intervals(frame5):
    t = strata.strata_table(frame5, [FT], PROPS, n_boot=0)
    assert t.lo.isna().all() and t.estimate.notna().all()


# ---------------------------------------------------------------------------
# Inputs the figure must refuse
# ---------------------------------------------------------------------------

def test_shipped_tables_match_their_guides(frame5, frame13):
    for guide, frame in ((G5, frame5), (G13, frame13)):
        assert set(frame.lineage) == set(guide.lineage_ids)
        assert set(frame.model) == {m.key for m in guide.models}
        assert not frame.duplicated(["model", "lineage", "position", "aa"]).any()
        assert set(frame.nt_steps) == {1, 2, 3}
        assert frame.plm_prob.between(0, 1).all() and (frame.mut_prob > 0).all()
        assert (frame.ref_aa != frame.aa).all()
        assert not frame.isna().any().any()


def test_shipped_tables_are_small(frame5):
    for guide in (G5, G13):
        assert guide.table_path("replacement_scores").stat().st_size < 10e6


def test_nan_probability_is_refused(frame5):
    bad = frame5.copy()
    bad.loc[bad.index[5], "plm_prob"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        strata.strata_table(bad, [FT, BASE], PROPS, n_boot=0)


def test_missing_model_and_column_are_refused(frame5):
    with pytest.raises(ValueError, match="no rows for model"):
        strata.strata_table(frame5, ["not_a_model"], PROPS, n_boot=0)
    with pytest.raises(KeyError, match="missing columns"):
        strata.strata_table(frame5.drop(columns=["blosum62"]), [FT], PROPS, n_boot=0)


def test_models_scored_on_different_rows_are_refused(frame5):
    bad = frame5.drop(frame5.index[(frame5.model == BASE)][:10])
    with pytest.raises(ValueError, match="same replacements"):
        strata.strata_table(bad, [FT, BASE], PROPS, n_boot=0)


def test_a_stratum_too_small_to_correlate_is_refused(frame5):
    thin = frame5[(frame5.nt_steps != 3) | (frame5.lineage != G5.lineage_ids[0]) | (frame5.position < 3)]
    with pytest.raises(ValueError, match="fewer than"):
        strata.strata_table(thin, [FT], PROPS, n_boot=0)


def test_empty_table_is_refused(frame5, tmp_path):
    with pytest.raises(ValueError):
        strata.strata_table(frame5.iloc[0:0], [FT], PROPS, n_boot=0)
    with pytest.raises(ValueError, match="empty"):
        io.write_values(pd.DataFrame(), tmp_path, "x")


def test_asymmetry_refuses_stops_and_synonymous_rows(frame13):
    sub = frame13[frame13.model == FT].head(500)
    with pytest.raises(ValueError, match="outside the 20"):
        strata.pair_asymmetry(sub.assign(aa="*"), FT)
    with pytest.raises(ValueError, match="synonymous"):
        strata.pair_asymmetry(sub.assign(aa=sub.ref_aa), FT)


def test_asymmetry_refuses_too_few_pairs(frame13):
    with pytest.raises(ValueError, match="observed in both directions"):
        strata.pair_asymmetry(frame13[frame13.model == FT].head(40), FT)


# ---------------------------------------------------------------------------
# Guide: the five-lineage root provenance defects are recorded, not hidden
# ---------------------------------------------------------------------------

def test_five_lineage_guide_declares_unverified_and_reconstructed_roots():
    kinds = {lin.id: lin.root_type for lin in G5.lineages}
    assert kinds == {"G.1": "unverified", "J_int": "unverified", "J.2_int": "reconstructed",
                     "J.2.4": "reconstructed", "K": "unverified"}
    assert G5.n_lineages == 5 and G13.n_lineages == 13
    assert len(G5.warnings) == 3


def test_known_defect_k_root_header_names_a_different_lineage():
    """K.nt.fa says J.2.4.1. Pinned so that fixing it forces this test (and the guide) to change."""
    k = next(l for l in G5.lineages if l.id == "K")
    assert read_fasta_headers(k.root)[0].split("|")[-1] == "J.2.4.1"
    problems = root_header_problems(G5)
    assert any(p.startswith("K:") and "J.2.4.1" in p for p in problems)


# ---------------------------------------------------------------------------
# The scripts
# ---------------------------------------------------------------------------

def test_scripts_contain_no_lineage_model_or_count_literals():
    names = {x for g in (G5, G13) for x in [m.key for m in g.models] + g.lineage_ids if len(x) > 2}
    for script in ("make_fig02.py", "make_sD_partial_correlation.py"):
        text = (REPO_ROOT / "figures" / script).read_text()
        for name in names:
            assert name not in text, f"{script} names {name!r}"
        assert not re.search(r"#[0-9a-fA-F]{6}\b", text)
        assert not re.search(r"\b(h3n2|influenza|esm|sars|thirteen|five)\b|(?<![\w.])13(?![\w.])", text, re.I), script
        assert "savefig(" not in text and "rcParams" not in text


def test_figure_2_renders_and_states_its_sources(tmp_path):
    out = tmp_path / "o"
    rc = fig2.main(["--guide", str(G5.source), "--guide-asymmetry", str(G13.source),
                    "--out", str(out), "--n-boot", "10"])
    assert rc == 0
    for suffix in ("pdf", "png"):
        assert (out / f"{fig2.STEM}.{suffix}").stat().st_size > 10_000
    values = pd.read_csv(out / f"{fig2.STEM}_values.tsv", sep="\t")
    assert set(values.panel) == {"A", "B", "C"}
    assert set(values[values.panel == "A"].n_lineages) == {G5.n_lineages}
    assert set(values[values.panel != "A"].n_lineages) == {G13.n_lineages}
    c = est_value(values, "C", FT, "rho_overall")
    assert c == pytest.approx(GOLDEN_ASYM[FT]["rho_overall"], abs=TOL)
    caption = (out / f"{fig2.STEM}_caption.md").read_text()
    assert f"{G5.n_lineages}-lineage set" in caption and f"{G13.n_lineages}-lineage set" in caption
    assert "not as a control" in caption or "not a control" in caption.replace("\n", " ")
    assert "BLOSUM62 is not independent" in " ".join(caption.split())
    for acronym in ("Spearman", "BLOcks SUbstitution Matrix", "protein language model (PLM)"):
        assert acronym in " ".join(caption.split())


def est_value(values, panel, series, measure):
    row = values[(values.panel == panel) & (values.series == series) & (values.measure == measure)]
    assert len(row) == 1
    return float(row.value.iloc[0])


def test_figure_2_single_guide_default_uses_one_set_for_every_panel(tmp_path):
    out = tmp_path / "o"
    assert fig2.main(["--guide", str(G5.source), "--out", str(out), "--n-boot", "0"]) == 0
    values = pd.read_csv(out / f"{fig2.STEM}_values.tsv", sep="\t")
    assert set(values.n_lineages) == {G5.n_lineages}
    assert f"All three panels use the {G5.n_lineages}-lineage set" in " ".join(
        (out / f"{fig2.STEM}_caption.md").read_text().split())


def test_sD_partial_correlation_renders_and_matches_the_figure_2_correlations(tmp_path):
    out = tmp_path / "o"
    assert sD.main(["--guide", str(G5.source), "--out", str(out), "--n-boot", "10"]) == 0
    assert (out / f"{sD.STEM}.png").stat().st_size > 10_000
    values = pd.read_csv(out / f"{sD.STEM}_values.tsv", sep="\t")
    # BLOSUM62 was dropped from S4: it is not plotted, so it is not in the table.
    assert set(values.control.fillna("")) == {"", "neg_grantham"}
    row = values[(values.series == FT) & (values.stratum == "3") & (values.control == "neg_grantham")]
    # 0.1850 was the mean over the five lineages. S4 now reports the one lineage the
    # guide names, which moves this by 0.0009 -- inside its own 0.090-wide interval.
    assert float(row.estimate.iloc[0]) == pytest.approx(0.184127, abs=TOL)
    assert set(values.n_lineages) == {1}
    text = " ".join((out / f"{sD.STEM}_caption.md").read_text().split())
    # The caption must still say why it is gone, so nobody reinstates it.
    assert "Grantham" in text and "has been removed" in text


def test_stratum_x_puts_all_apart_and_label_spread_keeps_order():
    xs = fig2.stratum_x(["all", "1", "2", "3"])
    assert xs["all"] == 0 and xs["1"] - xs["all"] == fig2.STRATUM_GAP and xs["3"] - xs["2"] == 1
    out = fig2.spread([0.0, 0.01, 0.02], gap=0.1)
    assert out == sorted(out) and np.diff(out).min() >= 0.1 - 1e-12
