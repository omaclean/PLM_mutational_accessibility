"""S1: the spike table's models, and panel C's Grantham control -- the Figure 2D one."""

import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr

import make_sA_spike as sA
from mutacc.guide import REPO_ROOT, load_guide
from mutacc.strata import ALL, partial_from_ranks

GUIDE = load_guide(REPO_ROOT / "guides" / "sars_cov_2_spike.yaml")
EXPECTED = {"ESMC_600M_base": "base", "ESMC_600M_SC2ft": "finetune"}


def test_guide_models_are_the_esmc_pair():
    assert {m.key: m.role for m in GUIDE.models} == EXPECTED


def test_table_holds_only_those_models_with_equal_coverage():
    frame = GUIDE.read_table("replacement_scores", GUIDE.results_dir)
    counts = frame["model"].value_counts()
    assert set(counts.index) == set(EXPECTED)
    assert counts.nunique() == 1
    assert not any("ESM2" in m for m in frame["model"].unique())


def test_each_model_covers_every_lineage_and_probabilities_are_valid():
    frame = GUIDE.read_table("replacement_scores", GUIDE.results_dir)
    for _, part in frame.groupby("model"):
        assert set(part["lineage"]) == set(GUIDE.lineage_ids)
        assert part["plm_prob"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Panel C's statistical control, the same one Figure 2D carries
# ---------------------------------------------------------------------------

BASE = next(m.key for m in GUIDE.models if m.role == "base")
FT = next(m.key for m in GUIDE.models if m.role == "finetune")
CONTROL = sA.CONTROL[0]
TOL = 1e-5


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """S1 built once, with a small bootstrap: the point estimates do not depend on it."""
    out = tmp_path_factory.mktemp("sA")
    assert sA.main(["--guide", str(GUIDE.source), "--out", str(out), "--n-boot", "8"]) == 0
    stem = sA.stem_for(GUIDE)
    values = pd.read_csv(out / f"{stem}_values.tsv", sep="\t").fillna({"control": ""})
    return values, (out / f"{stem}_caption.md").read_text()


def test_values_carry_both_the_raw_and_the_grantham_controlled_rows(rendered):
    """The table used to ship with the ``control`` column empty. Panel C draws both
    markers now, so both series have to be on the record for every stratum."""
    values, _ = rendered
    strata = {ALL, "1", "2", "3"}
    for key in (BASE, FT):
        for control in ("", CONTROL):
            got = values[(values.measure == "plm_vs_access") & (values.series == key)
                         & (values.control == control)]
            assert set(got.stratum.astype(str)) == strata, (key, control)
            assert got.estimate.notna().all()
    assert set(values.control) == {"", CONTROL}


def test_the_partial_is_the_residual_correlation_not_the_raw_one(rendered):
    """Recompute one cell the long way: Spearman of the model and accessibility with
    Grantham's rank removed. A panel that quietly drew the raw number twice passes
    every other test here."""
    values, _ = rendered
    frame = GUIDE.read_table("replacement_scores", GUIDE.results_dir)
    sub = frame[(frame.model == FT) & (frame.lineage.astype(str) == str(GUIDE.lineage_ids[0]))
                & (frame.nt_steps == 3)]
    rx, ry, rz = (rankdata(sub[c]) for c in ("plm_prob", "mut_prob", CONTROL))
    expected = partial_from_ranks(spearmanr(rx, ry).statistic, spearmanr(rx, rz).statistic,
                                  spearmanr(ry, rz).statistic)
    row = values[(values.series == FT) & (values.stratum == "3") & (values.control == CONTROL)]
    # TOL absorbs the six decimal places the values file is written to.
    assert float(row.lineage_min.iloc[0]) - TOL <= expected <= float(row.lineage_max.iloc[0]) + TOL
    raw = values[(values.series == FT) & (values.stratum == "3") & (values.control == "")]
    assert float(row.estimate.iloc[0]) != pytest.approx(float(raw.estimate.iloc[0]), abs=1e-6)


def test_caption_reports_the_controlled_numbers_and_says_what_was_held_constant(rendered):
    values, caption = rendered
    text = " ".join(caption.split())
    assert "partial Spearman correlation with Grantham similarity held constant" in text
    kept = (values[(values.series == BASE) & (values.stratum == ALL) & (values.control == CONTROL)]
            .estimate.iloc[0])
    assert f"{kept:.2f}" in text, "the caption must quote the controlled estimate it plots"


def test_panel_c_draws_a_hollow_raw_point_and_a_filled_controlled_point():
    """Two markers per model, and the filled one is the partialled series. Drawn from a
    made-up table so the geometry is checked, not the data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strata = [ALL, "1", "2", "3"]
    rows = []
    for key in (BASE, FT):
        for control, value in (("", 0.60), (CONTROL, 0.30)):
            rows += [{"measure": "plm_vs_access", "series": key, "control": control,
                      "stratum": s, "estimate": value, "lo": value - 0.05, "hi": value + 0.05}
                     for s in strata]
    rows += [{"measure": "property_vs_access", "series": CONTROL, "control": "", "stratum": s,
              "estimate": 0.1, "lo": 0.05, "hi": 0.15} for s in strata]
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    sA.strata_panel(ax, table, GUIDE)
    drawn = [ln for ln in ax.get_lines() if ln.get_marker() == "o" and len(ln.get_xdata())]
    hollow = [ln for ln in drawn if ln.get_markerfacecolor() in ("white", "w")]
    filled = [ln for ln in drawn if ln not in hollow]
    assert len(hollow) == len(filled) == 2, "one raw and one controlled series per model"
    assert {round(float(y), 3) for ln in hollow for y in ln.get_ydata()} == {0.6}
    assert {round(float(y), 3) for ln in filled for y in ln.get_ydata()} == {0.3}
    # One arrow per model per stratum, drawn only because raw and controlled differ.
    arrows = [a for a in ax.texts if getattr(a, "arrow_patch", None) is not None]
    assert len(arrows) == 2 * len(strata)
    plt.close(fig)
