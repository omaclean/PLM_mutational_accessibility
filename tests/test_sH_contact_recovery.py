"""S13: contact recovery by region. Golden values are the archive's printed tables."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import make_sH_contact_recovery as sH
from mutacc.guide import REPO_ROOT, GuideError, load_guide
from mutacc import antigenic

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
SCRIPT = REPO_ROOT / "figures" / "make_sH_contact_recovery.py"


@pytest.fixture(scope="module")
def t():
    names = ["contact_strata_benchmark", "contact_residue_accuracy", "contact_positional_accuracy",
             "contact_regions", "contact_surface_vs_core", "epistasis_focal_sites"]
    return {n: GUIDE.read_table(n) for n in names}


@pytest.fixture(scope="module")
def anti():
    return sH.antigenic_stratum(GUIDE.read_table("antigenic_sites"),
                                 GUIDE.read_table("antigenic_residue_accuracy"))


def strata_row(t, structure, category):
    s = t["contact_strata_benchmark"]
    r = s[(s.structure == structure) & (s.category == category)]
    assert len(r) == 1
    return r.iloc[0]


# ---------------------------------------------------------------------------
# Golden values: the archive's strata tables (four decimals)
# ---------------------------------------------------------------------------

GOLD = {
    ("6WXB", "All Long-Range"): (114975, 1334, 0.7151, 0.7146, -0.0004, -0.0082, 0.0068, True),
    ("6WXB", "Head - Head (Antigenic)"): (35245, 751, 0.7152, 0.7347, 0.0195, 0.0059, 0.0324, False),
    ("6WXB", "Stalk Surface - Stalk Surface"): (2172, 14, 0.5887, 0.4901, -0.0986, -0.1376, -0.0611, False),
    ("6WXB", "Head Surface - Head Surface"): (2161, 28, 0.6735, 0.7409, 0.0674, -0.0163, 0.1919, True),
    ("7ZJ7", "All Long-Range"): (115440, 1011, 0.7361, 0.7531, 0.0170, 0.0084, 0.0249, False),
    ("7ZJ7", "Head - Stalk"): (58235, 142, 0.6396, 0.7174, 0.0778, 0.0361, 0.1164, False),
    ("7ZJ7", "Stalk Surface - Stalk Surface"): (2768, 6, 0.7932, 0.8505, 0.0573, -0.1243, 0.2590, True),
}


@pytest.mark.parametrize("key,gold", GOLD.items(), ids=lambda x: str(x)[:40])
def test_strata_golden_values(t, key, gold):
    r = strata_row(t, *key)
    n, c, b, f, d, lo, hi, cz = gold
    assert (r.n_pairs, r.n_contacts) == (n, c)
    assert r.auroc_base == pytest.approx(b, abs=1e-4) and r.auroc_ft == pytest.approx(f, abs=1e-4)
    assert r.delta == pytest.approx(d, abs=1e-4)
    assert r.delta_lo == pytest.approx(lo, abs=1e-4) and r.delta_hi == pytest.approx(hi, abs=1e-4)
    assert bool(r.crosses_zero) is cz


def test_shape(t):
    s = t["contact_strata_benchmark"]
    assert s.groupby("structure").size().to_dict() == {"6WXB": 11, "7ZJ7": 11}
    assert t["contact_residue_accuracy"].groupby("structure").size().to_dict() == {"6WXB": 551, "7ZJ7": 551}
    assert len(t["contact_positional_accuracy"]) == 551 and len(t["contact_surface_vs_core"]) == 11


def test_whole_structure_verdict_differs_between_structures(t):
    """The dev README: the fine-tuning verdict differs between the two structures."""
    a = strata_row(t, "6WXB", "All Long-Range")
    b = strata_row(t, "7ZJ7", "All Long-Range")
    assert bool(a.crosses_zero) and not bool(b.crosses_zero) and b.delta > 0


def test_strata_checks_pass_on_shipped_tables(t):
    gaps = sH.check_strata(t["contact_strata_benchmark"], t["contact_surface_vs_core"])
    assert gaps["delta"] < 1.1e-4 and gaps["surface_core_n_pairs"] == 0
    assert gaps["surface_core_auroc_ft"] < 5.2e-5


def test_surface_vs_core_spearman_golden(t):
    sc = t["contact_surface_vs_core"].set_index("category")
    assert sc.loc["All Long-Range", "spearman_base"] == pytest.approx(-0.48913, abs=1e-5)
    assert sc.loc["Stalk Core - Stalk Core", "spearman_ft"] == pytest.approx(0.049485, abs=1e-5)


# ---------------------------------------------------------------------------
# The residue-resampling interval (new in this repo)
# ---------------------------------------------------------------------------

def test_residue_interval_is_wider_and_contains_the_estimate(t):
    s = t["contact_strata_benchmark"]
    w_pair, w_res = s.delta_hi - s.delta_lo, s.delta_hi_resid - s.delta_lo_resid
    assert (w_res > w_pair).all(), "resampling residues should never be narrower than resampling pairs here"
    assert (s.delta_lo_resid <= s.delta + 1e-4).all() and (s.delta_hi_resid >= s.delta - 1e-4).all()
    assert (s.n_boot_resid >= 360).all() and (s.n_boot_resid <= 400).all() and (s.n_residues > 400).all()


def test_residue_interval_flags_match_intervals(t):
    s = t["contact_strata_benchmark"]
    assert (((s.delta_lo_resid < 0) & (s.delta_hi_resid > 0)) == s.crosses_zero_resid).all()
    summ = sH.strata_summary(s).set_index("structure")
    # by pairs several strata look clearly different; by residues almost none do
    assert summ.loc["6WXB", "n_better_pair"] == 2 and summ.loc["6WXB", "n_worse_pair"] == 3
    assert summ.loc["6WXB", "n_better_resid"] == 0 and summ.loc["6WXB", "n_worse_resid"] == 1
    assert summ.loc["7ZJ7", "n_better_pair"] == 4 and summ.loc["7ZJ7", "n_better_resid"] == 0
    assert (summ.n_better_pair + summ.n_worse_pair + summ.n_unclear_pair == 11).all()
    assert (summ.n_better_resid + summ.n_worse_resid + summ.n_unclear_resid == 11).all()


def test_weighted_auroc_matches_rank_auroc_and_handles_ties():
    from scipy.stats import rankdata
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("exp", REPO_ROOT / "tools" / "export_s07_s13.py")
    exp = importlib.util.module_from_spec(spec); sys.modules["exp"] = exp; spec.loader.exec_module(exp)
    rng = np.random.default_rng(3)
    v = rng.normal(size=500); c = rng.uniform(size=500) < 0.2
    _, inv = np.unique(v, return_inverse=True)
    plain = (rankdata(v)[c].mean() - (c.sum() + 1) / 2) / (~c).sum()
    assert exp.weighted_auroc(inv, inv.max() + 1, np.ones(500), c) == pytest.approx(plain, abs=1e-12)
    tied = np.round(v, 0)
    _, inv = np.unique(tied, return_inverse=True)
    plain = (rankdata(tied)[c].mean() - (c.sum() + 1) / 2) / (~c).sum()
    assert exp.weighted_auroc(inv, inv.max() + 1, np.ones(500), c) == pytest.approx(plain, abs=1e-12)
    assert np.isnan(exp.weighted_auroc(inv, inv.max() + 1, np.ones(500), np.zeros(500, bool)))
    w = np.zeros(500); w[:50] = 1
    assert np.isnan(exp.weighted_auroc(inv, inv.max() + 1, w, np.zeros(500, bool)))


def test_weighted_auroc_with_integer_weights_equals_repeating_rows():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("exp2", REPO_ROOT / "tools" / "export_s07_s13.py")
    exp = importlib.util.module_from_spec(spec); sys.modules["exp2"] = exp; spec.loader.exec_module(exp)
    rng = np.random.default_rng(5)
    v = rng.normal(size=200); c = rng.uniform(size=200) < 0.3
    w = rng.integers(0, 4, 200).astype(float)
    rep = np.repeat(np.arange(200), w.astype(int))
    _, inv = np.unique(v, return_inverse=True)
    _, inv2 = np.unique(v[rep], return_inverse=True)
    a = exp.weighted_auroc(inv, inv.max() + 1, w, c)
    b = exp.weighted_auroc(inv2, inv2.max() + 1, np.ones(len(rep)), c[rep])
    assert a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------------------
# Moving averages and numbering
# ---------------------------------------------------------------------------

def test_moving_averages_reproduce_the_archive(t):
    mine = sH.positional_series(t["contact_residue_accuracy"])
    assert sH.check_positional(mine, t["contact_positional_accuracy"]) <= 5.1e-5
    assert list(mine.columns) == list(t["contact_positional_accuracy"].columns)


def test_positional_golden_points(t):
    p = t["contact_positional_accuracy"].set_index("res_id")
    assert p.loc[200, "6WXB_ft"] == pytest.approx(0.9, abs=0.05)         # head: high everywhere
    assert np.isnan(p.loc[1, "6WXB_ft"]) or p.loc[1, "6WXB_ft"] < 0.6
    assert p["6WXB_ft"].isna().sum() == 51 and p["7ZJ7_ft"].isna().sum() == 63


def test_a_tampered_moving_average_is_caught(t):
    bad = t["contact_positional_accuracy"].copy()
    bad.loc[300, "6WXB_delta"] += 0.01
    with pytest.raises(ValueError, match="differ"):
        sH.check_positional(sH.positional_series(t["contact_residue_accuracy"]), bad)


def test_shifted_by_the_signal_peptide_is_caught(t):
    """A table shifted by 16 positions must not pass as the archive's."""
    shifted = t["contact_positional_accuracy"].copy()
    cols = [c for c in shifted.columns if c != "res_id"]
    shifted[cols] = shifted[cols].shift(16)
    with pytest.raises(ValueError):
        sH.check_positional(sH.positional_series(t["contact_residue_accuracy"]), shifted)


def test_blank_positions_must_match(t):
    bad = t["contact_positional_accuracy"].copy()
    bad.loc[0:5, "7ZJ7_base"] = 0.5
    with pytest.raises(ValueError, match="blank"):
        sH.check_positional(sH.positional_series(t["contact_residue_accuracy"]), bad)


def test_moving_average_window_edges():
    v = pd.Series(np.arange(1.0, 101.0))
    m = sH.moving_average(v)
    assert m.iloc[50] == pytest.approx(51.0)                 # centred, 25 wide
    assert m.iloc[0] == pytest.approx(np.mean(np.arange(1, 14)))   # edge: 13 residues in the window
    short = sH.moving_average(pd.Series([np.nan] * 20 + [1.0] * 5))
    assert short.iloc[:5].isna().all()                       # fewer than 8 scored: blank
    assert sH.moving_average(pd.Series([np.nan] * 40)).isna().all()


def test_regions_tile_the_sequence_and_match_site_labels(t):
    start = sH.check_regions(t["contact_regions"], t["epistasis_focal_sites"], 551)
    assert start == 330
    dom = t["contact_regions"].query("kind == 'domain'")
    assert dom.start.iloc[0] == 1 and dom.end.iloc[-1] == 551


def test_region_errors_are_caught(t):
    r, sites = t["contact_regions"].copy(), t["epistasis_focal_sites"]
    with pytest.raises(ValueError, match="1-551|run"):
        sH.check_regions(r, sites, 567)                      # a length shifted by the signal peptide
    bad = r.copy(); i = bad.index[bad.name.str.contains("fusion")][0]
    bad.loc[i, "start"] = 331
    with pytest.raises(ValueError, match="gap|overlap"):
        sH.check_regions(bad, sites, 551)
    bad = r.copy(); i = bad.index[bad.name.str.contains("fusion")][0]
    bad.loc[i, ["start"]] = 329; bad.loc[bad.index[bad.name.str.contains("stem-side")][0], "end"] = 328
    with pytest.raises(ValueError, match="HA2 position|HA1 does not end"):
        sH.check_regions(bad, sites, 551)
    off = r.copy()
    for col in ("start", "end"):
        off.loc[off.kind == "domain", col] += 16
    with pytest.raises(ValueError, match="run"):
        sH.check_regions(off, sites, 551)
    wrong = sites.copy(); wrong.loc[wrong.focal == "T328A", "region"] = "HA2"
    with pytest.raises(ValueError, match="labelled"):
        sH.check_regions(r, wrong, 551)
    wrong = sites.copy(); wrong.loc[wrong.region == "HA2", "region_pos"] = 65   # 49 + 16
    with pytest.raises(ValueError, match="HA2 position"):
        sH.check_regions(r, wrong, 551)
    with pytest.raises(ValueError, match="no domain"):
        sH.check_regions(r.assign(kind="other"), sites, 551)


def test_fusion_peptide_and_head_regions_golden(t):
    r = t["contact_regions"].set_index("name")
    assert (r.loc["HA2 fusion peptide", "start"], r.loc["HA2 fusion peptide", "end"]) == (330, 352)
    assert (r.loc["HA1 head (Cys52 to Cys277)", "start"], r.loc["HA1 head (Cys52 to Cys277)", "end"]) == (52, 277)
    pipe = t["contact_regions"].query("kind == 'strata_head_definition'").iloc[0]
    assert (pipe.start, pipe.end) == (50, 320)


# ---------------------------------------------------------------------------
# Malformed strata / residue tables
# ---------------------------------------------------------------------------

def test_strata_validation(t):
    s = t["contact_strata_benchmark"]
    with pytest.raises(ValueError, match="lacks columns"):
        sH.check_strata(s.drop(columns=["delta_lo_resid"]))
    with pytest.raises(ValueError, match="duplicate"):
        sH.check_strata(pd.concat([s, s.iloc[:1]]))
    bad = s.copy(); bad.loc[0, "auroc_ft"] = 1.2
    with pytest.raises(ValueError, match="outside"):
        sH.check_strata(bad)
    bad = s.copy(); bad.loc[0, "auroc_ft"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sH.check_strata(bad)
    bad = s.copy(); bad.loc[3, "delta_lo_resid"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sH.check_strata(bad)
    bad = s.copy(); bad.loc[2, "delta"] += 0.01
    with pytest.raises(ValueError, match="not fine-tuned minus base"):
        sH.check_strata(bad)
    bad = s.copy(); bad.loc[2, "n_contacts"] = bad.loc[2, "n_pairs"] + 1
    with pytest.raises(ValueError, match="contact counts"):
        sH.check_strata(bad)
    bad = s.copy(); bad.loc[4, "base_rate"] += 0.5
    with pytest.raises(ValueError, match="base rate"):
        sH.check_strata(bad)
    bad = s.copy(); bad.loc[6, "crosses_zero"] = not bad.loc[6, "crosses_zero"]
    with pytest.raises(ValueError, match="crosses_zero"):
        sH.check_strata(bad)
    bad = s.copy(); bad.loc[1, ["delta_lo", "delta_hi"]] = [0.05, -0.05]
    with pytest.raises(ValueError, match="low end"):
        sH.check_strata(bad)


def test_surface_core_disagreement_is_caught(t):
    s, sc = t["contact_strata_benchmark"], t["contact_surface_vs_core"].copy()
    sc.loc[2, "auroc_ft"] += 0.01
    with pytest.raises(ValueError, match="surface-vs-core"):
        sH.check_strata(s, sc)
    sc = t["contact_surface_vs_core"].copy(); sc.loc[2, "n_pairs"] += 1
    with pytest.raises(ValueError, match="n_pairs"):
        sH.check_strata(s, sc)
    sc = t["contact_surface_vs_core"].copy(); sc["auroc_ft"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        sH.check_strata(s, sc)
    sc = t["contact_surface_vs_core"].copy(); sc["category"] = "x"
    with pytest.raises(ValueError, match="shares no strata"):
        sH.check_strata(s, sc)


def test_residue_validation(t):
    r = t["contact_residue_accuracy"]
    sH.validate_residues(r)
    with pytest.raises(ValueError, match="lacks columns"):
        sH.validate_residues(r.drop(columns=["delta"]))
    bad = r.copy(); bad.loc[bad.index[10], "res_id"] = 12
    with pytest.raises(ValueError, match="1..N"):
        sH.validate_residues(bad)
    bad = r.copy(); bad["res_id"] = bad["res_id"] + 16
    with pytest.raises(ValueError, match="1..N"):
        sH.validate_residues(bad)
    bad = r.copy(); i = bad.index[(~bad.modelled) & (bad.structure == "6WXB")][0]
    bad.loc[i, "auroc_ft"] = 0.6
    with pytest.raises(ValueError, match="not resolved"):
        sH.validate_residues(bad)
    bad = r.copy(); i = bad.index[bad.auroc_ft.notna()][0]
    bad.loc[i, "auroc_base"] = -0.1
    with pytest.raises(ValueError, match="outside"):
        sH.validate_residues(bad)
    bad = r.copy(); i = bad.index[bad.auroc_ft.notna()][0]
    bad.loc[i, "delta"] += 0.1
    with pytest.raises(ValueError, match="delta"):
        sH.validate_residues(bad)
    with pytest.raises(ValueError, match="different numbers"):
        sH.validate_residues(r.iloc[:-1].assign(res_id=r.iloc[:-1].res_id))


def test_resolved_residue_counts_are_plausible(t):
    r = t["contact_residue_accuracy"]
    n = r.groupby("structure").modelled.sum().to_dict()
    assert n["6WXB"] == 551 - 51 - 0 or n["6WXB"] > 400      # residues 9-501 minus the disordered loop
    assert n["7ZJ7"] > 400
    assert not r[~r.modelled].auroc_ft.notna().any()


def test_all_nan_residue_table_gives_a_blank_series_not_a_pass(t):
    r = t["contact_residue_accuracy"].copy()
    for c in ("auroc_ft", "auroc_base", "delta"):
        r[c] = np.nan
    r["modelled"] = False
    r["n_contacts"] = 0
    series = sH.positional_series(r)
    with pytest.raises(ValueError):
        sH.check_positional(series, t["contact_positional_accuracy"])


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

def test_no_absolute_paths_in_script():
    assert not any(n in SCRIPT.read_text() for n in ("/" + "home", "/" + "software", "~" + "/"))


def test_caption_is_data_driven(t):
    text = " ".join(sH.caption(t["contact_strata_benchmark"], t["contact_residue_accuracy"],
                               t["contact_regions"], "Base", "Tuned").split())
    assert "6WXB: over all long-range pairs the AUROC goes from 0.715 to 0.715" in text
    assert "7ZJ7: over all long-range pairs the AUROC goes from 0.736 to 0.753" in text
    assert "position 330" in text and "mature numbering" in text
    assert "resamples residues" in text


def test_model_labels_come_from_the_guide():
    assert [m.role for m in GUIDE.models_with_role("base")] and len(GUIDE.models_with_role("finetune")) == 1


# ---------------------------------------------------------------------------
# The antigenic panel, merged in from the retired figure S8
# ---------------------------------------------------------------------------

def test_antigenic_panel_is_one_class_on_one_gauge(anti):
    """The sites are pooled and only zero-sum exists, so the panel asks a single question."""
    assert anti["label"] == antigenic.GAUGE_LABEL == "Zero-sum"
    assert "gauge" not in anti["groups"].columns, "the gauge is a constant, not a column"
    assert set(anti["groups"]["stratum"]) == {"antigenic", "rest"}
    assert "site" not in anti["groups"].columns
    assert anti["n_sites"] == 5 and len(anti["contrasts"]) == 2


def test_antigenic_panel_matches_the_archive(anti):
    g = anti["groups"].set_index(["structure", "stratum"])
    assert g.loc[("6WXB", "antigenic"), "mean_delta"] == pytest.approx(0.03206, abs=1e-4)
    assert g.loc[("6WXB", "rest"), "mean_delta"] == pytest.approx(0.03504, abs=1e-4)
    assert int(g.loc[("7ZJ7", "antigenic"), "n_residues"]) == 125


def test_the_gauge_is_fixed_not_chosen_from_the_table():
    """There is no headline flag to read any more: the gauge is picked in code and nowhere else."""
    assert antigenic.GAUGE == "zerosum"
    assert not hasattr(sH, "headline_gauge"), "the headline-gauge lookup is retired"
    # a table that does not carry the one gauge is refused rather than quietly falling back
    with pytest.raises(ValueError, match="no 'zerosum' rows"):
        antigenic.select_gauge(pd.DataFrame({"gauge": ["prob", "wt"], "delta": [0.0, 0.1]}))


def test_caption_reports_the_antigenic_null_and_renumbers_the_later_panels(t, anti):
    text = " ".join(sH.caption(t["contact_strata_benchmark"], t["contact_residue_accuracy"],
                               t["contact_regions"], "Base", "Tuned", anti).split())
    assert "Panel D: the same per-residue gain at the published antigenic residues" in text
    assert "Panel E: AUROC for each residue's own partners" in text and "Panel F: the same difference" in text
    assert "not specialised to the antigenic sites" in text
    assert "permutation p = 0.85" in text and "permutation p = 0.71" in text


def test_caption_would_not_call_a_real_difference_a_null(t, anti):
    """The wording follows the numbers: a planted difference must not be captioned as a null."""
    planted = dict(anti)
    c = anti["contrasts"].copy()
    c[["difference", "ci_lo", "ci_hi"]] = [0.1, 0.05, 0.15]
    planted["contrasts"] = c
    text = " ".join(sH.caption(t["contact_strata_benchmark"], t["contact_residue_accuracy"],
                               t["contact_regions"], "Base", "Tuned", planted).split())
    assert "excludes zero in 2 of 2 structures" in text
    assert "not specialised" not in text


@pytest.mark.slow
def test_script_renders(tmp_path):
    assert sH.main(["--guide", str(GUIDE.source), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "sH_contact_recovery.png").stat().st_size > 50_000
    v = pd.read_csv(tmp_path / "sH_contact_recovery_values.tsv", sep="\t")
    assert len(v) == 22
    along = pd.read_csv(tmp_path / "sH_contact_recovery_along_sequence_values.tsv", sep="\t")
    assert len(along) == 551 and along.res_id.iloc[0] == 1 and along.res_id.iloc[-1] == 551
    assert (tmp_path / "sH_contact_recovery_caption.md").exists()
    a = pd.read_csv(tmp_path / "sH_contact_recovery_antigenic_values.tsv", sep="\t")
    assert set(a["table"]) == {"group", "contrast"} and set(a["gauge"]) == {"zerosum"}


def test_a_guide_without_both_models_is_refused(tmp_path):
    import yaml
    g = yaml.safe_load(GUIDE.source.read_text())
    g["models"] = g["models"][:1]
    g["results_dir"] = str(GUIDE.results_dir.relative_to(REPO_ROOT))
    for lin in g["lineages"]:
        pass
    path = REPO_ROOT / "guides" / "_tmp_s14_test.yaml"
    path.write_text(yaml.safe_dump(g))
    try:
        with pytest.raises(GuideError, match="one base and one fine-tuned"):
            sH.main(["--guide", str(path), "--out", str(tmp_path)])
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# Exporter: bootstrap behaviour on synthetic data, manifest rows
# ---------------------------------------------------------------------------

def _exporter():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("exp3", REPO_ROOT / "tools" / "export_s07_s13.py")
    exp = importlib.util.module_from_spec(spec); sys.modules["exp3"] = exp; spec.loader.exec_module(exp)
    return exp


def _synthetic(n_res=60, seed=0, gain=0.0):
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(n_res, k=6)
    contact = rng.uniform(size=len(ii)) < 0.15
    base = rng.normal(size=len(ii)) + 0.5 * contact
    ft = base + gain * contact + rng.normal(scale=0.01, size=len(ii))
    return ft, base, contact, ii, jj, np.arange(n_res)


def test_residue_bootstrap_is_seeded_and_brackets_a_known_gain():
    exp = _exporter()
    ft, base, contact, ii, jj, idx = _synthetic(gain=1.0)
    a = exp.residue_bootstrap(ft, base, contact, ii, jj, idx, 80, np.random.default_rng(1))
    b = exp.residue_bootstrap(ft, base, contact, ii, jj, idx, 80, np.random.default_rng(1))
    assert a == b
    assert a[0] > 0 and a[1] > a[0] and a[2] == 80          # a real gain is clearly above zero
    ft0, base0, c0, i0, j0, idx0 = _synthetic(gain=0.0)
    lo, hi, _ = exp.residue_bootstrap(ft0, base0, c0, i0, j0, idx0, 80, np.random.default_rng(2))
    assert lo < 0 < hi                                       # no gain: the interval straddles zero


def test_residue_bootstrap_gives_up_when_contacts_are_missing():
    exp = _exporter()
    ft, base, contact, ii, jj, idx = _synthetic()
    lo, hi, n = exp.residue_bootstrap(ft, base, np.zeros_like(contact), ii, jj, idx, 20, np.random.default_rng(0))
    assert np.isnan(lo) and np.isnan(hi) and n == 0


def test_manifest_lists_every_new_table_with_the_right_shape():
    man = pd.read_csv(GUIDE.results_dir / "MANIFEST.tsv", sep="\t").set_index("file")
    names = ["epistasis_cells", "epistasis_null_draws", "epistasis_focal_sites", "contact_strata_benchmark",
             "contact_surface_vs_core", "contact_residue_accuracy", "contact_positional_accuracy", "contact_regions"]
    for n in names:
        row = man.loc[f"{n}.tsv.gz"]
        frame = GUIDE.read_table(n)
        assert (row.rows, row.columns) == frame.shape, n
        assert not str(row.source).startswith("/") or ".." in str(row.source), "manifest source must be relative to the archive"
        assert len(str(row.source_sha256_16)) == 16 and isinstance(row.used_for, str)
    assert not man.index.duplicated().any()
