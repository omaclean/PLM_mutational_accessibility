"""Antigenic sites: the numbering, the statistics and the shipped tables.

The antigenic comparison used to be figure S8 on its own. It is a null, so it is now one panel
of S13 (``make_sH_contact_recovery.antigenic_stratum``) with the five sites pooled into a single class and only
the zero-sum gauge measured on. These tests cover ``mutacc.antigenic``, which is where the
numbers come from; the panel itself is tested in ``test_sH_contact_recovery.py``.

The shipped tables still carry five gauges and are still tested for it -- that is provenance.
The code reads one: ``select_gauge`` drops the rest as the table loads.
"""

import numpy as np
import pandas as pd
import pytest

import make_sH_contact_recovery as sH
from mutacc import antigenic
from mutacc.guide import REPO_ROOT, load_guide

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")


@pytest.fixture(scope="module")
def sites():
    return GUIDE.read_table("antigenic_sites")


@pytest.fixture(scope="module")
def residues():
    return GUIDE.read_table("antigenic_residue_accuracy")


@pytest.fixture(scope="module")
def data():
    """What S13 draws: the zero-sum gauge, the five sites pooled into one class."""
    return sH.antigenic_stratum(GUIDE.read_table("antigenic_sites"),
                                 GUIDE.read_table("antigenic_residue_accuracy"))


@pytest.fixture(scope="module")
def summaries(sites, residues):
    """Both summaries, one generator, consumed groups first and then sites.

    Intervals depend on that order, so it is fixed here and the golden values below are tied
    to it. They are no longer the archive's: the archive's code bootstrapped all five gauges
    and threw four away, and dropping that moved the interval ends by up to about 0.002. No
    point estimate moved and no conclusion changed.
    """
    d = antigenic.annotate_residues(residues, antigenic.load_sites(sites))
    letters = sorted(set(antigenic.load_sites(sites).values()))
    rng = np.random.default_rng(antigenic.SEED)
    return {"groups": antigenic.group_summary(d, rng),
            "per_site": antigenic.site_summary(d, letters, rng), "d": d}


# ---------------------------------------------------------------------------
# Numbering
# ---------------------------------------------------------------------------

def test_shipped_sites_are_in_mature_and_precursor_numbering_sixteen_apart(sites):
    assert len(sites) == 130 and sites["ha1_position"].is_unique
    assert set(sites["ha0_position"] - sites["ha1_position"]) == {16}
    assert sites["ha1_position"].max() <= 329
    assert sites.groupby("site").size().to_dict() == {"A": 19, "B": 21, "C": 27, "D": 41, "E": 22}


@pytest.mark.parametrize("shift", [0, 15, 17, 32])
def test_a_table_shifted_once_more_or_not_at_all_is_refused(sites, shift):
    bad = sites.assign(ha0_position=sites["ha1_position"] + shift)
    with pytest.raises(ValueError, match="exactly 16"):
        antigenic.load_sites(bad)


def test_a_table_with_mixed_offsets_is_refused(sites):
    bad = sites.copy()
    bad.loc[0, "ha0_position"] += 1
    with pytest.raises(ValueError, match="exactly 16"):
        antigenic.load_sites(bad)


def test_duplicate_missing_and_absent_columns_are_refused(sites):
    with pytest.raises(ValueError, match="more than once"):
        antigenic.load_sites(pd.concat([sites, sites.iloc[:1]]))
    with pytest.raises(ValueError, match="no site letter"):
        antigenic.load_sites(sites.assign(site=[None] + list(sites["site"][1:])))
    with pytest.raises(KeyError, match="ha1_position"):
        antigenic.load_sites(sites.drop(columns="ha1_position"))
    with pytest.raises(ValueError, match="empty"):
        antigenic.load_sites(sites.iloc[0:0])


def test_joining_on_the_precursor_column_instead_would_change_the_answer(sites, residues):
    """The reason the join key matters: the two numberings select different residues."""
    # annotate_residues already reduces to the zero-sum gauge, so these are one row per residue
    a = antigenic.annotate_residues(residues, antigenic.load_sites(sites))
    b = antigenic.annotate_residues(residues, dict(zip(sites["ha0_position"], sites["site"])))
    assert a.groupby("structure")["antigenic"].sum().to_dict() == {"6WXB": 126, "7ZJ7": 125}
    assert a["antigenic"].sum() != b["antigenic"].sum() or not (a["antigenic"].values == b["antigenic"].values).all()


def test_sites_agree_with_the_corpus_backbone_numbering_at_the_k_clade_anchors():
    """The mature numbering used for the join is the numbering of the backbone the model was scored on."""
    mem = GUIDE.read_table("corpus_membership").drop_duplicates("residue").set_index("residue")["backbone_aa"]
    anchors = {144: "S", 158: "N", 160: "I", 173: "Q", 189: "K"}
    assert all(mem[r] == aa for r, aa in anchors.items())


def test_the_focal_residues_fall_in_their_literature_sites(sites):
    site = dict(zip(sites["ha1_position"], sites["site"]))
    assert site[144] == "A" and site[158] == "B" and site[160] == "B" and site[189] == "B" and site[173] == "D"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_paired_bootstrap_small_sample_is_nan_and_consumes_no_random_numbers():
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    mean, lo, hi, n = antigenic.paired_bootstrap([0.1, 0.2, np.nan, 0.3], rng)
    assert np.isnan([mean, lo, hi]).all() and n == 3
    assert rng.bit_generator.state == state


def test_paired_bootstrap_golden_values():
    d = np.arange(1, 11, dtype=float)
    mean, lo, hi, n = antigenic.paired_bootstrap(d, np.random.default_rng(0), 500)
    assert mean == 5.5 and n == 10 and 3.5 < lo < 5.5 < hi < 7.5
    again = antigenic.paired_bootstrap(d, np.random.default_rng(0), 500)
    assert again == (mean, lo, hi, n)


def test_group_summary_is_reproducible(summaries):
    """Means are the archive's exactly; the interval ends are this code's and must not drift."""
    g = summaries["groups"].set_index(["structure", "stratum"])
    a = g.loc[("6WXB", "antigenic")]
    assert a["mean_delta"] == pytest.approx(0.03206269841269842, abs=1e-12)
    assert (a["ci_lo"], a["ci_hi"]) == pytest.approx((0.0067243849206349215, 0.05925099206349204), abs=1e-9)
    assert a["n_residues"] == 126
    r = g.loc[("7ZJ7", "rest")]
    assert r["mean_delta"] == pytest.approx(0.03433005464480874, abs=1e-12)
    assert (r["ci_lo"], r["ci_hi"]) == pytest.approx((0.017161051912568306, 0.052317377049180304), abs=1e-9)
    assert "gauge" not in summaries["groups"].columns


def test_dropping_the_other_gauges_moved_no_point_estimate(summaries):
    """The mean of a group does not depend on the random stream, so it is archive-exact."""
    g = summaries["groups"].set_index(["structure", "stratum"])["mean_delta"]
    for key, archive in ((("6WXB", "antigenic"), 0.03206269841269842),
                         (("6WXB", "rest"), 0.034944),
                         (("7ZJ7", "antigenic"), 0.040171),
                         (("7ZJ7", "rest"), 0.03433005464480874)):
        assert g.loc[key] == pytest.approx(archive, abs=1e-6)


def test_per_site_shows_site_c_above_the_rest(summaries):
    """``site_summary`` is not drawn -- S13 pools the sites -- but it is still the evidence
    behind the note that site C alone is above the rest, so it stays tested."""
    s = summaries["per_site"].set_index(["structure", "site"])
    c = s.loc[("6WXB", "C")]
    assert (c["mean_delta"], c["n_residues"]) == pytest.approx((0.150, 27), abs=1e-3)
    assert c["ci_lo"] > 0
    assert s.loc[("7ZJ7", "C"), "ci_lo"] > 0
    assert "gauge" not in summaries["per_site"].columns


def test_the_headline_result_is_a_null(data):
    """The point of the panel: pooled over all five sites, neither structure moves off zero."""
    g = data["contrasts"]
    assert len(g) == 2
    assert not ((g["ci_lo"] > 0) | (g["ci_hi"] < 0)).any()
    assert (g["p_perm"] > 0.05).all()
    h = g.set_index("structure")
    assert h.loc["6WXB", "difference"] == pytest.approx(-0.0029, abs=1e-3)
    assert h.loc["7ZJ7", "difference"] == pytest.approx(0.0058, abs=1e-3)


def test_sH_contact_recovery_groups_are_exactly_the_single_gauge_summary(data, summaries):
    """S13 does nothing to the groups beyond calling ``group_summary`` on the zero-sum rows."""
    mine = data["groups"].set_index(["structure", "stratum"])
    direct = summaries["groups"].set_index(["structure", "stratum"])
    for col in ("mean_delta", "ci_lo", "ci_hi", "n_residues"):
        assert mine[col].reindex(direct.index).to_numpy() == pytest.approx(direct[col].to_numpy(), abs=1e-12)


def test_contrast_detects_a_real_difference_and_not_a_null():
    rng = np.random.default_rng(2)
    a, b = rng.normal(1.0, 0.3, 60), rng.normal(0.0, 0.3, 80)
    real = antigenic.contrast(a, b, n_boot=300, n_perm=500)
    assert real["ci_lo"] > 0 and real["p_perm"] < 0.01 and real["n_a"] == 60
    null = antigenic.contrast(rng.normal(0, 1, 50), rng.normal(0, 1, 50), n_boot=300, n_perm=500)
    assert null["ci_lo"] < 0 < null["ci_hi"] and null["p_perm"] > 0.05


def test_contrast_identical_groups_has_zero_difference_and_p_one():
    x = np.linspace(0, 1, 20)
    c = antigenic.contrast(x, x.copy(), n_boot=50, n_perm=200)
    assert c["difference"] == 0.0 and c["p_perm"] == 1.0


def test_contrast_ignores_nans_is_seeded_and_refuses_tiny_groups():
    a, b = np.arange(10.0), np.arange(10.0) + 5
    one = antigenic.contrast(np.r_[a, np.nan], b, n_boot=30, n_perm=50, seed=3)
    two = antigenic.contrast(a, b, n_boot=30, n_perm=50, seed=3)
    assert one == two and one["n_a"] == 10
    assert antigenic.contrast(a, b, n_boot=30, n_perm=50, seed=4) != two
    with pytest.raises(ValueError, match="at least 5"):
        antigenic.contrast(a[:4], b)
    with pytest.raises(ValueError, match="at least 5"):
        antigenic.contrast([np.nan] * 9, b)


def test_annotate_residues_rejects_bad_tables(sites, residues):
    site_of = antigenic.load_sites(sites)
    with pytest.raises(KeyError, match="delta"):
        antigenic.annotate_residues(residues.drop(columns="delta"), site_of)
    with pytest.raises(ValueError, match="fine-tuned minus base"):
        antigenic.annotate_residues(residues.assign(delta=residues["delta"] + 0.01), site_of)
    with pytest.raises(ValueError, match="no scored residues"):
        antigenic.annotate_residues(residues.assign(in_domain=False), site_of)
    with pytest.raises(ValueError, match="empty"):
        antigenic.annotate_residues(residues.iloc[0:0], site_of)


def test_shipped_residue_table_is_clean(residues):
    assert set(residues["structure"]) == {"6WXB", "7ZJ7"}
    assert residues["delta"].notna().all() and not residues.duplicated(["structure", "res_id", "gauge"]).any()
    assert residues.groupby("gauge")["headline"].first().sum() == 1
    assert residues[["auroc_finetune", "auroc_base"]].stack().between(0, 1).all()
    assert (residues["delta"] - (residues["auroc_finetune"] - residues["auroc_base"])).abs().max() < antigenic.ROUNDING_TOLERANCE


def test_shipped_pair_table_covers_four_strata_and_seven_gauges_with_one_headline():
    p = GUIDE.read_table("antigenic_pair_benchmark")
    assert p["gauge"].nunique() == 7 and len(p) == 28
    assert p.groupby("stratum")["n_pairs"].first().sum() == 2 * 114975        # strata partition, plus the all-pairs row
    assert p[p.headline]["gauge"].nunique() == 1
    parts = p[p.gauge == "zerosum"].set_index("antigenic_ends")["n_pairs"]
    assert parts.loc[0] + parts.loc[1] + parts.loc[2] == 114975
    assert (p["delta"] - (p["auroc_finetune"] - p["auroc_base"])).abs().max() < 2e-3


def test_manifest_lists_every_antigenic_table_with_a_relative_source():
    m = pd.read_csv(GUIDE.results_dir / "MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("file")
    for name in ("antigenic_sites", "antigenic_residue_accuracy", "antigenic_pair_benchmark"):
        row = m.loc[f"{name}.tsv.gz"]
        assert row["source_sha256_16"] and not row["source"].startswith("/")
