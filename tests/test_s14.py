"""S14: lineage-set validity audit. Library statistics, shipped tables, numbering, and the figure."""

import numpy as np
import pandas as pd
import pytest

import make_s14 as s14
from mutacc import audit
from mutacc.accessibility import GENETIC_CODE, BASES, reference_codons
from mutacc.guide import REPO_ROOT, load_guide

G13 = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
G5 = load_guide(REPO_ROOT / "guides" / "h3n2_5lineage.yaml")
FT, BS = "ESM2_650M_HA80", "ESM2_650M_HA80_raw"
RATES = {(a, b): 1.0 for a in BASES for b in BASES if a != b}


@pytest.fixture(scope="module")
def scores13():
    return G13.read_table("replacement_scores")


@pytest.fixture(scope="module")
def scores5():
    return G5.read_table("replacement_scores")


@pytest.fixture(scope="module")
def observed13():
    return G13.read_table("replacement_observed")


@pytest.fixture(scope="module")
def membership():
    return G13.read_table("corpus_membership")


def row(table, key, col):
    return float(G13.read_table(table).set_index("model").loc[key, col])


# ---------------------------------------------------------------------------
# Golden values: the shipped tables against the archive's published audit numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("guide,key,auc", [(G13, FT, 0.610353), (G13, BS, 0.510454), (G5, FT, 0.608819), (G5, BS, 0.510574)])
def test_strand_b_point_estimates_match_the_archive(guide, key, auc):
    t = guide.read_table("audit_within_replacement").set_index("model")
    assert t.loc[key, "auc"] == pytest.approx(auc, abs=2e-6)
    assert t.loc[key, "n_pairs_scored"] == 54


def test_strand_b_intervals_are_clear_of_chance_for_finetune_only():
    for g in (G13, G5):
        t = g.read_table("audit_within_replacement").set_index("model")
        assert t.loc[FT, "ci_lo"] > 0.5
        assert t.loc[BS, "ci_lo"] < 0.5 < t.loc[BS, "ci_hi"]
        assert (t["n_boot"] == 2000).all()


@pytest.mark.parametrize("guide,key,rho,p", [(G13, FT, -0.0368801, 0.221878), (G13, BS, -0.0460831, 0.160784),
                                             (G5, FT, -0.0300579, 0.269573), (G5, BS, -0.0263153, 0.29887)])
def test_strand_c_matches_the_archive(guide, key, rho, p):
    t = guide.read_table("audit_persistence").set_index("model")
    assert t.loc[key, "mean_rho"] == pytest.approx(rho, abs=2e-6)
    assert t.loc[key, "p_perm"] == pytest.approx(p, abs=2e-6)
    assert t.loc[key, "null_lo"] < t.loc[key, "mean_rho"] < t.loc[key, "null_hi"]
    assert t.loc[key, "n_aa"] == 18


def test_interval_unit_means_match_figure_three_means_and_the_site_interval_is_wider():
    t = G13.read_table("audit_interval_unit").set_index("model")
    assert t.loc[FT, "mean_auroc"] == pytest.approx(0.919492, abs=1e-5)
    assert t.loc[BS, "mean_auroc"] == pytest.approx(0.772142, abs=1e-5)
    assert t.loc[BS, "width_ratio"] > 2 > t.loc[FT, "width_ratio"] > 1   # base is hit much harder than fine-tuned
    assert (t["n_lineages"] == G13.n_lineages).all() and (t["n_sites"] == 566).all()


def test_corpus_split_matches_the_archive_point_estimates():
    t = G13.read_table("audit_corpus_split").set_index("in_corpus")
    assert t.loc[False, "n_cells"] == 5129 and t.loc[False, "n_observed"] == 160
    assert t.loc[False, "auroc_finetune"] == pytest.approx(0.888877, abs=1e-5)
    assert t.loc[False, "auroc_base"] == pytest.approx(0.745993, abs=1e-5)
    assert t.loc[True, "n_cells"] == 5369 and t.loc[True, "n_observed"] == 1555
    assert (t["ci_lo"] > 0).all()                        # advantage clear of zero in both groups


@pytest.mark.slow
def test_library_reproduces_the_archives_bootstrap_exactly_with_the_archives_draw_counts(scores13, observed13, membership):
    """The archive ran 500, 500 and 600 draws; with those counts and its seed the intervals are identical."""
    r = audit.within_replacement(scores13[scores13.model == FT], n_boot=500)
    assert (r["lo"], r["hi"]) == pytest.approx((0.573319, 0.641353), abs=2e-6)
    i = audit.interval_unit(scores13[scores13.model == BS], observed13, n_boot=500)
    assert i["lineage_boot_width"] == pytest.approx(0.011977, abs=2e-6)
    assert i["site_boot_width"] == pytest.approx(0.0354408, abs=2e-6)
    s = audit.corpus_split(scores13, observed13, membership, finetune=FT, base=BS, n_boot=600).set_index("in_corpus")
    assert (s.loc[False, "ci_lo"], s.loc[False, "ci_hi"]) == pytest.approx((0.107145, 0.181554), abs=2e-6)


def test_the_archives_three_times_wider_is_a_small_sample_artefact_not_a_property():
    """At 500 draws the base ratio read 3.0; at 2000 it is about 2.5. Pin the direction, not the decimal."""
    ratio = row("audit_interval_unit", BS, "width_ratio")
    assert 2.0 < ratio < 3.2


# ---------------------------------------------------------------------------
# Self-consistency of the shipped tables with the scores they came from
# ---------------------------------------------------------------------------

def _tables(g, full):
    return s14.load_tables(g, g.results_dir, full=full)


def test_self_check_passes_on_shipped_tables():
    assert max(s14.check_point_estimates(G13, G13.results_dir, _tables(G13, True)).values()) < 1e-9
    assert max(s14.check_point_estimates(G5, G5.results_dir, _tables(G5, False)).values()) < 1e-9


def test_self_check_catches_a_tampered_table():
    t = _tables(G13, True)
    t["within"].loc[0, "auc"] += 0.001
    with pytest.raises(ValueError, match="disagree"):
        s14.check_point_estimates(G13, G13.results_dir, t)
    t = _tables(G13, True)
    t["split"].loc[0, "n_cells"] += 1
    with pytest.raises(ValueError, match="cell count"):
        s14.check_point_estimates(G13, G13.results_dir, t)
    t = _tables(G13, True)
    t["interval"].loc[1, "mean_auroc"] -= 0.01
    with pytest.raises(ValueError, match="disagree"):
        s14.check_point_estimates(G13, G13.results_dir, t)


def test_manifest_rows_exist_for_every_audit_table_and_record_seed_and_draws():
    for g in (G13, G5):
        m = pd.read_csv(g.results_dir / "MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("file")
        for name in g.tables:
            if name.startswith("audit_"):
                assert "seed=" in m.loc[g.tables[name].name, "source"], name
        assert not m["source"].str.startswith("/").any()


def test_every_shipped_audit_table_uses_only_the_guides_models():
    for g in (G13, G5):
        keys = {m.key for m in g.models}
        for name in ("audit_within_replacement", "audit_persistence"):
            assert set(g.read_table(name)["model"]) == keys


# ---------------------------------------------------------------------------
# Numbering: the +16 offset between precursor (HA0) and mature numbering is applied once
# ---------------------------------------------------------------------------

def test_membership_is_keyed_on_precursor_numbering_with_the_offset_applied_exactly_once(membership):
    assert ((membership["position"] - membership["residue"]) == audit.HA_PRECURSOR_OFFSET).all()
    assert membership["residue"].min() == 1 and membership["position"].min() == 1 + audit.HA_PRECURSOR_OFFSET


def test_backbone_residue_equals_the_scored_reference_residue_at_the_precursor_position_only(scores13, membership):
    """J.2 is the corpus backbone's own clade: 100% identity at +16, chance at +0 or +32."""
    ref = scores13.loc[scores13["lineage"] == "J.2"].groupby("position")["ref_aa"].first()
    bb = membership.drop_duplicates("position").set_index("position")["backbone_aa"]
    at = lambda shift: (ref.index.to_series().add(shift).map(bb).reindex(ref.index).values == ref.values)  # noqa: E731
    same = pd.Series(at(0), index=ref.index)
    assert same[ref.index >= 17].mean() > 0.99
    for shift in (-audit.HA_PRECURSOR_OFFSET, audit.HA_PRECURSOR_OFFSET, 1, -1):
        shifted = ref.index.to_series().add(shift).map(bb).reindex(ref.index)
        ok = (shifted.values == ref.values)[shifted.notna().values]
        assert ok.mean() < 0.2, shift


def test_the_k_clade_replacement_anchors_sit_where_the_numbering_says(membership):
    """K2N T135K S144N N158D I160K Q173R K189R T328A: mature numbering, wild type at the literal residue."""
    anchors = {2: "K", 135: "T", 144: "S", 158: "N", 160: "I", 173: "Q", 189: "K", 328: "T"}
    bb = membership.drop_duplicates("residue").set_index("residue")["backbone_aa"]
    assert all(bb[r] == aa for r, aa in anchors.items())
    assert sum(bb.get(r + audit.HA_PRECURSOR_OFFSET) == aa for r, aa in anchors.items()) < 4


def test_signal_peptide_cells_drop_out_of_the_corpus_split(scores13, observed13, membership):
    cells = audit._corpus_cells(scores13, observed13, membership, FT, BS)
    assert cells["position"].min() == audit.HA_PRECURSOR_OFFSET + 1
    assert len(cells) == 10498 == int(G13.read_table("audit_corpus_split")["n_cells"].sum())


@pytest.mark.parametrize("guide", [G13, G5], ids=["13", "5"])
def test_every_root_translates_to_the_residue_that_was_scored(guide):
    """A root file that is not the sequence the model scored would shift every codon-level number."""
    scores = guide.read_table("replacement_scores")
    ref = scores.groupby(["lineage", "position"])["ref_aa"].first()
    for lin in guide.lineages:
        sub = ref.loc[lin.id]
        codons = reference_codons(lin.root, sub.index)
        assert [GENETIC_CODE[c] for c in codons] == sub.tolist(), lin.id


# ---------------------------------------------------------------------------
# Library: small adversarial inputs
# ---------------------------------------------------------------------------

def test_rank_auroc_golden_values():
    assert audit.rank_auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert audit.rank_auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert audit.rank_auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert audit.rank_auroc([1, 2, 3, 4, 5], [1, 0, 1, 0, 0]) == pytest.approx(1 / 6)   # positives 1 and 3; only 3 beats a negative (2)
    assert np.isnan(audit.rank_auroc([1, 2, 3], [0, 0, 0]))
    assert np.isnan(audit.rank_auroc([1, 2, 3], [1, 1, 1]))
    with pytest.raises(ValueError):
        audit.rank_auroc([1, 2], [1])


def test_rank_auroc_agrees_with_the_library_metric():
    from mutacc.metrics import auroc
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=300), rng.random(300) < 0.3
    assert audit.rank_auroc(s, y) == pytest.approx(auroc(s, y), abs=1e-12)


def _within_frame(n_sites=60, seed=0, signal=0.0, lineages=("a", "b")):
    rng = np.random.default_rng(seed)
    rows = []
    for lin in lineages:
        for pos in range(1, n_sites + 1):
            for k, aa in enumerate("ACDEFGHIKLMNPQRSTVW"):
                step = 1 + (k + pos) % 3
                rows.append((lin, pos, "Y", aa, rng.normal() + signal * (step == 1), step))
    return pd.DataFrame(rows, columns=["lineage", "position", "ref_aa", "aa", "plm_prob", "nt_steps"])


def test_within_replacement_detects_a_planted_signal_and_is_chance_without_one():
    null = audit.within_replacement(_within_frame(), n_boot=0)
    planted = audit.within_replacement(_within_frame(signal=3.0), n_boot=0)
    assert abs(null["auc"] - 0.5) < 0.05
    assert planted["auc"] > 0.8
    assert planted["n_lineages"] == 2


def test_within_replacement_is_seeded_and_a_different_seed_differs():
    f = _within_frame()
    a, b = audit.within_replacement(f, n_boot=20, seed=1), audit.within_replacement(f, n_boot=20, seed=1)
    c = audit.within_replacement(f, n_boot=20, seed=2)
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"]) and (a["lo"], a["hi"]) != (c["lo"], c["hi"])


@pytest.mark.parametrize("break_it,exc,match", [
    (lambda f: f.drop(columns="nt_steps"), KeyError, "nt_steps"),
    (lambda f: f.iloc[0:0], ValueError, "empty"),
    (lambda f: f.assign(plm_prob=np.where(np.arange(len(f)) == 3, np.nan, f.plm_prob)), ValueError, "non-finite"),
    (lambda f: pd.concat([f, f.iloc[:1]]), ValueError, "twice"),
    (lambda f: f.assign(nt_steps=2), ValueError, "no replacement block"),
])
def test_within_replacement_rejects_malformed_input(break_it, exc, match):
    with pytest.raises(exc, match=match):
        audit.within_replacement(break_it(_within_frame()), n_boot=0)


def test_within_replacement_refuses_two_models_mixed_together():
    f = _within_frame()
    with pytest.raises(ValueError, match="twice"):
        audit.within_replacement(pd.concat([f.assign(model="x"), f.assign(model="y")]), n_boot=0)


def test_all_nan_bootstrap_is_an_error_not_a_blank_interval():
    with pytest.raises(ValueError, match="undefined"):
        audit._percentile_ci(np.full(5, np.nan))


def _persistence_frame(codon="GCT", n=8):
    rows = []
    for pos in range(1, n + 1):
        for k in range(19):
            rows.append(("m", "L", pos, GENETIC_CODE[codon], 0.01 * (k + 1)))
    codons = pd.DataFrame({"lineage": "L", "position": range(1, n + 1), "ref_codon": codon})
    return pd.DataFrame(rows, columns=["model", "lineage", "position", "ref_aa", "plm_prob"]), codons


def test_synonymous_fraction_golden_values():
    assert audit.synonymous_fraction("GCT", RATES) == pytest.approx(3 / 9)      # Ala: 3 of 9 changes silent at one-rate
    assert audit.synonymous_fraction("ATG", RATES) == 0.0                         # Met has no synonym
    with pytest.raises(ValueError):
        audit.synonymous_fraction("GC", RATES)
    with pytest.raises(ValueError):
        audit.synonymous_fraction("GCN", RATES)
    with pytest.raises(ValueError, match="positive rate"):
        audit.synonymous_fraction("GCT", {k: 0.0 for k in RATES})


def test_persistence_site_table_refuses_a_codon_that_does_not_translate_to_the_scored_residue():
    f, c = _persistence_frame()
    c["ref_codon"] = "TGG"
    with pytest.raises(ValueError, match="does not translate"):
        audit.persistence_site_table(f, c, RATES)


def test_persistence_site_table_refuses_missing_codons_and_wrong_alternative_counts():
    f, c = _persistence_frame()
    with pytest.raises(ValueError, match="no reference codon"):
        audit.persistence_site_table(f, c.iloc[1:], RATES)
    with pytest.raises(ValueError, match="alternatives"):
        audit.persistence_site_table(f.iloc[1:], c, RATES)


def test_persistence_with_no_eligible_residue_raises():
    f, c = _persistence_frame(n=3)     # fewer than six sites per residue
    site = audit.persistence_site_table(f, c, RATES)
    with pytest.raises(ValueError, match="enough sites"):
        audit.persistence(site, n_perm=10)


def test_persistence_is_seeded_and_finds_a_planted_negative_association():
    rng = np.random.default_rng(3)
    rows, codons = [], []
    cod = ["GCT", "GCC", "GCA", "GCG"]
    for pos in range(1, 41):
        c = cod[pos % 4]
        for k in range(19):
            rows.append(("m", "L", pos, "A", 1e-3 * (k + 1)))
        codons.append(("L", pos, c))
    f = pd.DataFrame(rows, columns=["model", "lineage", "position", "ref_aa", "plm_prob"])
    rates = {(a, b): {("T", "C"): 10.0, ("A", "G"): 4.0, ("G", "T"): 2.0}.get((a, b), 1.0) for a in BASES for b in BASES if a != b}
    site = audit.persistence_site_table(f, pd.DataFrame(codons, columns=["lineage", "position", "ref_codon"]), rates)
    site["escape"] = -site["S"] + rng.normal(scale=1e-3, size=len(site))
    s1, _ = audit.persistence(site, n_perm=200, seed=4)
    s2, _ = audit.persistence(site, n_perm=200, seed=4)
    pd.testing.assert_frame_equal(s1, s2)
    assert s1.loc[0, "mean_rho"] < -0.5 and s1.loc[0, "p_perm"] < 0.05


def _alignment(offset_gaps=3):
    backbone = "-" * offset_gaps + "MKA-LV"
    train = ["-" * offset_gaps + "MKAGLV", "-" * offset_gaps + "MRA-LV", "-" * offset_gaps + "MKAGIV"]
    return backbone, train


def test_corpus_membership_applies_the_offset_once_and_counts_per_column():
    bb, tr = _alignment()
    m = audit.corpus_membership(bb, tr)
    assert set(m["position"] - m["residue"]) == {16}
    assert m["position"].min() == 17 and m["position"].max() == 21        # five ungapped backbone residues
    k2 = m[m["residue"] == 2].set_index("aa")["corpus_count"]
    assert k2.to_dict() == {"K": 2, "R": 1}
    assert (m[m["residue"] == 4]["corpus_depth"] == 3).all()    # residue 4 is backbone column 5; column 4 is a backbone gap


def test_corpus_membership_rejects_unaligned_or_empty_corpora():
    bb, tr = _alignment()
    with pytest.raises(ValueError, match="not aligned"):
        audit.corpus_membership(bb, tr + ["MK"])
    with pytest.raises(ValueError, match="no training"):
        audit.corpus_membership(bb, [])


def test_corpus_split_refuses_a_shifted_membership_table(scores13, observed13, membership):
    shifted = membership.assign(position=membership["position"] + 1)
    with pytest.raises(ValueError, match="offset"):
        audit.corpus_split(scores13, observed13, shifted, finetune=FT, base=BS, n_boot=0)


def test_corpus_split_refuses_unknown_models_and_missing_observed(scores13, observed13, membership):
    with pytest.raises(ValueError, match="need models"):
        audit.corpus_split(scores13, observed13, membership, finetune="nope", base=BS, n_boot=0)
    with pytest.raises(ValueError, match="no observed flag"):
        audit.corpus_split(scores13, observed13[~((observed13.position == 100) & (observed13.aa == 'A'))], membership, finetune=FT, base=BS, n_boot=0)


def test_corpus_split_is_seeded(scores13, observed13, membership):
    a = audit.corpus_split(scores13, observed13, membership, finetune=FT, base=BS, n_boot=8, seed=5)
    b = audit.corpus_split(scores13, observed13, membership, finetune=FT, base=BS, n_boot=8, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_interval_unit_rejects_one_lineage_and_unmatched_observations(scores13, observed13):
    one = scores13[(scores13.model == FT) & (scores13.lineage == "K")]
    with pytest.raises(ValueError, match="at least two"):
        audit.interval_unit(one, observed13, n_boot=5)
    two = scores13[(scores13.model == FT) & scores13.lineage.isin(["K", "J"])]
    with pytest.raises(ValueError, match="no observed flag"):
        audit.interval_unit(two, observed13[observed13.lineage == "K"], n_boot=5)
    with pytest.raises(ValueError, match="twice"):
        audit.interval_unit(scores13[scores13.lineage.isin(["K", "J"])], observed13, n_boot=5)


def test_interval_unit_widths_are_positive_and_seeded(scores13, observed13):
    d = scores13[(scores13.model == BS) & scores13.lineage.isin(["K", "J", "J.2"])]
    a = audit.interval_unit(d, observed13, n_boot=15, seed=9)
    b = audit.interval_unit(d, observed13, n_boot=15, seed=9)
    assert a == b and a["lineage_boot_width"] > 0 and a["site_boot_width"] > 0


# ---------------------------------------------------------------------------
# The figure script
# ---------------------------------------------------------------------------

def test_by_role_refuses_missing_extra_and_repeated_models():
    t = G13.read_table("audit_within_replacement")
    with pytest.raises(ValueError, match="twice"):
        s14.by_role(pd.concat([t, t.iloc[:1]]), G13, "x")
    with pytest.raises(ValueError, match="not in the guide"):
        s14.by_role(t.assign(model=["a", "b"]), G13, "x")
    with pytest.raises(ValueError, match="no row"):
        s14.by_role(t.iloc[:1], G13, "x")
    assert set(s14.by_role(t, G13, "x").index) == {"finetune", "base"}


def test_a_guide_without_one_base_and_one_finetuned_model_is_refused(tmp_path):
    import yaml
    doc = yaml.safe_load(G13.source.read_text())
    doc["models"] = doc["models"][:1]
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="expected one"):
        s14.role_key(load_guide(p))


def test_draw_rejects_a_corpus_table_with_the_wrong_rows():
    t = _tables(G13, True)
    t["split"] = t["split"].iloc[:1]
    with pytest.raises(ValueError, match="exactly one row"):
        s14.draw([G5, G13], [_tables(G5, False), t], t)


@pytest.mark.slow
def test_script_renders_labels_match_data_and_caption_matches_tables(tmp_path):
    assert s14.main(["--guide", str(G13.source), "--guide-compare", str(G5.source), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "s14_lineage_set_validity.png").stat().st_size > 20_000
    assert (tmp_path / "s14_lineage_set_validity.pdf").stat().st_size > 5_000
    values = pd.read_csv(tmp_path / "s14_lineage_set_validity_values.tsv", sep="\t")
    # Trimmed to two panels on 13 Oct 2026: within-replacement AUROC, and the corpus split.
    assert set(values["panel"]) == {"A", "B"}
    a = values[(values.panel == "A") & (values.group == "13 lineages") & (values.series == "finetune")].iloc[0]
    assert a["value"] == pytest.approx(0.6104, abs=1e-4) and a["n_lineages"] == 13
    text = " ".join((tmp_path / "s14_lineage_set_validity_caption.md").read_text().split())
    assert "0.610" in text and "0.609" in text
    assert "+0.143" in text and "5 lineages and 13 lineages" in text


def test_figure_labels_come_from_the_guides_and_axes_match(tmp_path):
    fig = s14.draw([G5, G13], [_tables(G5, False), _tables(G13, True)], _tables(G13, True))
    axA, axB = fig.axes[:2]
    assert len(fig.axes) == 2, "the figure is two panels: A and the corpus split"
    assert [t.get_text() for t in axA.get_xticklabels()] == ["5 lineages", "13 lineages"]
    assert "160/5,129" in axB.get_xticklabels()[0].get_text()
    assert axA.get_ylabel() and axB.get_ylabel()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_main_refuses_to_compare_a_run_with_itself(tmp_path):
    with pytest.raises(ValueError, match="same run"):
        s14.main(["--guide", str(G13.source), "--guide-compare", str(G13.source), "--out", str(tmp_path)])


def test_a_single_set_figure_draws_without_a_comparison():
    import matplotlib.pyplot as plt
    t = _tables(G13, True)
    fig = s14.draw([G13], [t], t)
    assert [x.get_text() for x in fig.axes[0].get_xticklabels()] == ["13 lineages"]
    plt.close(fig)
