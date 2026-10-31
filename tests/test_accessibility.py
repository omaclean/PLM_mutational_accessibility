"""The accessibility module must reproduce the pipeline's own mut_prob, and its variants must behave."""

import numpy as np
import pytest

from mutacc import accessibility as A
from mutacc.guide import REPO_ROOT, load_guide

SPECTRA = REPO_ROOT / "data" / "spectra"
Q = A.load_spectrum(SPECTRA / "influenza_h3n2.csv")


def test_genetic_code_is_the_standard_one():
    assert A.GENETIC_CODE["ATG"] == "M" and A.GENETIC_CODE["TGG"] == "W"
    assert sum(1 for a in A.GENETIC_CODE.values() if a == "*") == 3
    assert sorted(set(A.GENETIC_CODE.values()) - {"*"}) == sorted(A.AA20)
    counts = {a: sum(1 for x in A.GENETIC_CODE.values() if x == a) for a in A.AA20}
    assert counts["L"] == counts["R"] == counts["S"] == 6 and counts["M"] == counts["W"] == 1


def test_reproduces_the_shipped_mut_prob_for_every_replacement():
    g = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
    scores = g.read_table("replacement_scores")
    one = scores[scores["model"] == g.models[0].key]
    codons = []
    for lin, grp in one.groupby("lineage", sort=False):
        lineage = next(l for l in g.lineages if l.id == lin)
        codons += list(zip(grp.index, A.reference_codons(lineage.root, grp["position"])))
    codon_of = dict(codons)
    ref = [codon_of[i] for i in one.index]
    mine = A.accessibility(ref, one["aa"], Q)
    assert np.allclose(mine, one["mut_prob"], rtol=1e-9, atol=0)


def test_steps_match_the_shipped_stratum_column():
    g = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
    scores = g.read_table("replacement_scores")
    lin = g.lineages[0]
    sub = scores[(scores.model == g.models[0].key) & (scores.lineage == lin.id)].head(600)
    codons = A.reference_codons(lin.root, sub["position"])
    mine = [A.nucleotide_steps(c, a) for c, a in zip(codons, sub["aa"])]
    assert list(sub["nt_steps"]) == mine


def test_a_one_step_route_is_more_accessible_than_a_three_step_one():
    t = A.codon_aa_table(Q)
    assert t[A.codon_index("TTT"), A.aa_index("L")] > t[A.codon_index("TTT"), A.aa_index("W")] * 100


def test_single_step_changes_are_real_routes():
    routes = A.single_step_changes("GGG", "E")        # GGG -> GAG
    assert routes == [("G", "A")]
    assert A.single_step_changes("TGG", "A") == []


def test_variants_have_the_defining_algebra():
    assert np.allclose(A.transpose(A.transpose(Q)), Q)
    s = A.symmetrise(Q)
    assert np.allclose(s, s.T) and np.allclose(A.symmetrise(s), s)
    assert np.allclose(A.complement(A.complement(Q)), Q)
    assert np.allclose(A.relabel(Q, (0, 1, 2, 3)), Q)
    assert A.relative_asymmetry(s) == pytest.approx(0.0) and 0 < A.relative_asymmetry(Q) < 1
    assert A.relative_asymmetry(A.transpose(Q)) == pytest.approx(A.relative_asymmetry(Q))


def test_complement_swaps_a_to_g_with_t_to_c():
    c = A.complement(Q)
    ag, tc = BASES = "ACGT".index("A"), "ACGT".index("T")
    assert c["ACGT".index("T"), "ACGT".index("C")] == pytest.approx(Q["ACGT".index("A"), "ACGT".index("G")])


def test_symmetrisation_leaves_the_total_rate_unchanged():
    assert A.symmetrise(Q).sum() == pytest.approx(Q.sum())


def test_shuffle_keeps_the_rates_and_is_seeded():
    a = A.shuffle(Q, np.random.default_rng(1)); b = A.shuffle(Q, np.random.default_rng(1))
    assert np.array_equal(a, b) and np.diag(a).sum() == 0
    assert sorted(a[~np.eye(4, dtype=bool)]) == sorted(Q[~np.eye(4, dtype=bool)])


def test_there_are_24_relabellings_starting_with_the_identity():
    perms = A.all_relabellings()
    assert len(perms) == 24 and perms[0] == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        A.relabel(Q, (0, 0, 1, 2))


def test_relative_matrices_need_a_scale_and_probabilities_are_refused_otherwise():
    jc = A.load_spectrum(SPECTRA / "jc69.csv")
    with pytest.raises(ValueError, match="row_sum"):
        A.codon_aa_table(jc)
    t = A.codon_aa_table(jc, row_sum=3.5e-4)
    assert np.isfinite(t).all() and (t >= 0).all()


def test_within_a_stratum_the_ranking_does_not_depend_on_the_scale():
    jc = A.load_spectrum(SPECTRA / "k80_kappa2.csv")
    a = A.codon_aa_table(jc, row_sum=3.5e-4)[A.codon_index("GGG"), A.aa_index("E")]
    b = A.codon_aa_table(jc, row_sum=7e-4)[A.codon_index("GGG"), A.aa_index("E")]
    assert b / a == pytest.approx(2.0, rel=1e-3)


@pytest.mark.parametrize("bad", ["", "AXG"])
def test_reference_codons_reject_bad_sequences(tmp_path, bad):
    f = tmp_path / "r.fa"
    f.write_text(">r\nATGAAA" + bad + "\n")
    with pytest.raises(ValueError):
        A.reference_codons(f, [3])


#: Metadata that lives beside the matrices rather than being one of them.
SPECTRA_SOURCES = SPECTRA / "spectra_sources.csv"


def shipped_spectrum_keys():
    """The keys ``spectra_sources.csv`` declares, which is the list of record."""
    import pandas as pd
    return list(pd.read_csv(SPECTRA_SOURCES)["key"])


def test_every_shipped_spectrum_loads():
    keys = shipped_spectrum_keys()
    assert keys, "spectra_sources.csv declares no spectra"
    for key in keys:
        q = A.load_spectrum(SPECTRA / f"{key}.csv")
        assert q.shape == (4, 4) and np.diag(q).sum() == 0


def test_the_matrix_files_and_the_source_table_agree():
    """Neither an undeclared matrix nor a declared one with no file.

    The glob and the table are two lists of the same thing; when they drift, a figure
    silently plots a subset and its caption counts the other list.
    """
    on_disk = {p.stem for p in SPECTRA.glob("*.csv")} - {SPECTRA_SOURCES.stem}
    declared = set(shipped_spectrum_keys())
    assert on_disk == declared, (
        f"undeclared matrix files: {sorted(on_disk - declared)}; "
        f"declared but missing: {sorted(declared - on_disk)}"
    )
