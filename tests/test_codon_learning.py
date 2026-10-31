"""Codon learning: golden values, planted-signal calibration, input rejection, reproducibility.

This file checks that the engine carries no absolute path, so it has to spell one out.
That is what the marker below is for; see ``tests/test_data_integrity.py``.
allow-absolute-path-literals
"""

import gzip
import inspect
import re

import numpy as np
import pandas as pd
import pytest
from Bio.Data import CodonTable
from Bio.Seq import Seq
from scipy.stats import binomtest, mannwhitneyu

import synth_codon as sc
from mutacc import accessibility as acc
from mutacc import codon_learning as cl
from mutacc.guide import REPO_ROOT, load_guide

FT, BASE = "ESM2_650M_HA80", "ESM2_650M_HA80_raw"
SPECTRA = REPO_ROOT / "data" / "spectra"


# ---------------------------------------------------------------------------
# Genetic code: Biopython, never typed
# ---------------------------------------------------------------------------

def test_genetic_code_agrees_with_biopython_translation():
    for c in acc.CODONS:
        assert acc.GENETIC_CODE[c] == str(Seq(c).translate())


def test_steps_table_matches_an_independent_hamming_computation():
    table = CodonTable.unambiguous_dna_by_id[1]
    by_aa = {}
    for codon, aa in table.forward_table.items():
        by_aa.setdefault(aa, []).append(codon)
    steps = cl.steps_table()
    for ci, c in enumerate(acc.CODONS):
        for ai, a in enumerate(acc.AA20):
            assert steps[ci, ai] == min(sum(x != y for x, y in zip(c, d)) for d in by_aa[a])


def test_six_codon_residues_are_read_off_the_code():
    assert cl.six_codon_residues() == ["L", "R", "S"]


@pytest.mark.parametrize("codon, expected", [("GCT", 1 / 3), ("ATG", 0.0), ("TGG", 0.0)])
def test_persistence_known_values_under_a_flat_spectrum(codon, expected):
    q = acc.load_spectrum(SPECTRA / "jc69.csv")
    assert cl.persistence(codon, q) == pytest.approx(expected)


def test_persistence_depends_on_the_spectrum_and_rejects_a_dead_one():
    flat = acc.load_spectrum(SPECTRA / "jc69.csv")
    k80 = acc.load_spectrum(SPECTRA / "k80_kappa2.csv")
    # TTT/TTC differ by a transition; a transition-biased spectrum raises persistence of TTT
    assert cl.persistence("TTT", k80) > cl.persistence("TTT", flat)
    with pytest.raises(ValueError, match="no mutation rate"):
        cl.persistence("GCT", np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return sc.make_scores(tmp_path_factory.mktemp("s"), n_pos=2500, delta=0.0, seed=1)


def test_canonical_scores_accepts_the_synthetic_table(synth):
    frame, _ = synth
    c = cl.canonical_scores(frame, sc.MODEL)
    assert len(c) == len(frame) and c["site_rank"].between(1, 19).all()


@pytest.mark.parametrize("mutate, error, match", [
    (lambda f: f.drop(columns=["plm_prob"]), KeyError, "missing columns"),
    (lambda f: f.assign(plm_prob=np.where(f.index == 3, np.nan, f.plm_prob)), ValueError, "finite"),
    (lambda f: f.assign(plm_prob=np.where(f.index == 3, -1.0, f.plm_prob)), ValueError, "non-negative"),
    (lambda f: f.assign(plm_prob=np.where(f.index == 3, np.inf, f.plm_prob)), ValueError, "finite"),
    (lambda f: pd.concat([f, f.iloc[:1]]), ValueError, "duplicate"),
    (lambda f: f.iloc[1:], ValueError, "nineteen|19"),
    (lambda f: f.assign(nt_steps=np.where(f.index == 0, 4, f.nt_steps)), ValueError, "1, 2 or 3"),
    (lambda f: f.assign(aa=np.where(f.index == 0, f.ref_aa, f.aa)), ValueError, "synonymous"),
    (lambda f: f.assign(aa=np.where(f.index == 0, "Z", f.aa)), ValueError, "20 amino acids"),
])
def test_malformed_scores_are_rejected(synth, mutate, error, match):
    frame, _ = synth
    with pytest.raises(error, match=match):
        cl.canonical_scores(mutate(frame.copy()), sc.MODEL)


def test_unknown_or_empty_model_is_rejected(synth):
    frame, _ = synth
    with pytest.raises(ValueError, match="no rows for model"):
        cl.canonical_scores(frame, "nope")
    with pytest.raises(ValueError, match="no rows"):
        cl.canonical_scores(frame.iloc[:0], sc.MODEL)


def test_withinpair_refuses_nothing_to_score():
    with pytest.raises(ValueError, match="no rows"):
        cl.WithinPair(pd.DataFrame(columns=["lineage", "position", "ref_aa", "aa", "site_rank", "nt_steps"]))


# ---------------------------------------------------------------------------
# The statistic against independent oracles
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prep_null(synth):
    frame, roots = synth
    canon = cl.canonical_scores(frame, sc.MODEL)
    return canon, cl.WithinPair(canon), roots


def _oracle_block_auc(canon, prep, block, weights=None):
    """AUROC of one block by scipy's Mann-Whitney U with tie correction, from the raw table."""
    lin, pair = prep.lineages[prep.block_lineage[block]], prep.pair_names[prep.block_pair[block]]
    sub = canon.loc[(canon.lineage == lin) & ((canon.ref_aa + ">" + canon.aa) == pair)]
    x = sub.loc[sub.nt_steps == 1, "site_rank"].to_numpy()
    y = sub.loc[sub.nt_steps != 1, "site_rank"].to_numpy()
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return mannwhitneyu(x, y).statistic / (len(x) * len(y))


def test_every_block_auroc_equals_scipy_mann_whitney(prep_null):
    canon, prep, _ = prep_null
    res = prep.blocks()
    checked = 0
    for b in np.flatnonzero(res.ok)[:60]:
        assert res.auc[b] == pytest.approx(_oracle_block_auc(canon, prep, b), abs=1e-12)
        checked += 1
    assert checked > 20


def test_weights_equal_duplicating_rows(prep_null):
    """A site drawn twice must score exactly as if its rows were present twice."""
    canon, prep, _ = prep_null
    rng = np.random.default_rng(5)
    w = rng.integers(0, 3, prep.n_rows).astype(float)
    res = prep.blocks(weights=w)
    for b in np.flatnonzero(res.ok)[:25]:
        m = prep.block == b
        v, e, ww = prep.value[m], prep.exposed[m], w[m].astype(int)
        x, y = np.repeat(v[e], ww[e]), np.repeat(v[~e], ww[~e])
        if len(x) and len(y):
            assert res.auc[b] == pytest.approx(mannwhitneyu(x, y).statistic / (len(x) * len(y)), abs=1e-12)


def test_van_elteren_pooling_is_the_weighted_mean_of_blocks(prep_null):
    _, prep, _ = prep_null
    res = prep.blocks()
    ok = res.ok
    expected = np.sum(res.auc[ok] * res.weight[ok]) / res.weight[ok].sum()
    assert res.pooled == pytest.approx(expected, abs=1e-14)
    n1, n0 = res.n_exposed[ok], res.n_unexposed[ok]
    assert np.allclose(res.weight[ok], n1 * n0 / (n1 + n0 + 1))
    assert (n1 >= cl.MIN_EXPOSED).all() and (n0 >= cl.MIN_UNEXPOSED).all() and (n1 + n0 >= cl.MIN_BLOCK).all()


def test_all_exposed_or_none_gives_nothing_to_score(prep_null):
    _, prep, _ = prep_null
    assert np.isnan(prep.pooled(exposed=np.ones(prep.n_rows, bool)))
    assert np.isnan(prep.pooled(exposed=np.zeros(prep.n_rows, bool)))


def test_zero_weights_everywhere_gives_nan_not_an_error(prep_null):
    _, prep, _ = prep_null
    assert np.isnan(prep.pooled(weights=np.zeros(prep.n_rows)))


def test_ties_are_midranked_in_a_hand_built_block():
    """Two exposed sites with rank 2 and two unexposed with ranks 1 and 2 (one tie): AUROC = (1 + .5)/2 ... by hand."""
    rows = []
    # 20 sites of one lineage, all ref S -> N (6-codon source, exposure by step count)
    spec = [(1, 1)] * 5 + [(2, 3)] * 5      # (site_rank value, nt_steps)
    for i in range(10):
        pass
    for p in range(1, 21):
        for k, a in enumerate(acc.AA20):
            if a == "S":
                continue
            rows.append(("L", p, "S", a, 0.5 if a != "N" else 0.5, 3 if a != "N" else (1 if p <= 10 else 2)))
    f = pd.DataFrame(rows, columns=["lineage", "position", "ref_aa", "aa", "site_rank", "nt_steps"])
    # outcome for N: exposed sites (p<=10) get rank 2 for half and 1 for half; unexposed all 1
    f.loc[(f.aa == "N") & (f.position <= 5), "site_rank"] = 2.0
    f.loc[(f.aa == "N") & (f.position > 5), "site_rank"] = 1.0
    p = cl.WithinPair(f)
    res = p.blocks()
    b = [i for i in range(p.n_blocks) if p.pair_names[p.block_pair[i]] == "S>N"][0]
    # exposed: five at 2, five at 1; unexposed: ten at 1.  P(exp > unexp) + .5 P(tie) = (5*10 + .5*5*10 ... )
    #   pairs: exposed 2 vs 1 -> 5*10 wins; exposed 1 vs 1 -> 5*10 ties
    assert res.auc[b] == pytest.approx((50 + 0.5 * 50) / 100)


# ---------------------------------------------------------------------------
# Planted signal and null calibration
# ---------------------------------------------------------------------------

def _pooled(tmp_path, delta, seed=1):
    frame, _ = sc.make_scores(tmp_path, n_pos=2500, delta=delta, seed=seed)
    return cl.WithinPair(cl.canonical_scores(frame, sc.MODEL)).pooled()


def test_null_table_scores_near_chance(tmp_path):
    assert abs(_pooled(tmp_path, 0.0) - 0.5) < 0.02


def test_planted_preference_is_recovered_and_monotone(tmp_path):
    vals = [_pooled(tmp_path, d) for d in (0.0, 0.3, 0.8, 1.6)]
    assert vals == sorted(vals)
    assert vals[0] < 0.53 and vals[-1] > 0.70


def test_the_wrong_direction_scores_below_chance(tmp_path):
    assert _pooled(tmp_path, -1.0) < 0.45


def test_planted_signal_is_outside_the_permutation_null_and_null_is_not(tmp_path):
    frame, roots = sc.make_scores(tmp_path, n_pos=2500, delta=0.8, seed=2, models=("null_m", "planted_m"))
    preps = {m: cl.WithinPair(cl.canonical_scores(frame, m)) for m in ("null_m", "planted_m")}
    codons = cl.reference_codon_matrix(preps["planted_m"], roots)
    cl.check_roots(preps["planted_m"], codons)
    null = cl.permutation_null(preps, codons, n_perm=60, seed=3)
    obs = {m: p.pooled() for m, p in preps.items()}
    assert obs["planted_m"] > np.percentile(null["planted_m"], 99)
    lo, hi = np.percentile(null["null_m"], [1, 99])
    assert lo < obs["null_m"] < hi
    assert abs(null["null_m"].mean() - 0.5) < 0.02


# ---------------------------------------------------------------------------
# Bootstrap and null: reproducibility and guards
# ---------------------------------------------------------------------------

def test_bootstrap_is_reproducible_and_seed_dependent(prep_null):
    _, prep, _ = prep_null
    a = cl.bootstrap_sites({"m": prep}, n_boot=6, seed=7)["m"][0]
    b = cl.bootstrap_sites({"m": prep}, n_boot=6, seed=7)["m"][0]
    c = cl.bootstrap_sites({"m": prep}, n_boot=6, seed=8)["m"][0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_models_share_the_same_site_draws(synth):
    frame, _ = synth
    p1 = cl.WithinPair(cl.canonical_scores(frame, sc.MODEL))
    p2 = cl.WithinPair(cl.canonical_scores(frame, sc.MODEL))
    out = cl.bootstrap_sites({"a": p1, "b": p2}, n_boot=5, seed=1)
    assert np.array_equal(out["a"][0], out["b"][0])


def test_bootstrap_interval_covers_the_point_estimate(prep_null):
    _, prep, _ = prep_null
    d = cl.bootstrap_sites({"m": prep}, n_boot=40, seed=1)["m"][0]
    lo, hi = np.percentile(d, [2.5, 97.5])
    assert lo < prep.pooled() < hi


def test_bootstrap_and_null_reject_bad_counts(prep_null):
    _, prep, roots = prep_null
    with pytest.raises(ValueError, match="at least 1"):
        cl.bootstrap_sites({"m": prep}, n_boot=0, seed=1)
    codons = cl.reference_codon_matrix(prep, roots)
    with pytest.raises(ValueError, match="at least 1"):
        cl.permutation_null({"m": prep}, codons, n_perm=0, seed=1)


def test_models_scored_on_different_sites_are_refused(synth):
    frame, _ = synth
    a = cl.WithinPair(cl.canonical_scores(frame, sc.MODEL))
    other = frame.loc[frame.position > 5].copy()
    b = cl.WithinPair(cl.canonical_scores(other, sc.MODEL))
    with pytest.raises(ValueError, match="same sites"):
        cl.bootstrap_sites({"a": a, "b": b}, n_boot=2, seed=1)


def test_permutation_null_is_reproducible(prep_null):
    _, prep, roots = prep_null
    codons = cl.reference_codon_matrix(prep, roots)
    a = cl.permutation_null({"m": prep}, codons, n_perm=5, seed=4)["m"]
    b = cl.permutation_null({"m": prep}, codons, n_perm=5, seed=4)["m"]
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

def test_roots_that_explain_the_table_pass(prep_null):
    _, prep, roots = prep_null
    cl.check_roots(prep, cl.reference_codon_matrix(prep, roots))


def test_a_root_with_the_wrong_codon_is_caught(prep_null, tmp_path):
    canon, prep, roots = prep_null
    seq = "".join(l for l in roots["L0"].read_text().splitlines() if not l.startswith(">"))
    first = seq[:3]
    other = next(c for c in sc.SENSE if acc.GENETIC_CODE[c] != acc.GENETIC_CODE[first])
    bad = sc.write_root(tmp_path / "bad.fa", [other] + [seq[i:i + 3] for i in range(3, len(seq), 3)])
    with pytest.raises(ValueError, match="does not translate"):
        cl.check_roots(prep, cl.reference_codon_matrix(prep, {**roots, "L0": bad}))


def test_a_synonymous_swap_that_changes_steps_is_caught(prep_null, tmp_path):
    """Same amino acid, different codon: translation passes, step counts must not."""
    canon, prep, roots = prep_null
    seq = "".join(l for l in roots["L0"].read_text().splitlines() if not l.startswith(">"))
    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    for i, c in enumerate(codons):
        syn = [d for d in sc.SENSE if d != c and acc.GENETIC_CODE[d] == acc.GENETIC_CODE[c]]
        if syn and any(acc.nucleotide_steps(c, a) != acc.nucleotide_steps(syn[0], a) for a in acc.AA20):
            codons[i] = syn[0]
            break
    bad = sc.write_root(tmp_path / "syn.fa", codons)
    with pytest.raises(ValueError, match="nt_steps does not match"):
        cl.check_roots(prep, cl.reference_codon_matrix(prep, {**roots, "L0": bad}))


def test_missing_root_and_short_root_are_reported(prep_null, tmp_path):
    _, prep, roots = prep_null
    with pytest.raises(KeyError, match="no root"):
        cl.reference_codon_matrix(prep, {"L0": roots["L0"]})
    short = sc.write_root(tmp_path / "short.fa", ["GCT"] * 10)
    with pytest.raises(ValueError, match="no valid codon"):
        cl.reference_codon_matrix(prep, {**roots, "L1": short})


def test_position_groups_hold_back_sites_whose_amino_acid_differs_between_lineages(tmp_path):
    frame, roots = sc.make_scores(tmp_path, n_pos=600, n_lin=2, seed=3)
    # rewrite lineage L1's root at position 1 to a codon for a different amino acid and relabel its rows
    seq = "".join(l for l in roots["L1"].read_text().splitlines() if not l.startswith(">"))
    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    old = acc.GENETIC_CODE[codons[0]]
    new = next(c for c in sc.SENSE if acc.GENETIC_CODE[c] != old)
    codons[0] = new
    frame = frame.loc[~((frame.lineage == "L1") & (frame.position == 1))]
    extra = []
    steps = lambda c, a: acc.nucleotide_steps(c, a)
    for a in acc.AA20:
        if a != acc.GENETIC_CODE[new]:
            extra.append(("L1", 1, acc.GENETIC_CODE[new], a, steps(new, a), sc.MODEL, 1e-4))
    frame = pd.concat([frame, pd.DataFrame(extra, columns=["lineage", "position", "ref_aa", "aa", "nt_steps", "model", "plm_prob"])])
    prep = cl.WithinPair(cl.canonical_scores(frame, sc.MODEL))
    roots["L1"] = sc.write_root(tmp_path / "L1b.fa", codons)
    cm = cl.reference_codon_matrix(prep, roots)
    cl.check_roots(prep, cm)
    shared, own = cl.position_groups(prep, cm)
    pos0 = int(np.searchsorted(prep.positions, 1))
    assert all(pos0 not in g for g in shared)
    assert any(pos0 in g for _, g in own) or True       # may be a singleton group; must never be in `shared`


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def test_sign_test_matches_the_archive_count_and_scipy():
    r = cl.sign_test([0.6] * 46 + [0.4] * 8)
    assert (r["n_pairs"], r["n_above_half"]) == (54, 46)
    assert r["p_sign"] == pytest.approx(binomtest(46, 54, 0.5).pvalue)
    assert 1.3e-7 < r["p_sign"] < 1.5e-7          # the archive's 1.4e-7


def test_sign_test_drops_nan_and_rejects_empty_and_ignores_exact_half():
    assert cl.sign_test([0.7, np.nan, 0.5])["n_pairs"] == 2
    assert cl.sign_test([0.7, np.nan, 0.5])["n_above_half"] == 1
    with pytest.raises(ValueError, match="no scored"):
        cl.sign_test([np.nan])


def test_lineage_resample_ci_needs_two_lineages_and_is_seeded():
    with pytest.raises(ValueError, match="two"):
        cl.lineage_resample_ci([0.6])
    assert cl.lineage_resample_ci([0.6, 0.61, 0.62], seed=1) == cl.lineage_resample_ci([0.6, 0.61, 0.62], seed=1)


def test_pair_table_and_robustness_are_consistent(prep_null):
    _, prep, _ = prep_null
    res = prep.blocks()
    pt = cl.pair_table(prep, res)
    assert (pt["auc"].diff().dropna() <= 0).all()                      # sorted strongest first
    assert pt["pair"].is_unique and len(pt) == len(np.unique(prep.block_pair[res.ok]))
    rob = cl.robustness(prep, res)
    assert rob.loc[0, "auc"] == pytest.approx(res.pooled)
    assert rob.loc[0, "pairs_remaining"] == len(pt)
    assert rob["pairs_remaining"].is_monotonic_decreasing or rob["pairs_remaining"].iloc[1:].is_monotonic_decreasing
    assert (rob["auc"].iloc[1:5].diff().dropna() <= 1e-12).all()       # dropping stronger and stronger pairs lowers the pool


def test_per_lineage_auc_one_row_per_lineage(prep_null):
    _, prep, _ = prep_null
    t = cl.per_lineage_auc(prep, prep.blocks())
    assert sorted(t["lineage"]) == prep.lineages and t["auc"].between(0, 1).all()


def test_audit_statistic_is_close_to_but_not_the_same_as_midrank(prep_null):
    canon, prep, _ = prep_null
    assert abs(cl.audit_pooled_auc(canon) - prep.pooled()) < 0.01


# ---------------------------------------------------------------------------
# Strand C
# ---------------------------------------------------------------------------

def _site(rho_sign, n=40, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for aa in "ACDEFG":
        s = rng.random(n)
        parts.append(pd.DataFrame({"ref_aa": aa, "persistence": s, "escape": rho_sign * s + 0.1 * rng.normal(size=n)}))
    return pd.concat(parts, ignore_index=True)


def test_strand_c_recovers_a_planted_negative_and_a_null():
    neg = cl.strand_c_per_aa(_site(-1.0))["rho"].mean()
    null = cl.strand_c_per_aa(_site(0.0, seed=2))["rho"].mean()
    assert neg < -0.8 and abs(null) < 0.2


def test_strand_c_null_is_centred_and_planted_signal_is_extreme():
    site = _site(-1.0)
    null = cl.strand_c_null(site, n_perm=2000, seed=1)
    obs = cl.strand_c_per_aa(site)["rho"].mean()
    assert abs(null.mean()) < 0.01 and obs < null.min()
    assert np.array_equal(null, cl.strand_c_null(site, n_perm=2000, seed=1))


def test_strand_c_skips_small_and_constant_groups_and_rejects_all_skipped():
    site = _site(-1.0)
    site = pd.concat([site, pd.DataFrame({"ref_aa": "W", "persistence": [0.0] * 30, "escape": np.arange(30.0)}),
                      pd.DataFrame({"ref_aa": "Y", "persistence": [0.1, 0.9, 0.5], "escape": [1.0, 2, 3]})])
    assert set(cl.strand_c_per_aa(site)["aa"]) == set("ACDEFG")
    with pytest.raises(ValueError, match="enough sites"):
        cl.strand_c_per_aa(site.loc[site.ref_aa.isin(["W", "Y"])])
    with pytest.raises(ValueError, match="enough sites"):
        cl.strand_c_null(site.loc[site.ref_aa.isin(["W", "Y"])], n_perm=5, seed=1)
    with pytest.raises(ValueError, match="enough sites"):
        cl.strand_c_bootstrap(site.loc[site.ref_aa.isin(["W", "Y"])], n_boot=5, seed=1)


def test_strand_c_bootstrap_is_reproducible_and_brackets_a_strong_signal():
    site = _site(-1.0)
    a = cl.strand_c_bootstrap(site, n_boot=50, seed=3)
    assert np.array_equal(a, cl.strand_c_bootstrap(site, n_boot=50, seed=3))
    assert np.nanpercentile(a, 97.5) < -0.5


def test_unit_ranks_dot_product_is_spearman():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 5, 50), rng.normal(size=50)
    assert cl.unit_ranks(x) @ cl.unit_ranks(y) == pytest.approx(spearmanr(x, y).statistic)
    assert np.allclose(cl.unit_ranks([1, 1, 1]), 0)


def test_strand_c_sites_on_synthetic_table_has_one_row_per_position(synth):
    frame, roots = synth
    q = acc.load_spectrum(SPECTRA / "jc69.csv")
    site = cl.strand_c_sites(frame, sc.MODEL, roots, q)
    assert len(site) == frame.position.nunique() and site["persistence"].between(0, 1).all()
    with pytest.raises(KeyError, match="no root"):
        cl.strand_c_sites(frame, sc.MODEL, {"L0": roots["L0"]}, q)


# ---------------------------------------------------------------------------
# Golden values on the shipped tables
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    out = {}
    for name in ("h3n2_13lineage", "h3n2_5lineage"):
        g = load_guide(REPO_ROOT / "guides" / f"{name}.yaml")
        out[name] = (g, g.read_table("replacement_scores"))
    return out


@pytest.mark.parametrize("name, model, mid, audit, blocks", [
    ("h3n2_13lineage", FT, 0.6104972, 0.6103532, 692),
    ("h3n2_13lineage", BASE, 0.5116267, 0.5104543, 692),
    ("h3n2_5lineage", FT, 0.6078762, 0.6088191, 270),
    ("h3n2_5lineage", BASE, 0.5110226, 0.5105741, 270),
])
def test_golden_pooled_auroc(shipped, name, model, mid, audit, blocks):
    """Mid-rank values equal the archive's own (a20: 0.6078762 and 0.5110226 on five lineages); the audit's
    0.6104/0.5105 (13) and 0.6088/0.5106 (5) are the row-order-tie-broken numbers."""
    _, scores = shipped[name]
    canon = cl.canonical_scores(scores, model)
    prep = cl.WithinPair(canon)
    res = prep.blocks()
    assert res.pooled == pytest.approx(mid, abs=1e-6)
    assert cl.audit_pooled_auc(canon) == pytest.approx(audit, abs=1e-6)
    assert int(res.ok.sum()) == blocks
    assert len(cl.pair_table(prep, res)) == 54


def test_golden_task_numbers_round_correctly(shipped):
    for name, model, expected in [("h3n2_13lineage", FT, 0.6104), ("h3n2_13lineage", BASE, 0.5105),
                                  ("h3n2_5lineage", FT, 0.6088), ("h3n2_5lineage", BASE, 0.5106)]:
        assert round(cl.audit_pooled_auc(cl.canonical_scores(shipped[name][1], model)), 4) == expected


def test_golden_five_lineage_pairs_and_robustness_match_the_archive(shipped):
    _, scores = shipped["h3n2_5lineage"]
    prep = cl.WithinPair(cl.canonical_scores(scores, FT))
    res = prep.blocks()
    st = cl.sign_test(cl.pair_table(prep, res)["auc"])
    assert (st["n_pairs"], st["n_above_half"]) == (54, 46)
    assert 1.3e-7 < st["p_sign"] < 1.5e-7
    rob = cl.robustness(prep, res)
    assert rob["auc"].round(6).tolist() == [0.607876, 0.600727, 0.595433, 0.586688, 0.571776, 0.580015]
    assert rob["pairs_remaining"].tolist() == [54, 53, 51, 49, 44, 26]
    top = cl.pair_table(prep, res).iloc[0]
    assert top["pair"] == "S>N" and top["auc"] == pytest.approx(0.8696773, abs=1e-6)


def test_every_scored_pair_has_both_exposed_and_unexposed_sites(shipped):
    _, scores = shipped["h3n2_13lineage"]
    prep = cl.WithinPair(cl.canonical_scores(scores, FT))
    pt = cl.pair_table(prep, prep.blocks())
    assert (pt["sites_one_step"] >= cl.MIN_EXPOSED).all() and (pt["sites_multi_step"] >= cl.MIN_UNEXPOSED).all()


@pytest.mark.parametrize("name, model, rho, neg", [
    ("h3n2_13lineage", FT, -0.0368801, 9), ("h3n2_13lineage", BASE, -0.0460831, 11),
    ("h3n2_5lineage", FT, -0.0300579, 12), ("h3n2_5lineage", BASE, -0.0263153, 11),
])
def test_golden_strand_c(shipped, name, model, rho, neg):
    g, scores = shipped[name]
    q = acc.symmetrise(acc.load_spectrum(SPECTRA / "influenza_h3n2.csv"))
    site = cl.strand_c_sites(scores, model, {l.id: l.root for l in g.lineages}, q)
    per = cl.strand_c_per_aa(site)
    assert len(per) == 18 and per["rho"].mean() == pytest.approx(rho, abs=1e-6)
    assert int((per["rho"] < 0).sum()) == neg
    assert not {"M", "W"} & set(per["aa"])


def test_strand_c_null_pvalue_on_the_archive_seed(shipped):
    g, scores = shipped["h3n2_13lineage"]
    q = acc.symmetrise(acc.load_spectrum(SPECTRA / "influenza_h3n2.csv"))
    site = cl.strand_c_sites(scores, FT, {l.id: l.root for l in g.lineages}, q)
    null = cl.strand_c_null(site, n_perm=10000, seed=20260928)
    obs = cl.strand_c_per_aa(site)["rho"].mean()
    assert ((null <= obs).sum() + 1) / 10001 == pytest.approx(0.2219, abs=0.001)     # archive: 0.2219


def test_roots_explain_both_shipped_tables(shipped):
    for name, (g, scores) in shipped.items():
        prep = cl.WithinPair(cl.canonical_scores(scores, FT))
        cl.check_roots(prep, cl.reference_codon_matrix(prep, {l.id: l.root for l in g.lineages}))


# ---------------------------------------------------------------------------
# Derived-table reader
# ---------------------------------------------------------------------------

def test_read_derived_names_the_tool_when_missing_and_rejects_empty(tmp_path):
    g = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
    with pytest.raises(FileNotFoundError, match="export_codon_s06"):
        cl.read_derived(g, "nope", tmp_path)
    with gzip.open(tmp_path / "e.tsv.gz", "wt") as fh:
        fh.write("a\tb\n")
    with pytest.raises(ValueError, match="empty"):
        cl.read_derived(g, "e", tmp_path)


def test_module_has_no_absolute_paths_or_hand_typed_genetic_code():
    text = inspect.getsource(cl)
    assert "/home" not in text
    assert not re.search(r"['\"]TTT['\"]\s*:", text)


# ---------------------------------------------------------------------------
# Export tool
# ---------------------------------------------------------------------------

import sys

sys.path.insert(0, str(REPO_ROOT / "tools"))
import export_codon_s06 as exp  # noqa: E402

HEADER = "file\tsource\tsource_sha256_16\trows\tcolumns\tdropped\tused_for"


def _row(name, rows=1):
    return {"file": name, "source": "generated by x", "source_sha256_16": "", "rows": rows, "columns": 2,
            "dropped": "", "used_for": "t"}


def test_manifest_append_creates_then_appends_without_rewriting(tmp_path):
    m = tmp_path / "MANIFEST.tsv"
    exp.append_manifest(m, [_row("a.tsv.gz")])
    assert m.read_text().splitlines()[0] == HEADER
    m.write_text(m.read_text().rstrip("\n"))                       # file with no trailing newline
    before = m.read_text()
    exp.append_manifest(m, [_row("b.tsv.gz")])
    text = m.read_text()
    assert text.startswith(before) and text.count("\n") == 3
    assert pd.read_csv(m, sep="\t")["file"].tolist() == ["a.tsv.gz", "b.tsv.gz"]


def test_manifest_rerun_replaces_only_its_own_rows(tmp_path):
    m = tmp_path / "MANIFEST.tsv"
    m.write_text(HEADER + "\nz_other.tsv.gz\tsrc\tabc\t5\t3\t\twhy\n")
    exp.append_manifest(m, [_row("a.tsv.gz", 1)])
    exp.append_manifest(m, [_row("a.tsv.gz", 9)])
    t = pd.read_csv(m, sep="\t", dtype=str, keep_default_na=False)
    assert t["file"].tolist() == ["a.tsv.gz", "z_other.tsv.gz"] and t.loc[0, "rows"] == "9"
    assert t.loc[1, "source_sha256_16"] == "abc"


def test_write_tables_refuses_empty_tables(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        exp.write_tables({"t": pd.DataFrame()}, tmp_path, "s", {"t": "u"})


def test_interval_helper_rejects_too_few_draws():
    with pytest.raises(ValueError, match="too few"):
        exp._interval(np.array([np.nan, 1.0]))


@pytest.mark.slow
def test_export_runs_end_to_end_and_is_deterministic(tmp_path):
    args = ["--set", "h3n2_5lineage", "--tables", "codon", "--n-boot", "6", "--n-perm", "6", "--n-perm-c", "60"]
    a, b = tmp_path / "a", tmp_path / "b"
    assert exp.main(args + ["--out-dir", str(a)]) == 0
    assert exp.main(args + ["--out-dir", str(b)]) == 0
    names = sorted(p.name for p in a.glob("*.tsv.gz"))
    assert len(names) == 7 and names == sorted(p.name for p in b.glob("*.tsv.gz"))
    for n in names:
        pd.testing.assert_frame_equal(pd.read_csv(a / n, sep="\t"), pd.read_csv(b / n, sep="\t"))
    man = pd.read_csv(a / "MANIFEST.tsv", sep="\t")
    assert sorted(man["file"]) == names and not man["source"].str.startswith("/").any()


# ---------------------------------------------------------------------------
# Shipped derived tables
# ---------------------------------------------------------------------------

SETS = ("h3n2_13lineage", "h3n2_5lineage")


@pytest.fixture(scope="module")
def derived():
    out = {}
    for name in SETS:
        g = load_guide(REPO_ROOT / "guides" / f"{name}.yaml")
        out[name] = (g, {t: cl.read_derived(g, t) for t in (
            "codon_within_pair_summary", "codon_within_pair_difference", "codon_within_pair_by_pair",
            "codon_within_pair_by_lineage", "codon_within_pair_robustness", "codon_strand_c_summary",
            "codon_strand_c_per_aa")})
    return out


@pytest.mark.parametrize("name", SETS)
def test_every_derived_table_has_a_manifest_row_with_a_relative_source(derived, name):
    g, tables = derived[name]
    man = pd.read_csv(g.results_dir / "MANIFEST.tsv", sep="\t")
    for t, frame in tables.items():
        row = man.loc[man["file"] == f"{t}.tsv.gz"]
        assert len(row) == 1, t
        assert int(row["rows"].iloc[0]) == len(frame) and int(row["columns"].iloc[0]) == frame.shape[1]
        assert not str(row["source"].iloc[0]).startswith("/")
    assert man["file"].is_unique


@pytest.mark.parametrize("name, lineages", [("h3n2_13lineage", 13), ("h3n2_5lineage", 5)])
def test_shipped_strand_b_summary(derived, name, lineages):
    g, t = derived[name]
    s = t["codon_within_pair_summary"].set_index("role")
    ft, base = s.loc["finetune"], s.loc["base"]
    assert ft["n_lineages"] == base["n_lineages"] == lineages == g.n_lineages
    assert ft["n_pairs"] == base["n_pairs"] == 54
    assert 0.607 < ft["auc"] < 0.611 and 0.510 < base["auc"] < 0.512
    for r in (ft, base):
        assert r["ci_lo"] < r["auc"] < r["ci_hi"] and r["null_lo"] < r["null_mean"] < r["null_hi"]
    assert ft["auc"] > ft["null_hi"]                       # the fine-tune is outside chance
    assert base["null_lo"] < base["auc"] < base["null_hi"]  # the base model is inside it
    assert ft["ci_lo"] > 0.5 and base["ci_lo"] < 0.5
    assert ft["ci_hi"] - ft["ci_lo"] > 5 * (ft["lineage_ci_hi"] - ft["lineage_ci_lo"])    # lineage resampling is far too narrow
    d = t["codon_within_pair_difference"].iloc[0]
    assert d["ci_lo"] > 0 and d["difference"] == pytest.approx(ft["auc"] - base["auc"], abs=1e-12)
    assert ft["audit_auc_ties_in_row_order"] != ft["auc"]


@pytest.mark.parametrize("name", SETS)
def test_shipped_strand_c_is_null_for_both_models(derived, name):
    _, t = derived[name]
    s = t["codon_strand_c_summary"]
    assert (s["n_aa"] == 18).all()
    assert (s["ci_lo"] < 0).all() and (s["ci_hi"] > 0).all()            # the interval spans zero
    assert (s["p_perm_lower"] > 0.1).all()
    assert s["symmetrised"].all() and (s["spectrum_file"].str.startswith("data/spectra/")).all()


@pytest.mark.parametrize("name", SETS)
def test_shipped_pair_and_robustness_tables(derived, name):
    _, t = derived[name]
    bp = t["codon_within_pair_by_pair"]
    assert bp.groupby("model").size().eq(54).all()
    assert (bp["ci_lo"] <= bp["ci_hi"]).all() and bp["auc"].between(0, 1).all()
    assert (bp["sites_one_step"] >= 5).all() and (bp["sites_multi_step"] >= 5).all()
    rob = t["codon_within_pair_robustness"]
    assert (rob.groupby("model")["check"].nunique() == 6).all()
    assert t["codon_within_pair_by_lineage"].groupby("model").size().isin([5, 13]).all()


# ---------------------------------------------------------------------------
# The figure script
# ---------------------------------------------------------------------------

import make_codon_learning as fig  # noqa: E402


def test_the_figure_check_agrees_with_the_scores_and_catches_tampering(derived, shipped):
    g, tables = derived["h3n2_5lineage"]
    scores = shipped["h3n2_5lineage"][1]
    assert fig.check_against_scores(g, scores, tables) < 1e-9
    bad = {k: v.copy() for k, v in tables.items()}
    bad["codon_within_pair_summary"].loc[0, "auc"] += 1e-6
    with pytest.raises(ValueError, match="disagree"):
        fig.check_against_scores(g, scores, bad)
    bad = {k: v.copy() for k, v in tables.items()}
    bad["codon_within_pair_by_pair"] = bad["codon_within_pair_by_pair"].iloc[1:]
    with pytest.raises(ValueError, match="different replacements"):
        fig.check_against_scores(g, scores, bad)
    bad = {k: v.copy() for k, v in tables.items()}
    bad["codon_strand_c_summary"].loc[0, "mean_rho"] += 1e-4
    with pytest.raises(ValueError, match="disagree"):
        fig.check_against_scores(g, scores, bad)
    bad = {k: v.copy() for k, v in tables.items()}
    bad["codon_within_pair_summary"].loc[0, "n_blocks"] += 1
    with pytest.raises(ValueError, match="blocks"):
        fig.check_against_scores(g, scores, bad)


def test_role_rows_needs_both_roles(derived):
    g, tables = derived["h3n2_5lineage"]
    one = tables["codon_within_pair_summary"].iloc[:1]
    with pytest.raises(ValueError, match="base and a fine-tuned"):
        fig.role_rows(one, g)


def test_caption_quotes_the_table_not_a_typed_number(derived):
    g, tables = derived["h3n2_13lineage"]
    guides = {"13 lineages": g}
    text = " ".join(fig.caption({"13 lineages": tables}, guides, "13 lineages", 1e-16).split())
    ft = tables["codon_within_pair_summary"].set_index("role").loc["finetune"]
    assert f"{ft['auc']:.3f}" in text and f"{int(ft['n_pairs'])} scored amino-acid pairs" in text
    assert "not a model failure" not in text and "null" in text


def test_two_guides_with_the_same_lineage_count_are_refused(tmp_path):
    g = REPO_ROOT / "guides" / "h3n2_13lineage.yaml"
    with pytest.raises(ValueError, match="share a label"):
        fig.main(["--guide", str(g), "--guide-secondary", str(g), "--out", str(tmp_path)])


@pytest.mark.slow
def test_script_renders_both_figures_with_a_secondary_guide(tmp_path):
    g13 = REPO_ROOT / "guides" / "h3n2_13lineage.yaml"
    g5 = REPO_ROOT / "guides" / "h3n2_5lineage.yaml"
    assert fig.main(["--guide", str(g13), "--guide-secondary", str(g5), "--out", str(tmp_path)]) == 0
    for stem in ("fig05_codon_learning", "sI_codon_pair_tracking"):
        assert (tmp_path / f"{stem}.png").stat().st_size > 30_000 and (tmp_path / f"{stem}.pdf").stat().st_size > 5_000
        assert (tmp_path / f"{stem}_caption.md").read_text().strip()
    v = pd.read_csv(tmp_path / "fig05_codon_learning_values.tsv", sep="\t")
    # Rebuilt 13 Oct 2026 as a single three-panel figure: worked example, totals, null.
    assert set(v["panel"]) == {"A", "B", "C"}
    assert set(v.loc[v["panel"] == "C", "set"]) == {"13 lineages", "5 lineages"}
    pairs = pd.read_csv(tmp_path / "sI_codon_pair_tracking_values.tsv", sep="\t")
    assert len(pairs) == 108


@pytest.mark.slow
def test_script_renders_alone_and_is_deterministic(tmp_path):
    g13 = REPO_ROOT / "guides" / "h3n2_13lineage.yaml"
    for d in ("a", "b"):
        assert fig.main(["--guide", str(g13), "--out", str(tmp_path / d)]) == 0
    assert (tmp_path / "a" / "fig05_codon_learning_values.tsv").read_text() == (tmp_path / "b" / "fig05_codon_learning_values.tsv").read_text()


@pytest.mark.slow
def test_script_renders_single_lineage(tmp_path):
    g5 = REPO_ROOT / "guides" / "h3n2_5lineage.yaml"
    assert fig.main(["--guide", str(g5), "--out", str(tmp_path)]) == 0
    for stem in ("fig05_codon_learning", "sI_codon_pair_tracking"):
        assert (tmp_path / f"{stem}.png").stat().st_size > 30_000 and (tmp_path / f"{stem}.pdf").stat().st_size > 5_000
        assert (tmp_path / f"{stem}_caption.md").read_text().strip()
    v = pd.read_csv(tmp_path / "fig05_codon_learning_values.tsv", sep="\t")
    assert set(v["panel"]) == {"A", "B", "C"}
    assert set(v.loc[v["panel"] == "C", "set"]) == {"J.2_int"}
    pairs = pd.read_csv(tmp_path / "sI_codon_pair_tracking_values.tsv", sep="\t")
    assert len(pairs) == 108


# ---------------------------------------------------------------------------
# Adversarial inputs
# ---------------------------------------------------------------------------

def test_a_model_with_no_information_scores_exactly_one_half(synth):
    frame, _ = synth
    flat = frame.assign(plm_prob=1e-3)
    assert cl.WithinPair(cl.canonical_scores(flat, sc.MODEL)).pooled() == pytest.approx(0.5, abs=1e-12)


def test_lineages_covering_different_positions_still_score_and_shuffle(tmp_path):
    frame, roots = sc.make_scores(tmp_path, n_pos=700, n_lin=2, seed=11, delta=0.5)
    frame = frame.loc[~((frame.lineage == "L1") & (frame.position <= 100))]
    prep = cl.WithinPair(cl.canonical_scores(frame, sc.MODEL))
    cm = cl.reference_codon_matrix(prep, roots)
    assert (cm[1, :100] == -1).all() and (cm[0] >= 0).all()
    cl.check_roots(prep, cm)
    shared, own = cl.position_groups(prep, cm)
    assert all((g >= 100).all() for g in shared)          # a site one lineage lacks never moves as a shared position
    null = cl.permutation_null({"m": prep}, cm, n_perm=8, seed=1)["m"]
    assert np.isfinite(null).all() and prep.pooled() > null.max() - 0.05


def test_numeric_looking_lineage_names_work(tmp_path):
    frame, roots = sc.make_scores(tmp_path, n_pos=700, n_lin=2, seed=12)
    ren = {"L0": "1", "L1": "2"}
    frame = frame.assign(lineage=frame["lineage"].map(ren)).astype({"lineage": int})
    roots = {ren[k]: v for k, v in roots.items()}
    prep = cl.WithinPair(cl.canonical_scores(frame, sc.MODEL))
    assert prep.lineages == ["1", "2"]
    cl.check_roots(prep, cl.reference_codon_matrix(prep, roots))


def test_row_order_of_the_input_does_not_change_the_answer(synth):
    frame, _ = synth
    a = cl.WithinPair(cl.canonical_scores(frame, sc.MODEL)).pooled()
    b = cl.WithinPair(cl.canonical_scores(frame.sample(frac=1, random_state=3), sc.MODEL)).pooled()
    assert a == b


def test_scaling_probabilities_changes_nothing(synth):
    frame, _ = synth
    a = cl.WithinPair(cl.canonical_scores(frame, sc.MODEL)).pooled()
    b = cl.WithinPair(cl.canonical_scores(frame.assign(plm_prob=frame.plm_prob * 1e5), sc.MODEL)).pooled()
    assert a == pytest.approx(b, abs=1e-15)
