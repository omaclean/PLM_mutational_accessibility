"""Red team: accessibility and the symmetry battery against independent brute force.

Every reference here is written differently from the code under test: genetic code from
Biopython's ``translate``, accessibility by explicit enumeration of all 64 target codons with a
Hamming distance, spectrum variants by explicit loops.

Bugs these tests pinned:

* ``reference_codons`` accepted a negative position (``-1`` read a codon from the END of the
  sequence), a float position, and a FASTA with several records (concatenated silently). Fixed.
* ``symmetry.direction_contrast`` ranked scores among ALL one-step rows but divided by the number
  of KEPT rows, so its "rank in [0, 1]" ran up to n_one_step / n_kept and the contrast changed
  when excluded rows were added. Fixed: rank among the kept rows. NOTE the shipped
  ``symmetry_*_direction`` tables were produced before this fix and are not regenerated here.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from Bio.Seq import Seq
from scipy.stats import spearmanr

from mutacc import accessibility as A
from mutacc import dms_sweep as D
from mutacc import symmetry as SY


def translate(codon):
    return str(Seq(codon).translate())


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def random_spectrum(rng, scale=0.04):
    q = rng.uniform(0.2, 1.0, (4, 4)) * scale
    np.fill_diagonal(q, 0.0)
    return q


def brute_accessibility(ref, aa, q):
    """Sum, over every codon that translates to ``aa``, of the product of per-position probabilities."""
    p = q.copy()
    for i in range(4):
        p[i, i] = 1.0 - q[i].sum()
    total = 0.0
    for target in map("".join, itertools.product("ACGT", repeat=3)):
        if translate(target) != aa:
            continue
        prob = 1.0
        for a, b in zip(ref, target):
            prob *= p["ACGT".index(a), "ACGT".index(b)]
        total += prob
    return total


# ---------------------------------------------------------------------------
# Genetic code
# ---------------------------------------------------------------------------

def test_genetic_code_equals_biopython_translate_for_all_64_codons():
    assert len(A.GENETIC_CODE) == 64
    for codon, aa in A.GENETIC_CODE.items():
        assert translate(codon) == aa, codon
    assert sorted(set(A.GENETIC_CODE.values()) - {"*"}) == sorted(A.AA20)
    assert [c for c, a in A.GENETIC_CODE.items() if a == "*"] == ["TAA", "TAG", "TGA"]


def test_the_two_code_tables_in_the_repo_agree():
    codons = ["".join(c) for c in itertools.product("TCAG", repeat=3)]
    for codon, aa in zip(codons, D.STANDARD_CODE):
        assert A.GENETIC_CODE[codon] == aa, codon


# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------

def test_accessibility_equals_explicit_enumeration_for_random_spectra():
    rng = np.random.default_rng(201)
    for _ in range(6):
        q = random_spectrum(rng)
        table = A.codon_aa_table(q)
        for ref in rng.choice(A.CODONS, 12):
            for aa in A.AA20:
                assert table[A.codon_index(ref), A.aa_index(aa)] == pytest.approx(
                    brute_accessibility(ref, aa, q), rel=1e-12, abs=1e-300), (ref, aa)


def test_accessibility_function_indexes_the_right_cell():
    rng = np.random.default_rng(202)
    q = random_spectrum(rng)
    refs = list(rng.choice([c for c in A.CODONS if A.GENETIC_CODE[c] != "*"], 40))
    aas = list(rng.choice(list(A.AA20), 40))
    got = A.accessibility(refs, aas, q)
    want = [brute_accessibility(r, a, q) for r, a in zip(refs, aas)]
    assert np.allclose(got, want, rtol=1e-12, atol=0)


def test_each_row_plus_the_stop_mass_is_one():
    rng = np.random.default_rng(203)
    q = random_spectrum(rng)
    table = A.codon_aa_table(q)
    for ref in A.CODONS:
        stop = brute_accessibility(ref, "*", q)
        assert table[A.codon_index(ref)].sum() + stop == pytest.approx(1.0, abs=1e-12)


def test_equal_rate_spectrum_gives_a_product_of_hamming_powers():
    """Q = eps off the diagonal: P(codon -> codon) = eps^d (1 - 3 eps)^(3 - d) with d the Hamming distance."""
    eps = 0.01
    q = np.full((4, 4), eps)
    np.fill_diagonal(q, 0.0)
    table = A.codon_aa_table(q)
    for ref in ("ATG", "CTT", "TGG", "AGA", "GCC"):
        for aa in A.AA20:
            want = sum(eps ** hamming(ref, c) * (1 - 3 * eps) ** (3 - hamming(ref, c))
                       for c in A.CODONS if translate(c) == aa)
            assert table[A.codon_index(ref), A.aa_index(aa)] == pytest.approx(want, rel=1e-12)


def test_leading_order_in_a_small_rate_is_the_number_of_nucleotide_steps():
    """For tiny eps, accessibility ~ (routes) * eps^steps, so 1, 2 and 3 step products separate by orders of magnitude."""
    eps = 1e-6
    q = np.full((4, 4), eps)
    np.fill_diagonal(q, 0.0)
    table = A.codon_aa_table(q)
    seen = set()
    for ref in A.CODONS:
        if A.GENETIC_CODE[ref] == "*":
            continue
        for aa in A.AA20:
            if aa == A.GENETIC_CODE[ref]:
                continue
            steps = A.nucleotide_steps(ref, aa)
            ratio = table[A.codon_index(ref), A.aa_index(aa)] / eps ** steps
            assert 0.99 < ratio < 30, (ref, aa, steps, ratio)       # a handful of routes at most
            seen.add(steps)
    assert seen == {1, 2, 3}


def test_nucleotide_steps_and_single_step_changes_match_brute_force():
    for ref in A.CODONS:
        if A.GENETIC_CODE[ref] == "*":
            continue
        neighbours = {}
        for pos in range(3):
            for b in "ACGT":
                if b != ref[pos]:
                    mutant = ref[:pos] + b + ref[pos + 1:]
                    neighbours.setdefault(translate(mutant), []).append((ref[pos], b))
        for aa in A.AA20:
            want_steps = min(hamming(ref, c) for c in A.CODONS if translate(c) == aa)
            assert A.nucleotide_steps(ref, aa) == want_steps, (ref, aa)
            assert sorted(A.single_step_changes(ref, aa)) == sorted(neighbours.get(aa, [])), (ref, aa)


def test_reversing_the_spectrum_reverses_the_codon_to_codon_probability():
    """P_Q(TGG -> ATG) = P_(Q^T)(ATG -> TGG) when the stay-probabilities do not change.

    That needs equal row and column sums, so Q is a weighted sum of cyclic shifts (rows and columns
    both sum to a + b + c). Met and Trp each have one codon, so accessibility IS that probability.
    The stay-probability comes from row sums, so for a general Q the identity does not hold; pinned
    in the second half.
    """
    shifts = [np.roll(np.eye(4), k, axis=1) for k in (1, 2, 3)]
    q = 0.05 * shifts[0] + 0.02 * shifts[1] + 0.01 * shifts[2]
    assert np.allclose(q.sum(axis=0), q.sum(axis=1)) and not np.allclose(q, q.T)
    forward = A.accessibility(["TGG"], ["M"], q)[0]
    assert forward == pytest.approx(A.accessibility(["ATG"], ["W"], A.transpose(q))[0], rel=1e-12)
    assert forward != pytest.approx(A.accessibility(["TGG"], ["M"], A.transpose(q))[0])
    rng = np.random.default_rng(204)
    general = random_spectrum(rng)
    assert A.accessibility(["TGG"], ["M"], general)[0] != pytest.approx(
        A.accessibility(["ATG"], ["W"], A.transpose(general))[0], rel=1e-6)


def test_the_two_accessibility_implementations_agree():
    """mutacc.accessibility (ACGT order, kron) vs mutacc.dms_sweep (TCAG order, explicit loops)."""
    rng = np.random.default_rng(205)
    for reversible in (False, True):
        q = random_spectrum(rng)
        frame = pd.DataFrame(q, index=list("ACGT"), columns=list("ACGT"))
        theirs = D.accessibility_table(frame, reversible=reversible)
        mine = A.codon_aa_table(A.symmetrise(q) if reversible else q)
        for codon in theirs.index:
            for aa in theirs.columns:
                assert theirs.loc[codon, aa] == pytest.approx(mine[A.codon_index(codon), A.aa_index(aa)], rel=1e-12)
        assert set(theirs.columns) == set(A.AA20)


def test_log_accessibility_floors_zeros_and_refuses_unknown_labels():
    frame = pd.DataFrame(np.array([[0, .1, .2, .1], [.1, 0, .1, .2], [.2, .1, 0, .1], [.1, .2, .1, 0.]]),
                         index=list("ACGT"), columns=list("ACGT"))
    table = D.accessibility_table(frame, reversible=False)
    table.loc["ATG", "W"] = 0.0
    out = D.log_accessibility(["ATG", "ATG"], ["W", "K"], table)
    assert np.isfinite(out).all() and out[0] < out[1]
    with pytest.raises(ValueError, match="not in the accessibility table"):
        D.log_accessibility(["XYZ"], ["W"], table)
    with pytest.raises(ValueError, match="not in the accessibility table"):
        D.log_accessibility(["ATG"], ["*"], table)


# ---------------------------------------------------------------------------
# Spectrum handling
# ---------------------------------------------------------------------------

def test_as_probabilities_rows_sum_to_one_and_refuses_unscaled_matrices():
    rng = np.random.default_rng(206)
    q = random_spectrum(rng)
    assert np.allclose(A.as_probabilities(q).sum(axis=1), 1.0)
    big = q * 100
    with pytest.raises(ValueError, match="probabilities"):
        A.as_probabilities(big)
    scaled = A.as_probabilities(big, row_sum=0.1)
    assert np.allclose(scaled.sum(axis=1), 1.0)
    assert (np.diag(scaled) > 0.5).all()
    assert (scaled >= 0).all()


def test_complement_transpose_symmetrise_and_relabel_against_explicit_loops():
    rng = np.random.default_rng(207)
    q = random_spectrum(rng)
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    out = A.complement(q)
    for i, a in enumerate("ACGT"):
        for j, b in enumerate("ACGT"):
            assert out[i, j] == q["ACGT".index(comp[a]), "ACGT".index(comp[b])]
    assert np.array_equal(A.complement(A.complement(q)), q)
    assert np.array_equal(A.transpose(A.transpose(q)), q)
    assert np.allclose(A.symmetrise(q), A.symmetrise(q).T) and np.allclose(A.symmetrise(q), A.symmetrise(A.transpose(q)))
    assert A.relative_asymmetry(A.symmetrise(q)) == pytest.approx(0, abs=1e-15)
    assert A.relative_asymmetry(A.complement(q)) == pytest.approx(A.relative_asymmetry(q))
    perms = A.all_relabellings()
    assert len(perms) == len(set(perms)) == 24 and perms[0] == (0, 1, 2, 3)
    for perm in perms[:6]:
        r = A.relabel(q, perm)
        for i in range(4):
            for j in range(4):
                assert r[i, j] == q[perm[i], perm[j]]
    assert np.array_equal(A.relabel(q, (0, 1, 2, 3)), q)
    for bad in ([0, 1, 2], [0, 0, 1, 2], [1, 2, 3, 4]):
        with pytest.raises(ValueError):
            A.relabel(q, bad)


def test_shuffle_keeps_the_rates_the_zero_diagonal_and_is_seeded():
    rng = np.random.default_rng(208)
    q = random_spectrum(rng)
    a = A.shuffle(q, np.random.default_rng(1))
    b = A.shuffle(q, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert (np.diag(a) == 0).all()
    assert sorted(a[~np.eye(4, dtype=bool)]) == sorted(q[~np.eye(4, dtype=bool)])
    assert not np.array_equal(a, A.shuffle(q, np.random.default_rng(2)))


def test_load_spectrum_refuses_bad_files(tmp_path):
    def write(body):
        path = tmp_path / "s.csv"
        path.write_text(body)
        return path

    good = "from_base,A,C,G,T\nA,0,1,2,1\nC,1,0,1,2\nG,2,1,0,1\nT,1,2,1,0\n"
    assert A.load_spectrum(write(good)).shape == (4, 4)
    for bad in (good.replace("2,1,0,1", "-2,1,0,1"), good.replace("1,0,1,2", "nan,0,1,2"),
                "from_base,A,C,G,T\nA,0,0,0,0\nC,0,0,0,0\nG,0,0,0,0\nT,0,0,0,0\n",
                "from_base,A,C,G\nA,0,1,2\nC,1,0,1\nG,2,1,0\n"):
        with pytest.raises(ValueError):
            A.load_spectrum(write(bad))


def test_load_spectrum_zeroes_the_diagonal():
    import io
    q = A.load_spectrum(io.StringIO("from_base,A,C,G,T\nA,5,1,2,1\nC,1,5,1,2\nG,2,1,5,1\nT,1,2,1,5\n"))
    assert (np.diag(q) == 0).all()


# ---------------------------------------------------------------------------
# reference_codons
# ---------------------------------------------------------------------------

def write_fasta(tmp_path, text):
    path = tmp_path / "r.fa"
    path.write_text(text)
    return path


def test_reference_codons_one_based_and_case_and_line_endings(tmp_path):
    path = write_fasta(tmp_path, ">r\r\natgaag\r\nGCTtga\r\n")
    assert A.reference_codons(path, [1, 2, 3, 4]) == ["ATG", "AAG", "GCT", "TGA"]
    assert A.reference_codons(path, [4, 1]) == ["TGA", "ATG"]
    assert A.reference_codons(path, np.array([2])) == ["AAG"]


@pytest.mark.parametrize("bad", [0, -1, -4, 5, 1.5, 100])
def test_reference_codons_refuses_positions_outside_the_sequence(tmp_path, bad):
    path = write_fasta(tmp_path, ">r\nATGAAGGCTTGA\n")
    with pytest.raises(ValueError):
        A.reference_codons(path, [bad])


def test_reference_codons_refuses_n_bases_ragged_ends_and_multiple_records(tmp_path):
    with pytest.raises(ValueError, match="no valid codon"):
        A.reference_codons(write_fasta(tmp_path, ">r\nATGNAG\n"), [2])
    with pytest.raises(ValueError, match="no valid codon"):
        A.reference_codons(write_fasta(tmp_path, ">r\nATGAA\n"), [2])
    with pytest.raises(ValueError, match="exactly one"):
        A.reference_codons(write_fasta(tmp_path, ">a\nATG\n>b\nAAG\n"), [1])
    with pytest.raises(ValueError, match="exactly one"):
        A.reference_codons(write_fasta(tmp_path, "ATGAAG\n"), [1])


# ---------------------------------------------------------------------------
# Symmetry battery
# ---------------------------------------------------------------------------

def all_rows_unit(rng, q, name="u", extra_excluded=0, steps_wanted=None):
    """One site per (sense codon, target amino acid): every possible replacement exactly once."""
    sites, codons, aas, ref_aas, steps = [], [], [], [], []
    site = 0
    for ref in A.CODONS:
        if A.GENETIC_CODE[ref] == "*":
            continue
        for aa in A.AA20:
            if aa == A.GENETIC_CODE[ref]:
                continue
            site += 1
            sites.append(site)
            codons.append(A.codon_index(ref))
            aas.append(A.aa_index(aa))
            ref_aas.append(A.aa_index(A.GENETIC_CODE[ref]))
            steps.append(A.nucleotide_steps(ref, aa))
    n = len(sites)
    truth = A.codon_aa_table(q)[codons, aas]
    scores = {"noise": rng.normal(size=n), "truth": np.log(truth)}
    return SY.Unit(name, np.array(sites), np.array(steps), np.array(codons), np.array(aas), np.array(ref_aas), scores)


def test_favoured_classes_are_the_faster_direction_of_each_pair():
    q = np.zeros((4, 4))
    q[0, 2], q[2, 0] = 0.09, 0.01       # A>G favoured, G>A not
    q[1, 3], q[3, 1] = 0.02, 0.02       # C<>T equal: left out
    q[0, 1], q[1, 0] = 0.01, 0.03       # C>A favoured
    fav, dis = SY.favoured_classes(q)
    assert fav == {("A", "G"), ("C", "A")} and dis == {("G", "A"), ("A", "C")}
    assert SY.favoured_classes(A.symmetrise(q)) == (set(), set())


def test_flip_variants_are_64_distinct_directional_assignments():
    rng = np.random.default_rng(209)
    q = random_spectrum(rng)
    variants = SY.flip_variants(q)
    assert len(variants) == 64
    assert np.array_equal(variants[0], q) and np.array_equal(variants[63], q.T)
    assert len({v.tobytes() for v in variants}) == 64
    for v in variants:
        assert np.allclose(A.symmetrise(v), A.symmetrise(q))
        assert sorted(v[~np.eye(4, dtype=bool)]) == sorted(q[~np.eye(4, dtype=bool)])


def test_orientation_rho_equals_mean_scipy_spearman_with_ties_and_no_boot_needed():
    rng = np.random.default_rng(210)
    q = random_spectrum(rng)
    units = [all_rows_unit(rng, q, "u1"), all_rows_unit(rng, q, "u2")]
    tab = SY.orientation_battery(units, q, ["noise", "truth"], strata=(1, 2, "all"), n_boot=3, seed=1)
    for stratum in (1, 2, "all"):
        for variant, qq in SY.orientation_variants(q).items():
            vals = []
            for u in units:
                rows = np.arange(u.n) if stratum == "all" else np.where(u.steps == stratum)[0]
                target = SY.accessibility_for(u, qq)[rows]
                vals.append(spearmanr(u.scores["noise"][rows], target).statistic)
            row = tab[(tab.kind == "rho") & (tab.score == "noise") & (tab.target == variant) & (tab.stratum == stratum)]
            assert row["value"].iloc[0] == pytest.approx(np.mean(vals), abs=1e-12)
    # a score that IS the truth prefers the real spectrum to its transpose in every stratum
    d = tab[(tab.kind == "delta") & (tab.score == "truth") & (tab.versus == "transposed")]
    assert (d["value"] > 0).all()


def test_symmetric_score_has_exactly_zero_direction_contrast():
    """A score that depends only on the unordered amino-acid pair cannot see direction."""
    rng = np.random.default_rng(211)
    q = random_spectrum(rng)
    u = all_rows_unit(rng, q)
    pair_value = {}
    sym = np.empty(u.n)
    for i in range(u.n):
        key = tuple(sorted((int(u.ref_aa[i]), int(u.aa[i]))))
        sym[i] = pair_value.setdefault(key, rng.normal())
    u.scores["sym"] = sym
    out = SY.direction_contrast([u], q, ["sym", "noise"], n_boot=4, seed=3).set_index("score")
    assert out.loc["sym", "contrast"] == pytest.approx(0.0, abs=1e-12)
    assert out.loc["accessibility (truth)", "contrast"] > 0.05
    assert abs(out.loc["noise", "contrast"]) < out.loc["accessibility (truth)", "contrast"]


def test_direction_contrast_ignores_rows_it_excludes():
    """Adding one-step rows that have several routes (so are excluded) must not move the contrast."""
    rng = np.random.default_rng(212)
    q = random_spectrum(rng)
    u = all_rows_unit(rng, q)
    out = {}
    for extra in (0, 1):
        scores = {k: v.copy() for k, v in u.scores.items()}
        site, steps, codon, aa, ref_aa = u.site, u.steps, u.ref_codon, u.aa, u.ref_aa
        if extra:
            multi = [i for i in range(u.n) if steps[i] == 1 and len(A.single_step_changes(
                A.CODONS[codon[i]], A.AA20[aa[i]])) > 1]
            assert multi, "the code has one-step replacements with several routes"
            idx = np.array(multi * 5)
            new_sites = site.max() + 1 + np.arange(len(idx))
            site = np.r_[site, new_sites]
            steps, codon, aa, ref_aa = (np.r_[x, x[idx]] for x in (steps, codon, aa, ref_aa))
            scores = {k: np.r_[v, 10 * rng.normal(size=len(idx))] for k, v in scores.items()}
        unit = SY.Unit("u", site, steps, codon, aa, ref_aa, scores)
        out[extra] = SY.direction_contrast([unit], q, ["noise", "truth"], n_boot=3, seed=1).set_index("score")["contrast"]
    assert out[0]["noise"] == pytest.approx(out[1]["noise"], abs=1e-12)
    assert out[0]["truth"] == pytest.approx(out[1]["truth"], abs=1e-12)


def test_no_bootstrap_replicate_of_the_direction_contrast_is_undefined():
    """A resample that misses every row of a rare amino-acid pair used to give 0 * NaN = NaN, and
    nanpercentile then dropped that replicate without a word."""
    from unittest import mock

    rng = np.random.default_rng(217)
    q = random_spectrum(rng)
    u = all_rows_unit(rng, q)
    seen = []
    real = np.nanpercentile

    def spy(a, *args, **kw):
        a = np.asarray(a)
        if a.ndim == 1:
            seen.append(int(np.isnan(a).sum()))
        return real(a, *args, **kw)

    with mock.patch.object(np, "nanpercentile", spy):
        out = SY.direction_contrast([u], q, ["noise", "truth"], n_boot=25, seed=2)
    assert seen and sum(seen) == 0
    assert out[["lo", "hi"]].notna().all().all()


def test_direction_contrast_is_bounded_by_the_rank_scale():
    rng = np.random.default_rng(213)
    q = random_spectrum(rng)
    u = all_rows_unit(rng, q)
    out = SY.direction_contrast([u], q, ["truth", "noise"], n_boot=3, seed=1)
    assert (out["contrast"].abs() <= 1.0).all()


def test_null_battery_ranks_the_identity_first_and_is_seeded():
    rng = np.random.default_rng(214)
    q = random_spectrum(rng)
    u = all_rows_unit(rng, q)
    a = SY.null_battery([u], q, ["truth", "noise"], n_shuffle=20, seed=4)
    b = SY.null_battery([u], q, ["truth", "noise"], n_shuffle=20, seed=4)
    pd.testing.assert_frame_equal(a, b)
    truth = a[a.score == "truth"]
    assert (truth["pct_relabel"] <= 1 / 24 + 1e-12).all()        # the identity is the best of 24 for the truth
    assert (a["pct_relabel"] >= 1 / 24 - 1e-12).all()             # and the identity itself is always counted


def test_battery_is_seeded_end_to_end():
    rng = np.random.default_rng(215)
    q = random_spectrum(rng)
    u = all_rows_unit(rng, q)
    a = SY.orientation_battery([u], q, ["noise"], strata=(1, "all"), n_boot=6, seed=9)
    b = SY.orientation_battery([u], q, ["noise"], strata=(1, "all"), n_boot=6, seed=9)
    c = SY.orientation_battery([u], q, ["noise"], strata=(1, "all"), n_boot=6, seed=10)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["lo"], c["lo"], equal_nan=True)


def test_units_from_frame_refuses_holes_and_unknown_labels():
    rng = np.random.default_rng(216)
    base = pd.DataFrame({"lineage": ["a"] * 12, "position": range(12), "ref_codon": ["ATG"] * 12,
                         "aa": list("ACDEFGHIKLNP"), "ref_aa": ["M"] * 12, "nt_steps": [1] * 12,
                         "s": rng.normal(size=12)})
    assert SY.units_from_frame(base, ["s"])[0].n == 12
    holey = base.copy()
    holey.loc[3, "s"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        SY.units_from_frame(holey, ["s"])
    with pytest.raises(KeyError):
        SY.units_from_frame(base.drop(columns="ref_aa"), ["s"])
    bad = base.copy()
    bad.loc[0, "aa"] = "*"
    with pytest.raises(KeyError):
        SY.units_from_frame(bad, ["s"])
    bad = base.copy()
    bad.loc[0, "ref_codon"] = "NNN"
    with pytest.raises(KeyError):
        SY.units_from_frame(bad, ["s"])
