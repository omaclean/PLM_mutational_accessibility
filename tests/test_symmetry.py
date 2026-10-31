"""The symmetry battery is tested on scores whose relationship to the spectrum is known by design."""

import numpy as np
import pandas as pd
import pytest

from mutacc import accessibility as A
from mutacc import symmetry as S
from mutacc.guide import REPO_ROOT

Q = A.load_spectrum(REPO_ROOT / "data" / "spectra" / "influenza_h3n2.csv")
SYM = A.symmetrise(Q)
NONSTOP = [c for c in A.CODONS if A.GENETIC_CODE[c] != "*"]


def synthetic(n_sites=160, n_units=2, noise=0.15, seed=0):
    """Rows for random reference codons with known scores. Units share their sites, as lineages do."""
    rng = np.random.default_rng(seed)
    codons = [NONSTOP[i] for i in rng.integers(0, len(NONSTOP), n_sites)]
    rows = []
    pair_effect = {}
    for u in range(n_units):
        for site, c in enumerate(codons, 1):
            ref = A.GENETIC_CODE[c]
            for aa in A.AA20:
                if aa == ref:
                    continue
                pair = tuple(sorted((ref, aa)))
                pair_effect.setdefault(pair, rng.normal())
                rows.append(dict(lineage=f"U{u}", position=site, ref_codon=c, ref_aa=ref, aa=aa,
                                 nt_steps=A.nucleotide_steps(c, aa), pair=pair))
    df = pd.DataFrame(rows)
    d_acc = np.log10(A.accessibility(df.ref_codon, df.aa, Q))
    s_acc = np.log10(A.accessibility(df.ref_codon, df.aa, SYM))
    z = lambda v: (v - v.mean()) / v.std()
    df["truth_dir"] = z(d_acc) + noise * rng.normal(size=len(df))
    df["truth_sym"] = z(s_acc) + noise * rng.normal(size=len(df))
    df["noise"] = rng.normal(size=len(df))
    df["pair_sim"] = [pair_effect[p] for p in df.pair]          # symmetric by construction, no noise
    return df.drop(columns="pair")


@pytest.fixture(scope="module")
def units():
    return S.units_from_frame(synthetic(), ["truth_dir", "truth_sym", "noise", "pair_sim"])


SCORES = ["truth_dir", "truth_sym", "noise", "pair_sim"]


def get(frame, **kw):
    out = frame
    for k, v in kw.items():
        out = out[out[k] == v]
    assert len(out) == 1, kw
    return out.iloc[0]


# ---------------------------------------------------------------------------
# Orientation battery
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def orient(units):
    return S.orientation_battery(units, Q, SCORES, n_boot=40, seed=1)


def test_a_directional_score_prefers_the_real_spectrum_over_its_symmetrisation(orient):
    d = get(orient, kind="delta", score="truth_dir", versus="symmetrised", stratum=1)
    assert d.lo > 0 and d.value > 0.05


def test_a_symmetric_score_prefers_the_symmetrised_spectrum(orient):
    d = get(orient, kind="delta", score="truth_sym", versus="symmetrised", stratum=1)
    assert d.hi < 0


def test_a_directional_score_beats_the_transposed_spectrum(orient):
    d = get(orient, kind="delta", score="truth_dir", versus="transposed", stratum=1)
    assert d.lo > 0


def test_pure_noise_shows_no_preference(orient):
    for versus in ("symmetrised", "transposed", "complemented"):
        d = get(orient, kind="delta", score="noise", versus=versus, stratum=1)
        assert d.lo < 0 < d.hi


def test_the_symmetric_amino_acid_score_has_no_directional_signature(orient):
    """A score that is the same in both directions cannot beat the transposed spectrum by much,
    which is the control the original S04 comparison lacked."""
    d = get(orient, kind="delta", score="pair_sim", versus="transposed", stratum=1)
    assert abs(d.value) < 0.15


def test_correlations_are_in_range_and_cover_every_cell(orient):
    rho = orient[orient.kind == "rho"]
    assert len(rho) == len(SCORES) * 4 * 4
    assert rho["value"].between(-1, 1).all()
    assert (orient.lo <= orient.hi + 1e-12).all()


def test_orientation_battery_is_seeded(units):
    a = S.orientation_battery(units, Q, ["truth_dir"], n_boot=10, seed=5, strata=(1,))
    b = S.orientation_battery(units, Q, ["truth_dir"], n_boot=10, seed=5, strata=(1,))
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Null battery
# ---------------------------------------------------------------------------

def test_a_score_built_from_the_real_spectrum_ranks_it_first_among_relabellings(units):
    nb = S.null_battery(units, Q, ["truth_dir", "noise"], n_shuffle=60, seed=2)
    t = get(nb, score="truth_dir", stratum=1)
    assert t.pct_relabel == pytest.approx(1 / 24) and t.z_shuffle > 2
    n = get(nb, score="noise", stratum=1)
    assert abs(n.z_shuffle) < 3


def test_null_battery_includes_the_identity_as_relabelling_zero(units):
    nb = S.null_battery(units, Q, ["truth_dir"], n_shuffle=5, strata=(1,))
    assert get(nb, score="truth_dir", stratum=1).best_relabel == 0


# ---------------------------------------------------------------------------
# Direction-flip null
# ---------------------------------------------------------------------------

def test_flip_variants_keep_the_rates_and_the_symmetric_part():
    vs = S.flip_variants(Q)
    assert len(vs) == 64 and np.array_equal(vs[0], Q) and np.allclose(vs[63], Q.T)
    for v in vs:
        assert np.allclose(A.symmetrise(v), SYM)
        assert sorted(v[~np.eye(4, dtype=bool)]) == sorted(Q[~np.eye(4, dtype=bool)])


def test_a_directional_score_ranks_the_real_direction_first_of_64(units):
    fn = S.flip_null(units, Q, ["truth_dir", "truth_sym", "noise", "pair_sim"], strata=(1,))
    assert get(fn, score="truth_dir", stratum=1).pct_flip == pytest.approx(1 / 64)


def test_a_symmetric_score_has_no_preference_among_flips(units):
    fn = S.flip_null(units, Q, ["truth_sym"], strata=(1,))
    assert get(fn, score="truth_sym", stratum=1).pct_flip > 0.2


def test_noise_has_no_preference_among_flips(units):
    fn = S.flip_null(units, Q, ["noise"], strata=(1,))
    assert get(fn, score="noise", stratum=1).pct_flip > 0.05


# ---------------------------------------------------------------------------
# Direction contrast
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def contrast(units):
    return S.direction_contrast(units, Q, SCORES, n_boot=30, seed=3)


def test_a_symmetric_score_reads_exactly_zero(contrast):
    """The built-in negative control: with the unordered pair absorbed there is nothing left."""
    r = get(contrast, score="pair_sim")
    assert abs(r.contrast) < 1e-12


def test_the_truth_has_a_positive_contrast_and_a_directional_score_inherits_it(contrast):
    truth = get(contrast, score="accessibility (truth)")
    assert truth.contrast > 0.05 and truth.fraction_of_truth == pytest.approx(1.0)
    d = get(contrast, score="truth_dir")
    assert d.contrast > 0 and 0.2 < d.fraction_of_truth <= 1.2
    assert get(contrast, score="truth_sym").contrast < d.contrast


def test_noise_has_no_contrast(contrast):
    r = get(contrast, score="noise")
    assert r.lo < 0 < r.hi


def test_favoured_classes_follow_the_higher_rate():
    fav, dis = S.favoured_classes(Q)
    assert ("A", "G") in fav and ("T", "C") in fav
    assert ("G", "A") in dis and ("C", "T") in dis
    assert S.favoured_classes(SYM) == (set(), set())


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-8, 1e-12])
def test_favoured_classes_do_not_depend_on_the_scale_of_the_rates(scale):
    """Regression: an absolute tolerance once made every rate in a 1e-8-scaled spectrum look equal."""
    assert S.favoured_classes(Q * scale) == S.favoured_classes(Q)


def test_a_symmetric_spectrum_has_nothing_to_contrast(units):
    with pytest.raises(ValueError, match="no favoured direction"):
        S.direction_contrast(units, SYM, ["noise"], n_boot=2)


# ---------------------------------------------------------------------------
# Planted signal and rank robustness
# ---------------------------------------------------------------------------

def test_planting_directional_signal_raises_the_preference_for_the_real_spectrum(units):
    ps = S.planted_signal(units, Q, ["pair_sim"], lambdas=(0.0, 0.5, 1.0), strata=(1,))
    d = ps[ps.planted == "direct"].set_index("lam")
    assert d.loc[1.0, "delta_sym"] > d.loc[0.5, "delta_sym"] > d.loc[0.0, "delta_sym"] - 1e-9
    s = ps[ps.planted == "symmetrised"].set_index("lam")
    assert s.loc[1.0, "delta_sym"] < 0


def test_rank_robustness_reports_all_three_methods_and_the_tie_counts(units):
    rr = S.rank_robustness(units, Q, ["truth_dir"], strata=(1,), n_tiebreak=5)
    assert set(rr.method) == {"average", "tiebreak", "log"}
    dv = rr.attrs["distinct_values"]
    assert (dv.distinct_symmetrised <= dv.distinct_direct * 2).all() and (dv.rows > 0).all()


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def test_units_refuse_missing_columns_and_nans():
    df = synthetic(n_sites=20, n_units=1)
    with pytest.raises(KeyError):
        S.units_from_frame(df.drop(columns="ref_codon"), ["noise"])
    bad = df.copy(); bad.loc[0, "noise"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        S.units_from_frame(bad, ["noise"])


def test_a_non_finite_score_is_refused():
    df = synthetic(n_sites=20, n_units=1)
    df.loc[0, "noise"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        S.units_from_frame(df, ["noise"])
