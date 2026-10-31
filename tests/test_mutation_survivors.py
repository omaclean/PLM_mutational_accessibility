"""Tests written from the mutants the suite let through (Batch 1, ``dev/mutation/``).

``dev/mutation/run_mutation.py`` changed one operator or constant at a time in ``metrics``,
``accessibility`` and ``strata`` and ran the covering tests. 51 of 300 mutants survived. Most
of those are the same few gaps, and each test below says which survivors it was written to kill.

**Survivors that are not gaps** (an equivalent mutant is one no input can tell apart):

* ``metrics.py`` lines 92, 106, 169: ``outcome > 0.5`` against ``>= 0.5`` or ``> 0.55``.
  ``_check`` has already rejected anything but exactly 0.0 and 1.0, so no outcome sits between.
* ``metrics.py`` line 39, ``sort_index(axis=1)`` -> ``axis=0``: ``DataFrame.pivot`` already
  returns its columns sorted, so the call is redundant. The test below still pins the order,
  so removing the redundancy later cannot unsort the alpha axis unnoticed.
* ``accessibility.py`` line 104 (``np.round(sums, 4)``): a digit in an error message.
* ``accessibility.py`` line 232 (``int(p) < 1`` against ``< 0``): position 0 still raises, from
  the codon-length check one line later; only the message differs, and it is pinned below.
* ``strata.py`` line 354, ``np.triu_indices(..., k=1)`` -> ``k=0``: the diagonal pairs it adds
  are (A, A), which no table can hold (a replacement never has ``ref == aa``), so they come out
  NaN and the finiteness filter drops them. ``test_every_unordered_pair...`` pins the 190.
* ``strata.py`` line 138 (``rtol=1e-6`` -> ``1.1e-6``): a tolerance no figure sits within a
  factor of 1.1 of.
* ``strata.py`` line 324, ``total > 0`` against ``>= 0``: at 0 both give NaN; only a warning
  differs, which ``test_an_all_zero_asymmetry_has_no_share_and_raises_no_warning`` pins.

Every expected value here is hand-worked, written as a loop, or taken from the definition,
never from the function under test.
"""

from __future__ import annotations

import inspect
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from mutacc import accessibility as A
from mutacc import metrics as M
from mutacc import strata as S

from test_redteam_metrics import make_replacement_frame

# ===========================================================================
# metrics.bootstrap_mean_ci          survivors: 53, 55, 66, 69, 71 (x4)
# ===========================================================================

DATA13 = pd.DataFrame(np.random.default_rng(0).random((13, 3)))


def _linear_quantile(sorted_column: np.ndarray, percent: float) -> float:
    """numpy's default (linear) percentile, written out: position ``(n-1) * p/100``."""
    pos = (len(sorted_column) - 1) * percent / 100.0
    low = int(np.floor(pos))
    high = min(low + 1, len(sorted_column) - 1)
    return sorted_column[low] + (pos - low) * (sorted_column[high] - sorted_column[low])


def reference_interval(values: np.ndarray, n_boot: int, level: float, seed: int):
    """The bootstrap interval from its definition: resample rows, average, take tail quantiles."""
    n = values.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    means = np.array([values[row].mean(axis=0) for row in idx])
    tail = 100.0 * (1.0 - level) / 2.0
    lo = [_linear_quantile(np.sort(means[:, j]), tail) for j in range(means.shape[1])]
    hi = [_linear_quantile(np.sort(means[:, j]), 100.0 - tail) for j in range(means.shape[1])]
    return np.array(lo), np.array(hi)


def test_the_default_interval_is_pinned_so_a_changed_default_cannot_move_a_published_band():
    """Seed 20261009, 2000 draws, 95%: the three defaults every figure's band depends on."""
    lo, hi = M.bootstrap_mean_ci(DATA13)
    assert lo == pytest.approx([0.3884367223912019, 0.4294199185404482, 0.31029925456271684], abs=1e-12)
    assert hi == pytest.approx([0.7008744968119707, 0.7089811515412735, 0.6796016681234157], abs=1e-12)


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9, 0.95, 0.99])
def test_interval_endpoints_are_the_tail_quantiles_of_the_stated_level(level):
    lo, hi = M.bootstrap_mean_ci(DATA13, n_boot=400, level=level, seed=7)
    want_lo, want_hi = reference_interval(DATA13.to_numpy(), 400, level, 7)
    assert lo == pytest.approx(want_lo, abs=1e-12)
    assert hi == pytest.approx(want_hi, abs=1e-12)


def test_a_wider_level_gives_a_wider_interval_and_the_same_median_draws():
    narrow = M.bootstrap_mean_ci(DATA13, level=0.5)
    wide = M.bootstrap_mean_ci(DATA13, level=0.99)
    assert (wide[0] < narrow[0]).all() and (wide[1] > narrow[1]).all()


def test_two_lineages_are_enough_and_one_row_can_be_drawn():
    """With rows {0} and {1}: all-zero, all-one and mixed draws are 1/4, 1/4, 1/2, so the 95%
    interval is [0, 1]. A resampler that never draws row 0 collapses it to [1, 1]."""
    lo, hi = M.bootstrap_mean_ci(pd.DataFrame([[0.0], [1.0]]))
    assert (lo[0], hi[0]) == (0.0, 1.0)


def test_one_lineage_has_no_bootstrap():
    with pytest.raises(ValueError, match="at least 2"):
        M.bootstrap_mean_ci(pd.DataFrame([[0.5, 0.6]]))


def test_the_default_draw_count_and_seed_are_the_documented_ones():
    params = inspect.signature(M.bootstrap_mean_ci).parameters
    assert (params["n_boot"].default, params["seed"].default, params["level"].default) == (2000, 20261009, 0.95)


# ===========================================================================
# metrics.lineage_matrix             survivor: 39 (sort_index axis)
# ===========================================================================

def test_the_alpha_axis_is_sorted_even_when_the_table_is_not():
    """A band drawn over an unsorted alpha axis would zig-zag; the matrix must order it."""
    alphas = [0.5, -1.0, 1.0, 0.0, -0.5]
    long = pd.DataFrame([
        {"lineage": lin, "alpha": a, "model": "m", "model_variant": "plm_alpha_sweep", "x": a * 10 + i}
        for i, lin in enumerate(["b", "a", "c"]) for a in alphas
    ])
    wide = M.lineage_matrix(long, "x", model="m")
    assert list(wide.columns) == sorted(alphas)
    assert wide.loc["a", -1.0] == -10 + 1 and wide.loc["c", 1.0] == 10 + 2


# ===========================================================================
# accessibility                      survivors: 102 (x2), 112, 213, 232
# ===========================================================================

def _off_diagonal(rate: float) -> np.ndarray:
    q = np.full((4, 4), rate)
    np.fill_diagonal(q, 0.0)
    return q


def test_a_row_summing_to_exactly_one_is_refused_and_just_under_one_is_not():
    """The contract in the error message is ``< 1``: a base that cannot stay put is not a
    per-site probability of change."""
    with pytest.raises(ValueError, match="per-site probabilities"):
        A.as_probabilities(_off_diagonal(1 / 3))
    p = A.as_probabilities(_off_diagonal(1 / 3 - 1e-6))
    assert np.diag(p) == pytest.approx(1e-6 * 3, abs=1e-12)
    assert p.sum(axis=1) == pytest.approx(1.0, abs=1e-12)


def test_a_row_summing_to_more_than_one_is_refused_not_given_a_negative_stay_probability():
    with pytest.raises(ValueError, match="per-site probabilities"):
        A.as_probabilities(_off_diagonal(0.5))                    # rows sum to 1.5


def test_a_relative_matrix_rescaled_to_row_sum_one_is_refused_too():
    with pytest.raises(ValueError, match="per-site probabilities"):
        A.as_probabilities(_off_diagonal(0.2), row_sum=1.0)


def test_the_codon_target_is_64_by_20_with_one_amino_acid_per_sense_codon_and_none_for_stops():
    target = A._TARGET
    assert target.shape == (64, 20)
    stops = [i for i, c in enumerate(A.CODONS) if A.GENETIC_CODE[c] == "*"]
    assert len(stops) == 3
    assert (target[stops] == 0).all()
    sense = np.delete(target, stops, axis=0)
    assert (sense.sum(axis=1) == 1).all() and set(np.unique(target)) == {0.0, 1.0}
    for i, codon in enumerate(A.CODONS):
        if A.GENETIC_CODE[codon] != "*":
            assert target[i].argmax() == A._AA_INDEX[A.GENETIC_CODE[codon]]
    assert (target.sum(axis=0) > 0).all(), "an amino acid no codon encodes"


def test_relative_asymmetry_on_worked_cases():
    """sum|Q-Q'| / sum(Q+Q'): 0 symmetric, 1 one-way, and 0.5 for rates 3 and 1."""
    one_way = np.zeros((4, 4)); one_way[0, 1] = 1.0
    assert A.relative_asymmetry(one_way) == pytest.approx(1.0, abs=1e-15)

    three_to_one = np.zeros((4, 4)); three_to_one[0, 1], three_to_one[1, 0] = 3.0, 1.0
    assert A.relative_asymmetry(three_to_one) == pytest.approx(4.0 / 8.0, abs=1e-15)

    symmetric = _off_diagonal(0.3)
    assert A.relative_asymmetry(symmetric) == 0.0

    two_pairs = np.zeros((4, 4))
    two_pairs[0, 1], two_pairs[1, 0] = 2.0, 0.0
    two_pairs[2, 3], two_pairs[3, 2] = 1.0, 1.0
    assert A.relative_asymmetry(two_pairs) == pytest.approx(4.0 / 8.0, abs=1e-15)   # (2+2) / (4+2+2)


def test_position_zero_is_refused_as_not_one_based(tmp_path):
    fasta = tmp_path / "root.fa"
    fasta.write_text(">r\nATGGCCAAA\n")
    with pytest.raises(ValueError, match="1-based integers"):
        A.reference_codons(fasta, [0])
    with pytest.raises(ValueError, match="1-based integers"):
        A.reference_codons(fasta, [1.5])
    assert A.reference_codons(fasta, [1, 3]) == ["ATG", "AAA"]


# ===========================================================================
# strata.primary_lineage             survivors: 80 (x2), 81, 82 (x2)
# ===========================================================================

def _lineage_frame(counts: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame({"lineage": [name for name, n in counts.items() for _ in range(n)]})


def test_the_lineage_with_the_most_scored_replacements_wins():
    frame = _lineage_frame({"A": 5, "B": 9, "C": 7})
    pairs = [("A", "reconstructed"), ("B", "reconstructed"), ("C", "reconstructed")]
    assert S.primary_lineage(frame, pairs, None) == "B"


def test_an_unverified_root_is_never_chosen_while_a_verified_one_exists_however_large():
    frame = _lineage_frame({"A": 50, "B": 9, "C": 7})
    pairs = [("A", "unverified"), ("B", "reconstructed"), ("C", "reconstructed")]
    assert S.primary_lineage(frame, pairs, None) == "B"


def test_when_every_root_is_unverified_the_largest_is_still_chosen():
    frame = _lineage_frame({"A": 5, "B": 9})
    assert S.primary_lineage(frame, [("A", "unverified"), ("B", "unverified")], None) == "B"


def test_a_tie_on_size_is_broken_by_the_sorted_identifier_whatever_the_guide_order():
    frame = _lineage_frame({"B": 4, "A": 4, "C": 4})
    pairs = [("C", "x"), ("B", "x"), ("A", "x")]
    assert S.primary_lineage(frame, pairs, None) == "A"


def test_a_lineage_the_table_does_not_hold_is_not_a_candidate():
    frame = _lineage_frame({"B": 3})
    assert S.primary_lineage(frame, [("A", "x"), ("B", "x")], None) == "B"


# ===========================================================================
# strata statistics                  survivors: 108, 324 (x2)
# ===========================================================================

def test_a_partial_correlation_with_a_perfectly_correlated_control_is_nan_not_infinite():
    """r_xz = 1 makes the denominator 0. Dividing anyway gives +-inf when r_xy != r_yz."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(S.partial_from_ranks(0.2, 1.0, 0.7))
        assert np.isnan(S.partial_from_ranks(0.2, 0.7, -1.0))


def test_a_partial_correlation_matches_the_textbook_value():
    r_xy, r_xz, r_yz = 0.6, 0.5, 0.4
    want = (0.6 - 0.5 * 0.4) / np.sqrt((1 - 0.25) * (1 - 0.16))
    assert S.partial_from_ranks(r_xy, r_xz, r_yz) == pytest.approx(want, abs=1e-15)


def test_an_additive_asymmetry_at_a_small_scale_is_still_fully_explained():
    """A share is scale-free. A guard that tested the total against 1, not 0, returned NaN for
    every asymmetry measured in tenths of a log unit."""
    first, second = np.triu_indices(6, k=1)
    score = np.array([0.003, -0.002, 0.005, 0.001, -0.004, 0.002])
    values = score[first] - score[second]
    assert (values ** 2).sum() < 1
    resid, share = S._additive_residual(values, first, second, 6)
    assert share == pytest.approx(1.0, abs=1e-9)
    assert np.abs(resid).max() < 1e-12


def test_an_all_zero_asymmetry_has_no_share_and_raises_no_warning():
    first, second = np.triu_indices(6, k=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resid, share = S._additive_residual(np.zeros(len(first)), first, second, 6)
    assert np.isnan(share) and (resid == 0).all()


# ===========================================================================
# strata.strata_table                survivors: 96, 178, 228, 229, 255, 257, 270, 278 (x4), 282 (x5)
# ===========================================================================

def _stratum_sizes(frame: pd.DataFrame, lineage: str) -> dict:
    return frame[(frame.model == "fine") & (frame.lineage == lineage)].nt_steps.value_counts().to_dict()


def _with_exact_rows(rng_seed: int, rows_in_stratum_3: int) -> pd.DataFrame:
    """Three lineages; lineage ``L2`` holds exactly ``rows_in_stratum_3`` rows at three steps.

    Steps are assigned by position in the table so both models see the same value, as
    ``wide_scores`` requires; everything else is spread over strata 1 and 2.
    """
    frame = make_replacement_frame(np.random.default_rng(rng_seed))
    fine = frame[frame.model == "fine"].index
    steps = np.where(np.arange(len(fine)) % 2 == 0, 1, 2)
    lineages = frame.loc[fine, "lineage"].to_numpy()
    mine = np.flatnonzero(lineages == "L2")[: rows_in_stratum_3]
    steps[mine] = 3
    # keep the other lineages' stratum 3 well populated
    others = np.flatnonzero(lineages != "L2")[::3]
    steps[others] = 3
    frame["nt_steps"] = np.tile(steps, 2)
    return frame


PROPS = ["neg_grantham"]


def test_a_stratum_of_exactly_twenty_rows_is_estimated_and_nineteen_is_refused_by_name():
    ok = _with_exact_rows(1, 20)
    assert _stratum_sizes(ok, "L2")[3] == 20
    table = S.strata_table(ok, ["fine", "base"], PROPS, n_boot=0)
    assert np.isfinite(table["estimate"]).all()

    thin = _with_exact_rows(1, 19)
    assert _stratum_sizes(thin, "L2")[3] == 19
    with pytest.raises(ValueError, match=r"lineage 'L2', stratum 3: fewer than 20 rows"):
        S.strata_table(thin, ["fine", "base"], PROPS, n_boot=0)


def test_with_two_thin_strata_the_error_names_the_first_in_lineage_then_stratum_order():
    """One statistic per block (a single model, no property) makes each offending block one
    entry, so naming the second instead of the first would show here and nowhere else."""
    frame = _with_exact_rows(1, 19)                       # L2's stratum 3 has 19 rows
    l3 = frame[(frame.lineage == "L3") & (frame.nt_steps == 3)].index[:-5]
    frame.loc[l3, "nt_steps"] = 1                         # and so does L3's, with 5
    with pytest.raises(ValueError) as caught:
        S.strata_table(frame, ["fine"], [], n_boot=0)
    assert "lineage 'L2', stratum 3" in str(caught.value) and "L3" not in str(caught.value)


def test_the_minimum_row_count_is_twenty():
    assert S.MIN_ROWS == 20


def test_the_default_draw_count_and_seed_of_the_site_bootstrap_are_the_documented_ones():
    params = inspect.signature(S.strata_table).parameters
    assert (params["n_boot"].default, params["seed"].default, params["level"].default) == (200, 20261009, 0.95)


@pytest.mark.parametrize("level, tails", [(0.95, [2.5, 97.5]), (0.8, [10.0, 90.0]), (0.5, [25.0, 75.0])])
def test_the_site_bootstrap_takes_the_percentiles_of_the_stated_level(monkeypatch, level, tails):
    seen = []
    real = np.nanpercentile
    monkeypatch.setattr(np, "nanpercentile", lambda a, q, **kw: seen.append(list(q)) or real(a, q, **kw))
    frame = make_replacement_frame(np.random.default_rng(3), lineages=("L1", "L2"))
    S.strata_table(frame, ["fine", "base"], PROPS, n_boot=6, level=level)
    assert len(seen) == 1 and seen[0] == pytest.approx(tails, abs=1e-9)


def test_the_cross_lineage_bootstrap_takes_the_percentiles_of_the_stated_level(monkeypatch):
    seen = []
    real = np.percentile
    monkeypatch.setattr(np, "percentile", lambda a, q, **kw: seen.append(list(q)) or real(a, q, **kw))
    M.bootstrap_mean_ci(DATA13, n_boot=10, level=0.9)
    assert seen == [pytest.approx([5.0, 95.0], abs=1e-9)]


def test_a_single_bootstrap_draw_still_gives_an_interval_and_zero_draws_gives_none():
    frame = make_replacement_frame(np.random.default_rng(4), lineages=("L1", "L2"))
    one = S.strata_table(frame, ["fine", "base"], PROPS, n_boot=1)
    assert one["lo"].notna().all() and (one["lo"] == one["hi"]).all()
    none = S.strata_table(frame, ["fine", "base"], PROPS, n_boot=0)
    assert none["lo"].isna().all() and none["hi"].isna().all()


def _bootstrap_with_valid_draws(monkeypatch, n_boot: int, n_valid: int):
    """Run the bootstrap with the first ``n_valid`` draws defined and the rest NaN everywhere."""
    frame = make_replacement_frame(np.random.default_rng(5), lineages=("L1", "L2"))
    real = S._strata_vectors
    calls = {"n": 0}

    def fake(blocks, strata, model_keys, properties, idx_per_block=None):
        out, stack = real(blocks, strata, model_keys, properties, idx_per_block)
        calls["n"] += 1
        if idx_per_block is not None and calls["n"] - 1 > n_valid:    # call 1 is the point estimate
            out = np.full_like(out, np.nan)
        return out, stack

    monkeypatch.setattr(S, "_strata_vectors", fake)
    return S.strata_table(frame, ["fine", "base"], PROPS, n_boot=n_boot)


@pytest.mark.filterwarnings("ignore:All-NaN slice")      # the planted undefined draws are the point
@pytest.mark.parametrize("n_boot, n_valid, raises", [
    (10, 5, False),     # exactly half defined: allowed (kills "<" -> "<=")
    (10, 4, True),      # one short of half: refused (kills "// 2" -> "// 3")
    (1, 0, True),       # a single draw that is undefined: refused (kills "max(1," -> "max(0,")
    (2, 1, False),      # one of two defined: allowed (kills "max(1," -> "max(2,")
    (3, 1, False),
    (3, 0, True),
])
def test_the_bootstrap_is_refused_exactly_when_fewer_than_half_the_draws_are_defined(
        monkeypatch, n_boot, n_valid, raises):
    if raises:
        with pytest.raises(ValueError, match="more than half the bootstrap replicates"):
            _bootstrap_with_valid_draws(monkeypatch, n_boot, n_valid)
    else:
        assert len(_bootstrap_with_valid_draws(monkeypatch, n_boot, n_valid)) > 0


# ===========================================================================
# strata.pair_asymmetry              survivors: 90, 96, 354
# ===========================================================================

AAS = list(S.AA_ORDER)


def _pair_frame(pairs, plm: dict, mut: dict) -> pd.DataFrame:
    """One row per ordered pair (a, b) in ``pairs`` and its reverse, one model, one site each."""
    rows = []
    for k, (a, b) in enumerate(pairs):
        for x, y in ((a, b), (b, a)):
            rows.append(dict(model="m", lineage="L", position=k, ref_aa=x, aa=y,
                             plm_prob=plm[(x, y)], mut_prob=mut[(x, y)]))
    return pd.DataFrame(rows)


def _random_probs(pairs, seed):
    rng = np.random.default_rng(seed)
    both = [(x, y) for a, b in pairs for x, y in ((a, b), (b, a))]
    return ({k: float(rng.uniform(1e-4, 0.3)) for k in both}, {k: float(rng.uniform(1e-4, 0.3)) for k in both})


ALL_PAIRS = list(itertools.combinations(AAS, 2))


def test_every_unordered_pair_of_the_twenty_amino_acids_appears_once_and_none_with_itself():
    assert len(ALL_PAIRS) == 190
    plm, mut = _random_probs(ALL_PAIRS, 1)
    table = S.pair_asymmetry(_pair_frame(ALL_PAIRS, plm, mut), "m")

    assert len(table) == 190
    assert (table.aa_from != table.aa_to).all()
    key = {tuple(sorted(p, key=S.AA_ORDER.index)) for p in zip(table.aa_from, table.aa_to)}
    assert key == set(ALL_PAIRS)


def test_the_asymmetry_is_log10_of_forward_over_reverse_in_the_published_order():
    plm, mut = _random_probs(ALL_PAIRS, 2)
    table = S.pair_asymmetry(_pair_frame(ALL_PAIRS, plm, mut), "m").set_index(["aa_from", "aa_to"])
    for (a, b), row in table.iterrows():
        assert S.AA_ORDER.index(a) < S.AA_ORDER.index(b)
        assert row.plm_asym == pytest.approx(np.log10(plm[(a, b)]) - np.log10(plm[(b, a)]), abs=1e-12)
        assert row.mut_asym == pytest.approx(np.log10(mut[(a, b)]) - np.log10(mut[(b, a)]), abs=1e-12)


def test_a_zero_probability_is_floored_at_1e_minus_32_before_the_log():
    plm, mut = _random_probs(ALL_PAIRS, 3)
    a, b = ALL_PAIRS[0]
    plm[(a, b)] = 0.0
    mut[(b, a)] = 0.0
    table = S.pair_asymmetry(_pair_frame(ALL_PAIRS, plm, mut), "m").set_index(["aa_from", "aa_to"])
    assert table.loc[(a, b), "plm_asym"] == pytest.approx(-32.0 - np.log10(plm[(b, a)]), abs=1e-12)
    assert table.loc[(a, b), "mut_asym"] == pytest.approx(np.log10(mut[(a, b)]) + 32.0, abs=1e-12)


def _connected_pairs(n_pairs: int):
    """``n_pairs`` unordered pairs, built from a clique on six amino acids plus spokes from a seventh."""
    clique = list(itertools.combinations(AAS[:6], 2))
    spokes = [(AAS[6], AAS[i]) for i in range(6)]
    return (clique + spokes)[:n_pairs]


def test_twenty_observed_pairs_are_enough_and_nineteen_are_not():
    ok = _connected_pairs(20)
    plm, mut = _random_probs(ok, 4)
    assert len(S.pair_asymmetry(_pair_frame(ok, plm, mut), "m")) == 20

    short = _connected_pairs(19)
    plm, mut = _random_probs(short, 4)
    with pytest.raises(ValueError, match="only 19 amino-acid pairs"):
        S.pair_asymmetry(_pair_frame(short, plm, mut), "m")
