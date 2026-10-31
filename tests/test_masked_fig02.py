"""Figure 2B and 2D recomputed on masked marginals: the swap, the gates, and the artefacts.

The published panels read the model's probability from a single unmasked forward pass over
the lineage root, so the model sees the residue it is scoring. ``dev/masked_fig02/`` builds
the masked-marginal counterpart for one lineage and recomputes the panels on it.

The arithmetic here runs on synthetic profiles and needs no GPU and no artefacts. The checks
that read the real masked profiles skip when those are absent, because they are a GPU
product and are not rebuilt by the test suite.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "unmasked_check"
LINEAGE = "J.2_int"
MODELS = ("ESM2_650M_HA80_raw", "ESM2_650M_HA80")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "dev" / "masked_fig02" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load("compare_fig02")


# ---------------------------------------------------------------------------
# Helpers: a tiny synthetic lineage
# ---------------------------------------------------------------------------
AA20 = list("ARNDCQEGHILKMFPSTWYV")


def write_profile(path: Path, sequence: str, matrix: np.ndarray) -> None:
    positions = list(range(1, len(sequence) + 1))
    head = pd.DataFrame([list(sequence)], index=["sequence"], columns=positions)
    body = pd.DataFrame(matrix, index=AA20, columns=positions)
    pd.concat([head, body], axis=0).to_csv(path, header=False)


def toy(tmp_path: Path, n_sites: int = 12, seed: int = 0):
    """A replacement table and the two profiles it is consistent with, for both models.

    Returns ``(frame, sequence, {model: (unmasked, masked)})``. The frame's ``plm_prob`` is
    read out of the unmasked profile, so the comparison script's first gate must pass on it.
    """
    rng = np.random.default_rng(seed)
    sequence = "".join(rng.choice(AA20, n_sites))
    profiles, rows = {}, []
    for model in MODELS:
        unmasked = rng.random((20, n_sites)) + 1e-3
        unmasked /= unmasked.sum(axis=0)
        masked = rng.random((20, n_sites)) + 1e-3
        masked /= masked.sum(axis=0)
        profiles[model] = (unmasked, masked)
        for j in range(n_sites):
            ref = sequence[j]
            for aa in AA20:
                if aa == ref:
                    continue
                rows.append(dict(
                    model=model, lineage=LINEAGE, position=j + 1, ref_aa=ref, aa=aa,
                    plm_prob=unmasked[AA20.index(aa), j],
                    mut_prob=float(10 ** rng.uniform(-14, -6)),
                    nt_steps=int(rng.integers(1, 4)),
                    blosum62=float(rng.integers(-4, 4)),
                    neg_grantham=float(-rng.uniform(5, 215)),
                ))
    frame = pd.DataFrame(rows)
    # The non-model columns are properties of the codon and the amino acids, so they must be
    # identical between models; the toy builder draws them twice, so copy the first model's.
    first = frame[frame.model == MODELS[0]].sort_values(["position", "aa"]).reset_index(drop=True)
    out = []
    for model in MODELS:
        sub = frame[frame.model == model].sort_values(["position", "aa"]).reset_index(drop=True)
        for column in ("mut_prob", "nt_steps", "blosum62", "neg_grantham"):
            sub[column] = first[column]
        out.append(sub)
    return pd.concat(out, ignore_index=True), sequence, profiles


def write_toy_profiles(tmp_path: Path, sequence, profiles) -> Path:
    for model, (unmasked, masked) in profiles.items():
        write_profile(tmp_path / f"{LINEAGE}__{model}__unmasked_profile.csv", sequence, unmasked)
        write_profile(tmp_path / f"{LINEAGE}__{model}__masked_profile.csv", sequence, masked)
    return tmp_path


# ---------------------------------------------------------------------------
# Profile I/O and the cell lookup
# ---------------------------------------------------------------------------

def test_read_profile_round_trips_a_written_one(tmp_path):
    rng = np.random.default_rng(3)
    sequence = "".join(rng.choice(AA20, 9))
    matrix = rng.random((20, 9))
    write_profile(tmp_path / "p.csv", sequence, matrix)
    got = compare.read_profile(tmp_path / "p.csv")
    assert list(got.index) == AA20
    assert list(got.columns) == list(range(1, 10))
    np.testing.assert_allclose(got.to_numpy(), matrix, atol=1e-12)


def test_lookup_reads_the_named_cell_not_its_transpose(tmp_path):
    """A transposed or off-by-one read is the failure that would silently pass everything
    downstream, so the cells are pinned by construction: cell (aa, pos) holds pos*100 + index."""
    sequence = "A" * 7
    matrix = np.array([[pos * 100 + i for pos in range(1, 8)] for i in range(20)], dtype=float)
    write_profile(tmp_path / "p.csv", sequence, matrix)
    profile = compare.read_profile(tmp_path / "p.csv")
    frame = pd.DataFrame([dict(aa="C", position=3), dict(aa="V", position=7), dict(aa="A", position=1)])
    np.testing.assert_allclose(compare.lookup(frame, profile),
                               [300 + AA20.index("C"), 700 + AA20.index("V"), 100 + AA20.index("A")])


def test_lookup_rejects_an_unknown_amino_acid(tmp_path):
    sequence = "A" * 4
    write_profile(tmp_path / "p.csv", sequence, np.ones((20, 4)))
    profile = compare.read_profile(tmp_path / "p.csv")
    with pytest.raises(ValueError, match="absent from the profile"):
        compare.lookup(pd.DataFrame([dict(aa="X", position=1)]), profile)


def test_lookup_rejects_a_position_off_the_end(tmp_path):
    sequence = "A" * 4
    write_profile(tmp_path / "p.csv", sequence, np.ones((20, 4)))
    profile = compare.read_profile(tmp_path / "p.csv")
    with pytest.raises(ValueError, match="outside the profile"):
        compare.lookup(pd.DataFrame([dict(aa="C", position=5)]), profile)


# ---------------------------------------------------------------------------
# Gate 1: the published column must be reproduced before anything is swapped
# ---------------------------------------------------------------------------

def test_swap_passes_its_gate_and_changes_only_plm_prob(tmp_path):
    frame, sequence, profiles = toy(tmp_path)
    write_toy_profiles(tmp_path, sequence, profiles)
    masked, gate = compare.swap_in_masked(frame, tmp_path, LINEAGE, list(MODELS), tol=1e-9)

    assert set(gate) == set(MODELS) and max(gate.values()) < 1e-9
    compare.check_untouched(frame, masked)                  # must not raise
    for model in MODELS:
        want = profiles[model][1]
        sub = masked[masked.model == model]
        got = want[[AA20.index(a) for a in sub.aa], sub.position.to_numpy() - 1]
        np.testing.assert_allclose(sub.plm_prob.to_numpy(), got, atol=1e-12)


def test_swap_refuses_when_the_published_column_is_not_reproduced(tmp_path):
    frame, sequence, profiles = toy(tmp_path)
    write_toy_profiles(tmp_path, sequence, profiles)
    nudged = frame.copy()
    nudged.loc[nudged.index[0], "plm_prob"] += 1e-3
    with pytest.raises(SystemExit, match="GATE FAILED"):
        compare.swap_in_masked(nudged, tmp_path, LINEAGE, list(MODELS), tol=1e-9)


def test_swap_refuses_when_a_profile_is_missing(tmp_path):
    frame, sequence, profiles = toy(tmp_path)
    write_toy_profiles(tmp_path, sequence, profiles)
    (tmp_path / f"{LINEAGE}__{MODELS[0]}__masked_profile.csv").unlink()
    with pytest.raises(SystemExit, match="run score_masked_profile.py first"):
        compare.swap_in_masked(frame, tmp_path, LINEAGE, list(MODELS), tol=1e-9)


# ---------------------------------------------------------------------------
# Gate 2: nothing but the model may move
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("column", ["mut_prob", "nt_steps", "blosum62", "neg_grantham", "position"])
def test_check_untouched_catches_any_other_column_moving(tmp_path, column):
    frame, sequence, profiles = toy(tmp_path)
    write_toy_profiles(tmp_path, sequence, profiles)
    masked, _ = compare.swap_in_masked(frame, tmp_path, LINEAGE, list(MODELS), tol=1e-9)
    broken = masked.copy()
    # An additive nudge, not ``x * 2 + 1``: that form leaves -1 unchanged, and blosum62 is
    # drawn from small integers, so the test would pass itself whenever it drew one.
    broken.loc[broken.index[0], column] = broken.loc[broken.index[0], column] + 7
    with pytest.raises(SystemExit, match=f"column {column!r} changed"):
        compare.check_untouched(frame, broken)


def test_check_untouched_catches_a_non_positive_probability(tmp_path):
    frame, sequence, profiles = toy(tmp_path)
    write_toy_profiles(tmp_path, sequence, profiles)
    masked, _ = compare.swap_in_masked(frame, tmp_path, LINEAGE, list(MODELS), tol=1e-9)
    masked.loc[masked.index[0], "plm_prob"] = 0.0
    with pytest.raises(SystemExit, match="non-positive"):
        compare.check_untouched(frame, masked)


def test_check_untouched_catches_a_dropped_row(tmp_path):
    frame, sequence, profiles = toy(tmp_path)
    write_toy_profiles(tmp_path, sequence, profiles)
    masked, _ = compare.swap_in_masked(frame, tmp_path, LINEAGE, list(MODELS), tol=1e-9)
    with pytest.raises(SystemExit, match="row count changed"):
        compare.check_untouched(frame, masked.iloc[1:])


# ---------------------------------------------------------------------------
# The panel statistics the comparison reports
# ---------------------------------------------------------------------------

class _Model:
    def __init__(self, key, role):
        self.key, self.role = key, role


class _Guide:
    name = "toy"
    models = [_Model(MODELS[0], "base"), _Model(MODELS[1], "finetune")]


def test_panel_ab_deltas_are_masked_minus_unmasked(tmp_path):
    frame, sequence, profiles = toy(tmp_path, n_sites=40, seed=7)
    write_toy_profiles(tmp_path, sequence, profiles)
    masked, _ = compare.swap_in_masked(frame, tmp_path, LINEAGE, list(MODELS), tol=1e-9)
    fig02ab = _load_figure_module("make_fig02ab")

    table = compare.panel_ab(fig02ab, frame, masked, _Guide(), LINEAGE)
    assert set(table.panel) == {"A", "B"}
    np.testing.assert_allclose(table.spearman_delta, table.spearman_masked - table.spearman_unmasked)
    np.testing.assert_allclose(table.pearson_delta, table.pearson_masked - table.pearson_unmasked)
    for _, row in table.iterrows():
        want = fig02ab.correlations(frame[frame.model == row.model])
        assert row.spearman_unmasked == pytest.approx(want["spearman_r"])


def _load_figure_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "figures" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_panel_d_flags_estimates_outside_the_published_interval(tmp_path):
    """The ``outside?`` column is the comparison's only judgement, so it is pinned directly."""
    frame, sequence, profiles = toy(tmp_path, n_sites=60, seed=11)
    write_toy_profiles(tmp_path, sequence, profiles)
    masked, _ = compare.swap_in_masked(frame, tmp_path, LINEAGE, list(MODELS), tol=1e-9)

    table = compare.panel_d(frame, masked, _Guide(), LINEAGE, n_boot=30, seed=5)
    np.testing.assert_allclose(table.estimate_delta, table.estimate_masked - table.estimate_unmasked)
    expected = (table.estimate_masked < table.lo_unmasked) | (table.estimate_masked > table.hi_unmasked)
    assert (table.outside_published_interval == expected).all()
    assert (table.lo_unmasked <= table.hi_unmasked).all()


# ---------------------------------------------------------------------------
# Gate 3: the unmasked arm must reproduce what the panels shipped
# ---------------------------------------------------------------------------

def _shipped_pair(tmp_path: Path):
    """A fake ``output/`` holding values files that agree with a tiny comparison pair."""
    ab = pd.DataFrame([dict(panel="B", model=MODELS[1], role="finetune", lineage=LINEAGE, n=5,
                            spearman_unmasked=0.5, spearman_masked=0.4, spearman_delta=-0.1,
                            pearson_unmasked=0.2, pearson_masked=0.1, pearson_delta=-0.1)])
    d = pd.DataFrame([dict(measure="plm_vs_access", series=MODELS[1], control="", stratum="all",
                           estimate_unmasked=0.5, estimate_masked=0.4)])
    out = tmp_path / "output" / "unmasked_check"
    out.mkdir(parents=True)
    pd.DataFrame([
        dict(panel="B", series=MODELS[1], group="all", measure="spearman_r", value=0.5),
        dict(panel="B", series=MODELS[1], group="all", measure="pearson_r", value=0.2),
    ]).to_csv(out.parent / "fig02ab_scatter_panels_values.tsv", sep="\t", index=False)
    pd.DataFrame([
        dict(measure="plm_vs_access", series=MODELS[1], control="", stratum="all", estimate=0.5),
    ]).to_csv(out.parent / "fig02d_strata_panel_values.tsv", sep="\t", index=False)
    return ab, d, out


def test_shipped_gate_passes_when_the_unmasked_arm_agrees(tmp_path):
    ab, d, out = _shipped_pair(tmp_path)
    worst = compare.gate_against_shipped(ab, d, out, tol=1e-9)
    assert set(worst) == {"fig02ab", "fig02d"} and max(worst.values()) < 1e-12


def test_shipped_gate_refuses_when_the_panel_disagrees(tmp_path):
    ab, d, out = _shipped_pair(tmp_path)
    ab.loc[0, "spearman_unmasked"] = 0.6
    with pytest.raises(SystemExit, match="does not reproduce the shipped panel values"):
        compare.gate_against_shipped(ab, d, out, tol=1e-9)


def test_shipped_gate_refuses_a_partial_match(tmp_path):
    ab, d, out = _shipped_pair(tmp_path)
    ab.loc[0, "model"] = "some_other_model"
    with pytest.raises(SystemExit, match="panel A/B values matched"):
        compare.gate_against_shipped(ab, d, out, tol=1e-9)


def test_shipped_gate_skips_a_panel_with_no_values_file(tmp_path):
    ab, d, out = _shipped_pair(tmp_path)
    (out.parent / "fig02d_strata_panel_values.tsv").unlink()
    assert set(compare.gate_against_shipped(ab, d, out, tol=1e-9)) == {"fig02ab"}


# ---------------------------------------------------------------------------
# The real artefacts, when they are present
# ---------------------------------------------------------------------------
needs_artefacts = pytest.mark.skipif(
    not all((OUT / f"{LINEAGE}__{m}__{k}_profile.csv").exists() for m in MODELS
            for k in ("masked", "unmasked")),
    reason="masked profiles are a GPU product; run dev/masked_fig02/score_masked_profile.py",
)


@needs_artefacts
@pytest.mark.parametrize("model", MODELS)
def test_real_masked_profile_is_a_probability_matrix(model):
    profile = compare.read_profile(OUT / f"{LINEAGE}__{model}__masked_profile.csv")
    values = profile.to_numpy()
    assert values.shape == (20, 566)
    assert np.isfinite(values).all()
    assert (values > 0).all() and (values < 1).all()
    sums = values.sum(axis=0)
    # Sliced from a softmax over the full 33-token vocabulary and deliberately not
    # renormalised, so the columns sum to at most one, and in practice to almost exactly one.
    assert (sums <= 1.0 + 1e-6).all()
    assert sums.min() > 0.99


@needs_artefacts
@pytest.mark.parametrize("model", MODELS)
def test_masking_actually_changed_the_numbers(model):
    unmasked = compare.read_profile(OUT / f"{LINEAGE}__{model}__unmasked_profile.csv").to_numpy()
    masked = compare.read_profile(OUT / f"{LINEAGE}__{model}__masked_profile.csv").to_numpy()
    changed = ~np.isclose(masked, unmasked, rtol=1e-3, atol=1e-9)
    assert changed.mean() > 0.9, "masking barely moved the profile; the mask may not be landing"


@needs_artefacts
@pytest.mark.parametrize("model", MODELS)
def test_unmasked_profile_leaks_the_wild_type_residue(model):
    """The reason the two estimators differ at all: with the residue visible the model mostly
    copies it. This is the direction of the effect, so it is asserted rather than assumed."""
    path = OUT / f"{LINEAGE}__{model}__unmasked_profile.csv"
    raw = pd.read_csv(path, header=None, index_col=0)
    sequence = "".join(raw.loc["sequence"].astype(str))
    unmasked = compare.read_profile(path).to_numpy()
    masked = compare.read_profile(OUT / f"{LINEAGE}__{model}__masked_profile.csv").to_numpy()
    rows = [AA20.index(a) for a in sequence]
    cols = np.arange(len(sequence))
    assert unmasked[rows, cols].mean() > masked[rows, cols].mean()


@needs_artefacts
def test_provenance_records_a_passed_gate():
    manifest = json.loads((OUT / "fig02_masked_provenance.json").read_text())
    assert manifest["lineage"] == LINEAGE
    assert manifest["estimator"]["renormalised_over_20"] is False
    for model in MODELS:
        entry = manifest["models"][model]
        assert entry["gate_passed"] is True
        assert entry["gate_max_abs_diff"] <= manifest["tol"]
        assert entry["sequence_length"] == 566


@needs_artefacts
def test_reported_panel_b_correlation_still_reproduces():
    """The number in ``FIG02_REPORT.md`` must be the one the panel's own function gives today."""
    table = pd.read_csv(OUT / "fig02ab_estimator_comparison.tsv", sep="\t")
    row = table[table.panel == "B"].iloc[0]
    assert row.n == 10754
    assert row.spearman_unmasked == pytest.approx(0.655, abs=0.001)
    assert row.spearman_masked == pytest.approx(0.642, abs=0.001)

    report = (OUT / "FIG02_REPORT.md").read_text()
    assert f"{row.spearman_unmasked:.3f} -> {row.spearman_masked:.3f}" in report


@needs_artefacts
def test_real_unmasked_arm_matches_the_shipped_panels():
    """The published panels and this comparison's baseline must be the same numbers."""
    ab = pd.read_csv(OUT / "fig02ab_estimator_comparison.tsv", sep="\t")
    d = pd.read_csv(OUT / "fig02d_estimator_comparison.tsv", sep="\t")
    worst = compare.gate_against_shipped(ab, d, OUT, tol=1e-5)
    assert set(worst) == {"fig02ab", "fig02d"}


@needs_artefacts
def test_panel_d_comparison_covers_every_stratum_and_series():
    table = pd.read_csv(OUT / "fig02d_estimator_comparison.tsv", sep="\t").fillna({"control": ""})
    plm = table[table.measure == "plm_vs_access"]
    assert set(plm.stratum.astype(str)) == {"all", "1", "2", "3"}
    assert set(plm.series) == set(MODELS)
    assert set(plm.control) == {"", "neg_grantham", "blosum62"}
    # The headline of the comparison: masking moves the fine-tuned model's within-stratum
    # correlation down, and far enough at three nucleotide changes to leave the interval.
    three = plm[(plm.stratum.astype(str) == "3") & (plm.series == MODELS[1]) & (plm.control == "")]
    assert len(three) == 1
    assert three.iloc[0].estimate_delta < -0.05
    assert bool(three.iloc[0].outside_published_interval)
