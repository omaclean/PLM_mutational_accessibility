"""Pins the explanation for the lowest-scoring spectrum in S06.

The puzzle: S05 says the two influenza matrices predict each other at 83-85% of variance,
yet S06 puts the second one LAST of the measured spectra, in every stratum and for both
models. These tests pin the chain that resolves it, including the two explanations that
turned out to be WRONG, so that nobody re-derives them and writes them into a caption.

Working: dev/spectrum_swap_anomaly/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mutacc import accessibility as acc  # noqa: E402
from mutacc import spectrum_swap as sw  # noqa: E402

SPECTRA = REPO_ROOT / "data" / "spectra"
BASES = "ACGT"
OFFDIAG = [(i, j) for i in range(4) for j in range(4) if i != j]
NAMES = [f"{BASES[i]}>{BASES[j]}" for i, j in OFFDIAG]
REFERENCE, OTHER = "influenza_h3n2", "influenza_h1n1"
PANEL = {
    "influenza_h3n2": "Influenza H3N2 (Pauly et al.)", "influenza_h1n1": "Influenza H1N1",
    "sars_cov_2": "SARS-CoV-2", "human": "Human germline", "ecoli": "E. coli",
    "yeast": "Yeast", "k80_kappa2": "Transitions twice as fast (K80)",
    "jc69": "All changes equal (JC69)",
}


def load(name):
    return acc.load_spectrum(SPECTRA / f"{name}.csv")


def rates(q):
    return pd.Series([q[i, j] for i, j in OFFDIAG], index=NAMES, dtype=float)


def shares(q):
    r = rates(q)
    return r / r.sum()


@pytest.fixture(scope="module")
def universe():
    code = acc._standard_code()
    codons = sorted(c for c, a in code.items() if a != "*")
    aas = sorted({a for a in code.values() if a != "*"})
    rows = [(c, a) for c in codons for a in aas if code[c] != a]
    steps = np.array([acc.nucleotide_steps(c, a) for c, a in rows])
    return rows, steps


def access(rows, name, scale=None):
    codons = [c for c, _ in rows]
    aas = [a for _, a in rows]
    scale = sw.mean_row_sum(load(REFERENCE)) if scale is None else scale
    return sw.merge_rounding_ties(sw.condition_accessibility(codons, aas, load(name), scale))


# ---------------------------------------------------------------------------
# The premise: the two matrices really are alike, but only in one sense
# ---------------------------------------------------------------------------

def test_the_two_matrices_agree_on_values_but_not_on_ranks():
    a, b = shares(load(REFERENCE)), shares(load(OTHER))
    assert pearsonr(a, b)[0] > 0.97, "the premise of the puzzle: by value they are nearly the same"
    assert spearmanr(a, b)[0] < 0.85, "by rank they are not; this is what the puzzle turns on"


def test_the_value_similarity_is_carried_by_the_two_dominant_transitions():
    a, b = shares(load(REFERENCE)), shares(load(OTHER))
    big = ["A>G", "T>C"]
    assert a[big].sum() > 0.6 and b[big].sum() > 0.6
    rest_a, rest_b = a.drop(big), b.drop(big)
    rest_a, rest_b = rest_a / rest_a.sum(), rest_b / rest_b.sum()
    assert pearsonr(rest_a, rest_b)[0] < 0.7, (
        "drop the two shared transitions and the agreement largely goes; a variance score over "
        "all twelve cells is reading the two cells the viruses have in common"
    )


# ---------------------------------------------------------------------------
# Not a parsing fault
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wrong", ["transpose", "complement", "symmetrise"])
def test_no_misreading_of_the_file_fits_the_reference_better(wrong):
    q, ref = load(OTHER), shares(load(REFERENCE))
    as_read = pearsonr(ref, shares(q))[0]
    altered = pearsonr(ref, shares(getattr(acc, wrong)(q)))[0]
    assert altered < as_read, f"{wrong} fits better than the file as read; suspect the parser"


def test_the_identity_base_order_is_the_best_relabelling():
    q, ref = load(OTHER), shares(load(REFERENCE))
    best = max(acc.all_relabellings(), key=lambda p: pearsonr(ref, shares(acc.relabel(q, p)))[0])
    assert best == (0, 1, 2, 3), f"column order looks wrong: {best} fits better than identity"


# ---------------------------------------------------------------------------
# The two explanations that are WRONG. Kept so they are not proposed again.
# ---------------------------------------------------------------------------

def test_refuted_agreement_with_the_reference_does_not_explain_the_ordering(universe):
    """It agrees with the reference BEST of any alternative, and still scores nearly last."""
    rows, steps = universe
    one = steps == 1
    ref = access(rows, REFERENCE)[one]
    agree = {label: spearmanr(access(rows, key)[one], ref)[0] for key, label in PANEL.items()}
    others = {k: v for k, v in agree.items() if k != PANEL[REFERENCE]}
    assert max(others, key=others.get) == PANEL[OTHER], (
        "the spectrum that scores lowest is the one that agrees with the reference most; "
        "agreement cannot be what the panel reads"
    )


def test_refuted_resolution_does_not_explain_the_ordering(universe):
    """Tied/compressed accessibility does not order the panel either."""
    rows, steps = universe
    one = steps == 1
    rho = _figure_rho()
    distinct = {}
    for key, label in PANEL.items():
        if label not in rho:
            continue
        distinct[label] = len(np.unique(access(rows, key)[one]))
    labels = sorted(distinct)
    r = spearmanr([distinct[k] for k in labels], [rho[k] for k in labels])[0]
    assert abs(r) < 0.5, f"resolution explains more than expected (rho {r:+.3f}); revisit the caption"


# ---------------------------------------------------------------------------
# The explanation that holds
# ---------------------------------------------------------------------------

def _figure_rho(model="ESM2_650M_HA80", stratum="1"):
    t = pd.read_csv(REPO_ROOT / "output" / "sF_spectrum_swap_values.tsv", sep="\t")
    t = t[(t["series"] == model) & (t["stratum"].astype(str) == stratum)]
    return dict(zip(t["label"], t["estimate"]))


@pytest.fixture(scope="module")
def model_preference():
    """Mean percentile of the model's score, by the base change a one-step replacement needs."""
    from mutacc.guide import load_guide
    guide = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
    s = guide.read_table("replacement_scores")
    s = s[(s["model"] == "ESM2_650M_HA80") & (s["nt_steps"] == 1)].copy()
    s["codon"] = sw.row_codons(s, {lin.id: lin.root for lin in guide.lineages})
    pairs = s[["codon", "aa"]].drop_duplicates()
    pairs["change"] = [
        (lambda o: f"{o[0][0]}>{o[0][1]}" if len(o) == 1 else None)(acc.single_step_changes(c, a))
        for c, a in zip(pairs["codon"], pairs["aa"])
    ]
    s = s.merge(pairs, on=["codon", "aa"], how="left")
    s = s[s["change"].notna()].copy()
    s["pct"] = s.groupby("lineage")["plm_prob"].rank(pct=True)
    return s.groupby("change")["pct"].mean().reindex([c for c in NAMES])


def test_the_model_prefers_transitions_over_transversions(model_preference):
    p = model_preference
    assert p.idxmax() == "A>G", f"expected A>G to top the model's preference, got {p.idxmax()}"
    assert p.idxmin() == "G>C", f"expected G>C to sit last, got {p.idxmin()}"
    assert p["G>A"] > 0.6 > p["T>G"]


def test_matching_the_models_preference_predicts_the_panel(model_preference):
    rho = _figure_rho()
    pairs = []
    for key, label in PANEL.items():
        r = rates(load(key))
        if r.nunique() == 1 or label not in rho:   # JC69 has no ordering to match
            continue
        pairs.append((spearmanr(r.reindex(model_preference.index), model_preference)[0], rho[label]))
    x, y = zip(*pairs)
    assert spearmanr(x, y)[0] > 0.9, (
        "how well a spectrum's rate order matches the model's per-change preference is what "
        f"this panel reads; got Spearman {spearmanr(x, y)[0]:.3f} over {len(pairs)} spectra"
    )


def test_the_lowest_scoring_spectrum_is_the_one_that_matches_the_preference_worst(model_preference):
    scores = {}
    for key, label in PANEL.items():
        r = rates(load(key))
        if r.nunique() == 1:
            continue
        scores[label] = spearmanr(r.reindex(model_preference.index), model_preference)[0]
    assert min(scores, key=scores.get) == PANEL[OTHER]
    assert scores[PANEL[OTHER]] < 0.2 < scores[PANEL[REFERENCE]]


def test_the_gap_is_carried_by_the_minor_cells_not_by_one_of_them(model_preference):
    """No single rate explains it; G>A leads, then G>C, then C>T."""
    q_other, q_ref = load(OTHER), load(REFERENCE)

    def match(q):
        return spearmanr(rates(q).reindex(model_preference.index), model_preference)[0]

    base, target = match(q_other), match(q_ref)
    recovered = {}
    for k, (i, j) in enumerate(OFFDIAG):
        q = q_other.copy()
        q[i, j] = q_ref[i, j]
        recovered[NAMES[k]] = (match(q) - base) / (target - base)
    order = sorted(recovered, key=recovered.get, reverse=True)
    assert order[:3] == ["G>A", "G>C", "C>T"], f"attribution changed: {order[:3]}"
    assert recovered["G>A"] < 0.5, "no single cell should explain the gap on its own"
    assert sum(recovered[c] for c in order[:3]) > 0.6


def test_the_caption_states_the_mechanism():
    text = " ".join((REPO_ROOT / "output" / "sF_spectrum_swap_caption.md").read_text().split())
    for phrase in ("does not", "agreement with the reference spectrum", "G>A", "G>C",
                   "preference", "dev/spectrum_swap_anomaly"):
        assert phrase in text, f"the caption no longer explains the mechanism: missing {phrase!r}"
