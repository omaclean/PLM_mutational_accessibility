"""Carpet-bombing tests for core mutacc mathematical edge cases, invariants, and algorithms.

Tests:
1. mutacc.strata: partial_from_ranks, _additive_residual exact least-squares recovery, pair asymmetry.
2. mutacc.symmetry: favoured_classes scaling invariance, reciprocal ties, orientation variants mathematical properties.
3. mutacc.accessibility: genetic code translation invariants, codon Hamming distance, stop codons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ==============================================================================
# 1. mutacc.strata mathematical properties
# ==============================================================================

def test_partial_from_ranks():
    from mutacc.strata import partial_from_ranks

    # Case 1: When x and y are identical, partial correlation is 1.0
    r_xy = 1.0
    r_xz = 0.5
    r_yz = 0.5
    assert np.isclose(partial_from_ranks(r_xy, r_xz, r_yz), 1.0)

    # Case 2: When x and y correlation is entirely mediated by z:
    # r_xy = r_xz * r_yz => partial correlation must be 0.0
    r_xz = 0.6
    r_yz = 0.5
    r_xy = r_xz * r_yz  # 0.30
    assert np.isclose(partial_from_ranks(r_xy, r_xz, r_yz), 0.0)

    # Case 3: When z is completely uncorrelated with both (r_xz = r_yz = 0)
    # partial correlation must equal r_xy
    assert np.isclose(partial_from_ranks(0.42, 0.0, 0.0), 0.42)


def test_additive_residual_exact_recovery():
    from mutacc.strata import _additive_residual, AA_ORDER

    n_aa = len(AA_ORDER)
    first, second = np.triu_indices(n_aa, k=1)

    # Plant purely additive scores s_i
    rng = np.random.default_rng(42)
    true_scores = rng.normal(0, 1, size=n_aa)

    # Values are pure difference s_i - s_j
    values = true_scores[first] - true_scores[second]

    resid, share = _additive_residual(values, first, second, n_aa)

    # Since values are purely additive, residual must be identically zero
    assert np.allclose(resid, 0.0, atol=1e-12), "Additive residual should be 0 for pure additive signal"
    assert np.isclose(share, 1.0), "Explained share must be 1.0 for pure additive signal"

    # Now add orthogonal noise
    noise = rng.normal(0, 0.5, size=len(values))
    # Project noise orthogonal to design
    design = np.zeros((len(values), n_aa))
    design[np.arange(len(values)), first] = 1.0
    design[np.arange(len(values)), second] = -1.0
    coef, *_ = np.linalg.lstsq(design, noise, rcond=None)
    pure_noise = noise - design @ coef

    noisy_values = values + pure_noise
    resid_noisy, share_noisy = _additive_residual(noisy_values, first, second, n_aa)

    assert 0.0 < share_noisy < 1.0
    assert np.allclose(resid_noisy, pure_noise, atol=1e-12)


# ==============================================================================
# 2. mutacc.symmetry mathematical properties & edge cases
# ==============================================================================

def test_favoured_classes_scaling_invariance():
    from mutacc.symmetry import favoured_classes

    # Create an asymmetric 4x4 matrix
    q = np.array([
        [0.0, 5.0, 2.0, 1.0],  # A->C > C->A
        [1.0, 0.0, 4.0, 3.0],  # C->G > G->C
        [2.0, 2.0, 0.0, 6.0],  # G->T < T->G
        [1.0, 3.0, 8.0, 0.0],
    ])

    fav_raw, dis_raw = favoured_classes(q)
    assert len(fav_raw) > 0

    # Scale by 1e-8 (like human mutation spectrum)
    fav_scaled, dis_scaled = favoured_classes(q * 1.2e-8)
    assert fav_raw == fav_scaled, "Scaling must not alter favoured substitution classes"
    assert dis_raw == dis_scaled, "Scaling must not alter disfavoured substitution classes"


def test_favoured_classes_symmetric_matrix():
    from mutacc.symmetry import favoured_classes

    # Symmetric matrix: all reciprocal rates equal
    q_sym = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 4.0, 5.0],
        [2.0, 4.0, 0.0, 6.0],
        [3.0, 5.0, 6.0, 0.0],
    ])

    fav, dis = favoured_classes(q_sym)
    assert len(fav) == 0, "Symmetric matrix must have empty favoured classes"
    assert len(dis) == 0, "Symmetric matrix must have empty disfavoured classes"


def test_orientation_variants_properties():
    from mutacc.symmetry import orientation_variants

    q = np.array([
        [0.0, 4.0, 1.0, 2.0],
        [2.0, 0.0, 5.0, 3.0],
        [1.5, 3.5, 0.0, 7.0],
        [2.2, 1.1, 4.4, 0.0],
    ])

    variants = orientation_variants(q)
    assert set(variants.keys()) == {"direct", "symmetrised", "transposed", "complemented"}

    direct = variants["direct"]
    symmetrised = variants["symmetrised"]
    transposed = variants["transposed"]
    complemented = variants["complemented"]

    # 1. Symmetrised is strictly symmetric
    assert np.allclose(symmetrised, symmetrised.T)
    assert np.allclose(symmetrised, 0.5 * (direct + direct.T))

    # 2. Transposed is strictly transpose of direct
    assert np.allclose(transposed, direct.T)

    # 3. Complemented permutes A<->T and C<->G
    # Bases: A=0, C=1, G=2, T=3 => comp: A->T (0->3), C->G (1->2), G->C (2->1), T->A (3->0)
    perm = [3, 2, 1, 0]
    expected_comp = direct[np.ix_(perm, perm)]
    assert np.allclose(complemented, expected_comp)


# ==============================================================================
# 3. mutacc.accessibility invariants
# ==============================================================================

def test_nucleotide_steps_invariants():
    from mutacc.accessibility import nucleotide_steps

    # Test identical codon
    assert nucleotide_steps("AAA", "K") == 0
    assert nucleotide_steps("ATG", "M") == 0

    # Test 1-nt change
    assert nucleotide_steps("ATG", "I") == 1  # ATG -> ATA (Met to Ile)
    assert nucleotide_steps("GAA", "D") == 1  # GAA -> GAC/GAT (Glu to Asp)

    # Test 2-nt change
    assert nucleotide_steps("ATG", "P") == 2  # ATG -> CCG (Met to Pro: 2 nt)

    # Test 3-nt change (Lys has only AAA and AAG, both differ from TTT at all 3 positions)
    assert nucleotide_steps("TTT", "K") == 3
