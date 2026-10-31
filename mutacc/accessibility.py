"""Mutational accessibility from a 4x4 nucleotide spectrum, and the variants of a spectrum.

*Accessibility* of replacing a reference codon by a target amino acid is the probability that
the codon changes into any codon of that amino acid, with each of the three codon positions
changing independently under the nucleotide matrix. It is the quantity the pipeline calls
``mut_prob``; ``tests/test_accessibility.py`` checks that this module reproduces the shipped
``mut_prob`` column to floating-point precision.

Spectra are 4x4 matrices ``Q[i, j]`` = rate of changing base ``i`` into base ``j`` on the **coding**
(+) strand, in the order A, C, G, T. Pauly et al. (2017), the source of the influenza matrices,
reports rates in the coding sense, so no strand flip is needed for them.

The variants below exist to ask whether a model, or any score, tracks the *direction* of a
spectrum and not just its overall shape:

``transpose``    rate of j into i. Same rates, every direction reversed.
``symmetrise``   (Q + Q^T) / 2. Direction averaged away; keeps transition/transversion structure.
``complement``   the same spectrum read on the opposite strand, Q'[i, j] = Q[c(i), c(j)].
``relabel``      bases renamed by a permutation. The 24 relabellings include the identity.
``shuffle``      the twelve off-diagonal rates permuted at random; a null with the same rate values.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "BASES", "CODONS", "AA20", "GENETIC_CODE",
    "load_spectrum", "as_probabilities", "codon_aa_table", "codon_index", "aa_index",
    "accessibility", "nucleotide_steps", "single_step_changes",
    "transpose", "symmetrise", "complement", "relabel", "shuffle", "all_relabellings",
    "reference_codons", "relative_asymmetry",
]

BASES = "ACGT"
CODONS = ["".join(t) for t in itertools.product(BASES, repeat=3)]
_BASES_ARR = np.array(list(BASES))

def _standard_code() -> dict[str, str]:
    """NCBI translation table 1, taken from Biopython rather than typed in."""
    from Bio.Data import CodonTable

    table = CodonTable.unambiguous_dna_by_id[1]
    code = dict(table.forward_table)
    code.update({c: "*" for c in table.stop_codons})
    return {c: code[c] for c in CODONS}


#: Standard genetic code, keyed in the same ACGT-product order as CODONS.
GENETIC_CODE = _standard_code()
AA20 = "ACDEFGHIKLMNPQRSTVWY"

_CODON_INDEX = {c: i for i, c in enumerate(CODONS)}
_AA_INDEX = {a: i for i, a in enumerate(AA20)}


def codon_index(codon: str) -> int:
    return _CODON_INDEX[codon]


def aa_index(aa: str) -> int:
    return _AA_INDEX[aa]


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def load_spectrum(path: str | Path) -> np.ndarray:
    """Read a 4x4 spectrum CSV (header ``from_base,A,C,G,T``). Diagonal is zeroed."""
    frame = pd.read_csv(path, index_col=0)
    try:
        q = frame.loc[list(BASES), list(BASES)].to_numpy(dtype=float)
    except KeyError as exc:
        raise ValueError(f"{path}: need rows and columns {list(BASES)}") from exc
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError(f"{path}: rates must be finite and non-negative")
    np.fill_diagonal(q, 0.0)
    if not q.any():
        raise ValueError(f"{path}: all rates are zero")
    return q


def as_probabilities(q: np.ndarray, *, row_sum: float | None = None) -> np.ndarray:
    """Per-site change probabilities with the stay-probability on the diagonal.

    A matrix whose rows already sum to less than one is used as given. Matrices on a relative
    scale (JC69, K80, the cellular spectra) are rescaled so the *mean* row sum is ``row_sum``;
    pass it explicitly. Within a nucleotide-step stratum the stay-probabilities are close to 1
    and rankings do not depend on the scale; across strata they do.
    """
    q = np.array(q, dtype=float)
    np.fill_diagonal(q, 0.0)
    sums = q.sum(axis=1)
    if row_sum is not None:
        q = q * (row_sum / sums.mean())
        sums = q.sum(axis=1)
    if (sums >= 1.0).any():
        raise ValueError(
            f"row sums {np.round(sums, 4).tolist()} are not per-site probabilities (< 1); "
            "pass row_sum= to rescale a relative matrix"
        )
    p = q.copy()
    p[np.diag_indices(4)] = 1.0 - sums
    return p


_TARGET = np.zeros((64, 20))
for _i, _c in enumerate(CODONS):
    _aa = GENETIC_CODE[_c]
    if _aa != "*":
        _TARGET[_i, _AA_INDEX[_aa]] = 1.0


def codon_aa_table(q: np.ndarray, *, row_sum: float | None = None) -> np.ndarray:
    """64 x 20 table: probability that a codon changes into any codon of each amino acid.

    The synonymous column of a codon's own amino acid is included (it is not a replacement);
    callers select replacements by amino acid. Stop codons are never targets.
    """
    p = as_probabilities(q, row_sum=row_sum)
    codon_to_codon = np.kron(np.kron(p, p), p)       # same ordering as CODONS
    return codon_to_codon @ _TARGET


def accessibility(ref_codons, aas, q: np.ndarray, *, row_sum: float | None = None) -> np.ndarray:
    """Accessibility of each (reference codon, target amino acid) under spectrum ``q``."""
    table = codon_aa_table(q, row_sum=row_sum)
    ci = np.fromiter((_CODON_INDEX[c] for c in ref_codons), dtype=int, count=len(ref_codons))
    ai = np.fromiter((_AA_INDEX[a] for a in aas), dtype=int, count=len(aas))
    return table[ci, ai]


# ---------------------------------------------------------------------------
# Geometry of the genetic code
# ---------------------------------------------------------------------------

def nucleotide_steps(ref_codon: str, aa: str) -> int:
    """Fewest nucleotide changes taking ``ref_codon`` to any codon of amino acid ``aa``."""
    return min(
        sum(x != y for x, y in zip(ref_codon, c))
        for c, a in GENETIC_CODE.items() if a == aa
    )


def single_step_changes(ref_codon: str, aa: str) -> list[tuple[str, str]]:
    """The (from_base, to_base) of every single-nucleotide route from the codon to ``aa``."""
    out = []
    for pos in range(3):
        for b in BASES:
            if b == ref_codon[pos]:
                continue
            mutant = ref_codon[:pos] + b + ref_codon[pos + 1:]
            if GENETIC_CODE[mutant] == aa:
                out.append((ref_codon[pos], b))
    return out


# ---------------------------------------------------------------------------
# Variants of a spectrum
# ---------------------------------------------------------------------------

def transpose(q: np.ndarray) -> np.ndarray:
    return np.array(q, dtype=float).T.copy()


def symmetrise(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=float)
    return (q + q.T) / 2.0


_COMPLEMENT = [BASES.index(b) for b in "TGCA"]    # A<->T, C<->G


def complement(q: np.ndarray) -> np.ndarray:
    """The same spectrum read on the other strand: Q'[i, j] = Q[c(i), c(j)]."""
    q = np.array(q, dtype=float)
    c = _COMPLEMENT
    return q[np.ix_(c, c)]


def relabel(q: np.ndarray, perm) -> np.ndarray:
    """Rename bases: new base ``i`` is old base ``perm[i]``."""
    perm = list(perm)
    if sorted(perm) != [0, 1, 2, 3]:
        raise ValueError(f"not a permutation of 0..3: {perm}")
    return np.array(q, dtype=float)[np.ix_(perm, perm)]


def all_relabellings() -> list[tuple[int, ...]]:
    """All 24 base permutations; the first is the identity."""
    return list(itertools.permutations(range(4)))


def shuffle(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the twelve off-diagonal rates at random. Same multiset of rates."""
    q = np.array(q, dtype=float)
    off = ~np.eye(4, dtype=bool)
    values = q[off].copy()
    rng.shuffle(values)
    out = np.zeros_like(q)
    out[off] = values
    return out


def relative_asymmetry(q: np.ndarray) -> float:
    """sum |Q - Q^T| / sum (Q + Q^T): 0 for a symmetric spectrum, 1 when every rate is one-way."""
    q = np.array(q, dtype=float)
    return float(np.abs(q - q.T).sum() / (q + q.T).sum())


# ---------------------------------------------------------------------------
# Reference codons from root sequences
# ---------------------------------------------------------------------------

def reference_codons(root_fasta: str | Path, positions) -> list[str]:
    """Codon at each 1-based ``position`` of a single-record coding FASTA.

    Raises if a position is below 1 or outside the sequence, the file is not exactly one record,
    or the codon contains a non-ACGT base.
    """
    lines = [ln.strip() for ln in Path(root_fasta).read_text().splitlines() if ln.strip()]
    if sum(ln.startswith(">") for ln in lines) != 1:
        raise ValueError(f"{root_fasta}: expected exactly one FASTA record")
    seq = "".join(ln for ln in lines if not ln.startswith(">")).upper()
    out = []
    for p in positions:
        if int(p) != p or int(p) < 1:
            raise ValueError(f"{root_fasta}: positions are 1-based integers, got {p!r}")
        codon = seq[(int(p) - 1) * 3: int(p) * 3]
        if len(codon) != 3 or set(codon) - set(BASES):
            raise ValueError(f"{root_fasta}: no valid codon at position {p}: {codon!r}")
        out.append(codon)
    return out
