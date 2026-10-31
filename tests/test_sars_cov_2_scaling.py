"""The SARS-CoV-2 spectrum's absolute scale must recompute from things that ship.

The manuscript states that `data/spectra/sars_cov_2.csv` is the De Maio et al. (2021)
Table 1 matrix put on an absolute scale by a per-site per-replication rate of 1.3e-6
(Amicone et al. 2022) and the SARS-CoV-2 genome composition, quoting a scaling factor of
8.186e-7. Neither the composition nor the genome behind it was in the repo, so a reader
could not check either; `data/reference_genomes/NC_045512.2.fasta` is now that genome and
these tests are that check.

The arithmetic: with `q` the published relative rates and `pi` the genome composition, the
factor `f` is the one that makes the composition-weighted mean of `f * q` equal 1.3e-6.
Equivalently, the *shipped* matrix's own composition-weighted mean must already be 1.3e-6.
"""

import numpy as np
import pytest

from mutacc import accessibility as A
from mutacc.guide import REPO_ROOT

SPECTRUM = REPO_ROOT / "data" / "spectra" / "sars_cov_2.csv"
GENOME = REPO_ROOT / "data" / "reference_genomes" / "NC_045512.2.fasta"

#: Per-site per-replication mutation rate, Amicone et al. (2022), as the Methods state it.
RATE = 1.3e-6
#: The scaling factor the Methods quote.
PUBLISHED_FACTOR = 8.186e-7
#: The genome composition the Methods quote, as percentages in A, C, G, T order.
PUBLISHED_COMPOSITION = (29.9, 18.4, 19.6, 32.1)

BASES = "ACGT"


def genome() -> str:
    """The Wuhan-Hu-1 sequence, uppercased, with the FASTA header dropped."""
    lines = GENOME.read_text().splitlines()
    assert lines[0].startswith(">NC_045512.2"), lines[0][:60]
    return "".join(lines[1:]).upper()


def composition() -> np.ndarray:
    """Base frequencies in A, C, G, T order."""
    seq = genome()
    return np.array([seq.count(b) for b in BASES], dtype=float) / len(seq)


def weighted_mean(q: np.ndarray, pi: np.ndarray) -> float:
    """Composition-weighted mean per-site mutation rate of a spectrum."""
    return float(pi @ q.sum(axis=1))


def test_the_genome_is_the_wuhan_hu_1_reference():
    """Pinned so that a truncated or silently re-fetched file is an error, not a shift.

    Every number below is a function of this sequence, so its identity is the premise of
    the whole file.
    """
    seq = genome()
    assert len(seq) == 29903
    assert set(seq) == set(BASES), f"unexpected characters: {sorted(set(seq) - set(BASES))}"
    assert [seq.count(b) for b in BASES] == [8954, 5492, 5863, 9594]


def test_the_published_composition_is_this_genome_rounded():
    """The four percentages in the Methods are Wuhan-Hu-1's, to the precision printed."""
    pct = 100 * composition()
    assert [round(x, 1) for x in pct] == list(PUBLISHED_COMPOSITION)


def test_the_shipped_matrix_already_carries_the_amicone_rate():
    """The scale is self-consistent: weighting the shipped matrix by the composition
    returns the rate it was scaled to.

    This is the statement the Methods make, checked end to end rather than through the
    factor, and it needs no published De Maio values to do it.
    """
    q = A.load_spectrum(SPECTRUM)
    assert weighted_mean(q, composition()) == pytest.approx(RATE, rel=1e-3)


def test_the_published_factor_recomputes_from_the_genome():
    """Recovering De Maio's relative rates and re-deriving the factor returns 8.186e-7.

    It does not return it *exactly* — the shipped matrix is stored to three significant
    figures, so the relative rates recovered from it are rounded and the factor they give
    is 8.1798e-7, 0.08% low. That is a property of the stored precision, not a different
    scaling, and the next test pins that reading.
    """
    q = A.load_spectrum(SPECTRUM)
    de_maio = q / PUBLISHED_FACTOR
    factor = RATE / weighted_mean(de_maio, composition())
    assert factor == pytest.approx(PUBLISHED_FACTOR, rel=1e-3)


def test_the_residual_is_inside_the_stored_precision():
    """1.3e-6 lies within the rounding interval of the three-significant-figure matrix.

    Perturbing every entry by half of its last stored digit moves the weighted mean across
    an interval that contains the target rate, so the 0.08% residual is consistent with
    rounding alone. If a future edit re-scales the matrix for real, the point estimate
    leaves this interval and the test above fails first; this one says why it cannot be
    blamed on precision.
    """
    q = A.load_spectrum(SPECTRUM)
    pi = composition()

    # Half the place value of each entry's last significant digit, zeros excluded.
    nz = q > 0
    half = np.zeros_like(q)
    half[nz] = 10.0 ** (np.floor(np.log10(q[nz])) - 2) / 2

    lo = weighted_mean(q - half, pi)
    hi = weighted_mean(q + half, pi)
    assert lo < RATE < hi, f"{RATE:.4e} outside [{lo:.4e}, {hi:.4e}]"


def test_the_composition_is_the_genome_and_not_the_spike_cds():
    """Identifies *which* sequence the Methods' percentages came from.

    Spike was the only SARS-CoV-2 nucleotide sequence in the repo before the genome was added,
    and it is not a stand-in: it is depleted in G and enriched in T relative to the genome, and
    using any of the five roots would move the factor by 1.6-2.3%. The stated 19.6% G settles
    it on its own — no spike CDS here reaches 19%.
    """
    q = A.load_spectrum(SPECTRUM)
    genome_factor = RATE / weighted_mean(q / PUBLISHED_FACTOR, composition())

    roots = sorted((REPO_ROOT / "data" / "roots" / "sars_cov_2_spike").glob("*.nt.fa"))
    assert roots, "no spike roots to compare against"
    for root in roots:
        seq = "".join(l.strip() for l in root.read_text().splitlines()
                      if not l.startswith(">")).upper()
        pi = np.array([seq.count(b) for b in BASES], dtype=float)
        pi /= pi.sum()
        spike_factor = RATE / weighted_mean(q / PUBLISHED_FACTOR, pi)
        off = abs(spike_factor / PUBLISHED_FACTOR - 1)
        assert off > 0.015, f"{root.name} is not meaningfully distinguishable ({off:.4f})"
        g_pct = 100 * pi[BASES.index("G")]
        assert g_pct < 19.0, f"{root.name} G content {g_pct:.2f}% is genome-like"

    assert abs(genome_factor / PUBLISHED_FACTOR - 1) < 0.002


def test_within_stratum_ranking_does_not_depend_on_the_factor():
    """Why the 0.08% does not reach a figure.

    Accessibility is used as a ranking inside a nucleotide-step stratum, where a common
    positive scale cancels. The factor sets the units the Methods report, not any result.
    """
    q = A.load_spectrum(SPECTRUM)
    codons = ["ATG", "GAA", "TTT", "CGC", "AGT"]
    aas = ["K", "D", "L", "H", "N"]
    a = A.accessibility(codons, aas, q)
    b = A.accessibility(codons, aas, q * 1.0008)
    assert np.array_equal(np.argsort(a), np.argsort(b))
