"""Authoritative regression tests for lineage definitions and the J.2 to K transition.

Pins:
1. Root sequence files exist and are valid coding sequences.
2. J.2_int vs K divergence comprises EXACTLY 9 amino acid mutations (8 in HA1, 1 in HA2).
3. Residue 145 (precursor codon 161) is identically Serine (agt) in J.2 and K, so the
   published N145S is a phantom and NOT a J.2 -> K mutation. It arose from diffing a
   stale stored table against a live reference FASTA, not from a vaccine-strain
   reference; see LINEAGE_DEFINITIONS.md section 6 and tests/test_fig01_mutation_labels.py.
4. All 9 mutations are single-nucleotide steps under the genetic code.
5. Coordinate transformations between precursor, mature HA1, and mature HA2 coordinates.
6. Identity between 5-lineage J.2_int and 13-lineage J.2 translated proteins.
7. Epistasis focal sites table contains exactly these 9 mutations and excludes 145.
8. Cross-pathogen SARS-CoV-2 Pangolin lineage roots are valid.
"""

from __future__ import annotations

import pathlib
import pytest
from Bio import SeqIO
from Bio.Seq import Seq

from mutacc.guide import REPO_ROOT, load_guide

# Paths
ROOTS_5 = REPO_ROOT / "data" / "roots" / "h3n2_5lineage"
ROOTS_13 = REPO_ROOT / "data" / "roots" / "h3n2_13lineage"
ROOTS_SC2 = REPO_ROOT / "data" / "roots" / "sars_cov_2_spike"
DOC_PRIMARY = REPO_ROOT / "LINEAGE_DEFINITIONS.md"
DOC_ALIAS_DEF = REPO_ROOT / "lineage_definitions.md"
DOC_ALIAS_TYPO = REPO_ROOT / "lineage_definitons.md"
DOC_DIR_README = REPO_ROOT / "lineage_definitions" / "README.md"

# Canonical 9 clade-defining mutations between J.2_int and K
# (precursor_codon_1based, mature_label, domain, domain_pos, j2_codon, k_codon, j2_aa, k_aa)
CANONICAL_9_MUTATIONS = [
    (18, "K2N", "HA1", 2, "aaa", "aac", "K", "N"),
    (151, "T135K", "HA1", 135, "aca", "aaa", "T", "K"),
    (160, "S144N", "HA1", 144, "agt", "aat", "S", "N"),
    (174, "N158D", "HA1", 158, "aac", "gac", "N", "D"),
    (176, "I160K", "HA1", 160, "ata", "aaa", "I", "K"),
    (189, "Q173R", "HA1", 173, "caa", "cga", "Q", "R"),
    (205, "K189R", "HA1", 189, "aag", "agg", "K", "R"),
    (344, "T328A", "HA1", 328, "acc", "gcc", "T", "A"),
    (394, "HA2:S49N", "HA2", 49, "agt", "aat", "S", "N"),
]


# ---------------------------------------------------------------------------
# Documentation & Symlink Integrity
# ---------------------------------------------------------------------------

def test_lineage_definitions_documentation_and_symlinks_exist():
    """All requested files, aliases, and directory READMEs must exist."""
    assert DOC_PRIMARY.exists(), f"Missing primary doc: {DOC_PRIMARY}"
    assert DOC_ALIAS_DEF.exists(), f"Missing alias: {DOC_ALIAS_DEF}"
    assert DOC_ALIAS_TYPO.exists(), f"Missing alias: {DOC_ALIAS_TYPO}"
    assert DOC_DIR_README.exists(), f"Missing folder readme: {DOC_DIR_README}"

    # Verify content mentions all 9 mutations and explains N145S
    content = DOC_PRIMARY.read_text()
    for _, label, _, _, _, _, _, _ in CANONICAL_9_MUTATIONS:
        assert label in content, f"Doc missing mention of {label}"
    assert "N145S" in content, "Doc must explain N145S artifact"
    assert "Pangolin" in content or "pango" in content.lower(), "Doc must explain Pangolin definition"


# ---------------------------------------------------------------------------
# Sequence Integrity & Protein Translation
# ---------------------------------------------------------------------------

def test_h3n2_root_fastas_exist():
    """Root FASTA files for J.2 and K must exist in both 5-lineage and 13-lineage."""
    assert (ROOTS_5 / "J.2_int.nt.fa").exists()
    assert (ROOTS_5 / "K.nt.fa").exists()
    assert (ROOTS_13 / "J.2.nt.fa").exists()
    assert (ROOTS_13 / "K.nt.fa").exists()


def test_j2_and_k_coding_sequences_are_in_frame():
    """Sequences must be multiples of 3, begin with ATG, and translate without stops."""
    for path in [ROOTS_5 / "J.2_int.nt.fa", ROOTS_5 / "K.nt.fa",
                 ROOTS_13 / "J.2.nt.fa", ROOTS_13 / "K.nt.fa"]:
        record = SeqIO.read(path, "fasta")
        assert len(record.seq) % 3 == 0, f"{path.name} length not divisible by 3"
        assert record.seq[:3].upper() == "ATG", f"{path.name} does not start with ATG"
        prot = record.seq.translate()
        assert "*" not in prot[:-1], f"{path.name} contains premature stop codon"


def test_j2_int_5lineage_and_13lineage_have_identical_protein_translation():
    """5-lineage J.2_int and 13-lineage J.2 differ only by a synonymous NT mutation."""
    j2_5 = SeqIO.read(ROOTS_5 / "J.2_int.nt.fa", "fasta")
    j2_13 = SeqIO.read(ROOTS_13 / "J.2.nt.fa", "fasta")
    prot_5 = j2_5.seq.translate()
    prot_13 = j2_13.seq.translate()
    assert prot_5 == prot_13, "J.2_int (5-lineage) and J.2 (13-lineage) translated proteins differ!"


# ---------------------------------------------------------------------------
# Exact Mutation Differences (J.2 -> K)
# ---------------------------------------------------------------------------

def _find_amino_acid_differences(seq_ancestor: Seq, seq_descendant: Seq):
    prot_anc = seq_ancestor.translate()
    prot_desc = seq_descendant.translate()
    assert len(prot_anc) == len(prot_desc)
    diffs = []
    for i in range(len(prot_anc)):
        if prot_anc[i] != prot_desc[i]:
            c_anc = str(seq_ancestor[i*3:(i+1)*3]).lower()
            c_desc = str(seq_descendant[i*3:(i+1)*3]).lower()
            diffs.append((i + 1, prot_anc[i], prot_desc[i], c_anc, c_desc))
    return diffs


def test_j2_int_vs_k_exact_nine_amino_acid_differences():
    """J.2_int vs K must yield EXACTLY 9 amino acid differences."""
    j2 = SeqIO.read(ROOTS_5 / "J.2_int.nt.fa", "fasta")
    k = SeqIO.read(ROOTS_5 / "K.nt.fa", "fasta")
    diffs = _find_amino_acid_differences(j2.seq, k.seq)
    assert len(diffs) == 9, f"Expected exactly 9 mutations, found {len(diffs)}: {diffs}"


def test_j2_vs_k_13lineage_exact_nine_amino_acid_differences():
    """J.2 (13-lineage) vs K (13-lineage) must also yield EXACTLY 9 amino acid differences."""
    j2 = SeqIO.read(ROOTS_13 / "J.2.nt.fa", "fasta")
    k = SeqIO.read(ROOTS_13 / "K.nt.fa", "fasta")
    diffs = _find_amino_acid_differences(j2.seq, k.seq)
    assert len(diffs) == 9, f"Expected exactly 9 mutations, found {len(diffs)}: {diffs}"


def test_mutation_domain_split_eight_ha1_one_ha2():
    """8 mutations must be in HA1 (mature 1-329, precursor 17-345) and 1 in HA2 (precursor 346-566)."""
    j2 = SeqIO.read(ROOTS_5 / "J.2_int.nt.fa", "fasta")
    k = SeqIO.read(ROOTS_5 / "K.nt.fa", "fasta")
    diffs = _find_amino_acid_differences(j2.seq, k.seq)
    ha1_diffs = [d for d in diffs if 17 <= d[0] <= 345]
    ha2_diffs = [d for d in diffs if 346 <= d[0] <= 566]
    signal_diffs = [d for d in diffs if d[0] < 17]

    assert len(signal_diffs) == 0, f"Unexpected difference in signal peptide: {signal_diffs}"
    assert len(ha1_diffs) == 8, f"Expected 8 HA1 mutations, found {len(ha1_diffs)}: {ha1_diffs}"
    assert len(ha2_diffs) == 1, f"Expected 1 HA2 mutation, found {len(ha2_diffs)}: {ha2_diffs}"


# ---------------------------------------------------------------------------
# Verification of Residue 145 (Post-Mortem of N145S Artifact)
# ---------------------------------------------------------------------------

def test_residue_145_is_identically_serine_in_both_roots():
    """Precursor codon 161 corresponds to mature HA1 residue 145 (161 - 16 = 145).
    In both J.2 and K, codon 161 must be agt -> S (Serine).
    """
    j2 = SeqIO.read(ROOTS_5 / "J.2_int.nt.fa", "fasta")
    k = SeqIO.read(ROOTS_5 / "K.nt.fa", "fasta")

    # 1-based codon 161 -> 0-based slice [160*3 : 161*3]
    j2_c161 = str(j2.seq[160*3:161*3]).lower()
    k_c161 = str(k.seq[160*3:161*3]).lower()

    assert j2_c161 == "agt", f"J.2 codon 161 expected agt, got {j2_c161}"
    assert k_c161 == "agt", f"K codon 161 expected agt, got {k_c161}"

    j2_aa145 = Seq(j2_c161).translate()
    k_aa145 = Seq(k_c161).translate()

    assert j2_aa145 == "S", f"J.2 mature residue 145 is {j2_aa145}, expected S"
    assert k_aa145 == "S", f"K mature residue 145 is {k_aa145}, expected S"


def test_n145s_is_not_a_mutation_between_j2_and_k():
    """Assert that N145S does not exist between J.2_int and K."""
    j2 = SeqIO.read(ROOTS_5 / "J.2_int.nt.fa", "fasta")
    k = SeqIO.read(ROOTS_5 / "K.nt.fa", "fasta")
    diffs = _find_amino_acid_differences(j2.seq, k.seq)

    # Check that precursor position 161 (mature 145) is not in diffs
    diff_positions = [d[0] for d in diffs]
    assert 161 not in diff_positions, "Codon 161 (mature 145) should NOT differ between J.2 and K!"


# ---------------------------------------------------------------------------
# Canonical Mutation Table & Coordinate Transformations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prec_pos,label,domain,dom_pos,c_j2,c_k,aa_j2,aa_k", CANONICAL_9_MUTATIONS)
def test_each_canonical_mutation_matches_root_sequences(prec_pos, label, domain, dom_pos,
                                                        c_j2, c_k, aa_j2, aa_k):
    """Every canonical mutation must match exactly at the nucleotide and amino acid levels."""
    j2 = SeqIO.read(ROOTS_5 / "J.2_int.nt.fa", "fasta")
    k = SeqIO.read(ROOTS_5 / "K.nt.fa", "fasta")

    # 1-based codon index
    idx = prec_pos - 1
    actual_c_j2 = str(j2.seq[idx*3:(idx+1)*3]).lower()
    actual_c_k = str(k.seq[idx*3:(idx+1)*3]).lower()

    assert actual_c_j2 == c_j2, f"At {label}: J.2 codon is {actual_c_j2}, expected {c_j2}"
    assert actual_c_k == c_k, f"At {label}: K codon is {actual_c_k}, expected {c_k}"
    assert Seq(actual_c_j2).translate() == aa_j2
    assert Seq(actual_c_k).translate() == aa_k

    # Test coordinate transformation
    if domain == "HA1":
        assert prec_pos - 16 == dom_pos, f"HA1 math failed for {label}: {prec_pos} - 16 != {dom_pos}"
    elif domain == "HA2":
        assert prec_pos - 345 == dom_pos, f"HA2 math failed for {label}: {prec_pos} - 345 != {dom_pos}"
        assert prec_pos - 16 == 378, f"Full mature precursor math failed for {label}"


@pytest.mark.parametrize("prec_pos,label,domain,dom_pos,c_j2,c_k,aa_j2,aa_k", CANONICAL_9_MUTATIONS)
def test_all_canonical_mutations_are_single_nucleotide_steps(prec_pos, label, domain, dom_pos,
                                                              c_j2, c_k, aa_j2, aa_k):
    """Every one of the 9 substitutions must be reachable by a 1-nt step (Hamming distance 1)."""
    assert len(c_j2) == 3 and len(c_k) == 3
    hamming_dist = sum(1 for a, b in zip(c_j2, c_k) if a != b)
    assert hamming_dist == 1, f"Mutation {label} ({c_j2} -> {c_k}) is {hamming_dist} nt steps, expected 1"


# ---------------------------------------------------------------------------
# Epistasis Focal Sites Table Integrity
# ---------------------------------------------------------------------------

def test_epistasis_focal_sites_contains_canonical_nine():
    """GUIDE epistasis_focal_sites must contain exactly the 9 canonical mutations."""
    guide = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
    sites = guide.read_table("epistasis_focal_sites")

    assert len(sites) == 9, f"epistasis_focal_sites must have exactly 9 rows, found {len(sites)}"

    focals = set(sites["focal"])
    expected_focals = {m[1] for m in CANONICAL_9_MUTATIONS}
    assert focals == expected_focals, f"Mismatch in focal sites: {focals ^ expected_focals}"

    # Verify site 145 is NOT present
    assert "N145S" not in focals
    assert 145 not in set(sites["region_pos"])
    assert 145 not in set(sites["seq_pos"])


# ---------------------------------------------------------------------------
# SARS-CoV-2 Pangolin Lineage Roots
# ---------------------------------------------------------------------------

def test_sarscov2_pangolin_roots_are_valid():
    """Verify SARS-CoV-2 Spike Pangolin lineage roots exist and translate properly."""
    expected_lineages = ["Wuhan-Hu-1_Reference", "JN.1", "LP.8.1", "XEC", "NB.1.8.1"]
    for lin in expected_lineages:
        fa_path = ROOTS_SC2 / f"{lin}.nt.fa"
        assert fa_path.exists(), f"Missing SARS-CoV-2 Pangolin root: {fa_path}"
        record = SeqIO.read(fa_path, "fasta")
        assert len(record.seq) % 3 == 0
        assert record.seq[:3].upper() == "ATG"
        prot = record.seq.translate()
        assert "*" not in prot[:-1]
