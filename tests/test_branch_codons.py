"""S11 engine: nucleotide changes per codon per branch. Planted truths on a synthetic tree.

Every tree here is SYNTHETIC. Nothing in this file is a result about any virus. The planted
tree lives in tests/fixtures/sK_tiny/ and is described in comments next to its expectations.

This file checks that the engine and its fixtures carry no absolute path, so it has to spell
some out. That is what the marker below is for; see ``tests/test_data_integrity.py``.
allow-absolute-path-literals
"""

import gzip
import random
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from Bio.Data import CodonTable

from mutacc import branch_codons as bc

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "sK_tiny"
sys.path.insert(0, str(ROOT / "tools"))
import branch_codon_tables as cli  # noqa: E402

REF = "ATGGCTAAACTGTCTGGATTTCGAANA"   # M A K L S G F R X: 9 codons, the last holds an N


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_fixture(cap=10, include_terminal=False, tips=None):
    tree = bc.parse_newick((FIX / "tree.nwk").read_text())
    muts = bc.parse_mutation_paths((FIX / "paths.txt").read_text().splitlines(), tree)
    with open(FIX / "aln.fasta") as fh:
        _, aln_ref = bc.first_fasta_record(fh)
    ref = bc.ungap(aln_ref)
    assert ref == REF
    cds = bc.check_frame(ref)
    tips = set(tips or (FIX / "tips.txt").read_text().split())
    clade, start, size, n_all, n_tip = bc.main_clade_and_start(tree, tips, tips, 0.95)
    res = bc.walk_branches(tree, muts, ref, cds, start, n_tip, max_nt=cap, include_terminal=include_terminal)
    return tree, muts, ref, cds, start, res


def build(newick, paths, ref, tips, *, cap=10, include_terminal=False, purity=0.95):
    tree = bc.parse_newick(newick)
    muts = bc.parse_mutation_paths(paths.splitlines(), tree)
    cds = bc.check_frame(ref)
    clade, start, size, n_all, n_tip = bc.main_clade_and_start(tree, set(tips), set(tips), purity)
    return bc.walk_branches(tree, muts, ref, cds, start, n_tip, max_nt=cap, include_terminal=include_terminal), start


def row(df, branch):
    return df.loc[df["branch"] == branch].iloc[0]


# ---------------------------------------------------------------------------
# genetic code
# ---------------------------------------------------------------------------

def test_genetic_code_is_the_standard_table_from_biopython():
    table = CodonTable.unambiguous_dna_by_id[1]
    assert len(bc.CODON2AA) == 64
    for codon, aa in bc.CODON2AA.items():
        assert aa == "*" if codon in table.stop_codons else aa == table.forward_table[codon]
    assert sorted(c for c, a in bc.CODON2AA.items() if a == "*") == ["TAA", "TAG", "TGA"]
    assert len(bc.AA2CODONS) == 21 and len(bc.AMINO_ACIDS) == 20
    assert sum(len(v) for v in bc.AA2CODONS.values()) == 64


def test_known_codon_distances():
    assert bc.MIN_AA_DISTANCE[("L", "I")] == 1      # CTT -> ATT, though CTG -> ATx needs two
    assert bc.min_distance_from_codon("CTG", "I") == 2
    assert bc.min_distance_from_codon("AAA", "R") == 1
    assert bc.MIN_AA_DISTANCE[("K", "K")] == 0
    assert all(bc.MIN_AA_DISTANCE[(a, b)] == bc.MIN_AA_DISTANCE[(b, a)] for a in bc.AMINO_ACIDS for b in bc.AMINO_ACIDS)
    assert max(bc.MIN_AA_DISTANCE.values()) == 3 and min(v for (a, b), v in bc.MIN_AA_DISTANCE.items() if a != b) == 1


def test_min_from_parent_never_below_min_code_never_above_observed():
    for a, b in ((x, y) for x in bc.CODON2AA for y in bc.CODON2AA):
        aa, bb = bc.CODON2AA[a], bc.CODON2AA[b]
        if aa == bb or "*" in (aa, bb):
            continue
        assert bc.MIN_AA_DISTANCE[(aa, bb)] <= bc.min_distance_from_codon(a, bb) <= bc.hamming(a, b)


def test_a_synonymous_ancestor_separates_the_two_comparators():
    """Planted: CTG -> ATA (L to I) took 2 changes. The code allows 1 (CTT -> ATT) but THIS parent
    codon needs 2, so n_nt equals n_min_from_parent while n_min_code is smaller."""
    cat, a1, a2, n_code, n_par, n_nt = bc._classify("CTG", "ATA")
    assert (cat, a1, a2, n_code, n_par, n_nt) == ("nonsyn", "L", "I", 1, 2, 2)


def test_hamming_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        bc.hamming("AC", "ACG")


# ---------------------------------------------------------------------------
# newick and mutation parsing
# ---------------------------------------------------------------------------

def test_newick_basic_structure_and_order():
    t = bc.parse_newick("((a:1,b:2)x:3,c:4)root;")
    assert t.root == "root" and t.children["x"] == ["a", "b"]
    assert t.preorder == ["root", "x", "a", "b", "c"]
    assert t.parent["a"] == "x" and t.tips == {"a", "b", "c"}


@pytest.mark.parametrize("text", ["", "   ", "((a,b)x,c;", "(a,b))x;", "(a,b)x,c;", "(a,a)x;", "(a,b)x;extra",
                                  "(a,b)x;(c,d)y;", "()x;", "((a,b)x,(c,d)x)r;"])
def test_newick_rejects_malformed_input(text):
    with pytest.raises(bc.InputError):
        bc.parse_newick(text)


def test_newick_single_tip_and_unlabelled_internal_nodes():
    assert bc.parse_newick("a;").tips == {"a"}
    t = bc.parse_newick("((a,b),c);")
    assert len(t.children) == 5 and all(n.startswith("_u") or n in "abc" for n in t.children)


def test_newick_deep_chain_does_not_recurse():
    n = 30000
    text = "(" * n + "t" + "".join(f")n{i}" for i in range(n)) + ";"
    t = bc.parse_newick(text)
    assert len(t.preorder) == n + 1 and t.preorder[-1] == "t"


def test_newick_tolerates_newlines_and_missing_lengths():
    t = bc.parse_newick("(a,\n b)x\n;\n")
    assert t.children["x"] == ["a", "b"]


def test_mutation_paths_parse_and_order():
    got = bc.parse_mutation_paths(["n1: A5G,C7T", "n2: ", "t1: G9A"])
    assert got == {"n1": [("A", 5, "G"), ("C", 7, "T")], "t1": [("G", 9, "A")]}


@pytest.mark.parametrize("line", ["n1 A5G", "n1: A5N", "n1: A0G", "n1: A5A", "n1: 5G", ": A5G", "n1: a5g"])
def test_mutation_paths_reject_bad_records(line):
    with pytest.raises(bc.InputError):
        bc.parse_mutation_paths([line])


def test_mutation_paths_reject_repeated_and_unknown_nodes():
    with pytest.raises(bc.InputError, match="twice"):
        bc.parse_mutation_paths(["n1: A5G", "n1: C6T"])
    tree = bc.parse_newick("(a,b)n1;")
    with pytest.raises(bc.InputError, match="not in the tree"):
        bc.parse_mutation_paths(["zzz: A5G"], tree)


# ---------------------------------------------------------------------------
# reference, frame, gaps
# ---------------------------------------------------------------------------

def test_first_record_only_and_ungap():
    with open(FIX / "aln.fasta") as fh:
        name, seq = bc.first_fasta_record(fh)
    assert name == "SYNTHETIC_ref" and seq.count("-") == 4 and len(seq) == 31
    assert bc.ungap(seq) == REF
    assert bc.ungap("ac.g-t") == "ACGT"


@pytest.mark.parametrize("lines", [[], ["ACGT"], [">x"], [">x", ""], ["\n"]])
def test_first_record_rejects_empty_or_headerless(lines):
    with pytest.raises(bc.InputError):
        bc.first_fasta_record(lines)


def test_frame_accepts_good_reference_and_final_stop():
    assert bc.check_frame(REF) == bc.Cds(1, 27)
    assert bc.check_frame("ATGAAATAA").n_codons == 3          # final stop allowed
    assert bc.check_frame("CCATGAAATTT", 3, 11).n_codons == 3  # explicit coding region


@pytest.mark.parametrize("ref, kw", [("ATGAAAT", {}), ("ATGTAAAAA", {}), ("ATGAAATGA" + "AAA", {}),
                                    ("ATGAAA", {"start": 0}), ("ATGAAA", {"end": 7}), ("ATGAAA", {"start": 5, "end": 3})])
def test_frame_failures_raise(ref, kw):
    with pytest.raises(bc.FrameError):
        bc.check_frame(ref, **kw)


def test_frame_failure_points_at_the_utr_offset():
    ref = "GG" + "ATGAAACCC" + "A"     # 2-base 5' UTR; frame 0 reads GGA TGA ... (stop at codon 2)
    with pytest.raises(bc.FrameError, match=r"codon 2 .*starting at (2|3|4) reads without an internal stop"):
        bc.check_frame(ref)
    bc.check_frame(ref[2:].rstrip("A") + "", 1, 9)          # the true start reads cleanly


def test_a_gap_block_that_would_shift_the_frame_does_not_shift_codon_numbers():
    """The alignment has a 4-column gap in the reference after codon 3. Counting alignment
    columns would put codon 4's first base at column 14 and call it codon 5; counting along the
    reference record keeps it codon 4. Mutations are in record coordinates."""
    ref_aln = "ATGGCTAAA----CTGTCTGGATTTCGAANA"
    assert len(ref_aln) % 3 != 0                      # the archive's frame check on width would trip or mislead
    ref = bc.ungap(ref_aln)
    assert (ref_aln.index("CTG") // 3) + 1 == 5 and ref.index("CTG") // 3 + 1 == 4
    res, _ = build("((a,b)x,c)r;", "x: C10A,G12A", ref, {"a", "b", "c"}, cap=10, include_terminal=False)
    ev = [e for e in res.events]
    assert [e.codon for e in ev] == [4] and ev[0].codon_from == "CTG" and ev[0].codon_to == "ATA"


def test_a_gap_block_not_a_multiple_of_three_is_still_fine_because_only_the_record_counts():
    ref = bc.ungap("ATGGCTAAA-----CTGTCTGGATTTCGAANA")
    assert ref == REF and len(ref) % 3 == 0
    bc.check_frame(ref)


def test_a_reference_with_a_frameshifting_deletion_is_refused():
    with pytest.raises(bc.FrameError):
        bc.check_frame(bc.ungap("ATGGCTAAACT-TCTGGATTTCGAANA"))


def test_vcf_reference_check_passes_and_catches_one_wrong_base():
    lines = (FIX / "ref.vcf").read_text().splitlines()
    assert bc.check_reference_against_vcf(REF, lines) == 23
    bad = [l.replace("\tG\tT", "\tC\tT", 1) if i == 5 else l for i, l in enumerate(lines)]
    with pytest.raises(bc.InputError, match="disagrees"):
        bc.check_reference_against_vcf(REF, bad)


def test_vcf_in_alignment_columns_is_detected_when_the_record_has_gaps():
    """If a VCF counted alignment columns (gap kept), REF bases would not match the ungapped record."""
    gapped = "ATGGCTAAA----CTGTCTGGATTTCGAANA"
    lines = ["#h", f"c\t10\t.\t{gapped[9]}\tT\t.\t.\t.", f"c\t13\t.\t{gapped[12]}\tT\t.\t.\t.", f"c\t14\t.\t{gapped[13]}\tT\t.\t.\t."]
    with pytest.raises(bc.InputError):
        bc.check_reference_against_vcf(REF, lines)


def test_vcf_edge_cases():
    with pytest.raises(bc.InputError):
        bc.check_reference_against_vcf(REF, ["#only header"])
    with pytest.raises(bc.InputError, match="outside"):
        bc.check_reference_against_vcf(REF, ["c\t999\t.\tA\tT"])
    with pytest.raises(bc.InputError):
        bc.check_reference_against_vcf(REF, ["c\t1\t."])


# ---------------------------------------------------------------------------
# planted truth on the fixture tree
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def planted():
    tree, muts, ref, cds, start, res = load_fixture()
    return res, bc.branch_table(res), bc.column_table(res, ref, cds), start


def test_walk_starts_at_the_right_node_and_counts_what_it_skipped(planted):
    res, _, _, start = planted
    assert start == "n1"
    s = res.stats
    assert s["ancestral_mutations"] == 1           # n1's own branch (C22A) is ancestral, not scored
    assert s["branches_walked"] == 21 and s["kept"] == 7
    assert s["skipped_terminal"] == 13 and s["skipped_long"] == 1 and s["skipped_zero_nt"] == 0
    assert s["mutations_outside_cds"] == 0


def test_branch_table_rows_and_boundary(planted):
    _, b, _, _ = planted
    assert list(b["branch"]) == ["n2", "n3", "n4", "n5", "n6", "n8", "n9"]     # n7 has exactly 10 changes: out
    assert "n7" not in set(b["branch"]) and "R" not in set(b["branch"]) and "n1" not in set(b["branch"])
    assert list(b.columns) == bc.BRANCH_COLUMNS and "terminal" not in b.columns
    assert (b["nt_total"] < 10).all() and row(b, "n8")["nt_total"] == 9


def expect(b, branch, **kw):
    r = row(b, branch)
    for k, v in kw.items():
        assert r[k] == v, (branch, k, r[k], v)
    others = [c for c in bc.BRANCH_COLUMNS if c not in kw and c not in ("branch", "parent", "n_target_tips", "nt_total",
                                                                       "nt_in_cds", "codons_touched")]
    assert all(r[c] == 0 for c in others), (branch, {c: r[c] for c in others if r[c]})


def test_planted_branch_counts(planted):
    _, b, _, _ = planted
    # n2: C5T (GCT->GTT, A to V, 1 nt), A9G (AAA->AAG, synonymous, 1 nt)
    expect(b, "n2", nt_total=2, nt_in_cds=2, codons_touched=2, syn_1nt=1, nonsyn_1nt=1)
    # n3: CTG->ATA (2 nt, the parent needs 2), TCT->ACA (2 nt, one would do), AGA->TGA (stop; AGA from the ancestor n1)
    expect(b, "n3", nt_total=5, codons_touched=3, nonsyn_2nt=2, nonsyn_2nt_min1=1, dropped_stop=1)
    # n4: TTT->GCC (F to A, 3 nt, parent needs 2); TGA->CGA (stop in the parent)
    expect(b, "n4", nt_total=4, codons_touched=2, nonsyn_3nt=1, nonsyn_3nt_min2=1, dropped_stop=1)
    # n5: AAA->CGC (K to R, 3 nt, one would do). The n2 synonymous change is NOT inherited here
    expect(b, "n5", nt_total=3, codons_touched=1, nonsyn_3nt=1, nonsyn_3nt_min1=1)
    # n6: two records at one position (GGA->GAA->GTA), a change and change back, and an ambiguous codon
    expect(b, "n6", nt_total=5, codons_touched=3, nonsyn_1nt=1, reverted=1, dropped_ambiguous=1)
    # n8 follows the skipped 10-change branch n7, so its parents are n7's codons
    expect(b, "n8", nt_total=9, codons_touched=4, nonsyn_1nt=1, nonsyn_2nt=1, nonsyn_2nt_min1=1, nonsyn_3nt=2)
    # n9: the same TCT->ACA as n3 has: recurrence
    expect(b, "n9", nt_total=2, codons_touched=1, nonsyn_2nt=1, nonsyn_2nt_min1=1)


def test_events_planted_detail(planted):
    res, _, _, _ = planted
    ev = {(e.branch, e.codon): e for e in res.events}
    e = ev[("n3", 4)]
    assert (e.codon_from, e.codon_to, e.aa_from, e.aa_to, e.n_nt, e.n_mut) == ("CTG", "ATA", "L", "I", 2, 2)
    assert (e.n_min_from_parent, e.n_min_code, e.isolated) == (2, 1, False)
    e = ev[("n3", 5)]
    assert (e.codon_from, e.codon_to, e.n_min_from_parent) == ("TCT", "ACA", 1)
    e = ev[("n3", 8)]
    assert (e.codon_from, e.codon_to, e.category) == ("AGA", "TGA", "dropped_stop")      # ancestral C22A applied
    assert ev[("n4", 8)].category == "dropped_stop" and ev[("n4", 8)].codon_from == "TGA"
    e = ev[("n5", 3)]
    assert (e.n_nt, e.n_mut, e.n_min_from_parent, e.isolated) == (3, 3, 1, True)          # isolated: all 3 changes in this codon
    e = ev[("n6", 6)]
    assert (e.codon_from, e.codon_to, e.n_nt, e.n_mut, e.isolated) == ("GGA", "GTA", 1, 2, False)
    assert ev[("n6", 5)].category == "reverted" and ev[("n6", 5)].n_nt == 0 and ev[("n6", 5)].n_mut == 2
    e = ev[("n6", 9)]
    assert (e.category, e.codon_from, e.codon_to) == ("dropped_ambiguous", "ANA", "GNA")
    e = ev[("n8", 4)]                                          # parent GTG (n7 changed C10G) -> GAA
    assert (e.codon_from, e.codon_to, e.n_nt, e.n_min_from_parent, e.n_min_code) == ("GTG", "GAA", 2, 1, 1)
    assert ev[("n8", 5)].n_min_from_parent == 3 and ev[("n8", 5)].n_nt == 3
    assert ev[("n8", 7)].codon_from == "TTT" and ev[("n8", 7)].codon_to == "ATT"
    assert not any(e.branch == "n7" for e in res.events)       # skipped branch makes no events


def test_isolated_counts_records_not_net_differences(planted):
    res, _, _, _ = planted
    iso = {(e.branch, e.codon) for e in res.events if e.isolated}
    assert iso == {("n5", 3), ("n9", 5)}      # n9: both of its changes are in codon 5


def test_planted_column_table(planted):
    _, _, c, _ = planted
    assert len(c) == 9 and list(c["codon"]) == list(range(1, 10))
    assert list(c.columns) == bc.COLUMN_COLUMNS
    assert list(c["ref_codon"]) == [REF[i:i + 3] for i in range(0, 27, 3)] and c["ref_aa"].iloc[-1] == "X"

    def cell(codon, **kw):
        r = c.loc[c["codon"] == codon].iloc[0]
        for k, v in kw.items():
            assert r[k] == v, (codon, k, r[k], v)
        return r

    cell(1, events=0, n_distinct_aa_changes=0)
    cell(2, events=1, nonsyn_1nt=1)
    cell(3, events=2, syn_1nt=1, nonsyn_3nt=1, nonsyn_3nt_min1=1, n_distinct_aa_changes=1)
    cell(4, events=2, nonsyn_2nt=2, nonsyn_2nt_min1=1, n_distinct_aa_changes=2)
    # recurrence collapses: S->T twice (n3, n9) is one distinct change, S->K once, plus a revert
    cell(5, events=4, nonsyn_2nt=2, nonsyn_2nt_min1=2, nonsyn_3nt=1, reverted=1, n_distinct_aa_changes=2)
    cell(6, events=2, nonsyn_1nt=1, nonsyn_3nt=1, n_distinct_aa_changes=2)
    cell(7, events=2, nonsyn_1nt=1, nonsyn_3nt=1, nonsyn_3nt_min2=1)
    cell(8, events=2, dropped_stop=2)
    cell(9, events=1, dropped_ambiguous=1)


def test_column_table_totals_agree_with_branch_table(planted):
    _, b, c, _ = planted
    for col in [*bc._CLASS_COUNTS, *bc._EXCESS, *bc._OTHER]:
        assert b[col].sum() == c[col].sum(), col
    assert c["events"].sum() == b["codons_touched"].sum()
    for n in (1, 2, 3):
        assert (b[f"nonsyn_{n}nt"] >= 0).all()
    assert (b["nonsyn_2nt_min1"] <= b["nonsyn_2nt"]).all()
    assert (b["nonsyn_3nt_min1"] + b["nonsyn_3nt_min2"] <= b["nonsyn_3nt"]).all()
    assert (b["codons_touched"] == b[[*bc._CLASS_COUNTS, *bc._OTHER]].sum(axis=1)).all()
    assert (b["nt_in_cds"] <= b["nt_total"]).all()


def test_terminal_branches_appear_only_when_asked():
    _, _, ref, cds, _, res = load_fixture(include_terminal=True)
    b = bc.branch_table(res, include_terminal=True)
    assert "terminal" in b.columns
    assert list(b.loc[b["terminal"] == 1, "branch"]) == ["t1", "t2"]       # every other tip has no change
    assert res.stats["skipped_terminal"] == 0 and res.stats["skipped_zero_nt"] == 11
    expect_t1 = row(b, "t1")
    assert expect_t1["nonsyn_1nt"] == 1 and row(b, "t2")["syn_1nt"] == 1
    assert len(b) == 9


@pytest.mark.parametrize("cap, kept", [(9, ["n2", "n3", "n4", "n5", "n6", "n9"]),
                                       (10, ["n2", "n3", "n4", "n5", "n6", "n8", "n9"]),
                                       (11, ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"]),
                                       (None, ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"]),
                                       (0, ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"]),
                                       (3, ["n2", "n9"]),
                                       (2, [])])
def test_the_under_ten_boundary_is_strict(cap, kept):
    """'Fewer than 10': a branch with 9 changes is in, one with 10 is out, 11 is in only when the cap is raised."""
    _, _, _, _, _, res = load_fixture(cap=cap)
    assert [r.branch for r in res.branches] == kept


def test_skipped_long_branch_still_passes_its_state_to_children():
    _, _, _, _, _, a = load_fixture(cap=10)
    _, _, _, _, _, b = load_fixture(cap=None)
    ea = {(e.branch, e.codon): e for e in a.events if e.branch == "n8"}
    eb = {(e.branch, e.codon): e for e in b.events if e.branch == "n8"}
    assert ea == eb


def test_zero_change_branch_is_not_a_row_and_is_counted():
    res, _ = build("((a,b)x,(c,d)y)r;", "x: A1G\n", "ATGAAA", {"a", "b", "c", "d"})
    assert res.stats["skipped_zero_nt"] == 1 and [r.branch for r in res.branches] == ["x"]


def test_mutations_outside_the_coding_region_count_toward_nt_total_only():
    ref = "GGATGAAACCC"                    # coding region 3..8 (ATGAAA); bases 1-2 and 9-11 are UTR
    cds = bc.check_frame(ref, 3, 8)
    tree = bc.parse_newick("(((a,b)x,c)y,d)r;")
    muts = {"x": [("G", 1, "A"), ("A", 3, "C"), ("C", 9, "T")]}
    _, start, _, _, n_tip = bc.main_clade_and_start(tree, {"a", "b", "c"}, {"a", "b", "c"}, 1.0)
    res = bc.walk_branches(tree, muts, ref, cds, start, n_tip)
    assert [r.branch for r in res.branches] == ["x"]
    assert (res.branches[0].nt_total, res.branches[0].nt_in_cds) == (3, 1)
    assert res.stats["mutations_outside_cds"] == 2
    assert res.events[0].codon == 1 and res.events[0].isolated is False


def test_root_mutations_are_applied_to_the_parent_state():
    """The archive skipped the root's own mutations. Here the root carries one and a child must see it."""
    tree = bc.parse_newick("((a,b)x,c)r;")
    cds = bc.check_frame("ATGAAA")
    muts = {"r": [("A", 4, "C")], "x": [("C", 4, "G")]}            # AAA -> CAA at the root, then CAA -> GAA
    clade, start, _, _, nt = bc.main_clade_and_start(tree, {"a", "b", "c"}, {"a", "b", "c"}, 1.0)
    assert start == "r"
    res = bc.walk_branches(tree, muts, "ATGAAA", cds, "r", nt)
    assert [(e.branch, e.codon_from, e.codon_to) for e in res.events] == [("x", "CAA", "GAA")]
    assert res.stats["ancestral_mutations"] == 1


def test_wrong_reference_is_a_hard_error_not_a_silent_miscount():
    with pytest.raises(bc.InputError, match="tracked sequence"):
        build("((a,b)x,c)r;", "x: G1A", "ATGAAA", {"a", "b", "c"})
    with pytest.raises(bc.InputError, match="beyond"):
        build("((a,b)x,c)r;", "x: A100G", "ATGAAA", {"a", "b", "c"})


def test_stop_in_reference_ancestor_codon_is_dropped_not_counted_as_amino_acid_change():
    res, _ = build("((a,b)x,c)r;", "x: A1T", "ATGAAA", {"a", "b", "c"})     # ATG -> TTG is L: fine
    assert res.events[0].category == "nonsyn"
    res, _ = build("((a,b)x,c)r;", "x: A4T", "ATGAAA", {"a", "b", "c"})
    assert res.events[0].codon_to == "TAA" and res.events[0].category == "dropped_stop"


def test_two_records_at_one_position_restore_cleanly_when_leaving_the_branch():
    """Archive bug: undo restored in forward order, so a position changed twice came back wrong."""
    res, _ = build("(((a,b)x,c)y,(d,e)z)r;", "x: A4G,G4T\nz: A4C", "ATGAAA", {"a", "b", "c", "d", "e"}, purity=1.0)
    ev = {e.branch: e for e in res.events}
    assert (ev["x"].codon_to, ev["x"].category, ev["x"].n_mut, ev["x"].n_nt) == ("TAA", "dropped_stop", 2, 1)
    assert ev["z"].codon_from == "AAA" and ev["z"].codon_to == "CAA"


def test_no_target_tips_and_bad_purity_are_refused():
    tree = bc.parse_newick("(a,b)r;")
    with pytest.raises(bc.InputError):
        bc.main_clade_and_start(tree, set(), set(), 0.9)
    with pytest.raises(ValueError):
        bc.main_clade_and_start(tree, {"a"}, {"a"}, 0)
    with pytest.raises(ValueError):
        bc.main_clade_and_start(tree, {"a"}, {"a"}, 1.5)


def test_single_target_tip_gives_no_branches_not_a_crash():
    res, start = build("((a,b)x,c)r;", "x: A1T", "ATGAAA", {"a"})
    assert start == "a" and res.branches == []
    b = bc.branch_table(res)
    assert b.empty and list(b.columns) == bc.BRANCH_COLUMNS
    c = bc.column_table(res, "ATGAAA", bc.Cds(1, 6))
    assert len(c) == 2 and c["events"].sum() == 0


def test_stray_mislabelled_tips_do_not_drag_the_start_to_the_root():
    tree = bc.parse_newick("(((a,b)x,(c,d)y)clade,stray)r;")
    tips = {"a", "b", "c", "d"}
    clade, start, *_ = bc.main_clade_and_start(tree, tips | {"stray"} - {"stray"}, tips, 0.95)
    assert clade == "clade" and start == "clade"


def test_numbering_offset_is_off_unless_given_and_exact_when_given(planted):
    res, _, _, _ = planted
    cds = bc.Cds(1, 27)
    assert "site_number" not in bc.column_table(res, REF, cds).columns
    c = bc.column_table(res, REF, cds, numbering_offset=2)
    assert list(c["site_number"]) == list(range(-1, 8)) and list(c.columns)[1] == "site_number"
    e = bc.events_frame(res, numbering_offset=2)
    assert (e["site_number"] == e["codon"] - 2).all()


def test_ref_pos_is_one_based_along_the_record_with_a_utr():
    ref = "GG" + "ATGAAACCC"
    cds = bc.check_frame(ref, 3, 11)
    res = bc.WalkResult()
    c = bc.column_table(res, ref, cds)
    assert list(c["ref_pos"]) == [3, 6, 9] and list(c["ref_codon"]) == ["ATG", "AAA", "CCC"]


# ---------------------------------------------------------------------------
# an independent oracle on random trees
# ---------------------------------------------------------------------------

def random_case(seed):
    rng = random.Random(seed)
    n_codons = rng.randint(3, 10)
    while True:
        ref = "".join(rng.choice("ACGT") for _ in range(3 * n_codons))
        if all(bc.CODON2AA[ref[i:i + 3]] != "*" for i in range(0, len(ref), 3)):
            break
    names = iter(f"n{i}" for i in range(10_000))
    tips = []

    def grow(depth):
        name = next(names)
        if depth == 0 or rng.random() < 0.25:
            tips.append(name)
            return name, []
        kids = [grow(depth - 1) for _ in range(rng.randint(2, 3))]
        return name, kids

    root = grow(4)
    while len(tips) < 4:
        tips.clear()
        root = grow(5)

    def newick(node):
        name, kids = node
        return name if not kids else "(" + ",".join(newick(k) for k in kids) + ")" + name

    seqs = {}
    muts = {}

    def assign(node, parent_seq):
        name, kids = node
        seq = list(parent_seq)
        recs = []
        for _ in range(rng.choice([0, 0, 1, 2, 3, 5, 9, 10, 12])):
            p = rng.randrange(len(seq))
            to = rng.choice([b for b in "ACGT" if b != seq[p]])
            recs.append((seq[p], p + 1, to))
            seq[p] = to
        if recs:
            muts[name] = recs
        seqs[name] = "".join(seq)
        for k in kids:
            assign(k, seq)

    assign(root, ref)
    return newick(root) + ";", muts, ref, tips, seqs, root


def oracle(tree, muts, seqs, start, n_tip, ref, cap, include_terminal):
    """Events from complete node sequences, with no undo bookkeeping."""
    out = []
    for node in tree.preorder:
        if node == tree.root or n_tip[node] == 0:
            continue
        if not any(a == start for a in _ancestors(tree, node)):
            continue
        if node == start:
            continue
        recs = muts.get(node, [])
        terminal = not tree.children[node]
        if (terminal and not include_terminal) or (cap and len(recs) >= cap) or not recs:
            continue
        par = seqs[tree.parent[node]]
        child = seqs[node]
        codons = sorted({(p - 1) // 3 for _, p, _ in recs})
        for c in codons:
            a, b = par[3 * c:3 * c + 3], child[3 * c:3 * c + 3]
            n_mut = sum((p - 1) // 3 == c for _, p, _ in recs)
            if a == b:
                cat, n_nt, n_par = "reverted", 0, None
            else:
                n_nt = sum(x != y for x, y in zip(a, b))
                if "*" in (bc.CODON2AA[a], bc.CODON2AA[b]):
                    cat, n_par = "dropped_stop", None
                elif bc.CODON2AA[a] == bc.CODON2AA[b]:
                    cat, n_par = "syn", 0
                else:
                    cat = "nonsyn"
                    n_par = min(sum(x != y for x, y in zip(a, cod)) for cod in bc.AA2CODONS[bc.CODON2AA[b]])
            out.append((node, c + 1, a, b, cat, n_nt, n_mut, n_par, len(recs) == n_mut))
    return out


def _ancestors(tree, node):
    while True:
        yield node
        if node == tree.root:
            return
        node = tree.parent[node]


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("include_terminal, cap", [(False, 10), (True, 10), (True, None)])
def test_random_trees_match_the_independent_oracle(seed, include_terminal, cap):
    newick, muts, ref, tips, seqs, _ = random_case(seed)
    tree = bc.parse_newick(newick)
    paths = "\n".join(f"{n}: " + ",".join(f"{f}{p}{t}" for f, p, t in r) for n, r in muts.items())
    parsed = bc.parse_mutation_paths(paths.splitlines(), tree)
    rng = random.Random(seed)
    targets = {t for t in tips if rng.random() < 0.7} or {tips[0], tips[1]}
    cds = bc.check_frame(ref)
    clade, start, _, _, n_tip = bc.main_clade_and_start(tree, targets, targets, 0.9)
    res = bc.walk_branches(tree, parsed, ref, cds, start, n_tip, max_nt=cap, include_terminal=include_terminal)
    got = [(e.branch, e.codon, e.codon_from, e.codon_to, e.category, e.n_nt, e.n_mut, e.n_min_from_parent, e.isolated)
           for e in res.events]
    want = oracle(tree, parsed, seqs, start, n_tip, ref, cap, include_terminal)
    assert sorted(got) == sorted(want)
    # table internals agree with the events they were built from
    b = bc.branch_table(res, include_terminal=include_terminal)
    c = bc.column_table(res, ref, cds)
    assert b["codons_touched"].sum() == len(got) == c["events"].sum()
    for col in [*bc._CLASS_COUNTS, *bc._EXCESS, *bc._OTHER]:
        assert b[col].sum() == c[col].sum()
    if cap:
        assert (b["nt_total"] < cap).all()


def test_walk_is_deterministic():
    a = load_fixture()[-1]
    b = load_fixture()[-1]
    assert a.events == b.events and a.branches == b.branches and a.stats == b.stats


def test_walk_leaves_no_state_behind_between_calls():
    tree, muts, ref, cds, start, _ = load_fixture()
    n_tip = bc.count_below(tree, set((FIX / "tips.txt").read_text().split()))
    first = bc.walk_branches(tree, muts, ref, cds, start, n_tip)
    second = bc.walk_branches(tree, muts, ref, cds, start, n_tip)
    assert first.events == second.events


# ---------------------------------------------------------------------------
# table files
# ---------------------------------------------------------------------------

def test_table_files_are_byte_identical_across_runs_and_round_trip(tmp_path, planted):
    _, b, c, _ = planted
    p1 = bc.write_table(b, tmp_path / "a" / "b.tsv.gz", {"note": "SYNTHETIC test tree"})
    p2 = bc.write_table(b, tmp_path / "b" / "b.tsv.gz", {"note": "SYNTHETIC test tree"})
    assert p1.read_bytes() == p2.read_bytes()
    back, prov = bc.read_table(p1)
    assert prov == {"note": "SYNTHETIC test tree"}
    pd.testing.assert_frame_equal(back, b, check_dtype=False)
    assert gzip.open(p1, "rt").read().startswith("# note: SYNTHETIC test tree\nbranch\t")


def test_plain_tsv_also_round_trips(tmp_path, planted):
    _, _, c, _ = planted
    p = bc.write_table(c, tmp_path / "c.tsv")
    back, prov = bc.read_table(p)
    assert prov == {} and list(back["codon"]) == list(c["codon"])
    assert list(back["ref_codon"]) == list(c["ref_codon"])


@pytest.mark.parametrize("value", ["/home/someone/file", "see /data/x", "line\nbreak"])
def test_provenance_refuses_absolute_paths_and_newlines(tmp_path, planted, value):
    with pytest.raises(ValueError):
        bc.write_table(planted[1], tmp_path / "x.tsv", {"tree": value})


def test_branch_with_numeric_looking_names_reads_back_as_text(tmp_path):
    df = pd.DataFrame({"branch": ["123", "0045"], "parent": ["7", "8"], "x": [1, 2]})
    p = bc.write_table(df, tmp_path / "t.tsv.gz")
    back, _ = bc.read_table(p)
    assert list(back["branch"]) == ["123", "0045"]


def test_column_docs_cover_every_output_column():
    cols = set(bc.BRANCH_COLUMNS) | set(bc.COLUMN_COLUMNS) | {"terminal", "site_number"}
    assert cols <= set(bc.COLUMN_DOCS), cols - set(bc.COLUMN_DOCS)
    assert all(len(v) > 15 and "\n" not in v for v in bc.COLUMN_DOCS.values())


def test_summary_text_reports_the_counts(planted):
    res, b, c, _ = planted
    text = bc.summary_text(res, b, c, ["header line"])
    assert text.startswith("header line") and "scored: 7" in text and "skipped for too many changes: 1" in text
    assert "synonymous : [1, 0, 0]" in text and "amino acid changing : [3, 4, 4]" in text
    assert "2 nt where 1 would do 3; 3 nt where 1 would do 1; 3 nt where 2 would do 1" in text
    assert "dropped (stop) 2; dropped (ambiguous) 1; reverted 1" in text
    assert "codon columns: 9; columns with an amino-acid change: 6" in text


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def cli_args(tmp_path, *extra, pb=False):
    base = ["--aln", str(FIX / "aln.fasta"), "--tips", str(FIX / "tips.txt"), "--outdir", str(tmp_path / "out")]
    src = [] if pb else ["--nwk", str(FIX / "tree.nwk"), "--paths", str(FIX / "paths.txt"), "--vcf", str(FIX / "ref.vcf")]
    return [*src, *base, *extra]


def test_cli_end_to_end_matches_the_library(tmp_path, capsys):
    assert cli.main(cli_args(tmp_path, "--note", "SYNTHETIC FIXTURE: not data", "--dump-events")) == 0
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == {"branch_codons.tsv.gz", "column_codons.tsv.gz", "events.tsv.gz",
                                              "long_range_nodes.tsv.gz", "long_range_changes.tsv.gz",
                                              "baseline_codons.tsv.gz", "summary.txt"}
    b, prov = bc.read_table(out / "branch_codons.tsv.gz")
    assert prov["note"] == "SYNTHETIC FIXTURE: not data" and prov["max_branch_nt"].startswith("10")
    assert prov["reference_record"] == "SYNTHETIC_ref" and prov["tree"] == "tree.nwk"
    assert list(b["branch"]) == ["n2", "n3", "n4", "n5", "n6", "n8", "n9"]
    ev, _ = bc.read_table(out / "events.tsv.gz")
    assert len(ev) == b["codons_touched"].sum() and "terminal" not in ev.columns
    c, _ = bc.read_table(out / "column_codons.tsv.gz")
    assert len(c) == 9
    assert "checked against 23 VCF sites" in (out / "summary.txt").read_text()
    assert str(tmp_path) not in "".join(gzip.open(out / "branch_codons.tsv.gz", "rt").readlines()[:12])


def test_cli_is_deterministic(tmp_path):
    cli.main(cli_args(tmp_path / "1"))
    cli.main(cli_args(tmp_path / "2"))
    for name in ("branch_codons.tsv.gz", "column_codons.tsv.gz"):
        assert (tmp_path / "1" / "out" / name).read_bytes() == (tmp_path / "2" / "out" / name).read_bytes()


def test_cli_terminal_option_and_cap_option(tmp_path):
    cli.main(cli_args(tmp_path, "--include-terminal", "--max-branch-nt", "11"))
    b, _ = bc.read_table(tmp_path / "out" / "branch_codons.tsv.gz")
    assert "terminal" in b.columns and {"t1", "t2", "n7"} <= set(b["branch"])


def test_cli_codon_region_and_offset(tmp_path):
    cli.main(cli_args(tmp_path, "--numbering-offset", "1"))
    c, _ = bc.read_table(tmp_path / "out" / "column_codons.tsv.gz")
    assert list(c["site_number"])[:2] == [0, 1]


def test_cli_frame_failure_exits_with_a_message(tmp_path):
    with pytest.raises(bc.FrameError):
        cli.main(cli_args(tmp_path, "--cds-end", "26"))
    assert not (tmp_path / "out" / "branch_codons.tsv.gz").exists()


def test_cli_refuses_missing_inputs_and_bad_options(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["--aln", str(FIX / "aln.fasta"), "--tips", str(FIX / "tips.txt"), "--outdir", str(tmp_path)])
    with pytest.raises(SystemExit, match="threads"):
        cli.main(cli_args(tmp_path, "--threads", "15"))
    with pytest.raises(SystemExit, match="threads"):
        cli.main(cli_args(tmp_path, "--threads", "0"))
    with pytest.raises(SystemExit):
        cli.main(cli_args(tmp_path, "--max-branch-nt", "-1"))


def test_cli_empty_and_foreign_tip_lists(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n")
    args = cli_args(tmp_path)
    args[args.index("--tips") + 1] = str(empty)
    with pytest.raises(SystemExit, match="none of the --tips"):
        cli.main(args)
    foreign = tmp_path / "foreign.txt"
    foreign.write_text("nope1\nnope2\n")
    args[args.index("--tips") + 1] = str(foreign)
    with pytest.raises(SystemExit, match="none of the --tips"):
        cli.main(args)


def test_cli_partial_tip_list_is_reported_and_can_be_made_fatal(tmp_path):
    tips = tmp_path / "tips.txt"
    tips.write_text((FIX / "tips.txt").read_text() + "ghost\n")
    args = cli_args(tmp_path)
    args[args.index("--tips") + 1] = str(tips)
    cli.main(args)
    _, prov = bc.read_table(tmp_path / "out" / "branch_codons.tsv.gz")
    assert prov["target_tips_not_in_tree"] == "1"
    with pytest.raises(SystemExit, match="not in the tree"):
        cli.main([*args, "--require-all-tips"])


def test_cli_nothing_passes_the_filters_is_an_error_not_an_empty_table(tmp_path):
    with pytest.raises(SystemExit, match="no branch passed"):
        cli.main(cli_args(tmp_path, "--max-branch-nt", "2"))


def test_cli_wrong_reference_is_caught_by_the_vcf_check(tmp_path):
    aln = tmp_path / "aln.fasta"
    aln.write_text(">x\nAAGGCTAAACTGTCTGGATTTCGAANA\n")          # base 2 differs from the VCF REF
    args = cli_args(tmp_path)
    args[args.index("--aln") + 1] = str(aln)
    with pytest.raises(bc.InputError, match="disagrees with the VCF"):
        cli.main(args)


def test_cli_without_vcf_still_checks_every_mutation_base(tmp_path):
    aln = tmp_path / "aln.fasta"
    aln.write_text(">x\nAAGGCTAAACTGTCTGGATTTCGAANA\n")
    args = cli_args(tmp_path)
    i = args.index("--vcf")
    del args[i:i + 2]
    args[args.index("--aln") + 1] = str(aln)
    with pytest.raises(bc.InputError, match="tracked sequence"):
        cli.main(args)


def test_cli_pb_mode_runs_matutils_with_the_requested_threads(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check, capture_output):
        calls.append(cmd)
        d = Path(cmd[cmd.index("-d") + 1])
        d.mkdir(parents=True, exist_ok=True)
        if "-t" in cmd:
            (d / "pb.nwk").write_text((FIX / "tree.nwk").read_text())
            (d / "pb_paths.txt").write_text((FIX / "paths.txt").read_text())
        else:
            (d / "pb_ref.vcf").write_text((FIX / "ref.vcf").read_text())

    monkeypatch.setattr(subprocess, "run", fake_run)
    pb = tmp_path / "tree.pb"
    pb.write_bytes(b"")
    assert cli.main([*cli_args(tmp_path, "--threads", "3", pb=True), "--pb", str(pb)]) == 0
    assert len(calls) == 2 and all(c[0] == "matUtils" and c[c.index("-T") + 1] == "3" for c in calls)
    assert any("-A" in c for c in calls) and any("-v" in c and "-n" in c for c in calls)
    b, prov = bc.read_table(tmp_path / "out" / "branch_codons.tsv.gz")
    assert prov["tree"] == "tree.pb" and len(b) == 7


def test_cli_missing_matutils_is_a_clear_exit(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("matUtils")

    monkeypatch.setattr(subprocess, "run", boom)
    pb = tmp_path / "tree.pb"
    pb.write_bytes(b"")
    with pytest.raises(SystemExit, match="matUtils is not on PATH"):
        cli.main([*cli_args(tmp_path, pb=True), "--pb", str(pb)])


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------

ABSOLUTE = re.compile(r"[\"'](/home|/data|/software|/tmp|/Users)\b|oml4h")


@pytest.mark.parametrize("path", [ROOT / "mutacc" / "branch_codons.py", ROOT / "tools" / "branch_codon_tables.py"],
                         ids=lambda p: p.name)
def test_no_absolute_paths_in_sources(path):
    assert not ABSOLUTE.search(path.read_text())


def test_fixture_files_are_marked_synthetic_and_carry_no_paths():
    for f in FIX.iterdir():
        text = f.read_text()
        assert "/home" not in text and "oml4h" not in text
    assert (FIX / "aln.fasta").read_text().startswith(">SYNTHETIC")


# ---------------------------------------------------------------------------
# skipped codons: columns where the protein of interest has no residue
# ---------------------------------------------------------------------------

def walk_with_skip(skip):
    tree = bc.parse_newick((FIX / "tree.nwk").read_text())
    muts = bc.parse_mutation_paths((FIX / "paths.txt").read_text().splitlines(), tree)
    cds = bc.check_frame(REF)
    tips = set((FIX / "tips.txt").read_text().split())
    clade, start, size, n_all, n_tip = bc.main_clade_and_start(tree, tips, tips, 0.95)
    return bc.walk_branches(tree, muts, REF, cds, start, n_tip, max_nt=10, skip_codons=skip)


def test_skipped_codon_makes_no_events_but_changes_nothing_else():
    full = walk_with_skip(())
    busiest = pd.Series([e.codon for e in full.events]).value_counts().index[0]
    skipped = walk_with_skip({busiest})
    assert busiest not in {e.codon for e in skipped.events}
    kept = [(e.branch, e.codon, e.codon_to) for e in full.events if e.codon != busiest]
    assert [(e.branch, e.codon, e.codon_to) for e in skipped.events] == kept
    # the branch list and every branch's total are unchanged; only the in-coding-region count falls
    assert [(b.branch, b.nt_total) for b in skipped.branches] == [(b.branch, b.nt_total) for b in full.branches]
    assert skipped.stats["mutations_in_skipped_codons"] > 0
    lost = sum(b.nt_in_cds for b in full.branches) - sum(b.nt_in_cds for b in skipped.branches)
    assert lost == skipped.stats["mutations_in_skipped_codons"]


def test_a_skipped_codon_still_passes_its_state_to_later_branches():
    """Skipping hides events; it must not freeze the tracked sequence, or later parents would be wrong."""
    full = walk_with_skip(())
    busiest = pd.Series([e.codon for e in full.events]).value_counts().index[0]
    skipped = walk_with_skip({busiest})
    after = {(e.branch, e.codon): e.codon_from for e in skipped.events}
    before = {(e.branch, e.codon): e.codon_from for e in full.events if e.codon != busiest}
    assert after == before


# ---------------------------------------------------------------------------
# the long comparison: oldest scored node against the newest internal nodes
# ---------------------------------------------------------------------------

LONG_REF = "ATGGCTAAA"                          # M A K
LONG_TREE = "((((a,b)x2,e)x1,(c,d)y)s,f)r;"      # newest internal nodes: x2 and y (x1 and s have internal children)
LONG_PATHS = "r:\ns:\nx1: G4A\nx2: C5T\ny: T6C,A7C\n"


def long_range(skip=(), paths=LONG_PATHS):
    tree = bc.parse_newick(LONG_TREE)
    muts = bc.parse_mutation_paths(paths.splitlines(), tree)
    cds = bc.check_frame(LONG_REF)
    n_tip = bc.count_below(tree, tree.tips)
    return bc.compare_to_start(tree, muts, LONG_REF, cds, "s", n_tip, skip_codons=skip)


def test_long_comparison_picks_only_the_newest_internal_nodes():
    lr = long_range()
    assert lr.n_nodes == 2 and sorted(lr.per_node["node"]) == ["x2", "y"]


def test_a_change_made_in_two_steps_needs_two_nucleotides_from_the_start_codon():
    """GCT (Ala) -> ACT -> ATT (Ile): each branch changed one position, but from GCT the nearest Ile codon is 2 away."""
    row = long_range().per_node.set_index("node").loc["x2"]
    assert (row["nonsyn"], row["min1"], row["min2"], row["min3"], row["nt2"]) == (1, 0, 1, 0, 1)


def test_a_one_step_change_and_a_synonymous_change_on_the_other_node():
    row = long_range().per_node.set_index("node").loc["y"]
    assert (row["nonsyn"], row["min1"], row["min2"], row["syn"]) == (1, 1, 0, 1)


def test_distinct_changes_table_matches_the_per_node_table():
    lr = long_range()
    ch = lr.changes.set_index(["codon", "aa_to"])
    assert ch.loc[(2, "I"), "n_min"] == 2 and ch.loc[(3, "Q"), "n_min"] == 1
    assert ch["nodes"].sum() == lr.per_node["nonsyn"].sum()
    assert (lr.per_node["nonsyn"] == lr.per_node[["min1", "min2", "min3"]].sum(axis=1)).all()
    assert (lr.per_node["nonsyn"] == lr.per_node[["nt1", "nt2", "nt3"]].sum(axis=1)).all()


def test_skipped_codons_are_left_out_of_the_long_comparison():
    lr = long_range(skip={2})
    assert lr.per_node.set_index("node").loc["x2", "nonsyn"] == 0
    assert 2 not in set(lr.changes["codon"])


def test_a_back_mutation_leaves_no_difference_from_the_start_node():
    lr = long_range(paths="r:\ns:\nx1: G4A\nx2: A4G\ny: A7C\n")
    assert lr.per_node.set_index("node").loc["x2", "nonsyn"] == 0


def test_a_very_long_branch_still_passes_its_state_down():
    """A branch of 10 or more changes is not scored by the branch walk, but the long comparison applies it."""
    many = "r:\ns:\nx1: G4A,C5T,T6C,A7C,A8C,A1T,T2A,G3C,A9C,C9T\nx2: \ny: \n"
    lr = long_range(paths=many)
    assert lr.n_nodes == 2 and lr.per_node.set_index("node").loc["x2", "nonsyn"] > 0
