"""How many times a replacement arose: planted truths on a synthetic tree.

Every tree in this file is SYNTHETIC and every count is planted by hand in the comment
beside it. Nothing here is a result about any virus.

The thing worth testing is the definition, not the arithmetic. An origin is a branch that
gained a residue the lineage root does not carry, so the tests below plant, one at a time:
a replacement that arose twice in different parts of the tree, the same residue reached by
two different codons, a synonymous change, a reversion back to the root residue, a change
through a stop and through an ambiguous codon, and a codon the reference numbering has no
position for. Each must be counted or ignored for its own reason.
"""

import pandas as pd
import pytest

from mutacc import branch_codons as bc
from mutacc import homoplasy as hp

# M K S G F  -- five codons, so positions 1..5 of a (pretend) root protein.
REF = "ATGAAAAGTGGATTT"
#: Identity numbering: reference codon c is root-protein position c. Real runs shift by
#: the signal peptide and drop codons H3 has no residue at; those are tested separately.
IDENT = {c: c for c in range(1, 6)}
ROOT_PROTEIN = "MKSGF"


def walk(newick: str, paths: str, tips, ref: str = REF):
    tree = bc.parse_newick(newick)
    muts = bc.parse_mutation_paths(paths.splitlines(), tree)
    cds = bc.check_frame(ref)
    tips = set(tips)
    n_tip = bc.count_below(tree, tips)
    start = bc.mrca(tree, tips, n_tip)
    return bc.walk_branches(tree, muts, ref, cds, start, n_tip,
                            max_nt=None, include_terminal=True)


def frame(newick, paths, tips, ref=REF, root=ROOT_PROTEIN, numbering=None):
    return hp.origins_frame(walk(newick, paths, tips, ref), numbering or IDENT, root)


# ---------------------------------------------------------------------------
# mrca: the scope the walk is given
# ---------------------------------------------------------------------------

def test_mrca_is_the_lowest_node_holding_every_tip():
    tree = bc.parse_newick("(((a,b)ab,c)abc,d)root;")
    assert bc.mrca(tree, ["a", "b"]) == "ab"
    assert bc.mrca(tree, ["a", "c"]) == "abc"
    assert bc.mrca(tree, ["a", "d"]) == "root"
    assert bc.mrca(tree, ["b"]) == "b"


def test_mrca_refuses_a_tip_the_tree_does_not_have():
    tree = bc.parse_newick("((a,b)ab,c)root;")
    with pytest.raises(bc.InputError, match="not in the tree"):
        bc.mrca(tree, ["a", "zzz"])
    with pytest.raises(bc.InputError, match="no tips"):
        bc.mrca(tree, [])


def test_a_paraphyletic_set_returns_the_whole_clades_ancestor():
    """Eight of the thirteen real lineages are 'clade minus its subclades'. The ancestor of
    such a set is the whole clade's, and that is the answer, not an error."""
    tree = bc.parse_newick("((own1,(sub1,sub2)sub)inner,out)root;")
    assert bc.mrca(tree, ["own1", "sub1"]) == "inner"


# ---------------------------------------------------------------------------
# What counts as an origin
# ---------------------------------------------------------------------------

def test_one_replacement_arising_twice_is_one_homoplasy_with_two_origins():
    # Codon 2 (AAA, K) -> AAT (N) on two sister branches: two origins, one route.
    got = frame("((x)left,(y)right)root;",
                "left: A6T\nright: A6T\n", ["x", "y"])
    assert list(got["position"]) == [2]
    assert got.iloc[0]["aa"] == "N"
    assert got.iloc[0]["n_origins"] == 2
    assert got.iloc[0]["n_routes"] == 1
    assert got.iloc[0]["routes"] == "AAA>AAT"


def test_the_same_residue_by_two_different_codons_is_two_routes():
    # Codon 5 (TTT, F) -> CTT and -> TTA, both Leu: one replacement, two origins, two routes.
    got = frame("((x)left,(y)right)root;",
                "left: T13C\nright: T15A\n", ["x", "y"])
    assert len(got) == 1
    row = got.iloc[0]
    assert (row["aa"], row["n_origins"], row["n_routes"]) == ("L", 2, 2)
    assert row["routes"] == "TTT>CTT,TTT>TTA"


def test_a_synonymous_change_is_not_an_origin():
    # Codon 2 AAA -> AAG is Lys either way.
    assert frame("((x)left,(y)right)root;", "left: A6G\n", ["x", "y"]).empty


def test_a_change_back_to_the_root_residue_is_not_a_replacement():
    # Codon 2 AAA(K) -> AAT(N) on 'left', then back to AAA(K) on 'x'. The reversion is a
    # change, but K is what the root carries, so it is not one of the counted replacements.
    got = frame("((x)left,(y)right)root;", "left: A6T\nx: T6A\n", ["x", "y"])
    assert list(got["aa"]) == ["N"]
    assert got.iloc[0]["n_origins"] == 1


def test_a_stop_and_an_ambiguous_codon_make_no_origin():
    # Codon 2 AAA -> TAA is a stop.
    assert frame("((x)left,(y)right)root;", "left: A4T\n", ["x", "y"]).empty
    # Codon 3 holds an N, so AGT -> NAT is dropped_ambiguous, not nonsyn.
    ref = REF[:6] + "N" + REF[7:]
    assert frame("((x)left,(y)right)root;", "left: G8A\n", ["x", "y"], ref=ref).empty


def test_a_codon_with_no_root_position_is_skipped():
    """The UShER reference carries 18 codons H3 has no residue at. A change there cannot be
    attributed to a position, and must not be guessed onto a neighbouring one."""
    got = frame("((x)left,(y)right)root;", "left: A6T\n", ["x", "y"],
                numbering={1: 1, 3: 3, 4: 4, 5: 5})      # codon 2 has no position
    assert got.empty


def test_internal_and_terminal_origins_are_counted_apart():
    # Codon 2 -> AAT on the internal branch 'left' and on the tip 'y'.
    got = frame("((x)left,y)root;", "left: A6T\ny: A6T\n", ["x", "y"])
    row = got.iloc[0]
    assert row["n_origins"] == 2
    assert row["n_internal_origins"] == 1


def test_the_position_must_fit_the_root_protein():
    with pytest.raises(ValueError, match="outside the"):
        frame("((x)left,(y)right)root;", "left: A6T\n", ["x", "y"],
              numbering={2: 999})


# ---------------------------------------------------------------------------
# The numbering the counts hang on
# ---------------------------------------------------------------------------

def test_numbering_shifts_by_the_signal_peptide_and_drops_unnumbered_codons():
    column = pd.DataFrame({"codon": [1, 2, 3, 4], "site_number": [11.0, 12.0, None, 13.0]})
    assert hp.numbering_from_column_table(column) == {1: 27, 2: 28, 4: 29}
    assert hp.SIGNAL_PEPTIDE == 16


def test_numbering_refuses_a_repeated_or_colliding_codon():
    with pytest.raises(ValueError, match="repeats a codon"):
        hp.numbering_from_column_table(
            pd.DataFrame({"codon": [1, 1], "site_number": [11.0, 12.0]}))
    with pytest.raises(ValueError, match="map to one"):
        hp.numbering_from_column_table(
            pd.DataFrame({"codon": [1, 2], "site_number": [11.0, 11.0]}))
    with pytest.raises(ValueError, match="site_number"):
        hp.numbering_from_column_table(pd.DataFrame({"codon": [1]}))


# ---------------------------------------------------------------------------
# Folding up to one row per lineage
# ---------------------------------------------------------------------------

def observed_table(rows):
    return pd.DataFrame(rows, columns=["lineage", "position", "aa", "obs_present"])


def test_the_summary_divides_by_what_the_tree_can_place():
    """A replacement the tree cannot place is not evidence of a single origin, so it must
    not sit in the denominator and quietly pull the percentage down."""
    origins = pd.DataFrame({
        "position": [2, 3], "aa": ["N", "A"], "n_origins": [2, 1], "n_routes": [1, 1],
        "n_internal_origins": [2, 0], "routes": ["AAA>AAT", "AGT>GCT"]})
    observed = observed_table([
        ("L", 2, "N", 1),       # placed, a homoplasy
        ("L", 3, "A", 1),       # placed, one origin
        ("L", 4, "W", 1),       # observed and covered, but the tree places no origin
        ("L", 9, "W", 1),       # observed at a position the reference does not cover
        ("L", 5, "C", 0),       # not observed at all
    ])
    got = hp.summarise({"L": origins}, observed, covered_positions=[1, 2, 3, 4, 5],
                       tree_tips={"L": 7}).iloc[0]
    assert got["n_observed"] == 4
    assert got["n_observed_covered"] == 3
    assert got["n_placed"] == 2
    assert got["n_unplaced"] == 1
    assert got["n_homoplasy"] == 1
    assert got["pct_multi_origin"] == pytest.approx(50.0)
    assert got["n_covered_positions"] == 5


def test_the_two_processes_partition_the_homoplasies():
    origins = pd.DataFrame({
        "position": [1, 2, 3], "aa": ["A", "N", "L"], "n_origins": [2, 3, 1],
        "n_routes": [1, 2, 1], "n_internal_origins": [2, 1, 0],
        "routes": ["x>y", "a>b,a>c", "p>q"]})
    observed = observed_table([("L", 1, "A", 1), ("L", 2, "N", 1), ("L", 3, "L", 1)])
    got = hp.summarise({"L": origins}, observed, [1, 2, 3], {"L": 4}).iloc[0]
    assert got["n_homoplasy"] == 2
    assert got["n_homoplasy_one_route"] == 1
    assert got["n_homoplasy_multi_route"] == 1
    assert got["n_homoplasy_one_route"] + got["n_homoplasy_multi_route"] == got["n_homoplasy"]
    assert got["n_origins_total"] == 6
    assert got["max_origins"] == 3


def test_a_placed_replacement_the_observed_table_does_not_hold_is_reported():
    """The sub-alignment and the tree are not the same sequence set; a replacement only the
    tree has is a disagreement worth seeing, not something to fold into the counts."""
    origins = pd.DataFrame({
        "position": [2], "aa": ["N"], "n_origins": [2], "n_routes": [1],
        "n_internal_origins": [2], "routes": ["AAA>AAT"]})
    got = hp.summarise({"L": origins}, observed_table([("L", 2, "N", 0)]),
                       [1, 2, 3], {"L": 4}).iloc[0]
    assert got["n_unseen"] == 1
    assert got["n_placed"] == 0
    assert pd.isna(got["pct_multi_origin"])


def test_every_summary_column_is_documented():
    origins = pd.DataFrame({
        "position": [2], "aa": ["N"], "n_origins": [1], "n_routes": [1],
        "n_internal_origins": [1], "routes": ["AAA>AAT"]})
    got = hp.summarise({"L": origins}, observed_table([("L", 2, "N", 1)]), [2], {"L": 3})
    assert list(got.columns) == [name for name, _ in hp.SUMMARY_COLUMNS]
    assert all(why for _, why in hp.SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------
# rarefaction: making the homoplasy share comparable across lineages
# ---------------------------------------------------------------------------

def test_an_origin_is_visible_when_one_of_its_tips_is_drawn():
    # One tip of ten carries it; sample five of the ten, so it is seen half the time.
    assert hp.p_origin_visible(1, 10, 5) == pytest.approx(0.5)
    # Six of ten carry it and five are drawn: at most four can be missed, so it is certain.
    assert hp.p_origin_visible(6, 10, 5) == 1.0
    # Sampling every tip sees every origin, whatever its support.
    assert hp.p_origin_visible(3, 10, 10) == 1.0
    # Sampling none sees none.
    assert hp.p_origin_visible(3, 10, 0) == 0.0


def test_origin_visibility_matches_the_hypergeometric_it_is_written_as():
    from math import comb
    for n_total, n_target, support in [(10, 4, 3), (25, 7, 2), (50, 13, 9), (12, 11, 5)]:
        expected = 1 - comb(n_total - support, n_target) / comb(n_total, n_target)
        assert hp.p_origin_visible(support, n_total, n_target) == pytest.approx(expected)


def test_origin_visibility_refuses_impossible_sizes():
    with pytest.raises(ValueError):
        hp.p_origin_visible(1, 10, 11)          # more tips sampled than exist
    with pytest.raises(ValueError):
        hp.p_origin_visible(0, 10, 5)           # an origin with no descendants
    with pytest.raises(ValueError):
        hp.p_origin_visible(11, 10, 5)          # more descendants than the lineage has


def test_two_or_more_is_the_poisson_binomial_tail():
    assert hp.p_two_or_more([]) == 0.0
    assert hp.p_two_or_more([1.0]) == 0.0               # one origin is never a homoplasy
    assert hp.p_two_or_more([0.5, 0.5]) == pytest.approx(0.25)
    assert hp.p_two_or_more([1.0, 1.0]) == pytest.approx(1.0)
    # Three independent halves: 1 - P(0) - P(1) = 1 - 1/8 - 3/8.
    assert hp.p_two_or_more([0.5, 0.5, 0.5]) == pytest.approx(0.5)
    # Unequal probabilities, by hand: P(>=2) = p1p2 + p1p3 + p2p3 - 2p1p2p3.
    a, b, c = 0.2, 0.3, 0.7
    assert hp.p_two_or_more([a, b, c]) == pytest.approx(a*b + a*c + b*c - 2*a*b*c)


def test_rarefying_to_full_depth_returns_the_counts_unchanged():
    # Two replacements, one with two origins and one with a single origin, in a 10-tip
    # lineage. At full depth the answer must be the raw count: 2 placed, 1 homoplasy.
    placed, homoplasies = hp.rarefied_counts([[3, 4], [5]], n_total=10, n_target=10)
    assert placed == pytest.approx(2.0)
    assert homoplasies == pytest.approx(1.0)


def test_rarefying_loses_recurrence_before_it_loses_the_replacement():
    # The same replacement, two origins of one tip each, in a 100-tip lineage sampled to
    # ten. Seeing it at all is far likelier than seeing both of its origins.
    placed, homoplasies = hp.rarefied_counts([[1, 1]], n_total=100, n_target=10)
    assert 0 < homoplasies < placed < 1
    assert homoplasies == pytest.approx(0.1 * 0.1)


def test_rarefied_homoplasy_falls_as_the_sample_shrinks():
    supports = [[2, 2], [5, 1], [3]]
    shares = []
    for n_target in (100, 60, 30, 10):
        placed, homoplasies = hp.rarefied_counts(supports, 100, n_target)
        shares.append(homoplasies / placed)
    assert shares == sorted(shares, reverse=True)


def _supports_frame(supports):
    return pd.DataFrame({
        "position": range(1, len(supports) + 1),
        "aa": ["N"] * len(supports),
        "n_origins": [len(s) for s in supports],
        "supports": [",".join(str(x) for x in s) for s in supports],
    })


def test_rarefy_frame_leaves_the_smallest_lineage_alone():
    # The lineage that sets the depth is sampled at full depth, so its rarefied share must
    # equal its raw one. That is the arithmetic check that the rarefaction is honest.
    per_lineage = {"small": _supports_frame([[2, 1], [3]]),
                   "big": _supports_frame([[20, 10], [30]])}
    got = hp.rarefy_frame(per_lineage, {"small": 6, "big": 60}).set_index("lineage")
    assert (got["rarefied_to"] == 6).all()
    assert got.loc["small", "rarefied_n_placed"] == pytest.approx(2.0)
    assert got.loc["small", "rarefied_pct_multi_origin"] == pytest.approx(50.0)
    # The big lineage is sampled at a tenth of its depth, so it loses recurrence: its
    # rarefied share must fall below the 50% its full-depth counts give. That drop is the
    # whole point -- the raw column reads high on deep lineages for no biological reason.
    assert got.loc["big", "rarefied_pct_multi_origin"] < 50.0


def test_rarefy_frame_needs_the_supports_column():
    bare = pd.DataFrame({"position": [1], "aa": ["N"], "n_origins": [2]})
    with pytest.raises(ValueError, match="supports"):
        hp.rarefy_frame({"L": bare}, {"L": 10})


def test_rarefy_frame_refuses_a_depth_deeper_than_a_lineage():
    with pytest.raises(ValueError, match="cannot rarefy"):
        hp.rarefy_frame({"L": _supports_frame([[1]])}, {"L": 5}, n_target=10)


def test_every_rarefied_column_is_documented():
    got = hp.rarefy_frame({"L": _supports_frame([[2, 1]])}, {"L": 4})
    assert list(got.columns) == [name for name, _ in hp.RAREFIED_COLUMNS]
    assert all(why for _, why in hp.RAREFIED_COLUMNS)


def test_origins_frame_records_each_origins_descendant_tips():
    # A gained at codon 2 on two separate branches; the frame must carry both supports,
    # largest first, so the rarefaction has what it needs without a second walk.
    newick = "((t1,t2)i1,(t3,t4)i2)root;"
    paths = "i1: A4G\ni2: A4G\n"
    tree = bc.parse_newick(newick)
    muts = bc.parse_mutation_paths(paths.splitlines(), tree)
    tips = {"t1", "t2", "t3", "t4"}
    n_tip = bc.count_below(tree, tips)
    start = bc.mrca(tree, tips, n_tip)
    result = bc.walk_branches(tree, muts, REF, bc.check_frame(REF), start, n_tip,
                              max_nt=None, include_terminal=True)
    got = hp.origins_frame(result, IDENT, ROOT_PROTEIN, n_tip)
    row = got[(got["position"] == 2) & (got["aa"] == "E")].iloc[0]
    assert row["n_origins"] == 2
    assert row["supports"] == "2,2"


def test_origins_frame_omits_supports_when_it_is_not_given_tip_counts():
    got = frame("((t1,t2)i1,(t3,t4)i2)root;", "i1: A4G\n", ["t1", "t2", "t3", "t4"])
    assert "supports" not in got.columns
