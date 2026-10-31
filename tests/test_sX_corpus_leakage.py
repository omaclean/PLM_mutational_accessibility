"""Table SX: the lineage-defining replacements against the fine-tuning corpus.

The table answers a reviewer's question about leakage, so what is under test is mostly
provenance: that the replacements are read at the coordinates LINEAGE_DEFINITIONS.md
defines them at, that the counts are the corpus's own, and that the caption's claims are
the table's.

The strongest test here is ``test_agrees_with_corpus_membership``. The shipped
``corpus_membership.tsv.gz`` counts the same corpus through the project's multiple
alignment; this table counts it through pairwise alignment to the ancestral backbone.
The two routes share no code, so where both are well defined they have to agree, and a
numbering or frame slip in either shows up as a mismatch rather than a plausible number.

The co-occurrence table the exporter also writes is no longer drawn (the figure is one
block since 20 Oct 2026), but it is still exported and still tested here: dropping a
panel is not a reason to stop checking the numbers behind it.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import make_sX_corpus_leakage as sX
from mutacc.guide import REPO_ROOT
from mutacc.io import caption_label

sys.path.insert(0, str(REPO_ROOT / "tools"))
import export_sX_corpus_leakage as exporter  # noqa: E402

RESULTS = REPO_ROOT / "results" / "h3n2_13lineage"

#: The corpus's own bounds, from the public set the model was fine-tuned on.
CORPUS_SEQUENCES = 8099
CORPUS_LAST_DATE = "2015-02-27"
SUBSET_SEQUENCES = 1282
H3_SEQUENCES = 2530

#: Counts of the new amino acid over the whole corpus, pinned 20 Oct 2026. These are
#: the numbers the reviewer reply quotes, so a silent change must fail here.
EXPECTED_CORPUS_DERIVED = {
    "HA1:K2N": 372, "HA1:T135K": 231, "HA1:S144N": 1397, "HA1:N158D": 368,
    "HA1:I160K": 1764, "HA1:Q173R": 146, "HA1:K189R": 335, "HA1:T328A": 70,
    "HA2:S49N": 2345,
}

#: Counts of the new amino acid within the H3 sequences of the corpus, whatever the
#: host, pinned 22 Oct 2026. The human subset is nested inside these, and the gap between
#: the two columns is what the manuscript reads: HA1:T328A is the row where the residue
#: is in the subtype at all (16 sequences) but in none of the human ones.
EXPECTED_H3_DERIVED = {
    "HA1:K2N": 268, "HA1:T135K": 63, "HA1:S144N": 1061, "HA1:N158D": 89,
    "HA1:I160K": 1714, "HA1:Q173R": 24, "HA1:K189R": 302, "HA1:T328A": 16,
    "HA2:S49N": 2114,
}

#: The one replacement absent from the human subset while present in the subtype around
#: it. The distinction is load-bearing in the manuscript, so it is pinned by name.
ABSENT_FROM_HUMAN_SUBSET = "HA1:T328A"

#: Sites where the pairwise and multiple-alignment routes see the same sequences and so
#: must return the same counts exactly.
SHARED_DENOMINATOR = (
    "HA1:T135K", "HA1:I160K", "HA1:Q173R", "HA1:K189R", "HA1:T328A", "HA2:S49N",
)

#: Sites where they legitimately differ, and why. K2N sits at the signal-peptide
#: boundary, where the multiple alignment reads only 3,241 of the 8,099 sequences
#: against the pairwise route's 6,757; S144N and N158D sit in the 158-loop, which
#: carries indels across subtypes, so a column there is not the same object as a
#: backbone coordinate. Both routes must still find the residue, which is the claim the
#: table makes; the frequency quoted at these three is the pairwise one.
INDEL_ADJACENT = ("HA1:K2N", "HA1:S144N", "HA1:N158D")


@pytest.fixture(scope="module")
def frame():
    return sX.load(RESULTS)


@pytest.fixture(scope="module")
def pairs():
    return pd.read_csv(RESULTS / "k_corpus_leakage_pairs.tsv.gz", sep="\t")


def test_the_replacements_are_the_defined_ones(frame):
    assert list(frame["replacement"]) == [
        label for label, _s, _a, _d in exporter.REPLACEMENTS
    ]


def test_every_new_residue_occurs_in_the_corpus(frame):
    """The answer to the reviewer: all of them, without exception."""
    absent = frame[frame["corpus_derived"] == 0]
    assert absent.empty, f"new residue absent from the corpus: {list(absent['replacement'])}"


def test_corpus_counts_are_pinned(frame):
    assert dict(zip(frame["replacement"], frame["corpus_derived"])) == EXPECTED_CORPUS_DERIVED


def test_subtype_counts_are_pinned(frame):
    assert dict(zip(frame["replacement"], frame["h3_derived"])) == EXPECTED_H3_DERIVED


def test_the_groups_nest(frame):
    """Human subset within subtype within training data, counts and sizes alike."""
    assert (frame["subset_derived"] <= frame["h3_derived"]).all()
    assert (frame["h3_derived"] <= frame["corpus_derived"]).all()
    assert (frame["subset_sequences"] < frame["h3_sequences"]).all()
    assert (frame["h3_sequences"] < frame["corpus_sequences"]).all()
    assert (frame["h3_sequences"] == H3_SEQUENCES).all()


def test_one_replacement_is_in_the_subtype_but_not_the_human_subset(frame):
    """The manuscript sentence turns on this row, so it is pinned by name.

    Reading "absent from the human subset" as "absent from the subtype" is the error the
    split column exists to prevent: the residue is in 16 subtype sequences.
    """
    split = frame[(frame["h3_derived"] > 0) & (frame["subset_derived"] == 0)]
    assert list(split["replacement"]) == [ABSENT_FROM_HUMAN_SUBSET]
    assert int(split["h3_derived"].iloc[0]) > 0


def test_corpus_bounds_are_carried_by_the_table(frame):
    """The figure prints these, so they must be data and not typed into the script."""
    assert (frame["corpus_sequences"] == CORPUS_SEQUENCES).all()
    assert (frame["corpus_last_date"] == CORPUS_LAST_DATE).all()
    assert frame["subset"].nunique() == 1


def test_counts_never_exceed_their_denominator(frame):
    assert (frame["corpus_derived"] <= frame["corpus_typed"]).all()
    assert (frame["h3_derived"] <= frame["h3_typed"]).all()
    assert (frame["subset_derived"] <= frame["subset_typed"]).all()
    assert (frame["subset_derived"] <= frame["corpus_derived"]).all()
    assert (frame["corpus_typed"] <= CORPUS_SEQUENCES).all()
    assert (frame["subset_typed"] == SUBSET_SEQUENCES).all()


def test_percentages_match_their_counts(frame):
    """Every share is over its whole group, not over the sequences readable at the site.

    The denominator is the claim the column header makes. On readable-sequence
    denominators two rows four sequences apart printed shares a point apart (K2N 5.5%
    against N158D 4.5%), which is what this pins shut.
    """
    for prefix in ("corpus", "h3", "subset"):
        expected = 100.0 * frame[f"{prefix}_derived"] / frame[f"{prefix}_sequences"]
        assert frame[f"{prefix}_pct"].round(4).equals(expected.round(4))


def test_first_seen_predates_the_corpus_cutoff(frame):
    """Nothing in the corpus may post-date its own close, or it is not the corpus."""
    years = pd.to_numeric(frame["corpus_first_year"], errors="coerce")
    assert years.notna().all()
    assert (years <= int(CORPUS_LAST_DATE[:4])).all()


def test_three_sites_are_the_majority_of_the_subset(frame):
    """The finding that goes beyond the question asked, so it is pinned.

    At these three the new residue is not merely present but is what the subset mostly
    carried, which makes them returns to a common state rather than novel changes.
    """
    majority = set(frame.loc[frame["subset_pct"] > 50.0, "replacement"])
    assert majority == {"HA1:S144N", "HA1:I160K", "HA2:S49N"}


def test_every_replacement_is_in_one_agreement_class():
    """No site may be dropped from the cross-check by being in neither list."""
    labels = {label for label, _s, _a, _d in exporter.REPLACEMENTS}
    assert set(SHARED_DENOMINATOR) | set(INDEL_ADJACENT) == labels
    assert not set(SHARED_DENOMINATOR) & set(INDEL_ADJACENT)


def test_agrees_with_corpus_membership(frame):
    """Pairwise-to-backbone and alignment-column counting must land on the same numbers."""
    membership = pd.read_csv(RESULTS / "corpus_membership.tsv.gz", sep="\t")
    by_site = {
        (int(row.position), row.aa): int(row.corpus_count) for row in membership.itertuples()
    }
    depth = dict(zip(membership["position"], membership["corpus_depth"]))

    for label, site, ancestral, derived in exporter.REPLACEMENTS:
        row = frame[frame["replacement"] == label].iloc[0]
        backbone = membership.loc[membership["position"] == site, "backbone_aa"].iloc[0]
        assert backbone == ancestral, f"{label}: the two routes disagree on the backbone residue"
        found = by_site.get((site, derived), 0)
        if label in SHARED_DENOMINATOR:
            assert found == pytest.approx(row["corpus_derived"], abs=1)
            # Denominators may part by a sequence or two: this table drops an ambiguity
            # code as unreadable where the alignment's depth still counts it.
            assert depth[site] == pytest.approx(row["corpus_typed"], abs=3)
        else:
            assert found > 0, f"{label}: the alignment route does not see the new residue"


def test_interacting_pairs_are_the_flagged_ones(pairs):
    flagged = pairs[pairs["interacting"]]
    got = {frozenset((row.replacement_a, row.replacement_b)) for row in flagged.itertuples()}
    assert got == {frozenset(pair) for pair in exporter.INTERACTING_PAIRS}


def test_every_flagged_pair_co_occurs(pairs):
    """The second half of the reviewer's question, exported though no longer drawn."""
    quiet = pairs[pairs["interacting"] & (pairs["corpus_both_derived"] == 0)]
    assert quiet.empty


def test_pair_counts_are_bounded_by_their_singles(pairs, frame):
    """A pair cannot be carried more often than either of its residues is."""
    singles = dict(zip(frame["replacement"], frame["corpus_derived"]))
    for row in pairs.itertuples():
        assert row.corpus_both_derived <= min(
            singles[row.replacement_a], singles[row.replacement_b]
        )


def test_most_but_not_all_pairs_co_occur(pairs):
    assert len(pairs) == 36
    assert int((pairs["corpus_both_derived"] > 0).sum()) == 33


def test_backbone_check_catches_a_numbering_slip():
    """``check_backbone`` is the guard against three coordinate systems in one draft."""
    with pytest.raises(ValueError, match="not the K the replacement is defined from"):
        exporter.check_backbone("M" * 600)


def test_backbone_check_catches_a_short_backbone():
    with pytest.raises(ValueError, match="past the end"):
        exporter.check_backbone("MKT")


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, "0.0%"), (0.037, "<0.1%"), (0.3, "0.3%"), (99.532, "99.5%")],
)
def test_percent_formatter(value, expected):
    assert sX._pct(value) == expected


def test_check_rejects_a_broken_table(frame):
    with pytest.raises(ValueError, match="empty"):
        sX.check(frame.head(0))
    with pytest.raises(ValueError, match="appears twice"):
        sX.check(pd.concat([frame, frame.head(1)], ignore_index=True))
    inflated = frame.copy()
    inflated.loc[0, "corpus_derived"] = inflated.loc[0, "corpus_typed"] + 1
    with pytest.raises(ValueError, match="exceeds the sequences readable"):
        sX.check(inflated)
    unnested = frame.copy()
    unnested.loc[0, "subset_derived"] = unnested.loc[0, "h3_derived"] + 1
    with pytest.raises(ValueError, match="more than the subtype"):
        sX.check(unnested)
    unnested = frame.copy()
    unnested.loc[0, "h3_derived"] = unnested.loc[0, "corpus_derived"] + 1
    with pytest.raises(ValueError, match="more than the training data"):
        sX.check(unnested)


def test_caption_states_what_the_table_shows(frame):
    text = sX.caption(frame)
    assert "All 9 of the new amino acids are already present in the training data" in text
    assert ABSENT_FROM_HUMAN_SUBSET in text
    assert frame["h3_label"].iloc[0] in text
    for label in ("HA1:S144N", "HA1:I160K", "HA2:S49N"):
        assert label in text
    assert frame["subset"].iloc[0] in text
    # the caption must not name itself; the label comes from the numbering map
    assert not text.lstrip().lower().startswith(("figure", "table", "**figure", "**table"))


def test_the_table_carries_a_manuscript_label():
    assert caption_label(sX.STEM).startswith("Table ")
