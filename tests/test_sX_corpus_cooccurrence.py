"""The higher-order carriage check behind Table SX's "no more than five" sentence.

``tools/check_sX_corpus_cooccurrence.py`` answers the one claim in the Table SX caption
that no shipped table can support. ``k_corpus_leakage_pairs.tsv.gz`` is pairwise, so it
can say two derived residues were seen together and never that five were; the caption
nonetheless asserts that no corpus sequence carries more than five of the nine. That is
an order-9 statement, it needs the corpus at sequence level, and the corpus is not
shipped here.

So the tests split in two:

* ``carried()`` is the whole of the scientific content -- it decides, per sequence, which
  replacements count -- and it is pure, so it is unit-tested here against synthetic rows.
  The cases that matter are the ones that make the count a *floor*: a gap or an ambiguity
  at a site is not carriage, which is the conservative direction for a claim phrased as
  "no sequence carries more than N".
* The end-to-end run needs the 4.6 MB public FASTA. It is skipped unless ``HA80_CORPUS``
  points at a copy, so the suite stays offline by default:

      HA80_CORPUS=/path/to/ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.fas pytest tests/test_sX_corpus_cooccurrence.py

``test_expectations_match_the_shipped_pins`` is the one that earns its place without the
corpus: it ties the script's hard-coded expectations to the counts
``test_sX_corpus_leakage.py`` already pins from the shipped table, so the two files cannot
drift into disagreeing about what the corpus contains.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mutacc.guide import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))
import check_sX_corpus_cooccurrence as check  # noqa: E402
import export_sX_corpus_leakage as exporter  # noqa: E402

from test_sX_corpus_leakage import EXPECTED_CORPUS_DERIVED  # noqa: E402

#: A row in which every site shows its ancestral residue: carriage must be empty.
ANCESTRAL_ROW = {label: ancestral for label, _site, ancestral, _derived in exporter.REPLACEMENTS}
#: A row in which every site shows its derived residue: carriage must be all nine.
DERIVED_ROW = {label: derived for label, _site, _ancestral, derived in exporter.REPLACEMENTS}

ALL_LABELS = frozenset(ANCESTRAL_ROW)


def test_ancestral_row_carries_nothing() -> None:
    assert check.carried(ANCESTRAL_ROW) == frozenset()


def test_derived_row_carries_all_nine() -> None:
    assert check.carried(DERIVED_ROW) == ALL_LABELS


def test_one_derived_residue_is_carried_alone() -> None:
    row = dict(ANCESTRAL_ROW, **{"HA1:T328A": "A"})
    assert check.carried(row) == {"HA1:T328A"}


@pytest.mark.parametrize("no_state", sorted(exporter.NO_STATE))
def test_a_site_with_no_state_is_not_carriage(no_state: str) -> None:
    """A gap, an ambiguity or a stop at a site must not count as the derived residue.

    This is what keeps the per-sequence count a floor. If any of these were counted the
    "no more than five" claim could be satisfied or broken by alignment artefacts rather
    than by sequence content.
    """
    row = dict(DERIVED_ROW, **{"HA1:I160K": no_state})
    assert check.carried(row) == ALL_LABELS - {"HA1:I160K"}


def test_a_third_residue_is_not_carriage() -> None:
    """Neither ancestral nor derived: a site that mutated to something else entirely."""
    row = dict(DERIVED_ROW, **{"HA1:Q173R": "W"})
    assert check.carried(row) == ALL_LABELS - {"HA1:Q173R"}


def test_no_derived_residue_is_itself_a_no_state_code() -> None:
    """Documents why ``carried()``'s ``not in NO_STATE`` guard cannot hide a real call.

    The guard is belt-and-braces: it only ever fires on a residue that is not the derived
    one anyway. If a replacement were ever defined with a derived residue inside
    NO_STATE the guard would silently drop it, so the premise is pinned rather than
    assumed.
    """
    derived = {d for _label, _site, _ancestral, d in exporter.REPLACEMENTS}
    assert derived.isdisjoint(exporter.NO_STATE)


def test_replacements_are_well_formed() -> None:
    labels = [label for label, _s, _a, _d in exporter.REPLACEMENTS]
    assert len(labels) == 9
    assert len(set(labels)) == 9
    for label, _site, ancestral, derived in exporter.REPLACEMENTS:
        assert ancestral != derived, label


def test_expectations_match_the_shipped_pins() -> None:
    """The script's constants must agree with the table the suite already pins.

    Without the corpus this is the test that stops the two files drifting: the rarest
    replacement and its count are read back out of ``test_sX_corpus_leakage``'s own
    expectations rather than retyped.
    """
    rarest = min(EXPECTED_CORPUS_DERIVED, key=EXPECTED_CORPUS_DERIVED.get)
    assert check.EXPECT_MIN_LABEL == rarest
    assert check.EXPECT_MIN_COUNT == EXPECTED_CORPUS_DERIVED[rarest]
    assert check.EXPECT_SEQUENCES > max(EXPECTED_CORPUS_DERIVED.values())
    assert check.EXPECT_PAIRS == 36
    assert len(check.EXPECT_NEVER_COOCCUR) == check.EXPECT_PAIRS - check.EXPECT_PAIRS_COOCCURRING
    for pair in check.EXPECT_NEVER_COOCCUR:
        assert pair <= ALL_LABELS, pair
        assert len(pair) == 2, pair


def _corpus() -> Path | None:
    raw = os.environ.get("HA80_CORPUS")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


@pytest.mark.slow
@pytest.mark.skipif(_corpus() is None, reason="set HA80_CORPUS to the public HA-80 FASTA")
def test_end_to_end_against_the_public_corpus() -> None:
    """Every pinned expectation holds on the real corpus, including the max of five."""
    corpus = _corpus()
    assert corpus is not None  # for type checkers; skipif already guarantees it
    assert (
        check.main(
            [
                "--corpus",
                str(corpus),
                "--root",
                str(REPO_ROOT / "data" / "roots" / "h3n2_13lineage" / "J.2.nt.fa"),
            ]
        )
        == 0
    )
