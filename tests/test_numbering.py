"""The figure numbering map is complete, consistent, and matches the scripts.

``guides/manuscript/figure_numbers.yaml`` is the only place a manuscript figure number lives.
That only helps if the map and the figure scripts cannot drift apart, which is what
this file pins:

* every output stem declared by a figure script has exactly one entry;
* every entry names a stem something actually produces;
* no two stems claim one manuscript label, and the numbering has no gaps.

The stems are read out of the source the same way ``test_output_provenance.py``
reads them -- ``STEM``-ish module constants plus the composite layouts -- so a new
figure that forgets the map fails here rather than in the manuscript.
"""

from __future__ import annotations

import ast
import re

import pytest
import yaml

from mutacc.guide import REPO_ROOT
from mutacc.numbering import NUMBERING_PATH, NumberingError, load_numbering

from test_output_provenance import DYNAMIC_STEMS, SCRIPTS, _declared_outputs

NUMBERING = load_numbering()


OUTPUT = REPO_ROOT / "output"


def _is_a_real_stem(stem: str) -> bool:
    """``_declared_outputs`` returns every bare string handed to a writing call,
    which includes column names and the ``STEM + "_summary"`` fragments. A stem
    is real if ``output/`` holds a file it names."""
    return any((OUTPUT / f"{stem}{suffix}").exists()
               for suffix in ("_values.tsv", ".pdf", ".png", ".tsv"))


def _script_stems() -> dict[str, str]:
    """stem -> the script that declares it, for stems a figure script owns."""
    found: dict[str, str] = {}
    for path in SCRIPTS:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and "STEM" in target.id and isinstance(node.value.value, str):
                    found.setdefault(node.value.value, path.name)
        # Side tables written under a name of their own rather than a constant.
        for stem in _declared_outputs(path)[0]:
            if _is_a_real_stem(stem):
                found.setdefault(stem, path.name)
    # The composites are assembled by panelise.py from the layouts, not by a
    # STEM constant, and sA's stem is built from the guide's virus name.
    for stem in DYNAMIC_STEMS:
        found.setdefault(stem, "panelise.py / make_sA_spike.py")
    return found


SCRIPT_STEMS = _script_stems()


def test_the_stem_scan_found_the_figure_set():
    """Guard the scan itself: an empty read would make every test below vacuous."""
    assert len(SCRIPT_STEMS) > 15, sorted(SCRIPT_STEMS)
    assert "sK_branch_codons" in SCRIPT_STEMS
    assert "fig01_composite" in SCRIPT_STEMS


@pytest.mark.parametrize("stem", sorted(SCRIPT_STEMS), ids=lambda s: s)
def test_every_declared_stem_has_exactly_one_numbering_entry(stem):
    """A figure with no entry has no manuscript number, and nothing would say so."""
    assert stem in NUMBERING, (
        f"{stem} is written by {SCRIPT_STEMS[stem]} but has no entry in "
        f"{NUMBERING_PATH.name}; give it a label, a parent, or retired: true"
    )


@pytest.mark.parametrize("stem", sorted(NUMBERING.stems()), ids=lambda s: s)
def test_every_numbering_entry_names_a_real_stem(stem):
    """The other direction: an entry for a figure nobody draws is stale."""
    assert stem in SCRIPT_STEMS, (
        f"{NUMBERING_PATH.name} carries {stem}, which no script in figures/ or tools/ "
        f"declares. Drop the entry, or the stem was renamed and the map missed it."
    )


def test_no_two_stems_claim_one_label():
    """The collision this file exists to stop (two tables both labelled S16)."""
    seen: dict[str, str] = {}
    for stem, entry in NUMBERING.entries.items():
        if entry.label is None:
            continue
        assert entry.label not in seen, f"{entry.label} is claimed by {seen[entry.label]} and {stem}"
        seen[entry.label] = stem


def test_the_numbering_is_contiguous_from_one():
    """A gap means a figure was dropped without re-numbering.

    The length is read from the map rather than written here, because a figure being
    added or dropped is the expected kind of change and should not need this test
    edited; a *gap* in the sequence is the defect being looked for.

    Only the *figures* are held to a run: main text 1..N, supplement A..N. Supplementary
    tables take a letter that names them rather than places them (the one shipped table is
    "Table SZ"), so a stem moving between the two sets has to close the hole it leaves in
    the figures without being measured against the tables at all.
    """
    for section in ("main", "supplementary"):
        numbers = sorted(e.number for e in NUMBERING.numbered(section).values()
                         if e.kind == "figure")
        assert numbers, f"{section} has no numbered figures"
        assert numbers == list(range(1, len(numbers) + 1)), f"{section} figure"


def test_the_supplementary_figures_are_lettered_and_the_table_is_not_in_their_run():
    """The 20 Oct 2026 change: a supplementary label is a letter, so the draft can be
    re-ordered without every caption, document and test following it around.

    A digit here would be the old scheme leaking back in, and would read as a manuscript
    number that nothing has assigned yet.
    """
    supplementary = NUMBERING.numbered("supplementary")
    figures = [e.label for e in supplementary.values() if e.kind == "figure"]
    assert figures == [f"Figure S{chr(ord('A') + i)}" for i in range(len(figures))]
    tables = sorted(e.label for e in supplementary.values() if e.kind == "table")
    # Tables are not held to a contiguous run (see mutacc.numbering._CONTIGUOUS), so
    # this is a roster rather than a sequence: X joined it on 20 Oct 2026 and W, the
    # leave-one-lineage-out alpha selection, on 21 Oct 2026.
    assert tables == ["Table SW", "Table SX", "Table SY", "Table SZ"], tables
    # The tables are lettered from the end of the alphabet, deliberately outside the
    # figures' A..O run, so that adding a supplementary figure never moves one.
    assert NUMBERING["sZ_lineage_panel"].number == 26
    assert NUMBERING["sY_dms_alpha_optima"].number == 25


def test_a_numbered_supplementary_label_is_rejected(tmp_path):
    """"Figure S1" is the scheme the letters replaced; accepting it would let half
    the supplement drift back to digits one entry at a time."""
    path = _write(tmp_path, {"supplementary": {"a": {"label": "Figure S1"}}})
    with pytest.raises(NumberingError, match="not of the form"):
        load_numbering(path)
    path = _write(tmp_path, {"supplementary": {"a": {"label": "Table S1"}}})
    with pytest.raises(NumberingError, match="not of the form"):
        load_numbering(path)


def test_a_supplementary_table_may_take_a_letter_outside_the_figure_run(tmp_path):
    """Table SZ beside figures SA and SB is the shipped arrangement, and it must
    load: the tables are named, not placed, so Z does not mean 25 tables are missing."""
    path = _write(tmp_path, {"supplementary": {
        "a": {"label": "Figure SA"}, "b": {"label": "Figure SB"}, "t": {"label": "Table SZ"},
    }})
    numbering = load_numbering(path)
    assert numbering.label_for("t") == "Table SZ"


def test_the_draft_ordering_of_15_oct_is_what_the_map_says():
    """The four moves and the swap, pinned to digits rather than to "it loaded".

    If the draft re-orders again this test changes with the map -- that is the
    point. What it stops is the map changing by accident.
    """
    assert NUMBERING.label_for("sH_contact_recovery") == "Figure SH"
    assert NUMBERING.label_for("sL_dms_sc2") == "Figure SL"
    assert NUMBERING.label_for("sM_within_stratum") == "Figure SM"
    assert NUMBERING.label_for("sN_simulation_control") == "Figure SN"
    # SL and SM swap, so neither may keep its old slot.
    assert NUMBERING["sM_within_stratum"].number == 13
    assert NUMBERING["sL_dms_sc2"].number == 12


def test_panels_take_their_parents_number():
    assert NUMBERING.label_for("fig01d_codon_accessibility") == "Figure 1"
    assert NUMBERING.label_for("fig02d_strata_panel") == "Figure 2"
    assert NUMBERING.label_for("sM_peak_alpha_by_stratum") == "Figure SM"


def test_the_retired_figures_have_no_number():
    """Retired means the code stays and the manuscript slot goes; asking for a
    number must fail loudly rather than return a stale one."""
    for stem in ("s14_lineage_set_validity", "fig02_model_vs_accessibility", "sF_spectrum_delta"):
        assert NUMBERING[stem].retired, stem
        with pytest.raises(NumberingError, match="retired"):
            NUMBERING.label_for(stem)


def test_an_unknown_stem_raises_rather_than_guessing():
    with pytest.raises(NumberingError, match="not in figure_numbers.yaml"):
        NUMBERING.label_for("s99_not_a_figure")


def test_caption_prefix_is_what_a_caption_opens_with():
    assert NUMBERING.caption_prefix("sH_contact_recovery") == "Figure SH."
    assert NUMBERING.caption_prefix("fig03_alpha_sweep") == "Figure 3."


# --- the loader rejects malformed maps -------------------------------------
#
# The map is hand-edited every time the draft moves, so the ways it can be
# mis-edited are the ways this breaks. Each case is a plausible edit.

def _write(tmp_path, data):
    path = tmp_path / "figure_numbers.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_a_duplicated_label_is_rejected(tmp_path):
    path = _write(tmp_path, {"supplementary": {
        "a": {"label": "Figure SA"}, "b": {"label": "Figure SA"},
    }})
    with pytest.raises(NumberingError, match="claimed by both"):
        load_numbering(path)


def test_a_gap_in_the_numbering_is_rejected(tmp_path):
    path = _write(tmp_path, {"supplementary": {
        "a": {"label": "Figure SA"}, "b": {"label": "Figure SC"},
    }})
    with pytest.raises(NumberingError, match="numbering is"):
        load_numbering(path)


def test_an_entry_that_is_both_numbered_and_retired_is_rejected(tmp_path):
    path = _write(tmp_path, {"main": {"a": {"label": "Figure 1", "retired": True}}})
    with pytest.raises(NumberingError, match="exactly one of"):
        load_numbering(path)


def test_an_entry_with_nothing_set_is_rejected(tmp_path):
    path = _write(tmp_path, {"main": {"a": {"note": "to do"}}})
    with pytest.raises(NumberingError, match="exactly one of"):
        load_numbering(path)


def test_a_parent_that_is_not_in_the_map_is_rejected(tmp_path):
    path = _write(tmp_path, {"main": {
        "a": {"label": "Figure 1"}, "b": {"parent": "nonexistent"},
    }})
    with pytest.raises(NumberingError, match="not in the map"):
        load_numbering(path)


def test_a_supplementary_label_in_the_main_section_is_rejected(tmp_path):
    """The sections are not interchangeable: "Figure SA" under main would print
    a supplementary label in the main text."""
    path = _write(tmp_path, {"main": {"a": {"label": "Figure SA"}}})
    with pytest.raises(NumberingError, match="not of the form"):
        load_numbering(path)


def test_an_unknown_key_is_rejected(tmp_path):
    """A typo ("labels:") would otherwise read as an entry with nothing set."""
    path = _write(tmp_path, {"main": {"a": {"label": "Figure 1", "colour": "blue"}}})
    with pytest.raises(NumberingError, match="unknown key"):
        load_numbering(path)


def test_a_missing_map_raises(tmp_path):
    with pytest.raises(NumberingError, match="no figure numbering map"):
        load_numbering(tmp_path / "absent.yaml")


# --- captions take their number from the map (A1b stage 2) -----------------

CAPTIONS = sorted((REPO_ROOT / "output").glob("*_caption.md"))


@pytest.mark.parametrize("path", CAPTIONS, ids=lambda p: p.name)
def test_every_shipped_caption_opens_with_the_number_the_map_gives_it(path):
    """The caption is where the number reaches the manuscript text.

    Before the map, three styles were in use ("Supplementary figure.", "Figure
    S18.", "Supplementary Figure S16.") and two of the numbers were ones the
    draft had already moved.
    """
    from mutacc.io import caption_label

    stem = path.name[: -len("_caption.md")]
    if stem not in NUMBERING:
        pytest.skip(f"{stem} is a table, not a figure; supplementary tables are not in the map yet")
    opening = " ".join(path.read_text().split())[:120]
    entry = NUMBERING[stem]
    if entry.retired:
        assert not re.match(r"\**\s*(Supplementary\s+)?Figure\b", opening), (
            f"{path.name} is retired and must not carry a number: {opening[:60]!r}"
        )
        return
    panels = {"fig01d": "D", "fig02ab": "A, B", "fig02c": "C", "fig02d": "D"}
    panel = next((letter for prefix, letter in panels.items() if stem.startswith(prefix)), None)
    expected = f"{caption_label(stem, panel)}."
    assert expected in opening[: len(expected) + 4], (
        f"{path.name} opens {opening[:60]!r}; the map says {expected}"
    )


def test_write_caption_refuses_a_caption_that_names_its_own_number(tmp_path):
    """The mechanism, not just the current files: a number typed into caption
    text is how "Figure S18" outlived the draft moving it to S12."""
    from mutacc.io import write_caption

    for text in ("**Figure S18. A title.** Body.",
                 "**Supplementary figure. A title.** Body.",
                 "Supplementary Figure S16. A title."):
        with pytest.raises(ValueError, match="names its own number"):
            write_caption(text, tmp_path, "sK_branch_codons")


def test_write_caption_numbers_from_the_map_and_adds_a_panel_letter(tmp_path):
    from mutacc.io import write_caption

    path = write_caption("**A title.** Body text.", tmp_path, "sH_contact_recovery")
    assert path.read_text().startswith("**Figure SH. A title.**")

    path = write_caption("**A title.** Body.", tmp_path, "fig02d_strata_panel", panel="D")
    assert path.read_text().startswith("**Figure 2D. A title.**")


def test_write_caption_gives_a_retired_figure_no_number(tmp_path):
    from mutacc.io import write_caption

    path = write_caption("**A title.** Body.", tmp_path, "s14_lineage_set_validity")
    assert path.read_text().startswith("**A title.**")


def test_write_caption_refuses_a_stem_the_map_does_not_know(tmp_path):
    from mutacc.io import write_caption

    with pytest.raises(NumberingError, match="not in figure_numbers.yaml"):
        write_caption("**A title.** Body.", tmp_path, "s99_invented")


# --- derived stems -----------------------------------------------------------

def test_a_run_time_variant_of_a_stem_takes_that_figures_number():
    """A script may build a stem for a *condition* of the same figure: sK on a
    planted tree, sG under a non-default gauge. Same figure, same number."""
    assert NUMBERING.label_for("sK_branch_codons_SYNTHETIC_FIXTURE") == "Figure SK"
    assert NUMBERING.label_for("sG_epistasis_null_wt") == "Figure SG"
    # ... but it is not an entry, so it cannot satisfy the completeness tests.
    assert "sK_branch_codons_SYNTHETIC_FIXTURE" not in NUMBERING


def test_a_variant_resolves_to_the_longest_matching_stem():
    """sM has three entries sharing a prefix; a variant must not pick the short one."""
    assert NUMBERING.label_for("sM_within_stratum_draft") == "Figure SM"
    assert NUMBERING.label_for("sM_peak_alpha_by_stratum_raw") == "Figure SM"


def test_a_genuinely_new_figure_still_raises():
    """The fallback must not turn an unmapped new figure into a silent number."""
    with pytest.raises(NumberingError, match="not in figure_numbers.yaml"):
        NUMBERING.label_for("s20_brand_new")
    with pytest.raises(NumberingError, match="not in figure_numbers.yaml"):
        NUMBERING.label_for("s11_branch")        # a prefix of a stem is not a variant of it
