"""The prose agrees with the tables and with the figure numbering map.

``test_redteam_figures.py`` checks that a figure's *caption* matches its values
table. The ``.md`` files do not get that treatment, and they are what goes into
the manuscript and what a reader of the repository believes.

This file is the start of that (docs/notes/extended_red_team_tests.md, Batch 5). It begins
with the claims that are cheap and unambiguous:

* every figure script has a row in ``figures/README.md`` -- the drift that made
  the README document a ``make_s15.py`` that no longer exists (A3);
* no document asserts a manuscript figure number that contradicts
  ``guides/manuscript/figure_numbers.yaml``;
* the three published spectra percentages, and the two Fig 3 lineage means,
  against the tables they come from.

A claim whose table is not shipped is listed here as unverifiable rather than
skipped quietly -- see ``UNVERIFIABLE`` at the end.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from mutacc.guide import REPO_ROOT
from mutacc.numbering import load_numbering

from test_output_provenance import DYNAMIC_STEMS

NUMBERING = load_numbering()
OUTPUT = REPO_ROOT / "output"
FIGURES = REPO_ROOT / "figures"


# ---------------------------------------------------------------------------
# figures/README.md documents the scripts that exist
# ---------------------------------------------------------------------------

README = (FIGURES / "README.md").read_text()
SCRIPT_NAMES = sorted(p.name for p in FIGURES.glob("make_*.py")) + ["panelise.py"]


@pytest.mark.parametrize("name", SCRIPT_NAMES, ids=lambda n: n)
def test_every_figure_script_has_a_row_in_the_readme(name):
    """A script with no row is an undocumented figure; a row with no script is a
    reader following instructions that cannot run (A3: ``make_s15.py``)."""
    assert f"`{name}`" in README, f"figures/README.md has no row for {name}"


def test_the_readme_names_no_script_that_is_gone():
    named = set(re.findall(r"`(make_\w+\.py)`", README))
    missing = sorted(n for n in named if not (FIGURES / n).exists())
    assert not missing, f"figures/README.md documents scripts that do not exist: {missing}"


# ---------------------------------------------------------------------------
# No document contradicts the numbering map
# ---------------------------------------------------------------------------

DOCS = (
    sorted(REPO_ROOT.glob("*.md"))
    + sorted((REPO_ROOT / "docs").rglob("*.md"))
    + [FIGURES / "README.md"]
)

#: Documents that deliberately discuss the *old* numbering: the plan that
#: describes the move, and the log of what moved.
EXEMPT = {"docs/notes/extended_red_team_tests.md", "docs/notes/PLAN.md", "RUNS.md"}

#: A row of the script table: "| `make_sH_contact_recovery.py` | SH | ..." -- the script
#: in the first column and what the README calls its figure in the second.
_ROW = re.compile(r"^\|\s*~?~?`(make_\w+\.py)`~?~?\s*\|\s*~?~?([^|]*?)~?~?\s*\|")


def _stems_of(script: str) -> set[str]:
    """The output stems a figure script declares.

    Read as constants, except for the one script whose stem is built at run time from the
    guide's virus name; that one is named in ``test_output_provenance.DYNAMIC_STEMS``, and
    taking it from there rather than hard-coding it here keeps a single list of the stems a
    static read cannot resolve.
    """
    text = (FIGURES / script).read_text()
    found = set(re.findall(r"^\w*STEM\w*\s*=\s*[\"\'](\w+)[\"\']", text, re.M))
    if not found:
        letter = re.match(r"make_(s[A-Z])_", script)
        if letter:
            found = {s for s in DYNAMIC_STEMS if s.startswith(f"{letter.group(1)}_")}
    return found


@pytest.mark.parametrize("line", [l for l in README.splitlines() if _ROW.match(l)],
                         ids=lambda l: _ROW.match(l).group(1))
def test_the_readme_figure_column_agrees_with_the_map(line):
    """Every supplementary label the Figure column claims is one the map gives that
    script.

    Before 20 Oct 2026 this test could only check the weaker thing -- that a number in
    this column was the script's own *stem* number and not some other figure's -- because
    stems carried digits that the draft had moved away from. Now a script, its stem and
    its label share a letter, so the column can be held to the map directly.

    A retired figure has no label, so a row that claims one for it fails here.
    """
    script, claimed = _ROW.match(line).groups()
    stems = _stems_of(script)
    if not stems:
        # make_sA_spike.py builds its stem from the guide's virus name, so there is no
        # constant to read and nothing here to check it against.
        pytest.skip(f"{script} declares no stem constant")
    labels = {NUMBERING.label_for(stem).replace("Figure ", "").replace("Table ", "")
              for stem in stems if not NUMBERING[stem].retired}
    for letter in re.findall(r"\bS[A-Z]\b", claimed):
        assert letter in labels, (
            f"figures/README.md gives {script} the label S{letter[1:]}, which the map does "
            f"not: it writes {sorted(labels) or 'no numbered figure'}. The label comes from "
            f"{NUMBERING.path.name}, not from here."
        )


def test_the_readme_figure_column_carries_no_supplementary_digit(line=None):
    """"S13" in this column would be the old stem numbering, which no longer exists
    anywhere: the file is `sH_contact_recovery` now."""
    stray = []
    for row in README.splitlines():
        m = _ROW.match(row)
        if m and re.search(r"\bS\d+\b", m.group(2)) and "~~" not in row:
            stray.append(row.split("|")[1].strip())
    assert not stray, f"figures/README.md still numbers these scripts' figures: {stray}"


#: The root README's figure table: the first column names the figure, a later one holds the
#: command. It had kept the 15 Oct 2026 draft's digits until 2 Nov 2026 -- and not even those
#: consistently, since it called ``sM_within_stratum`` S12 where the map's old column says
#: S13. Nothing checked it, because the test above is written for ``figures/README.md``'s
#: script-first rows. A reader who follows a digit here lands on the wrong supplementary
#: figure, which is the one mistake this numbering scheme exists to prevent.
ROOT_README = (REPO_ROOT / "README.md").read_text()

#: "| SH contact recovery ... | `python figures/make_sH_contact_recovery.py ...` | ... |"
_ROOT_ROW = re.compile(r"^\|\s*([^|]*?)\s*\|\s*`python (?:figures|tools)/")


def test_the_root_readme_figure_table_carries_no_supplementary_digit():
    stray = [m.group(1) for m in (_ROOT_ROW.match(r) for r in ROOT_README.splitlines())
             if m and re.search(r"\bS\d+\b", m.group(1))]
    assert not stray, (
        f"README.md still labels these rows with the old supplementary numbers: {stray}. "
        f"The supplement is lettered; see FIGURE_LABELS.md."
    )


# ---------------------------------------------------------------------------
# Published numbers in the prose, against the tables they come from
# ---------------------------------------------------------------------------

def _values(stem: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT / f"{stem}_values.tsv", sep="\t")


def test_the_two_fig3_lineage_means_are_what_the_table_holds():
    """0.9284 and 0.7894, quoted in figures/README.md, are the lineage means at
    alpha = 0 under the masked marginals estimator.

    Repinned 23 Oct 2026 for the lineage reassignment; the audit's pair, quoted here
    until then, was 0.9294 / 0.7927."""
    claimed = {m for m in re.findall(r"0\.9284|0\.7894", README)}
    assert claimed == {"0.9284", "0.7894"}, "figures/README.md no longer quotes both means"
    frame = _values("fig03_alpha_sweep")
    at_zero = frame.loc[
        (frame["alpha"].astype(float).abs() < 1e-12)
        & (frame["metric"] == "site_logistic_auroc")
    ].set_index("model")["mean"]
    assert not at_zero.empty, "fig03 values table has no alpha = 0 site_logistic_auroc row"
    # Named cells, not "some number in the table rounds to this": the claim is
    # about the fine-tuned and base models specifically, and swapping the two
    # would otherwise pass.
    assert round(float(at_zero["ESM2_650M_HA80"]), 4) == 0.9284
    assert round(float(at_zero["ESM2_650M_HA80_raw"]), 4) == 0.7894


#: Claims whose table is not shipped. Listed, not skipped silently, so that the
#: count of unverifiable numbers in the manuscript is visible rather than zero
#: by omission. Same pattern as test_redteam_data.py's two GISAID skips.
UNVERIFIABLE = {
    "84.8% / 43.6% / 51.0% (figures/README.md, make_sE_spectra_codon_space row)":
        "the published spectra percentages are computed from the spectrum comparison "
        "passed with --results; the comparison input is archive-only. The figure's own "
        "values table is checked by tests/test_spectra.py instead.",
    "190-pair asymmetry Spearman 0.539 (figures/README.md, make_fig02 row)":
        "fig02 is retired; its values table lives in dev/retired_figures/ and is not "
        "part of the shipped set.",
}


def test_the_unverifiable_claims_are_listed_not_forgotten():
    """A placeholder with teeth: it fails if the list is emptied without the
    claims being checked somewhere, and it keeps the count honest."""
    assert UNVERIFIABLE, "nothing is unverifiable? then check the claims instead of listing them"
    for claim, reason in UNVERIFIABLE.items():
        assert reason.strip(), claim


# ---------------------------------------------------------------------------
# FIGURE_LABELS.md is the reader's copy of the map, and must not drift from it
# ---------------------------------------------------------------------------
#
# The map is machine-read; FIGURE_LABELS.md is what a person (or the manuscript's
# final renumbering pass) reads instead. A row that disagrees with the map is worse
# than no row, because it is the one a human would trust.

LABELS_DOC = REPO_ROOT / "FIGURE_LABELS.md"

#: "| Figure SH | S8 | `sH_contact_recovery` | `make_sH_contact_recovery.py` | ... |"
_LABEL_ROW = re.compile(r"^\|\s*((?:Figure|Table)\s+S[A-Z])\s*\|[^|]*\|\s*`(\w+)`\s*\|")


def _label_rows() -> dict[str, str]:
    return {m.group(1): m.group(2)
            for m in (_LABEL_ROW.match(l) for l in LABELS_DOC.read_text().splitlines()) if m}


def test_figure_labels_doc_lists_every_lettered_entry_once():
    rows = _label_rows()
    expected = {e.label for e in NUMBERING.numbered("supplementary").values()}
    assert set(rows) == expected, (
        f"FIGURE_LABELS.md and {NUMBERING.path.name} disagree on which labels exist: "
        f"only in the doc {sorted(set(rows) - expected)}, only in the map "
        f"{sorted(expected - set(rows))}"
    )
    assert len(rows) == len(set(rows.values())), "a stem appears on two rows"


@pytest.mark.parametrize("label", sorted(_label_rows()), ids=lambda s: s)
def test_each_figure_labels_row_gives_the_stem_the_map_gives(label):
    stem = _label_rows()[label]
    assert NUMBERING.label_for(stem) == label, (
        f"FIGURE_LABELS.md puts {stem} at {label}; {NUMBERING.path.name} says "
        f"{NUMBERING.label_for(stem)}"
    )


def test_no_supplementary_label_in_the_docs_is_a_digit():
    """The letters replaced digits on 20 Oct 2026. A document that reverts one is
    asserting a manuscript number that nothing has assigned."""
    for path in [LABELS_DOC, FIGURES / "README.md"]:
        text = path.read_text()
        # Rows of the old-number translation column are the one legitimate use, and
        # they are introduced by the words that say so.
        body = text.split("## The set")[0] if path == LABELS_DOC else text
        stray = re.findall(r"\b(?:Figure|Table)\s+S\d+\b", body)
        assert not stray, f"{path.name} still carries numbered supplementary labels: {stray}"
