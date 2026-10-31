"""The values table behind Figure 1B/1C, checked against the panels that are actually shipped.

Both panels carry the nine marked replacements and, next to each, the rank they were drawn at
(``HA2:S49N (N=7)``). Those labels are parsed back out of the placed PDFs here and the
exported table is required to agree with them, which is what stops the drawn figure and the
table drifting apart.

**Both panels are now held to the integer** (23 Oct 2026). They used to be the cropped halves
of a frozen render, and panel C was allowed to land anywhere inside its tie block: mutational
accessibility depends only on the codon and the target residue, so it takes about 860 distinct
values across 10,754 replacements, ties run about thirty deep, and that render was drawn with
an unstable sort. Demanding exact agreement would have been pinning the behaviour of a sort.

``figures/make_fig01bc.py`` now draws both panels from this very table, so the question the
old test could not ask is gone: a tie takes the first rank of its block in the table and in
the panel alike. What the exact check buys is the opposite direction -- it fails if the panel
is ever rebuilt from different values, or if the two stop being rebuilt together.

``test_tied_ranks_are_the_first_of_their_block`` keeps the tie structure itself under test,
since that is the property the ranks depend on and it is no longer implied by anything else.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mutacc.guide import GuideError, REPO_ROOT, load_guide

# matplotlib MUST be imported before PyMuPDF; see tests/test_fig01_mutation_labels.py for the
# full account. The ordering is load-bearing for the whole pytest session, not just this file.
import matplotlib  # noqa: E402,F401
matplotlib.use("Agg")

pytest.importorskip("fitz", reason="PyMuPDF is required to read the rendered panel")

import tools.export_fig01bc_values as exporter  # noqa: E402

VALUES = REPO_ROOT / "output" / f"{exporter.STEM}_values.tsv"
FIG01_LAYOUT = REPO_ROOT / "guides" / "composites" / "fig01.yaml"

#: ``HA2:S49N (N=7)`` -- a marked replacement and the rank it was drawn at.
RENDERED_LABEL = re.compile(r"((?:HA2:)?[A-Z]\d{1,3}[A-Z])\s*\(N=(\d+)\)")



@pytest.fixture(scope="module")
def panels_dir(tmp_path_factory) -> Path:
    """Draw both panels once, into a temporary directory, and hand back where they went.

    The panels are build intermediates: ``tools/build_figures.py`` deletes them after
    panelising unless ``--keep-panels`` is passed, so a test that read the path the layout
    names would pass or fail on whether a build had been run. Drawing them here makes the
    check a round trip instead -- table to panel to parsed label and back to the table --
    which is the thing worth testing, and it needs no build to have happened.
    """
    import make_fig01bc

    out = tmp_path_factory.mktemp("fig01bc")
    if make_fig01bc.main(["--panels-out", str(out), "--out", str(out)]) != 0:
        pytest.fail("figures/make_fig01bc.py did not run cleanly")
    return out


def _rendered_ranks(key: str, panels_dir: Path) -> dict[str, int]:
    """``mutation -> rank`` as drawn into the panel, read back out of the PDF's text."""
    import fitz

    import make_fig01bc

    stem = make_fig01bc.PANELS[key.removeprefix("1")][0]
    with fitz.open(panels_dir / f"{stem}.pdf") as doc:
        words = doc[0].get_text("words")
    text = " ".join(w[4] for w in sorted(words, key=lambda w: (round(w[1]), w[0])))
    found = {m.group(1): int(m.group(2)) for m in RENDERED_LABEL.finditer(text)}
    assert found, f"no '(N=...)' rank labels found in the drawn panel for {key}"
    return found


def test_layout_places_the_generated_panels():
    """The composite must name the files make_fig01bc.py writes, not a frozen render.

    The round trip above would keep passing if the layout quietly went back to placing a
    frozen PDF, so the link between what is tested and what is shipped is pinned here.
    """
    import make_fig01bc

    layout = yaml.safe_load(FIG01_LAYOUT.read_text())
    placed = {p["key"]: p["file"] for p in layout["panels"] if p.get("key") in {"1B", "1C"}}
    assert set(placed) == {"1B", "1C"}
    for key, path in placed.items():
        stem = make_fig01bc.PANELS[key.removeprefix("1")][0]
        assert path == f"build/panels/{stem}.pdf", (
            f"fig01.yaml places {path!r} for {key}; expected the generated panel"
        )
        assert "crop_left_frac" not in path and "frozen_panels" not in path


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    if not VALUES.exists():
        pytest.skip(f"{VALUES.name} not built; run tools/export_fig01bc_values.py")
    return pd.read_csv(VALUES, sep="\t")


# ---------------------------------------------------------------------------
# Shape and internal consistency
# ---------------------------------------------------------------------------

def test_columns_and_panels(table):
    assert list(table.columns) == exporter.COLS
    assert set(table["panel"]) == set(exporter.PANELS)
    # Both panels rank the very same replacements; only the order differs.
    keyed = {p: set(zip(rows["position"], rows["target_aa"]))
             for p, rows in table.groupby("panel")}
    assert len(set(map(frozenset, keyed.values()))) == 1, (
        "the two panels do not rank the same set of replacements"
    )


def test_every_alternative_residue_is_ranked_once(table):
    for panel, rows in table.groupby("panel"):
        assert len(rows) == rows["n_ranked"].iloc[0]
        assert not rows.duplicated(["position", "target_aa"]).any()
        # 19 alternatives at every site, and never the reference residue itself.
        assert (rows.groupby("position").size() == 19).all()
        assert (rows["ref_aa"] != rows["target_aa"]).all()


def test_ranks_are_competition_ranks(table):
    for panel, rows in table.groupby("panel"):
        rows = rows.sort_values("rank_tie_first")
        assert (rows["rank"] == rows["rank_tie_first"]).all()
        assert (rows["rank_tie_first"] <= rows["rank_tie_last"]).all()
        assert rows["rank"].min() == 1
        assert rows["rank_tie_last"].max() == len(rows)
        # A tie block is exactly the rows sharing a first rank, so its width must be their
        # count. Grouped on the rank rather than on ``value``: the file is written to six
        # significant figures, so two genuinely distinct values can print alike and would
        # merge into one group here even though the exporter ranked them apart.
        width = rows["rank_tie_last"] - rows["rank_tie_first"] + 1
        assert (rows.groupby("rank_tie_first")["rank_tie_first"].transform("size") == width).all()
        # Descending: a larger value never gets a worse rank.
        ordered = rows.sort_values("rank_tie_first")["value"].to_numpy()
        assert (ordered[:-1] >= ordered[1:]).all()


def test_marked_replacements_are_the_same_nine_on_both_panels(table):
    marked = {p: set(rows.loc[rows["observed"], "mutation"])
              for p, rows in table.groupby("panel")}
    assert len(set(map(frozenset, marked.values()))) == 1
    assert len(next(iter(marked.values()))) == 9


# ---------------------------------------------------------------------------
# Agreement with the rendered panel that the layout actually places
# ---------------------------------------------------------------------------

def test_marked_replacements_match_the_rendered_panel(table, panels_dir):
    """The table marks exactly the replacements the drawn panel labels."""
    for key, panel in (("1B", "B"), ("1C", "C")):
        rendered = set(_rendered_ranks(key, panels_dir))
        exported = set(table.loc[table["observed"] & (table["panel"] == panel), "mutation"])
        assert exported == rendered, (
            f"panel {panel}: table marks {sorted(exported)}, the placed render labels "
            f"{sorted(rendered)}"
        )


@pytest.mark.parametrize("key, panel", [("1B", "B"), ("1C", "C")])
def test_panel_ranks_reproduce_the_rendered_panel(table, panels_dir, key, panel):
    """Both panels are drawn from this table, so both must match it to the integer."""
    rendered = _rendered_ranks(key, panels_dir)
    rows = table[(table["panel"] == panel) & table["observed"]]
    exported = dict(zip(rows["mutation"], rows["rank"]))
    assert exported == rendered, (
        f"panel {panel} ranks disagree with the placed render: "
        f"{ {k: (exported.get(k), v) for k, v in rendered.items() if exported.get(k) != v} }"
    )


def test_tied_ranks_are_the_first_of_their_block(table):
    """A marked rank must be its tie block's own rank, not a position inside the block.

    This is the property the exact check above now rests on, and the one the frozen render
    broke: with an unstable sort the two replacements tied at 928 were drawn 13 places
    apart. Panel C is where it bites -- accessibility takes about 860 distinct values across
    10,754 replacements -- so the test asserts that panel really does carry ties, otherwise
    it would pass vacuously if the data ever changed shape.
    """
    marked = table[table["observed"]]
    assert (marked["rank"] == marked["rank_tie_first"]).all()
    panel_c = marked[marked["panel"] == "C"]
    assert (panel_c["rank_tie_last"] > panel_c["rank_tie_first"]).any(), (
        "panel C is expected to have tied marked replacements; if this fails the "
        "accessibility values have changed and this test is no longer checking anything"
    )


# ---------------------------------------------------------------------------
# Numbering
# ---------------------------------------------------------------------------

def test_site_labels_follow_the_rendered_numbering(table):
    """The three regimes, pinned at their boundaries against the exporter's own constants."""
    offset = exporter.HA_PRECURSOR_OFFSET
    assert exporter.site_label(offset + 1) == "1"
    assert exporter.site_label(offset) == f"{exporter.PRESEQUENCE_PREFIX}0"
    assert exporter.site_label(1) == f"{exporter.PRESEQUENCE_PREFIX}-{offset - 1}"
    assert exporter.site_label(offset - 1) == f"{exporter.PRESEQUENCE_PREFIX}-1"
    last_first_chain = offset + exporter.SECOND_CHAIN_START - 1
    assert exporter.site_label(last_first_chain) == str(exporter.SECOND_CHAIN_START - 1)
    assert exporter.site_label(last_first_chain + 1) == f"{exporter.SECOND_CHAIN_PREFIX}1"
    # And every site in the shipped table is one of the three forms, numbered from 1 upwards.
    forms = re.compile(
        rf"^(?:{exporter.PRESEQUENCE_PREFIX}-?\d+|\d+|{re.escape(exporter.SECOND_CHAIN_PREFIX)}\d+)$"
    )
    assert table["site"].astype(str).map(lambda s: bool(forms.match(s))).all()


def test_mutation_labels_put_the_chain_prefix_first():
    assert exporter.mutation_label("328", "T", "A") == "T328A"
    assert exporter.mutation_label(f"{exporter.SECOND_CHAIN_PREFIX}49", "S", "N") == "HA2:S49N"


# ---------------------------------------------------------------------------
# The bug class that put a phantom replacement into the published panel
# ---------------------------------------------------------------------------

def test_a_stale_stored_reference_is_refused():
    """A table whose ref_aa no longer matches its root FASTA must raise, not export.

    This is the N145S defect exactly: the stored table described a root that had since been
    corrected, the diff against the comparison lineage picked up a replacement that does not
    exist, and nothing warned. See LINEAGE_DEFINITIONS.md section 6.
    """
    guide = load_guide(exporter.DEFAULT_GUIDE)
    focal = {lin.id: lin for lin in guide.lineages}[guide.primary_lineage]
    frame = guide.read_table("replacement_scores")
    frame = frame[frame["model"] == guide.models[0].key]

    # Unedited, the shipped table agrees with the root it describes.
    exporter.check_reference_against_root(frame, focal)

    stale = frame.copy()
    target = (stale["lineage"] == focal.id) & (stale["position"] == stale["position"].min())
    stale.loc[target, "ref_aa"] = "W" if frame.loc[target, "ref_aa"].iloc[0] != "W" else "A"
    with pytest.raises(GuideError, match="predates an edit to the root"):
        exporter.check_reference_against_root(stale, focal)


def test_comparison_lineage_must_be_a_real_other_lineage():
    guide = load_guide(exporter.DEFAULT_GUIDE)
    with pytest.raises(GuideError, match="must differ from the focal lineage"):
        exporter.build(guide, guide.primary_lineage, "finetune")
    with pytest.raises(GuideError, match="is not in"):
        exporter.build(guide, "not-a-lineage", "finetune")
