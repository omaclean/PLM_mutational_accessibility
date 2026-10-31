"""Regression tests for the `N145S` phantom mutation in Figure 1B/C.

Background. The published 2 Jul 2026 render of Figure 1B/C marked TEN replacements for the
H3N2 J.2 -> K transition. The tenth, `N145S`, does not exist: residue 145 is Serine in J.2,
J.2.4 and K alike. It appeared because the panel diffed a STORED results table's `ref_aa`
column against a LIVE reference FASTA; the `J.2_int` root was corrected at precursor codon
161 (`aat` -> `agt`) on 15 Jun 2026 and the May tables were never rebuilt. Nothing warned.
Full account in `LINEAGE_DEFINITIONS.md` section 6.

Nothing in the suite caught this, because every existing test checked the *tables* while the
defect lived in a *rendered* figure. These tests therefore pin three different layers:

1. The rendered output. `output/fig01_composite.pdf` and the panel the layout actually
   places must carry exactly the canonical nine labels and no site-145 label at all.
2. The layout. `guides/composites/fig01.yaml` must not place the defective 2 Jul file,
   which is retained for provenance only.
3. The class of bug. For every shipped guide, a stored table's `ref_aa` column must still
   agree with the translation of the root FASTA it claims to describe. This is the
   invariant whose violation produced `N145S`, and it is checked independently of Figure 1.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest
import yaml
from Bio.Seq import Seq

from mutacc.guide import REPO_ROOT

# matplotlib MUST be imported before PyMuPDF. If fitz loads first it pulls in a conflicting
# libjpeg/freetype and Pillow breaks for the rest of the pytest session, which shows up as
# unrelated failures in every downstream figure-rendering test. Importing fitz here (rather
# than lazily inside the helpers) is deliberate: it makes the ordering explicit and local.
import matplotlib  # noqa: E402,F401  -- import order is load-bearing; see above
matplotlib.use("Agg")

pytest.importorskip("fitz", reason="PyMuPDF is required to read the rendered panels")

FIG01_LAYOUT = REPO_ROOT / "guides" / "composites" / "fig01.yaml"
FIG01_COMPOSITE = REPO_ROOT / "output" / "fig01_composite.pdf"
DEFECTIVE_PANEL = REPO_ROOT / "data" / "frozen_panels" / "fig1bc_ranked_mutations.pdf"

CANONICAL_NINE = {
    "K2N", "T135K", "S144N", "N158D", "I160K",
    "Q173R", "K189R", "T328A", "HA2:S49N",
}

# Any label at mature site 145, in either HA1 or the raw precursor numbering (161) that the
# un-canonicalised renders used. `N145S` is the published one; the others guard the variants.
SITE_145_PATTERN = re.compile(r"\b(?:HA2:)?[A-Z](?:145|161)[A-Z]\b")
LABEL_PATTERN = re.compile(r"\b(?:HA2:)?[A-Z]\d{1,3}[A-Z]\b")


def _pdf_text(path: Path) -> str:
    import fitz  # safe: matplotlib was imported at module scope, above

    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def _labels_in(path: Path) -> set[str]:
    return set(LABEL_PATTERN.findall(_pdf_text(path)))


def _placed_panel_files() -> list[Path]:
    layout = yaml.safe_load(FIG01_LAYOUT.read_text())
    return [
        REPO_ROOT / panel["file"]
        for panel in layout["panels"]
        if panel.get("key") in {"1B", "1C"}
    ]


# ---------------------------------------------------------------------------
# 1. The rendered figure
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FIG01_COMPOSITE.exists(), reason="fig01_composite.pdf not built")
def test_composite_has_no_site_145_label():
    """The shipped Figure 1 must not claim any replacement at residue 145."""
    text = _pdf_text(FIG01_COMPOSITE)
    assert "N145S" not in text, (
        "N145S is back in output/fig01_composite.pdf. Residue 145 is Serine in J.2, J.2.4 "
        "and K alike; see LINEAGE_DEFINITIONS.md section 6."
    )
    stray = SITE_145_PATTERN.findall(text)
    assert not stray, f"Figure 1 labels a replacement at site 145/161: {sorted(set(stray))}"


@pytest.mark.skipif(not FIG01_COMPOSITE.exists(), reason="fig01_composite.pdf not built")
def test_composite_mutation_labels_are_exactly_the_canonical_nine():
    found = {label for label in _labels_in(FIG01_COMPOSITE) if label in CANONICAL_NINE or SITE_145_PATTERN.fullmatch(label)}
    assert found == CANONICAL_NINE, (
        f"Figure 1 should mark exactly the nine clade-defining replacements.\n"
        f"  missing: {sorted(CANONICAL_NINE - found)}\n"
        f"  unexpected: {sorted(found - CANONICAL_NINE)}"
    )


def test_placed_panels_carry_exactly_the_canonical_nine():
    """Whatever file the layout places for 1B/1C must itself be correct."""
    for panel in _placed_panel_files():
        assert panel.exists(), f"fig01.yaml places a missing panel: {panel}"
        labels = {label for label in _labels_in(panel) if LABEL_PATTERN.fullmatch(label)}
        assert "N145S" not in labels, f"{panel.name} still marks the phantom N145S"
        assert labels == CANONICAL_NINE, (
            f"{panel.name} marks {sorted(labels)}; expected the canonical nine "
            f"{sorted(CANONICAL_NINE)}"
        )


# ---------------------------------------------------------------------------
# 2. The layout must not reinstate the defective render
# ---------------------------------------------------------------------------

def test_defective_panel_is_retained_but_never_placed():
    assert DEFECTIVE_PANEL.exists(), (
        "The 2 Jul 2026 render is kept for provenance; do not delete it."
    )
    # It is kept precisely because it is wrong, so it must still be wrong.
    assert "N145S" in _pdf_text(DEFECTIVE_PANEL), (
        "fig1bc_ranked_mutations.pdf is supposed to be the defective published copy. "
        "If it has been replaced, update data/frozen_panels/README.md and this test."
    )
    placed = {panel.resolve() for panel in _placed_panel_files()}
    assert DEFECTIVE_PANEL.resolve() not in placed, (
        "guides/composites/fig01.yaml places the DEFECTIVE 2 Jul panel. It marks ten "
        "replacements, one of which does not exist."
    )


# ---------------------------------------------------------------------------
# 3. The underlying class of bug: stored table vs reference FASTA
# ---------------------------------------------------------------------------

def _translate_root(path: Path) -> str:
    nt = "".join(
        line.strip()
        for line in path.read_text().splitlines()
        if not line.startswith(">")
    )
    return str(Seq(nt).translate()).rstrip("*")


def _stored_ref_aa(table: Path, lineage: str) -> str:
    by_position: dict[int, str] = {}
    with gzip.open(table, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        try:
            i_lin, i_pos, i_aa = (header.index(c) for c in ("lineage", "position", "ref_aa"))
        except ValueError:
            return ""
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[i_lin] == lineage:
                by_position[int(fields[i_pos])] = fields[i_aa]
    return "".join(by_position[key] for key in sorted(by_position))


@pytest.mark.parametrize("guide_name", ["h3n2_5lineage", "h3n2_13lineage"])
def test_stored_tables_still_match_their_reference_fastas(guide_name):
    """A stored table's ref_aa must agree with the root FASTA it describes.

    This is the invariant whose violation produced N145S. If a root is ever edited without
    re-running the scoring step, this fails and names the offending residue.
    """
    guide = yaml.safe_load((REPO_ROOT / "guides" / f"{guide_name}.yaml").read_text())
    table = REPO_ROOT / guide["results_dir"] / "replacement_scores.tsv.gz"
    if not table.exists():
        pytest.skip(f"{table} not shipped")

    for lineage in guide["lineages"]:
        root = REPO_ROOT / lineage["root"]
        if not root.exists():
            continue
        stored = _stored_ref_aa(table, lineage["id"])
        if not stored:
            continue
        on_disk = _translate_root(root)
        mismatches = [
            f"{table_aa}{index + 1}{disk_aa}"
            for index, (table_aa, disk_aa) in enumerate(zip(stored, on_disk))
            if table_aa != disk_aa
        ]
        assert not mismatches and len(stored) == len(on_disk), (
            f"{guide_name}/{lineage['id']}: the stored table disagrees with {root.name} "
            f"at {len(mismatches)} position(s): {mismatches[:10]}. The table predates an "
            f"edit to the root; re-run the scoring step rather than re-plotting."
        )


def test_site_145_is_serine_in_every_j_lineage_and_in_k():
    """Precursor codon 161 (mature H3 145) is Serine throughout the J.2 -> K path."""
    roots = REPO_ROOT / "data" / "roots" / "h3n2_13lineage"
    for name in ("J", "J.2", "J.2.4", "K"):
        path = roots / f"{name}.nt.fa"
        if not path.exists():
            pytest.skip(f"{path} not shipped")
        protein = _translate_root(path)
        assert protein[160] == "S", (
            f"{name} has {protein[160]} at precursor 161 (mature H3 145), expected S. "
            f"If a root has genuinely changed, LINEAGE_DEFINITIONS.md must be revised "
            f"and every downstream table re-run, not re-plotted."
        )
