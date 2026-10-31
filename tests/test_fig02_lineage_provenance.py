"""Which lineage each panel of Figure 2 is drawn from.

Figure 2 is a composite of four panels with three different provenances, and only
one of them is built from code in this repository:

===== ===================================== =========================
Panel Source                                Lineage
===== ===================================== =========================
2A    ``figures/make_fig02ab.py``           the guide's primary_lineage
2B    ``figures/make_fig02ab.py``           the guide's primary_lineage
2C    ``figures/make_fig02c.py``            three lineages, not one (from digitised values)
2D    ``figures/make_fig02d.py``            the guide's primary_lineage
===== ===================================== =========================

So "Figure 2 is drawn from one lineage" is true of 2A, 2B and 2D and false for
2C, which redraws a published three-lineage mean whose table no longer exists. 2A and 2B were frozen PDFs until 16 Oct 2026; the frozen files are kept, and
the tests below show they were drawn from all five lineages pooled (which is why
they were five-fold overplotted). These tests pin exactly that much and no more.
The point is to stop the 2C gap being closed by writing a lineage into a caption:
that panel carries no such claim, and asserting one in prose would be a provenance
claim the file does not support.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
import yaml

from mutacc.guide import REPO_ROOT, load_guide

GUIDE_5 = load_guide(REPO_ROOT / "guides" / "h3n2_5lineage.yaml")
LAYOUT = yaml.safe_load((REPO_ROOT / "guides" / "composites" / "fig02.yaml").read_text())
PANELS = {p["key"]: p for p in LAYOUT["panels"]}
FROZEN = REPO_ROOT / "data" / "frozen_panels"

#: The Spearman correlations printed inside the frozen scatter panels these replace, and the
#: files they are in. The panels are no longer placed by the layout; the files are kept.
FROZEN_RHO = {"2A": 0.274, "2B": 0.652}
FROZEN_FILES = {"2A": "fig2a_scatter_esm2_raw.pdf", "2B": "fig2b_scatter_esm2_ha80.pdf"}
MODEL_ROLE = {"2A": "base", "2B": "finetune"}


# ---------------------------------------------------------------------------
# 2D: the one panel whose lineage is determined by code
# ---------------------------------------------------------------------------

def test_the_five_lineage_guide_pins_the_primary_lineage_to_j2_int():
    """2D follows ``primary_lineage``, so the guide is where the choice lives.

    Unlike the thirteen-lineage set -- where nothing is declared and the choice
    falls to an alphabetical tie-break (tests/test_known_defects.py) -- this
    guide names its lineage outright.
    """
    assert GUIDE_5.primary_lineage == "J.2_int"
    assert "J.2_int" in {lin.id for lin in GUIDE_5.lineages}


def test_the_shipped_panel_d_values_are_that_one_lineage():
    frame = pd.read_csv(REPO_ROOT / "output" / "fig02d_strata_panel_values.tsv", sep="\t")
    assert set(frame["lineage"].astype(str)) == {"J.2_int"}, (
        "panel 2D was drawn from a different lineage than the guide declares"
    )


# ---------------------------------------------------------------------------
# 2A and 2B: frozen, and consistent with J.2_int without identifying it
# ---------------------------------------------------------------------------

def _frozen_text(filename):
    import matplotlib            # noqa: F401  -- must import before fitz, or Pillow dies
    fitz = pytest.importorskip("fitz")
    with fitz.open(FROZEN / filename) as doc:
        return " ".join(page.get_text() for page in doc)


@pytest.mark.parametrize("key", ["2A", "2B"])
def test_frozen_scatter_still_prints_the_correlation_it_was_published_with(key):
    """The frozen panels are images, so their numbers can only be read back out."""
    text = _frozen_text(FROZEN_FILES[key])
    found = re.search(r"Spearman[^=]*=\s*(-?\d+\.\d+)", text)
    assert found, f"{key}: no Spearman value in the panel text"
    assert float(found.group(1)) == FROZEN_RHO[key]


@pytest.mark.parametrize("key", ["2A", "2B"])
def test_the_frozen_scatter_was_drawn_from_all_lineages_pooled(key):
    """The printed correlation is the one over every lineage stacked together.

    Pooled, it reproduces the printed value to the three decimals it was printed
    with; no single lineage does for the fine-tuned model (they run 0.634 to
    0.664). Stacking the five lineages puts five near-copies of each replacement
    in the cloud, which is what the redraw (make_fig02ab.py) removes.
    """
    scores = GUIDE_5.read_table("replacement_scores", GUIDE_5.results_dir)
    model = next(m.key for m in GUIDE_5.models if m.role == MODEL_ROLE[key])
    rows = scores[scores["model"] == model]
    assert rows["lineage"].nunique() == len(GUIDE_5.lineages)
    pooled = rows["plm_prob"].corr(rows["mut_prob"], method="spearman")
    assert round(pooled, 3) == FROZEN_RHO[key], f"{key}: pooled rho is {pooled:.4f}"
    primary = rows[rows["lineage"].astype(str) == GUIDE_5.primary_lineage]
    assert len(primary) * len(GUIDE_5.lineages) == len(rows), "lineages are scored at the same sites"


@pytest.mark.parametrize("key", ["2A", "2B"])
def test_the_redrawn_panels_are_the_one_primary_lineage(key):
    frame = pd.read_csv(REPO_ROOT / "output" / "fig02ab_scatter_panels_values.tsv", sep="\t")
    sub = frame[frame["panel"] == key[1]]
    assert set(sub["lineage"].astype(str)) == {GUIDE_5.primary_lineage}
    n = sub[sub["measure"] == "n_replacements"]["value"].sum()
    scores = GUIDE_5.read_table("replacement_scores", GUIDE_5.results_dir)
    one = scores[(scores["lineage"].astype(str) == GUIDE_5.primary_lineage)
                 & (scores["model"] == next(m.key for m in GUIDE_5.models if m.role == MODEL_ROLE[key]))]
    assert n == len(one), "the panel must plot every replacement of that one lineage"


def test_the_frozen_panels_do_not_claim_a_lineage():
    """If a lineage name ever appears in these panels, this file's premise changes
    and the caption may finally state one."""
    for key in ("2A", "2B"):
        text = _frozen_text(FROZEN_FILES[key])
        for lineage in (lin.id for lin in GUIDE_5.lineages):
            assert lineage not in text, f"{key} names lineage {lineage}; update this module"


# ---------------------------------------------------------------------------
# 2C: the panel that is not one lineage, and cannot be made into one
# ---------------------------------------------------------------------------

def test_panel_c_says_where_its_numbers_and_lineages_come_from():
    """2C's numbers were read back off the published raster and are a three-lineage mean.

    The subheader no longer says so (the lineages are pseudo-replicates, so the user dropped
    it), which moves the statement to the caption and the values table, where it must stay:
    a panel drawn from three lineages beside three drawn from one is a provenance fact.
    """
    caption = " ".join((REPO_ROOT / "output" / "fig02c_epoch_panel_caption.md").read_text().split())
    assert "means over three lineages" in caption and "pseudo-replicates" in caption
    assert "read back from the published version" in caption
    values = pd.read_csv(REPO_ROOT / "output" / "fig02c_epoch_panel_values.tsv", sep="\t")
    assert values["source"].str.startswith("digitised from").all()
    readme = (FROZEN / "README.md").read_text()
    assert "Uses 3 lineages" in readme and "No generator anywhere" in readme


def test_panel_c_subheader_no_longer_mentions_lineages():
    assert "lineage" not in PANELS["2C"]["tagline"].lower()


def test_no_panel_tagline_names_a_lineage():
    """The layout's own rule: the lineage comes from the guide, and repeating it
    in a tagline would let the two drift apart silently."""
    for key, panel in PANELS.items():
        text = f"{panel.get('title', '')} {panel.get('tagline', '')}"
        for lineage in (lin.id for lin in GUIDE_5.lineages):
            assert lineage not in text, f"{key} names {lineage} in its header"
