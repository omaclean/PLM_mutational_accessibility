"""Tests for the masked-marginal estimator check (figures/make_sO_masked_marginals.py).

The figure's whole claim is "masking is the only thing that changed". Most of what is worth
testing is therefore about the *tables* holding that invariant -- same arms, same alpha grid,
same n -- rather than about the drawing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "figures"))

import make_sO_masked_marginals as mm  # noqa: E402
from mutacc.guide import load_guide  # noqa: E402
from mutacc.numbering import load_numbering  # noqa: E402

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
OUTPUT = REPO_ROOT / "output"


@pytest.fixture(scope="module")
def curves():
    return mm.load_curves(GUIDE, GUIDE.results_dir)


@pytest.fixture(scope="module")
def strata():
    return mm.load_strata(GUIDE, GUIDE.results_dir)


# --- the invariant the figure rests on ------------------------------------------------------

def test_both_estimators_cover_the_same_arms_on_the_same_alpha_grid(curves):
    """A missing arm or a shifted grid would read as an effect of masking."""
    grids = {}
    for (panel, model, scoring), g in curves.groupby(["panel_label", "model_label", "scoring"]):
        grids[(panel, model, scoring)] = tuple(np.round(sorted(g["alpha"]), 6))
    panels = {(p, m) for p, m, _ in grids}
    for panel, model in panels:
        a = grids[(panel, model, "published")]
        b = grids[(panel, model, "masked")]
        assert a == b, f"{panel}/{model}: the two estimators are on different alpha grids"
    assert len(set(grids.values())) == 1, "not every arm shares one alpha grid"


def test_masking_changes_the_scores_but_not_the_rows_behind_them(curves):
    """Same replacements, same sites -- only the probability column differs."""
    keys = ["panel_label", "model_label"]
    counts = curves.groupby(keys + ["scoring"])[["n", "n_sites"]].first().reset_index()
    wide = counts.pivot_table(index=keys, columns="scoring", values=["n", "n_sites"])
    assert (wide[("n", "masked")] == wide[("n", "published")]).all()
    assert (wide[("n_sites", "masked")] == wide[("n_sites", "published")]).all()

    # ...and it must actually have changed something, or the table is the published one twice
    peaks = curves.groupby(keys + ["scoring"])["rho_best"].first().reset_index()
    pw = peaks.pivot_table(index=keys, columns="scoring", values="rho_best")
    assert (pw["masked"] != pw["published"]).any(), "masked and published curves are identical"


def test_every_influenza_panel_of_figure_four_is_present(curves):
    """The figure is the masked version of Figure 4; a dropped panel would be a silent gap."""
    summary = GUIDE.read_table("dms_peak_summary", GUIDE.results_dir)
    expected = set(summary.loc[summary["virus_tag"] == "IAV", "panel_label"])
    assert set(curves["panel_label"]) == expected
    assert len(expected) == 4


def test_the_sweep_is_the_published_one_rerun_not_a_reimplementation():
    """The builder must drive make_fig04.compute_curves, not its own copy of the sweep."""
    source = (REPO_ROOT / "dev" / "masked_dms" / "build_masked_tables.py").read_text()
    assert "fig4.compute_curves(" in source
    assert "sweep_curve(" not in source, "the builder re-implements the sweep instead of reusing it"


# --- panel E: no Grantham ------------------------------------------------------------------

def test_panel_e_drops_every_control(strata):
    """Figure 2D's Grantham arrows ask a different question; this panel asks only about
    the estimator, so nothing controlled may reach it."""
    assert strata["control"].isna().all()
    assert set(strata["measure"]) == {mm.STRATA_MEASURE}
    assert "neg_grantham" not in set(strata["series"])
    assert "blosum62" not in set(strata["series"])


def test_panel_e_covers_all_and_each_codon_distance(strata):
    assert set(strata["stratum"].astype(str)) == {"all", "1", "2", "3"}
    assert set(strata["role"]) == {"base", "finetune"}


def test_panel_e_carries_an_interval_for_both_estimators(strata):
    for kind in ("unmasked", "masked"):
        lo, est, hi = (strata[f"lo_{kind}"], strata[f"estimate_{kind}"], strata[f"hi_{kind}"])
        assert (lo <= est).all() and (est <= hi).all(), f"{kind}: estimate outside its interval"


def test_the_source_table_still_holds_the_controls_even_though_the_panel_drops_them():
    """Dropping them from the figure must not mean throwing the run away."""
    table = GUIDE.read_table(mm.STRATA_TABLE, GUIDE.results_dir)
    assert set(table["control"].dropna()) == {"neg_grantham", "blosum62"}


# --- drawing and the shipped output ---------------------------------------------------------

def test_a_table_missing_one_estimator_is_refused(curves):
    thin = curves[~((curves["panel_label"] == curves["panel_label"].iloc[0])
                    & (curves["scoring"] == "masked"))]
    with pytest.raises(ValueError, match="both scorings"):
        mm.load_curves(_FakeGuide(thin), Path("."))


class _FakeGuide:
    def __init__(self, frame):
        self._frame = frame

    def read_table(self, *_args, **_kwargs):
        return self._frame


def test_the_figure_draws_one_axes_per_panel_plus_the_strata_panel(curves, strata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = mm.draw(curves, strata)
    try:
        assert len(fig.axes) == len(mm.panel_order(curves)) + 1
        # panel_title sets the title at loc="left", which matplotlib stores apart from the
        # centre one; get_title() with no argument would return "" for every panel.
        titles = [ax.get_title(loc="left") for ax in fig.axes]
        assert any("codon distance" in t for t in titles), titles
    finally:
        plt.close(fig)


def test_the_strata_panel_can_be_left_out(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = mm.draw(curves, None)
    try:
        assert len(fig.axes) == len(mm.panel_order(curves))
    finally:
        plt.close(fig)


def test_the_figure_has_a_manuscript_number():
    assert load_numbering().label_for(mm.STEM) == "Figure SO"


def test_the_shipped_caption_quotes_only_numbers_the_table_supports(curves, strata):
    caption = (OUTPUT / f"{mm.STEM}_caption.md").read_text()
    d = mm.deltas(curves).set_index(["panel_label", "model_role"])
    ft = d.xs("finetune", level="model_role")["d_rho_best"]
    assert f"{int((ft > 0).sum())} of {len(ft)}" in caption
    assert "Grantham" in caption, "the caption must say the control was deliberately dropped"


def test_the_shipped_values_table_has_both_panel_groups():
    values = pd.read_csv(OUTPUT / f"{mm.STEM}_values.tsv", sep="\t")
    assert set(values["panel"]) == {"A-D", "E"}
