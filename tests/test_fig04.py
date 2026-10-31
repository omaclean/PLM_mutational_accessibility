"""Figure 4: DMS fitness against the alpha-weighted score.

Numbers are checked three ways: against an independent scipy calculation, against the
archive's own tables (shipped beside the scores), and against golden values read from that
archive. The self-check against ``dms_alpha_curves`` is the guard the source plotter carried
and is kept as a test: a recomputed sweep that drifts from the archive's must fail.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from mutacc.io import caption_label
from scipy.stats import spearmanr

import make_fig04 as fig4
from mutacc import style
from mutacc.dms_sweep import accessibility_table, log_accessibility
from mutacc.guide import REPO_ROOT, load_guide

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
SCRIPT = REPO_ROOT / "figures" / "make_fig04.py"


@pytest.fixture(scope="module")
def scores():
    return GUIDE.read_table("dms_scores")


@pytest.fixture(scope="module")
def summary():
    return GUIDE.read_table("dms_peak_summary")


@pytest.fixture(scope="module")
def reference():
    return GUIDE.read_table("dms_alpha_curves")


@pytest.fixture(scope="module")
def curves(scores, summary):
    """Every arm, no bootstrap: the sweep itself (about 6 s)."""
    return fig4.compute_curves(scores, summary, n_boot=0)


def arm(curves, dataset, trait, model_name):
    sub = curves[(curves.dataset == dataset) & (curves.trait == trait) & (curves.model_name == model_name)]
    assert len(sub) == len(fig4.alpha_grid()), f"{dataset}/{trait}/{model_name}"
    return sub.sort_values("alpha")


# ---------------------------------------------------------------------------
# The self-check against the archive's own curves
# ---------------------------------------------------------------------------

def test_recomputed_curves_reproduce_the_archive_to_1e6(curves, reference):
    worst = fig4.self_check(curves, reference)
    assert worst < 1e-6
    assert curves["model_name"].nunique() == reference["model"].nunique()


def test_self_check_fails_when_the_reference_drifts(curves, reference):
    drifted = reference.copy()
    drifted.loc[drifted.index[100], "rho"] += 1e-5
    with pytest.raises(fig4.SelfCheckError, match="differs from the archive"):
        fig4.self_check(curves, drifted)


def test_self_check_fails_when_a_curve_is_missing_from_the_reference(curves, reference):
    first = curves.iloc[0]
    gap = reference[~((reference.dataset == first.dataset) & (reference.model == first.model_name)
                      & (reference.trait == first.trait))]
    with pytest.raises(fig4.SelfCheckError, match="absent"):
        fig4.self_check(curves, gap)


def test_self_check_fails_when_the_scores_are_corrupted(scores, summary, reference):
    """Swapping the accessibility column changes the sweep; the check must notice."""
    bad = scores.copy()
    bad["log_acc"] = np.random.default_rng(0).permutation(bad["log_acc"].to_numpy())
    changed = fig4.compute_curves(bad, summary, n_boot=0, max_arms=2)
    with pytest.raises(fig4.SelfCheckError):
        fig4.self_check(changed, reference)


# ---------------------------------------------------------------------------
# Golden values (read from the archive's a31_alpha table) and independent calculation
# ---------------------------------------------------------------------------

GOLDEN = [
    # dataset, trait, model, rho at alpha 0, best alpha, rho at best
    ("MA22", "MDCKSIAT1 cell entry", "ESM-2 650M base", 0.41720, -0.05, 0.41944),
    ("MA22", "MDCKSIAT1 cell entry", "HA-80 fine-tuned", 0.28973, -0.17, 0.32325),
    ("MA22", "sera escape", "HA-80 fine-tuned", 0.12028, -0.21, 0.14594),
    ("Perth09", "preference ratio", "ESM-2 650M base", 0.45898, -0.04, 0.46030),
    ("BA2", "spike mediated entry", "SC2 fine-tuned", 0.28623, -0.19, 0.35071),
    ("XBB15", "spike mediated entry", "ESM-C 600M base", 0.10386, 0.01, 0.10503),
]


@pytest.mark.parametrize("dataset,trait,model,rho0,best,rho_best", GOLDEN)
def test_golden_peak_values(curves, dataset, trait, model, rho0, best, rho_best):
    sub = arm(curves, dataset, trait, model)
    assert sub["rho_alpha0"].iloc[0] == pytest.approx(rho0, abs=5e-6)
    assert sub["best_alpha"].iloc[0] == pytest.approx(best, abs=1e-9)
    assert sub["rho_best"].iloc[0] == pytest.approx(rho_best, abs=5e-6)


@pytest.mark.parametrize("alpha", [-1.0, -0.3, 0.0, 0.5, 1.0])
def test_sweep_equals_scipy_spearman(scores, curves, alpha):
    """An independent calculation on one arm: plain scipy on the raw columns."""
    dataset, trait, model = "MA22", "pH stability", "HA-80 fine-tuned"
    sub = scores[(scores.dataset == dataset) & (scores.model_name == model)]
    sub = sub[sub[trait].notna()]
    expected = spearmanr(sub["log_plm"] + alpha * sub["log_acc"], sub[trait]).statistic
    got = arm(curves, dataset, trait, model).set_index("alpha")["rho"].loc[alpha]
    assert got == pytest.approx(expected, abs=1e-9)


def test_replacement_counts_match_the_shipped_summary(curves, summary):
    pub = summary[summary.spectrum == "published"]
    for r in pub.itertuples():
        sub = arm(curves, r.dataset, r.trait, r.model_name)
        # The summary counts replacements with a finite phenotype after a31's own dropna.
        assert sub["n"].iloc[0] == r.n_variants, (r.dataset, r.trait, r.model_name)


# ---------------------------------------------------------------------------
# Flags: optima are marked only where the archive says they are interpretable
# ---------------------------------------------------------------------------

def test_recomputed_flags_match_the_shipped_summary(curves):
    fig4.check_flags(curves)


def test_binding_arms_run_negative_and_are_read_in_size(curves):
    """ACE2-type receptor-binding arms run negative throughout; the optimum is the largest |rho|.

    Under the archive's largest-rho rule every one of these was unreadable, because maximising a
    negative curve walks to whichever end dilutes hardest. Reading them in size has to recover an
    optimum strictly inside the swept range wherever one is marked at all.
    """
    arms = fig4._arm_table(curves)
    binding = arms[arms["trait"].str.contains(fig4.RECEPTOR_TRAIT, case=False)]
    assert len(binding) >= 2
    assert (binding["rho_alpha0"] < 0).all()
    assert (binding["rho_best"] < 0).all()                      # the sign of the curve is kept
    assert (binding["rho_best"].abs() >= binding["rho_alpha0"].abs()).all()
    marked = binding[binding["interpretable"]]
    assert len(marked) >= 1, "reading in size must recover at least one receptor-binding optimum"
    assert (marked["best_alpha"].abs() < 1.0).all()             # interior, not at an end


def test_flag_mismatch_raises(curves):
    broken = curves.copy()
    broken["interpretable_in_archive"] = ~broken["interpretable_in_archive"]
    with pytest.raises(fig4.SelfCheckError, match="interpretable flag"):
        fig4.check_flags(broken)


# ---------------------------------------------------------------------------
# The bootstrap band reproduces the published one (first arm: generator order matters)
# ---------------------------------------------------------------------------

def test_band_reproduces_the_published_band_for_the_first_arm(scores, summary):
    band = GUIDE.read_table("dms_alpha_bands_published")
    got = fig4.compute_curves(scores, summary, max_arms=1)          # n_boot = 200, seed = the archive's
    first = got.iloc[0]
    ref = band[(band.dataset == first.dataset) & (band.model_name == first.model_name)
               & (band.trait == first.trait)].set_index("alpha").reindex(np.round(got["alpha"], 6))
    # the shipped band is written to 6 decimals, so half a unit in the last place is the floor
    assert np.abs(got["ci_lo"].to_numpy() - ref["ci_lo"].to_numpy()).max() < 1e-6
    assert np.abs(got["ci_hi"].to_numpy() - ref["ci_hi"].to_numpy()).max() < 1e-6
    assert (got["ci_lo"] <= got["rho"] + 1e-9).all() and (got["rho"] <= got["ci_hi"] + 1e-9).all()


def test_band_is_deterministic_and_seed_dependent(scores, summary):
    a = fig4.compute_curves(scores, summary, n_boot=8, max_arms=1)
    b = fig4.compute_curves(scores, summary, n_boot=8, max_arms=1)
    c = fig4.compute_curves(scores, summary, n_boot=8, max_arms=1, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["ci_lo"], c["ci_lo"])


# ---------------------------------------------------------------------------
# Reversible variant
# ---------------------------------------------------------------------------

def _spectra(scores):
    return fig4.load_spectra(GUIDE, GUIDE.results_dir, scores)


def test_published_accessibility_is_reproduced_from_the_shipped_matrices(scores):
    """Validates the codon/accessibility port: the non-reversible table equals the archive's log_acc."""
    spectra = _spectra(scores)
    for name, g in scores.groupby("spectrum_matrix"):
        table = accessibility_table(spectra[name], reversible=False)
        assert np.abs(log_accessibility(g["ref_codon"], g["aa"], table) - g["log_acc"]).max() < 1e-9


# ---------------------------------------------------------------------------
# Bad input raises; it never draws a blank panel
# ---------------------------------------------------------------------------

def test_empty_scores_raise(scores, summary):
    with pytest.raises(ValueError, match="empty"):
        fig4.compute_curves(scores.iloc[0:0], summary, n_boot=0)


def test_empty_summary_raises(scores, summary):
    with pytest.raises(ValueError, match="empty"):
        fig4.compute_curves(scores, summary.iloc[0:0], n_boot=0)


@pytest.mark.parametrize("column", ["log_plm", "log_acc"])
def test_nan_in_the_score_columns_raises(scores, summary, column):
    broken = scores.copy()
    broken.loc[broken.index[5], column] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fig4.compute_curves(broken, summary, n_boot=0, max_arms=1)


def test_all_nan_trait_raises_not_a_blank_panel(scores, summary):
    first = fig4.published_arms(summary).iloc[0]
    broken = scores.copy()
    broken[first.trait] = np.nan
    with pytest.raises(ValueError, match="scored replacements"):
        fig4.compute_curves(broken, summary, n_boot=0, max_arms=1)


def test_a_model_name_that_never_joins_raises(scores, summary):
    broken = summary.copy()
    broken.loc[broken.index[0], "model_name"] = "no such model"
    with pytest.raises(ValueError, match="scored replacements"):
        fig4.compute_curves(scores, broken, n_boot=0, max_arms=1)


def test_missing_column_and_unknown_role_raise(scores, summary):
    with pytest.raises(ValueError, match="lacks column"):
        fig4.compute_curves(scores.drop(columns=["site"]), summary, n_boot=0)
    broken = summary.copy()
    broken["model_role"] = "mystery"
    with pytest.raises(ValueError, match="unknown model role"):
        fig4.compute_curves(scores, broken, n_boot=0)


def test_duplicate_summary_rows_raise(scores, summary):
    with pytest.raises(ValueError, match="duplicate"):
        fig4.compute_curves(scores, pd.concat([summary, summary]), n_boot=0)


def test_drawing_nothing_raises():
    with pytest.raises(ValueError, match="no curves"):
        fig4.draw(pd.DataFrame())


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def split(curves):
    """What each figure actually draws: the panels of one virus."""
    return dict(fig4.split_by_virus(curves))


@pytest.fixture(scope="module")
def drawn(split):
    return {virus: fig4.draw(sub) for virus, sub in split.items()}


def test_the_split_keeps_every_panel_and_puts_influenza_first(curves, split):
    assert list(split) == [fig4.MAIN_VIRUS, "SC2"]
    assert sum(len(sub) for sub in split.values()) == len(curves)
    assert all(sub["virus_tag"].nunique() == 1 for sub in split.values())


def test_each_figure_letters_its_own_panels_from_a(drawn, split):
    for virus, fig in drawn.items():
        axes = [ax for ax in fig.axes if ax.get_visible() and ax.lines]
        assert len(axes) == len(fig4.panel_order(split[virus])), virus
        letters = [t.get_text() for ax in axes for t in ax.texts if t.get_fontweight() == "bold"]
        assert letters == [chr(ord("A") + i) for i in range(len(axes))], virus


def test_no_blank_cell_is_left_in_a_grid():
    assert fig4.column_count(4) == 2                  # 2x2, not 3 + 1
    assert fig4.column_count(6) == 3
    assert fig4.column_count(1) == 1


def test_every_panel_has_a_labelled_zero_line_and_no_label_collides(drawn):
    for virus, fig in drawn.items():
        axes = [ax for ax in fig.axes if ax.get_visible() and ax.lines]
        for ax in axes:
            assert fig4.ZERO_LABEL in [t.get_text() for t in ax.texts]
        assert max(fig.label_scores) < 1000, f"two labels overlap in the {virus} figure"


def test_series_are_coloured_by_role_only(drawn):
    allowed = {style.BASE, style.FINETUNE, style.NULL_GREY, style.INK_MUTED}
    for fig in drawn.values():
        for ax in fig.axes:
            for line in ax.lines:
                assert line.get_color() in allowed, line.get_color()


def test_alpha_is_never_shown_bare(drawn):
    for fig in drawn.values():
        texts = [t.get_text() for t in fig.texts]
        assert any("Weight on mutational accessibility" in t for t in texts), texts
        legend = fig.legends[0]
        assert len(legend.get_texts()) >= 3
        assert "Base model" in [t.get_text() for t in legend.get_texts()]


def test_unmarked_optima_say_so_on_the_panel(drawn, split):
    """Every unmarked curve is a SARS-CoV-2 one, so the supplementary figure carries them all.

    A curve with no filled optimum says so one of two ways: "no optimum marked", or, when it
    has a real interior optimum that is simply below the signal floor, that weight with the
    caveat beside it. Between them they must account for every non-interpretable curve.
    """
    for virus, fig in drawn.items():
        arms = fig4._arm_table(split[virus])
        n_unmarked = int((~arms["interpretable"]).sum())
        found = sum(t.get_text().count("no optimum marked") + t.get_text().count("below signal floor")
                    for ax in fig.axes for t in ax.texts)
        assert found == n_unmarked, virus
    assert int((~fig4._arm_table(split["SC2"])["interpretable"]).sum()) > 0


def test_main_writes_both_figures_with_values_and_captions(tmp_path, curves, monkeypatch):
    monkeypatch.setattr(fig4, "compute_curves", lambda *a, **k: curves)   # the sweep is tested above
    rc = fig4.main(["--guide", str(GUIDE.source), "--out", str(tmp_path)])
    assert rc == 0
    for virus, stem in ((fig4.MAIN_VIRUS, fig4.STEM), ("SC2", fig4.STEM_SC2)):
        for ext in ("pdf", "png"):
            assert (tmp_path / f"{stem}.{ext}").stat().st_size > 20_000, stem
        values = pd.read_csv(tmp_path / f"{stem}_values.tsv", sep="\t")
        assert set(values["virus_tag"]) == {virus}
        assert values["rho"].notna().all()
        text = " ".join((tmp_path / f"{stem}_caption.md").read_text().split())
        for phrase in ("deep mutational scanning (DMS)", "protein language model (PLM)",
                       "cluster bootstrap over DMS sites", "optimistic by an unknown amount"):
            assert phrase.lower() in text.lower(), (stem, phrase)
    total = sum(len(pd.read_csv(tmp_path / f"{s}_values.tsv", sep="\t"))
                for s in (fig4.STEM, fig4.STEM_SC2))
    assert total == curves.shape[0]                  # the split loses nothing


def test_each_figure_names_itself_in_its_caption(tmp_path, curves, monkeypatch):
    monkeypatch.setattr(fig4, "compute_curves", lambda *a, **k: curves)
    fig4.main(["--guide", str(GUIDE.source), "--out", str(tmp_path)])
    # The numbers are the map's, not this script's: the hard-coded
    # FIGURE_LABELS = {STEM: "Figure 4", STEM_SC2: "Figure S18"} still said S18
    # after the draft moved the SARS-CoV-2 figure, which is Figure SL.
    assert f"**{caption_label(fig4.STEM)}." in (tmp_path / f"{fig4.STEM}_caption.md").read_text()
    assert f"**{caption_label(fig4.STEM_SC2)}." in (tmp_path / f"{fig4.STEM_SC2}_caption.md").read_text()
    assert caption_label(fig4.STEM_SC2) == "Figure SL"


def test_caption_counts_come_from_the_data(split):
    for virus, sub in split.items():
        text = " ".join(fig4.caption(sub, n_boot=200).split())
        arms = fig4._arm_table(sub)
        n_flag = int((~arms["interpretable"]).sum())
        assert f"No optimum is marked on {n_flag} of {len(arms)} curves" in text, virus
        assert "(200 resamples)" in text


def test_only_the_sars_cov_2_caption_explains_the_binding_panels(split):
    """The receptor-binding panels are all SARS-CoV-2, and so is the sentence about them."""
    sc2 = " ".join(fig4.caption(split["SC2"], n_boot=200).split())
    iav = " ".join(fig4.caption(split[fig4.MAIN_VIRUS], n_boot=200).split())
    assert "Panels D and E measure receptor binding" in sc2
    assert "rather than a model failure" in sc2
    assert "receptor binding" not in iav


# ---------------------------------------------------------------------------
# House rules specific to this script (test_figure_rules.py covers the general ones)
# ---------------------------------------------------------------------------

def test_script_does_not_name_datasets_traits_or_models(scores, summary):
    source = SCRIPT.read_text()
    names = set(scores["dataset"]) | set(scores["model_name"]) | set(summary["trait"]) | set(summary["model_label"])
    names |= set(summary["organism"]) | set(summary["citation"])
    hits = [n for n in names if re.search(r'["\']' + re.escape(n) + r'["\']', source)]
    assert not hits, f"table values typed into the script: {hits}"


def test_script_takes_its_sweep_from_the_library_not_its_own_arithmetic():
    source = SCRIPT.read_text()
    assert "spearmanr" not in source and "rankdata" not in source
