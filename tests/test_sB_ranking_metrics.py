"""S2: the six ranking metrics, the frequency table panel F rests on, and the exporter that builds it."""

import numpy as np
import pandas as pd
import pytest

import make_sB_ranking_metrics as sB
from mutacc.guide import REPO_ROOT, load_guide
from mutacc.metrics import mcclish_floor
from tools import export_sB_frequency as exporter

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
KEY = sB.KEY


@pytest.fixture(scope="module")
def frame():
    return sB.load_scores(GUIDE, GUIDE.results_dir)


@pytest.fixture(scope="module")
def alphas():
    return sB.alpha_grid(GUIDE, GUIDE.results_dir)


@pytest.fixture(scope="module")
def recomputed(frame, alphas):
    return sB.metric_grid(frame, alphas, sB.FPR_CAP)


@pytest.fixture(scope="module")
def shipped():
    return sB.sweep_grid(GUIDE, GUIDE.results_dir)


@pytest.fixture(scope="module")
def grid(recomputed, shipped):
    return sB.join_grids(recomputed, shipped)


@pytest.fixture(scope="module")
def base(frame):
    return sB.baseline_grid(frame, GUIDE, GUIDE.results_dir, sB.FPR_CAP)


# ---------------------------------------------------------------------------
# The sweep grid
# ---------------------------------------------------------------------------

def test_alpha_grid_is_the_sweeps_own_101_points(alphas):
    assert len(alphas) == 101 and alphas[0] == -1.0 and alphas[-1] == 1.0
    assert np.isfinite(alphas).all() and np.allclose(np.diff(alphas), 0.02)


def test_every_scored_replacement_has_an_observed_flag(frame):
    assert frame["obs_present"].isin([0, 1]).all()
    assert len(frame) == 279_604


def test_recomputed_auroc_equals_the_shipped_sweep_table(recomputed):
    gap = sB.check_against_sweep(recomputed, GUIDE, GUIDE.results_dir)
    assert gap < 1e-6


def test_a_tampered_sweep_table_is_caught(recomputed, monkeypatch):
    real = GUIDE.read_table(sB.sweep_table_name())
    bad = real.copy()
    bad.loc[bad.index[5], "site_logistic_auroc"] += 0.01
    monkeypatch.setattr(type(GUIDE), "read_table", lambda self, name, results=None, **kw: bad)
    with pytest.raises(ValueError, match="disagrees"):
        sB.check_against_sweep(recomputed, GUIDE, GUIDE.results_dir)


def test_golden_auroc_at_alpha_zero_matches_the_audit(recomputed):
    """Under masked marginals: 0.9284 fine-tuned, 0.7894 base.

    Repinned 23 Oct 2026 for the lineage reassignment (was 0.9294 / 0.7927; the audit's
    unmasked pair was 0.9195 / 0.7721). The lineages are the same thirteen, delimited by
    the published rule rather than by nearest root."""
    at0 = recomputed[np.isclose(recomputed.alpha, 0.0)].groupby("model")["auroc"].mean()
    by_role = {m.role: m.key for m in GUIDE.models}
    assert at0[by_role["finetune"]] == pytest.approx(0.9284, abs=5e-4)
    assert at0[by_role["base"]] == pytest.approx(0.7894, abs=5e-4)


def test_prevalence_is_a_property_of_the_lineage_not_the_model(recomputed):
    wide = recomputed.pivot_table(index=["lineage", "alpha"], columns="model", values="prevalence")
    assert np.allclose(wide.iloc[:, 0], wide.iloc[:, 1])
    per = recomputed.groupby("lineage")["prevalence"].first()
    assert 0.015 < per.min() < 0.03 and 0.10 < per.max() < 0.12     # 2.0% to 10.9% (PLAN risk register)


def test_fold_enrichment_is_pr_auc_over_prevalence(recomputed):
    assert np.allclose(recomputed["prauc_fold"], recomputed["pr_auc"] / recomputed["prevalence"])


def test_partial_auroc_cannot_fall_below_the_mcclish_floor(recomputed):
    assert recomputed["pauroc_std"].min() >= mcclish_floor(sB.FPR_CAP) - 1e-9
    assert recomputed["pauroc_std"].max() <= 1.0 + 1e-9


def test_a_perfect_ranking_beats_chance_on_every_metric(recomputed):
    best = recomputed[np.isclose(recomputed.alpha, 1.0)]
    assert (best["prauc_fold"] > 1).all() and (best["pauroc_std"] > 0.5).all() and (best["auroc"] > 0.5).all()


def test_the_two_routes_cover_exactly_the_same_cells(grid, recomputed):
    assert len(grid) == len(recomputed)
    assert grid[[m for m, *_ in sB.PANELS]].notna().all().all()


def test_a_sweep_grid_missing_a_cell_is_an_error_not_a_hole(recomputed, shipped):
    with pytest.raises(ValueError, match="only one of the two routes"):
        sB.join_grids(recomputed, shipped.iloc[1:])


# ---------------------------------------------------------------------------
# Panels D and E: read from the sweep table, rebuilt from the scores to check
# ---------------------------------------------------------------------------

def test_panels_d_and_e_are_the_shipped_columns(grid):
    sweep = GUIDE.read_table(sB.sweep_table_name())
    sweep = sweep.loc[sweep["model_variant"] == sB.SWEEP_VARIANT].copy()
    sweep["alpha"] = sweep["alpha"].round(10)
    mine = grid.copy()
    mine["alpha"] = mine["alpha"].round(10)
    both = mine.merge(sweep, on=["model", "lineage", "alpha"], how="inner")
    assert len(both) == len(grid)
    for metric, column in sB.SWEEP_METRICS.items():
        assert np.array_equal(both[metric].to_numpy(), both[column].to_numpy())


def test_panels_d_and_e_rebuild_from_the_scores_and_frequencies(frame, shipped, alphas):
    """The figure reads them off the shelf; this is what entitles it to."""
    gaps = sB.check_sweep_metrics(frame, shipped, alphas)
    assert set(gaps) == set(sB.SWEEP_METRICS)
    for metric, (worst, bulk) in gaps.items():
        every_tol, bulk_tol = sB.SWEEP_METRIC_TOL[metric]
        assert worst <= every_tol and bulk <= bulk_tol


def test_a_rank_correlation_is_allowed_tie_flips_and_nothing_more():
    """``SWEEP_METRIC_TOL`` gives the rank correlation a looser per-cell floor than the
    linear one. That is for ties moving under the shipped table's rounding, not for a
    column that is a different quantity, so the bulk of cells is still held tight."""
    every_rank, bulk_rank = sB.SWEEP_METRIC_TOL["nonzero_spearman"]
    every_linear, _ = sB.SWEEP_METRIC_TOL["logfreq_pearson"]
    assert every_rank > every_linear
    assert bulk_rank == sB.SWEEP_TOL


def test_a_wrong_correlation_column_is_caught_even_inside_the_rank_floor(frame, shipped, alphas):
    """A shifted column stays under the per-cell floor but fails on the bulk of cells."""
    nudged = shipped.copy()
    nudged["nonzero_spearman"] = nudged["nonzero_spearman"] + 1e-4
    with pytest.raises(ValueError, match="different quantity, not rounding"):
        sB.check_sweep_metrics(frame, nudged, alphas)


def test_a_sweep_table_without_the_review_columns_refuses_to_draw(monkeypatch):
    real = GUIDE.read_table

    def fake(self, name, results=None, **kw):
        table = real(name, results, **kw)
        if name == sB.sweep_table_name():
            return table.drop(columns=[sB.SWEEP_METRICS["logfreq_pearson"]])
        return table

    monkeypatch.setattr(type(GUIDE), "read_table", fake)
    with pytest.raises(KeyError, match="does not carry"):
        sB.sweep_grid(GUIDE, GUIDE.results_dir)


# ---------------------------------------------------------------------------
# Panel F: picking the winners
# ---------------------------------------------------------------------------

def test_the_top_cut_is_taken_inside_each_lineage(frame):
    """A per-lineage quantile, not one threshold shared across lineages. The lineages differ
    in depth and in how long they ran, so a shared cut would hand some lineages every winner
    and others none."""
    thresholds, counts = {}, {}
    for lineage, g in frame.drop_duplicates(KEY).query("obs_freq > 0").groupby("lineage"):
        freq = g["obs_freq"].to_numpy(float)
        thresholds[lineage] = float(np.quantile(freq, 1.0 - sB.TOP_FRACTION))
        counts[lineage] = (len(freq), int(sB.winners_labels(freq).sum()))
    assert len(set(thresholds.values())) == len(thresholds), "two lineages share a threshold"
    assert max(thresholds.values()) > 5 * min(thresholds.values())
    # Each lineage keeps its own top slice, so the positive class scales with the lineage.
    for n_observed, n_winners in counts.values():
        # abs=2 because the cut cannot split a tie: C.1 has a three-way tie at the
        # boundary frequency (ranks 30-32 all at 0.003092), so its top 5% of 595
        # observed replacements is 32 rather than 30. Breaking such a tie arbitrarily
        # would be worse than overshooting by one or two.
        assert n_winners == pytest.approx(round(n_observed * sB.TOP_FRACTION), abs=2)


def test_winners_ignore_the_replacements_that_never_happened():
    """The point of panel F. Adding zero-frequency rows must change the answer, because an
    AUROC that counted them would be measuring the main figure's question again."""
    freq = np.array([0.9, 0.5, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01])
    score = np.array([9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0, 0.5])
    labels = sB.winners_labels(freq, top=0.2)
    assert labels.sum() == 2 and labels[0] == 1 and labels[-1] == 0

    from mutacc.metrics import auroc
    among_observed = auroc(score, labels)

    # The same two winners, but every unobserved replacement allowed into the negatives. They
    # are easy to beat, so the number moves -- and moves up.
    padded_score = np.r_[score, np.full(200, -5.0)]
    padded_labels = np.r_[labels, np.zeros(200)]
    with_zeros = auroc(padded_score, padded_labels)
    assert with_zeros != pytest.approx(among_observed, abs=1e-6)
    assert with_zeros > among_observed


def test_a_zero_frequency_row_is_refused_rather_than_counted():
    with pytest.raises(ValueError, match="observed replacements only"):
        sB.winners_labels(np.array([0.5, 0.0, 0.2]))


def test_a_lineage_with_no_usable_contrast_raises_rather_than_dropping_out():
    """Documented choice: a degenerate positive class is an error. The band is a bootstrap
    over lineages, so one that quietly held a different set at one end of the axis than at
    the other would not mean anything."""
    with pytest.raises(ValueError, match="no usable contrast"):
        sB.winners_labels(np.full(40, 0.25))          # every frequency equal: no negatives
    with pytest.raises(ValueError, match="no usable contrast"):
        sB.winners_labels(np.array([0.4, 0.4]))       # too few to cut
    assert sB.winners_labels(np.array([0.4, 0.1])).tolist() == [1.0, 0.0]


def test_ties_at_the_cut_join_the_winners():
    """Equality joins the positives, so a block of tied frequencies is not split by the order
    the rows happen to be in."""
    freq = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    labels = sB.winners_labels(freq, top=0.1)
    assert labels[:3].tolist() == [1.0, 1.0, 1.0] and labels[3:].sum() == 0


def test_winners_auroc_stays_in_range_and_beats_chance_at_the_optimum(recomputed):
    assert recomputed["winners_auroc"].between(0.0, 1.0).all()
    at0 = recomputed[np.isclose(recomputed.alpha, 0.0)].groupby("model")["winners_auroc"].mean()
    assert (at0 > 0.5).all()


def test_winners_auroc_golden_values(recomputed):
    """Mean over lineages, no accessibility term: base 0.5765, fine-tuned 0.7467,
    masked marginals. Repinned 23 Oct 2026 for the reassignment (was 0.5740 / 0.7394)."""
    at0 = recomputed[np.isclose(recomputed.alpha, 0.0)].groupby("model")["winners_auroc"].mean()
    by_role = {m.role: m.key for m in GUIDE.models}
    assert at0[by_role["base"]] == pytest.approx(0.5765, abs=5e-4)
    assert at0[by_role["finetune"]] == pytest.approx(0.7467, abs=5e-4)


def test_winners_auroc_golden_values_on_the_unmasked_arm():
    """``--scoring unmasked`` is a supported arm and has its own numbers: base 0.5765,
    fine-tuned 0.7317 (repinned 23 Oct 2026; was 0.5704 / 0.7288). Pinned here because the shipped figure uses the masked arm, so
    nothing else would notice the unmasked tables drifting. Only alpha = 0 is scored,
    which is the whole of the claim and costs one pass over the table."""
    frame = sB.load_scores(GUIDE, GUIDE.results_dir, scoring="unmasked")
    grid = sB.metric_grid(frame, np.array([0.0]), sB.FPR_CAP)
    at0 = grid.groupby("model")["winners_auroc"].mean()
    by_role = {m.role: m.key for m in GUIDE.models}
    assert at0[by_role["base"]] == pytest.approx(0.5765, abs=5e-4)
    assert at0[by_role["finetune"]] == pytest.approx(0.7317, abs=5e-4)


# ---------------------------------------------------------------------------
# The accessibility-only baseline
# ---------------------------------------------------------------------------

def test_accessibility_only_baseline_is_model_independent(base):
    assert len(base) == GUIDE.n_lineages
    assert base[[m for m, *_ in sB.PANELS]].notna().all().all()


def test_baselines_that_differ_between_models_are_refused(frame):
    broken = frame.copy()
    first = GUIDE.models[0].key
    mask = broken["model"] == first
    broken.loc[mask, "mut_prob"] = broken.loc[mask, "mut_prob"].sample(frac=1, random_state=0).to_numpy()
    with pytest.raises(ValueError, match="differs between models"):
        sB.baseline_grid(broken, GUIDE, GUIDE.results_dir, sB.FPR_CAP)


def test_accessibility_alone_picks_winners_barely_above_chance(base):
    """The contrast the caption rests on: accessibility is informative about which
    replacements are reachable and close to uninformative about which of them succeed."""
    assert float(base["winners_auroc"].mean()) == pytest.approx(0.533, abs=5e-3)
    assert float(base["auroc"].mean()) == pytest.approx(0.909, abs=5e-3)


# ---------------------------------------------------------------------------
# The shipped frequency table and the exporter that builds it
# ---------------------------------------------------------------------------

def test_the_frequency_table_round_trips():
    frequency = GUIDE.read_table("replacement_frequency")
    assert list(frequency.columns) == KEY + ["obs_freq", "depth"]
    assert len(frequency) == 139_802
    assert not frequency.duplicated(KEY).any()
    assert frequency["obs_freq"].between(0.0, 1.0).all()
    assert (frequency["depth"] >= 0).all()
    assert int((frequency["obs_freq"] > 0).sum()) == 6_392


def test_the_frequency_table_agrees_with_the_observed_flag_on_every_row():
    frequency = GUIDE.read_table("replacement_frequency")
    observed = GUIDE.read_table("replacement_observed")
    checked = exporter.check_against_observed(frequency, observed)
    assert checked == len(observed) == len(frequency)


def test_one_disagreeing_row_is_enough_to_stop_the_export():
    frequency = GUIDE.read_table("replacement_frequency").head(50).copy()
    observed = GUIDE.read_table("replacement_observed").merge(frequency[KEY], on=KEY)
    exporter.check_against_observed(frequency, observed)        # the untouched pair is fine
    frequency.loc[frequency.index[0], "obs_freq"] = 0.5 if not observed["obs_present"].iloc[0] else 0.0
    with pytest.raises(ValueError, match="disagree between obs_freq"):
        exporter.check_against_observed(frequency, observed)


def test_an_extra_replacement_in_one_table_is_caught():
    frequency = GUIDE.read_table("replacement_frequency").head(50)
    observed = GUIDE.read_table("replacement_observed").head(49)
    with pytest.raises(ValueError, match="cover different replacements"):
        exporter.check_against_observed(frequency, observed)


def test_the_exporter_stores_one_row_per_replacement_not_one_per_model():
    archive = pd.DataFrame({
        "lineage": ["A", "A", "A", "A"], "position": [1, 2, 1, 2], "aa": ["K", "R", "K", "R"],
        "model": ["m1", "m1", "m2", "m2"], "obs_freq": [0.5, 0.0, 0.5, 0.0],
        "obs_present": [1, 0, 1, 0], "depth": [10, 10, 10, 10],
    })
    out = exporter.collapse(archive)
    assert len(out) == 2 and list(out.columns) == KEY + ["obs_freq", "depth"]


def test_a_model_dependent_frequency_refuses_to_collapse():
    archive = pd.DataFrame({
        "lineage": ["A", "A"], "position": [1, 1], "aa": ["K", "K"],
        "model": ["m1", "m2"], "obs_freq": [0.5, 0.25],
        "obs_present": [1, 1], "depth": [10, 10],
    })
    with pytest.raises(ValueError, match="more than one 'obs_freq'"):
        exporter.collapse(archive)


def test_a_frequency_outside_zero_to_one_is_refused():
    archive = pd.DataFrame({
        "lineage": ["A"], "position": [1], "aa": ["K"], "model": ["m1"],
        "obs_freq": [7.0], "obs_present": [1], "depth": [10],
    })
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        exporter.collapse(archive)


def test_the_frequency_table_has_a_manifest_row_naming_the_archive():
    manifest = pd.read_csv(GUIDE.results_dir / "MANIFEST.tsv", sep="\t", dtype=str,
                           keep_default_na=False)
    row = manifest.loc[manifest["file"] == exporter.DEST]
    assert len(row) == 1
    # Since the 22 Oct 2026 reassignment the shipped table is no longer the archive
    # collapse this exporter performs: it is rebuilt from the reassigned sub-alignments
    # by dev/lineage_reassign/build_observed_tables.py. The manifest has to say which,
    # so the source is checked for naming a producer rather than for equalling the
    # exporter's own SOURCE constant, which now describes only how it *would* be built.
    source = row["source"].iloc[0]
    assert source and (source == exporter.SOURCE.as_posix()
                       or "build_observed_tables.py" in source)
    assert int(row["rows"].iloc[0]) == 139_802 and int(row["columns"].iloc[0]) == 5
    # Same archive file as the flag table it refines, so the two cannot be different vintages.
    flag = manifest.loc[manifest["file"] == exporter.OBSERVED]
    assert row["source_sha256_16"].iloc[0] == flag["source_sha256_16"].iloc[0]


def test_an_unflagged_replacement_is_an_error_not_a_silent_drop(monkeypatch):
    real = GUIDE.read_table

    def fake(self, name, results=None, **kw):
        t = real(name, results, **kw)
        return t.iloc[1:] if name == "replacement_observed" else t

    monkeypatch.setattr(type(GUIDE), "read_table", fake)
    with pytest.raises(ValueError, match="flag without a frequency|no observed flag"):
        sB.load_scores(GUIDE, GUIDE.results_dir)


def test_a_flag_that_contradicts_the_frequency_stops_the_figure(monkeypatch):
    real = GUIDE.read_table

    def fake(self, name, results=None, **kw):
        t = real(name, results, **kw)
        if name == "replacement_frequency":
            t = t.copy()
            t.loc[t.index[0], "obs_freq"] = 0.5     # row 0 is unobserved in the flag table
        return t

    monkeypatch.setattr(type(GUIDE), "read_table", fake)
    with pytest.raises(ValueError, match="flag and the observed frequency disagree"):
        sB.load_scores(GUIDE, GUIDE.results_dir)


# ---------------------------------------------------------------------------
# Bands, caption and the render
# ---------------------------------------------------------------------------

def test_bands_bracket_the_mean(grid):
    band = sB.bands(grid, GUIDE, n_boot=100, seed=1)
    assert (band["lo"] <= band["mean"] + 1e-12).all() and (band["mean"] <= band["hi"] + 1e-12).all()
    assert len(band) == len(sB.PANELS) * len(GUIDE.models) * 101


def test_the_figure_is_two_rows_of_three():
    assert len(sB.PANELS) == 2 * sB.N_COLUMNS


def test_caption_states_the_prevalence_range_the_axis_cut_and_the_null_result(grid, base):
    band = sB.bands(grid, GUIDE, n_boot=20, seed=1)
    observed = pd.Series({lin: 100 for lin in GUIDE.lineage_ids})
    text = " ".join(sB.caption(base, band, GUIDE, sB.FPR_CAP, sB.TOP_FRACTION, 20, observed).split())
    assert "prevalence runs from 1.6% to 10.1%" in text
    assert "anti-predictive" in text and "values table" in text
    assert "gains essentially nothing" in text
    # The contrast panel F exists to make, stated as numbers and not as a claim: accessibility
    # alone barely beats chance at picking the winners while being strong at saying which
    # replacements occur at all. Read off the same baseline table the caption formats.
    assert f"scores {float(base['winners_auroc'].mean()):.3f} at picking the" in text
    assert f"against {float(base['auroc'].mean()):.3f} at saying which" in text
    assert "close to uninformative about what" in text


@pytest.mark.slow
def test_script_renders_with_tables(tmp_path):
    assert sB.main(["--guide", str(GUIDE.source), "--out", str(tmp_path), "--n-boot", "50"]) == 0
    assert (tmp_path / "sB_ranking_metrics.png").stat().st_size > 20_000
    vals = pd.read_csv(tmp_path / "sB_ranking_metrics_values.tsv", sep="\t")
    assert vals["mean"].notna().all()
    assert set(vals["metric"]) == {m for m, *_ in sB.PANELS}
