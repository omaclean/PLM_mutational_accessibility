"""S6: correlation with accessibility under swapped spectra. Planted truth, guards, shipped-table goldens."""

import numpy as np
import pandas as pd
import pytest

import make_sF_spectrum_swap as sF
import synth_codon as sc
from mutacc import accessibility as acc
from mutacc import spectrum_swap as sw
from mutacc.guide import REPO_ROOT, load_guide

SPECTRA = REPO_ROOT / "data" / "spectra"
GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")


def q(stem):
    return acc.load_spectrum(SPECTRA / f"{stem}.csv")


REF, K80, FLAT = q("influenza_h3n2"), q("k80_kappa2"), q("jc69")


def synthetic(tmp_path, planted_q=None, seed=1, n_pos=1500):
    """Two models on a synthetic protein. The 'ft' model's probability IS accessibility under
    ``planted_q`` (times noise); the 'base' model is random. mut_prob is accessibility under REF."""
    frame, roots = sc.make_scores(tmp_path, n_pos=n_pos, n_lin=2, seed=seed, models=("base", "ft"), q=REF)
    rng = np.random.default_rng(seed + 7)
    codon = pd.Series(sc.SENSE)   # placeholder to keep linters quiet
    ref_codons = sw.row_codons(frame.loc[frame.model == "ft"], roots)
    ft = frame["model"] == "ft"
    if planted_q is not None:
        a = acc.accessibility(ref_codons, frame.loc[ft, "aa"].tolist(), planted_q, row_sum=sw.mean_row_sum(REF))
        frame.loc[ft, "plm_prob"] = a * np.exp(0.15 * rng.normal(size=int(ft.sum())))
    pair = frame[["ref_aa", "aa"]].drop_duplicates()
    pair["blosum62"] = rng.normal(size=len(pair))
    pair["neg_grantham"] = rng.normal(size=len(pair))
    return frame.merge(pair, on=["ref_aa", "aa"]), roots


def conditions(*extra):
    out = [sw.Condition("ref", "Reference", REF, True), sw.Condition("flat", "Flat", FLAT)]
    return out + list(extra)


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    frame, roots = synthetic(tmp_path_factory.mktemp("p"), planted_q=K80)
    conds = conditions(sw.Condition("k80", "K80", K80))
    table = sw.swap_table(frame, ["base", "ft"], ["blosum62", "neg_grantham"], roots, conds, n_boot=12, seed=3)
    return frame, roots, conds, table


# ---------------------------------------------------------------------------
# Tie merging
# ---------------------------------------------------------------------------

def test_noise_ties_are_merged_but_real_differences_survive():
    v = np.array([0.5, 0.5 + 1e-16, 0.25, 0.25 * (1 + 1e-6)])
    out = sw.merge_rounding_ties(v)
    assert out[0] == out[1] and out[2] != out[3] and len(np.unique(out)) == 3


def test_merge_rounding_ties_rejects_bad_values():
    for bad in ([np.nan, 1.0], [-1.0, 1.0], [np.inf]):
        with pytest.raises(ValueError):
            sw.merge_rounding_ties(np.array(bad))


def test_zero_is_a_valid_accessibility():
    assert sw.merge_rounding_ties(np.array([0.0, 1.0])).tolist() == [0.0, 1.0]


def test_flat_spectrum_accessibility_has_few_distinct_values_after_merging():
    codons = sc.SENSE
    aas = ["A"] * len(codons)
    a = sw.condition_accessibility(codons, aas, FLAT, sw.mean_row_sum(REF))
    assert len(np.unique(sw.merge_rounding_ties(a))) <= len(np.unique(a))


# ---------------------------------------------------------------------------
# Structure and planted truth
# ---------------------------------------------------------------------------

def test_table_has_every_condition_series_and_stratum_once(planted):
    _, _, conds, t = planted
    assert len(t) == len(conds) * 4 * 4
    assert not t.duplicated(["condition", "series", "stratum"]).any()
    assert set(t["stratum"]) == {"all", "1", "2", "3"} and set(t["kind"]) == {"model", "property"}
    assert np.isfinite(t[["estimate", "lo", "hi", "delta", "delta_lo", "delta_hi"]].to_numpy()).all()


def test_reference_rows_have_zero_delta_exactly(planted):
    t = planted[3]
    r = t.loc[t["reference"]]
    assert (r[["delta", "delta_lo", "delta_hi"]] == 0).all().all()
    assert r["condition"].nunique() == 1


def test_delta_is_reference_minus_condition(planted):
    t = planted[3]
    ref = t.loc[t["reference"]].set_index(["series", "stratum"])["estimate"]
    for row in t.loc[~t["reference"]].itertuples():
        assert row.delta == pytest.approx(ref[(row.series, row.stratum)] - row.estimate, abs=1e-12)


def test_interval_brackets_the_estimate_mostly(planted):
    t = planted[3]
    assert (t["lo"] <= t["hi"]).all() and (t["delta_lo"] <= t["delta_hi"]).all()
    assert ((t["lo"] <= t["estimate"] + 0.05) & (t["estimate"] - 0.05 <= t["hi"])).all()


def test_planted_spectrum_is_recovered_by_the_model_that_carries_it(planted):
    """The 'ft' model is built from K80 accessibility: it must correlate best with K80 of the three
    spectra, in every stratum, and the paired difference must exclude zero."""
    t = planted[3]
    for st in ("all", "1", "2", "3"):
        e = t.loc[(t.series == "ft") & (t.stratum == st)].set_index("condition")["estimate"]
        assert e["k80"] == e.max(), (st, e.to_dict())
        assert e["k80"] > 0.6
    d = t.loc[(t.series == "ft") & (t.stratum == "1") & (t.condition == "k80")].iloc[0]
    assert d["delta_hi"] < 0                       # reference minus K80 is negative: K80 fits better


def test_a_model_with_no_spectrum_in_it_shows_no_preference(planted):
    t = planted[3]
    b = t.loc[(t.series == "base") & (t.stratum == "all") & (t.condition == "k80")].iloc[0]
    assert abs(b["estimate"]) < 0.1


def test_swap_table_is_reproducible_and_seed_dependent(planted):
    frame, roots, conds, t = planted
    again = sw.swap_table(frame, ["base", "ft"], ["blosum62", "neg_grantham"], roots, conds, n_boot=12, seed=3)
    pd.testing.assert_frame_equal(t, again)
    other = sw.swap_table(frame, ["base", "ft"], ["blosum62", "neg_grantham"], roots, conds, n_boot=12, seed=4)
    assert not np.allclose(t["lo"], other["lo"])
    assert np.allclose(t["estimate"], other["estimate"])      # the point estimate never depends on the seed


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _call(frame, roots, conds, **kw):
    kw.setdefault("n_boot", 3)
    kw.setdefault("seed", 1)
    return sw.swap_table(frame, ["base", "ft"], ["blosum62", "neg_grantham"], roots, conds, **kw)


def test_exactly_one_reference_is_required(planted):
    frame, roots, _, _ = planted
    with pytest.raises(ValueError, match="exactly one"):
        _call(frame, roots, [sw.Condition("a", "A", FLAT), sw.Condition("b", "B", K80)])
    with pytest.raises(ValueError, match="exactly one"):
        _call(frame, roots, [sw.Condition("a", "A", REF, True), sw.Condition("b", "B", K80, True)])


def test_duplicate_condition_keys_are_rejected(planted):
    frame, roots, _, _ = planted
    with pytest.raises(ValueError, match="duplicate"):
        _call(frame, roots, [sw.Condition("a", "A", REF, True), sw.Condition("a", "B", K80)])


def test_a_reference_that_does_not_reproduce_mut_prob_is_rejected(planted):
    frame, roots, _, _ = planted
    with pytest.raises(ValueError, match="does not reproduce"):
        _call(frame, roots, [sw.Condition("a", "A", REF * 0.5, True)])


def test_zero_bootstrap_draws_rejected(planted):
    frame, roots, conds, _ = planted
    with pytest.raises(ValueError, match="at least 1"):
        _call(frame, roots, conds, n_boot=0)


def test_wrong_roots_are_rejected(planted, tmp_path):
    frame, roots, conds, _ = planted
    seq = "".join(l for l in roots["L0"].read_text().splitlines() if not l.startswith(">"))
    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    codons[0] = next(c for c in sc.SENSE if acc.GENETIC_CODE[c] != acc.GENETIC_CODE[codons[0]])
    bad = sc.write_root(tmp_path / "bad.fa", codons)
    with pytest.raises(ValueError, match="does not translate"):
        _call(frame, {**roots, "L0": bad}, conds)
    with pytest.raises(KeyError, match="no root"):
        _call(frame, {"L0": roots["L0"]}, conds)


def test_a_constant_score_column_is_reported_not_averaged_over(planted):
    frame, roots, conds, _ = planted
    broken = frame.assign(blosum62=1.0)
    with pytest.raises(ValueError, match="could not be computed"):
        _call(broken, roots, conds)


def test_models_scored_on_different_replacements_are_rejected(planted):
    frame, roots, conds, _ = planted
    broken = frame.drop(frame.loc[frame.model == "ft"].index[:3])
    with pytest.raises(ValueError, match="not scored on the same"):
        _call(broken, roots, conds)


def test_nonfinite_scores_are_rejected(planted):
    frame, roots, conds, _ = planted
    broken = frame.copy()
    broken.loc[broken.index[0], "plm_prob"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _call(broken, roots, conds)


# ---------------------------------------------------------------------------
# The figure module
# ---------------------------------------------------------------------------

def table_for_figure(planted):
    t = planted[3].copy()
    t["label"] = t["condition"]
    t["series_label"] = t["series"]
    t["role"] = t["series"].map({"base": "base", "ft": "finetune", "blosum62": "biochemistry", "neg_grantham": "biochemistry"})
    t["n_boot"] = 12
    return t


def test_validate_accepts_a_good_table_and_rejects_bad_ones(planted):
    t = table_for_figure(planted)
    sF.validate(t)
    with pytest.raises(KeyError, match="missing columns"):
        sF.validate(t.drop(columns=["delta_lo"]))
    with pytest.raises(ValueError, match="exactly one reference"):
        sF.validate(t.assign(reference=False))
    with pytest.raises(ValueError, match="duplicate"):
        sF.validate(pd.concat([t, t.iloc[:1]]))
    bad = t.copy()
    bad.loc[bad.index[0], "estimate"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sF.validate(bad)


def test_summarise_counts_by_paired_interval():
    rows = []
    for cond, dlo, dhi in [("a", -0.2, -0.1), ("b", 0.1, 0.2), ("c", -0.1, 0.1)]:
        rows.append({"condition": cond, "reference": False, "series": "m", "series_label": "M", "stratum": "1",
                     "delta_lo": dlo, "delta_hi": dhi})
    s = sF.summarise(pd.DataFrame(rows))
    assert s.iloc[0][["n_spectra", "alt_higher", "alt_lower"]].tolist() == [3, 1, 1]


def test_figure_draws_from_a_small_table(planted, tmp_path):
    t = table_for_figure(planted)
    sF.validate(t)
    from mutacc import style
    fig = sF.draw_swap(t)
    paths = style.save_figure(fig, tmp_path / "a")
    assert all(p.stat().st_size > 5_000 for p in paths)
    assert not hasattr(sF, "draw_delta"), "the delta figure was dropped"


# ---------------------------------------------------------------------------
# Shipped table
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    from mutacc import codon_learning as cl
    return cl.read_derived(GUIDE, "spectrum_swap_strata")


def test_shipped_table_is_complete(shipped):
    assert len(shipped) == shipped["condition"].nunique() * 4 * 4
    sF.validate(shipped)
    assert shipped["n_boot"].nunique() == 1


#: Values from the archive's a35 output (13 lineages, 200 resamples, five decimals). The archive does not merge
#: floating-point-noise ties, so agreement is to a few thousandths; flat-spectrum cells are not pinned (the archive's
#: values there move by 0.015 with rounding noise).
A35 = [("h3n2", "ESM2_650M_HA80", "1", 0.29858), ("h3n2", "ESM2_650M_HA80", "2", 0.24698),
       ("h3n2_symmetrised", "ESM2_650M_HA80", "1", 0.35783), ("sars_cov_2", "ESM2_650M_HA80", "1", 0.29697),
       ("human", "ESM2_650M_HA80", "1", 0.35350), ("k80", "ESM2_650M_HA80", "1", 0.35807),
       ("h3n2", "ESM2_650M_HA80_raw", "1", 0.07322), ("sars_cov_2", "ESM2_650M_HA80_raw", "1", 0.11891),
       ("human", "ESM2_650M_HA80_raw", "1", 0.15184), ("k80", "ESM2_650M_HA80_raw", "1", 0.14597)]


@pytest.mark.parametrize("cond, series, stratum, expected", A35)
def test_golden_against_the_archive(shipped, cond, series, stratum, expected):
    row = shipped.loc[(shipped.condition == cond) & (shipped.series == series) & (shipped.stratum == stratum)].iloc[0]
    # 5e-4, not the 4e-3 this used to allow. The goldens are written to five
    # decimals; a tolerance of 4e-3 let the third decimal move freely, so
    # 0.29858 -> 0.3025 would have passed. Measured 15 Oct 2026: the largest gap
    # between the shipped table and the archive is 3.3e-4 (human / fine-tuned,
    # stratum 1), so this is the tightest bound the reproduction actually meets.
    assert row["estimate"] == pytest.approx(expected, abs=5e-4)


def test_shipped_reference_matches_the_guides_own_accessibility(shipped):
    scores = GUIDE.read_table("replacement_scores")
    assert sF.check_reference(shipped, GUIDE, scores) < sF.SELF_CHECK_TOL


def test_a_tampered_reference_is_caught(shipped):
    scores = GUIDE.read_table("replacement_scores")
    bad = shipped.copy()
    bad.loc[bad["reference"] & (bad.series == "blosum62") & (bad.stratum == "1"), "estimate"] += 0.05
    with pytest.raises(ValueError, match="disagrees"):
        sF.check_reference(bad, GUIDE, scores)


def test_controls_move_with_the_models_across_spectra(shipped):
    """What the caption leans on: within one nucleotide change the no-model controls are not flat across spectra."""
    d = shipped.loc[(shipped.stratum == "1") & ~shipped.reference]
    for ser in ("neg_grantham", "blosum62"):
        assert d.loc[d.series == ser, "delta"].abs().max() > 0.05


@pytest.mark.slow
def test_script_renders(tmp_path):
    assert sF.main(["--guide", str(GUIDE.source), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "sF_spectrum_swap.png").stat().st_size > 20_000
    assert (tmp_path / "sF_spectrum_swap_caption.md").read_text().strip()
    # The delta figure was dropped: the correlations themselves are the figure now.
    assert not (tmp_path / "sF_spectrum_delta.png").exists()
