"""Test suite for Supplementary Figure S16 and Supplementary Table S16.

Validates:
1. make_sN_simulation_control.py execution and output generation (PDF, PNG, caption, values table).
2. Supplementary Table 16 (supl_table_16.tsv) format, columns, and completeness.
3. Golden values across all 4 panels of S16:
   - Panel A: 12-class substitution spectra percentages sum to 100% per arm; JC69 is flat (8.33%).
   - Panel B: Cross-correlation of PLMs vs mutation spectra across full accessibility matrix.
   - Panel C: Alpha sweep curves for fitness estimation: fine-tuned models have optimal alpha < 0.
   - Panel D: Cross-correlation in N=1 band only: specificity for generating spectra; flat FT is ~0.
4. export_sY_dms_alpha_optima.py: writes Table SY, the DMS alpha optima, and its caption.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_make_sN_simulation_control_execution(tmp_path: Path):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "figures" / "make_sN_simulation_control.py"),
        "--results", str(REPO_ROOT / "results" / "sim_v3_star90"),
        "--out", str(tmp_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"make_sN_simulation_control.py failed: {res.stderr}"

    assert (tmp_path / "sN_simulation_control.pdf").exists()
    assert (tmp_path / "sN_simulation_control.png").exists()
    assert (tmp_path / "sN_simulation_control_caption.md").exists()
    assert (tmp_path / "sN_simulation_control_values.tsv").exists()
    assert (tmp_path / "supl_table_16.tsv").exists()
    assert (tmp_path / "table_sN_simulation_control.tsv").exists()


def test_sN_simulation_control_golden_values():
    val_path = REPO_ROOT / "output" / "sN_simulation_control_values.tsv"
    assert val_path.exists(), "sN_simulation_control_values.tsv must exist in output/"

    df = pd.read_csv(val_path, sep="\t")
    assert not df.empty

    # Panel A: Spectra
    p_a = df[df["panel"] == "A"]
    assert len(p_a) == 48  # 4 arms x 12 classes
    for cond in p_a["condition"].unique():
        sub = p_a[p_a["condition"] == cond]
        assert np.isclose(sub["value"].sum(), 100.0, atol=0.2)

    # Panel C: Alpha sweep optima
    p_alpha = df[df["panel"] == "C"]
    opt_rows = p_alpha[p_alpha["category"] == "optimum"]
    assert len(opt_rows) == 5  # 5 models
    # Fine-tuned models must have negative optimal alpha (debiasing proposal bias)
    for _, r in opt_rows.iterrows():
        cond = r["condition"]
        if "FT" in cond:
            assert r["value"] < 0, f"Fine-tuned model {cond} optimal alpha must be < 0, got {r['value']}"
        elif "Base" in cond:
            assert np.isclose(r["value"], 0.0, atol=0.05), f"Base model optimal alpha should be near 0, got {r['value']}"

    # Panel B: Full accessibility matrix cross-correlations
    p_full = df[df["panel"] == "B"]
    assert len(p_full) == 20  # 5 models x 4 spectra
    assert (p_full["value"] > 0).all()  # All positive due to genetic code distance gradient

    # Panel D: N=1 band only cross-correlations
    p_d = df[df["panel"] == "D"]
    assert len(p_d) == 20  # 5 models x 4 spectra

    # Check flat JC69 FT is near zero across all spectra in 1-nt band
    flat_rows = p_d[p_d["condition"].str.contains("JC69|flat", case=False)]
    assert not flat_rows.empty
    assert (flat_rows["value"].abs() < 0.06).all(), f"Flat control correlations should be near 0 in 1-nt band, got {flat_rows['value'].tolist()}"

    # Check SC2 FT has highest correlation on SC2 spectrum
    sc2_rows = p_d[p_d["condition"].str.contains("SARS|SC2", case=False)]
    assert not sc2_rows.empty
    sc2_val_on_sc2 = sc2_rows[sc2_rows["category"].str.contains("SARS|SC2", case=False)]["value"].iloc[0]
    for _, r in sc2_rows.iterrows():
        assert sc2_val_on_sc2 >= r["value"], f"SC2 FT should peak on SC2, but got higher on {r['category']}"


def test_export_sY_dms_alpha_optima():
    out_table = REPO_ROOT / "output" / "sY_dms_alpha_optima.tsv"
    assert out_table.exists(), "sY_dms_alpha_optima.tsv must exist in output/"

    df = pd.read_csv(out_table, sep="\t")
    assert len(df) == 18  # 9 panels x 2 models
    assert "interpretable" in df.columns
    assert "alpha_best" in df.columns
    assert "delta_rho" in df.columns

    # Interpretable fine-tuned models must have negative alpha_best
    ft_interpretable = df[(df["model_role"] == "finetune") & (df["interpretable"] == True)]
    assert len(ft_interpretable) > 0
    alphas = [float(a) for a in ft_interpretable["alpha_best"]]
    assert all(a < 0 for a in alphas), f"All interpretable fine-tuned alpha_best must be negative: {alphas}"


# ---------------------------------------------------------------------------
# Actual golden values
# ---------------------------------------------------------------------------
# ``test_sN_simulation_control_golden_values`` above is named for goldens and contains none: row
# counts, a sum-to-100 and sign checks. Those pass for a table whose every value
# has drifted. The numbers below are the ones the figure is read for.

#: Panel C, the alpha optimum of each arm, from
#: output/sN_simulation_control_values.tsv. Two decimals is the grid the sweep
#: is reported on, so these are exact, not rounded.
S16_OPTIMA = {
    "Base (untuned)": 0.00,
    "JC69 (flat) FT": -0.16,
    "Human FT": -0.09,
    "SC2 FT": -0.13,
    "Influenza H3N2 FT": -0.22,
}

#: Panel D, each arm's correlation with its *own* spectrum in the one-nucleotide
#: band -- the diagonal of the matrix the panel exists to show.
S16_OWN_SPECTRUM = {"H3N2 FT": ("H3N2", 0.1451), "Human FT": ("Human", 0.1089),
                    "SC2 FT": ("SC2", 0.1618)}


@pytest.fixture(scope="module")
def s16_values():
    return pd.read_csv(REPO_ROOT / "output" / "sN_simulation_control_values.tsv", sep="\t")


@pytest.mark.parametrize("condition,expected", sorted(S16_OPTIMA.items()))
def test_each_arms_alpha_optimum_is_the_published_one(s16_values, condition, expected):
    rows = s16_values[(s16_values["panel"] == "C") & (s16_values["category"] == "optimum")]
    got = rows.loc[rows["condition"] == condition, "value"]
    assert len(got) == 1, f"{condition}: expected one optimum row, found {len(got)}"
    assert float(got.iloc[0]) == pytest.approx(expected, abs=5e-3)


@pytest.mark.parametrize("condition,spectrum_and_value", sorted(S16_OWN_SPECTRUM.items()))
def test_each_fine_tune_tracks_its_own_spectrum_most_closely(s16_values, condition, spectrum_and_value):
    """The claim S14 (repo stem sN) is in the manuscript for: a model fine-tuned on
    one virus learns *that virus's* accessibility, not accessibility in general.

    Restricted to the one-nucleotide band, where the genetic code cannot supply the
    correlation -- in panel B, which uses the full matrix, every arm correlates with
    every spectrum at 0.25-0.50 and the diagonal is invisible.
    """
    own, expected = spectrum_and_value
    row = s16_values[(s16_values["panel"] == "D") & (s16_values["condition"] == condition)]
    assert len(row) == 4, f"{condition}: expected 4 spectra, found {len(row)}"
    assert float(row.loc[row["category"] == own, "value"].iloc[0]) == pytest.approx(expected, abs=5e-4)
    best = row.loc[row["value"].idxmax(), "category"]
    assert best == own, (
        f"{condition} now tracks {best} more closely than its own {own} spectrum; "
        f"the simulation control no longer shows what the figure claims"
    )


def test_the_flat_control_arm_learns_no_spectrum(s16_values):
    """The negative control: fine-tuned under JC69, where there is no spectrum to
    learn, every correlation in the one-nucleotide band must sit near zero."""
    row = s16_values[(s16_values["panel"] == "D") & (s16_values["condition"] == "Flat FT")]
    assert len(row) == 4
    assert row["value"].abs().max() < 0.05, row[["category", "value"]].to_string(index=False)
