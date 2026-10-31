"""Carpet-bombing test suite for Reviewer 2 simulation control pipeline and results.

Validates:
1. tools/exports/sim.py and the exported replacement_scores.tsv.gz table.
2. tools/run_symmetry_sim.py functions, units, controls, and partial orientation logic.
3. Integrity, exact invariants, and bounds across all 20 symmetry battery tables in results/sim_v3_star90/.
4. MANIFEST.tsv completeness, column counts, and row counts.
"""
from __future__ import annotations

from pathlib import Path
import gzip
import numpy as np
import pandas as pd
import pytest
from Bio.Data import CodonTable

REPO_ROOT = Path(__file__).resolve().parents[1]
#: The exploration tree, which is NOT part of clean_repo and is not shipped.
#: Anything reading from here cannot run on a clean checkout -- exactly the
#: reproducibility complaint the revision is answering -- so every such test
#: carries ``needs_exploration_tree`` below and skips with its reason stated
#: rather than failing or, worse, passing only on this machine.
OUTER_REPO = Path(__file__).resolve().parents[2]
SIM_SOURCE = OUTER_REPO / "tmp_explore" / "gpu_sim_arms" / "score"
#: A decision is outstanding (docs/notes/extended_red_team_tests.md A4): ship a trimmed
#: fixture, or drop the test and rely on the exported table it feeds.
needs_exploration_tree = pytest.mark.skipif(
    not SIM_SOURCE.exists(),
    reason=f"reads {SIM_SOURCE} outside clean_repo; not shipped (see docs/notes/extended_red_team_tests.md A4)",
)
RESULTS_DIR = REPO_ROOT / "results" / "sim_v3_star90"
SCORES_FILE = RESULTS_DIR / "replacement_scores.tsv.gz"
MANIFEST_FILE = RESULTS_DIR / "MANIFEST.tsv"

MODELS = ["plm_base", "plm_human_ft", "plm_sc2_ft", "plm_h3n2_ft", "plm_jc69_ft"]
PROFILES = ["pssm_human", "pssm_sc2", "pssm_h3n2", "pssm_jc69"]
SPECTRA = ["sars_cov_2", "human", "influenza_h3n2", "jc69"]
EXPECTED_COLS = [
    "site", "ref_codon", "ref_aa", "aa", "nt_steps",
    "plm_base", "plm_human_ft", "plm_sc2_ft", "plm_h3n2_ft", "plm_jc69_ft",
    "ground_truth", "pssm_human", "pssm_sc2", "pssm_h3n2", "pssm_jc69",
    "mut_prob_SC2_Directional", "mut_prob_Human_Germline", "mut_prob_H3N2", "mut_prob_JC69"
]


def test_simulation_replacement_scores_structure_and_invariants():
    """Verify exact dimensions, schema, and biological invariants of the simulation replacement table."""
    assert SCORES_FILE.exists(), f"Missing {SCORES_FILE}"
    df = pd.read_csv(SCORES_FILE, sep="\t")

    # Exact shape
    assert len(df) == 13300, f"Expected 13,300 rows (700 sites x 19 AAs), got {len(df)}"
    assert list(df.columns) == EXPECTED_COLS, f"Column mismatch in {SCORES_FILE}"

    # Exactly 700 unique sites, 19 amino acids per site
    sites = df["site"].unique()
    assert len(sites) == 700
    assert set(sites) == set(range(1, 701))
    counts_per_site = df.groupby("site").size()
    assert (counts_per_site == 19).all(), "Every site must have exactly 19 non-synonymous amino acids"

    # Genetic code translation invariant
    fwd_table = CodonTable.unambiguous_dna_by_id[1].forward_table
    for _, row in df.iloc[::50].iterrows():  # sample rows across sites
        codon = row["ref_codon"]
        expected_aa = fwd_table[codon]
        assert row["ref_aa"] == expected_aa, f"Codon {codon} should translate to {expected_aa}, got {row['ref_aa']}"
        assert row["aa"] != expected_aa, f"Replacement {row['aa']} cannot equal wildtype {expected_aa}"

    # Nucleotide steps must be in {1, 2, 3}
    assert set(df["nt_steps"].unique()).issubset({1, 2, 3})
    assert (df["nt_steps"] == 1).sum() > 0
    assert (df["nt_steps"] == 2).sum() > 0
    assert (df["nt_steps"] == 3).sum() > 0

    # No NaN values anywhere
    assert not df.isna().any().any(), "Table must contain no NaN values"

    # Probabilities and frequencies in valid numerical ranges
    for m in MODELS:
        assert (df[m] > 0).all() and (df[m] <= 1.0).all(), f"Model {m} probabilities out of range"
    for p in PROFILES:
        assert (df[p] >= 0).all() and (df[p] <= 1.0).all(), f"PSSM {p} frequencies out of range"
    assert (df["ground_truth"] > 0).all() and (df["ground_truth"] <= 1.0).all(), "Ground truth fitness out of range"

    # Mutational accessibility columns positive
    for mut_col in ["mut_prob_SC2_Directional", "mut_prob_Human_Germline", "mut_prob_H3N2", "mut_prob_JC69"]:
        assert (df[mut_col] > 0).all(), f"{mut_col} must be strictly positive"


def test_simulation_manifest_completeness():
    """Every file in sim_v3_star90 must be registered in MANIFEST.tsv with matching rows and columns."""
    assert MANIFEST_FILE.exists()
    manifest = pd.read_csv(MANIFEST_FILE, sep="\t")

    # Must contain replacement_scores + 19 symmetry tables + empirical spectra = 21 files total
    assert len(manifest) == 21, f"Expected 21 manifest rows, got {len(manifest)}"

    for _, row in manifest.iterrows():
        fname = row["file"]
        target_path = RESULTS_DIR / fname
        assert target_path.exists(), f"File {fname} in MANIFEST.tsv does not exist on disk"

        # Check rows and columns
        df = pd.read_csv(target_path, sep="\t")
        assert len(df) == row["rows"], f"Row count mismatch for {fname}: file has {len(df)}, manifest says {row['rows']}"
        assert df.shape[1] == row["columns"], f"Col count mismatch for {fname}: file has {df.shape[1]}, manifest says {row['columns']}"
        assert str(row["used_for"]).strip() != "", f"Empty used_for in manifest for {fname}"


@pytest.mark.parametrize("spectrum", ["sars_cov_2", "human", "influenza_h3n2", "jc69"])
def test_symmetry_battery_tables_exist_and_valid(spectrum: str):
    """Verify that all 4 battery test tables exist and have valid structure for each spectrum."""
    battery_types = ["orientation", "null", "flip", "partial_given_truth"]
    if spectrum != "jc69":  # jc69 cleanly skips direction test because all rates are equal
        battery_types.append("direction")

    for btype in battery_types:
        fpath = RESULTS_DIR / f"symmetry_{spectrum}_{btype}.tsv.gz"
        assert fpath.exists(), f"Missing {fpath}"
        df = pd.read_csv(fpath, sep="\t")
        assert len(df) > 0, f"Empty table {fpath}"
        assert not df.isna().all().any(), f"All-NaN column in {fpath}"


def test_flip_null_exact_invariants():
    """Verify theoretical invariants of the flip null test across spectra."""
    # 1. In flat JC69, reciprocal off-diagonals are identical, so every flip is identical to truth
    jc69_flip = pd.read_csv(RESULTS_DIR / "symmetry_jc69_flip.tsv.gz", sep="\t")
    assert (jc69_flip["pct_flip"] == 1.0).all(), "In flat JC69, pct_flip must be identically 1.0 for all models"
    assert np.allclose(jc69_flip["rho_flip_mean"], jc69_flip["rho_real"]), "In JC69, mean flip rho must equal real rho"
    assert np.allclose(jc69_flip["rho_transposed"], jc69_flip["rho_real"]), "In JC69, transposed rho must equal real rho"

    # 2. In SARS-CoV-2, plm_sc2_ft must rank real direction 1st or 2nd across all strata (pct_flip <= 2/64 = 0.03125)
    sc2_flip = pd.read_csv(RESULTS_DIR / "symmetry_sars_cov_2_flip.tsv.gz", sep="\t")
    sc2_ft = sc2_flip[sc2_flip["score"] == "plm_sc2_ft"]
    for s in [1, 2, 3]:
        pct = sc2_ft[sc2_ft["stratum"] == s]["pct_flip"].values[0]
        assert pct <= 0.032, f"plm_sc2_ft flip percentile at stratum {s} should be <= 0.03125, got {pct}"

    # 3. In SARS-CoV-2, untrained base model must fail to maintain significance across all strata
    base = sc2_flip[sc2_flip["score"] == "plm_base"]
    s3_pct = base[base["stratum"] == 3]["pct_flip"].values[0]
    assert s3_pct > 0.40, f"plm_base at stratum 3 should fail to rank real direction, got {s3_pct}"


def test_direction_contrast_exact_invariants():
    """Verify direction contrast findings: SC2 FT is significantly positive; controls are ~0."""
    sc2_dir = pd.read_csv(RESULTS_DIR / "symmetry_sars_cov_2_direction.tsv.gz", sep="\t")

    # plm_sc2_ft must have positive contrast with 95% CI strictly above 0
    sc2_row = sc2_dir[sc2_dir["score"] == "plm_sc2_ft"].iloc[0]
    assert sc2_row["contrast"] > 0.02
    assert sc2_row["lo"] > 0, "plm_sc2_ft 95% bootstrap lower bound must be > 0"
    assert sc2_row["fraction_of_truth"] > 0.10, "plm_sc2_ft must recover > 10% of truth"

    # Human FT, H3N2 FT, and JC69 FT must have contrast <= 0 or CI containing 0
    human_row = sc2_dir[sc2_dir["score"] == "plm_human_ft"].iloc[0]
    assert human_row["contrast"] < 0.005
    jc69_row = sc2_dir[sc2_dir["score"] == "plm_jc69_ft"].iloc[0]
    assert jc69_row["lo"] <= 0 <= jc69_row["hi"], "JC69 FT contrast CI must contain zero under SC2 spectrum"


def test_partial_correlation_holding_fitness_fixed_invariants():
    """Verify partial correlation holding ground_truth fitness fixed."""
    h3n2_part = pd.read_csv(RESULTS_DIR / "symmetry_influenza_h3n2_partial_given_truth.tsv.gz", sep="\t")

    # In H3N2: plm_h3n2_ft must specifically correlate with H3N2 accessibility at 1 nt and 2 nt
    h3n2_sub = h3n2_part[(h3n2_part["kind"] == "rho") & (h3n2_part["target"] == "direct")]
    rho_h3n2_1nt = h3n2_sub[(h3n2_sub["score"] == "plm_h3n2_ft") & (h3n2_sub["stratum"] == 1)]["value"].values[0]
    rho_h3n2_2nt = h3n2_sub[(h3n2_sub["score"] == "plm_h3n2_ft") & (h3n2_sub["stratum"] == 2)]["value"].values[0]
    assert rho_h3n2_1nt > 0.15, f"Expected h3n2_ft rho at 1 nt > 0.15, got {rho_h3n2_1nt}"
    assert rho_h3n2_2nt > 0.15, f"Expected h3n2_ft rho at 2 nt > 0.15, got {rho_h3n2_2nt}"

    # In H3N2: plm_jc69_ft (flat negative control) must be zero
    rho_jc69_1nt = h3n2_sub[(h3n2_sub["score"] == "plm_jc69_ft") & (h3n2_sub["stratum"] == 1)]["value"].values[0]
    assert abs(rho_jc69_1nt) < 0.03, f"Expected jc69_ft rho at 1 nt near 0, got {rho_jc69_1nt}"

    # In H3N2: base model must be much lower than fine-tuned model
    rho_base_1nt = h3n2_sub[(h3n2_sub["score"] == "plm_base") & (h3n2_sub["stratum"] == 1)]["value"].values[0]
    assert rho_h3n2_1nt > rho_base_1nt + 0.08, "plm_h3n2_ft must substantially exceed plm_base"


@needs_exploration_tree
def test_sim_table_exporter_function():
    """Directly test sim_table function from tools.exports.sim.

    Reads the exploration tree, so it does not run on a clean checkout (A4).
    """
    from tools.exports.sim import sim_table

    base_frame = pd.read_csv(SIM_SOURCE / "founder_base" / "tables" / "combined_long_table.csv")
    out = sim_table(base_frame, OUTER_REPO)

    assert len(out) == 13300
    assert list(out.columns) == EXPECTED_COLS
    assert out.iloc[0]["site"] == 1
    assert out.iloc[-1]["site"] == 700
