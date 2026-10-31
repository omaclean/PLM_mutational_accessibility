"""Carpet bombing tests for simulation report script, HTML report integrity, and run_symmetry_sim.py.

Tests:
1. tools/plot_simulation_report.py helper functions.
2. clean_repo/reports/simulation_control_update_2026-10-13.html structure, validity, and assets integrity.
3. tools/run_symmetry_sim.py: pair_controls function, build_unit, and control score properties.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"
HTML_REPORT = REPORTS_DIR / "simulation_control_update_2026-10-13.html"


def test_html_report_and_embedded_assets_exist():
    """Verify HTML report exists and every referenced asset is a valid, non-empty PNG."""
    assert HTML_REPORT.exists(), "simulation_control_update_2026-10-13.html must exist"
    html_text = HTML_REPORT.read_text(encoding="utf-8")

    # Basic HTML structure assertions
    assert "<!DOCTYPE html>" in html_text
    assert "<title>Simulation Control & Progress Update Report" in html_text
    assert "</html>" in html_text

    # Extract all <img src="assets/..." tags
    img_srcs = re.findall(r'<img\s+src="([^"]+)"', html_text)
    assert len(img_srcs) >= 8, f"Expected at least 8 image tags, found {len(img_srcs)}"

    for src in img_srcs:
        # Resolve path relative to reports directory
        asset_path = REPORTS_DIR / src
        assert asset_path.exists(), f"Image asset {src} does not exist at {asset_path}"
        assert asset_path.stat().st_size > 1000, f"Image asset {src} is suspiciously small ({asset_path.stat().st_size} bytes)"

        # Verify PNG magic bytes
        with open(asset_path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", f"File {src} is not a valid PNG"


def test_pair_controls_on_real_guide():
    """Verify tools.run_symmetry_sim.pair_controls generates exact pair controls from shipped data."""
    from tools.run_symmetry_sim import pair_controls, CONTROLS

    guide_path = REPO_ROOT / "guides" / "h3n2_13lineage.yaml"
    assert guide_path.exists()

    controls_df = pair_controls(guide_path)

    # 20 x 19 = 380 ordered non-synonymous pairs
    assert len(controls_df) == 380, f"Expected 380 pairs, got {len(controls_df)}"
    assert set(controls_df.columns) == {"ref_aa", "aa", *CONTROLS}

    # No NaNs
    assert not controls_df.isna().any().any(), "Controls must contain no NaNs"

    # Check symmetry of distance metrics: Grantham distance must be symmetric
    merged = controls_df.merge(
        controls_df,
        left_on=["ref_aa", "aa"],
        right_on=["aa", "ref_aa"],
        suffixes=("_fwd", "_rev")
    )
    assert np.allclose(merged["neg_grantham_fwd"], merged["neg_grantham_rev"]), "Grantham distance must be symmetric"


def test_build_unit_sim():
    """Test build_unit on simulation replacement table."""
    from tools.run_symmetry_sim import build_unit, pair_controls, MODELS, PROFILES, SELECTION, CONTROLS

    guide_path = REPO_ROOT / "guides" / "h3n2_13lineage.yaml"
    table_path = REPO_ROOT / "results" / "sim_v3_star90" / "replacement_scores.tsv.gz"
    assert table_path.exists()

    controls = pair_controls(guide_path)
    units, frame = build_unit(table_path, controls)

    assert len(units) == 1
    unit = units[0]
    assert unit.name == "sim"
    assert len(np.unique(unit.site)) == 700

    # Ensure all required score columns exist in unit
    for name in MODELS + PROFILES + SELECTION + CONTROLS:
        assert name in unit.scores
        assert len(unit.scores[name]) == 13300


def test_plot_simulation_report_count_mutations(tmp_path: Path):
    """Test tools.plot_simulation_report.count_mutations helper logic."""
    from tools.plot_simulation_report import count_mutations, SUB_CLASSES

    # Create dummy founder FASTA (length 6: ATG AAA)
    founder_path = tmp_path / "founder.fna"
    founder_path.write_text(">founder\nATGAAA\n")

    # Create dummy descendant FASTA with planted mutations:
    # Seq 1: CTG AAA (A->C at pos 0)
    # Seq 2: TTG AAA (A->T at pos 0)
    # Seq 3: ATG AAG (A->G at pos 5)
    desc_path = tmp_path / "desc.fna"
    desc_path.write_text(">seq1\nCTGAAA\n>seq2\nTTGAAA\n>seq3\nATGAAG\n")

    pcts, total = count_mutations(founder_path, desc_path)

    assert total == 3
    assert np.isclose(pcts["A->C"], 100.0 / 3.0)
    assert np.isclose(pcts["A->T"], 100.0 / 3.0)
    assert np.isclose(pcts["A->G"], 100.0 / 3.0)
    assert pcts["C->A"] == 0.0
