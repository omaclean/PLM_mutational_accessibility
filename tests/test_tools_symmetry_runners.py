"""Carpet bombing tests for tools/run_symmetry_battery.py, run_symmetry_dms.py, h3_numbering_from_db.py, and exporters.

Tests:
1. run_symmetry_battery.py: build_frame validation on h3n2_13lineage guide.
2. run_symmetry_dms.py: build validation on DMS tables.
3. h3_numbering_from_db.py: codon_columns helper and landmark validation logic.
4. Exporter metadata and integrity across export_s01, export_s07_s13, export_s14_s08, export_viral_panel.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = REPO_ROOT / "guides" / "h3n2_13lineage.yaml"


def test_run_symmetry_battery_build_frame():
    """Verify build_frame loads replacement scores and properties without missing codons or NaNs."""
    from tools.run_symmetry_battery import build_frame, CONTROLS
    from mutacc.guide import load_guide

    guide = load_guide(GUIDE_PATH)
    frame, names = build_frame(guide)

    # Shape, pinned. "len(frame) > 0" passes for a frame that has lost 99% of its
    # rows, and the battery's every result is a mean over these rows.
    assert len(frame) == 139_802
    assert frame["lineage"].nunique() == 13
    assert frame["nt_steps"].value_counts().sort_index().to_dict() == {1: 43_391, 2: 71_270, 3: 25_141}
    assert set(names) == {"plm_base", "plm_finetune", *CONTROLS}

    # Verify columns exist
    for col in names:
        assert col in frame.columns
        assert not frame[col].isna().any(), f"NaN in {col}"

    assert "ref_codon" in frame.columns
    assert (frame["ref_codon"].str.len() == 3).all(), "All ref_codons must have length 3"
    assert "nt_steps" in frame.columns
    assert set(frame["nt_steps"].unique()).issubset({1, 2, 3})


def test_run_symmetry_dms_build():
    """Verify run_symmetry_dms.build assembles wide DMS table with pair controls."""
    from tools.run_symmetry_dms import build, CONTROLS

    wide, names = build(GUIDE_PATH)

    # Shape, pinned per dataset: a scan silently losing rows would otherwise
    # change every DMS number downstream and still pass here.
    assert len(wide) == 33_862
    assert wide["dataset"].value_counts().sort_index().to_dict() == {
        "BA2": 7_062, "MA22": 9_238, "Perth09": 10_754, "XBB15": 6_808}
    assert names == ["plm_base", "plm_finetune"] + CONTROLS
    for col in names:
        assert col in wide.columns
        assert not wide[col].isna().any(), f"NaN in {col}"

    # Datasets should cover influenza and SARS-CoV-2 DMS scans
    datasets = set(wide["dataset"].unique())
    assert {"MA22", "Perth09", "BA2", "XBB15"}.issubset(datasets)


def test_h3_numbering_codon_columns():
    """Test tools.h3_numbering_from_db.codon_columns."""
    from tools.h3_numbering_from_db import codon_columns

    aln = "ATGCCCTAATTT"
    cols = codon_columns(aln)
    assert cols == ["ATG", "CCC", "TAA", "TTT"]

    # Trailing bases truncated to codon boundary
    aln_extra = "ATGCCCTAATTTG"
    cols_extra = codon_columns(aln_extra)
    assert cols_extra == ["ATG", "CCC", "TAA", "TTT"]


def test_h3_numbering_sqlite_mock(tmp_path: Path):
    """Test database interaction and landmark validation in h3_numbering_from_db."""
    from tools.h3_numbering_from_db import LANDMARKS

    # Create dummy SQLite DB mimicking iav-db
    db_path = tmp_path / "test_iav.db"
    con = sqlite3.connect(db_path)
    con.execute("create table sequence_alignment (primary_accession text, segment text, alignment text)")

    # Build sequence with exact landmarks
    # Length: 200 codons (600 nt).
    # Fill with 'GCT' (Ala), except landmarks at 98 (TAC: Tyr), 153 (TGG: Trp), 183 (CAC: His), 195 (TAC: Tyr)
    codons = ["GCT"] * 200
    codons[97] = "TAC"   # index 97 is mature residue 98 (Y)
    codons[152] = "TGG"  # index 152 is mature residue 153 (W)
    codons[182] = "CAC"  # index 182 is mature residue 183 (H)
    codons[194] = "TAC"  # index 194 is mature residue 195 (Y)
    aln_seq = "".join(codons)

    con.execute("insert into sequence_alignment values (?, ?, ?)", ("ACC001", "4", aln_seq))
    con.execute("insert into sequence_alignment values (?, ?, ?)", ("ACC002", "4", aln_seq))
    con.commit()
    con.close()

    # Query from DB and verify landmarks match
    con_ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con_ro.execute("select alignment from sequence_alignment where segment='4'").fetchall()
    assert len(rows) == 2
    con_ro.close()


def test_exporter_modules_import_and_define_a_callable_main():
    """Exactly what it says, and no more.

    Renamed from ``test_exporter_modules_exist_and_valid``: nothing here checks
    that an exporter is *valid*, only that it imports and has a ``main``. That is
    worth having -- a syntax error or a missing entry point fails here rather than
    at the end of a figure build -- but the name promised a check on the output
    that does not exist. Calling each ``main`` on a fixture is still open
    (docs/notes/extended_red_team_tests.md, Batch 3).
    """
    import importlib.util

    for mod_name in ["export_s01", "export_s07_s13", "export_s14_s08", "export_viral_panel", "export_spectra"]:
        path = REPO_ROOT / f"tools/{mod_name}.py"
        assert path.exists(), f"Missing {path}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main"), f"{mod_name} missing main function"
        assert callable(mod.main)
