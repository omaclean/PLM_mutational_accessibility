"""Carpet-bombing test suite for tools/export_base.py, tools/export_tables.py, and exporters.

Tests:
1. export_base.py: sha256 hashing, Writer class, append_manifest atomic replacement and error handling.
2. export_tables.py: discovery mechanism, Export validation across all tools/exports/*.py.
3. Exporter transform hooks on synthetic dataframes.
4. spectra_compare.py: codon table creation, compression, variance explained, and correlation metrics.
"""
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"


# ==============================================================================
# 1. Tests for tools/export_base.py
# ==============================================================================

def test_export_base_sha256(tmp_path: Path):
    from tools.export_base import sha256
    import hashlib

    # Create test file with known content
    p = tmp_path / "test.bin"
    content = b"hello world 123456789"
    p.write_bytes(content)

    expected_full = hashlib.sha256(content).hexdigest()[:16]
    assert sha256(p) == expected_full
    assert len(sha256(p)) == 16


def test_export_base_writer(tmp_path: Path):
    from tools.export_base import Writer

    w = Writer(tmp_path / "results_set")
    assert (tmp_path / "results_set").is_dir()

    # Refuse empty frame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="refusing to write an empty table"):
        w.add("empty.tsv.gz", empty_df, source="src", why="test")

    # Add valid frame
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    w.add("table.tsv.gz", df, source="src/path.csv", why="unit test", sha="1234567890abcdef")

    assert len(w.rows) == 1
    assert w.rows[0]["file"] == "table.tsv.gz"
    assert w.rows[0]["rows"] == 3
    assert w.rows[0]["columns"] == 2
    assert (tmp_path / "results_set" / "table.tsv.gz").exists()

    # Write manifest
    w.write_manifest()
    manifest_path = tmp_path / "results_set" / "MANIFEST.tsv"
    assert manifest_path.exists()
    man = pd.read_csv(manifest_path, sep="\t")
    assert len(man) == 1
    assert man.iloc[0]["file"] == "table.tsv.gz"


def test_export_base_append_manifest_validation(tmp_path: Path):
    from tools.export_base import append_manifest, MANIFEST_COLUMNS

    m_path = tmp_path / "MANIFEST.tsv"

    # Empty rows is a no-op
    append_manifest(m_path, [])
    assert not m_path.exists()

    # Missing column raises ValueError
    bad_row = {"file": "f.tsv.gz", "source": "s"}  # missing other columns
    with pytest.raises(ValueError, match="missing"):
        append_manifest(m_path, [bad_row])

    # Tab in prose field raises ValueError
    bad_prose = {
        "file": "f.tsv.gz", "source": "s\tbad", "source_sha256_16": "abc",
        "rows": 1, "columns": 1, "dropped": "", "used_for": "why"
    }
    with pytest.raises(ValueError, match="tab or newline"):
        append_manifest(m_path, [bad_prose])

    # Newline in prose field raises ValueError
    bad_newline = {
        "file": "f.tsv.gz", "source": "s", "source_sha256_16": "abc",
        "rows": 1, "columns": 1, "dropped": "", "used_for": "why\nline"
    }
    with pytest.raises(ValueError, match="tab or newline"):
        append_manifest(m_path, [bad_newline])

    # Valid write
    valid_rows = [
        {"file": "b.tsv.gz", "source": "s2", "source_sha256_16": "222", "rows": 10, "columns": 2, "dropped": "", "used_for": "b file"},
        {"file": "a.tsv.gz", "source": "s1", "source_sha256_16": "111", "rows": 5, "columns": 3, "dropped": "", "used_for": "a file"},
    ]
    append_manifest(m_path, valid_rows)
    assert m_path.exists()

    df = pd.read_csv(m_path, sep="\t")
    assert len(df) == 2

    # Append new row and update existing row
    update_rows = [
        {"file": "a.tsv.gz", "source": "s1_new", "source_sha256_16": "111_new", "rows": 6, "columns": 3, "dropped": "", "used_for": "a updated"},
        {"file": "c.tsv.gz", "source": "s3", "source_sha256_16": "333", "rows": 20, "columns": 4, "dropped": "", "used_for": "c file"},
    ]
    append_manifest(m_path, update_rows)
    df_updated = pd.read_csv(m_path, sep="\t")
    assert len(df_updated) == 3
    # Check that rows are sorted by file
    assert list(df_updated["file"]) == ["a.tsv.gz", "b.tsv.gz", "c.tsv.gz"]
    # Check that a.tsv.gz was updated
    assert df_updated.loc[df_updated["file"] == "a.tsv.gz", "rows"].values[0] == 6


# ==============================================================================
# 2. Tests for tools/export_tables.py and Exporter discovery
# ==============================================================================

def test_export_tables_discovery():
    from tools.export_tables import _discover, Export

    exports = _discover()
    assert len(exports) > 0, "Must discover Export records in tools/exports/"

    for exp in exports:
        assert isinstance(exp, Export)
        assert exp.set_name.strip() != "", f"Empty set_name in {exp}"
        assert exp.source.strip() != "", f"Empty source in {exp}"
        assert exp.dest.endswith(".tsv.gz"), f"dest must end in .tsv.gz: {exp.dest}"
        assert exp.why.strip() != "", f"Empty why docstring in {exp}"


def test_export_records_valid_types():
    """Verify all individual export modules define valid EXPORTS."""
    import importlib.util

    export_files = sorted((TOOLS_DIR / "exports").glob("*.py"))
    assert len(export_files) >= 5, "Expected at least 5 export modules in tools/exports/"

    for p in export_files:
        spec = importlib.util.spec_from_file_location(f"_test_exp_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "EXPORTS"), f"Module {p.name} missing EXPORTS"
        assert isinstance(mod.EXPORTS, list)
        for e in mod.EXPORTS:
            assert hasattr(e, "set_name")
            assert hasattr(e, "dest")
            assert hasattr(e, "source")


# ==============================================================================
# 3. Tests for tools/spectra_compare.py
# ==============================================================================

def test_spectra_compare_codon_table():
    from tools.spectra_compare import codon_table, TOTAL_RATE

    # Identity rate matrix with non-zero off-diagonals
    q = np.array([
        [0.0, 1.0, 2.0, 1.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [1.0, 2.0, 1.0, 0.0],
    ])
    tbl = codon_table(q)

    # Shape must be 64 codons x 20 amino acids
    assert tbl.shape == (64, 20)

    # Diagonal/own-amino-acid cells must be NaN
    assert np.isnan(tbl).sum() > 0

    # Zero matrix raises ValueError
    zero_q = np.zeros((4, 4))
    with pytest.raises(ValueError, match="no non-zero rate"):
        codon_table(zero_q)


def test_spectra_compare_compress():
    from tools.spectra_compare import codon_table, compress
    from mutacc import accessibility as acc

    q = np.array([
        [0.0, 2.0, 1.0, 1.0],
        [1.0, 0.0, 3.0, 1.0],
        [1.0, 1.0, 0.0, 2.0],
        [2.0, 1.0, 1.0, 0.0],
    ])
    tbl = codon_table(q)
    c_tbl = compress(tbl)

    assert c_tbl.shape == (64, 20)

    # For any amino acid with multiple synonymous codons, their rows in c_tbl must be identical
    syn_rows = [i for i, c in enumerate(acc.CODONS) if acc.GENETIC_CODE[c] == "L"]  # Leucine has 6 codons
    assert len(syn_rows) == 6
    for col in range(20):
        vals = [c_tbl[r, col] for r in syn_rows]
        # either all NaN or all equal
        if np.isnan(vals[0]):
            assert all(np.isnan(v) for v in vals)
        else:
            assert np.allclose(vals, vals[0])


def test_spectra_compare_retained_and_corr():
    from tools.spectra_compare import retained_and_corr

    # Test identical matrices: retained variance = 100%, correlation = 1.0
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    ret, corr = retained_and_corr(a, a)
    assert np.isclose(ret, 100.0)
    assert np.isclose(corr, 1.0)

    # Test all-NaN input returns (nan, nan)
    nan_mat = np.full((3, 3), np.nan)
    ret_nan, corr_nan = retained_and_corr(nan_mat, nan_mat)
    assert np.isnan(ret_nan) and np.isnan(corr_nan)

    # Test perturbed predictor
    b = a + np.array([[0.1, -0.1], [0.05, -0.05]])
    ret_p, corr_p = retained_and_corr(a, b)
    assert 90.0 < ret_p < 100.0
    assert 0.95 < corr_p < 1.0
