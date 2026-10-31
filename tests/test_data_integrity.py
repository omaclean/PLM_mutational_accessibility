"""Red-team the INPUT data, whatever guide it is pointed at.

A figure drawn cleanly from corrupt input is a worse failure than a crash, so these
tests check the inputs directly: roots, tables, and the guide's claims about them.
Parametrised over every shipped guide.

Not yet ported (need GISAID-derived alignments that are not redistributable):
tip-date vs corpus-cutoff, per-lineage sequence uniqueness, alignment-column frame
checks. They are listed in docs/notes/PLAN.md Phase 6 and must be run on the authoring machine.
"""

import re

import numpy as np
import pandas as pd
import pytest

from mutacc.guide import REPO_ROOT, load_guide, read_fasta_headers, root_header_problems

GUIDES = sorted((REPO_ROOT / "guides").glob("*.yaml"))
STOPS = {"TAA", "TAG", "TGA"}


def read_single_sequence(path):
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    seq = "".join(ln for ln in lines if ln and not ln.startswith(">"))
    return seq.upper()


def pytest_generate_tests(metafunc):
    if "guide" in metafunc.fixturenames:
        metafunc.parametrize("guide", [load_guide(p) for p in GUIDES], ids=[p.stem for p in GUIDES])


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

def test_every_root_is_one_record_of_acgt(guide):
    for lin in guide.lineages:
        assert len(read_fasta_headers(lin.root)) == 1, lin.id
        seq = read_single_sequence(lin.root)
        assert seq and set(seq) <= set("ACGT"), f"{lin.id}: non-ACGT characters {set(seq) - set('ACGT')}"


def test_root_reading_frame_is_intact(guide):
    """Length divisible by 3 and no internal stop. A trailing stop is allowed."""
    for lin in guide.lineages:
        seq = read_single_sequence(lin.root)
        assert len(seq) % 3 == 0, f"{lin.id}: length {len(seq)} not divisible by 3"
        codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
        internal = [i for i, c in enumerate(codons[:-1]) if c in STOPS]
        assert not internal, f"{lin.id}: internal stop at codon(s) {internal[:5]}"


def test_roots_share_a_coordinate_system(guide):
    """Lineages are compared site by site, so every root must have the same length."""
    if guide.name in NOT_ALIGNED:
        pytest.skip(NOT_ALIGNED[guide.name])
    lengths = {lin.id: len(read_single_sequence(lin.root)) for lin in guide.lineages}
    assert len(set(lengths.values())) == 1, lengths


#: Guides whose roots are separate coding sequences with indels, not one alignment. Their
#: origin (database, accession) is not recorded in the archive, so the roots are `unverified`.
NOT_ALIGNED = {
    "sars_cov_2_spike": "spike coding sequences are 1,264 to 1,273 residues and are scored per lineage; "
                        "roots are unverified (no accession in the archive)",
}


def test_roots_are_not_all_identical(guide):
    if guide.n_lineages < 2:
        pytest.skip("single lineage")
    seqs = {read_single_sequence(lin.root) for lin in guide.lineages}
    assert len(seqs) > 1, "every root has the same sequence; the lineages would be indistinguishable"


def test_root_provenance_is_declared(guide):
    """No guide may lean on an unverified root without that being visible."""
    for lin in guide.lineages:
        assert lin.root_type in {"reconstructed", "observed", "unverified"}


# ---------------------------------------------------------------------------
# Provenance: the guide's claims agree with the headers
# ---------------------------------------------------------------------------

def test_headers_agree_with_guide(guide):
    problems = root_header_problems(guide)
    if guide.name in KNOWN_DEFECTS:
        # Pin the defect exactly, so fixing it forces this test to be updated
        # instead of the defect quietly disappearing from the record.
        assert sorted(p.split(":")[0] for p in problems) == sorted(KNOWN_DEFECTS[guide.name]), problems
    else:
        assert problems == [], problems


def test_clean_guide_has_no_unverified_roots(guide):
    if guide.name in KNOWN_DEFECTS or guide.name in NOT_ALIGNED:
        pytest.skip("guide carries documented provenance defects")
    assert guide.warnings == ()


#: Known provenance defects, from the 9 Oct audit (LINEAGE_VALIDITY.md section 4, Q3b).
#: The original five roots carry tip accessions on inferred sequences, and K.nt.fa's
#: header names lineage J.2.4.1. Resolve before any five-lineage number is quoted.
KNOWN_DEFECTS = {
    "h3n2_5lineage": ["J.2_int", "J.2.4", "K"],
}


def test_declared_reference_matches_root_headers(guide):
    """Where a header states ref=..., it must be the guide's reference."""
    for lin in guide.lineages:
        header = read_fasta_headers(lin.root)[0]
        m = re.search(r"\bref=([^|]+)", header)
        if m:
            assert m.group(1) == guide.reference, f"{lin.id}: header ref {m.group(1)!r} vs guide {guide.reference!r}"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def sweep_tables(guide):
    for name in guide.tables:
        if name.startswith("alpha_sweep"):
            yield name, guide.read_table(name)


def test_sweep_covers_exactly_the_guides_lineages_and_models(guide):
    for name, frame in sweep_tables(guide):
        sweep = frame.loc[frame["model_variant"] == "plm_alpha_sweep"]
        assert set(sweep["lineage"]) == set(guide.lineage_ids), name
        assert set(sweep["model"]) == {m.key for m in guide.models}, name


def test_sweep_grid_is_identical_for_every_lineage_and_model(guide):
    for name, frame in sweep_tables(guide):
        sweep = frame.loc[frame["model_variant"] == "plm_alpha_sweep"]
        grids = {tuple(sorted(g["alpha"].round(9))) for _, g in sweep.groupby(["lineage", "model"])}
        assert len(grids) == 1, f"{name}: alpha grid differs between lineage/model cells"
        assert not sweep.duplicated(["lineage", "model", "alpha"]).any(), name


def test_metrics_are_finite_and_in_range(guide):
    for name, frame in sweep_tables(guide):
        for col, lo, hi in [("site_logistic_auroc", 0, 1), ("mut_flat_global_spearman_r", -1, 1),
                            ("mut_flat_nonzero_pearson_r", -1, 1)]:
            if col not in frame:
                continue
            v = frame[col].to_numpy(float)
            assert np.isfinite(v).all(), f"{name}.{col} has non-finite values"
            assert v.min() >= lo and v.max() <= hi, f"{name}.{col} outside [{lo}, {hi}]"


def test_auroc_is_not_floored_at_chance(guide):
    """The old constrained fit clamped AUROC at exactly 0.5 wherever the score was
    anti-predictive. A column that still does that would draw a flat floor, not data."""
    for name, frame in sweep_tables(guide):
        sweep = frame.loc[frame["model_variant"] == "plm_alpha_sweep"]
        assert not (sweep["site_logistic_auroc"] == 0.5).any(), f"{name}: AUROC contains exact-0.5 floor values"
        negative = sweep.loc[sweep["alpha"] < 0, "site_logistic_auroc"]
        assert (negative < 0.5).any(), f"{name}: AUROC never drops below chance at alpha<0"


def test_clamped_columns_are_not_shipped(guide):
    for name, frame in sweep_tables(guide):
        assert not [c for c in frame.columns if c.endswith("_constrained_fit")], name


def test_prevalence_is_reported_per_lineage(guide):
    """Raw PR-AUC is not comparable across lineages with different prevalence; the
    per-lineage site counts must be present so that it can be reported, not pooled."""
    for name, frame in sweep_tables(guide):
        per = frame.groupby("lineage")["n_sites_used"].first()
        assert (per >= 1).all(), name
        assert len(per) == guide.n_lineages


def test_shipped_tables_have_manifest_rows(guide):
    manifest = pd.read_csv(guide.results_dir / "MANIFEST.tsv", sep="\t")
    shipped = set(manifest["file"])
    for name, rel in guide.tables.items():
        assert rel.name in shipped, f"{rel.name} is shipped without a MANIFEST row"
    # source paths must be relative to the exploration repo, never machine-specific
    assert not manifest["source"].str.startswith("/").any()


# ---------------------------------------------------------------------------
# Repo-wide hygiene
# ---------------------------------------------------------------------------

#: A test that *checks* the no-absolute-paths rule has to spell the needle it looks for,
#: so it would fail its own rule. Such a file opts out by carrying this marker on a line
#: of its own. The opt-out is honoured only under ``tests/``: it cannot silence the check
#: for a figure script, a tool or a guide, which is where a stray absolute path does harm.
ABSOLUTE_PATH_OPT_OUT = "allow-absolute-path-literals"


def test_no_absolute_paths_in_code_or_config():
    needles = ["/" + "home", "/" + "Users", "/" + "software"]
    bad = []
    for pattern in ("**/*.py", "**/*.yaml", "**/*.yml", "**/*.sh", "**/MANIFEST.tsv"):
        for path in REPO_ROOT.glob(pattern):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(errors="ignore")
            in_tests = "tests" in path.relative_to(REPO_ROOT).parts
            if in_tests and ABSOLUTE_PATH_OPT_OUT in text:
                continue
            bad += [f"{path.relative_to(REPO_ROOT)}: {n}" for n in needles if n in text]
    assert not bad, "\n".join(bad)


def test_the_opt_out_is_confined_to_the_test_tree():
    """A file outside ``tests/`` cannot buy its way out with the marker."""
    outside = [
        path.relative_to(REPO_ROOT)
        for pattern in ("**/*.py", "**/*.yaml", "**/*.yml", "**/*.sh")
        for path in REPO_ROOT.glob(pattern)
        if "__pycache__" not in path.parts
        and "tests" not in path.relative_to(REPO_ROOT).parts
        and ABSOLUTE_PATH_OPT_OUT in path.read_text(errors="ignore")
    ]
    assert not outside, f"the opt-out marker is only for tests/: {outside}"
