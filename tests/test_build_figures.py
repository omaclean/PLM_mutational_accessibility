"""``tools/build_figures.py``: the one command that rebuilds every figure.

A runner that silently skips a figure, or builds a composite from a stale panel, is worse
than no runner: it produces a confident-looking ``output/`` that no longer matches the code.
So these tests pin, in this order:

1. **Coverage** -- every figure script is a step, and the steps account for every shipped
   figure, both ways. A new ``make_*.py`` that nobody added to the runner fails here.
2. **Order** -- a composite waits for the panels it places, derived from the layout YAML.
3. **The comparison** -- the check that makes a rebuild mean something. Each test of it has a
   paired tampered input that must be caught; a comparison that never fails proves nothing.
4. **Execution** -- failures propagate and block dependants, on throwaway scripts.
5. **End to end** -- a real, fast subset is rebuilt into a scratch directory and must
   reproduce the shipped values tables and captions.
"""

from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mutacc.guide import REPO_ROOT
from mutacc.io import caption_label
from mutacc.numbering import load_numbering
from tools import build_figures as bf

OUTPUT = REPO_ROOT / "output"

#: Figure scripts that are not figures: ``make_fig02.py`` was retired as a figure on
#: 14 Oct 2026 and survives only as the home of ``PROPERTIES``/``stratum_x``/``spread``,
#: which ``make_sD_partial_correlation.py`` and ``make_fig02d.py`` import.
NOT_STEPS = {"make_fig02.py"}


def _shipped_pdfs() -> set[str]:
    retired = bf.retired_stems()
    return {p.name for p in OUTPUT.glob("*.pdf") if not p.name.startswith(retired)}


# ---------------------------------------------------------------------------
# 1. Coverage
# ---------------------------------------------------------------------------

def test_every_figure_script_is_a_step_and_every_step_script_exists():
    on_disk = {p.name for p in (REPO_ROOT / "figures").glob("make_*.py")} - NOT_STEPS
    used = {Path(s.script).name for s in bf.STEPS}
    assert on_disk - used == set(), f"figure scripts the runner never calls: {sorted(on_disk - used)}"
    for step in bf.STEPS:
        assert (REPO_ROOT / step.script).is_file(), step.name


def test_step_names_are_unique_and_every_dependency_names_a_step():
    names = [s.name for s in bf.STEPS]
    assert len(names) == len(set(names))
    for step in bf.STEPS:
        assert set(step.after) <= set(names), step.name


def test_every_path_argument_of_every_step_exists():
    """A guide or results directory renamed without the runner failing is found here,
    not an hour into a build."""
    for step in bf.STEPS:
        for arg in step.args:
            if arg.startswith("--") or not (arg.endswith((".yaml", ".gz")) or arg.startswith("results/")):
                continue
            assert (REPO_ROOT / arg).exists(), f"{step.name}: {arg}"


def test_the_steps_account_for_every_shipped_figure_pdf_and_nothing_more():
    claimed = {f for s in bf.STEPS if not s.retired for f in s.produces if f.endswith(".pdf")}
    shipped = _shipped_pdfs()
    assert shipped - claimed == set(), f"shipped, but no step says it builds it: {sorted(shipped - claimed)}"
    assert claimed - shipped == set(), f"a step claims a figure that is not shipped: {sorted(claimed - shipped)}"


def test_every_claimed_output_belongs_to_a_figure_in_the_numbering_map():
    numbering = load_numbering()
    for step in bf.STEPS:
        for f in step.produces:
            # A supplementary table is lettered like a figure since 20 Oct 2026 and is in the
            # map, so the extension is all that has to come off; the exemption this used to
            # make for a "table_" prefix would now hide a table that nothing numbers.
            stem = f.rsplit(".", 1)[0]
            assert stem in numbering or any(stem.startswith(k + "_") for k in numbering.entries), f


def test_retired_steps_are_exactly_the_ones_the_map_retires():
    retired = set(bf.retired_stems())
    for step in bf.STEPS:
        stems = {f.removesuffix(".pdf") for f in step.produces if f.endswith(".pdf")}
        assert step.retired == bool(stems & retired), step.name


# ---------------------------------------------------------------------------
# 2. Order
# ---------------------------------------------------------------------------

def test_composite_inputs_reads_the_output_panels_from_a_layout():
    assert bf.composite_inputs(REPO_ROOT / "guides/composites/fig02.yaml") == [
        "fig02a_scatter_panel.pdf", "fig02b_scatter_panel.pdf", "fig02c_epoch_panel.pdf",
        "fig02d_strata_panel.pdf"]
    assert bf.composite_inputs(REPO_ROOT / "guides/composites/fig01.yaml") == [
        "fig01b_ranked_model.pdf", "fig01c_ranked_accessibility.pdf",
        "fig01d_codon_accessibility.pdf"]


def test_every_output_panel_a_composite_places_has_exactly_one_step_that_builds_it():
    for layout in (REPO_ROOT / "guides" / "composites").glob("*.yaml"):
        for panel in bf.composite_inputs(layout):
            assert len(bf.producers_of(panel)) == 1, f"{layout.name}: {panel}"


def test_a_composite_comes_after_its_panels_and_the_table_after_figure_4():
    order = [s.name for s in bf.plan(None, include_retired=True)]
    assert order.index("fig01d") < order.index("fig01")
    assert order.index("fig02d") < order.index("fig02")
    assert order.index("fig02ab") < order.index("fig02")
    assert order.index("fig02c") < order.index("fig02")
    assert order.index("fig04") < order.index("table_dms_alpha")
    assert sorted(order) == sorted(s.name for s in bf.STEPS)


def test_asking_for_a_composite_pulls_in_its_panel():
    assert [s.name for s in bf.plan({"fig02"}, False)] == ["fig02ab", "fig02c", "fig02d", "fig02"]


def test_retired_figures_are_skipped_unless_asked_for():
    assert "s14" not in {s.name for s in bf.plan(None, include_retired=False)}
    assert "s14" in {s.name for s in bf.plan(None, include_retired=True)}


def test_an_unknown_step_name_is_refused():
    with pytest.raises(SystemExit, match="unknown step"):
        bf.plan({"s99"}, False)


def test_a_dependency_cycle_is_refused():
    a = bf.Step("a", "x.py", after=("b",))
    b = bf.Step("b", "x.py", after=("a",))
    with pytest.raises(SystemExit, match="cycle"):
        bf.plan(None, False, steps=(a, b))


def test_list_prints_every_default_step(capsys):
    assert bf.main(["--list"]) == 0
    printed = {line.split()[0] for line in capsys.readouterr().out.splitlines() if line.strip()}
    assert printed == {s.name for s in bf.STEPS if not s.retired}


# ---------------------------------------------------------------------------
# Composite layouts and the command line
# ---------------------------------------------------------------------------

def test_a_staging_root_points_every_built_directory_at_this_build_and_data_at_the_repo(tmp_path):
    out = tmp_path / "build"
    out.mkdir()
    root = bf.staging_root(out)

    assert (root / "output").resolve() == out.resolve()
    assert (root / "data").resolve() == (REPO_ROOT / "data").resolve()
    # a scratch build's composite must place that build's panels, not the repository's
    assert (root / "build" / "panels").resolve() == bf.panels_dir(out).resolve()
    # every panel path in every shipped layout resolves through it to a real file or a file
    # a step builds, using only repo-relative paths, which is all panelise accepts
    for layout in (REPO_ROOT / "guides" / "composites").glob("*.yaml"):
        for panel in yaml.safe_load(layout.read_text())["panels"]:
            rel = Path(panel["file"])
            assert not rel.is_absolute() and ".." not in rel.parts
            if rel.parts[0] == "data":
                assert (root / rel).is_file(), panel["file"]
            else:
                assert any(rel.as_posix().startswith(d + "/") for d in bf.BUILT_PANEL_DIRS), \
                    panel["file"]
                assert len(bf.producers_of(rel.name)) == 1, panel["file"]


def test_a_composite_panel_is_never_written_into_the_output_directory(tmp_path):
    """Panels are intermediates. ``output/`` is for finished figures, so they go elsewhere."""
    assert bf.panels_dir(bf.SHIPPED) == bf.SHIPPED_PANELS
    assert bf.SHIPPED not in bf.SHIPPED_PANELS.parents
    scratch_panels = bf.panels_dir(tmp_path)
    assert scratch_panels.parent == tmp_path and scratch_panels.name.startswith("_")

    panel_steps = [s for s in bf.STEPS if s.panels]
    assert {s.name for s in panel_steps} == {"fig01bc", "fig01d", "fig02ab", "fig02c", "fig02d"}
    for step in panel_steps:
        assert not step.produces, f"{step.name}: a panel step writes no figure into output/"
        cmd = bf.command(step, bf.SHIPPED, "python")
        assert cmd[cmd.index("--panels-out") + 1] == str(bf.SHIPPED_PANELS)


def test_the_shipped_output_directory_holds_no_panel_renders():
    """The published tree is composites and standalone figures; no half-figures beside them."""
    names = {name for s in bf.STEPS for name in s.panels}
    stems = {Path(n).stem for n in names}
    strays = [p.name for p in bf.SHIPPED.iterdir()
              if p.stem in stems and p.suffix in {".pdf", ".png"}]
    assert strays == [], f"panel renders left in output/: {strays}"


def test_staging_twice_is_harmless_and_follows_a_changed_target(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    assert bf.staging_root(first) == bf.staging_root(first)
    assert (bf.staging_root(first) / "output").resolve() == first.resolve()
    assert (bf.staging_root(second) / "output").resolve() == second.resolve()


def test_a_build_into_the_shipped_directory_is_not_rooted_elsewhere(tmp_path):
    cmd = bf.command(bf.BY_NAME["fig02"], bf.SHIPPED, "python")
    assert "--root" not in cmd and cmd[cmd.index("--layout") + 1] == "guides/composites/fig02.yaml"


def test_a_scratch_build_roots_the_composite_in_its_own_staging_tree(tmp_path):
    cmd = bf.command(bf.BY_NAME["fig02"], tmp_path, "python")
    root = Path(cmd[cmd.index("--root") + 1])
    assert (root / "output").resolve() == tmp_path.resolve()
    assert cmd[cmd.index("--layout") + 1] == "guides/composites/fig02.yaml", "the layout itself is not rewritten"


def test_the_command_names_the_output_directory_the_way_each_script_expects(tmp_path):
    cmd = bf.command(bf.BY_NAME["sE"], tmp_path, "python")
    assert cmd[-2:] == ["--out", str(tmp_path)] and "--root" not in cmd
    table = bf.command(bf.BY_NAME["table_dms_alpha"], tmp_path, "python")
    assert table[-2:] == [str(REPO_ROOT), str(tmp_path)] and "--out" not in table


def test_the_table_exporter_reads_and_writes_the_directory_it_is_given(tmp_path):
    """Built into a scratch directory it must not read Figure 4's values from ``output/``."""
    import subprocess

    shipped_table = OUTPUT / "sY_dms_alpha_optima.tsv"
    before = shipped_table.read_bytes()
    (tmp_path / "fig04_alpha_vs_dms_values.tsv").write_text("not the real columns\n")
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "tools/export_sY_dms_alpha_optima.py"),
                           str(REPO_ROOT), str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode != 0, "it accepted a values table with no data and wrote an empty table"
    assert "is not a DMS values table" in proc.stderr
    assert not (tmp_path / "sY_dms_alpha_optima.tsv").exists()
    assert shipped_table.read_bytes() == before, "it wrote into output/"


# ---------------------------------------------------------------------------
# 3. The comparison
# ---------------------------------------------------------------------------

def _tsv(path: Path, frame: dict) -> Path:
    pd.DataFrame(frame).to_csv(path, sep="\t", index=False, float_format="%.6g")
    return path


BASE = {"name": ["a", "b", "c"], "rho": [0.123456, -0.5, 1e-9], "n": [10, 20, 30], "note": ["", "x", ""]}


def test_identical_values_tables_agree(tmp_path):
    a, b = _tsv(tmp_path / "a.tsv", BASE), _tsv(tmp_path / "b.tsv", BASE)
    assert bf.compare_values(a, b) == []


def test_one_unit_in_the_sixth_digit_is_not_a_difference(tmp_path):
    """``%.6g`` rounding: two values 1e-12 apart can print as neighbours."""
    a = _tsv(tmp_path / "a.tsv", BASE)
    b = _tsv(tmp_path / "b.tsv", {**BASE, "rho": [0.123457, -0.5, 1e-9]})
    assert bf.compare_values(a, b) == []


@pytest.mark.parametrize("change, expect", [
    ({"rho": [0.1235, -0.5, 1e-9]}, "rho"),                 # 4th-digit drift in a published number
    ({"rho": [0.123456, 0.5, 1e-9]}, "rho"),                # sign flipped
    ({"n": [10, 20, 31]}, "n"),                             # a count moved by one
    ({"name": ["a", "b", "z"]}, "name"),                    # a label changed
    ({"note": ["", "y", ""]}, "note"),
])
def test_a_tampered_cell_is_caught(tmp_path, change, expect):
    a = _tsv(tmp_path / "a.tsv", {**BASE, **change})
    b = _tsv(tmp_path / "b.tsv", BASE)
    problems = bf.compare_values(a, b)
    assert problems and expect in problems[0]


def test_a_number_replaced_by_text_is_caught(tmp_path):
    a = _tsv(tmp_path / "a.tsv", {**BASE, "rho": ["nan?", -0.5, 1e-9]})
    b = _tsv(tmp_path / "b.tsv", BASE)
    assert bf.compare_values(a, b)


def test_a_renamed_column_or_a_lost_row_is_caught(tmp_path):
    base = _tsv(tmp_path / "base.tsv", BASE)
    renamed = _tsv(tmp_path / "r.tsv", {("rho2" if k == "rho" else k): v for k, v in BASE.items()})
    short = _tsv(tmp_path / "s.tsv", {k: v[:2] for k, v in BASE.items()})
    assert "columns differ" in bf.compare_values(renamed, base)[0]
    assert "row count" in bf.compare_values(short, base)[0]


def test_two_missing_values_agree_and_one_missing_does_not(tmp_path):
    nan = {"x": [1.0, float("nan")]}
    a, b = _tsv(tmp_path / "a.tsv", nan), _tsv(tmp_path / "b.tsv", nan)
    c = _tsv(tmp_path / "c.tsv", {"x": [1.0, 2.0]})
    assert bf.compare_values(a, b) == []
    assert bf.compare_values(a, c)


def test_a_caption_that_gained_a_word_is_caught_and_an_identical_one_is_not(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text("**Figure S1.** one\ntwo\n")
    b.write_text("**Figure S1.** one\ntwo\n")
    assert bf.compare_text(a, b) == []
    b.write_text("**Figure S1.** one\ntwo!\n")
    assert "line 2, column 4" in bf.compare_text(a, b)[0]
    b.write_text("**Figure S1.** one\ntwo\nthree\n")
    assert "length differs" in bf.compare_text(a, b)[0]


def _two_dirs(tmp_path):
    built, shipped = tmp_path / "built", tmp_path / "shipped"
    built.mkdir(), shipped.mkdir()
    for d in (built, shipped):
        _tsv(d / "figx_values.tsv", BASE)
        (d / "figx_caption.md").write_text("**Figure 1.** text\n")
        (d / "figx.pdf").write_bytes(b"%PDF differs per build")
    (shipped / "figx.pdf").write_bytes(b"%PDF a different render")
    return built, shipped


def test_a_clean_build_has_no_problems_and_pdf_bytes_are_not_compared(tmp_path):
    built, shipped = _two_dirs(tmp_path)
    report = bf.check_build(built, shipped)
    assert set(report) == {"figx_values.tsv", "figx_caption.md"}
    assert not any(report.values())


def test_a_drifted_table_and_a_drifted_caption_are_both_reported(tmp_path):
    built, shipped = _two_dirs(tmp_path)
    _tsv(built / "figx_values.tsv", {**BASE, "n": [10, 20, 99]})
    (built / "figx_caption.md").write_text("**Figure 2.** text\n")
    report = bf.check_build(built, shipped)
    assert report["figx_values.tsv"] and report["figx_caption.md"]


def test_a_shipped_file_the_build_did_not_write_is_reported_only_on_a_full_build(tmp_path):
    built, shipped = _two_dirs(tmp_path)
    _tsv(shipped / "ghost_values.tsv", BASE)
    assert "<not rebuilt>" in bf.check_build(built, shipped, full=True)["ghost_values.tsv"][0]
    assert "ghost_values.tsv" not in bf.check_build(built, shipped, full=False)
    assert "ghost_values.tsv" not in bf.check_build(built, shipped, full=True, skipped_stems=("ghost",))


def test_a_built_file_with_no_shipped_twin_is_reported(tmp_path):
    built, shipped = _two_dirs(tmp_path)
    _tsv(built / "new_values.tsv", BASE)
    assert "no shipped copy" in bf.check_build(built, shipped)["new_values.tsv"][0]


def test_supplementary_tables_are_compared_too(tmp_path):
    built, shipped = _two_dirs(tmp_path)
    _tsv(built / "table_s99_x.tsv", BASE)
    _tsv(shipped / "table_s99_x.tsv", {**BASE, "n": [1, 2, 3]})
    assert bf.check_build(built, shipped)["table_s99_x.tsv"]


# ---------------------------------------------------------------------------
# 4. Execution, on throwaway scripts
# ---------------------------------------------------------------------------

def _script(path: Path, body: str) -> str:
    path.write_text(textwrap.dedent(
        f"""
        import argparse, pathlib
        ap = argparse.ArgumentParser(); ap.add_argument("--out", type=pathlib.Path)
        a = ap.parse_args()
        a.out.mkdir(parents=True, exist_ok=True)
        {body}
        """))
    return str(path)


def test_a_failing_step_is_reported_with_its_error_and_blocks_what_depends_on_it(tmp_path):
    good = bf.Step("good", _script(tmp_path / "good.py", '(a.out / "good.txt").write_text("x")'),
                   produces=("good.txt",))
    bad = bf.Step("bad", _script(tmp_path / "bad.py", 'raise SystemExit("boom")'))
    child = bf.Step("child", _script(tmp_path / "child.py", '(a.out / "child.txt").write_text("x")'),
                    after=("bad",), produces=("child.txt",))
    out = tmp_path / "out"
    results = {r.step.name: r for r in bf.execute([good, bad, child], out, jobs=2, python=sys.executable)}

    assert results["good"].ok and (out / "good.txt").exists()
    assert not results["bad"].ok and "boom" in results["bad"].detail
    assert not results["child"].ok and "bad failed" in results["child"].detail
    assert not (out / "child.txt").exists(), "a dependant ran after its dependency failed"


def test_a_step_that_exits_cleanly_but_writes_nothing_is_a_failure(tmp_path):
    lazy = bf.Step("lazy", _script(tmp_path / "lazy.py", "pass"), produces=("never_written.pdf",))
    (res,) = bf.execute([lazy], tmp_path / "out", jobs=1, python=sys.executable)
    assert not res.ok and res.missing == ["never_written.pdf"]


def test_the_steps_run_one_blas_thread_each(tmp_path):
    probe = bf.Step("probe", _script(tmp_path / "probe.py",
                                     'import os; (a.out / "t.txt").write_text(os.environ["OPENBLAS_NUM_THREADS"] '
                                     '+ os.environ["OMP_NUM_THREADS"] + os.environ["MPLBACKEND"])'),
                    produces=("t.txt",))
    bf.execute([probe], tmp_path / "out", jobs=1, python=sys.executable)
    assert (tmp_path / "out" / "t.txt").read_text() == "11Agg"


def test_main_exits_nonzero_when_a_step_fails(tmp_path, monkeypatch):
    bad = bf.Step("bad", _script(tmp_path / "bad.py", "raise SystemExit(3)"))
    monkeypatch.setattr(bf, "STEPS", (bad,))
    monkeypatch.setattr(bf, "BY_NAME", {"bad": bad})
    monkeypatch.setattr(bf, "plan", lambda selected, include_retired, steps=(bad,): [bad])
    assert bf.main(["--out", str(tmp_path / "o"), "--no-check", "--jobs", "1"]) == 1


# ---------------------------------------------------------------------------
# 5. End to end, on a fast real subset
# ---------------------------------------------------------------------------

FAST = ["fig01", "sC", "sE"]       # fig01 pulls in fig01d: a real composite built from a scratch panel


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("build")
    assert bf.main(["--out", str(out), "--only", *FAST, "--jobs", "2"]) == 0
    return out


def test_a_real_partial_build_reproduces_the_shipped_values_and_captions(built):
    report = bf.check_build(built, OUTPUT, full=False)
    assert len(report) >= 6, sorted(report)          # the check really looked at files
    assert {k: v for k, v in report.items() if v} == {}


def test_the_composite_was_built_from_the_scratch_panel_not_the_shipped_one(built):
    # the panel goes beside the build, not into it: the output directory is finished figures
    assert not (built / "fig01d_codon_accessibility.pdf").exists()
    assert (bf.panels_dir(built) / "fig01d_codon_accessibility.pdf").is_file()
    assert (built / "fig01_composite.pdf").is_file()
    assert (built / "fig01_composite.png").stat().st_size > 20_000


def test_a_partial_build_keeps_its_panels_but_a_whole_one_clears_them(built, tmp_path):
    """Panels are intermediates, so a finished build deletes them -- but only when every
    composite that places them ran. ``--only fig02ab`` is someone asking to see a panel."""
    assert (bf.panels_dir(built) / "fig01d_codon_accessibility.pdf").is_file(), \
        "fig02's composite did not run, so fig01's panel must survive"

    whole = tmp_path / "whole"
    assert bf.main(["--out", str(whole), "--only", "fig01", "fig02", "--jobs", "2"]) == 0
    assert (whole / "fig01_composite.pdf").is_file() and (whole / "fig02_composite.pdf").is_file()
    assert not bf.panels_dir(whole).exists(), "every consumer ran, so the panels should be gone"


def test_a_real_build_that_drifted_would_be_caught(built, tmp_path):
    shipped = tmp_path / "shipped"
    shutil.copytree(OUTPUT, shipped, ignore=shutil.ignore_patterns("*.pdf", "*.png"))
    target = shipped / "sE_spectra_in_codon_space_values.tsv"
    frame = pd.read_csv(target, sep="\t")
    column = next(c for c in frame.columns if pd.api.types.is_float_dtype(frame[c]))
    frame.loc[0, column] = frame.loc[0, column] * 1.01 + 0.01
    frame.to_csv(target, sep="\t", index=False, float_format="%.6g")

    report = bf.check_build(built, shipped, full=False)
    assert report["sE_spectra_in_codon_space_values.tsv"]


# ---------------------------------------------------------------------------
# 6. A clone has to be able to build
# ---------------------------------------------------------------------------

def _git(*args: str) -> list[str] | None:
    import subprocess

    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in out.stdout.split("\0") if line]


@pytest.mark.parametrize("directory", ["results", "data", "guides", "output"])
def test_no_file_a_figure_reads_or_ships_is_git_ignored(directory):
    """``results/`` was silently ignored from the first commit (the exploration archive's own
    ``results`` rule matched it), so a fresh clone had none of the tables every figure reads
    and nothing built. A file is fine if git tracks it or would add it; it is lost if neither."""
    tracked = _git("ls-files", "-z", "--", directory)
    if tracked is None:
        pytest.skip("not a git checkout")
    addable = _git("ls-files", "-z", "--others", "--exclude-standard", "--", directory)
    on_disk = {p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / directory).rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    lost = sorted(on_disk - set(tracked) - set(addable))
    assert not lost, f"git would never ship these, so a clone cannot rebuild: {lost[:8]}"


# ---------------------------------------------------------------------------
# 7. The alpha-optima table spans BOTH DMS values files
# ---------------------------------------------------------------------------

def _run_table_exporter(directory: Path):
    import subprocess

    return subprocess.run([sys.executable, str(REPO_ROOT / "tools/export_sY_dms_alpha_optima.py"),
                           str(REPO_ROOT), str(directory)], capture_output=True, text=True)


def _exported_from_shipped_values(tmp_path: Path) -> Path:
    for name in ("fig04_alpha_vs_dms_values.tsv", "sL_dms_sc2_values.tsv"):
        shutil.copy(OUTPUT / name, tmp_path / name)
    proc = _run_table_exporter(tmp_path)
    assert proc.returncode == 0, proc.stderr
    return tmp_path / "sY_dms_alpha_optima.tsv"


def test_the_exporter_covers_every_panel_and_model_in_both_values_files(tmp_path):
    """Found by a scratch rebuild: the exporter read only Figure 4's file (influenza), so it
    wrote 8 rows where the shipped table has 18 and silently dropped all of SARS-CoV-2."""
    made = pd.read_csv(_exported_from_shipped_values(tmp_path), sep="\t")
    shipped = pd.read_csv(OUTPUT / "sY_dms_alpha_optima.tsv", sep="\t")
    key = ["dataset", "trait", "model_name"]
    assert len(made) == len(shipped) == 18
    assert made.organism.nunique() == 2
    assert made[key].to_records(index=False).tolist() == shipped[key].to_records(index=False).tolist()


def test_the_shipped_alpha_optima_table_is_what_the_exporter_makes_from_the_shipped_values(tmp_path):
    # Was an xfail (defect 5): the shipped table had gone stale against the figures it
    # summarises, through a sign-aware optimum on the receptor-binding panels and a model
    # rename, and its caption was hand-written beside it so no rebuild could fix both.
    # Table and caption are now written together by the exporter; see FIGURE_LABELS.md.
    rebuilt = _exported_from_shipped_values(tmp_path).read_bytes()
    assert rebuilt == (OUTPUT / "sY_dms_alpha_optima.tsv").read_bytes()


def test_the_caption_is_written_beside_the_table_and_quotes_only_its_rows(tmp_path):
    """The caption used to be hand-written, so it outlived two changes to the table.

    Every count it gives is checked back against the rows the same run produced. The claims
    are about the *fitness* panels only: receptor binding correlates negatively, so "larger
    rho" is not "better" there and a range spanning both would mean nothing.
    """
    table = pd.read_csv(_exported_from_shipped_values(tmp_path), sep="\t")
    caption = " ".join((tmp_path / "sY_dms_alpha_optima_caption.md").read_text().split())

    assert caption.startswith(f"**{caption_label('sY_dms_alpha_optima')}.")
    fitness = table[~table["trait"].str.lower().str.contains("binding")]
    good = fitness[fitness["interpretable"]]
    assert f"On the {len(good)} interpretable fitness rows" in caption
    assert f"The {len(table) - len(fitness)} receptor-binding rows are left out" in caption
    assert f"**{int((~table['interpretable']).sum())} of {len(table)} rows are not interpretable**" in caption
    ft = good[good["model_role"] == "finetune"]["alpha_best"]
    assert f"best weight at {ft.min():+.2f} to {ft.max():+.2f}" in caption


def test_the_caption_does_not_quote_a_number_the_table_contradicts(tmp_path):
    """The guard that a hand-written caption had no way of having."""
    table = pd.read_csv(_exported_from_shipped_values(tmp_path), sep="\t")
    caption = " ".join((tmp_path / "sY_dms_alpha_optima_caption.md").read_text().split())
    wrong = len(table[~table["trait"].str.lower().str.contains("binding")]) + 1
    assert f"On the {wrong} interpretable fitness rows" not in caption


def test_without_the_sars_cov_2_values_the_exporter_fails_rather_than_writing_half_a_table(tmp_path):
    shutil.copy(OUTPUT / "fig04_alpha_vs_dms_values.tsv", tmp_path / "fig04_alpha_vs_dms_values.tsv")
    proc = _run_table_exporter(tmp_path)
    assert proc.returncode != 0
    assert not (tmp_path / "sY_dms_alpha_optima.tsv").exists()


# ---------------------------------------------------------------------------
# 8. The README's commands and the runner agree
# ---------------------------------------------------------------------------

def _readme_commands() -> dict[str, list[set[str]]]:
    import re

    text = (REPO_ROOT / "README.md").read_text()
    found: dict[str, list[set[str]]] = {}
    for script, rest in re.findall(r"`python (figures/[\w.]+\.py)([^`]*)`", text):
        found.setdefault(script, []).append(set(rest.split()))
    return found


def test_every_figure_command_in_the_readme_is_exactly_a_runner_step():
    """The README is what a reader types; the runner is what the tests exercise. A command in
    the README that differs from the step (a missing table argument, an old guide) is a figure
    the reader builds differently from the one that was checked."""
    commands = _readme_commands()
    steps = {}
    for s in bf.STEPS:
        steps.setdefault(s.script, []).append(set(s.args))
    for script, variants in commands.items():
        if script == "figures/make_fig02.py":          # retired, kept for its helpers
            continue
        for args in variants:
            assert args in steps.get(script, []), f"README: {script} {sorted(args)} is not a runner step"


def test_every_figure_the_runner_builds_has_a_command_in_the_readme():
    commands = _readme_commands()
    for step in bf.STEPS:
        if step.script.startswith("figures/"):
            assert set(step.args) in commands.get(step.script, []), \
                f"README has no command for {step.name}: {step.script} {' '.join(step.args)}"


def test_a_text_difference_is_reported_where_it_is_not_at_the_start_of_the_line(tmp_path):
    """The first version cut each line at 70 characters, which hid a hyphen-wrap difference
    that sat after column 80: the report showed two identical-looking strings."""
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    head = "x" * 85
    a.write_text(f"{head} fine tune\n")
    b.write_text(f"{head} fine-\ntune\n")
    message = bf.compare_text(a, b)[0]
    assert "line 1, column 91" in message and "fine tune" in message and "fine-" in message
