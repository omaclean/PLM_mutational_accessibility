#!/usr/bin/env python3
"""Rebuild every figure from the shipped tables, and say whether anything moved.

    python tools/build_figures.py                      # build into build/figures/, then check
    python tools/build_figures.py --list               # what would run, in order
    python tools/build_figures.py --only sE sN       # a subset (names from --list)
    python tools/build_figures.py --out output         # overwrite the shipped figures
    python tools/build_figures.py --jobs 1             # serial

Nothing here needs a GPU, a network or the exploration archive: every step reads the
derived tables under ``results/`` and the frozen panels under ``data/``. The models'
scoring runs are not part of this file; ``RUNS.md`` records how those tables were made.

**What a build proves.** A figure script is deterministic (every bootstrap is seeded), so a
rebuild must reproduce the shipped ``<stem>_values.tsv`` and ``<stem>_caption.md`` exactly.
With ``--check`` (the default whenever ``--out`` is not the shipped ``output/``) each is
compared against the shipped copy: values numerically, to one unit in the sixth significant
digit, which is the precision ``mutacc.io.write_values`` writes; captions as text. A
difference means a shipped figure no longer matches the code that claims to draw it, and the
exit status says so. The PDF and PNG are not compared: they embed fonts, ids and dates.

**Order.** Steps with no dependency run concurrently (``--jobs``, default 4, one BLAS thread
each so a build never takes more than ``--jobs`` cores). The composites, which place other
figures' panels, wait for them, and the DMS alpha table waits for Figure 4. That ordering is
read from the composite layouts, not typed: see ``composite_inputs``.

**A composite built into a scratch directory** places the *scratch* copies of its panels, not
the shipped ones (``staging_root``), so a stale panel cannot hide behind a fresh composite.

**Composite panels are intermediates** and are never written into the output directory: see
``panels_dir``. A build that ran every composite deletes them afterwards (``--keep-panels``
keeps them); a partial or failed build leaves them where the log says.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SHIPPED = REPO / "output"
DEFAULT_OUT = REPO / "build" / "figures"
#: Where a build over ``output/`` leaves its composite panels: the gitignored directory the
#: composite layouts name, and ``mutacc.guide.PANELS_DIR``, which the panel scripts default to.
SHIPPED_PANELS = REPO / "build" / "panels"

G13 = "guides/h3n2_13lineage.yaml"
G5 = "guides/h3n2_5lineage.yaml"
GSC2 = "guides/sars_cov_2_spike.yaml"

#: One unit in the sixth significant digit: ``write_values`` writes ``%.6g``, so two values
#: that agree to 1e-12 can still print as neighbours when they straddle a rounding edge.
RTOL = 2e-5
ATOL = 1e-9


@dataclass(frozen=True)
class Step:
    """One command. ``args`` are everything after the script; ``--out`` is added by the runner."""

    name: str
    script: str                      # path relative to the repository
    args: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()   # files this step must leave in the output directory
    panels: tuple[str, ...] = ()     # ...and in the panels directory (composite panels only)
    after: tuple[str, ...] = ()      # names of steps that must finish first
    retired: bool = False            # the code is kept; the manuscript has dropped the figure
    takes_out_flag: bool = True      # panelise and the table exporter name the directory differently


STEPS: tuple[Step, ...] = (
    # --- panels that composites place ---------------------------------------------------
    # Panels B and C read the exported ranking, so they wait for the table that writes it.
    Step("fig01bc", "figures/make_fig01bc.py", after=("table_fig01bc",),
         panels=("fig01b_ranked_model.pdf", "fig01c_ranked_accessibility.pdf")),
    Step("fig01d", "figures/make_fig01d.py", panels=("fig01d_codon_accessibility.pdf",)),
    Step("fig02ab", "figures/make_fig02ab.py", ("--guide", G5),
         panels=("fig02a_scatter_panel.pdf", "fig02b_scatter_panel.pdf")),
    Step("fig02c", "figures/make_fig02c.py", panels=("fig02c_epoch_panel.pdf",)),
    Step("fig02d", "figures/make_fig02d.py", ("--guide", G5), panels=("fig02d_strata_panel.pdf",)),
    # --- main figures -------------------------------------------------------------------
    Step("fig03", "figures/make_fig03.py", ("--guide", G13), produces=("fig03_alpha_sweep.pdf",)),
    Step("fig04", "figures/make_fig04.py", ("--guide", G13),
         produces=("fig04_alpha_vs_dms.pdf", "sL_dms_sc2.pdf")),
    Step("fig05", "figures/make_codon_learning.py", ("--guide", G5),
         produces=("fig05_codon_learning.pdf", "sI_codon_pair_tracking.pdf")),
    Step("fig01", "figures/panelise.py", ("--layout", "guides/composites/fig01.yaml"),
         after=("fig01bc", "fig01d"), takes_out_flag=True, produces=("fig01_composite.pdf",)),
    Step("fig02", "figures/panelise.py", ("--layout", "guides/composites/fig02.yaml"),
         after=("fig02ab", "fig02c", "fig02d"), takes_out_flag=True, produces=("fig02_composite.pdf",)),
    # --- supplementary figures ----------------------------------------------------------
    Step("sA", "figures/make_sA_spike.py", ("--guide", GSC2),
         produces=("sA_spike_model_vs_accessibility.pdf",)),
    Step("sB", "figures/make_sB_ranking_metrics.py", ("--guide", G13), produces=("sB_ranking_metrics.pdf",)),
    Step("sC", "figures/make_sC_viral_panel.py", ("--results", "results/viral_panel"),
         produces=("sC_viral_panel_recovery.pdf",)),
    Step("sD", "figures/make_sD_partial_correlation.py", ("--guide", G5), produces=("sD_partial_correlation.pdf",)),
    Step("sE", "figures/make_sE_spectra_codon_space.py", produces=("sE_spectra_in_codon_space.pdf",)),
    Step("sF", "figures/make_sF_spectrum_swap.py", ("--guide", G13), produces=("sF_spectrum_swap.pdf",)),
    Step("sG", "figures/make_sG_epistasis_null.py", ("--guide", G13), produces=("sG_epistasis_null.pdf",)),
    Step("sJ", "figures/make_sJ_dms_concordance.py", ("--guide", G13), produces=("sJ_dms_concordance.pdf",)),
    Step("sK", "figures/make_sK_branch_codons.py",
         ("--branch-table", "results/s11_h3n2_ha_tree/branch_codons.tsv.gz",
          "--column-table", "results/s11_h3n2_ha_tree/column_codons.tsv.gz",
          "--long-range-nodes", "results/s11_h3n2_ha_tree/long_range_nodes.tsv.gz",
          "--baseline-codons", "results/s11_h3n2_ha_tree/baseline_codons.tsv.gz",
          "--year-table", "results/s11_h3n2_ha_tree/year_counts.tsv.gz",
          "--plm-null-scores", "results/h3n2_13lineage/replacement_scores_masked.tsv.gz",
          "--plm-null-guide", G13,
          "--plm-null-lineage", "J.2"),
         produces=("sK_branch_codons.pdf",)),
    Step("sM", "figures/make_sM_within_stratum.py", ("--guide", G13), produces=("sM_within_stratum.pdf",)),
    Step("sH", "figures/make_sH_contact_recovery.py", ("--guide", G13), produces=("sH_contact_recovery.pdf",)),
    Step("s14", "figures/make_s14.py", ("--guide", G13, "--guide-compare", G5),
         produces=("s14_lineage_set_validity.pdf",), retired=True),
    Step("sZ", "figures/make_sZ_lineage_panel.py",
         ("--guide", G13, "--results", "results/h3n2_13lineage"),
         produces=("sZ_lineage_panel.pdf",)),
    # Reads only shipped tables. The corpus FASTA those tables were built from is
    # public but not shipped, so tools/export_sX_corpus_leakage.py is not a step here.
    Step("sX", "figures/make_sX_corpus_leakage.py",
         ("--results", "results/h3n2_13lineage"),
         produces=("sX_corpus_leakage.pdf",)),
    Step("sN", "figures/make_sN_simulation_control.py", produces=("sN_simulation_control.pdf",)),
    Step("sO", "figures/make_sO_masked_marginals.py", ("--guide", G13),
         produces=("sO_masked_marginals.pdf",)),
    Step("sP", "figures/make_sP_alpha_curves.py", ("--guide", G13),
         produces=("sP_alpha_curves.pdf",)),
    # --- tables derived from a figure's values -----------------------------------------
    Step("table_dms_alpha", "tools/export_sY_dms_alpha_optima.py", after=("fig04",), takes_out_flag=False,
         produces=("sY_dms_alpha_optima.tsv",)),
    # Figure 1B/1C are placed as a frozen render, so there is no panel step to hang this on:
    # it recomputes what those panels plot from the shipped replacement table. The comparison
    # lineage is named here rather than in the script, as the guides are.
    Step("table_fig01bc", "tools/export_fig01bc_values.py", ("--comparison", "K"),
         takes_out_flag=False, produces=("fig01bc_ranked_mutations_values.tsv",)),
    # The two quantities those panels rank, against each other: how closely they agree,
    # for both models, on raw, semi-log and log-log probabilities and on ranks. Reads the same shipped
    # table, so it hangs on no figure step either.
    Step("table_fig01bc_corr", "tools/export_fig01bc_correlations.py", takes_out_flag=False,
         produces=("fig01bc_correlation_stats_values.tsv",)),
    # --- tables derived from the shipped results tables --------------------------------
    # Reads the per-lineage sweeps directly, not Figure 3's values table, so it has no
    # dependency on any figure step.
    Step("table_loo_alpha", "tools/export_sW_loo_alpha.py", takes_out_flag=False,
         produces=("sW_loo_alpha_selection_values.tsv",)),
)

BY_NAME = {s.name: s for s in STEPS}


# ---------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------

#: Layout directories whose panels this file builds. ``data/frozen_panels`` is deliberately not
#: one: those panels have no generator (data/frozen_panels/README.md) and are placed as shipped.
BUILT_PANEL_DIRS = ("output", "build/panels")


def composite_inputs(layout: Path) -> list[str]:
    """Basenames of the panels a composite layout takes from a directory this file builds."""
    spec = yaml.safe_load(Path(layout).read_text())
    files = [Path(p["file"]) for p in spec.get("panels", [])]
    return [f.name for f in files
            if any(f.as_posix().startswith(d + "/") for d in BUILT_PANEL_DIRS)]


def producers_of(filename: str, steps=STEPS) -> list[Step]:
    return [s for s in steps if filename in s.produces or filename in s.panels]


def derived_dependencies(steps=STEPS) -> dict[str, set[str]]:
    """Explicit ``after`` edges plus, for each composite, whoever makes its ``output/`` panels."""
    deps = {s.name: set(s.after) for s in steps}
    for s in steps:
        if "--layout" in s.args:
            layout = REPO / s.args[s.args.index("--layout") + 1]
            for name in composite_inputs(layout):
                deps[s.name] |= {p.name for p in producers_of(name, steps)}
    return deps


def plan(selected: set[str] | None, include_retired: bool, steps=STEPS) -> list[Step]:
    """Steps to run, with whatever they depend on pulled in, in dependency order."""
    deps = derived_dependencies(steps)
    by = {s.name: s for s in steps}
    want = {s.name for s in steps if include_retired or not s.retired} if selected is None else set(selected)
    unknown = want - set(by)
    if unknown:
        raise SystemExit(f"unknown step(s): {sorted(unknown)}; --list shows the names")
    frontier = list(want)
    while frontier:
        for d in deps[frontier.pop()]:
            if d not in want:
                want.add(d)
                frontier.append(d)
    ordered, done = [], set()
    while len(ordered) < len(want):
        ready = [by[n] for n in sorted(want - done) if deps[n] <= done]
        if not ready:
            raise SystemExit(f"dependency cycle among {sorted(want - done)}")
        for s in ready:
            ordered.append(s)
            done.add(s.name)
    return ordered


# ---------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------

@dataclass
class Result:
    step: Step
    ok: bool
    seconds: float
    detail: str = ""
    missing: list[str] = field(default_factory=list)


def _env(threads: int = 1) -> dict[str, str]:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    env["MPLBACKEND"] = "Agg"
    return env


def scratch_dir(out: Path) -> Path:
    """Where this build's intermediates go: its panels and its per-step logs.

    A scratch build keeps them inside itself, which is already a scratch tree. A build over the
    shipped ``output/`` must not: that directory is the published figures, and a build used to
    drop ``_logs/`` into it. Those go to the gitignored ``build/`` instead.
    """
    return REPO / "build" if Path(out).resolve() == SHIPPED.resolve() else Path(out)


def panels_dir(out: Path) -> Path:
    """Where this build's composite panels go. Never inside ``out``.

    Panels are intermediates, so a build must not leave them beside the finished figures. A
    build over the shipped ``output/`` puts them in the gitignored ``build/panels/`` the layouts
    name; a scratch build keeps its own, beside that build's ``_logs``.
    """
    return scratch_dir(out) / ("panels" if Path(out).resolve() == SHIPPED.resolve() else "_panels")


def staging_root(out: Path) -> Path:
    """A tree a composite's relative panel paths resolve in, with ``output/`` being ``out``.

    ``panelise`` only accepts repo-relative panel paths (a layout that reaches elsewhere is
    refused on purpose), so a scratch build cannot simply rewrite them. Instead the layout
    stays as shipped and is given a root in which ``output`` is the scratch directory and
    ``data`` is the repository's: the composite then places the panels *this build* made.
    """
    root = Path(out).resolve() / "_root"
    root.mkdir(parents=True, exist_ok=True)
    for name, target in (("output", Path(out).resolve()), ("data", REPO / "data"),
                         ("build/panels", panels_dir(out).resolve())):
        link = root / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() == target:
            continue                      # two composites may be staged at once
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            link.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            pass
    return root


def command(step: Step, out: Path, python: str) -> list[str]:
    """The argv for one step. A composite is rooted so that ``output/`` means ``out``."""
    cmd = [python, str(REPO / step.script), *step.args]
    if "--layout" in step.args and Path(out).resolve() != SHIPPED.resolve():
        cmd += ["--root", str(staging_root(out))]
    if step.panels:
        cmd += ["--panels-out", str(panels_dir(out))]
    if step.takes_out_flag:
        cmd += ["--out", str(out)]
    else:
        cmd += [str(REPO), str(out)]       # the table exporter: <repo> <output dir>
    return cmd


def run_step(step: Step, out: Path, python: str, log_dir: Path) -> Result:
    start = time.monotonic()
    log = log_dir / f"{step.name}.log"
    proc = subprocess.run(command(step, out, python), cwd=REPO, env=_env(),
                          capture_output=True, text=True)
    log.write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
    took = time.monotonic() - start
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        return Result(step, False, took, f"exit {proc.returncode}: " + " | ".join(tail))
    missing = [f for f in step.produces if not (out / f).is_file()]
    missing += [f for f in step.panels if not (panels_dir(out) / f).is_file()]
    if missing:
        return Result(step, False, took, "exit 0 but did not write " + ", ".join(missing), missing)
    return Result(step, True, took)


def execute(steps: list[Step], out: Path, jobs: int, python: str) -> list[Result]:
    out.mkdir(parents=True, exist_ok=True)
    panels_dir(out).mkdir(parents=True, exist_ok=True)
    log_dir = scratch_dir(out) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    deps = derived_dependencies(steps)
    pending = {s.name: s for s in steps}
    finished: dict[str, Result] = {}
    running: dict[cf.Future, str] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        while pending or running:
            for name, step in list(pending.items()):
                blockers = deps[name] & set(deps)
                failed = [b for b in blockers if b in finished and not finished[b].ok]
                if failed:
                    finished[name] = Result(step, False, 0.0, f"not run: {', '.join(failed)} failed")
                    del pending[name]
                elif blockers <= set(finished) and len(running) < max(1, jobs):
                    running[pool.submit(run_step, step, out, python, log_dir)] = name
                    del pending[name]
                    print(f"  start  {name}", flush=True)
            if not running:
                continue
            done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                name = running.pop(fut)
                res = fut.result()
                finished[name] = res
                print(f"  {'ok   ' if res.ok else 'FAIL '}  {name:<16} {res.seconds:7.1f}s  {res.detail}",
                      flush=True)
    return [finished[s.name] for s in steps]


# ---------------------------------------------------------------------------------------
# Checking a build against the shipped figures
# ---------------------------------------------------------------------------------------

def compare_values(built: Path, shipped: Path) -> list[str]:
    """Differences between two values tables; an empty list means they agree."""
    a = pd.read_csv(built, sep="\t", dtype=object, keep_default_na=False)
    b = pd.read_csv(shipped, sep="\t", dtype=object, keep_default_na=False)
    if list(a.columns) != list(b.columns):
        return [f"columns differ: built {list(a.columns)} vs shipped {list(b.columns)}"]
    if len(a) != len(b):
        return [f"row count differs: built {len(a)} vs shipped {len(b)}"]
    problems = []
    for col in a.columns:
        x, y = a[col], b[col]
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        numeric = xn.notna() & yn.notna()
        # a cell that is a number on one side and text on the other is a difference
        mixed = (xn.notna() != yn.notna()) & ~((x == "") & (y == ""))
        text = ~numeric & ~mixed
        bad = mixed | (text & (x != y))
        if numeric.any():
            close = np.isclose(xn[numeric], yn[numeric], rtol=RTOL, atol=ATOL, equal_nan=True)
            bad.loc[numeric[numeric].index[~close]] = True
        for i in bad[bad].index[:3]:
            problems.append(f"{col}[row {i}]: built {x[i]!r} vs shipped {y[i]!r}")
        if bad.sum() > 3:
            problems.append(f"{col}: {int(bad.sum())} cells differ in all")
    return problems


def compare_text(built: Path, shipped: Path) -> list[str]:
    a, b = built.read_text(), shipped.read_text()
    if a == b:
        return []
    for n, (x, y) in enumerate(zip(a.splitlines(), b.splitlines()), 1):
        if x != y:
            at = len(os.path.commonprefix([x, y]))
            lo = max(0, at - 25)
            return [f"line {n}, column {at + 1}: built ...{x[lo:at + 25]!r} vs shipped ...{y[lo:at + 25]!r}"]
    return [f"length differs: built {len(a.splitlines())} lines vs shipped {len(b.splitlines())}"]


CHECKED_SUFFIXES = {"_values.tsv": compare_values, "_caption.md": compare_text}


def checkable(path: Path):
    for suffix, fn in CHECKED_SUFFIXES.items():
        if path.name.endswith(suffix):
            return fn
    if path.name.startswith(("table_", "supl_table")) and path.suffix == ".tsv":
        return compare_values
    return None


def check_build(built_dir: Path, shipped_dir: Path = SHIPPED, full: bool = True,
                skipped_stems: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """``{file: [problems]}`` for every checkable built file; plus shipped files nothing rebuilt.

    ``full`` adds ``"<not rebuilt>"`` entries for checkable shipped files the build did not
    write: a figure that ships but no script regenerates is exactly what a final build must
    surface.
    """
    report: dict[str, list[str]] = {}
    for path in sorted(Path(built_dir).iterdir()):
        fn = checkable(path)
        if fn is None or not path.is_file():
            continue
        twin = Path(shipped_dir) / path.name
        report[path.name] = fn(path, twin) if twin.exists() else ["no shipped copy to compare with"]
    if full:
        for path in sorted(Path(shipped_dir).iterdir()):
            if checkable(path) and not (Path(built_dir) / path.name).exists() \
                    and not any(path.name.startswith(s) for s in skipped_stems):
                report[path.name] = ["<not rebuilt>: shipped, but this build did not write it"]
    return report


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------

def retired_stems() -> tuple[str, ...]:
    """Stems the numbering map retires: their shipped files are not 'missing' from a build."""
    from mutacc.numbering import load_numbering

    return tuple(stem for stem, entry in load_numbering().entries.items() if entry.retired)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"where to build (default {DEFAULT_OUT.relative_to(REPO)}/; "
                         "'output' overwrites the shipped figures)")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run just these steps (and what they need)")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--jobs", type=int, default=4, help="steps to run at once (default 4)")
    ap.add_argument("--no-check", action="store_true", help="skip the comparison with output/")
    ap.add_argument("--include-retired", action="store_true", help="also build retired figures")
    ap.add_argument("--keep-panels", action="store_true",
                    help="keep the composite panels after the build instead of deleting them "
                         "(they are intermediates; see panels_dir)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    steps = plan(set(args.only) if args.only else None, args.include_retired)
    deps = derived_dependencies(steps)
    if args.list:
        for s in steps:
            need = f"  (after {', '.join(sorted(deps[s.name]))})" if deps[s.name] else ""
            print(f"{s.name:<16} {s.script} {' '.join(s.args)}{need}")
        return 0

    out = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
    overwriting = out.resolve() == SHIPPED.resolve()
    print(f"building {len(steps)} steps into {out}  ({args.jobs} at once)")
    results = execute(steps, out, args.jobs, args.python)
    failed = [r for r in results if not r.ok]
    shutil.rmtree(out / "_root", ignore_errors=True)

    # The panels are intermediates, so a finished build clears them. Only when every composite
    # that places them was part of this build, though: `--only fig02ab` is someone asking to look
    # at a panel, and deleting it under them would be perverse. A failed build keeps them too,
    # so whoever reads the logs can see what the panel looked like.
    built_panels = {name for s in steps for name in s.panels}
    consumers = {s.name for s in steps if "--layout" in s.args}
    all_consumers = {s.name for s in STEPS if "--layout" in s.args}
    if built_panels and not failed and not args.keep_panels and consumers == all_consumers:
        shutil.rmtree(panels_dir(out), ignore_errors=True)
    elif built_panels:
        print(f"\ncomposite panels left in {panels_dir(out)}")

    status = 0
    if failed:
        status = 1
        print(f"\n{len(failed)} step(s) failed; logs in {scratch_dir(out) / '_logs'}")
    if overwriting:
        print("\nbuilt over the shipped output/; `git diff --stat output/` shows what moved")
    elif not args.no_check:
        full = args.only is None
        report = check_build(out, SHIPPED, full=full, skipped_stems=retired_stems())
        moved = {k: v for k, v in report.items() if v}
        print(f"\ncompared {len(report)} values/caption files with output/: "
              f"{len(report) - len(moved)} identical, {len(moved)} differ")
        for name, problems in moved.items():
            print(f"  DIFF  {name}")
            for p in problems[:8]:
                print(f"          {p}")
        if moved:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
