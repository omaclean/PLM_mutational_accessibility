#!/usr/bin/env python3
"""How the accessibility weight is selected, and how well the data pin it down.

    python tools/export_sW_loo_alpha.py                     # writes into output/
    python tools/export_sW_loo_alpha.py <repo> <out_dir>    # how tools/build_figures.py calls it

Answers Reviewer 2's "alpha is chosen on the data it is scored on"
(``PREPRINT_AND_REVIEWS.md``, Reviewer 2, uncertainty row), for both halves of the manuscript
that select a weight. One file, two kinds of row, told apart by ``scope``:

* **Figure 3, the lineage sweeps** (``scope`` ``fold`` and ``mean``). Figure 3 picks the weight
  at the peak of the mean-over-lineages curve and reports what that weight scores on the same
  lineages; these rows pick the weight on twelve lineages and score it on the thirteenth,
  thirteen times over. The arithmetic is :mod:`mutacc.loo_alpha`.
* **Figure 4 and Figure SL, the deep mutational scans** (``scope`` ``dms_alpha_star``). Holding
  a scan out is not available there -- each panel is one experiment -- so the question is asked
  the other way round: how well do these sites pin the selected weight down? Sites are
  resampled with replacement, the whole rho-versus-alpha curve is recomputed inside every
  replicate, and the replicate's own argmax is taken, giving a percentile interval for alpha*
  itself. The arithmetic is :mod:`mutacc.dms_alpha_ci`.

Nothing is computed in this file: it joins the two tables, writes them, and writes a caption
whose every number is read back off the rows.

**The DMS half is slow.** Every replicate re-ranks every replacement at all 201 weights, so a
default run is about 110 core-minutes (18 panel-model arms x 1000 replicates); ``--dms-jobs``
spreads it over processes and ``--dms-boot`` is there for a quick structural check, not for
the shipped file. The lineage half takes under two seconds.

**Both scoring arms are written, always.** Figure 3 draws the masked-marginal table by
default and the wild-type (unmasked) one under ``--scoring unmasked``, so a single arm
chosen by a flag would make the shipped file depend on how it was last run -- and the two
arms disagree about where the held-out weight lands. The ``scoring`` column says which is
which, and a reader comparing this table with the figure takes the ``masked`` rows.

**The DMS rows are the masked-marginal arm only.** That is the probability Figure 4, Figure SL
and Table SY report; the wild-type (unmasked) arm survives in the figure pipeline as the gate
that reproduces the archive's curves, and its optima are not what the manuscript quotes. The
``scoring`` column says so on every row.

**No p-value is computed, here or in either module.** For the lineage rows: the thirteen
lineages are not independent replicates (residual outcome co-occurrence about 0.391; effective
number of independent lineages 1.01 to 1.12), so a sign test or a paired t-test over the folds
is anti-conservative, and 13 of 13 is a sign test's floor of 2 / 2^13 whatever the effect size.
For the DMS rows: the question is where one selected weight sits, which an interval answers
directly, and the panels are not independent of one another either. The effect size, its
percentile interval and the counts are what this table carries and what should be quoted.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from mutacc.dms_alpha_ci import (                                        # noqa: E402
    ALPHA_STAR, DMS_METRIC, alpha_star_table, iter_arms,
)
from mutacc.dms_alpha_ci import N_BOOT as DMS_N_BOOT                      # noqa: E402
from mutacc.guide import REPO_ROOT, load_guide                           # noqa: E402
from mutacc.io import write_caption, write_values                        # noqa: E402
from mutacc.loo_alpha import FOLD, MEAN, ALPHA_REFERENCE, loo_alpha_table  # noqa: E402
from mutacc.masked_dms import apply_masked                                # noqa: E402

STEM = "sW_loo_alpha_selection"

#: The guide whose lineage set and models the manuscript's Figure 3 is drawn from.
DEFAULT_GUIDE = REPO_ROOT / "guides" / "h3n2_13lineage.yaml"

#: ``scoring`` label -> the guide table that holds that arm's sweep. The order is the
#: order the rows are written in, masked first because that is the arm Figure 3 draws.
ARMS = {
    "masked": "alpha_sweep_by_lineage_masked",
    "unmasked": "alpha_sweep_by_lineage",
}

#: The two ranking metrics Figure 3 and Figure SB report for the binary task. Both are
#: swept, because they answer the reviewer's question differently: AUROC is prevalence-free
#: and PR-AUC is not, and the draft quotes both.
METRICS = ("site_logistic_auroc", "site_logistic_pr_auc")

#: The scoring arm the DMS rows are computed on. One value, not both: see the module docstring.
DMS_SCORING = "masked"

#: The weight grid the DMS sweep is searched over. It must be the grid
#: ``figures/make_fig04.py`` draws (``ALPHA_LIMIT``, ``ALPHA_STEP``), or the interval would be
#: an interval on a different axis from the optimum the figure marks; the point estimates
#: written here are checked against the shipped ``sY_dms_alpha_optima.tsv`` by the tests.
DMS_ALPHA_LIMIT, DMS_ALPHA_STEP = 1.0, 0.01

#: Processes the DMS bootstrap is spread over. Four, like ``tools/build_figures.py``'s own
#: default, so that a build running this step alongside three others does not oversubscribe a
#: small machine. Each arm is seeded from its own position, so the result does not depend on
#: this number (:func:`mutacc.dms_alpha_ci.arm_row`).
DMS_JOBS = 4

#: Share of a row's resamples allowed at an end of the searched range before the caption calls
#: the row out. Not a rule the table applies -- every row carries its own two fractions -- but
#: a threshold the caption has to name in order to count anything, and a twentieth of the
#: replicates against an edge is already enough to drag a percentile endpoint onto it.
DMS_PILE_UP = 0.05

#: The per-panel optimum table whose ``alpha_best`` these rows recompute, named in the caption
#: and checked against in the tests.
SY_STEM = "sY_dms_alpha_optima"

COLS = [
    "scoring", "metric", "model", "model_label", "model_role", "scope", "held_out_lineage",
    "alpha_star", "n_tied_alpha", "metric_at_alpha_star", "metric_at_reference",
    "estimate", "ci_lo", "ci_hi", "n_folds", "n_folds_gain_positive",
    # Only the DMS rows carry these; they are empty on a lineage row rather than filled with a
    # stand-in, because a fold has no panel, no sites of its own and no edge to pile up against.
    "panel_label", "organism", "trait", "n", "n_sites", "n_boot",
    "frac_boundary_low", "frac_boundary_high", "boundary", "weak_signal", "interpretable",
]

#: Columns a lineage row fills and a DMS row does not, and the other way round. Named once so
#: that the two halves cannot drift into sharing a column that means different things.
LOO_ONLY = ("held_out_lineage", "n_tied_alpha", "n_folds", "n_folds_gain_positive")
DMS_ONLY = ("panel_label", "organism", "trait", "n", "n_sites", "n_boot",
            "frac_boundary_low", "frac_boundary_high", "boundary", "weak_signal",
            "interpretable")


def dms_alpha_grid() -> np.ndarray:
    """The sweep grid, built exactly as ``figures/make_fig04.alpha_grid`` builds it."""
    return np.round(np.arange(-DMS_ALPHA_LIMIT, DMS_ALPHA_LIMIT + DMS_ALPHA_STEP / 2,
                              DMS_ALPHA_STEP), 6)


def build(guide, results: Path | None = None) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Every (scoring arm, metric, model): thirteen folds and their summary row.

    The swept range is returned beside the table. The caption needs it to say which
    held-out weights sit at the edge of the search, and reading it off the sweep rather
    than writing -1 to +1 into the text is what keeps that sentence true if the grid is
    ever widened. Both arms must share the grid, or the two halves of the table would not
    be on one axis.
    """
    model_keys = [m.key for m in guide.models]
    parts, grids = [], set()
    for scoring, table in ARMS.items():
        frame = guide.read_table(table, results)
        guide.check_lineages(frame["lineage"].unique(), f"{table} ({scoring})")
        swept = frame.loc[frame["alpha"].notna(), "alpha"]
        grids.add((round(float(swept.min()), 6), round(float(swept.max()), 6)))
        part = loo_alpha_table(frame, model_keys, METRICS)
        part.insert(0, "scoring", scoring)
        parts.append(part)
    if len(grids) != 1:
        raise SystemExit(f"the scoring arms are swept over different alpha ranges: {sorted(grids)}")
    out = pd.concat(parts, ignore_index=True)
    out["model_label"] = out["model"].map({m.key: m.label for m in guide.models})
    out["model_role"] = out["model"].map({m.key: m.role for m in guide.models})
    # A fold has no panel, no sites of its own and no edge to pile up against, so the DMS
    # columns stay empty here rather than taking a zero that would read as a measurement.
    for column in DMS_ONLY:
        out[column] = np.nan
    return out[COLS], grids.pop()


def build_dms(guide, results: Path | None = None, *, n_boot: int = DMS_N_BOOT,
              jobs: int = DMS_JOBS) -> pd.DataFrame:
    """One row per (DMS panel, model): the selected weight and its bootstrap interval.

    The scores are the guide's ``dms_scores`` with the masked marginal joined on, which is the
    frame Figure 4 draws from; the arms and their order come from ``dms_peak_summary``, as
    there. Nothing is recomputed from a figure's values table -- this reads the same inputs the
    figure does, so the point estimate it writes is the figure's own optimum rather than a copy
    of a number parsed back out of an output file.

    ``jobs`` only spreads the work: every arm is seeded from its position, so one process and
    eight give the same table (``tests/test_loo_alpha.py`` asserts it).
    """
    scores = guide.read_table("dms_scores", results)
    summary = guide.read_table("dms_peak_summary", results)
    scores = apply_masked(scores, guide.read_table("dms_scores_masked", results))
    arms = iter_arms(scores, summary)
    started = time.time()
    if jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
            table = alpha_star_table(arms, dms_alpha_grid(), n_boot=n_boot, runner=pool.map)
    else:
        table = alpha_star_table(arms, dms_alpha_grid(), n_boot=n_boot)
    print(f"[dms] {len(table)} arms x {n_boot} site-resamples in {time.time() - started:.0f}s "
          f"on {jobs} process(es)")

    out = pd.DataFrame({
        "scoring": DMS_SCORING,
        "metric": DMS_METRIC,
        "model": table["model"],
        "model_label": table["model_label"],
        "model_role": table["model_role"],
        "scope": ALPHA_STAR,
        # The weight the figure marks, chosen on all the sites, with the two correlations it is
        # read between: the same three numbers a fold row carries, for the same reason.
        "alpha_star": table["alpha_star"],
        "metric_at_alpha_star": table["rho_best"],
        "metric_at_reference": table["rho_alpha0"],
        # ``estimate`` is the quantity ci_lo and ci_hi bracket, as on a summary row. Here that
        # is the bootstrap MEDIAN of alpha*, not the point estimate beside it: the interval is
        # the spread of the replicates' own argmaxes, so the centre quoted with it has to come
        # from the same distribution. The two differ by a grid step or so where the curve is
        # asymmetric about its peak, and that difference is itself worth seeing.
        "estimate": table["alpha_star_median"],
        "ci_lo": table["ci_lo"],
        "ci_hi": table["ci_hi"],
        "panel_label": table["panel_label"],
        "organism": table["organism"],
        "trait": table["trait"],
        "n": table["n"],
        "n_sites": table["n_sites"],
        "n_boot": table["n_boot_used"],
        "frac_boundary_low": table["frac_boundary_low"],
        "frac_boundary_high": table["frac_boundary_high"],
        "boundary": table["boundary"],
        "weak_signal": table["weak_signal"],
        "interpretable": table["interpretable"],
    })
    # Written as text, as the flags in ``sY_dms_alpha_optima.tsv`` are. Left as booleans they
    # would meet a column of NaNs when the two halves of the table are concatenated, and
    # pandas resolves bool against float by making the flags 1.0 and 0.0 -- which still reads
    # as a flag, in a file whose other flags say True.
    for column in ("boundary", "weak_signal", "interpretable"):
        out[column] = out[column].map({True: "True", False: "False"})
    out["held_out_lineage"] = ""
    for column in LOO_ONLY[1:]:
        out[column] = np.nan
    return out[COLS]


def _signed_range(values) -> str:
    """``"+0.60 to +0.66"``, or the single value where every fold agrees."""
    lo, hi = min(values), max(values)
    return f"{lo:+.2f}" if lo == hi else f"{lo:+.2f} to {hi:+.2f}"


def _flag(column: pd.Series) -> pd.Series:
    """A True/False column as booleans, whether it came from memory or back off the TSV.

    The flags are written as ``True``/``False`` text and are empty on a lineage row, so a
    column read back from the file is object-typed and a plain ``astype(bool)`` would make
    every empty cell True. The caption is built from the table either way -- in the run that
    writes it and, in the tests, from the shipped file -- and has to say the same thing both
    times.
    """
    return column.astype(str).str.lower().eq("true")


def _and(items) -> str:
    """``"a, b and c"``."""
    items = list(items)
    return items[0] if len(items) < 2 else ", ".join(items[:-1]) + " and " + items[-1]


def _dms_line(rows: pd.DataFrame, name: str) -> str:
    """One clause of the DMS result sentence, every number read off ``rows``."""
    if rows.empty:
        return f"No row is interpretable for {name}."
    widths = rows["ci_hi"] - rows["ci_lo"]
    return (f"Over the {len(rows)} interpretable rows for {name} the selected weight is "
            f"{_signed_range(rows['alpha_star'])}, with 95% intervals running from "
            f"{rows['ci_lo'].min():+.2f} to {rows['ci_hi'].max():+.2f} and spanning "
            f"{widths.min():.2f} to {widths.max():.2f}.")


def caption(guide, table: pd.DataFrame, alpha_range: tuple[float, float]) -> str:
    """The caption, with every quoted number read off ``table``.

    Nothing is asserted here that the rows do not say: a rebuild on different data writes
    different sentences rather than false ones, which is the rule
    ``tools/export_sY_dms_alpha_optima.py`` set for the tables in this directory.
    """
    folds = table[table["scope"] == FOLD]
    means = table[table["scope"] == MEAN]
    drawn = means[means["scoring"] == "masked"]
    n_folds = int(means["n_folds"].iloc[0])

    def line(role: str, metric: str, name: str) -> str:
        row = drawn[(drawn["model_role"] == role) & (drawn["metric"] == metric)].iloc[0]
        grid = folds[(folds["scoring"] == "masked") & (folds["model_role"] == role)
                     & (folds["metric"] == metric)]["alpha_star"]
        return (f"{row['estimate']:+.3f} [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}] in "
                f"{name}, with every fold's weight landing at {_signed_range(grid)} and "
                f"{int(row['n_folds_gain_positive'])} of {n_folds} folds above zero")

    sentences = []
    for role, label in (("finetune", "fine-tuned"), ("base", "base")):
        parts = [line(role, "site_logistic_auroc", "AUROC"),
                 line(role, "site_logistic_pr_auc", "PR-AUC")]
        sentences.append(f"The {label} model gains {'; and '.join(parts)}.")
    at_edge = folds[folds["alpha_star"].isin(alpha_range)]
    edge = ""
    if not at_edge.empty:
        labels = sorted(set(at_edge["model_label"]))
        edge = (f" In {len(at_edge)} of the {len(folds)} folds the held-out weight sits at "
                f"an edge of the searched range ({alpha_range[0]:+.2f} to "
                f"{alpha_range[1]:+.2f}), all of them {' and '.join(labels)}: the training "
                f"curve never turns over there, so that weight is a bound rather than an "
                f"estimate.")
    agree = means.groupby(["metric", "model"])["estimate"].apply(
        lambda s: np.sign(s).nunique() == 1).all()

    dms = table[table["scope"] == ALPHA_STAR]
    readable = dms[_flag(dms["interpretable"])]
    dms_sentences = [
        _dms_line(readable[(readable["organism"] == organism) & (readable["model_role"] == role)],
                  f"the {label} models on the {organism} panels")
        for organism in sorted(set(readable["organism"]))
        for role, label in (("finetune", "fine-tuned"), ("base", "base"))
    ]
    unreadable = dms[~_flag(dms["interpretable"])]
    n_edge = int(_flag(unreadable["boundary"]).sum())
    n_weak = int((_flag(unreadable["weak_signal"]) & ~_flag(unreadable["boundary"])).sum())
    piled = dms[(dms["frac_boundary_low"] + dms["frac_boundary_high"]) > DMS_PILE_UP]
    pile_up = (
        f" In {len(piled)} of the {len(dms)} rows more than {DMS_PILE_UP:.0%} of the replicates "
        f"put the optimum at an end of the searched range (up to "
        f"{(piled['frac_boundary_low'] + piled['frac_boundary_high']).max():.0%}, on "
        f"{_and(sorted(set(piled['panel_label'] + ', ' + piled['model_label'])))}). Those "
        f"replicates found the end of the range rather than a peak, and an interval that "
        f"includes them is not an interval on an interior optimum."
        if len(piled) else
        f" No row has more than {DMS_PILE_UP:.0%} of its replicates at an end of the searched "
        f"range, so no interval below is standing on a pile-up against an edge."
    )
    widths = dms["ci_hi"] - dms["ci_lo"]

    return f"""
    **The weight on mutational accessibility chosen without the lineage it is scored on,
    and how tightly the deep mutational scans pin that weight down.**
    Figure 3 reads its result off the peak of the curve averaged over all {n_folds}
    {guide.virus} lineages, and reports what that weight scores on those same lineages.
    This table removes that circularity. Each lineage is held out in turn; the weight alpha
    is chosen at the peak of the mean curve over the other {n_folds - 1} lineages, with the
    held-out lineage taking no part in the choice; and that weight is then scored on the
    held-out lineage alone. The gain is its score there minus its score at alpha = {ALPHA_REFERENCE:g},
    which is the model with no accessibility weighting at all. Every lineage is held out
    exactly once, giving {n_folds} separate choices of weight and {n_folds} gains per model
    and metric. The folds are separate, not independent: any two of them train on
    {n_folds - 2} of the same {n_folds - 1} lineages, and the lineages themselves are not
    independent of one another either (see below).

    {' '.join(sentences)}{edge}
    The masked-marginal rows are quoted above because they are the arm Figure 3 draws; the
    wild-type (unmasked) rows are in the table and
    {'agree in sign throughout' if agree else 'do not agree in sign throughout'}.

    **The deep mutational scans, where no scan can be held out.** The {len(dms)} rows with
    `scope` = `{ALPHA_STAR}` are the {dms['panel_label'].nunique()} panels of Figure 4 and
    Figure SL, each scored by a base and a virus-specific fine-tuned model. Each panel is a
    single experiment, so there is no second scan to choose the weight on; what can be asked
    instead is how well the measured sites pin the chosen weight down. Sites are resampled
    with replacement -- replacements at one site share a backbone and are not independent --
    the whole correlation-versus-alpha curve is recomputed on the resampled replacements, and
    **that replicate's own best weight is taken**, {int(dms['n_boot'].max()):,} times per row.
    `alpha_star` is the weight the figure marks, chosen on all the sites; `estimate` is the
    median of the resampled weights and `ci_lo` to `ci_hi` their 2.5th and 97.5th percentiles.
    {' '.join(dms_sentences)}

    What those intervals are not. The best weight is an **argmax**, which jumps: a replicate in
    which one grid point tips above its neighbour moves the answer by a whole step, so the
    resampled distribution is lumpy on the 0.01 grid and can have more than one mode. What is
    quoted is a percentile interval of that distribution and nothing more -- not a confidence
    interval with a coverage guarantee, which would need the smooth, asymptotically normal
    statistic an argmax is not. Nothing is smoothed, trimmed or re-centred to tighten it: where
    a curve is a broad plateau the interval comes back wide, and here they span
    {widths.min():.2f} to {widths.max():.2f} in alpha. A wide one is the finding that those
    sites do not locate the weight, not a defect of the estimator.{pile_up}
    {len(unreadable)} of the {len(dms)} rows are not interpretable at all and their
    `alpha_star` must not be quoted: {n_edge} put the best weight at the end of the searched
    range and {n_weak} more have too little correlation at alpha = {ALPHA_REFERENCE:g} for a
    weight to improve on.

    Columns. `scoring` is the model probability used -- the masked marginal the manuscript
    reports, or the wild-type marginal used before 19 Oct 2026; the DMS rows are the masked arm
    only, which is the one Figure 4 draws. `metric` is the ranking metric, or
    `{DMS_METRIC}` for a DMS row, whose curve is a Spearman rank correlation with the
    laboratory measurement. `model`, `model_label` and `model_role` name the scoring model.
    `scope` is
    `{FOLD}` for one held-out lineage, `{MEAN}` for the summary over them, and `{ALPHA_STAR}`
    for a DMS panel.
    `held_out_lineage` names the lineage taking no part in the choice; `alpha_star` is the
    weight chosen without it and `n_tied_alpha` how many weights shared that training peak;
    `metric_at_alpha_star` and `metric_at_reference` are the held-out lineage's score at
    that weight and at alpha = {ALPHA_REFERENCE:g}. `estimate` is the difference between
    them -- the gain -- on a fold row, and the mean of those gains on the summary row, where
    `ci_lo` and `ci_hi` bracket it. `n_folds_gain_positive` counts the folds whose gain is
    above zero.

    On a DMS row the same three weight columns hold the selected weight, the correlation at it
    and the correlation at alpha = {ALPHA_REFERENCE:g}, and `estimate`, `ci_lo` and `ci_hi`
    hold the resampled median and interval described above. `panel_label`, `organism` and
    `trait` name the scan and the phenotype; `n` and `n_sites` count the replacements measured
    and the protein sites they sit at; `n_boot` is the number of resamples that produced a
    usable curve. `frac_boundary_low` and `frac_boundary_high` are the fractions of those
    resamples whose best weight landed at the negative and positive ends of the searched range.
    `boundary`, `weak_signal` and `interpretable` are the flags of the point estimate, on the
    same rule the figure marks its optima with: **read `alpha_star` only where `interpretable`
    is True.** Columns that belong to the other kind of row are left empty rather than filled
    with a stand-in.

    Intervals on the lineage rows are 95% percentile bootstrap intervals of the mean over
    folds, resampling the
    {n_folds} held-out gains. They are the precision of that mean. The selection is not
    re-run inside a resample, because it has already been made out of sample in every fold;
    and, as for every interval in this project, the lineages share their sites, so the
    interval does not say how a new, unrelated lineage would behave. Intervals on the DMS rows
    are 95% percentile intervals over resampled sites, and there the selection **is** re-run
    inside every resample, because that is the quantity being bracketed. Neither kind says how
    an unrelated lineage or an unmeasured protein would behave.

    **No p-value is given, and none should be computed from these rows.** The lineages are
    not independent replicates -- their reference proteins are 96.6 to 99.7% identical and
    the effective number of independent lineages is between 1.01 and 1.12 -- so a sign test
    or a paired t-test over the folds would be anti-conservative by about that factor. A
    sign test has a second problem on top: {n_folds} of {n_folds} is its floor whatever the
    effect size, so it would report the fold count and call it evidence. The effect size,
    its interval and the fold count are the result. The DMS rows carry none either, and for a
    reason of their own: the question there is where one weight sits, which the interval
    answers directly, and the two models of a panel score the very same replacements, so any
    set of such p-values would be correlated by construction.

    **Source.** The shipped per-lineage alpha sweeps, both scoring arms:
    {folds.groupby(['scoring', 'metric', 'model']).ngroups} combinations of arm, metric and
    model, each resolved into {n_folds} folds. Computed by `mutacc.loo_alpha`; nothing here
    is read from an earlier table of results. The DMS rows come from the same shipped scan
    scores and masked-marginal probabilities Figure 4 is drawn from, searched over the same
    weight grid, and are computed by `mutacc.dms_alpha_ci`; their `alpha_star` is the optimum
    `{SY_STEM}.tsv` reports for the same panel and model, recomputed rather than copied.
    """


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # Positional, and both optional: tools/build_figures.py calls a table exporter as
    # ``<script> <repo> <out_dir>`` with no flags at all.
    parser.add_argument("repo", nargs="?", type=Path, default=REPO_ROOT,
                        help="repository root (default: the one this script lives in)")
    parser.add_argument("out_dir", nargs="?", type=Path, default=None,
                        help="where to write the table and caption (default: <repo>/output)")
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE,
                        help="run guide naming the lineages and models")
    parser.add_argument("--results", type=Path, default=None,
                        help="read the guide's tables from here instead of its results_dir")
    parser.add_argument("--dms-boot", type=int, default=DMS_N_BOOT,
                        help=f"site resamples behind each DMS row (default {DMS_N_BOOT}). "
                             f"Lower values are for checking the shape of the output, not for "
                             f"the shipped table: they move every interval")
    parser.add_argument("--dms-jobs", type=int, default=DMS_JOBS,
                        help=f"processes the DMS bootstrap is spread over (default {DMS_JOBS}); "
                             f"the table does not depend on this")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = args.repo / "output"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    guide = load_guide(args.guide)
    table, alpha_range = build(guide, args.results)
    # The lineage rows first and the DMS rows after, which is the order the caption reads them
    # in; ``scope`` is what a reader filters on either way.
    table = pd.concat([table, build_dms(guide, args.results, n_boot=args.dms_boot,
                                        jobs=args.dms_jobs)], ignore_index=True)
    path = write_values(table, args.out_dir, STEM)
    print(f"wrote {path} ({len(table)} rows)")
    print(f"wrote {write_caption(caption(guide, table, alpha_range), args.out_dir, STEM).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
