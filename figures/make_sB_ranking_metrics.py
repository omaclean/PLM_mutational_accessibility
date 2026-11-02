"""S2: the ranking metrics that the main sweep figure does not carry.

    python figures/make_sB_ranking_metrics.py --guide guides/h3n2_13lineage.yaml

The main figure reports plain AUROC because its chance value, 0.5, is the same for every
lineage. The metrics it displaced are each conditional on something it cannot show, so they
are drawn here on the same sweep with the same band. The two rows ask different questions.

The **top row** asks the main figure's question -- can the score tell the replacements that
happened from the ones that did not -- and scores every possible replacement:

  A  PR-AUC               precision averaged over recall. A random ranker scores the
                          prevalence of observed replacements, which differs between
                          lineages, so there is no single chance line; each lineage's own
                          chance level is drawn as a tick in the right margin.
  B  PR-AUC / prevalence  the same number divided by its own chance level, so chance is 1
                          for every lineage and the lineages can be averaged.
  C  Partial AUROC        the ROC area over low false-positive rates only, standardised so
                          chance is 0.5 and perfect is 1.

The **bottom row** is put only to the replacements that were observed at all. Telling a
replacement that occurred from one that did not is a question about what is possible; these
ask whether a score can order what evolution actually did:

  D  Spearman             rank correlation between the score and the observed frequency.
  E  Pearson, log scale   the same correlation as a linear one, on the logarithm of the
                          frequency rather than the frequency itself. Asked for in review.
  F  Picking the winners  AUROC with the negative class cut down to replacements that
                          occurred: the most frequent few per cent of a lineage's observed
                          replacements against the rest of them. Nothing is won here by
                          knowing which changes are possible, because they all happened.

Where the numbers come from. A, B, C and F are rebuilt from the per-replacement tables:

    score = log10(model probability) + alpha * log10(accessibility)

D and E are read from the columns the sweep table already ships, and are then recomputed from
the same per-replacement tables as a cross-check, as the main figure's AUROC is. Any
disagreement stops the run rather than being drawn.

The raw-scale Pearson correlation is checked against its shipped column with the other two but
is **not** drawn. It is the quantity the review asked to see replaced: observed frequencies
span four orders of magnitude, so on the raw scale a handful of near-fixed replacements set
the answer, and D and E are the two ways of not letting them. It stays in the cross-check so
that dropping it from the panels cannot hide a table that has drifted.

F needs to know how *often* each replacement was observed, which the 0/1 flag table cannot
say. That frequency is shipped by ``tools/export_sB_frequency.py``; nothing here reads the
exploration archive.

A lineage whose top-frequency class cannot be cut -- too few observed replacements, or all of
them at one frequency -- is an error rather than a dropped row, because a band over a changing
set of lineages would mean something different at each end of the axis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from mutacc import style
from mutacc.guide import Guide, parse_figure_args
from mutacc.io import write_caption, write_values
from mutacc.masked_dms import add_scoring_argument
from mutacc.metrics import (bootstrap_mean_ci, lineage_matrix, mcclish_floor, mean_over_lineages,
                            partial_auroc, pr_auc, roc_points, winners_labels)

STEM = "sB_ranking_metrics"
FLOOR = 1e-32           # probabilities are clipped here before the log, as in the pipeline
FPR_CAP = 0.05
TOP_FRACTION = 0.05     # top share of a lineage's observed replacements for winners AUROC
SWEEP_TOL = 1e-6        # recomputed metrics vs the shipped sweep table
KEY = ["lineage", "position", "aa"]
BASELINE_ROLE = "accessibility"
RIGHT_PAD = 0.12        # distance of the accessibility-only marker beyond the last alpha
TICK_PAD = 0.34         # distance of the per-lineage chance ticks (panel A)
SWEEP_VARIANT = "plm_alpha_sweep"
BASELINE_VARIANT = "mutation_accessibility_only"

#: Metric -> the shipped sweep column it is **read from**. Two of these are panels D and E;
#: ``nonzero_pearson``, the raw-scale correlation, is read and cross-checked but not drawn --
#: it is the quantity the review asked to see replaced, and keeping it in the check means
#: dropping it from the panels cannot hide a table that has drifted. See the module docstring.
SWEEP_METRICS = {"nonzero_pearson": "mut_flat_nonzero_pearson_r",
                 "logfreq_pearson": "mut_flat_logfreq_nonzero_pearson_r",
                 "nonzero_spearman": "mut_flat_nonzero_spearman_r"}

#: How closely each of those has to be reproduced: ``(every cell, the 99th percentile of cells)``.
SWEEP_METRIC_TOL = {"nonzero_pearson": (SWEEP_TOL, SWEEP_TOL),
                    "logfreq_pearson": (SWEEP_TOL, SWEEP_TOL),
                    "nonzero_spearman": (1e-3, SWEEP_TOL)}
#: The share of cells held to the tighter of the two tolerances.
BULK_QUANTILE = 0.99
#: Recomputed metric -> the shipped sweep column it is **checked against**. Plain AUROC is not
#: drawn here; agreeing with the published sweep on it is what says these scores are those scores.
SWEEP_CHECKS = {"auroc": "site_logistic_auroc"}

#: Figure height, in inches, for the 2x3 grid at the double-column width. Set from the plot box
#: rather than the page: a row of plots is about 1.7 in tall here, which is what the 1x3 this
#: grew out of settled on, and that leaves them a little wider than they are tall.
FIG_HEIGHT = 4.95
#: Inches constrained_layout leaves between panels, and around the figure, on top of each
#: panel's own tight bounding box.
LAYOUT_PAD = 0.014

#: metric column, title, tagline, y label, chance (None = differs by lineage)
PANELS = [
    ("pr_auc", "PR-AUC", "mean precision over recall;\nchance = prevalence", "PR-AUC", None),
    ("prauc_fold", "PR-AUC ÷ prevalence", "the same score, as fold\nenrichment over chance", "Fold over chance",
     (1.0, "chance = 1", "right")),
    ("pauroc_std", "Partial AUROC", "AUROC for false positive\nrates ≤ {cap}", "Partial AUROC",
     (0.5, "chance = 0.5", "right")),
    ("nonzero_spearman", "Spearman, observed only", "rank correlation across\nnon-zero frequencies only",
     "Spearman ρ", (0.0, "no relationship = 0", "right")),
    ("logfreq_pearson", "Pearson on log frequency", "the same correlation as a linear\none, on the logged frequency",
     "Pearson correlation", (0.0, "no relationship = 0", "right")),
    ("winners_auroc", "Picking the winners", "top {top} by frequency against\nthe rest of the observed ones",
     "AUROC among observed", (0.5, "chance = 0.5", "left")),
]
#: Panels per row. The number of rows is derived from this and the length of ``PANELS``, so
#: adding a seventh panel grows the grid rather than silently dropping off the end of it.
N_COLUMNS = 3


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def sweep_table_name(scoring: str = "masked") -> str:
    return "alpha_sweep_by_lineage_masked" if scoring == "masked" else "alpha_sweep_by_lineage"


def scores_table_name(scoring: str = "masked") -> str:
    return "replacement_scores_masked" if scoring == "masked" else "replacement_scores"


def load_scores(guide: Guide, results: Path, scoring: str = "masked") -> pd.DataFrame:
    """Per-replacement scores joined to the observed flag and frequency. Any mismatch is an error."""
    table = scores_table_name(scoring)
    scores = guide.read_table(table, results)
    guide.check_lineages(scores["lineage"].unique(), f"{guide.name} {table}")
    observed = guide.read_table("replacement_observed", results)
    if observed.duplicated(KEY).any():
        raise ValueError("replacement_observed has duplicate (lineage, position, aa) keys")
    frequency = guide.read_table("replacement_frequency", results)
    if frequency.duplicated(KEY).any():
        raise ValueError("replacement_frequency has duplicate (lineage, position, aa) keys")
    labels = observed.merge(frequency[KEY + ["obs_freq"]], on=KEY, how="outer",
                            validate="one_to_one", indicator=True)
    if (labels["_merge"] != "both").any():
        raise ValueError(f"{int((labels['_merge'] != 'both').sum())} replacements carry a flag "
                         "without a frequency, or a frequency without a flag")
    if ((labels["obs_freq"] > 0) != (labels["obs_present"] > 0)).any():
        raise ValueError("the observed flag and the observed frequency disagree; they are "
                         "derived from one table and must not")
    merged = scores.merge(labels.drop(columns="_merge"), on=KEY, how="left", validate="many_to_one")
    if merged["obs_present"].isna().any():
        raise ValueError(f"{int(merged['obs_present'].isna().sum())} scored replacements have no observed flag")
    unused = len(observed) - len(merged.drop_duplicates(KEY))
    if unused:
        raise ValueError(f"{unused} observed-flag rows match no scored replacement")
    return merged


def alpha_grid(guide: Guide, results: Path, scoring: str = "masked") -> np.ndarray:
    """The sweep figure's own grid, so S2 and Figure 3 share an x-axis."""
    sweep = guide.read_table(sweep_table_name(scoring), results, usecols=["alpha", "model_variant"])
    # The accessibility-only rows carry a placeholder alpha; it is not a point on the sweep.
    grid = sweep.loc[sweep["model_variant"] == SWEEP_VARIANT, "alpha"].round(10)
    if not np.isfinite(grid.to_numpy(float)).all():
        raise ValueError("the sweep table has non-finite alpha values")
    return np.sort(grid.unique())


def log_scores(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lp = np.log10(group["plm_prob"].clip(lower=FLOOR).to_numpy(float))
    lm = np.log10(group["mut_prob"].clip(lower=FLOOR).to_numpy(float))
    return lp, lm, group["obs_present"].to_numpy(float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_row(score: np.ndarray, y: np.ndarray, cap: float,
               nonzero: np.ndarray, winners: np.ndarray) -> dict:
    """The recomputed metrics for one score vector: three over all replacements, one over the observed."""
    prevalence = float(y.mean())
    fpr, tpr = roc_points(score, y)
    _, pstd = partial_auroc(fpr, tpr, cap)
    p = pr_auc(score, y)
    w_fpr, w_tpr = roc_points(score[nonzero], winners)
    return {"prevalence": prevalence, "pr_auc": p, "prauc_fold": p / prevalence,
            "pauroc_std": pstd, "winners_auroc": float(np.trapezoid(w_tpr, w_fpr)),
            "auroc": float(np.trapezoid(tpr, fpr))}


def lineage_parts(group: pd.DataFrame, top: float) -> tuple:
    """Everything one (model, lineage) cell needs, built once for the whole alpha grid."""
    lp, lm, y = log_scores(group)
    nonzero = group["obs_freq"].to_numpy(float) > 0
    freq = group["obs_freq"].to_numpy(float)[nonzero]
    return lp, lm, y, nonzero, freq, winners_labels(freq, top)


def metric_grid(frame: pd.DataFrame, alphas: np.ndarray, cap: float,
                top: float = TOP_FRACTION) -> pd.DataFrame:
    """Per (model, lineage, alpha): the recomputed panel metrics, prevalence and AUROC."""
    rows = []
    for (model, lineage), g in frame.groupby(["model", "lineage"], sort=False):
        lp, lm, y, nonzero, _freq, winners = lineage_parts(g, top)
        for a in alphas:
            rows.append({"model": model, "lineage": lineage, "alpha": float(a),
                         **metric_row(lp + float(a) * lm, y, cap, nonzero, winners)})
    out = pd.DataFrame(rows)
    out["model_variant"] = SWEEP_VARIANT      # the shape lineage_matrix expects
    return out


def sweep_grid(guide: Guide, results: Path, scoring: str = "masked") -> pd.DataFrame:
    """The two correlation panels, read from the columns the sweep table already ships."""
    sweep = guide.read_table(sweep_table_name(scoring), results)
    missing = [c for c in SWEEP_METRICS.values() if c not in sweep.columns]
    if missing:
        raise KeyError(f"the sweep table does not carry {missing}; it cannot draw these panels")
    out = sweep.loc[sweep["model_variant"] == SWEEP_VARIANT,
                    ["model", "lineage", "alpha"] + list(SWEEP_METRICS.values())].copy()
    out["alpha"] = out["alpha"].round(10)
    return out.rename(columns={v: k for k, v in SWEEP_METRICS.items()})


def join_grids(recomputed: pd.DataFrame, shipped: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, lineage, alpha) carrying every panel metric. Any gap is an error."""
    mine = recomputed.copy()
    mine["alpha"] = mine["alpha"].round(10)
    both = mine.merge(shipped, on=["model", "lineage", "alpha"], how="outer", indicator=True)
    if (both["_merge"] != "both").any():
        raise ValueError(f"{int((both['_merge'] != 'both').sum())} (model, lineage, alpha) cells "
                         "are scored by only one of the two routes")
    return both.drop(columns="_merge")


def baseline_grid(frame: pd.DataFrame, guide: Guide, results: Path, cap: float,
                  top: float = TOP_FRACTION, scoring: str = "masked") -> pd.DataFrame:
    """Accessibility alone: score = log10(accessibility), no model term, no alpha.

    The recomputed metrics are scored here directly; the two correlation panels take the sweep
    table's own accessibility-only rows, the same place their swept values come from.
    """
    rows = []
    for (model, lineage), g in frame.groupby(["model", "lineage"], sort=False):
        _lp, lm, y, nonzero, _freq, winners = lineage_parts(g, top)
        rows.append({"model": model, "lineage": lineage,
                     **metric_row(lm, y, cap, nonzero, winners)})
    out = pd.DataFrame(rows)
    # It cannot depend on the model; prove that instead of assuming it.
    for m in ("pr_auc", "pauroc_std", "winners_auroc", "auroc"):
        wide = out.pivot(index="lineage", columns="model", values=m)
        if not np.allclose(wide.to_numpy(), wide.iloc[:, [0]].to_numpy(), atol=1e-12):
            raise ValueError(f"accessibility-only baseline differs between models on {m!r}")

    sweep = guide.read_table(sweep_table_name(scoring), results)
    only = sweep.loc[sweep["model_variant"] == BASELINE_VARIANT,
                     ["model", "lineage"] + list(SWEEP_METRICS.values())]
    if only.empty:
        raise ValueError("the sweep table has no accessibility-only rows to take a baseline from")
    for column in SWEEP_METRICS.values():
        wide = only.pivot(index="lineage", columns="model", values=column)
        if not np.allclose(wide.to_numpy(), wide.iloc[:, [0]].to_numpy(), atol=1e-9):
            raise ValueError(f"the shipped accessibility-only baseline differs between models on {column!r}")
    only = only.drop_duplicates("lineage").rename(columns={v: k for k, v in SWEEP_METRICS.items()})
    merged = out.drop_duplicates("lineage").merge(
        only.drop(columns="model"), on="lineage", how="outer", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("the recomputed and shipped accessibility-only baselines cover different lineages")
    return merged.drop(columns="_merge")


def check_against_sweep(grid: pd.DataFrame, guide: Guide, results: Path, tol: float = SWEEP_TOL,
                        scoring: str = "masked") -> float:
    """Recomputed AUROC must equal the shipped sweep table's. Returns the largest gap."""
    sweep = guide.read_table(sweep_table_name(scoring), results)
    column = SWEEP_CHECKS["auroc"]
    sweep = sweep.loc[sweep["model_variant"] == SWEEP_VARIANT,
                      ["model", "lineage", "alpha", column]].copy()
    sweep["alpha"] = sweep["alpha"].round(10)
    mine = grid[["model", "lineage", "alpha", "auroc"]].copy()
    mine["alpha"] = mine["alpha"].round(10)
    both = mine.merge(sweep, on=["model", "lineage", "alpha"], how="outer", indicator=True)
    if (both["_merge"] != "both").any():
        raise ValueError(f"{int((both['_merge'] != 'both').sum())} (model, lineage, alpha) cells are in only one table")
    gap = float((both["auroc"] - both[column]).abs().max())
    if gap > tol:
        raise ValueError(f"recomputed AUROC disagrees with the shipped sweep table by {gap:.2e} (tolerance {tol:g})")
    return gap


def recompute_sweep_metrics(frame: pd.DataFrame, alphas: np.ndarray) -> pd.DataFrame:
    """Rebuild the two correlation panels from the per-replacement tables.

    Both are scored over the replacements observed at all: the rank correlation with the
    frequency, and the linear correlation with its logarithm.
    """
    rows = []
    for (model, lineage), g in frame.groupby(["model", "lineage"], sort=False):
        lp, lm, _y = log_scores(g)
        nonzero = g["obs_freq"].to_numpy(float) > 0
        freq = g["obs_freq"].to_numpy(float)[nonzero]
        log_freq = np.log10(freq)
        lp, lm = lp[nonzero], lm[nonzero]
        for a in alphas:
            score = lp + float(a) * lm
            rows.append({
                "model": model,
                "lineage": lineage,
                "alpha": float(a),
                "nonzero_pearson": float(pearsonr(score, freq).statistic),
                "logfreq_pearson": float(pearsonr(score, log_freq).statistic),
                "nonzero_spearman": float(spearmanr(score, freq).statistic),
            })
    return pd.DataFrame(rows)


def check_sweep_metrics(frame: pd.DataFrame, shipped: pd.DataFrame, alphas: np.ndarray,
                        tolerances: dict | None = None) -> dict:
    """The read columns must equal the same correlations rebuilt from the scores and frequencies.

    The panels are drawn from the shipped columns. This is what entitles them to be: it says
    the sweep table, the per-replacement scores and the frequency table are three views of one
    run rather than three vintages that happen to sit in the same directory.

    Each metric is held to two tolerances, one on every cell and a tighter one on the bulk of
    them; see ``SWEEP_METRIC_TOL`` for why a rank correlation needs both. Returns
    ``{metric: (largest gap, gap at the bulk quantile)}``.
    """
    tolerances = SWEEP_METRIC_TOL if tolerances is None else tolerances
    mine = recompute_sweep_metrics(frame, alphas)
    mine["alpha"] = mine["alpha"].round(10)
    both = mine.merge(shipped, on=["model", "lineage", "alpha"], how="outer",
                      suffixes=("_mine", "_shipped"), indicator=True)
    if (both["_merge"] != "both").any():
        raise ValueError(f"{int((both['_merge'] != 'both').sum())} (model, lineage, alpha) cells "
                         "are in only one of the recomputed and shipped correlation tables")
    gaps = {}
    for metric, column in SWEEP_METRICS.items():
        every, bulk_tol = tolerances[metric]
        difference = (both[f"{metric}_mine"] - both[f"{metric}_shipped"]).abs()
        worst, bulk = float(difference.max()), float(difference.quantile(BULK_QUANTILE))
        if worst > every:
            raise ValueError(f"the shipped {column!r} column disagrees with the same correlation "
                             f"rebuilt from the scores by {worst:.2e} in at least one cell "
                             f"(tolerance {every:g})")
        if bulk > bulk_tol:
            raise ValueError(f"the shipped {column!r} column disagrees with the same correlation "
                             f"rebuilt from the scores by {bulk:.2e} across {BULK_QUANTILE:.0%} of "
                             f"cells (tolerance {bulk_tol:g}); that is a different quantity, not "
                             "rounding")
        gaps[metric] = (worst, bulk)
    return gaps


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def bands(grid: pd.DataFrame, guide: Guide, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for metric, *_ in PANELS:
        for model in guide.models:
            wide = lineage_matrix(grid, metric, model=model.key)
            lo, hi = bootstrap_mean_ci(wide, n_boot=n_boot, seed=seed)
            for a, m, l, h in zip(wide.columns, mean_over_lineages(wide), lo, hi):
                rows.append({"metric": metric, "model": model.key, "alpha": a, "mean": m, "lo": l, "hi": h})
    return pd.DataFrame(rows)


def peak(band: pd.DataFrame, metric: str, model: str) -> tuple[float, float, float]:
    """Where a curve is highest: ``(alpha, value there, value at alpha = 0)``."""
    sub = band[(band["metric"] == metric) & (band["model"] == model)].sort_values("alpha")
    best = sub.loc[sub["mean"].idxmax()]
    at_zero = sub.loc[np.isclose(sub["alpha"], 0.0), "mean"]
    if len(at_zero) != 1:
        raise ValueError(f"the grid does not contain alpha = 0 exactly for {metric!r}")
    return float(best["alpha"]), float(best["mean"]), float(at_zero.iloc[0])


def draw(base: pd.DataFrame, band: pd.DataFrame, guide: Guide, cap: float, top: float = TOP_FRACTION):
    fig = style.figure(style.WIDTH_DOUBLE, FIG_HEIGHT)
    # The engine's defaults leave a gutter wide enough to read as a break in the figure, and
    # this grid has nothing to put in it: every label and header is inside its panel's own box.
    fig.get_layout_engine().set(w_pad=LAYOUT_PAD, h_pad=LAYOUT_PAD, wspace=0.0, hspace=0.0)
    n_rows = -(-len(PANELS) // N_COLUMNS)
    # sharex hides the tick labels everywhere but the bottom row, so the weight is read off
    # the bottom of the figure and labelled once underneath it.
    axes = np.asarray(fig.subplots(n_rows, N_COLUMNS, sharex=True)).reshape(-1)
    x_lo, x_hi = float(band["alpha"].min()), float(band["alpha"].max())
    prevalence = base.sort_values("lineage")["prevalence"].to_numpy()
    for ax, (metric, title, tagline, ylabel, ref) in zip(axes, PANELS):
        for model in guide.models:
            colour = style.colour_for_role(model.role)
            sub = band[(band["metric"] == metric) & (band["model"] == model.key)].sort_values("alpha")
            ax.fill_between(sub["alpha"], sub["lo"], sub["hi"], color=colour, alpha=0.18, linewidth=0)
            ax.plot(sub["alpha"], sub["mean"], color=colour, label=model.label)
        ax.scatter([x_hi + RIGHT_PAD], [float(base[metric].mean())], marker="s", s=22, clip_on=False, zorder=5,
                   color=style.colour_for_role(BASELINE_ROLE), label="Accessibility alone")
        if ref is not None:
            style.reference_line(ax, ref[0], ref[1], label_loc=ref[2])
            style.cut_axis_below(ax, ref[0])
        else:
            ax.plot([x_hi + TICK_PAD] * len(prevalence), prevalence, linestyle="none", marker="_",
                    markersize=5, color=style.NULL_GREY, clip_on=False, label="Chance, one tick per lineage")
            ax.set_ylim(bottom=0.0)
        style.reference_line(ax, 0.0, "α = 0", axis="x", label_side="below",
                             color=style.INK_MUTED, zorder=2, **style.ALPHA_ZERO_RULE)
        ax.set_xlim(x_lo - 0.05, x_hi + 2 * RIGHT_PAD)
        style.finalise_axes(ax, ylabel=ylabel)
        style.panel_title(ax, title, tagline.replace("{cap}", f"{cap:.0%}").replace("{top}", f"{top:.0%}"))
    fig.supxlabel("Weight on mutational accessibility (α); 0 = model alone", fontsize=8)
    # Panel A is the only one with the per-lineage chance ticks, so its handles are the full set.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), handlelength=1.4)
    style.panel_letters(axes, dx=-0.20, dy=1.14)
    return fig


def caption(base: pd.DataFrame, band: pd.DataFrame, guide: Guide, cap: float, top: float,
            n_boot: int, n_observed: pd.Series) -> str:
    n = guide.n_lineages
    lo, hi = base["prevalence"].min(), base["prevalence"].max()
    a = band["alpha"]
    roles = {m.role: m for m in guide.models}
    gains = {}
    for role, model in roles.items():
        for metric in ("nonzero_spearman", "logfreq_pearson", "winners_auroc"):
            a_star, best, at_zero = peak(band, metric, model.key)
            gains[role, metric] = (a_star, best, at_zero)

    def phrase(role: str, metric: str) -> str:
        """``0.7394 to 0.7394 at α = 0``: four decimals because panel F's fine-tuned gain is
        a few ten-thousandths, and rounding it to three would print a curve moving from a
        number to itself, which reads as a slip rather than as the result. A peak sitting at
        zero is written "0", not "+0"; it means the sweep never improves on the model alone."""
        a_star, best, at_zero = gains[role, metric]
        where = "0" if a_star == 0 else f"{a_star:+g}"
        return f"{at_zero:.4f} to {best:.4f} at α = {where}"

    return f"""
    **Six ranking metrics across the accessibility weight.**
    The main figure reports AUROC over every possible replacement. These are the metrics it
    displaced, together with two that ask a narrower question, drawn on the same sweep
    (weight α from {a.min():g} to {a.max():g}; 0 is the model alone). Lines are means over the
    {n} lineages and the orange square is accessibility with no model.

    **The top row scores every possible replacement**, observed against unobserved, which is
    the main figure's question. (A) PR-AUC, the area under the precision-recall curve, and the
    metric first reported. A random ranker scores the prevalence -- the fraction of possible
    replacements that were observed -- and prevalence runs from {lo:.1%} to {hi:.1%} between
    lineages. There is therefore no single chance line; each lineage's own chance level is a
    grey tick in the right margin, and the mean curve is shown because it was reported, not
    because the quantity it averages is comparable between lineages. (B) The same number
    divided by each lineage's prevalence: fold enrichment over chance, where chance is 1 for
    every lineage and the mean is meaningful. (C) Partial AUROC: the ROC area over
    false-positive rates up to {cap:.0%}, standardised so that chance is 0.5 and a perfect
    ranking is 1. The metric cannot fall below {mcclish_floor(cap):.4f}, so everything worse
    than chance is compressed into the gap beneath 0.5.

    **The bottom row is put only to the replacements that were observed at all**, between
    {int(n_observed.min()):,} and {int(n_observed.max()):,} of them per lineage. (D) Spearman
    rank correlation between the score and the observed frequency. (E) The same correlation as
    a linear one, computed on the logarithm of the frequency rather than on the frequency
    itself, as asked in review; observed frequencies span four orders of magnitude, so on the
    raw scale the few replacements that went nearly to fixation set the answer, and the rank
    correlation in D and the log scale in E are the two ways of not letting them. (F) AUROC
    again, with the negative class cut down to replacements that occurred: positives are the
    most frequent {top:.0%} of a lineage's observed replacements and negatives are the rest of
    them, so the zero-frequency replacements take no part. Panel F asks whether a score can
    tell a replacement that swept from one that merely appeared.

    **What the bottom row shows.** The fine-tuned model gains essentially nothing from the
    explicit accessibility term on any of the three: {phrase('finetune', 'nonzero_spearman')}
    in D, {phrase('finetune', 'logfreq_pearson')} in E, and
    {phrase('finetune', 'winners_auroc')} in F, every one of them inside the width of its own
    band. The base model does gain materially on D and E --
    {phrase('base', 'nonzero_spearman')} and {phrase('base', 'logfreq_pearson')}, still rising
    at the end of the sweep -- and much less on F, {phrase('base', 'winners_auroc')}.
    Accessibility on its own scores {float(base['winners_auroc'].mean()):.3f} at picking the
    winners in F, against {float(base['auroc'].mean()):.3f} at saying which replacements occur
    at all, the quantity Figure 3 reports. That contrast is the point of this row: mutational
    accessibility is informative about what is reachable and close to uninformative about what
    succeeds once it is reached, and a model that has learned the reachable part has nothing
    left to gain from being told it again.

    Bands are 95% percentile bootstrap intervals of the mean, resampling lineages
    ({n_boot:,} resamples). They show the precision of the mean over these {n} lineages.
    Lineages are disjoint sets of sequences but can share sites, so the bands do not show how
    well a model would do on a new lineage. Every panel but A stops its vertical axis just
    under chance; below it the score is anti-predictive. Every plotted value, including those,
    is in the values table. Panels D and E are the sweep table's own columns; they, the
    raw-scale Pearson correlation that is checked but not drawn, and the scores behind the
    other four panels are checked against each other before the figure is drawn, and a
    disagreement stops the run.
    """


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_scoring_argument(pre)
    pre.add_argument("--fpr-cap", type=float, default=FPR_CAP)
    pre.add_argument("--top-fraction", type=float, default=TOP_FRACTION)
    pre.add_argument("--n-boot", type=int, default=2000)
    pre.add_argument("--seed", type=int, default=20261009)
    known, rest = pre.parse_known_args(argv)
    args = parse_figure_args(__doc__.split("\n\n")[0], rest)
    guide = args.guide

    frame = load_scores(guide, args.results, scoring=known.scoring)
    for m in guide.models:
        if m.key not in set(frame["model"]):
            raise ValueError(f"guide model {m.key!r} is not in {scores_table_name(known.scoring)}")
    alphas = alpha_grid(guide, args.results, scoring=known.scoring)
    recomputed = metric_grid(frame, alphas, known.fpr_cap, known.top_fraction)
    shipped = sweep_grid(guide, args.results, scoring=known.scoring)
    grid = join_grids(recomputed, shipped)
    base = baseline_grid(frame, guide, args.results, known.fpr_cap, known.top_fraction, scoring=known.scoring)
    gap = check_against_sweep(recomputed, guide, args.results, scoring=known.scoring)
    gaps = check_sweep_metrics(frame, shipped, alphas)
    band = bands(grid, guide, known.n_boot, known.seed)
    n_observed = frame.drop_duplicates(KEY).query("obs_freq > 0").groupby("lineage").size()

    fig = draw(base, band, guide, known.fpr_cap, known.top_fraction)
    paths = style.save_figure(fig, args.out / STEM)
    out = band.copy()
    out["baseline_mean"] = out["metric"].map({m: float(base[m].mean()) for m, *_ in PANELS})
    write_values(out, args.out, STEM)
    # ``auroc`` is not a panel but is what panel F's number is read against in the caption.
    write_values(base[["lineage", "prevalence"] + [m for m, *_ in PANELS] + ["auroc"]],
                 args.out, "sB_accessibility_only_by_lineage")
    write_caption(caption(base, band, guide, known.fpr_cap, known.top_fraction, known.n_boot, n_observed),
                  args.out, STEM)
    print(f"self-check: AUROC recomputed from the scores agrees with the sweep table to {gap:.1e}")
    for name, (worst, bulk) in gaps.items():
        print(f"self-check: the shipped {name!r} column agrees with the same correlation rebuilt "
              f"from the scores to {worst:.1e} everywhere, {bulk:.1e} across {BULK_QUANTILE:.0%} of cells")
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
