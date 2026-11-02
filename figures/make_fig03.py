"""Figure 3: how well the model predicts evolution, as accessibility is weighted in.

    python figures/make_fig03.py --guide guides/h3n2_13lineage.yaml

Reads the guide's ``alpha_sweep_by_lineage_masked`` table (or ``alpha_sweep_by_lineage`` under
``--scoring unmasked``). Draws one curve per model in the guide (coloured by role), a band that
is the bootstrap interval of the mean over lineages, and the accessibility-only baseline as a point.

The model probability is the **masked marginal**: each position is masked in turn and predicted
from its context. ``--scoring unmasked`` redraws the single unmasked pass version.

Panels:
  A  Spearman correlation between PLM score and observed frequency over all possible replacements.
  B  AUROC over all possible replacements: distinguishing observed from unobserved replacements.
  C  AUROC among observed replacements: distinguishing the top 5% most frequent circulating
     replacements from the rest of the observed replacements ("picking the winners").
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from mutacc import style
from mutacc.guide import Guide, parse_figure_args
from mutacc.io import write_caption, write_values
from mutacc.masked_dms import add_scoring_argument
from mutacc.metrics import bootstrap_mean_ci, lineage_matrix, mean_over_lineages, roc_points, winners_labels

STEM = "fig03_alpha_sweep"

#: (column, metric name, plain-words tagline, y label, reference line). The column
#: names are the table's schema. The title names the metric exactly; the tagline says
#: in plain words what is being asked and is deliberately smaller.
PANELS = [
    ("mut_flat_global_spearman_r", "Spearman correlation",
     "PLM score vs observed frequency,\nall possible AA replacements", "Spearman ρ", (0.0, "no relationship = 0")),
    ("site_logistic_auroc", "AUROC: all possible AAs",
     "Binary prediction: AA\nreplacement observed or not", "AUROC", (0.5, "chance = 0.5")),
    ("winners_auroc", "AUROC: top 5% circulating",
     "Top 5% by frequency vs\nother observed replacements", "AUROC", (0.5, "chance = 0.5")),
]

#: Metrics that are not plotted as a panel, but are computed and included in the values table.
EXTRA_TABLE_METRICS = [
    "mut_flat_nonzero_pearson_r",
    "mut_flat_nonzero_spearman_r",
]

TABLE_METRICS = [p[0] for p in PANELS] + EXTRA_TABLE_METRICS

BASELINE_ROLE = "accessibility"
BASELINE_X_PAD = 0.12

TOP_FRACTION = 0.05
FLOOR = 1e-32
KEY = ["lineage", "position", "aa"]

#: alpha = 0 is the reference the whole figure is read against -- it is the model with no
#: accessibility weighting at all. It gets a full reference line, not a hairline.
ALPHA_REFERENCE = 0.0


def compute_winners_auroc(
    guide: Guide,
    results: Path | None = None,
    alphas: np.ndarray | None = None,
    top: float = TOP_FRACTION,
    scoring: str = "masked",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes winners_auroc across (model, lineage, alpha) and for the accessibility baseline."""
    score_table = (
        "replacement_scores_masked"
        if scoring == "masked" and "replacement_scores_masked" in guide.tables
        else "replacement_scores"
    )
    scores = guide.read_table(score_table, results)
    guide.check_lineages(scores["lineage"].unique(), f"{guide.name} {score_table}")
    observed = guide.read_table("replacement_observed", results)
    frequency = guide.read_table("replacement_frequency", results)
    labels = observed.merge(
        frequency[KEY + ["obs_freq"]], on=KEY, how="outer", validate="one_to_one"
    )
    merged = scores.merge(labels, on=KEY, how="left", validate="many_to_one")

    if alphas is None:
        sweep_table = (
            "alpha_sweep_by_lineage_masked"
            if scoring == "masked" and "alpha_sweep_by_lineage_masked" in guide.tables
            else "alpha_sweep_by_lineage"
        )
        sweep = guide.read_table(sweep_table, results, usecols=["alpha", "model_variant"])
        grid = sweep.loc[sweep["model_variant"] == "plm_alpha_sweep", "alpha"].round(10)
        alphas = np.sort(grid.unique())

    sweep_rows = []
    base_rows = []
    for (model, lineage), g in merged.groupby(["model", "lineage"], sort=False):
        lp = np.log10(g["plm_prob"].clip(lower=FLOOR).to_numpy(float))
        lm = np.log10(g["mut_prob"].clip(lower=FLOOR).to_numpy(float))
        nonzero = g["obs_freq"].to_numpy(float) > 0
        freq = g["obs_freq"].to_numpy(float)[nonzero]
        winners = winners_labels(freq, top)
        lp_nz, lm_nz = lp[nonzero], lm[nonzero]

        for a in alphas:
            score = lp_nz + float(a) * lm_nz
            w_fpr, w_tpr = roc_points(score, winners)
            sweep_rows.append({
                "model": model,
                "lineage": lineage,
                "alpha_round": round(float(a), 10),
                "winners_auroc": float(np.trapezoid(w_tpr, w_fpr)),
            })

        w_fpr, w_tpr = roc_points(lm_nz, winners)
        base_rows.append({
            "model": model,
            "lineage": lineage,
            "winners_auroc": float(np.trapezoid(w_tpr, w_fpr)),
        })

    return pd.DataFrame(sweep_rows), pd.DataFrame(base_rows)


def ensure_winners_auroc(
    frame: pd.DataFrame,
    guide: Guide,
    *,
    results: Path | None = None,
    top: float = TOP_FRACTION,
    scoring: str = "masked",
) -> pd.DataFrame:
    """Ensure frame contains the 'winners_auroc' column, computing it if missing."""
    if "winners_auroc" in frame.columns:
        return frame

    sweep_alphas = np.sort(
        frame.loc[frame["model_variant"] == "plm_alpha_sweep", "alpha"].round(10).unique()
    )
    sweep_df, base_df = compute_winners_auroc(
        guide, results, alphas=sweep_alphas, top=top, scoring=scoring
    )

    out = frame.copy()
    out["alpha_round"] = out["alpha"].round(10)

    sweep_mask = out["model_variant"] == "plm_alpha_sweep"
    base_mask = out["is_mutation_only_baseline"].astype(bool)

    s_merged = out.loc[sweep_mask].merge(
        sweep_df, on=["model", "lineage", "alpha_round"], how="left"
    )
    b_merged = out.loc[base_mask].merge(
        base_df, on=["model", "lineage"], how="left"
    )
    other = out.loc[~sweep_mask & ~base_mask].copy()
    if not other.empty:
        other["winners_auroc"] = np.nan

    combined = pd.concat([s_merged, b_merged, other], ignore_index=True).drop(
        columns=["alpha_round"]
    )
    return combined


def band_table(
    frame: pd.DataFrame,
    guide: Guide,
    *,
    metrics: list[str] | tuple[str, ...] | None = None,
    n_boot: int = 2000,
    seed: int = 20261009,
    results: Path | None = None,
) -> pd.DataFrame:
    """One row per (panel metric, model, alpha): mean and bootstrap interval."""
    if metrics is None:
        metrics = TABLE_METRICS
    if "winners_auroc" in metrics and "winners_auroc" not in frame.columns:
        frame = ensure_winners_auroc(frame, guide, results=results)
    rows = []
    for metric in metrics:
        for model in guide.models:
            wide = lineage_matrix(frame, metric, model=model.key)
            if wide.shape[0] != guide.n_lineages:
                raise ValueError(
                    f"{metric}/{model.key}: table has {wide.shape[0]} lineages, guide declares "
                    f"{guide.n_lineages}"
                )
            if set(wide.index) != set(guide.lineage_ids):
                raise ValueError(f"{metric}/{model.key}: lineages in table differ from the guide")
            lo, hi = bootstrap_mean_ci(wide, n_boot=n_boot, seed=seed)
            mean = mean_over_lineages(wide)
            for a, m, l, h in zip(wide.columns, mean, lo, hi):
                rows.append({"metric": metric, "model": model.key, "alpha": a, "mean": m, "lo": l, "hi": h})
    return pd.DataFrame(rows)


def baseline_table(
    frame: pd.DataFrame,
    guide: Guide,
    *,
    metrics: list[str] | tuple[str, ...] | None = None,
    n_boot: int = 2000,
    seed: int = 20261009,
    results: Path | None = None,
) -> pd.DataFrame:
    """Accessibility-only score, averaged over lineages with bootstrap intervals."""
    if metrics is None:
        metrics = TABLE_METRICS
    if "winners_auroc" in metrics and "winners_auroc" not in frame.columns:
        frame = ensure_winners_auroc(frame, guide, results=results)
    base = frame.loc[frame["is_mutation_only_baseline"].astype(bool)]
    # The baseline does not depend on the model; keep one model's rows, and check.
    per_model = {k: g for k, g in base.groupby("model")}
    first = next(iter(per_model.values()))
    for key, other in per_model.items():
        for metric in metrics:
            if metric in other and not np.allclose(
                other.sort_values("lineage")[metric].to_numpy(),
                first.sort_values("lineage")[metric].to_numpy(),
            ):
                raise ValueError(f"accessibility-only baseline differs between models on {metric!r} ({key})")
    if first["lineage"].nunique() != guide.n_lineages:
        raise ValueError("baseline rows do not cover every lineage in the guide")
    rows = []
    for m in metrics:
        s = first.set_index("lineage")[m]
        wide = s.to_frame()
        lo, hi = bootstrap_mean_ci(wide, n_boot=n_boot, seed=seed)
        rows.append({"metric": m, "mean": float(s.mean()), "lo": float(lo[0]), "hi": float(hi[0])})
    return pd.DataFrame(rows)


def draw(frame: pd.DataFrame, guide: Guide, *, results: Path | None = None):
    if "winners_auroc" not in frame.columns:
        frame = ensure_winners_auroc(frame, guide, results=results)
    bands = band_table(frame, guide, results=results)
    baseline = baseline_table(frame, guide, results=results).set_index("metric")["mean"]

    fig = style.figure(style.WIDTH_DOUBLE, 3.25)
    axes = fig.subplots(1, len(PANELS), sharex=True)
    x_hi = float(bands["alpha"].max())
    x_lo = float(bands["alpha"].min())

    for ax, (metric, title, tagline, ylabel, ref) in zip(axes, PANELS):
        for model in guide.models:
            colour = style.colour_for_role(model.role)
            sub = bands[(bands["metric"] == metric) & (bands["model"] == model.key)].sort_values("alpha")
            ax.fill_between(sub["alpha"], sub["lo"], sub["hi"], color=colour, alpha=0.18, linewidth=0)
            ax.plot(sub["alpha"], sub["mean"], color=colour, label=model.label)
            # Nothing is drawn at alpha = 0, but the rule claims the curve has a value there,
            # so the grid is still checked for it.
            at_zero = sub.loc[np.isclose(sub["alpha"], ALPHA_REFERENCE), "mean"]
            if len(at_zero) != 1:
                raise ValueError(f"{metric}/{model.key}: {len(at_zero)} rows at alpha = {ALPHA_REFERENCE}, expected 1")
        ax.scatter(
            [x_hi + BASELINE_X_PAD], [baseline[metric]], marker="s", s=22,
            color=style.colour_for_role(BASELINE_ROLE), zorder=5, clip_on=False,
            label="Accessibility alone",
        )
        style.reference_line(ax, ref[0], ref[1], label_loc="right")
        style.reference_line(ax, ALPHA_REFERENCE, "α = 0", axis="x", label_side="below",
                             color=style.INK_MUTED, zorder=2, **style.ALPHA_ZERO_RULE)
        ax.set_xlim(x_lo - 0.05, x_hi + 2 * BASELINE_X_PAD)
        style.cut_axis_below(ax, ref[0])
        style.finalise_axes(ax, ylabel=ylabel)
        style.panel_title(ax, title, tagline)
    fig.supxlabel(style.ALPHA_SWEEP_XLABEL, fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    style.legend_above(fig, handles, labels, ncol=len(labels), handlelength=1.4)
    style.panel_letters(axes)
    return fig, bands, baseline


def hidden_text(bands: pd.DataFrame) -> str:
    """Which panels have curves under the cut line, read from the bands themselves."""
    letters = "ABC"[:len(PANELS)]
    hidden = [letter for letter, (metric, _, _, _, ref) in zip(letters, PANELS)
              if (bands.loc[bands["metric"] == metric, "mean"] < ref[0]).any()]
    rest = [letter for letter in letters if letter not in hidden]
    if not hidden:
        return "No curve falls below the line in any panel."
    names = " and ".join(f"({h})" for h in hidden)
    tail = f" In {' and '.join(f'({r})' for r in rest)} no curve falls below the line." if rest else ""
    return f"In {names} the curves fall under that line for some weights and are not drawn there.{tail}"


def caption(guide: Guide, bands: pd.DataFrame) -> str:
    n = guide.n_lineages
    hidden_text_ = hidden_text(bands)
    finetune_key = next((m.key for m in guide.models if m.role == "finetune"), guide.models[0].key)
    sub_c = bands[(bands["metric"] == "winners_auroc") & (bands["model"] == finetune_key)]
    if not sub_c.empty:
        best_c = sub_c.loc[sub_c["mean"].idxmax()]
        zero_c = sub_c.loc[np.isclose(sub_c["alpha"], 0.0), "mean"].iloc[0]
        phrase_c = f"{zero_c:.4f} at α = 0 to {best_c['mean']:.4f} at best α"
    else:
        phrase_c = "flat across α"
    return f"""
    **Prediction of observed amino-acid replacements as the weight on
    mutational accessibility changes.**
    Each panel scores how well the product of the model's pseudo-probability and accessibility^α
    predicts replacements seen in sequence data, as the weight α on accessibility
    runs from {bands['alpha'].min():g} to {bands['alpha'].max():g}. α = 0 is the model alone:
    it is marked by the dashed vertical rule, and each curve is read off where it crosses
    that rule. The orange square is accessibility with no model at all.
    The unit throughout is one candidate replacement -- each of the 19 non-reference amino
    acids at every site, scored within a lineage. "PLM score" in the panel subtitles is the
    quantity being swept: the protein language model's **pseudo-probability** for that amino
    acid at that site -- the probability the masked model assigns it, which is a score over the
    20 amino acids and not a calibrated probability that the replacement will happen --
    multiplied by accessibility^α. At α = 0 it is the pseudo-probability alone.
    (A) Spearman rank correlation between the score and the observed frequency of every
    possible replacement, including the ones never seen (frequency 0). (B) AUROC over all
    possible replacements (area under the receiver-operating-characteristic curve):
    the chance that a replacement that occurs scores higher than one that does not; 0.5 is chance.
    (C) AUROC among observed replacements: distinguishing the most frequent 5% of observed
    replacements from the remaining observed replacements ("picking the winners"). While
    mutational accessibility strongly predicts which replacements occur at all ((B), accessibility
    alone = 0.909), it provides almost no signal for distinguishing high-frequency from low-frequency
    circulating variants ((C), accessibility alone = 0.524, near chance). Weighting in mutational
    accessibility provides essentially no additional predictive gain for the fine-tuned model
    in (C) ({phrase_c}), indicating that mutational accessibility governs
    evolutionary reachability rather than post-emergence fitness or fixation.

    Each vertical axis stops at the no-relationship level (0, or 0.5 for AUROC).
    Below it the score is anti-predictive. {hidden_text_} All plotted values, including
    those, are in the accompanying values table.

    Lines are means over {n} lineages, each scored separately. Bands are 95%
    percentile bootstrap intervals of that mean, resampling lineages. They show the
    precision of the mean over these {n} lineages. Lineages are disjoint sets of
    sequences but can share sites, so the bands do not show how well a model would do
    on a new, unrelated lineage.
    """


def main(argv=None) -> int:
    args = parse_figure_args(__doc__.split("\n\n")[0], argv, extra=add_scoring_argument)
    guide = args.guide
    table_name = (
        "alpha_sweep_by_lineage_masked"
        if args.scoring == "masked"
        else "alpha_sweep_by_lineage"
    )
    frame = guide.read_table(table_name, args.results)
    frame = ensure_winners_auroc(frame, guide, results=args.results, scoring=args.scoring)
    fig, bands, baseline = draw(frame, guide, results=args.results)
    paths = style.save_figure(fig, args.out / STEM)
    base_df = baseline_table(frame, guide, results=args.results).set_index("metric")
    out = bands.copy()
    out["baseline_mean"] = out["metric"].map(base_df["mean"])
    out["baseline_lo"] = out["metric"].map(base_df["lo"])
    out["baseline_hi"] = out["metric"].map(base_df["hi"])
    base_rows = []
    for m in TABLE_METRICS:
        base_rows.append({
            "metric": m,
            "model": "Mutational_accessibility_alone",
            "alpha": np.nan,
            "mean": float(base_df.loc[m, "mean"]),
            "lo": float(base_df.loc[m, "lo"]),
            "hi": float(base_df.loc[m, "hi"]),
            "baseline_mean": float(base_df.loc[m, "mean"]),
            "baseline_lo": float(base_df.loc[m, "lo"]),
            "baseline_hi": float(base_df.loc[m, "hi"]),
        })
    out = pd.concat([out, pd.DataFrame(base_rows)], ignore_index=True)
    write_values(out, args.out, STEM)
    write_caption(caption(guide, bands), args.out, STEM)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())

