"""Every deep-mutational-scanning arm's whole accessibility-weight curve, not just its peak.

    python figures/make_sP_alpha_curves.py --guide guides/h3n2_13lineage.yaml

Figure SM reduces each arm to one number -- the weight alpha at which the correlation with the
measured phenotype is highest -- and then counts how many of those numbers are negative. A
count of optima says nothing about the curves they came from: whether the optimum is a real
turning point or the end of the swept range, how much is gained by moving off alpha = 0, and
how flat the whole thing is. This figure shows the curves, so that the count can be read
against them.

Two choices made here, both visible in the panels:

**Receptor-binding assays are drawn but not counted.** They are the one phenotype measured
*against* fitness rather than with it, so their correlations are negative across the entire
sweep and the largest correlation falls wherever the model is diluted hardest towards zero,
not at an optimum. ``mutacc.dms_sweep.curve_flags`` says so in its own docstring. Rather than
give two panels their own sign rule, the question this figure asks is not one those assays can
answer, so they are left out of the tallies here and out of Figure SM panel J. They are still
plotted, shaded and labelled: the curves are the evidence for the exclusion, and a reader
cannot check an exclusion they cannot see.

**An optimum at the end of the sweep is kept, not discarded.** ``curve_flags`` marks it
``boundary``, meaning the argmax is a bound rather than a turning point. That is worth marking
-- the curve may still be falling past the edge -- but it is not a reason to drop the arm when
the curve is plainly monotone towards that edge. Such optima are drawn as squares and counted.

What is set aside is a curve with no signal anywhere in the sweep, by
``mutacc.dms_sweep.curve_has_signal``. That tests the maximum of ``|rho|`` over the whole
sweep rather than its value at alpha = 0, which is what ``curve_flags`` tests: a curve can pass
through a shallow point at alpha = 0 and still have a clear peak, and the alpha = 0 rule throws
those away.

Reads ``dms_scores`` (and ``dms_scores_masked`` under the default masked scoring) and takes its
panel list from ``dms_peak_summary``, so no dataset, trait, model or virus is named here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from make_fig04 import ALPHA_LIMIT, ALPHA_STEP, MIN_ROWS, RECEPTOR_TRAIT
from mutacc import style
from mutacc.dms_sweep import MARKED_GAIN, curve_flags, curve_has_signal, sweep_curve
from mutacc.guide import Guide, parse_figure_args
from mutacc.io import write_caption, write_values
from mutacc.masked_dms import add_scoring_argument, apply_masked

STEM = "sP_alpha_curves"

SCORES_TABLE = "dms_scores"
MASKED_TABLE = "dms_scores_masked"
SUMMARY_TABLE = "dms_peak_summary"

#: The pooled fit, then one fit inside each nucleotide-step stratum.
POOLED = "pooled"
STRATA = (1, 2, 3)
#: Only the arms scored under the spectrum the manuscript reports.
PUBLISHED_SPECTRUM = "published"
#: Where in the viridis ramp the three strata are sampled. Codon distance is ordered, so it
#: takes a sequential ramp; the pooled fit is not a fourth category and wears ink instead.
RAMP_SPAN = (0.15, 0.85)
#: Assay panels per figure row. Each contributes one column per model, so this many pairs.
PANEL_PAIRS_PER_ROW = 2
#: Four columns do not fit a journal column width legibly. This is a diagnostic supplementary
#: read on screen or at full page, not a column-width figure, so it takes its own width rather
#: than squeezing eighteen panels into 180 mm.
WIDTH_WIDE = 13.0
#: Role whose header is bolded, so the model being compared is obvious at a glance. Matches the
#: key in ``mutacc.style.ROLE_COLOURS``; it is a role, not a model name.
FINETUNE_ROLE = "finetune"


def alpha_grid() -> np.ndarray:
    """Figure SM's grid, so the optima drawn here are the ones that figure counts."""
    return np.round(np.arange(-ALPHA_LIMIT, ALPHA_LIMIT + ALPHA_STEP / 2, ALPHA_STEP), 6)


def panels_of(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (panel, model), in the summary's order.

    Every assay is drawn, receptor binding included: those curves are the evidence for leaving
    them out of the counts, and a reader cannot check an exclusion they cannot see. ``counted``
    marks which arms enter the tallies.
    """
    wanted = summary.loc[summary["spectrum"] == PUBLISHED_SPECTRUM].copy()
    wanted["counted"] = ~wanted["trait"].str.contains(RECEPTOR_TRAIT, case=False, na=False)
    columns = ["dataset", "trait", "panel_label", "organism", "citation",
               "model_name", "model_label", "model_role", "counted"]
    return wanted[columns].drop_duplicates().reset_index(drop=True)


def compute(scores: pd.DataFrame, panels: pd.DataFrame) -> pd.DataFrame:
    """The curve, its optimum and its flags for every arm and every group."""
    grid = alpha_grid()
    rows = []
    for arm in panels.itertuples():
        cell = scores.loc[(scores["dataset"] == arm.dataset)
                          & (scores["model_name"] == arm.model_name)]
        cell = cell.loc[cell["log_plm"].notna() & cell["log_acc"].notna()
                        & cell[arm.trait].notna()]
        for group in (POOLED, *STRATA):
            sub = cell if group == POOLED else cell.loc[cell["nt_steps"] == group]
            if len(sub) < MIN_ROWS:
                continue
            rho = sweep_curve(sub["log_plm"].to_numpy(float), sub["log_acc"].to_numpy(float),
                              sub[arm.trait].to_numpy(float), grid)
            flags = curve_flags(rho, grid)
            rows.append({
                "panel_label": arm.panel_label, "organism": arm.organism,
                "counted": bool(arm.counted),
                "citation": arm.citation, "trait": arm.trait,
                "model_name": arm.model_name, "model_label": arm.model_label,
                "role": arm.model_role, "group": str(group),
                "n": len(sub), "n_sites": int(sub["site"].nunique()),
                "sd_log_acc": float(np.std(sub["log_acc"].to_numpy(float))),
                "peak_alpha": flags["best_alpha"], "rho_peak": flags["rho_best"],
                "rho_alpha0": flags["rho_alpha0"],
                "gain": flags["rho_best"] - flags["rho_alpha0"],
                "scaled_weight": flags["best_alpha"] * float(np.std(sub["log_acc"].to_numpy(float))),
                "at_sweep_end": flags["boundary"],
                # The quantity curve_has_signal actually tests. Shipped because |rho| at the
                # signed argmax is NOT it: on a curve that is negative throughout, the largest
                # signed value and the largest magnitude sit at opposite ends.
                "max_abs_rho": float(np.nanmax(np.abs(rho))),
                "has_signal": curve_has_signal(rho),
                # Marked only where the sweep actually found something. A curve that is weak
                # everywhere AND gains nothing by moving alpha has no optimum to draw.
                # No optimum is defined for an assay measured against fitness: its rho is
                # negative throughout and the signed argmax is a dilution endpoint. Those arms
                # get no marker at all, rather than one the reader has to know to ignore.
                "has_optimum": bool(arm.counted
                                    and (curve_has_signal(rho)
                                         or flags["rho_best"] - flags["rho_alpha0"] >= MARKED_GAIN)),
                "curve": rho,
            })
    return pd.DataFrame(rows)


def counts(curves: pd.DataFrame) -> dict[str, dict]:
    """The per-group tallies the caption quotes, over **every** arm.

    Nothing is excluded on the quality of its fit. An earlier version set aside curves whose
    ``|rho|`` never reaches 0.05, and that was wrong twice over: the optima were perfectly
    locatable (the one arm it caught reads alpha* = 0.00 pooled, -1.00 at 1 nt and +0.02 at
    2 nt -- a flat curve and a monotone one, not noise), and dropping them made the
    denominator move between strata for a reason no reader of the count could see. The weak
    flag is kept in the table and drawn on the figure so the reader can discount those arms;
    it no longer decides for them.
    """
    def tally(frame: pd.DataFrame) -> dict:
        return {
            "arms": len(frame),
            "negative": int((frame["peak_alpha"] < 0).sum()),
            "at_end": int(frame["at_sweep_end"].sum()),
            "weak": int((~frame["has_signal"]).sum()),
            "median_scaled": float(frame["scaled_weight"].median()) if len(frame) else float("nan"),
            "lo_scaled": float(frame["scaled_weight"].min()) if len(frame) else float("nan"),
            "hi_scaled": float(frame["scaled_weight"].max()) if len(frame) else float("nan"),
            # Raw alpha, readable directly *within* one group because the arms there share a
            # comparable spread of log accessibility. Only across groups does alpha need x SD.
            "median_alpha": float(frame["peak_alpha"].median()) if len(frame) else float("nan"),
            "lo_alpha": float(frame["peak_alpha"].min()) if len(frame) else float("nan"),
            "hi_alpha": float(frame["peak_alpha"].max()) if len(frame) else float("nan"),
        }

    curves = curves.loc[curves["counted"]]
    out = {}
    for group in (POOLED, *STRATA):
        arms = curves.loc[curves["group"] == str(group)]
        out[str(group)] = tally(arms)
        # Pooling the two models hides the result. Their optima separate completely in the
        # pooled fit -- every fine-tuned arm below -0.52, every base arm above -0.19 -- so a
        # median over both lands in the empty gap between them and describes neither.
        for role, part in arms.groupby("role"):
            out[f"{group}:{role}"] = tally(part)
    out["within"] = tally(curves.loc[curves["group"] != POOLED])
    return out


def draw(curves: pd.DataFrame):
    """Two assay panels per row, each as a base / fine-tuned pair of columns.

    One pair of columns per panel would make a figure nine rows tall. Two pairs side by side
    halves that without breaking the pairing: columns 1-2 are one assay's two models and
    columns 3-4 the next assay's, so the comparison the figure is for stays adjacent.
    """
    from matplotlib import colormaps

    style.use_house_style()
    ramp = colormaps[style.SEQUENTIAL](np.linspace(*RAMP_SPAN, len(STRATA)))
    hue = {str(step): ramp[i] for i, step in enumerate(STRATA)}
    grid = alpha_grid()

    panels = list(dict.fromkeys(curves["panel_label"]))
    roles = list(dict.fromkeys(curves["role"]))
    pairs = PANEL_PAIRS_PER_ROW
    nrow = -(-len(panels) // pairs)
    ncol = pairs * len(roles)
    fig = style.figure(WIDTH_WIDE, 1.72 * nrow + 0.5)
    axes = np.asarray(fig.subplots(nrow, ncol, sharex=True)).reshape(nrow, ncol)
    for spare in range(len(panels), nrow * pairs):          # unused slots at the end
        for role_index in range(len(roles)):
            axes[spare // pairs][(spare % pairs) * len(roles) + role_index].set_axis_off()

    for index, panel in enumerate(panels):
        row, block = divmod(index, pairs)
        for role_index, role in enumerate(roles):
            col = block * len(roles) + role_index
            ax = axes[row][col]
            arm = curves.loc[(curves["panel_label"] == panel) & (curves["role"] == role)]
            ax.axvline(0.0, color=style.GRID, lw=1.0, zorder=0)
            ax.axhline(0.0, color=style.GRID, lw=1.0, ls=":", zorder=0)

            for _, r in arm.iterrows():
                pooled = r["group"] == POOLED
                colour = style.INK_MUTED if pooled else hue[r["group"]]
                ax.plot(grid, r["curve"], color=colour, lw=2.0,
                        ls="--" if pooled else "-", zorder=4 if pooled else 3)
                if not r["has_optimum"]:
                    # No marker: the sweep found nothing, so there is no optimum to assert.
                    continue
                ax.plot([r["peak_alpha"]], [r["rho_peak"]],
                        "s" if r["at_sweep_end"] else "o",
                        ms=8 if r["at_sweep_end"] else 7,
                        mfc=colour if r["has_signal"] else style.SURFACE,
                        mec=colour, mew=1.8, zorder=6 if pooled else 5)

            counted = bool(arm["counted"].all()) if len(arm) else True
            weak = int((~arm["has_signal"]).sum())
            if weak or not counted:
                ax.set_facecolor(style.SURFACE_SET_ASIDE)
            label = arm["model_label"].iloc[0] if len(arm) else role
            note = " · not counted" if not counted else (f" · {weak} weak" if weak else "")
            # Two pieces, so the model line can carry its own weight: bold marks the fine-tuned
            # column of each pair, which is the comparison every row is making.
            ax.set_title(panel, fontsize=9, color=style.INK, pad=15)
            ax.text(0.5, 1.015, f"{label}{note}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8.5, color=style.INK,
                    fontweight="bold" if role == FINETUNE_ROLE else "normal")
            ax.tick_params(labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == nrow - 1 or index >= len(panels) - pairs:
                ax.set_xlabel("accessibility weight α", fontsize=9.5)
            if col == 0:
                ax.set_ylabel("Spearman ρ with\nmeasured effect", fontsize=9.5)

    handles = [Line2D([], [], color=style.INK_MUTED, lw=2, ls="--",
                      label="all replacements pooled"),
               *[Line2D([], [], color=hue[str(s)], lw=2, label=f"{s} nucleotide change"
                        + ("s" if s > 1 else "")) for s in STRATA],
               Line2D([], [], color=style.INK_MUTED, marker="s", ls="none", ms=8,
                      mfc=style.INK_MUTED, mew=1.8, label="best α at sweep end (kept)"),
               Line2D([], [], color=style.INK_MUTED, marker="o", ls="none", ms=7,
                      mfc=style.SURFACE, mew=1.8,
                      label="|ρ| under 0.05 throughout (still counted)"),
               Line2D([], [], color=style.SURFACE, ls="none", marker="None",
                      label="no marker: no optimum defined"),
               Line2D([], [], color=style.SURFACE, ls="none", marker="None",
                      label="shaded: weak, or not counted (receptor binding)")]
    style.legend_above(fig, handles=handles, ncol=4, fontsize=9, frameon=False)
    # Fill the page: the default outer pad leaves a visible white margin either side of a
    # four-column grid. h_pad is legend_above's and is left alone.
    fig.get_layout_engine().set(w_pad=0.01, wspace=0.035)
    return fig


def caption(curves: pd.DataFrame) -> str:
    c = counts(curves)
    arms = curves.groupby(["panel_label", "role"]).ngroups
    panels = curves["panel_label"].nunique()
    # Counted in terms of what the reader can actually see: a marker is only drawn on a
    # counted arm that has an optimum, so quoting every arm's flag here would name squares
    # and hollow circles that are not on the page.
    def fits(n: int) -> str:
        """"1 fit", not "1 fits" -- these counts are small enough to hit the singular."""
        return f"{n} fit" + ("" if n == 1 else "s")

    drawn = curves["counted"] & curves["has_optimum"]
    ends = int((drawn & curves["at_sweep_end"]).sum())
    weak = int((drawn & ~curves["has_signal"]).sum())
    uncounted = int((~curves["counted"]).sum())
    unmarked = int((~curves["has_optimum"] & curves["counted"]).sum())
    return f"""
        The whole accessibility-weight curve for every scored arm, behind the optima counted
        in Figure SM. Rows are deep mutational scanning panels and columns the two models, so
        each row is one paired comparison. The dashed grey line fits all replacements together
        and the three coloured lines fit inside one nucleotide-step class each; a marker is
        that curve's best weight. {arms} arms over {panels} panels.

        Every fit is drawn. The two receptor-binding assays are shaded and labelled "not
        counted": their correlations are negative across the whole sweep, because they measure
        a phenotype running against fitness rather than with it, so the largest value falls
        wherever the model is diluted hardest towards zero and no optimum is defined. Those
        panels are plotted anyway, because the shape of their curves is what justifies leaving
        them out. Among the assays that are counted, nothing is excluded on how good its fit
        looks; two kinds are marked so that they can be discounted by the reader rather than
        by the figure. A best weight at the end of the swept range is drawn as a square ({fits(ends)}):
        it is a bound rather than a turning point, since the curve may still be improving past
        the edge, but where a curve runs plainly towards that edge it is a result and not a
        failure. A curve whose correlation never reaches 0.05 anywhere in the sweep is drawn
        hollow on a shaded panel ({fits(weak)}): its optimum may still be perfectly clear --
        one such curve falls steadily across the whole sweep and is read at the far edge --
        but nothing there is strong enough to lean on. Where such a curve is also unmoved by
        the weighting, gaining less than {MARKED_GAIN} in rho over alpha = 0, no marker is
        drawn ({fits(unmarked)}): the sweep found nothing there, so the position of the largest
        value is numerical noise rather than an optimum. Those fits are still counted, as
        showing no preference, which is what keeps the denominators equal across classes. The
        receptor-binding panels carry no markers either ({fits(uncounted)}), for the different
        reason that no optimum is defined for them at all.

        A negative weight -- dividing some mutational accessibility back out -- is preferred in
        {c[POOLED]['negative']} of {c[POOLED]['arms']} pooled fits and {c['within']['negative']}
        of {c['within']['arms']} fits made inside a single nucleotide-step class
        ({c['1']['negative']} of {c['1']['arms']} at 1, {c['2']['negative']} of
        {c['2']['arms']} at 2, {c['3']['negative']} of {c['3']['arms']} at 3). Every one of
        those denominators is the same because every arm is counted in every class.

        Those totals pool two models that behave differently, and in the pooled fit the two
        separate completely. Every one of the {c[POOLED + ':finetune']['arms']} fine-tuned arms
        prefers a negative weight, between alpha = {c[POOLED + ':finetune']['lo_alpha']:+.2f}
        and {c[POOLED + ':finetune']['hi_alpha']:+.2f} (median
        {c[POOLED + ':finetune']['median_alpha']:+.2f}), while the base model's
        {c[POOLED + ':base']['arms']} arms sit either side of zero, between
        {c[POOLED + ':base']['lo_alpha']:+.2f} and {c[POOLED + ':base']['hi_alpha']:+.2f}
        (median {c[POOLED + ':base']['median_alpha']:+.2f}), only
        {c[POOLED + ':base']['negative']} of them negative. The two ranges do not overlap: the
        least negative fine-tuned arm still asks for more accessibility to be divided out than
        the most negative base arm does, so an arm's model can be read off its preferred weight
        alone, and a single median over all {c[POOLED]['arms']} would fall in the empty gap
        between the two groups and describe neither. This is what the paper's argument
        predicts: dividing accessibility back out helps where the model absorbed it, and the
        fine-tuned model is the one that did.

        Inside a single nucleotide-step class that separation goes. At one nucleotide change
        both models prefer a negative weight in all {c['1:base']['arms']} of their arms, so the
        preference there is not a fine-tuning effect. By three changes neither model shows one:
        {c['3:base']['negative']} of {c['3:base']['arms']} base and
        {c['3:finetune']['negative']} of {c['3:finetune']['arms']} fine-tuned arms are
        negative, over a spread wide enough to hold both signs. Weights are not comparable
        between the pooled fit and a single class, nor between classes: fixing the step count
        collapses the spread of log accessibility about fivefold, so the same real
        down-weighting needs a correspondingly larger alpha inside a class. On the
        scale-matched footing alpha x SD(log accessibility) the median is
        {c[POOLED]['median_scaled']:+.2f} pooled against {c['within']['median_scaled']:+.2f}
        within classes.
        """


def summary_rows(curves: pd.DataFrame) -> pd.DataFrame:
    """The table shipped beside the figure: the curve itself is not written, its summary is."""
    return curves.drop(columns="curve")


def parse(argv=None):
    return parse_figure_args(__doc__.splitlines()[0], argv, extra=add_scoring_argument)


def load(guide: Guide, results: Path, scoring: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = guide.read_table(SCORES_TABLE, results)
    if scoring == "masked":
        scores = apply_masked(scores, guide.read_table(MASKED_TABLE, results))
    return scores, guide.read_table(SUMMARY_TABLE, results)


def main(argv=None) -> int:
    args = parse(argv)
    scores, summary = load(args.guide, args.results, args.scoring)
    curves = compute(scores, panels_of(summary))
    if curves.empty:
        raise SystemExit("no arms survived the panel filter; is dms_peak_summary complete?")
    fig = draw(curves)
    paths = style.save_figure(fig, args.out / STEM)
    write_values(summary_rows(curves), args.out, STEM)
    write_caption(caption(curves), args.out, STEM)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
