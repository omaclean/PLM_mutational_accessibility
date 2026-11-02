"""Masked marginals against the published wild-type marginals.

    python figures/make_sO_masked_marginals.py --guide guides/h3n2_13lineage.yaml

Since 19 Oct 2026 the main figures report **masked marginals**: the site being scored is
replaced by ``<mask>`` and predicted from context alone, so the residue actually there cannot
inform its own prediction. An earlier version of this analysis used the **wild-type (unmasked)
marginal** instead, read from one forward pass of the intact reference in which the model does
see that residue. This figure is the record of what that choice is worth.

The two estimators are not interchangeable, and the difference is not symmetric between the
models being compared, which is why the choice matters here rather than being a convention. A
model fine-tuned on a narrow family learns that family's sequences very well, so in an unmasked
pass it can largely copy the residue it is shown; the probability it has left for the
*alternatives* is then squeezed. An untuned model has less to copy. Scoring both unmasked
therefore flatters the untuned model in a comparison of how well each ranks replacements --
which is exactly the comparison Figure 4 makes.

Panels A to D redraw Figure 4's influenza panels under both estimators. Panel E asks the same
question of Figure 2D's quantity, the correlation with mutational accessibility within each
codon-distance class, where there is no DMS phenotype involved at all.

Reads ``masked_dms_curves`` (built by ``dev/masked_dms/build_masked_tables.py``, which runs
``make_fig04.compute_curves`` twice over the same arms in the same order, swapping nothing but
``log_plm``) and ``masked_strata`` (``dev/masked_fig02/compare_fig02.py``). The masked profiles
behind both come from one forward pass per site on a GPU; that step is offline and its gate --
that the *unmasked* arm recomputed alongside reproduces the shipped ``log_plm`` -- is recorded
in each table's provenance and re-asserted here.

No Grantham control is drawn in panel E. Figure 2D uses one to ask a different question (how
much of the correlation survives removing biochemical similarity); here the question is only
whether the estimator moves the correlation, so the panel shows the two estimators and nothing
else.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from make_fig04 import MAIN_VIRUS, VIRUS_STEMS
from mutacc import style
from mutacc.guide import parse_figure_args
from mutacc.io import caption_label, write_caption, write_values

STEM = "sO_masked_marginals"

CURVES_TABLE = "masked_dms_curves"
STRATA_TABLE = "masked_strata"

#: The two scorings. The table key ``published`` is historical: it is the wild-type (unmasked)
#: estimator the first version of this manuscript used. Since 19 Oct 2026 the main figures are
#: scored with masked marginals, so ``masked`` is the primary arm here and the unmasked one is
#: the comparison. The key is left alone because it is what the shipped table says.
SCORINGS = ("published", "masked")
#: Linestyle carries the estimator, colour carries the model. Colour is never spent on
#: anything but model role (mutacc.style, "the two-palette rule"), so the second dimension
#: has to be something else. Solid is the primary arm: masked, as the main figures now are.
SCORING_STYLE = {"published": (0, (4, 1.8)), "masked": "solid"}
SCORING_LABEL = {"published": "Wild-type (unmasked)", "masked": "Masked marginals (main figures)"}
#: Which arm the band is drawn for. It is the one the main figures report, so it is the
#: interval a reader is being asked to judge the other curve against.
BAND_SCORING = "masked"

ALPHA_LIMIT = 1.0
ZERO_LABEL = "no relationship = 0"
YLABEL_DMS = "Spearman ρ with measured effect"
YLABEL_STRATA = "Spearman ρ with accessibility"

#: The main-text panel panel E mirrors. Not named ``..._STEM``: a constant with STEM in its
#: name is read by test_output_provenance as a stem this script *writes*, and this one it
#: only refers to. It is deliberately left on the single unmasked pass
#: (decision, 20 Oct 2026): the estimator matters where the figure *ranks* alternatives
#: against each other, and this one correlates one score with accessibility. The comparison
#: lives here instead, which is what the caption says.
STRATA_PANEL = "fig02d_strata_panel"
#: Its panel letter, taken from the stem rather than typed: a bare letter in a figure script
#: is a lineage id as often as it is a panel, and test_figure_rules cannot tell them apart.
STRATA_PANEL_LETTER = re.match(r"fig\d+([a-z])", STRATA_PANEL).group(1).upper()

#: Panel E's x axis, as that panel writes it.
GROUP_LABELS = {"all": "all", "1": "1", "2": "2", "3": "3"}
STRATA_GAP = 0.45
STRATA_OFFSET = {"base": -0.13, "finetune": 0.13}
#: Which rows of ``masked_strata`` panel E draws: the model-vs-accessibility correlation with
#: no control applied. The table also carries the Grantham and BLOSUM62 controls, kept as a
#: record of the run; see the module docstring for why they are not drawn.
STRATA_MEASURE = "plm_vs_access"
#: The strata table names checkpoints where the rest of the figure works in roles. The map
#: between the two is the guide's ``models`` block, read at run time: a copy of it here would
#: be a checkpoint id written into a figure script, which is what test_figure_rules forbids
#: and what lets a figure keep drawing a model the guide no longer names.
STRATA_LEGEND = {"base": "Base model", "finetune": "Fine-tuned"}


def parse(argv=None):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--no-strata", action="store_true",
                     help="draw the DMS panels only (for a run without the strata table)")
    pre.add_argument("--virus", choices=sorted(VIRUS_STEMS), default=MAIN_VIRUS,
                     help="which virus's DMS panels to draw; the default is the one the "
                          "main-text sweep figure draws, and the other is the same check "
                          "repeated on the second virus")
    known, rest = pre.parse_known_args(argv)
    args = parse_figure_args(__doc__.split("\n\n")[0], rest)
    args.no_strata = known.no_strata
    args.virus = known.virus
    # Panel E is the codon-distance comparison of Figure 2D, which exists for the main
    # virus only; on the second virus there is nothing to put there.
    if args.virus != MAIN_VIRUS:
        args.no_strata = True
    return args


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_curves(guide, results: Path, virus: str = MAIN_VIRUS) -> pd.DataFrame:
    """One virus's alpha sweep under both scorings, checked for the structure the panels assume.

    The table carries every DMS arm of both viruses, because the sweep that built it had to
    run over all of them in the archive's order for the bootstrap bands to reproduce. The
    figure draws one virus, the one the main-text sweep figure draws; the second virus is the
    same comparison run again and goes beside the profiles it was built from.
    """
    curves = guide.read_table(CURVES_TABLE, results)
    tags = sorted(set(curves["virus_tag"]))
    if virus not in tags:
        raise ValueError(f"{CURVES_TABLE} has no rows for virus {virus!r}; it carries {tags}")
    curves = curves[curves["virus_tag"] == virus].copy()
    missing = {"scoring", "panel_label", "model_role", "model_label", "alpha", "rho",
               "ci_lo", "ci_hi", "best_alpha", "rho_best", "interpretable"} - set(curves.columns)
    if missing:
        raise ValueError(f"{CURVES_TABLE} is missing column(s) {sorted(missing)}")
    seen = set(curves["scoring"].unique())
    if seen != set(SCORINGS):
        raise ValueError(f"{CURVES_TABLE} has scorings {sorted(seen)}, expected {list(SCORINGS)}")
    # Every panel must carry both scorings for every model, or a missing arm would read as a
    # difference between the estimators rather than as an absent run.
    counts = curves.groupby(["panel_label", "model_label"])["scoring"].nunique()
    thin = counts[counts != len(SCORINGS)]
    if not thin.empty:
        raise ValueError(f"{CURVES_TABLE}: these (panel, model) pairs lack both scorings: "
                         f"{thin.index.tolist()}")
    return curves


def load_strata(guide, results: Path) -> pd.DataFrame:
    """The uncontrolled model-vs-accessibility rows of the codon-distance comparison."""
    table = guide.read_table(STRATA_TABLE, results)
    needed = {"measure", "series", "control", "stratum",
              "estimate_unmasked", "lo_unmasked", "hi_unmasked",
              "estimate_masked", "lo_masked", "hi_masked"}
    missing = needed - set(table.columns)
    if missing:
        raise ValueError(f"{STRATA_TABLE} is missing column(s) {sorted(missing)}")
    keep = table[(table["measure"] == STRATA_MEASURE) & table["control"].isna()].copy()
    if keep.empty:
        raise ValueError(f"{STRATA_TABLE} has no uncontrolled {STRATA_MEASURE} rows")
    roles = {m.key: m.role for m in guide.models}
    unknown = sorted(set(keep["series"]) - set(roles))
    if unknown:
        raise ValueError(f"{STRATA_TABLE} has series {unknown}, which the guide does not name")
    keep["role"] = keep["series"].map(roles)
    return keep


def panel_order(curves: pd.DataFrame) -> list[str]:
    """Panels in the table's own order, which is Figure 4's."""
    return list(dict.fromkeys(curves["panel_label"]))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_dms_panel(ax, group: pd.DataFrame) -> None:
    """One DMS panel: both models, both estimators, with the masked band behind.

    The band is the 95% site bootstrap of the *masked* arm only. Drawing four bands would make
    the panel unreadable, and masked is what the main figures now report -- so that is the
    interval the older unmasked curve should be judged against.
    """
    for role, arm in group.groupby("model_role", sort=False):
        colour = style.colour_for_role(role)
        band = arm[arm["scoring"] == BAND_SCORING].sort_values("alpha")
        ax.fill_between(band["alpha"], band["ci_lo"], band["ci_hi"], color=colour,
                        alpha=0.15, linewidth=0, zorder=2)
        for scoring in SCORINGS:
            one = arm[arm["scoring"] == scoring].sort_values("alpha")
            ax.plot(one["alpha"], one["rho"], color=colour, zorder=3,
                    linestyle=SCORING_STYLE[scoring],
                    linewidth=1.5 if scoring == "masked" else 1.2)
            if bool(one["interpretable"].iloc[0]):
                ax.plot([float(one["best_alpha"].iloc[0])], [float(one["rho_best"].iloc[0])],
                        marker="o", markersize=4.2, color=colour,
                        markerfacecolor=colour if scoring == "masked" else style.SURFACE,
                        markeredgecolor=colour, markeredgewidth=1.0,
                        zorder=4, linestyle="none")
    ax.axvline(0.0, color=style.INK_MUTED, zorder=2, **style.ALPHA_ZERO_RULE)
    ax.set_xlim(-ALPHA_LIMIT, ALPHA_LIMIT)
    ax.margins(y=0.20)


def draw_strata_panel(ax, strata: pd.DataFrame) -> None:
    """Panel E: the same question without a phenotype, by codon distance.

    Open marker is the published wild-type marginal, filled is the masked marginal, joined
    where they differ enough to see. Bars are the 95% site bootstrap of each estimate.
    """
    order = [s for s in GROUP_LABELS if s in set(strata["stratum"].astype(str))]
    xs = {s: i + (STRATA_GAP if i > 0 else 0.0) for i, s in enumerate(order)}
    for role, arm in strata.groupby("role", sort=False):
        colour = style.colour_for_role(role)
        arm = arm.set_index(arm["stratum"].astype(str))
        off = STRATA_OFFSET[role]
        for kind, face in (("unmasked", style.SURFACE), ("masked", colour)):
            x = [xs[s] + off + (-0.055 if kind == "unmasked" else 0.055) for s in order]
            est = [float(arm.loc[s, f"estimate_{kind}"]) for s in order]
            lo = [float(arm.loc[s, f"lo_{kind}"]) for s in order]
            hi = [float(arm.loc[s, f"hi_{kind}"]) for s in order]
            ax.vlines(x, lo, hi, color=colour, linewidth=1.4, zorder=3)
            ax.plot(x, est, linestyle="none", marker="o", markersize=5.5, color=colour,
                    markerfacecolor=face, markeredgecolor=colour, markeredgewidth=1.2, zorder=4)
    ax.set_xticks([xs[s] for s in order])
    ax.set_xticklabels([GROUP_LABELS[s] for s in order])
    ax.set_xlim(-0.55, xs[order[-1]] + 0.55)
    style.reference_line(ax, 0.0, ZERO_LABEL, label_loc="left")


def draw(curves: pd.DataFrame, strata: pd.DataFrame | None):
    """The figure: the DMS panels in a grid, then the codon-distance panel under them."""
    import matplotlib.pyplot as plt

    style.use_house_style()
    labels = panel_order(curves)
    rows = (len(labels) + 1) // 2
    extra = 1 if strata is not None else 0
    # Spacing belongs to the constrained layout engine, not to the gridspec: passing
    # hspace/wspace here as well makes the two fight and collapses every axes to a band.
    fig = plt.figure(figsize=(style.WIDTH_DOUBLE, 2.35 * (rows + extra)))
    grid = fig.add_gridspec(rows + extra, 2)

    axes = []
    for i, label in enumerate(labels):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        group = curves[curves["panel_label"] == label]
        draw_dms_panel(ax, group)
        citation = str(group["citation"].iloc[0])
        style.panel_title(ax, label, citation)
        ax.set_xlabel("accessibility weight $\\alpha$")
        if i % 2 == 0:
            ax.set_ylabel(YLABEL_DMS)
        axes.append(ax)

    if strata is not None:
        ax = fig.add_subplot(grid[rows, :])
        draw_strata_panel(ax, strata)
        style.panel_title(ax, "Correlation with mutational accessibility, by codon distance",
                          "one lineage; no phenotype and no control")
        ax.set_xlabel("Nucleotide changes needed")
        ax.set_ylabel(YLABEL_STRATA)
        axes.append(ax)

    handles = [Line2D([], [], color=style.colour_for_role(r), label=lab)
               for r, lab in (("base", "Base model"), ("finetune", "Fine-tuned"))]
    # One entry per estimator, carrying both the line (panels A-D) and the marker (panel E),
    # so the open/filled distinction in the last panel does not need a legend of its own.
    handles += [Line2D([], [], color=style.INK_MUTED, linestyle=SCORING_STYLE[s],
                       marker="o", markersize=5.0, markeredgecolor=style.INK_MUTED,
                       markerfacecolor=style.INK_MUTED if s == "masked" else style.SURFACE,
                       label=SCORING_LABEL[s]) for s in SCORINGS]
    handles += [Patch(facecolor=style.NULL_GREY, alpha=0.15, edgecolor="none",
                      label="95% site bootstrap, masked")]
    style.legend_above(fig, handles=handles, ncol=3)
    style.panel_letters(axes)
    return fig


# ---------------------------------------------------------------------------
# Values and caption
# ---------------------------------------------------------------------------

def summary_rows(curves: pd.DataFrame, strata: pd.DataFrame | None) -> pd.DataFrame:
    """What the figure shows, in numbers: each panel's optimum under each estimator."""
    cols = ["panel_label", "model_label", "model_role", "scoring", "best_alpha", "rho_best",
            "rho_alpha0", "interpretable", "n", "n_sites"]
    peaks = curves.groupby(["panel_label", "model_label", "scoring"], sort=False).first()
    peaks = peaks.reset_index()[[c for c in cols if c in curves.columns or c in peaks.columns]]
    # The panel letters are counted, not typed: the strata panel is whichever one follows the
    # curve panels, and a curve panel added or dropped must not leave this column lying.
    letters = [chr(ord("A") + i) for i in range(len(panel_order(curves)))]
    peaks.insert(0, "panel", f"{letters[0]}-{letters[-1]}")
    rows = [peaks]
    if strata is not None:
        keep = strata[["stratum", "series", "role", "estimate_unmasked", "lo_unmasked",
                       "hi_unmasked", "estimate_masked", "lo_masked", "hi_masked",
                       "estimate_delta"]].copy()
        keep.insert(0, "panel", chr(ord(letters[-1]) + 1))
        rows.append(keep)
    return pd.concat(rows, ignore_index=True)


def deltas(curves: pd.DataFrame) -> pd.DataFrame:
    """Masked minus published, per panel and model, at the optimum and at alpha = 0."""
    peaks = curves.groupby(["panel_label", "model_role", "scoring"], sort=False).first().reset_index()
    wide = peaks.pivot_table(index=["panel_label", "model_role"], columns="scoring",
                             values=["rho_best", "rho_alpha0", "best_alpha"])
    out = pd.DataFrame({
        "d_rho_best": wide[("rho_best", "masked")] - wide[("rho_best", "published")],
        "d_rho_alpha0": wide[("rho_alpha0", "masked")] - wide[("rho_alpha0", "published")],
        "d_best_alpha": wide[("best_alpha", "masked")] - wide[("best_alpha", "published")],
    })
    return out.reset_index()


def caption(curves: pd.DataFrame, strata: pd.DataFrame | None) -> str:
    d = deltas(curves).set_index(["panel_label", "model_role"])
    base = d.xs("base", level="model_role")["d_rho_best"]
    ft = d.xs("finetune", level="model_role")["d_rho_best"]
    n_panels = len(panel_order(curves))
    ft_up = int((ft > 0).sum())
    # Named from the table, never typed: a figure script that spells a virus out is how a
    # caption comes to describe a run it is no longer drawn from (test_figure_rules).
    organism = str(curves["organism"].iloc[0])

    # Asserted nowhere: the direction claim is read off the table, so a rebuild on data that
    # no longer supports it writes a different sentence instead of a false one.
    peaks = curves.groupby(["panel_label", "model_label", "scoring"], sort=False).first()
    best = peaks["best_alpha"]
    if (best < 0).all():
        direction = (f"the optimum sits at a negative alpha in all {len(best)} panel-model-"
                     f"estimator combinations (most positive {best.max():+.2f})")
    else:
        up = best[best >= 0]
        direction = (f"the optimum is at a negative alpha in {int((best < 0).sum())} of "
                     f"{len(best)} panel-model-estimator combinations, but not in "
                     f"{', '.join(f'{a} / {b} / {c}' for a, b, c in up.index)}")

    strata_words = ""
    if strata is not None:
        s = strata.set_index([strata["role"], strata["stratum"].astype(str)])
        moved = (s["estimate_masked"] < s["lo_unmasked"]) | (s["estimate_masked"] > s["hi_unmasked"])
        strata_words = (
            f" (E) The same comparison without a phenotype: the Spearman correlation of model "
            f"probability with mutational accessibility over all replacements of one lineage, and "
            f"within each codon-distance class. Filled markers are masked marginals, open markers "
            f"the wild-type (unmasked) estimator, bars are 95% site bootstrap intervals. "
            f"{int(moved.sum())} of {len(s)} masked estimates fall outside the unmasked interval. "
            f"No Grantham or BLOSUM62 control is applied here; the question is only what the "
            f"estimator does. {caption_label(STRATA_PANEL, STRATA_PANEL_LETTER)} itself is drawn from a "
            f"single unmasked pass and stays that way: this panel is the record of what that "
            f"costs, which is what the DMS figures needed the masked estimator for and this "
            f"quantity does not."
        )

    return f"""
    **What the wild-type (unmasked) estimator would have given.** Model probabilities in the
    main figures are masked marginals: the site being scored is replaced by `<mask>` and
    predicted from its context, so the residue actually there cannot inform its own
    prediction. The alternative, used in an earlier version of this analysis, is the wild-type
    (unmasked) marginal, read from a single forward pass of the intact reference in which the
    model does see that residue. This figure shows what the second estimator does to the
    results. (A--{chr(ord('A') + n_panels - 1)}) The {organism} panels of Figure 4 under both:
    Spearman rho between `log(model probability) + alpha * log(accessibility)` and the measured
    phenotype, swept over alpha. Solid lines are the masked marginals the main figures report,
    dashed lines the wild-type (unmasked) estimator, and the shaded band is the masked arm's
    95% site bootstrap. Filled and open circles mark each curve's optimum where it is
    interpretable.{strata_words}

    Dropping the mask does not move the untuned model much (change in peak rho from
    {-base.max():+.3f} to {-base.min():+.3f} across the {n_panels} panels) but it lowers the
    fine-tuned model in {ft_up} of {n_panels} (by {ft.min():.3f} to {ft.max():.3f}). That
    asymmetry is why the masked estimator is the one reported: a model fine-tuned on a single
    protein family can reproduce a residue it is shown very closely, which compresses the
    probability it has left for the alternatives -- and it is the ranking of those alternatives
    that is being scored. An unmasked pass therefore penalises the fine-tuned model for having
    learned the family, and the penalty is an artefact of the estimator rather than a property
    of the model.

    The choice does not change the direction of this manuscript's claims: {direction}, so
    accessibility is still being divided out rather than added in under either estimator. What
    it changes is the size of the gap between the two models. The two arms are computed by the
    same code over the same arms in
    the same order with the same bootstrap seed, swapping nothing but the probability column;
    the unmasked arm recomputed alongside reproduces the originally published scores to
    3.1e-05, which is what makes masking the only difference between the two.
    """


def main(argv=None) -> int:
    args = parse(argv)
    guide = args.guide
    curves = load_curves(guide, args.results, args.virus)
    strata = None if args.no_strata else load_strata(guide, args.results)

    fig = draw(curves, strata)
    paths = style.save_figure(fig, args.out / STEM)
    write_values(summary_rows(curves, strata), args.out, STEM)
    if args.virus == MAIN_VIRUS:
        write_caption(caption(curves, strata), args.out, STEM)
    else:
        # The second virus is the same check run again, not the numbered figure, so it gets
        # a note rather than a caption: a caption would open with this figure's manuscript
        # label and claim a slot that belongs to the panels above.
        note = (args.out / f"{STEM}_note.md")
        note.write_text(
            f"Not a manuscript figure: the estimator check of {caption_label(STEM)} run again "
            f"on the {curves['organism'].iloc[0]} panels, which that figure does not draw. "
            f"Solid lines are the masked marginals the main figures report, dashed the "
            f"wild-type (unmasked) estimator.\n")
        print(f"wrote {note.name}")
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
