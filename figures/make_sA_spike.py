"""S1: Figure 2's question asked of SARS-CoV-2 spike.

    python figures/make_sA_spike.py --guide guides/sars_cov_2_spike.yaml

Figure 2's test for a second virus, with ESM-C 600M as both the base and the fine-tuned model. The epoch curve (Figure 2C) is left out: the
only per-epoch checkpoints in the archive are for a different backbone, so there is nothing
honest to draw.

A, B  Model probability against mutational accessibility for one reference lineage, one dot per
      possible amino-acid replacement. A is the base model, B the model fine-tuned on spike, each
      in its Figure 2 colour. The three bands are the one-, two- and three-change replacements.
C     The rank correlation with accessibility computed inside each codon-distance group, where the
      genetic code cannot supply the correlation, for both models and the Grantham control.
      Same numbers and same resampling as Figure 2 (``mutacc.strata.strata_table``), and the
      same two-point reading as Figure 2D: the hollow marker is the plain Spearman correlation,
      the filled marker the partial correlation with Grantham similarity held constant, so the
      panel shows how much of each model's correlation survives the chemistry control.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mutacc import style
from mutacc.guide import parse_figure_args
from mutacc.io import write_caption, write_values
from mutacc.strata import ALL, strata_table

CONTROL = ("neg_grantham", "Grantham similarity (control)")
GROUP_LABELS = {ALL: "any", "1": "1", "2": "2", "3": "3"}
#: Figure 2D's layout: the two models sit either side of the tick, the control on it.
MODEL_OFFSET = (-0.16, 0.16)
GAP = 0.5       # extra space between "any" and the codon-distance groups


def parse(argv=None):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--n-boot", type=int, default=200)
    pre.add_argument("--seed", type=int, default=20261009)
    known, rest = pre.parse_known_args(argv)
    args = parse_figure_args(__doc__.split("\n\n")[0], rest)
    args.n_boot, args.seed = known.n_boot, known.seed
    return args


def scatter_panel(ax, frame: pd.DataFrame, model, lineage: str, rho: float, n_lineages: int) -> int:
    sub = frame[(frame["model"] == model.key) & (frame["lineage"] == lineage)]
    drawn = sub[(sub["plm_prob"] > 0) & (sub["mut_prob"] > 0)]
    # Same colour as the model wears in Figure 2; the number of changes is told by position.
    ax.scatter(drawn["plm_prob"], drawn["mut_prob"], s=1.6, color=style.colour_for_role(model.role),
               alpha=0.35, linewidths=0, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    tops = drawn.groupby("nt_steps")["mut_prob"].max()
    ax.set_ylim(top=tops[1] * 10 ** 1.4)
    for steps, top in tops.items():
        ax.text(0.02, top * 10 ** 0.35, f"{steps} change" + ("" if steps == 1 else "s"), fontsize=6.5,
                color=style.INK, va="bottom", ha="left", transform=ax.get_yaxis_transform())
    style.finalise_axes(ax, xlabel="Model probability of the replacement",
                        ylabel="Mutational accessibility")
    style.panel_title(ax, model.label, f"Spearman r = {rho:.2f} (mean of {n_lineages} lineages)")
    return len(sub) - len(drawn)


def strata_panel(ax, table: pd.DataFrame, guide) -> None:
    """Figure 2D's panel for this virus: raw and Grantham-controlled correlations together."""
    from matplotlib.lines import Line2D

    strata = list(table["stratum"].unique())
    xs = {s: i + (GAP if i > 0 else 0.0) for i, s in enumerate(strata)}
    models = sorted(guide.models, key=lambda m: {"base": 0, "finetune": 1}[m.role])
    for m, off in zip(models, MODEL_OFFSET):
        colour = style.colour_for_role(m.role)
        pick = lambda ctrl: table[(table["measure"] == "plm_vs_access")              # noqa: E731
                                  & (table["series"] == m.key) & (table["control"] == ctrl)
                                  ].set_index("stratum").loc[strata]
        raw, adj = pick(""), pick(CONTROL[0])
        x = np.array([xs[s] for s in strata]) + off

        # Hollow: the plain Spearman correlation. Filled with its interval: the partial
        # correlation, Grantham similarity held constant.
        ax.plot(x, raw["estimate"], linestyle="none", marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.2, zorder=4)
        ax.vlines(x, adj["lo"], adj["hi"], color=colour, linewidth=1.2, zorder=3)
        ax.plot(x, adj["estimate"], linestyle="none", marker="o", markersize=5, color=colour,
                markeredgecolor=colour, zorder=5)
        for s, r0, r1 in zip(strata, raw["estimate"], adj["estimate"]):
            if abs(r0 - r1) > 0.01:
                ax.annotate("", xy=(xs[s] + off, r1), xytext=(xs[s] + off, r0),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.0, shrinkA=3.5,
                                            shrinkB=3.5, mutation_scale=7), zorder=4)
        kept = adj.loc[ALL, "estimate"] / raw.loc[ALL, "estimate"]
        ax.text(xs[ALL] + off + (-0.07 if off < 0 else 0.07),
                (raw.loc[ALL, "estimate"] + adj.loc[ALL, "estimate"]) / 2, f"{kept:.0%}\nkept",
                ha="right" if off < 0 else "left", va="center", fontsize=6.5, color=colour)

    # The control itself, on the tick: Grantham against accessibility, no model involved.
    ctrl = table[(table["measure"] == "property_vs_access") & (table["series"] == CONTROL[0])
                 & (table["control"] == "")].set_index("stratum").loc[strata]
    x = np.array([xs[s] for s in strata])
    colour = style.colour_for_role("biochemistry")
    ax.vlines(x, ctrl["lo"], ctrl["hi"], color=colour, linewidth=1.1, alpha=0.8, zorder=2)
    ax.plot(x, ctrl["estimate"], linestyle="none", marker="_", markersize=10, markeredgewidth=1.6,
            color=colour, alpha=0.8, zorder=2)

    ax.axvline((xs[strata[0]] + xs[strata[1]]) / 2, color=style.GRID, linewidth=0.8, zorder=1)
    style.reference_line(ax, 0.0, "no relationship = 0", label_loc="left")
    ax.set_xticks([xs[s] for s in strata])
    ax.set_xticklabels([GROUP_LABELS.get(s, s) for s in strata])
    ax.set_xlim(-0.6, xs[strata[-1]] + 0.6)
    handles = [Line2D([], [], linestyle="none", marker="o", color=style.colour_for_role(m.role),
                      label=m.label) for m in models]
    handles += [
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
               markeredgecolor=style.INK_MUTED, markeredgewidth=1.2, label="before: Spearman r"),
        Line2D([], [], linestyle="none", marker="o", color=style.INK_MUTED,
               label="after: Grantham removed"),
        Line2D([], [], linestyle="none", marker="_", markersize=10, markeredgewidth=1.6,
               color=style.colour_for_role("biochemistry"), label=CONTROL[1]),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6, handlelength=1.0,
              borderaxespad=0.2, labelspacing=0.3)
    style.finalise_axes(ax, xlabel="Nucleotide changes needed to make the replacement",
                        ylabel="Spearman r with accessibility")
    style.panel_title(ax, "Model vs accessibility, codon distance held fixed",
                      "hollow: raw; filled: Grantham partialled out; interval resamples sites")


def lookup(table, series, stratum, column="estimate", control="") -> float:
    row = table[(table["series"] == series) & (table["stratum"] == stratum)
                & (table["control"] == control) & (table["measure"] != "")]
    return float(row[column].iloc[0])


def stem_for(guide) -> str:
    """sA_<last word of the virus>_model_vs_accessibility, e.g. the protein name."""
    return f"sA_{guide.virus.split()[-1].lower()}_model_vs_accessibility"


def caption(table, guide, ref_lineage: str, dropped: int) -> str:
    base = next(m for m in guide.models if m.role == "base")
    own = next(m for m in guide.models if m.role == "finetune")
    f = lambda s, k, c="estimate": f"{lookup(table, k, s, c):.2f}"            # noqa: E731
    # The same number with Grantham partialled out: the filled marker in panel C.
    g = lambda s, k, c="estimate": f"{lookup(table, k, s, c, CONTROL[0]):.2f}"  # noqa: E731
    return f"""
    **The Figure 2 test on a second virus: {guide.virus}.**
    Mutational accessibility is the probability that a nucleotide mutation process (here the
    spectrum for this virus, from data/spectra), given the reference codon at a site, produces a
    particular amino-acid replacement. The models are protein language models (PLMs): the base
    model as released, and the same model fine-tuned on {own.label.lower().removeprefix("fine-tuned on ")}.
    {guide.n_lineages} {guide.virus} lineages are used ({", ".join(guide.lineage_ids)}). They are variants of
    one protein and share almost every site, so they are near-duplicates, not independent
    replicates.

    **(A, B)** Every possible replacement at every site of {ref_lineage}, for the base model (A)
    and the fine-tune (B), in the colours used for those models in Figure 2. The three bands
    are the replacements needing one, two and three nucleotide changes.
    Axes are logarithmic{f"; {dropped} replacements with a probability of exactly zero are not drawn" if dropped else ""}.
    The correlation in the subtitle is the Spearman rank correlation (1 = same ranking, 0 =
    unrelated) over all replacements, averaged over the lineages; the dots are one lineage. It is partly produced by the genetic code,
    because replacements that need more changes are both rarer under mutation and less likely
    under the model; panel C removes that.

    **(C)** The same correlation inside each group of replacements that need the same number of
    nucleotide changes, where the code cannot drive it. "Any" pools all replacements. Markers are
    means over the lineages. As in Figure 2D, each model is shown twice: the hollow marker is the
    plain Spearman correlation and the filled marker is the partial Spearman correlation with
    Grantham similarity held constant, with the arrow between them and a 95% percentile bootstrap
    interval that resamples sites on the filled marker. Grantham (a chemical distance, sign
    flipped so that higher = more similar) is the chemistry control: it has no mutational
    information, and is drawn on the tick with its own interval. Pooling all replacements, the
    correlation that survives the control is {g(ALL, base.key)} of {f(ALL, base.key)} for the base
    model and {g(ALL, own.key)} of {f(ALL, own.key)} for the fine-tune. With three changes the
    controlled correlations are {g("3", base.key)} for the base model
    ({g("3", base.key, "lo")} to {g("3", base.key, "hi")}), {g("3", own.key)} for the fine-tune
    ({g("3", own.key, "lo")} to {g("3", own.key, "hi")}); Grantham itself is {f("3", CONTROL[0])}.

    The epoch curve of Figure 2C is not shown: the archive holds no per-epoch checkpoints of
    this backbone. The scores were produced by the exploration archive's pipeline; they are
    shipped as a table and are not regenerated by this repository.
    """


def main(argv=None) -> int:
    args = parse(argv)
    guide = args.guide
    frame = guide.read_table("replacement_scores", args.results)
    guide.check_lineages(frame["lineage"].unique(), f"{guide.name} replacement_scores")
    table = strata_table(frame, [m.key for m in guide.models], [CONTROL[0]],
                         n_boot=args.n_boot, seed=args.seed)

    base = next(m for m in guide.models if m.role == "base")
    own = next(m for m in guide.models if m.role == "finetune")
    ref_lineage = guide.lineages[0].id
    fig = style.figure(style.WIDTH_DOUBLE, 5.6)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    dropped = 0
    for ax, model in ((ax_a, base), (ax_b, own)):
        rho = lookup(table, model.key, ALL)
        dropped = max(dropped, scatter_panel(ax, frame, model, ref_lineage, rho, guide.n_lineages))
    strata_panel(ax_c, table, guide)
    for ax, letter, dx in ((ax_a, "A", -0.16), (ax_b, "B", -0.16), (ax_c, "C", -0.075)):
        style.panel_letter(ax, letter, dx=dx)

    stem = stem_for(guide)
    paths = style.save_figure(fig, args.out / stem)
    # Both the raw rows and the Grantham-controlled rows, since panel C now draws both.
    write_values(table, args.out, stem)
    write_caption(caption(table, guide, ref_lineage, dropped), args.out, stem)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
