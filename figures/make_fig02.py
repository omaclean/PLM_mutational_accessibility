"""Figure 2: does the model know about mutation supply, or only about chemistry?

    python figures/make_fig02.py --guide guides/h3n2_5lineage.yaml \\
        --guide-asymmetry guides/h3n2_13lineage.yaml

Reads each guide's ``replacement_scores`` table. Panels A and C are drawn from ``--guide``
and panel B from ``--guide-asymmetry`` (default: the same guide). Every panel states how many
lineages it used, counted from its guide.

A  Rank correlation between a model's probability and mutational accessibility, computed
   separately within each codon-distance stratum (replacements reachable by 1, 2 or 3
   nucleotide changes). Within a stratum the genetic code cannot supply the correlation.
   Grantham and BLOSUM62 are drawn as scores with no model and no codons in them. Grantham is
   the chemistry control. BLOSUM62 is only a comparator: an empirical substitution matrix
   already reflects which changes mutation makes easy, so it is not independent of
   accessibility.
B  Direction. For every pair of amino acids, how much easier is A -> B than B -> A under
   mutation, and under the model? A symmetric score (Grantham, BLOSUM62) is zero for every
   pair, so cannot express direction at all.
C  The part of panel B's agreement that is specific to a pair, after removing each amino
   acid's overall tendency.

RETIRED as a standalone figure (14 Oct 2026). Figure 2 is now output/fig02_composite.png,
whose panel D comes from make_fig02d.py. This module is kept because it is still the home of
PROPERTIES, TABLE, stratum_x and spread, which make_sD_partial_correlation.py and make_fig02d.py import, and it
still runs; but its own figure is not published and it still averages over all lineages
rather than using the single lineage the published panels now use.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd

from mutacc import style
from mutacc.guide import Guide, load_guide, parse_figure_args
from mutacc.io import write_caption, write_values
from mutacc.strata import ALL, asymmetry_summary, pair_asymmetry, strata_table

STEM = "fig02_model_vs_accessibility"
TABLE = "replacement_scores"

#: (column, label shown to the reader, marker, linestyle, hollow). Both are drawn in the
#: "biochemistry" role colour and told apart by marker and line style, never by colour alone.
PROPERTIES = [
    ("neg_grantham", "Grantham chemical\nsimilarity (control)", "s", "-", False),
    ("blosum62", "BLOSUM62\n(comparator)", "d", "--", True),
]
PROPERTY_ROLE = "biochemistry"
SERIES_OFFSET = {0: -0.21, 1: -0.07, 2: 0.07, 3: 0.21}
STRATUM_GAP = 1.5          # blank space between "all replacements" and the strata
LABEL_WRAP = 17

VALUE_COLUMNS = ["panel", "series", "measure", "group", "value", "lo", "hi", "x",
                 "n", "pooled", "lineage_min", "lineage_max", "n_lineages", "guide"]


def stratum_x(strata: list[str]) -> dict[str, float]:
    """x position of each stratum: 'all' at 0, then the codon-distance strata after a gap."""
    out, pos = {}, 0.0
    for s in strata:
        out[s] = pos
        pos = STRATUM_GAP if pos == 0.0 else pos + 1.0
    return out


def spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label heights apart so direct labels never overlap. Order is preserved."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out.tolist()


def series_specs(guide: Guide):
    """Everything panel A draws: key, display label, role, marker, linestyle, hollow, is_model."""
    specs = []
    for m in guide.models:
        specs.append(dict(key=m.key, label=textwrap.fill(m.label, LABEL_WRAP), role=m.role,
                          marker="o", linestyle="-", hollow=False, measure="plm_vs_access"))
    for col, label, marker, linestyle, hollow in PROPERTIES:
        specs.append(dict(key=col, label=label, role=PROPERTY_ROLE, marker=marker,
                          linestyle=linestyle, hollow=hollow, measure="property_vs_access"))
    return specs


def panel_strata(ax, table: pd.DataFrame, guide: Guide) -> None:
    strata = [s for s in table["stratum"].unique()]
    xs = stratum_x(strata)
    specs = series_specs(guide)
    last = strata[-1]
    ends = []
    for i, sp in enumerate(specs):
        sub = table[(table["measure"] == sp["measure"]) & (table["series"] == sp["key"])
                    & (table["control"] == "")].set_index("stratum").loc[strata]
        colour = style.colour_for_role(sp["role"])
        x = np.array([xs[s] for s in strata]) + SERIES_OFFSET[i]
        ax.vlines(x, sub["lo"], sub["hi"], color=colour, linewidth=1.0, alpha=0.9, zorder=3)
        ax.plot(x[1:], sub["estimate"].to_numpy()[1:], color=colour, linestyle=sp["linestyle"],
                linewidth=1.2, zorder=3)
        ax.plot(x, sub["estimate"], linestyle="none", marker=sp["marker"], markersize=4.2,
                markerfacecolor=style.SURFACE if sp["hollow"] else colour, markeredgecolor=colour,
                markeredgewidth=1.0, zorder=4, label=sp["label"].replace("\n", " "))
        ends.append((x[-1], float(sub["estimate"].iloc[-1])))
    heights = spread([y for _, y in ends], gap=0.085)
    for (x_end, _), y, sp in zip(ends, heights, specs):
        ax.text(x_end + 0.22, y, sp["label"], fontsize=6.3, color=style.INK, va="center", ha="left",
                linespacing=0.95)
    ax.axvline(STRATUM_GAP / 2, color=style.GRID, linewidth=0.8, zorder=1)
    style.reference_line(ax, 0.0, "no relationship = 0", label_loc="left")
    ax.set_xticks([xs[s] for s in strata])
    ax.set_xticklabels(["any" if s == ALL else s for s in strata])
    ax.set_xlim(-0.55, xs[last] + 2.0)
    ax.legend(loc="upper right", fontsize=6, handlelength=1.6, borderaxespad=0.2, labelspacing=0.35)


def panel_direction(ax, pairs: dict[str, pd.DataFrame], summaries: dict[str, dict], guide: Guide) -> None:
    for model in sorted(guide.models, key=lambda m: m.role == "finetune"):  # fine-tuned drawn on top
        p = pairs[model.key]
        s = summaries[model.key]
        ax.scatter(p["mut_asym"], p["plm_asym"], s=9, linewidths=0, alpha=0.6,
                   color=style.colour_for_role(model.role),
                   label=f"{model.label}: {s['n_agree']} of {s['n_pairs']} pairs agree")
    y_all = np.concatenate([pairs[k]["plm_asym"].to_numpy() for k in pairs])
    lo, hi = float(y_all.min()), float(y_all.max())
    ax.set_ylim(lo - 0.42 * (hi - lo), hi + 0.12 * (hi - lo))   # head-room for labels, floor for the legend
    style.reference_line(ax, 0.0, "Grantham, BLOSUM62 = 0", label_loc="left")
    style.reference_line(ax, 0.0, "equally easy both ways", axis="x")
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.2, markerscale=1.4,
              title="Pairs where model and mutation agree on direction", title_fontsize=6)


def panel_specific(ax, summaries: dict[str, dict], guide: Guide) -> None:
    groups = [("rho_overall", "All of\nthe direction"), ("rho_pair_specific", "Pair-specific\npart only")]
    step = 0.3
    for j, model in enumerate(guide.models):
        vals = [summaries[model.key][col] for col, _ in groups]
        x = np.arange(len(groups)) + (j - (len(guide.models) - 1) / 2) * step
        colour = style.colour_for_role(model.role)
        ax.vlines(x, 0, vals, color=colour, linewidth=1.6, zorder=3)
        ax.plot(x, vals, linestyle="none", marker="o", markersize=5.5, color=colour, zorder=4,
                label=model.label)
        side = 1 if j == len(guide.models) - 1 else -1     # labels point away from the neighbouring stem
        for xi, v in zip(x, vals):
            ax.text(xi + 0.07 * side, v, f"{v:.2f}", ha="left" if side > 0 else "right", va="center",
                    fontsize=6.5, color=style.INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for _, g in groups])
    ax.set_xlim(-0.55, len(groups) + 0.45)
    ax.set_ylim(-0.1, 0.75)
    style.reference_line(ax, 0.0, "no agreement = 0", label_loc="right")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.0, borderaxespad=0.2)


def build_values(table: pd.DataFrame, pairs, summaries, guide: Guide, guide_asym: Guide) -> pd.DataFrame:
    rows = []
    for r in table[table["control"] == ""].itertuples():
        rows.append(dict(panel="A", series=r.series, measure=r.measure, group=r.stratum, value=r.estimate,
                         lo=r.lo, hi=r.hi, n=r.n_rows, pooled=r.pooled, lineage_min=r.lineage_min,
                         lineage_max=r.lineage_max, n_lineages=guide.n_lineages, guide=guide.name))
    for key, p in pairs.items():
        for r in p.itertuples():
            rows.append(dict(panel="B", series=key, measure="direction", group=f"{r.aa_from}>{r.aa_to}",
                             value=r.plm_asym, x=r.mut_asym, n_lineages=guide_asym.n_lineages,
                             guide=guide_asym.name))
    for key, s in summaries.items():
        for col in ("rho_overall", "rho_pair_specific"):
            rows.append(dict(panel="C", series=key, measure=col, group=col, value=s[col], n=s["n_pairs"],
                             n_lineages=guide_asym.n_lineages, guide=guide_asym.name))
        rows.append(dict(panel="B", series=key, measure="n_agree", group="n_agree", value=s["n_agree"],
                         n=s["n_pairs"], n_lineages=guide_asym.n_lineages, guide=guide_asym.name))
        rows.append(dict(panel="B", series=key, measure="share_explained_by_amino_acid_tendency", group="plm",
                         value=s["share_by_amino_acid_plm"], n_lineages=guide_asym.n_lineages,
                         guide=guide_asym.name))
        rows.append(dict(panel="B", series=key, measure="share_explained_by_amino_acid_tendency", group="mut",
                         value=s["share_by_amino_acid_mut"], n_lineages=guide_asym.n_lineages,
                         guide=guide_asym.name))
    return pd.DataFrame(rows).reindex(columns=VALUE_COLUMNS)


def compute(frame: pd.DataFrame, frame_asym: pd.DataFrame, guide: Guide, guide_asym: Guide,
            *, n_boot: int, seed: int):
    guide.check_lineages(frame["lineage"].unique(), f"{guide.name} {TABLE}")
    guide_asym.check_lineages(frame_asym["lineage"].unique(), f"{guide_asym.name} {TABLE}")
    model_keys = [m.key for m in guide.models]
    table = strata_table(frame, model_keys, [p[0] for p in PROPERTIES], n_boot=n_boot, seed=seed)
    asym_keys = [m.key for m in guide_asym.models]
    pairs = {k: pair_asymmetry(frame_asym, k) for k in asym_keys}
    summaries = {k: asymmetry_summary(p) for k, p in pairs.items()}
    return table, pairs, summaries


def draw(table, pairs, summaries, guide: Guide, guide_asym: Guide):
    fig = style.figure(style.WIDTH_DOUBLE, 6.4)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.2])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    n_a, n_b = guide.n_lineages, guide_asym.n_lineages

    panel_strata(ax_a, table, guide)
    style.finalise_axes(
        ax_a, xlabel="Minimum nucleotide changes needed to make the replacement",
        ylabel="Correlation with mutational\naccessibility (Spearman r)",
        title=f"Model vs mutational accessibility, with codon distance held fixed ({n_a} lineages)")

    panel_direction(ax_b, pairs, summaries, guide_asym)
    style.finalise_axes(
        ax_b, xlabel="Mutation: how much easier A to B is than B to A (log10 ratio)",
        ylabel="Model: how much likelier A to B\nis than B to A (log10 ratio)",
        title=f"Which way is an amino-acid change easier? ({n_b} lineages)")

    panel_specific(ax_c, summaries, guide_asym)
    style.finalise_axes(
        ax_c, ylabel="Correlation between model and\nmutation directions (Spearman r)",
        title=f"Pair-specific direction ({n_b} lineages)")
    for ax, dx in ((ax_a, -0.075), (ax_b, -0.13), (ax_c, -0.2)):
        style.panel_letter(ax, "ABC"[[ax_a, ax_b, ax_c].index(ax)], dx=dx)
    return fig


def lookup(table: pd.DataFrame, series: str, stratum: str, column: str = "estimate", control: str = "") -> float:
    row = table[(table["series"] == series) & (table["stratum"] == stratum) & (table["control"] == control)]
    return float(row[column].iloc[0])


def fmt2(value: float) -> str:
    """Two decimals, never "-0.00"."""
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def lineage_spread_text(table: pd.DataFrame) -> str:
    """What the caption may say about the lineage-to-lineage spread, read from the numbers themselves."""
    rows = table[table["control"] == ""]
    width = (rows["hi"] - rows["lo"]).to_numpy(float)
    spread = (rows["lineage_max"] - rows["lineage_min"]).to_numpy(float)
    ok = width > 0
    if not ok.any():
        return "in the values table (lineage_min, lineage_max)"
    worst = float((spread[ok] / width[ok]).max())
    if worst < 1.0:
        return (f"narrower than the interval in every group (at most {worst:.0%} of its width; "
                "lowest and highest lineage are in the values table)")
    return "not narrower than the interval in every group (lowest and highest lineage are in the values table)"


def caption(table, summaries, guide: Guide, guide_asym: Guide) -> str:
    strata = [s for s in table["stratum"].unique() if s != ALL]
    last = strata[-1]
    ft = next(m for m in guide.models if m.role == "finetune")
    base = next(m for m in guide.models if m.role == "base")
    sf, sb = summaries[ft.key], summaries[base.key]
    props = dict((p[0], p[1].split("\n")[0]) for p in PROPERTIES)
    same = guide_asym.name == guide.name
    spread_text = lineage_spread_text(table)
    sets = (f"All three panels use the {guide.n_lineages}-lineage set ({guide.name})." if same else
            f"Panel A uses the {guide.n_lineages}-lineage set ({guide.name}); panels B and C use the "
            f"{guide_asym.n_lineages}-lineage set ({guide_asym.name}).")
    return f"""
    **Does the model carry information about mutation supply beyond chemistry?**
    {sets} Mutational accessibility is the probability that a nucleotide mutation process, given the
    reference codon at a site, produces a particular amino-acid replacement. The model is a protein
    language model (PLM) that assigns a probability to each possible replacement at each site of
    the {guide.virus}; "base" is the model as released and "fine-tuned" has been further trained
    on sequences of this protein.

    **(A)** Spearman rank correlation (1 = same ranking, 0 = unrelated, negative = opposite)
    between each score and mutational accessibility. "Any" pools every replacement. The other groups
    separate replacements by the minimum number of nucleotide changes needed (1, 2 or 3); within a
    group every replacement needs the same number of mutations, so codon distance itself cannot
    drive the correlation. Markers are means over {guide.n_lineages} lineages; vertical
    lines are 95% percentile bootstrap intervals that resample sites (a site brings all of its
    replacements with it, and the same sites are drawn in every lineage). The lineages share their
    sites, so the spread between them is not a generalisation interval and is not drawn; it is
    {spread_text}.
    Grantham (a chemical distance, sign flipped here so that higher = more similar) is the
    chemistry control: it is built from three amino-acid properties and has no mutational
    information. BLOSUM62 (the BLOcks SUbstitution Matrix, built from observed substitutions in
    related proteins) is shown only as a comparator, not as a control: substitutions seen in
    nature are already shaped by which changes mutation makes easy, so BLOSUM62 is not independent
    of accessibility. With {last} nucleotide changes the fine-tuned model's correlation is
    {fmt2(lookup(table, ft.key, last))} (95% interval {fmt2(lookup(table, ft.key, last, 'lo'))} to
    {fmt2(lookup(table, ft.key, last, 'hi'))}), the base model's is {fmt2(lookup(table, base.key, last))}
    ({fmt2(lookup(table, base.key, last, 'lo'))} to {fmt2(lookup(table, base.key, last, 'hi'))}),
    Grantham's is {fmt2(lookup(table, 'neg_grantham', last))} and BLOSUM62's is
    {fmt2(lookup(table, 'blosum62', last))}.

    **(B)** Each dot is one pair of amino acids that was observed in both directions
    ({sf['n_pairs']} pairs). Horizontal: log10 of mutational accessibility for A to B divided by
    that for B to A, averaged over all sites and lineages. Vertical: the same ratio for the model's
    probability. Which amino acid is called A is arbitrary (residues are ordered by side-chain
    class), and flipping a pair mirrors its dot through the centre without changing whether it
    agrees. Dots in the upper-right and lower-left quadrants agree on direction; the counts are in
    the legend. The Spearman correlation across pairs is {fmt2(sf['rho_overall'])} for the fine-tuned
    and {fmt2(sb['rho_overall'])} for the base model. Grantham and BLOSUM62 give the same score in both
    directions, so every pair sits at exactly zero and they cannot express direction. Pairs share
    amino acids, so these {sf['n_pairs']} points are not independent and no p-value is given.

    **(C)** Most of the direction in panel B is an overall tendency of each amino acid (the
    asymmetry of a pair is close to a difference between two per-amino-acid scores; these explain
    {100 * sf['share_by_amino_acid_mut']:.0f}% of the variance of the mutation asymmetry and
    {100 * sf['share_by_amino_acid_plm']:.0f}% and {100 * sb['share_by_amino_acid_plm']:.0f}% of the
    fine-tuned and base model's). Panel C repeats the correlation after fitting and removing that
    per-amino-acid part from both quantities, leaving what is specific to each pair:
    {fmt2(sf['rho_pair_specific'])} for the fine-tuned and {fmt2(sb['rho_pair_specific'])} for the base
    model. This is a descriptive comparison on one reference sequence per lineage, not a test of
    whether the model has learned the mutation process.
    """


def parse(argv=None):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--guide-asymmetry", type=Path, default=None,
                     help="guide for panels B and C (default: --guide)")
    pre.add_argument("--results-asymmetry", type=Path, default=None)
    pre.add_argument("--n-boot", type=int, default=200)
    pre.add_argument("--seed", type=int, default=20261009)
    known, rest = pre.parse_known_args(argv)
    args = parse_figure_args(__doc__.split("\n\n")[0], rest)
    args.guide_asym = load_guide(known.guide_asymmetry) if known.guide_asymmetry else args.guide
    args.results_asym = known.results_asymmetry or (
        args.results if args.guide_asym is args.guide else args.guide_asym.results_dir)
    args.n_boot, args.seed = known.n_boot, known.seed
    return args


def main(argv=None) -> int:
    args = parse(argv)
    guide, guide_asym = args.guide, args.guide_asym
    frame = guide.read_table(TABLE, args.results)
    frame_asym = frame if guide_asym is guide and args.results_asym == args.results \
        else guide_asym.read_table(TABLE, args.results_asym)
    table, pairs, summaries = compute(frame, frame_asym, guide, guide_asym, n_boot=args.n_boot, seed=args.seed)
    fig = draw(table, pairs, summaries, guide, guide_asym)
    paths = style.save_figure(fig, args.out / STEM)
    write_values(build_values(table, pairs, summaries, guide, guide_asym), args.out, STEM)
    write_caption(caption(table, summaries, guide, guide_asym), args.out, STEM)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
