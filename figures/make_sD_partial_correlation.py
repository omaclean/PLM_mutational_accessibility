"""Supplementary S4: is the model's link to accessibility just chemistry, by three routes?

    python figures/make_sD_partial_correlation.py --guide guides/h3n2_5lineage.yaml

Figure 2D asked whether a model's probability tracks mutational accessibility within each
codon-distance stratum. Here the same question is asked with amino-acid chemistry controlled
for, by three estimators chosen to fail in different ways rather than to agree by construction:

* **Partial Spearman** (top row). Each quantity is stripped of what the control's rank
  linearly predicts, and the residuals are correlated.
* **Grantham-matched Spearman** (top row). Replacements are binned by Grantham distance and
  correlated only against others of similar chemistry. This conditions exactly instead of
  regressing out, so a curved chemistry effect cannot leak through, which it can in a partial
  correlation.
* **Site-conditioned softmax exponent** (bottom row). A likelihood fitted to the model's own
  output, weighting replacements by the probability the model actually assigns them rather
  than treating all nineteen as equally important.

The first two share a weighting and differ in how the control is applied; the third differs in
both. Where all three agree the conclusion does not depend on the choice. Where they disagree
the figure says so rather than showing the one that reads best: see the caption on the
three-change group, where the rank estimators and the likelihood give opposite signs for the
untuned model.

Grantham is the control, and it is the only one drawn. BLOSUM62 used to sit beside it as a
comparator, but an empirical matrix of substitutions seen in nature already reflects which
changes mutation makes easy, so holding it constant removes genuine mutational signal along with
the chemistry and the residual is not "chemistry held constant". Showing it invited exactly the
reading it cannot support, so it is no longer plotted, captioned or written to the values table.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # make_fig02 sits beside this script

import argparse

import numpy as np
import pandas as pd

from make_fig02 import PROPERTIES, PROPERTY_ROLE, STRATUM_GAP, TABLE, stratum_x
from mutacc import style
from mutacc.guide import Guide, parse_figure_args
from mutacc.io import write_caption, write_values
from mutacc.linkfit import GRANTHAM_BINS, link_table
from mutacc.strata import ALL, primary_lineage, strata_table

STEM = "sD_partial_correlation"

#: Grantham only; see the module docstring for why BLOSUM62 was dropped.
CONTROL = next(p[0] for p in PROPERTIES if p[0] == "neg_grantham")

#: Series drawn in each row, as (measure, control, legend label, role, marker, linestyle,
#: hollow). The role decides the colour: a series carrying no chemistry control
#: wears the model's own colour, one that does wears the biochemistry colour.
ROW_RANK = [
    ("plm_vs_access", "", "Spearman ρ (no control)", None, "o", "-", False),
    ("plm_vs_access", CONTROL, "Partial Spearman ρ, Grantham rank removed",
     PROPERTY_ROLE, "s", "-", False),
    ("matched_rank", CONTROL, "Spearman ρ within Grantham bins (matched)",
     PROPERTY_ROLE, "v", "--", True),
]
ROW_FIT = [
    ("softmax_beta", "", "Softmax exponent (no control)", None, "o", "-", False),
    ("softmax_beta", CONTROL, "Softmax exponent, Grantham spline included",
     PROPERTY_ROLE, "s", "-", False),
]

#: One horizontal offset per drawn series, so overlapping intervals stay readable.
OFFSETS = {3: (-0.16, 0.0, 0.16), 2: (-0.10, 0.10)}


def compute(frame: pd.DataFrame, guide: Guide, *, n_boot: int, seed: int) -> pd.DataFrame:
    # The guide is checked against the whole table first, so a results set with the
    # wrong lineages still fails loudly, and only then is one lineage selected.
    guide.check_lineages(frame["lineage"].unique(), f"{guide.name} {TABLE}")
    chosen = primary_lineage(frame, [(lin.id, lin.root_type) for lin in guide.lineages],
                             guide.primary_lineage)
    one = frame[frame["lineage"].astype(str) == chosen]
    keys = [m.key for m in guide.models]
    ranks = strata_table(one, keys, [CONTROL], n_boot=n_boot, seed=seed)
    ranks = ranks[ranks["measure"] == "plm_vs_access"]
    fits = link_table(one, keys, CONTROL, n_boot=n_boot, seed=seed)
    # The plain Spearman is computed by both routes on purpose; it is drawn once, from the
    # table Figure 2D uses, and the copy is kept only so a test can confirm they agree.
    fits = fits[fits["measure"] != "spearman"]
    columns = ["measure", "series", "control", "stratum", "estimate", "lo", "hi", "n_rows"]
    table = pd.concat([ranks[columns], fits[columns]], ignore_index=True)
    return table.assign(lineage=chosen)


def panel(ax, table: pd.DataFrame, model, lines) -> None:
    strata = [s for s in table["stratum"].unique()]
    xs = stratum_x(strata)
    for off, (measure, control, label, role, marker, linestyle, hollow) in zip(
            OFFSETS[len(lines)], lines):
        sub = table[(table["series"] == model.key) & (table["measure"] == measure)
                    & (table["control"] == control)].set_index("stratum").loc[strata]
        colour = style.colour_for_role(role or model.role)
        x = np.array([xs[s] for s in strata]) + off
        ax.vlines(x, sub["lo"], sub["hi"], color=colour, linewidth=1.0, zorder=3)
        ax.plot(x[1:], sub["estimate"].to_numpy()[1:], color=colour, linestyle=linestyle,
                linewidth=1.2, zorder=3)
        ax.plot(x, sub["estimate"], linestyle="none", marker=marker, markersize=4.2,
                markerfacecolor=style.SURFACE if hollow else colour, markeredgecolor=colour,
                markeredgewidth=1.0, zorder=4, label=label)
    # No end labels. Direct labelling beats a legend at two series, but at three the labels
    # land on top of each other and on the data, and every series is already named once.
    ax.axvline(STRATUM_GAP / 2, color=style.GRID, linewidth=0.8, zorder=1)
    style.reference_line(ax, 0.0, "no relationship = 0", label_loc="left")
    ax.set_xticks([xs[s] for s in strata])
    ax.set_xticklabels(["any" if s == ALL else s for s in strata])
    ax.set_xlim(-0.5, xs[strata[-1]] + 1.7)
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.6, borderaxespad=0.2,
              labelspacing=0.3)


def draw(table: pd.DataFrame, guide: Guide):
    fig = style.figure(style.WIDTH_DOUBLE, 6.4)
    axes = np.atleast_2d(fig.subplots(2, len(guide.models), sharex=True, squeeze=False))
    # Every panel carries the model and the lineage, including the lower row. A supplementary
    # panel is quoted on its own often enough that it has to say what it is without the one
    # above it, and the row is told apart by its y-axis rather than by the title.
    rows = [(ROW_RANK, "Correlation with mutational\naccessibility (Spearman ρ)"),
            (ROW_FIT, "Implicit exponent on\naccessibility (softmax fit)")]
    for r, (lines, ylabel) in enumerate(rows):
        sub = table[table["measure"].isin({m for m, *_ in lines})]
        for c, model in enumerate(guide.models):
            ax = axes[r, c]
            panel(ax, sub, model, lines)
            # Both columns share one scale per row, so the two models are compared by eye
            # and not by reading two differently stretched axes.
            floor, ceiling = float(np.nanmin(sub["lo"])), float(np.nanmax(sub["hi"]))
            pad = ceiling - floor
            ax.set_ylim(floor - 0.05 * pad, ceiling + 0.22 * pad)
            style.finalise_axes(
                ax,
                xlabel="Minimum nucleotide changes needed" if r == len(rows) - 1 else "",
                ylabel=ylabel if c == 0 else "",
                title=f"{model.label} (lineage {table['lineage'].iloc[0]})")
    style.panel_letters(axes.ravel(order="C"), dx=-0.14)
    return fig


def lookup(table, measure, series, stratum, control, column="estimate"):
    row = table[(table["measure"] == measure) & (table["series"] == series)
                & (table["stratum"] == stratum) & (table["control"] == control)]
    return float(row[column].iloc[0])


def interval(table, measure, series, stratum, control) -> str:
    lo = lookup(table, measure, series, stratum, control, "lo")
    hi = lookup(table, measure, series, stratum, control, "hi")
    return f"{lookup(table, measure, series, stratum, control):.2f} [{lo:.2f}, {hi:.2f}]"


def caption(table: pd.DataFrame, guide: Guide) -> str:
    last = [s for s in table["stratum"].unique() if s != ALL][-1]
    ft = next(m for m in guide.models if m.role == "finetune")
    base = next(m for m in guide.models if m.role == "base")
    return f"""
    **Removing amino-acid chemistry from the model-accessibility link, three ways.**
    One lineage ({table['lineage'].iloc[0]}) from the set {guide.name}; one column per model.
    The horizontal axis is the minimum number of nucleotide changes a replacement needs, with
    "any" pooling all of them. Vertical lines are 95% percentile bootstrap intervals that
    resample **sites** -- whole positions, carrying all of their replacements -- because
    replacements at one site share a backbone and are not independent. All uncertainty shown is
    over sites.

    **Top row (panels A and B): rank-based, in correlation units.** The coloured circles repeat
    Figure 2D, the plain Spearman correlation between the model's probability of a replacement
    and that replacement's mutational accessibility. The two pink series control for chemistry,
    in two different ways, and the point of showing both is that they can disagree.

    *Partial Spearman (squares).* Take three quantities over the same replacements: the model's
    probability, the accessibility, and the Grantham score. Replace each by its rank. Regress
    the ranked model probability on the ranked Grantham score and keep the residual -- the part
    of the model's ranking that chemistry cannot linearly account for. Do the same for ranked
    accessibility. The partial Spearman r is the ordinary correlation between those two
    residuals; equivalently, and as computed here, it is
    (r_ma - r_mc * r_ac) / sqrt((1 - r_mc^2)(1 - r_ac^2)) from the three pairwise Spearman
    correlations between model (m), accessibility (a) and control (c). The two routes are
    algebraically identical.

    *Grantham-matched Spearman (hollow triangles).* The known weakness of a partial correlation
    is that it removes only the *monotone* part of the control's effect, so any curved
    dependence on chemistry survives into the residual and is still counted as model signal.
    This series avoids that by conditioning instead of regressing: replacements are sorted into
    {GRANTHAM_BINS} bins of Grantham distance, the correlation is computed within each bin
    among replacements of similar chemistry only, and the bin correlations are pooled on
    Fisher's z scale weighted by n - 3. Nothing is subtracted, so nothing non-monotone can leak
    through. It is on the same scale as the other two and can be read directly against them.

    **Bottom row (panels C and D): likelihood-based, in exponent units.** A rank correlation
    counts every replacement equally, so a swap between two the model has all but ruled out
    weighs as much as a swap between its top two -- and the model's probabilities span eight
    orders of magnitude at a site. This row removes that objection by fitting the model's own
    output as what it is, a softmax over amino acids at a site:
    q(site, aa) proportional to exp(beta * ln accessibility + f(Grantham)). Chemistry enters as
    a restricted cubic spline, so its effect is not forced to be monotone or even smooth in any
    particular direction. The fit is within each site, so the site intercepts cancel and only
    the ordering and spacing of replacements at a site informs it; each codon-distance group is
    normalised within itself, so every site contributes equal weight to its own group and the
    three-change estimate does not hang on a handful of high-probability sites. The "any" fit
    carries no codon-distance offsets, because across all nineteen replacements the step
    between the groups *is* most of the variation in log accessibility, and offsets plus a
    slope would be estimating the same quantity.

    *What beta is, and what it is not.* Beta is a power-law exponent, d ln q / d ln A: the model
    behaves as though it had multiplied its probabilities by accessibility raised to beta. It is
    therefore in the same units as the exponent the alpha sweep tunes from outside, and the
    comparison is valid in either log base, because an exponent is invariant to the base as long
    as one base is used on both sides. It is **not the same quantity** as that alpha and should
    not be quoted as if it were. The sweep's alpha is an external weight chosen to maximise
    agreement with a measured phenotype; beta is how much of that behaviour the model's own
    output already shows, with no phenotype involved. A test generates data from a known
    exponent and confirms the fit recovers it, so the scale is pinned rather than assumed.

    **What the three estimators agree on.** For "{ft.label}" the chemistry control changes
    almost nothing, by any route: with one nucleotide change the correlation is
    {interval(table, 'plm_vs_access', ft.key, '1', '')} alone and
    {interval(table, 'plm_vs_access', ft.key, '1', CONTROL)} with Grantham's rank removed, and
    the exponent is {interval(table, 'softmax_beta', ft.key, '1', '')} alone against
    {interval(table, 'softmax_beta', ft.key, '1', CONTROL)} with the spline. Its link to
    accessibility is not chemistry wearing another name. At {last} nucleotide changes the
    control does not merely fail to remove the signal, it uncovers more of it: the correlation
    rises from {interval(table, 'plm_vs_access', ft.key, last, '')} alone to
    {interval(table, 'plm_vs_access', ft.key, last, CONTROL)} with Grantham's rank removed, and
    the matched estimator agrees ({interval(table, 'matched_rank', ft.key, last, CONTROL)}).
    Chemistry was masking part of this model's preference for accessible replacements rather
    than supplying it. For "{base.label}" at two nucleotide changes all three estimators keep
    the residual signal clear of zero.

    **Where they disagree, and why that is in the figure.** Two cells do not reproduce across
    estimators, and both involve the untuned model. With one nucleotide change its
    chemistry-controlled signal is above zero on both rank estimators
    ({interval(table, 'plm_vs_access', base.key, '1', CONTROL)} partial,
    {interval(table, 'matched_rank', base.key, '1', CONTROL)} matched) but not distinguishable
    from zero once replacements are weighted by the probability the model assigns them
    ({interval(table, 'softmax_beta', base.key, '1', CONTROL)}). With {last} nucleotide changes
    the two rows take opposite signs: the rank estimators sit at or just above zero
    ({interval(table, 'plm_vs_access', base.key, last, CONTROL)} partial) while the exponent is
    negative ({interval(table, 'softmax_beta', base.key, last, CONTROL)}), meaning that among
    the replacements it considers at all plausible, the untuned model mildly *disprefers* the
    more accessible ones. Both readings are defensible and they answer slightly different
    questions -- all replacements equally, or weighted by the model's own confidence -- so the
    honest report is that the untuned model's residual link to accessibility is weak and depends
    on the weighting, while the fine-tuned model's does not depend on it at all. The claim this
    figure supports is the second one.

    The control score is Grantham chemical similarity, a distance built from composition,
    polarity and volume, sign flipped; it contains no model and no codon information. An earlier
    version also plotted BLOSUM62 (the BLOcks SUbstitution Matrix) beside it. That has been
    removed: BLOSUM62 summarises substitutions observed in nature, which already reflect which
    changes mutation makes easy, so holding it constant removes genuine mutational signal along
    with the chemistry, and the residual cannot be read as "chemistry held constant".

    **Why one lineage and not the set.** Earlier versions reported the mean over all
    {guide.n_lineages} lineages. Those lineages are scored at the same sites and are
    pseudo-replicates for this measurement: across the set the per-lineage estimates differ by a
    standard deviation of 0.001 to 0.013, against a site-bootstrap interval 0.04 to 0.11 wide,
    so the between-lineage spread is three to ninety-six times narrower than the uncertainty
    already shown. Averaging them moved no estimate by more than about 0.015 and added nothing
    to the interval, while reporting "{guide.n_lineages} lineages" invited the reader to hear
    replication that the shared sites do not provide. The lineage is chosen by a stated rule
    (provenance established rather than unverified, then the most scored replacements, then
    sorted identifier) rather than by hand. Every interval here is a statement about these
    sites, not about a new lineage or a new protein.
    """


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--n-boot", type=int, default=200)
    pre.add_argument("--seed", type=int, default=20261009)
    known, rest = pre.parse_known_args(argv)
    args = parse_figure_args(__doc__.split("\n\n")[0], rest)
    guide = args.guide
    table = compute(guide.read_table(TABLE, args.results), guide, n_boot=known.n_boot, seed=known.seed)
    fig = draw(table, guide)
    paths = style.save_figure(fig, args.out / STEM)
    write_values(table.assign(n_lineages=1, guide=guide.name), args.out, STEM)
    write_caption(caption(table, guide), args.out, STEM)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
