"""S5: every mutation spectrum compared against every other in codon space.

    python figures/make_sE_spectra_codon_space.py

A nucleotide mutation spectrum says how often each base changes into each other base. On its
own that is a 4x4 table and says nothing about proteins. Pushed through the genetic code it
becomes something a protein model could in principle have learned: for each of the 64 codons,
the chance of turning into each of the 20 amino acids.

Two spectra can then be compared by asking how much of one such table the other accounts for.
The statistic is the manuscript's own, the percent of variance in the target retained by the
predictor. It is 100 for a perfect account, 0 for one no better than the target's own average,
and negative for one that is worse than that average.

The "decomposition" in the figure's title is the compress-and-rebuild step. Each predictor's
64x20 table is first averaged down to 20x20, one row per amino acid the codon currently codes
for, and then expanded back to 64 rows. That throws away everything the table knew about
*which* codon a protein is using and keeps only which amino acid it codes for. The diagonal is
therefore how much of a spectrum's codon-level structure is codon-level at all, and the
off-diagonal cells say how much of one virus's codon table another spectrum accounts for after
the same treatment.

Panel B turns the same comparisons into a ranking: if you had to pick ONE matrix to
stand in for a mutation spectrum you had not measured, which would it be? Each spectrum is
scored by how it does against every other by correlation (B). Influenza is counted once there, so its two
subtypes cannot inflate each other. The answer is the uncomfortable one: the simple generic
matrices win and the focus spectrum comes last, which is the whole point of the figure.

Reads ``spectra_pairwise``, ``spectra_sources`` and ``spectra_generic_stand_in``, written by
``tools/export_spectra.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import ScaledTranslation

from mutacc import style
from mutacc.io import write_caption, write_values

STEM = "sE_spectra_in_codon_space"
#: Highlight colour for the focus spectrum in panels B and C. Everything else is grey: the
#: contrast is "this one against the rest", not a set of categories, and the house green
#: would read as the fine-tuned model.
FOCUS_COLOUR = style.CATEGORICAL[0]
OTHER_COLOUR = style.NULL_GREY
RESULTS_SET = "spectra_codon_space"
REPO_ROOT = Path(__file__).resolve().parents[1]

def _relative_luminance(rgb) -> float:
    """WCAG relative luminance of an RGB triple in 0-1."""
    channels = []
    for c in rgb[:3]:
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b) -> float:
    """WCAG contrast ratio between two RGB triples, 1 (identical) to 21 (black on white)."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    lo, hi = sorted((la, lb))
    return (hi + 0.05) / (lo + 0.05)


def text_colour_on(rgba) -> str:
    """Ink or surface, whichever actually reads better on this cell.

    Picked by measured contrast rather than by a cut-off on the colour scale, because the
    scale here is not symmetric about zero: a fixed fraction of the range lands at a
    different lightness on the two sides and leaves some cells sitting near the floor.
    """
    import matplotlib.colors as mcolors
    ink = mcolors.to_rgb(style.INK)
    surface = mcolors.to_rgb(style.SURFACE)
    return style.INK if _contrast(rgba, ink) >= _contrast(rgba, surface) else style.SURFACE


def load(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairwise = pd.read_csv(results / "spectra_pairwise.tsv.gz", sep="\t")
    sources = pd.read_csv(results / "spectra_sources.tsv.gz", sep="\t")
    stand_in = pd.read_csv(results / "spectra_generic_stand_in.tsv.gz", sep="\t")
    missing = {"target", "predictor", "retained_compressed", "retained_uncompressed", "corr"} - set(pairwise)
    if missing:
        raise ValueError(f"spectra_pairwise is missing {sorted(missing)}")
    sources = sources.sort_values("plot_order")
    keys = list(sources["key"])
    seen = set(pairwise["target"]) | set(pairwise["predictor"])
    if seen != set(keys):
        raise ValueError(f"the pairwise table and the sources table disagree: {sorted(seen ^ set(keys))}")
    if len(pairwise) != len(keys) ** 2:
        raise ValueError(f"expected {len(keys) ** 2} pairs, found {len(pairwise)}")
    need = {"key", "label", "mean_corr_to_others", "mean_retained_predicting_others",
            "free_nt_parameters", "is_focus", "n_others"}
    if need - set(stand_in):
        raise ValueError(f"spectra_generic_stand_in is missing {sorted(need - set(stand_in))}")
    if int(stand_in["is_focus"].astype(bool).sum()) != 1:
        raise ValueError("spectra_generic_stand_in needs exactly one focus spectrum")
    # Every spectrum scores against the same number of others, which is what makes the two
    # rankings comparable. Influenza is counted once upstream, so this is one fewer than the
    # heatmap has rows.
    if stand_in["n_others"].nunique() != 1:
        raise ValueError("spectra_generic_stand_in: the spectra are scored against different "
                         "numbers of others, so the means are not comparable")
    return pairwise, sources, stand_in


def matrix(pairwise: pd.DataFrame, keys: list[str], column: str) -> np.ndarray:
    """Square array of ``column``, rows targets and columns predictors, in ``keys`` order."""
    wide = pairwise.pivot(index="target", columns="predictor", values=column)
    values = wide.loc[keys, keys].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column}: the comparison matrix has a non-finite cell")
    return values


def merged_labels(sources: pd.DataFrame, stand_in: pd.DataFrame) -> list[str]:
    """Spectra in the heatmap that the ranking folded into a sibling, in heatmap order."""
    kept = set(stand_in["label"])
    return [str(l) for l in sources["label"] if l not in kept]


def merged_tagline(merged: list[str]) -> str:
    """Kept short on purpose.

    This sits over a half-width panel, and the constrained layout treats an over-long
    tagline as a width the axes must find: at 58 characters it squeezed both ranking
    panels to nothing. What "nt params" means is in the caption, not here.
    """
    if not merged:
        return "every spectrum counted once"
    return f"{' and '.join(merged)} counted once, not separately"


def draw_stand_in(ax, stand_in: pd.DataFrame, column: str, *, xlabel: str, title: str,
                  tagline: str, fmt: str) -> None:
    """One ranked bar panel: how well each spectrum stands in for all the others.

    Sorted by the column being drawn. The focus spectrum is the only coloured
    bar; the rest are grey, because the question is where the focus lands, not how the
    others compare with each other.
    """
    d = stand_in.sort_values(column, ascending=False).reset_index(drop=True)
    y = np.arange(len(d))[::-1].astype(float)
    values = d[column].to_numpy(dtype=float)
    colours = [FOCUS_COLOUR if f else OTHER_COLOUR for f in d["is_focus"].astype(bool)]
    ax.barh(y, values, color=colours, height=0.66, linewidth=0, zorder=2)

    span = float(np.nanmax(values) - min(0.0, float(np.nanmin(values))))
    pad = 0.02 * span
    for yy, v in zip(y, values):
        ax.annotate(fmt.format(v), xy=(v + (pad if v >= 0 else -pad), yy), fontsize=7.5,
                    color=style.INK, va="center", ha="left" if v >= 0 else "right", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    for yy, r in zip(y, d.itertuples()):
        ax.text(-0.02, yy + 0.03, r.label, ha="right", va="bottom", fontsize=8.0, fontweight="bold",
                color=style.INK, transform=ax.get_yaxis_transform())
        ax.text(-0.02, yy - 0.03, f"{int(r.free_nt_parameters)} nt params", ha="right", va="top",
                fontsize=6.5, color=style.INK_MUTED, transform=ax.get_yaxis_transform())
    ax.axvline(0.0, color=style.INK_MUTED, linewidth=0.8, zorder=1)
    ax.set_ylim(-0.7, len(d) - 0.3)
    # Room at both ends for the value labels, which sit outside the bars.
    lo = min(0.0, float(np.nanmin(values))) - 0.26 * span
    ax.set_xlim(lo, float(np.nanmax(values)) + 0.20 * span)
    ax.grid(axis="y", visible=False)
    style.finalise_axes(ax, xlabel=xlabel)
    ax.xaxis.label.set_size(8.5)
    style.panel_title(ax, title, tagline)
    clarify_tagline(ax, fontsize=8.5)


def clarify_tagline(ax, fontsize: float = 8.5) -> None:
    """Make the panel's tagline carry at normal reading distance."""
    tag = ax.texts[-1]
    tag.set_fontsize(fontsize)
    tag.set_color(style.INK_MUTED)
    tag.set_in_layout(False)
    # The helper places the tagline at 1.015 in AXES fractions, so how far above the axes it
    # lands depends on how tall that axes is: on the heatmap, which is three times the height
    # of the ranking panels, it rose into the title. Pin it a fixed 3 pt above the axes
    # instead, and give the title room to clear it.
    tag.set_position((0.0, 1.0))
    tag.set_transform(ax.transAxes
                      + ScaledTranslation(0.0, 3 / 72, ax.figure.dpi_scale_trans))
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=14)


def draw(pairwise: pd.DataFrame, sources: pd.DataFrame, stand_in: pd.DataFrame | None = None):
    keys = list(sources["key"])
    labels = list(sources["label"])
    values = matrix(pairwise, keys, "retained_compressed")

    # Double width, with height tuned so the heatmap spreads wide and cells are wider than tall.
    fig = style.figure(style.WIDTH_DOUBLE, 6.8 if stand_in is not None else 4.6)
    if stand_in is not None:
        # A across the top, B beneath.
        gs = fig.add_gridspec(2, 1, height_ratios=[1.7, 1.0])
        # The colour bar gets a dedicated column of the top subgridspec.
        top = gs[0].subgridspec(1, 2, width_ratios=[1.0, 0.030], wspace=0.015)
        ax = fig.add_subplot(top[0, 0])
        cax = fig.add_subplot(top[0, 1])
        # 3-column subgridspec decouples bottom column sizing from the 2-column top subgridspec.
        bottom = gs[1].subgridspec(1, 3, width_ratios=[1.0, 0.65, 0.01], wspace=0.01)
        ax_b = fig.add_subplot(bottom[0, 0])
    else:
        ax = fig.subplots()
        ax_b = cax = None

    # Centre the colour scale on zero: zero is "no better than the target's own average",
    # which is the only value on this scale that means something on its own.
    limit = float(np.nanmax(np.abs(values)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=100.0)
    # "auto" lets panel A spread wide across its row rather than collapsing to a tall vertical box.
    image = ax.imshow(values, cmap=style.DIVERGING, norm=norm,
                      aspect="auto" if stand_in is not None else "equal")

    for i in range(len(keys)):
        for j in range(len(keys)):
            v = values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                    color=text_colour_on(image.cmap(norm(v))))

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=2.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    style.finalise_axes(
        ax,
        xlabel="Spectrum used as the predictor, averaged over codons and rebuilt",
        ylabel="Spectrum being predicted",
    )
    ax.xaxis.label.set_size(9.5)
    ax.yaxis.label.set_size(9.5)
    style.panel_title(
        ax,
        "Percent of variance retained, one spectrum predicting another",
        "100 is a perfect account; 0 is no better than the target's own average; "
        "below 0 is worse than that",
    )
    ax.title.set_fontsize(10.0)
    clarify_tagline(ax, fontsize=8.5)
    bar = (fig.colorbar(image, cax=cax) if cax is not None
           else fig.colorbar(image, ax=ax, fraction=0.040, pad=0.02))
    bar.set_label("% of variance retained", fontsize=9.0)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8.0)

    if stand_in is not None:
        n = int(stand_in["n_others"].iloc[0])
        merged = merged_labels(sources, stand_in)
        draw_stand_in(ax_b, stand_in, "mean_corr_to_others",
                      xlabel="Mean Pearson r against every other spectrum",
                      title="Best generic stand-in, by correlation",
                      tagline=f"scored against the other {n} spectra", fmt="{:.3f}")
        style.panel_letters([ax], dx=-0.06, dy=1.055)
        style.panel_letters([ax_b], start="B", dx=-0.30, dy=1.16)
    return fig, values


def ranking_paragraph(stand_in: pd.DataFrame, merged: list[str]) -> str:
    """Panel B, every number read off the ranking table."""
    focus = stand_in.loc[stand_in["is_focus"].astype(bool)].iloc[0]
    n = int(stand_in["n_others"].iloc[0])
    by_corr = stand_in.sort_values("mean_corr_to_others", ascending=False).reset_index(drop=True)
    best_c = by_corr.iloc[0]
    rank_c = int(by_corr.index[by_corr["key"] == focus["key"]][0]) + 1
    simplest = stand_in.loc[stand_in["free_nt_parameters"].idxmin()]
    simplest_rank = int(by_corr.index[by_corr["key"] == simplest["key"]][0]) + 1
    merged_phrase = ("every spectrum is counted once" if not merged else
                     f"{' and '.join(merged)} is counted with its subtype rather than separately, "
                     "so a pair of close relatives cannot prop each other up")
    lead = (f"{best_c['label']} ranks first at {best_c['mean_corr_to_others']:.3f}")
    return f"""

    (**B**) The same comparisons asked as a ranking: if one matrix had to stand in for a
    spectrum nobody had measured, which should it be? Each spectrum is scored against the other
    {n} as a mean Pearson correlation; {merged_phrase}. Each bar is labelled with the spectrum's free nucleotide rate
    parameters, which is how much shape it can express at all. {lead}. {focus['label']}, the spectrum this paper
    is about, comes {rank_c} of {len(stand_in)} on correlation ({focus['mean_corr_to_others']:.3f}),
    below matrices with far fewer free parameters: {simplest['label']} has {int(simplest['free_nt_parameters'])}
    free nucleotide parameters against {focus['label']}'s {int(focus['free_nt_parameters'])} and still ranks
    {simplest_rank}. A spectrum fitted to one virus is a poor general stand-in, and being able
    to express more is not what makes a matrix transferable. Bars are grey except the focus;
    these are means over a fixed set of {n} comparisons, not estimates with a sampling
    distribution, so no interval is drawn.
    """


def caption(pairwise: pd.DataFrame, sources: pd.DataFrame,
            stand_in: pd.DataFrame | None = None) -> str:
    keys = list(sources["key"])
    labels = dict(zip(sources["key"], sources["label"]))
    focus = sources.loc[sources["is_focus"], "key"].iloc[0]

    def cell(target, predictor, column="retained_compressed"):
        row = pairwise[(pairwise["target"] == target) & (pairwise["predictor"] == predictor)]
        return float(row[column].iloc[0])

    # The matrix the manuscript contrasted the focus with. Taken from the sources table so
    # the caption cannot quote a published number against the wrong spectrum.
    marked = sources.loc[sources["manuscript_role"] == "comparator", "key"]
    if len(marked) != 1:
        raise ValueError(f"expected exactly one comparator spectrum, found {len(marked)}")
    comparator = marked.iloc[0]

    n_free = int(sources.loc[sources["key"] == focus, "free_nt_parameters"].iloc[0])
    n_comparator = int(sources.loc[sources["key"] == comparator, "free_nt_parameters"].iloc[0])

    ranking = "" if stand_in is None else ranking_paragraph(stand_in, merged_labels(sources, stand_in))

    # The point is made by the spectrum that is AS expressive as the focus and still does
    # worst, not by the one that does best: a close relative scoring highly would argue the
    # other way.
    as_expressive = [k for k in keys if k != focus
                     and int(sources.loc[sources["key"] == k, "free_nt_parameters"].iloc[0]) == n_free]
    if not as_expressive:
        raise ValueError("no spectrum is as expressive as the focus; the caption's claim has no support")
    rival = min(as_expressive, key=lambda k: cell(focus, k))

    return f"""
    **Mutation spectra compared in codon space.**
    Each nucleotide spectrum is pushed through the standard genetic code into a 64 x 20 table
    giving the chance that a codon changes into each amino acid, after scaling every spectrum to
    the same total mutation rate so that only the shape of the spectrum counts. The cell at row
    *i*, column *j* is the percent of variance in spectrum *i*'s table that spectrum *j*'s table
    accounts for, 100 x (1 - var(residual) / var(target)). Before being used as a predictor every
    table is averaged over the codons of each amino acid and rebuilt to 64 rows, which removes
    everything it knew about which codon is in use. The scale is centred on zero because zero is
    the only interpretable point on it: a predictor scoring zero is exactly as good as the
    target's own average, and the negative cells are predictors that are worse than that.
    The diagonal is each spectrum against itself after the same compression, so it measures how
    much of that spectrum's codon-space structure survives being reduced to amino acids; for
    {labels[focus]} it is {cell(focus, focus):.1f}%. The manuscript's two published numbers are
    both cells of this matrix: the {labels[focus]} diagonal, and {labels[comparator]} predicting
    {labels[focus]} at {cell(focus, comparator):.1f}%. Without the compression step that second
    comparison is {cell(focus, comparator, "retained_uncompressed"):.1f}%, a different quantity
    kept in the values table alongside. Expressivity alone does not explain that gap:
    {labels[rival]} has the same {n_free} free nucleotide parameters as {labels[focus]}, against
    {n_comparator} for {labels[comparator]}, yet scores {cell(focus, rival):.1f}% against
    {labels[focus]}'s table — far worse than the much simpler matrix the manuscript used. What a
    matrix can express is therefore not what decides whether it can stand in for this one.
    Spectra are ordered by how well they correlate with
    {labels[focus]}, which is the only reason the matrix is not symmetric in appearance; it is
    not symmetric in value either, because retained variance is scaled by the target's own
    variance and so depends on which of the pair is being predicted.{ranking}
    """


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "results" / RESULTS_SET,
                    help="directory holding the spectra comparison tables")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "output",
                    help="output directory for PDF + PNG, values and caption")
    args = ap.parse_args(argv)

    pairwise, sources, stand_in = load(args.results)
    fig, _ = draw(pairwise, sources, stand_in)
    paths = style.save_figure(fig, args.out / STEM)

    order = {k: i for i, k in enumerate(sources["key"])}
    out = pairwise.assign(target_order=pairwise["target"].map(order),
                          predictor_order=pairwise["predictor"].map(order))
    out = out.sort_values(["target_order", "predictor_order"])
    write_values(out, args.out, STEM)
    write_values(stand_in.sort_values("mean_corr_to_others", ascending=False),
                 args.out, f"{STEM}_generic_stand_in")
    write_caption(caption(pairwise, sources, stand_in), args.out, STEM)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
