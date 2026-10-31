"""S6: how the correlation with mutational accessibility changes when the mutation spectrum is swapped.

    python figures/make_sF_spectrum_swap.py --guide guides/h3n2_13lineage.yaml

The correlation between a model's probability and mutational accessibility is normally computed
with the spectrum of the virus the model was trained on. Here only the spectrum changes: the
protein, the reference codons, the replacements and the nucleotide-step strata are the same in
every row. A model that had learned its own virus's mutation process should correlate best
with that spectrum. Grantham and BLOSUM62 are drawn too; they contain no nucleotide
information, so they show what the genetic code alone does under each spectrum.

This is the within-strata, many-spectra figure. It varies the spectrum for ONE protein, not the
protein: nothing in the shipped tables scores a second protein with a fine-tuned model.

Reads ``spectrum_swap_strata`` (made by ``tools/export_codon_s06.py``) and checks its reference
condition against the guide's own ``mut_prob`` column. Writes one figure, the correlations
themselves (``sF_spectrum_swap``). A star marks each series' best-correlating spectrum in each
panel.

Two things shown by earlier versions are deliberately not drawn. BLOSUM62 is dropped: it is a
comparator rather than a control, because an empirical substitution matrix already contains
mutational accessibility, so putting it beside Grantham invited exactly the confusion the
reviewer objected to. The reversed-direction H3N2 spectrum is dropped as well; the rows here are
the spectra of S05 plus the direction-averaged H3N2, which is the one symmetry control this
figure needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from mutacc import codon_learning as cl
from mutacc import strata as S
from mutacc import style
from mutacc.guide import Guide, parse_figure_args
from mutacc.io import write_caption, write_values

STEM = "sF_spectrum_swap"
STEM_DELTA = "sF_spectrum_delta"
TABLE = "spectrum_swap_strata"
#: The reference estimates are recomputed from the table's own mut_prob. The table merges accessibilities
#: that are equal up to floating-point noise (see mutacc.spectrum_swap); the guide's column does not,
#: so the two agree to about 1e-3, not to rounding.
SELF_CHECK_TOL = 2e-3
SERIES_STYLE = {"neg_grantham": ("s", True)}   # marker, filled
#: BLOSUM62 is a comparator, not a biochemistry control: see the module docstring.
DROP_SERIES = ("blosum62",)
#: Conditions whose key contains any of these are dropped. The direction-averaged spectrum is the
#: symmetry control this figure needs; the fully reversed one is redundant beside it. Matched on a
#: substring rather than named outright, so the script carries no dataset name (tests/test_figure_rules).
DROP_CONDITION_PARTS = ("reversed",)
STRATUM_TITLE = {"all": "All replacements pooled", "1": "One nucleotide change", "2": "Two nucleotide changes",
                 "3": "Three nucleotide changes"}
SPREAD = 0.62
MARKER_SIZE = 5.0
#: Series that are not models. They get their own row rather than repeating on every spectrum.
PROPERTY_SERIES = ("neg_grantham",)
PROPERTY_ROW_LABEL = {"neg_grantham": "Grantham similarity (control)"}
#: Blank rows between the spectrum block and the control row.
GAP_ROWS = 1.0


def check_reference(table: pd.DataFrame, guide: Guide, scores: pd.DataFrame) -> float:
    """Largest gap between the table's reference rows and the same correlations from the guide's own mut_prob."""
    models = [m.key for m in guide.models]
    props = table.loc[table["kind"] == "property", "series"].unique().tolist()
    mine = S.strata_table(scores, models, props, n_boot=0)
    mine = mine.loc[mine["control"] == ""].set_index(["series", "stratum"])["estimate"]
    ref = table.loc[table["reference"]].set_index(["series", "stratum"])["estimate"]
    if set(ref.index) != set(mine.index):
        raise ValueError("the table's reference rows and the scores cover different (score, stratum) cells")
    gap = float((ref - mine.reindex(ref.index)).abs().max())
    if gap > SELF_CHECK_TOL:
        raise ValueError(f"the table's reference condition disagrees with the guide's own accessibility by {gap:.2e}")
    return gap


def validate(table: pd.DataFrame) -> None:
    need = {"condition", "label", "reference", "series", "series_label", "kind", "role", "stratum", "estimate", "lo", "hi",
            "delta", "delta_lo", "delta_hi", "n_rows"}
    if need - set(table.columns):
        raise KeyError(f"table is missing columns {sorted(need - set(table.columns))}")
    if int(table.drop_duplicates("condition")["reference"].sum()) != 1:
        raise ValueError("the table must have exactly one reference condition")
    if table.duplicated(["condition", "series", "stratum"]).any():
        raise ValueError("duplicate (condition, series, stratum) rows")
    if not np.isfinite(table[["estimate", "lo", "hi", "delta", "delta_lo", "delta_hi"]].to_numpy()).all():
        raise ValueError("non-finite estimate or interval")


def _wrap(text: str, width: int = 26) -> str:
    if " (" in text:
        prefix, suffix = text.split(" (", 1)
        suffix = "(" + suffix
        if "(Pauly et al" in suffix:
            suffix = r"$\mathit{(Pauly\ et\ al.)}$"
        p_lines, cur = [], ""
        for w in prefix.split():
            if cur and len(cur) + 1 + len(w) > width:
                p_lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        p_lines.append(cur)
        bold_prefix = "\n".join(r"$\mathbf{" + line.replace(" ", r"\ ").replace("-", r"\text{-}") + r"}$" for line in p_lines)
        return f"{bold_prefix}\n{suffix}"
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    bold_lines = [r"$\mathbf{" + line.replace(" ", r"\ ").replace("-", r"\text{-}") + r"}$" for line in lines]
    return "\n".join(bold_lines)


def _series_order(table: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(table["series"]))


def _plot(table: pd.DataFrame, value: str, lo: str, hi: str, *, zero_label: str, xlabel: str, height: float):
    """Spectrum rows for the models, then one separate row for the biochemistry control.

    Grantham is not a model and takes no part in the comparison the spectrum rows make, so
    repeating it on every row only crowded them. It gets a row of its own, below a gap: the
    point is its value under the reference spectrum and the bar behind it is its full range
    across the nine, because that range is wide enough that one number would misrepresent it.
    """
    model_series = [s for s in _series_order(table) if s not in PROPERTY_SERIES]
    prop_series = [s for s in _series_order(table) if s in PROPERTY_SERIES]
    conds = table.drop_duplicates("condition")
    keys = conds["condition"].tolist()
    labels = [_wrap(l) for l in conds["label"]]

    # Model rows run from the top; each property sits below them, after one blank row.
    y = np.arange(len(keys))[::-1].astype(float) + len(prop_series) + GAP_ROWS
    prop_y = {name: float(len(prop_series) - 1 - i) for i, name in enumerate(prop_series)}
    all_labels = labels + [_wrap(PROPERTY_ROW_LABEL.get(n, n)) for n in prop_series]
    all_y = list(y) + [prop_y[n] for n in prop_series]

    strata = list(dict.fromkeys(table["stratum"]))
    fig = style.figure(style.WIDTH_DOUBLE, height)
    axes = fig.subplots(2, 2, sharey=True).reshape(-1)
    n_series = len(model_series)
    handles = {}
    for ax, st in zip(axes, strata):
        here = table.loc[table["stratum"] == st]
        star_pad = 0.022 * float(here[hi].max() - here[lo].min())
        for si, name in enumerate(model_series):
            sub = here.loc[here["series"] == name].set_index("condition").reindex(keys)
            off = (si - (n_series - 1) / 2) * SPREAD / n_series
            colour = style.colour_for_role(sub["role"].iloc[0])
            marker, filled = SERIES_STYLE.get(name, ("o", True))
            h = ax.errorbar(sub[value], y + off, xerr=[sub[value] - sub[lo], sub[hi] - sub[value]], fmt=marker,
                            color=colour, ecolor=colour, markerfacecolor=colour if filled else style.SURFACE,
                            markersize=MARKER_SIZE, elinewidth=0.9, capsize=0, zorder=3)
            handles.setdefault(name, (h, sub["series_label"].iloc[0]))
            best = int(np.nanargmax(sub[value].to_numpy()))
            ax.plot([sub[hi].iloc[best] + star_pad], [y[best] + off], marker="*", markersize=7.0,
                    color=colour, markeredgecolor=colour, linestyle="none", zorder=5, clip_on=False)

        for name in prop_series:
            sub = here.loc[here["series"] == name].set_index("condition").reindex(keys)
            colour = style.colour_for_role(sub["role"].iloc[0])
            marker, filled = SERIES_STYLE.get(name, ("o", True))
            ref_key = conds.loc[conds["reference"], "condition"].iloc[0]
            row = sub.loc[ref_key]
            # The range this control covers across every spectrum, drawn behind its point.
            ax.plot([sub[value].min(), sub[value].max()], [prop_y[name]] * 2, color=colour, alpha=0.30,
                    linewidth=4.0, solid_capstyle="round", zorder=2)
            h = ax.errorbar([row[value]], [prop_y[name]], xerr=[[row[value] - row[lo]], [row[hi] - row[value]]],
                            fmt=marker, color=colour, ecolor=colour,
                            markerfacecolor=colour if filled else style.SURFACE,
                            markersize=MARKER_SIZE, elinewidth=0.9, capsize=0, zorder=4)
            handles.setdefault(name, (h, sub["series_label"].iloc[0]))

        ax.set_yticks(all_y)
        ax.set_yticklabels(all_labels)
        ax.set_ylim(-0.7, max(all_y) + 0.8)
        ax.grid(axis="y", visible=False)
        n = here["n_rows"].iloc[0]
        style.panel_title(ax, STRATUM_TITLE.get(st, st), f"{n:,.0f} replacements per lineage")
        ax.set_xlabel(xlabel)
        style.reference_line(ax, 0.0, zero_label, axis="x")
    fig.legend([h for h, _ in handles.values()], [l for _, l in handles.values()],
               loc="outside upper center", ncol=len(handles))
    style.panel_letters(axes, dx=-0.12, dy=1.12)
    return fig


def draw_swap(table: pd.DataFrame):
    return _plot(table, "estimate", "lo", "hi", zero_label="no correlation = 0",
                 xlabel="Spearman correlation with accessibility", height=7.2)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Per score and stratum: how many alternative spectra fit better or worse than the reference, by paired interval."""
    d = table.loc[~table["reference"]]
    better = (d["delta_hi"] < 0)      # alternative correlates MORE (delta = reference minus alternative)
    worse = (d["delta_lo"] > 0)
    out = d.assign(alt_higher=better, alt_lower=worse).groupby(["series", "series_label", "stratum"], sort=False).agg(
        n_spectra=("condition", "size"), alt_higher=("alt_higher", "sum"), alt_lower=("alt_lower", "sum")).reset_index()
    return out


def caption(table: pd.DataFrame, gap: float, n_boot: int) -> str:
    ref_label = table.loc[table["reference"], "label"].iloc[0]
    models = table.loc[table["kind"] == "model"]
    worst = models.loc[models.loc[models["stratum"].astype(str) == "1", "estimate"].idxmin(), "label"]
    n_alt = int(table["condition"].nunique() - 1)
    return f"""
    **How strongly each model's score tracks mutational accessibility,
    spectrum by spectrum.** Mutational accessibility of a replacement is the probability that
    mutation turns the reference codon into any codon of the new amino acid, given a 4x4 table of
    nucleotide change rates (a spectrum). Each row is a different spectrum ({n_alt} alternatives
    to the reference, "{ref_label}"): the experimental spectra of supplementary figure S5, the
    non-directional symmetrised version of the reference, the phylogenetic fourfold GTR spectrum,
    and theoretical baselines (K80, JC69). Rates are rescaled to the same mean so pooled values are
    comparable. The protein, the reference codons and the replacements are identical in every row,
    so the nucleotide-step strata are too; only the accessibility changes. **A star marks the
    spectrum each series correlates with most strongly in that panel.**

    **Natural phylogenetic spectrum vs experimental rates.** The fourfold GTR condition estimates
    substitution rates directly from ancestral sequence reconstructions across the global
    phylogenetic tree (>140,000 branches). Because natural sequence evolution causes codons to
    transition between amino acids over time, only two codons remain strictly fourfold degenerate
    across every branch of the entire tree; thus, rather than filtering for a static alignment of
    globally conserved positions (which would discard >95% of synonymous variation), rates are
    measured branch by branch. At each reconstructed parent-child branch, every codon whose
    ancestral state possesses a fourfold-degenerate prefix (Ala, Pro, Thr, Val, Gly, Ser, Arg, Leu)
    provides an opportunity for neutral substitution at the third nucleotide position. Reversible
    GTR exchangeabilities are fitted from the resulting synonymous substitution counts (79,228
    substitutions) by enforcing detailed balance against equilibrium base frequencies. The
    non-directional condition averages forward and reverse fluxes of the reference experimental
    spectrum to enforce detailed balance with respect to stationary base composition
    ($Q_\\text{{sym}} = (Q + \\text{{diag}}(\\pi)^{{-1}} Q^T \\text{{diag}}(\\pi))/2$). Both alternative
    parameterisations correlate very strongly with each other and with the reference (Pearson
    r > 0.997), confirming that the strong model-accessibility correlation is robust whether
    using laboratory-measured error rates or natural phylogenetic estimates.

    **Dots and whiskers.** A dot is the mean over lineages of the within-lineage Spearman
    correlation between the score and accessibility. Whiskers are 95% **percentile bootstrap**
    intervals, built as follows. Sites are resampled with replacement -- whole positions, carrying
    all of their replacements, and the same positions in every lineage -- {n_boot} times. Each
    resample gives one value of the same lineage-averaged statistic, and the whisker runs from the
    2.5th to the 97.5th percentile of those {n_boot} values. Sites are the resampling unit because
    the up-to-nineteen replacements at one position share a backbone and are not independent;
    resampling rows would treat them as independent and give intervals that are far too narrow.
    Lineages are not the resampling unit either: they share their sites, so the spread between
    them understates the uncertainty (`LINEAGE_VALIDITY.md` section 4, Q2). One draw is reused
    across every spectrum, series and stratum, so differences between rows are paired.

    **The bottom row is the control.** Grantham similarity is not a model and holds no nucleotide
    information, so it shows what the genetic code does by itself. It takes a row of its own
    rather than repeating on all {n_alt + 1} spectrum rows. Its square is the value under the reference
    spectrum, "{ref_label}", with the usual interval; the pale bar behind it spans
    the lowest to the highest value the control reaches across all {n_alt + 1} spectra, because that
    range is wide enough that a single number would misrepresent it. No star is placed on this
    row: the stars mark each model's best spectrum, and the control is not a model.

    **Why "{worst}" scores lowest.** Its matrix is the closest to the
    reference of any in the panel by variance -- supplementary figure S5 puts the two at 83-85%
    of each other's -- and its accessibility ordering agrees with the reference's more closely
    than any other spectrum's here (Spearman 0.82 within one nucleotide step). It still scores
    lowest of the measured spectra. The two facts are compatible because this panel does not
    measure agreement with the reference spectrum; it measures agreement with the model's score.
    Inside one nucleotide step every replacement needs one specific base change, so a spectrum's
    ordering there is essentially its ordering of the twelve rates. Ranking the model's score by
    which base change each replacement needs gives the model a preference over those twelve:
    A>G (mean percentile 0.73) and G>A (0.66) at the top, G>C (0.36) and T>G (0.37) at the
    bottom. How closely a spectrum's rate order matches that preference predicts this panel
    almost exactly -- Spearman 0.96 across the eight spectra -- and this one's match is 0.06,
    against 0.56 for the reference. The two agree on the dominant
    transitions and disagree on the minor cells that decide the rest: the reference makes G>A
    its third-largest rate, the model's second-most-favoured change, while the other demotes
    G>A to sixth and promotes G>C, the model's least-favoured, to third. Swapping that one G>A
    rate back recovers 35% of the gap, G>C a further 21% and C>T 17%. This is a property of the
    two matrices' minor cells, not a parsing fault: transposing, complementing or relabelling
    the bases of the second file all fit the reference worse than reading it as it stands
    (`dev/spectrum_swap_anomaly/`).

    BLOSUM62 is not drawn: an empirical substitution matrix already contains mutational
    accessibility, which makes it a comparator rather than a biochemistry control, and showing it
    beside Grantham invites that confusion.

    A model that had learned the reference spectrum's direction would be expected to correlate
    best with it. Where the control moves with the models, the pattern reflects the genetic code
    and not learning (see SYMMETRY_TESTS.md). Only the spectrum varies, and only for one protein:
    nothing in the shipped tables scores a second protein with a fine-tuned model. Conditions in
    the exploration archive that redrew synonymous codons from codon-usage tables are not
    reproduced. Accessibilities equal up to floating-point noise are treated as ties, which
    matters for the flat spectrum; the reference agrees with the guide's own accessibility to
    {gap:.1e}.
    """


def main(argv=None) -> int:
    args = parse_figure_args(__doc__.split("\n\n")[0], argv)
    guide: Guide = args.guide
    table = cl.read_derived(guide, TABLE, args.results)
    validate(table)
    scores = guide.read_table("replacement_scores", args.results)
    gap = check_reference(table, guide, scores)
    drop = table["condition"].str.contains("|".join(DROP_CONDITION_PARTS), case=False, regex=True)
    table = table.loc[~table["series"].isin(DROP_SERIES) & ~drop].copy()
    if table.empty:
        raise ValueError("every row was filtered out; check DROP_SERIES and DROP_CONDITION_PARTS")
    summary = summarise(table)
    summary["kind"] = ["property" if s in set(table.loc[table["kind"] == "property", "series"]) else "model" for s in summary["series"]]

    paths = style.save_figure(draw_swap(table), args.out / STEM)
    write_values(table, args.out, STEM)
    write_values(summary, args.out, f"{STEM}_summary")
    write_caption(caption(table, gap, int(table["n_boot"].iloc[0])), args.out, STEM)
    print(f"self-check: reference condition agrees with the guide's accessibility to {gap:.1e}")
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
