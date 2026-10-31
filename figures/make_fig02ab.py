"""Figure 2, panels A and B: model probability against mutational accessibility.

    python figures/make_fig02ab.py --guide guides/h3n2_5lineage.yaml

One point per replacement, coloured by the nucleotide changes it needs, for the base model (A)
and the fine-tuned model (B). Drawn from the guide's ``replacement_scores`` table on a single
lineage, the guide's ``primary_lineage``, as panel D is.

**Why a redraw.** The frozen panels these replace (``data/frozen_panels/fig2a_*``, ``fig2b_*``)
printed Spearman 0.274 and 0.652, which is the value over all five lineages pooled
(``tests/test_make_fig02ab.py`` reproduces both). Pooling stacks five near-copies of the same
10,754 replacements, because the lineages are scored at the same sites, so the clouds were
five-fold overplotted and the bulk of the mass could not be seen. One lineage is one copy of each
replacement; the points are small and translucent, so dense regions darken and an isolated
replacement is still visible.

The Spearman and Pearson correlations over all of a panel's replacements are written to the
values table, and the composite prints them as each panel's subheader. Pearson is taken on the
probabilities themselves, as the frozen panels did, so a handful of large values dominate it; it
is shown for continuity, and Spearman is the statistic the rest of the figure uses.

Two ways of saying what the colours mean, chosen with ``--labels``: ``legend`` (default) is a
"Mut_num" legend inside the plot's top left; ``text`` has no legend and writes the meaning in the
group's own colour beside each cloud, with larger points and a little more headroom. The panels
are drawn in the shared box ``style.PANEL_AXES`` and saved uncropped, so they line up with panels
C and D.

Writes PDF and PNG for each panel. They are panels, not figures: the composite
(``guides/composites/fig02.yaml``) places them and draws their headers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mutacc import style
from mutacc.guide import REPO_ROOT, parse_figure_args
from mutacc.io import write_caption, write_values
from mutacc.strata import primary_lineage

TABLE = "replacement_scores"
STEM_A = "fig02a_scatter_panel"
STEM_B = "fig02b_scatter_panel"
STEMS = {"base": STEM_A, "finetune": STEM_B}
VALUES_STEM = "fig02ab_scatter_panels"
LETTERS = {"base": "A", "finetune": "B"}
GROUPS = (1, 2, 3)

#: Drawn 3, 2, 1 so the one-change group, the one that matters most, sits on top.
DRAW_ORDER = (3, 2, 1)
LEGEND_TITLE = "Mut_num"
TICK_BASE = 100          # a tick every two decades on the x axis

#: The two label styles. Point size and opacity are chosen together: bigger points need a lower
#: opacity for the cloud to keep the same density, but not so low that a lone point disappears.
#: The limits are fixed so panels A and B share axes exactly, and cut close to the data.
VARIANTS = {
    "legend": dict(size=8.0, alpha=0.13, xlim=(1.5e-9, 3.0), ylim=(10 ** -15.8, 10 ** -2.9)),
    "text": dict(size=14.0, alpha=0.20, xlim=(1.5e-9, 3.0), ylim=(10 ** -16.9, 10 ** -2.5)),
}
#: The ``legend`` key sits inside the point cloud's top left, so it is set small and the cloud
#: is drawn at a lower opacity than the ``text`` variant: it labels three colours already shown,
#: and anything bigger competes with the data it sits on.
LEGEND_PT = 7.0
LEGEND_MARKER_PT = 5.0
TEXT_PT = 10.5
#: Where the coloured text sits in the ``text`` variant: group -> (log10 y, wording).
TEXT_LABELS = {1: (-2.85, "1 nucleotide change"), 2: (-6.35, "2 nucleotide changes"),
               3: (-16.3, "3 nucleotide changes")}
TEXT_X = 1.8e-9


def parse(argv=None):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--labels", choices=sorted(VARIANTS), default="legend")
    known, rest = pre.parse_known_args(argv)
    args = parse_figure_args(__doc__.split("\n\n")[0], rest)
    args.labels = known.labels
    return args


def group_colours() -> dict[int, str]:
    """Colour of each nucleotide-change group: the first three Dark2 colours, as in the panels
    this replaces (green, orange, purple). Taken from the colormap, not typed, so the figure
    scripts still carry no colour literals."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    cmap = colormaps["Dark2"]
    return {k: to_hex(cmap(i)) for i, k in enumerate(GROUPS)}


def log_columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log10 PLM probability, log10 mutation probability and group, refusing zeros and NaN.

    A zero cannot be placed on a log axis and silently dropping it would change the points the
    panel claims to show, so it is an error, not a filter.
    """
    plm = frame["plm_prob"].to_numpy(float)
    mut = frame["mut_prob"].to_numpy(float)
    if not (np.isfinite(plm).all() and np.isfinite(mut).all() and (plm > 0).all() and (mut > 0).all()):
        raise ValueError("plm_prob and mut_prob must be finite and positive to plot on log axes")
    group = frame["nt_steps"].to_numpy(int)
    if not set(group) <= set(GROUPS):
        raise ValueError(f"nt_steps must be one of {GROUPS}, got {sorted(set(group))}")
    return np.log10(plm), np.log10(mut), group


def correlations(frame: pd.DataFrame) -> dict[str, float]:
    """Spearman and Pearson correlation of PLM probability with mutation probability."""
    plm = frame["plm_prob"].to_numpy(float)
    mut = frame["mut_prob"].to_numpy(float)
    return dict(spearman_r=float(spearmanr(plm, mut)[0]), pearson_r=float(np.corrcoef(plm, mut)[0, 1]))


def subheader(stats: dict[str, float]) -> str:
    """The text the composite prints under a panel's title."""
    return f"Spearman r = {stats['spearman_r']:.3f}; Pearson r = {stats['pearson_r']:.3f}"


def draw(frame: pd.DataFrame, labels: str = "legend"):
    """One panel for one model's rows (a single lineage), in the shared panel box."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator

    if labels not in VARIANTS:
        raise ValueError(f"labels must be one of {sorted(VARIANTS)}, got {labels!r}")
    cfg = VARIANTS[labels]
    lx, ly, group = log_columns(frame)
    colours = group_colours()
    fig = style.figure(*style.PANEL_SIZE)
    ax = style.panel_axes(fig)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for k in DRAW_ORDER:
        sel = group == k
        ax.scatter(10 ** lx[sel], 10 ** ly[sel], s=cfg["size"], color=colours[k], alpha=cfg["alpha"],
                   linewidths=0, rasterized=True, zorder=2)
    ax.set_xlim(*cfg["xlim"])
    ax.set_ylim(*cfg["ylim"])
    ax.xaxis.set_major_locator(LogLocator(base=TICK_BASE, numticks=20))
    ax.minorticks_off()
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=style.PANEL_TICK_PT)
    if labels == "legend":
        handles = [Line2D([], [], linestyle="none", marker="o", markersize=LEGEND_MARKER_PT,
                          color=colours[k], label=str(k))
                   for k in GROUPS]
        ax.legend(handles=handles, title=LEGEND_TITLE, loc="upper left", frameon=False, fontsize=LEGEND_PT,
                  title_fontsize=LEGEND_PT, alignment="left", borderaxespad=0.2, borderpad=0.2,
                  labelspacing=0.3, handletextpad=0.2)
    else:
        for k, (log_y, words) in TEXT_LABELS.items():
            ax.text(TEXT_X, 10 ** log_y, words, color=colours[k], fontsize=TEXT_PT, fontweight="bold",
                    va="center", ha="left")
    style.finalise_axes(ax, xlabel="PLM probability", ylabel="Mutation probability")
    style.place_panel_labels(ax)
    return fig


def summary_rows(frame: pd.DataFrame, *, panel: str, model: str, lineage: str) -> list[dict]:
    """What the panel shows, in numbers: the two correlations and each group's size."""
    _, _, group = log_columns(frame)
    rows = [dict(panel=panel, series=model, group="all", measure=name, value=value, n=len(frame),
                 lineage=lineage) for name, value in correlations(frame).items()]
    for k in GROUPS:
        n = int((group == k).sum())
        rows.append(dict(panel=panel, series=model, group=str(k), measure="n_replacements",
                         value=float(n), n=n, lineage=lineage))
    return rows


def root_description(guide, lineage: str) -> dict:
    """Which root sequence the panels are drawn from, read from the guide and the file itself.

    Nothing about the root is typed here: the path and provenance class come from the guide's
    lineage entry and the header from the FASTA, so the caption cannot drift from what was scored.
    """
    entry = next(lin for lin in guide.lineages if lin.id == lineage)
    header = Path(entry.root).read_text().splitlines()[0].lstrip(">").strip()
    return dict(lineage=lineage, path=Path(entry.root).relative_to(REPO_ROOT).as_posix(),
                root_type=entry.root_type, header=header)


_ROOT_WORDS = {
    "reconstructed": ("an inferred ancestral sequence, not an observed one. Its FASTA header carries "
                      "a sequence accession, but the sequence itself is inferred, so the accession "
                      "should not be read as its source"),
    "observed": "an observed sequence",
}


def caption(summary: pd.DataFrame, root: dict, guide) -> str:
    """Caption text for panels A and B, every number read from ``summary``."""
    def get(panel, measure, group="all"):
        row = summary[(summary["panel"] == panel) & (summary["measure"] == measure)
                      & (summary["group"] == group)]
        return float(row["value"].iloc[0])

    sizes = [int(get("A", "n_replacements", str(k))) for k in GROUPS]
    label = {m.role: m.label for m in guide.models}
    kind = _ROOT_WORDS.get(root["root_type"], f"a root whose provenance is recorded as {root['root_type']}")
    return f"""
    **Model probability against mutational accessibility, base model and fine-tuned model.**
    Each dot is one amino-acid replacement at one site of the {guide.virus} reference root
    sequence ({sum(sizes):,} replacements), plotted at the model's probability for it
    (horizontal) and its mutational accessibility (vertical), both on log axes. Colour is
    `Mut_num`, the fewest nucleotide changes that make the replacement ({sizes[0]:,},
    {sizes[1]:,} and {sizes[2]:,} replacements need 1, 2 and 3). (A) {label['base']}. (B) The same
    replacements scored by the model: {label['finetune'].lower()}.

    One lineage only: the root sequence `{root['lineage']}` (`{root['path']}`), which is
    {kind}. Its header reads `{root['header']}`. The other lineages of the set are scored at the
    same sites, so they would add near-copies of these points, not independent ones, and are not
    drawn.

    Under each panel title: the Spearman rank correlation of the model's probability with
    mutational accessibility over all replacements ({get('A', 'spearman_r'):.3f} base,
    {get('B', 'spearman_r'):.3f} fine-tuned), and the Pearson correlation of the two probabilities
    ({get('A', 'pearson_r'):.3f} and {get('B', 'pearson_r'):.3f}). Pearson is taken on the
    probabilities themselves, not their logarithms, so a few large values dominate it and it says
    little about the bulk of the points. Mutation probability takes only a few hundred distinct
    values within a group, which is why the points fall in horizontal stripes.
    """


def main(argv=None) -> int:
    args = parse(argv)
    guide = args.guide
    frame = guide.read_table(TABLE, args.results)
    guide.check_lineages(frame["lineage"].unique(), f"{guide.name} {TABLE}")
    chosen = primary_lineage(frame, [(lin.id, lin.root_type) for lin in guide.lineages],
                             guide.primary_lineage)
    one = frame[frame["lineage"].astype(str) == chosen]
    models = {m.role: m.key for m in guide.models}
    rows, written = [], []
    for role in ("base", "finetune"):
        rows_m = one[one["model"] == models[role]]
        if rows_m.empty:
            raise ValueError(f"{guide.name}: no {TABLE} rows for model {models[role]} in lineage {chosen}")
        fig = draw(rows_m, args.labels)
        written += [p.name for p in style.save_panel(fig, args.panels_out / STEMS[role], suffixes=(".pdf", ".png"))]
        rows += summary_rows(rows_m, panel=LETTERS[role], model=models[role], lineage=chosen)
    summary = pd.DataFrame(rows)
    write_values(summary, args.out, VALUES_STEM)
    write_caption(caption(summary, root_description(guide, chosen), guide), args.out, VALUES_STEM,
                  panel="A, B")
    print("wrote", ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
