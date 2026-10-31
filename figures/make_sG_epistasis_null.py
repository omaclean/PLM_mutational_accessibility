"""S7: does the model's response to one substitution depend on another, or is that just what any substitution does?

    python figures/make_sG_epistasis_null.py --guide guides/h3n2_13lineage.yaml

The model scores every possible substitution at every position. Put substitution A on the
sequence first and the score of substitution B moves. The question is whether the moves seen
between the nine defining substitutions of one clade are larger than the move the same
substitution B makes when it is measured on a background that has nothing to do with the clade:
every one of the thousands of single substitutions at other sites. That pool is the null.

The shift is measured on one scale only: the zero-sum gauge of Zhang and Ovchinnikov (2024),
log probability minus its average over the 20 residues. The shipped tables still carry the
three other gauges the archive exported, but nothing here reads them -- ``select_gauge`` drops
them on the way in. The reason for the choice is in the caption and does not rest on which
gauge gives the most significant pairs: a plain change in probability has no room to move where
the model is already sure, and the zero-sum gauge is the one the contact analysis uses too.

Reads ``epistasis_cells`` (one row per gauge and directed pair, with the archive's placement
against the null), ``epistasis_null_draws`` (every null draw, per gauge and focal substitution)
and ``epistasis_focal_sites`` (mature numbering, HA1/HA2 position, baseline probabilities).

Checks it makes on every run:

* every pooled p-value is recomputed from the null draws and must match the shipped table (a
  p-value may move by one tied draw, never more);
* every Benjamini-Hochberg q-value is recomputed from the p-values;
* the site numbering is internally consistent: HA1 positions equal sequence positions, HA2
  positions count from the HA2 start, and no label is off by the 16-residue signal peptide.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from mutacc import style
from mutacc.guide import Guide, parse_figure_args
from mutacc.io import write_caption, write_values

STEM = "sG_epistasis_null"
Q_THRESHOLD = 0.05
SIGNAL_PEPTIDE = 16          # full-length numbering = mature numbering + 16; never used for labels
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
TIE_TOLERANCE = 2.5          # a p-value may differ from the shipped one by this many null draws

#: The only scale measured on. ``SCALE`` finishes "shift in the model score, on the ...".
GAUGE = "zerosum"
SCALE = "zero-sum scale"
DEFINITION = "log probability minus its average over the 20 residues"

CELL_COLUMNS = {
    "background", "focal", "bg_pos", "focal_pos", "delta", "null_n", "null_median", "null_sd",
    "p_pooled_up", "p_pooled_dn", "p_pooled_two", "p_stud_two", "q_bh_pooled", "q_bh_stud",
}
SITE_COLUMNS = {"focal", "index_label", "seq_pos", "wt", "alt", "region", "region_pos", "p_wt0", "p_alt0"}


# ---------------------------------------------------------------------------
# Inputs and their checks
# ---------------------------------------------------------------------------

def select_gauge(frame: pd.DataFrame, what: str) -> pd.DataFrame:
    """Keep the zero-sum rows and drop the gauge column.

    Done once, on the way in, so no function below takes a gauge argument and no panel can end
    up drawn on a different scale from its neighbour.
    """
    if "gauge" not in frame.columns:
        return frame
    out = frame.loc[frame["gauge"] == GAUGE]
    if out.empty:
        raise ValueError(f"epistasis {what} table has no {GAUGE!r} rows; it carries "
                         f"{sorted(set(frame['gauge']))}")
    return out.drop(columns=["gauge"]).reset_index(drop=True)


def validate_inputs(cells: pd.DataFrame, draws: pd.DataFrame, sites: pd.DataFrame) -> None:
    """Reject tables that would draw a wrong or blank figure. Raises ValueError naming the problem.

    Takes the tables already reduced to one gauge, so there is nothing here about gauges agreeing.
    """
    for name, frame, need in (("cells", cells, CELL_COLUMNS), ("sites", sites, SITE_COLUMNS),
                              ("draws", draws, {"focal", "delta"})):
        missing = need - set(frame.columns)
        if missing:
            raise ValueError(f"epistasis {name} table lacks columns {sorted(missing)}")
    if cells["delta"].isna().any() or draws["delta"].isna().any():
        raise ValueError("NaN in a shift or null draw")
    if not np.isfinite(cells["delta"]).all() or not np.isfinite(draws["delta"]).all():
        raise ValueError("non-finite shift or null draw")
    if cells.duplicated(["background", "focal"]).any():
        raise ValueError("duplicate (background, focal) rows in the cell table")
    if (cells["background"] == cells["focal"]).any():
        raise ValueError("a substitution is listed as its own background")
    stray = (set(cells["focal"]) | set(cells["background"])) - set(sites["focal"])
    if stray:
        raise ValueError(f"cells name substitutions missing from the site table: {sorted(stray)}")
    if sites["focal"].duplicated().any():
        raise ValueError("duplicate focal substitutions in the site table")
    sizes = draws.groupby("focal").size()
    if sizes.nunique() != 1:
        raise ValueError("null draws differ in size between focal substitutions")
    if set(draws["focal"]) != set(cells["focal"]):
        raise ValueError("null draws and cells cover different focal substitutions")
    expected = cells["null_n"].unique()
    if len(expected) != 1 or int(expected[0]) != int(sizes.iloc[0]):
        raise ValueError(f"null_n {list(expected)} does not match the {int(sizes.iloc[0])} draws shipped per focal substitution")
    check_numbering(sites)


def check_numbering(sites: pd.DataFrame) -> None:
    """Labels, positions and regions of the focal substitutions must agree with each other.

    The backbone is numbered from the mature protein (position 1 = first residue after the
    signal peptide). HA1 residues keep that number; HA2 residues count from the HA2 start. A
    label shifted by the 16-residue signal peptide would put a residue in the wrong place, so
    every label is rebuilt from its position and compared.
    """
    regions = set(sites["region"])
    if not regions <= {"HA1", "HA2"}:
        raise ValueError(f"unknown region(s) {sorted(regions - {'HA1', 'HA2'})}")
    ha1, ha2 = sites[sites["region"] == "HA1"], sites[sites["region"] == "HA2"]
    if (ha1["region_pos"] != ha1["seq_pos"]).any():
        raise ValueError("an HA1 site's region position differs from its sequence position")
    if len(ha2):
        starts = (ha2["seq_pos"] - ha2["region_pos"] + 1).unique()
        if len(starts) != 1:
            raise ValueError(f"HA2 sites imply different HA2 start positions {sorted(starts)}")
        if ha1["seq_pos"].max() >= starts[0]:
            raise ValueError("an HA1 site lies at or after the HA2 start")
        if (ha2["region_pos"] < 1).any():
            raise ValueError("an HA2 position below 1")
    for r in sites.itertuples():
        body = f"{r.wt}{int(r.region_pos)}{r.alt}"
        want = body if r.region == "HA1" else f"HA2:{body}"
        if r.focal != want:
            raise ValueError(f"label {r.focal!r} does not match region {r.region}, position {r.region_pos}, "
                             f"{r.wt}->{r.alt} (expected {want!r})")
        if r.index_label != f"{r.wt}{int(r.seq_pos)}{r.alt}":
            raise ValueError(f"index label {r.index_label!r} does not match sequence position {r.seq_pos}")
        if not (0.0 <= r.p_wt0 <= 1.0 and 0.0 <= r.p_alt0 <= 1.0):
            raise ValueError(f"{r.focal}: baseline probability outside [0, 1]")
        if r.wt == r.alt:
            raise ValueError(f"{r.focal}: substitution changes nothing")
        if int(r.seq_pos) in (int(r.region_pos) + SIGNAL_PEPTIDE,) and r.region == "HA1":
            raise ValueError(f"{r.focal}: position looks shifted by the {SIGNAL_PEPTIDE}-residue signal peptide")


# ---------------------------------------------------------------------------
# Statistics (ported from the archive and checked against it)
# ---------------------------------------------------------------------------

def empirical_p(null: np.ndarray, observed: float) -> tuple[float, float, float]:
    """Upper, lower and two-sided empirical p with the +1 correction (Phipson and Smyth)."""
    n = null.size
    up = (1 + int((null >= observed).sum())) / (n + 1)
    dn = (1 + int((null <= observed).sum())) / (n + 1)
    return up, dn, min(1.0, 2 * min(up, dn))


def benjamini_hochberg(p) -> np.ndarray:
    """BH q-values; NaN stays NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    vals = p[ok]
    order = np.argsort(vals)
    n = vals.size
    adj = np.minimum.accumulate((vals[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def null_by_focal(draws: pd.DataFrame) -> dict[str, np.ndarray]:
    return {f: g["delta"].to_numpy() for f, g in draws.groupby("focal", sort=False)}


def recompute_placement(cells: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Pooled p-values and BH q-values from the null draws, beside the shipped ones."""
    nulls = null_by_focal(draws)
    out = cells.copy()
    ps = np.array([empirical_p(nulls[r.focal], r.delta) for r in out.itertuples()])
    out["p_up_re"], out["p_dn_re"], out["p_two_re"] = ps[:, 0], ps[:, 1], ps[:, 2]
    out["q_pooled_re"] = benjamini_hochberg(out["p_two_re"].to_numpy())
    out["q_stud_re"] = benjamini_hochberg(out["p_stud_two"].to_numpy())
    return out.reset_index(drop=True)


def check_placement(recomputed: pd.DataFrame) -> dict[str, float]:
    """Raise if the shipped p/q values are not what the draws give. Returns the worst gaps."""
    shipped = ["p_pooled_up", "p_pooled_dn", "p_pooled_two", "p_stud_two", "q_bh_pooled", "q_bh_stud"]
    if recomputed[shipped].isna().any().any():
        raise ValueError("NaN in a shipped p- or q-value; a missing value would pass every comparison silently")
    if not recomputed[shipped].apply(lambda c: c.between(0, 1)).all().all():
        raise ValueError("a shipped p- or q-value lies outside [0, 1]")
    n = recomputed["null_n"].iloc[0]
    tol_p = TIE_TOLERANCE / (n + 1)
    gap_p = float(max((recomputed["p_up_re"] - recomputed["p_pooled_up"]).abs().max(),
                      (recomputed["p_dn_re"] - recomputed["p_pooled_dn"]).abs().max(),
                      (recomputed["p_two_re"] - recomputed["p_pooled_two"]).abs().max()))
    if gap_p > tol_p:
        raise ValueError(f"shipped pooled p-values disagree with the null draws by {gap_p:.2e} (tolerance {tol_p:.2e})")
    gap_stud = float((recomputed["q_stud_re"] - recomputed["q_bh_stud"]).abs().max())
    if gap_stud > 1e-9:
        raise ValueError(f"shipped studentised q-values are not the BH adjustment of their p-values (off by {gap_stud:.2e})")
    # q depends on a handful of p-values that may each move by one tied draw; allow that much
    gap_q = float((recomputed["q_pooled_re"] - recomputed["q_bh_pooled"]).abs().max())
    if gap_q > tol_p * 2:
        raise ValueError(f"shipped pooled q-values are not the BH adjustment of the recomputed p-values (off by {gap_q:.2e})")
    return {"p": gap_p, "q_pooled": gap_q, "q_stud": gap_stud}


def check_null_summaries(cells: pd.DataFrame, draws: pd.DataFrame, tol: float = 1e-9) -> float:
    """Median and SD of each null, recomputed from its draws, against the shipped columns."""
    worst = 0.0
    nulls = null_by_focal(draws)
    for r in cells.drop_duplicates("focal").itertuples():
        v = nulls[r.focal]
        worst = max(worst, abs(np.median(v) - r.null_median), abs(v.std(ddof=1) - r.null_sd))
    if worst > tol:
        raise ValueError(f"shipped null median/SD disagree with the draws by {worst:.2e}")
    return worst


def how_many_stand_out(cells: pd.DataFrame) -> dict[str, int]:
    """How many of the pairs stand out, under each of the two tests."""
    return {"n_cells": len(cells),
            "n_q_pooled": int((cells["q_bh_pooled"] < Q_THRESHOLD).sum()),
            "n_q_studentised": int((cells["q_bh_stud"] < Q_THRESHOLD).sum()),
            "n_unadjusted_p": int((cells["p_pooled_two"] < Q_THRESHOLD).sum())}


def null_bands(draws: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    nulls = null_by_focal(draws)
    rows = []
    for f in order:
        q = np.percentile(nulls[f], PERCENTILES)
        rows.append({"gauge": GAUGE, "focal": f, "n_draws": nulls[f].size,
                     **{f"p{p}": v for p, v in zip(PERCENTILES, q)}})
    return pd.DataFrame(rows)


def matrix_of(cells: pd.DataFrame, order: list[str], value: str = "delta") -> pd.DataFrame:
    """Rows = the substitution being scored, columns = the substitution already present (the preprint's orientation)."""
    m = pd.DataFrame(np.nan, index=order, columns=order)
    for r in cells.itertuples():
        m.loc[r.focal, r.background] = getattr(r, value)
    return m


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(cells: pd.DataFrame, draws: pd.DataFrame, sites: pd.DataFrame):
    order = list(sites["focal"])
    sub = cells
    scale, definition = SCALE, DEFINITION
    strong = sub["q_bh_pooled"] < Q_THRESHOLD
    n_cells = len(sub)

    # Panel C held the bottom row with the old cross-gauge panel beside it; with one gauge it
    # has the row to itself, which is what it wanted -- nine nulls side by side need the width.
    fig = style.figure(style.WIDTH_DOUBLE, 6.3)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86], width_ratios=[1.0, 1.0])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # --- A: the matrix --------------------------------------------------------------------
    mat = matrix_of(cells, order)
    qs = matrix_of(cells, order, "q_bh_pooled")
    vmax = float(np.nanmax(np.abs(mat.values)))
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = axA.imshow(mat.values, cmap=style.DIVERGING, norm=norm, aspect="auto")
    axA.grid(False)
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels(order, rotation=90)
    axA.set_yticks(range(len(order)))
    axA.set_yticklabels(order)
    axA.set_xlabel("Substitution already on the sequence")
    axA.set_ylabel("Substitution being scored")
    for spine in axA.spines.values():
        spine.set_visible(False)
    for i in range(len(order)):
        for j in range(len(order)):
            q = qs.values[i, j]
            if np.isfinite(q) and q < Q_THRESHOLD:
                axA.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=style.INK, linewidth=1.3))
                v = mat.values[i, j]
                axA.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.5,
                         color="white" if abs(v) > 0.55 * vmax else style.INK)
    cb = fig.colorbar(im, cax=axA.inset_axes([1.04, 0.05, 0.035, 0.9]))
    cb.set_label(f"Shift in model score ({scale})")
    cb.outline.set_visible(False)
    style.panel_title(axA, "Shift in the model score, every pair",
                      f"Outlined: stands out from the null (BH q < {Q_THRESHOLD:g}); number shown")

    # --- B: pooled null with the observed shifts ------------------------------------------
    nulls = null_by_focal(draws)
    pooled = np.concatenate([nulls[f] for f in order])
    lo, hi = np.percentile(pooled, [0.02, 99.98])
    span = max(abs(lo), abs(hi), float(sub["delta"].abs().max())) * 1.12
    edges = np.linspace(-span, span, 161)
    counts, _ = np.histogram(pooled, bins=edges)
    axB.stairs(np.where(counts > 0, counts, np.nan), edges, fill=True, color=style.NULL_GREY, alpha=0.7, linewidth=0)
    axB.set_yscale("log")
    top = counts.max()
    axB.set_ylim(0.6, top * 2000)
    axB.set_xlim(-span, span)
    rug_lo, rug_top = top * 2.5, top * 9
    for r in sub.sort_values("delta").itertuples():
        s = r.q_bh_pooled < Q_THRESHOLD
        axB.vlines(r.delta, rug_lo, rug_top, colors=style.INK if s else style.INK_MUTED,
                   linewidth=1.5 if s else 0.7, alpha=1.0 if s else 0.7, zorder=3)
    # label the pairs that stand out, staggered in height so labels cannot collide
    lab = sub[strong].sort_values("delta")
    heights = np.geomspace(top * 20, top * 300, max(len(lab), 2))[:len(lab)] if len(lab) else []
    for (_, r), h in zip(lab.iterrows(), heights):
        axB.annotate(f"{r.background} then {r.focal}\n{r.delta:+.2f}", xy=(r.delta, rug_top), xytext=(r.delta, h),
                     ha="left" if r.delta < 0 else "right", va="bottom", fontsize=5.5, color=style.INK,
                     arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": style.INK_MUTED})
    style.reference_line(axB, float(np.median(pooled)), "null median", axis="x")
    axB.set_xlabel(f"Shift in model score ({scale})")
    axB.set_ylabel("Null draws per bin (log scale)")
    axB.plot([], [], color=style.INK, linewidth=1.5, label=f"Clade pair, q < {Q_THRESHOLD:g}")
    axB.plot([], [], color=style.INK_MUTED, linewidth=0.7, label="Clade pair, not distinct")
    axB.legend(loc="center left", fontsize=6, bbox_to_anchor=(0.0, 0.42))
    style.panel_title(axB, f"The {n_cells} pairs against {len(pooled):,} null draws",
                      "Null: the same nine substitutions on backgrounds at other sites")

    # --- C: each focal substitution against its own null ----------------------------------
    x = np.arange(len(order))
    for i, f in enumerate(order):
        q = np.percentile(nulls[f], PERCENTILES)
        axC.fill_between([i - 0.36, i + 0.36], q[0], q[6], color=style.NULL_GREY, alpha=0.25, linewidth=0)
        axC.fill_between([i - 0.36, i + 0.36], q[1], q[5], color=style.NULL_GREY, alpha=0.40, linewidth=0)
        axC.fill_between([i - 0.36, i + 0.36], q[2], q[4], color=style.NULL_GREY, alpha=0.65, linewidth=0)
        axC.plot([i - 0.36, i + 0.36], [q[3], q[3]], color=style.INK, linewidth=0.9)
    for i, f in enumerate(order):
        cell = sub[sub["focal"] == f].reset_index(drop=True)
        offs = np.linspace(-0.27, 0.27, max(len(cell), 2))[:len(cell)]
        for off, r in zip(offs, cell.itertuples()):
            s = r.q_bh_pooled < Q_THRESHOLD
            axC.scatter(i + off, r.delta, s=22 if s else 10, zorder=4,
                        facecolor=style.INK if s else style.SURFACE,
                        edgecolor=style.INK if s else style.INK_MUTED, linewidth=0.8)
            if s:
                axC.annotate(r.background, (i + off, r.delta), xytext=(4, 0), textcoords="offset points",
                             fontsize=5.5, va="center", color=style.INK)
    axC.axhline(0, color=style.INK_MUTED, linewidth=0.5, zorder=1)
    axC.set_xticks(x)
    axC.set_xticklabels(order, rotation=45, ha="right")
    axC.grid(axis="x", visible=False)
    axC.set_xlabel("Substitution being scored")
    axC.set_ylabel(f"Shift in model score ({scale})")
    style.panel_title(axC, "Each substitution against its own null",
                      "Bands: 50, 90 and 98% of its null draws; line: null median; dots: its pairs in the clade")

    style.panel_letters([axA, axB, axC], dx=-0.14, dy=1.12)
    return fig


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------

def conserved_site_note(sites: pd.DataFrame) -> str:
    """The room-to-move argument, from the data: the most conserved focal site's baseline probability.

    This is why the gauge is the zero-sum one. It used to quote the width of that site's null on
    the probability scale as well, which meant reading a gauge the figure does not draw; the
    baseline probability alone makes the point, and makes it from the site table.
    """
    top = sites.sort_values("p_wt0", ascending=False).iloc[0]
    return (f"at the most conserved site, {top.focal}, the model gives the residue already there a baseline "
            f"probability of {top.p_wt0:.3f}, leaving a plain change in probability almost nowhere to go")


def caption(cells: pd.DataFrame, draws: pd.DataFrame, sites: pd.DataFrame) -> str:
    scale, definition = SCALE, DEFINITION
    sub = cells
    counts = how_many_stand_out(cells)
    n_cells, n_focal = len(sub), sites["focal"].nunique()
    n_pool_each = int(sub["null_n"].iloc[0])
    n_pool = n_pool_each * n_focal
    strong = sub[sub["q_bh_pooled"] < Q_THRESHOLD].sort_values("q_bh_pooled")
    top = sub.loc[sub["delta"].abs().idxmax()]
    if len(strong):
        standing = "; ".join(f"{r.background} then {r.focal} ({r.delta:+.2f}, q = {r.q_bh_pooled:.3f})"
                             for r in strong.itertuples())
        stands = (f"{len(strong)} of the {n_cells} pairs have a Benjamini-Hochberg q below {Q_THRESHOLD:g} "
                  f"against the pooled null: {standing}.")
    else:
        stands = f"None of the {n_cells} pairs has a Benjamini-Hochberg q below {Q_THRESHOLD:g} against the pooled null."
    return f"""
    **Are the shifts between the clade-defining substitutions larger than
    what any substitution does when its background changes?**
    The model gives a probability for every residue at every position. Place one substitution on
    the sequence first and the score of a second substitution moves. Here the {n_focal} substitutions
    that define the clade are taken in every ordered pair ({n_cells} pairs). To say how large a shift
    is, each substitution being scored is also measured on every background that carries one change at
    a site outside the clade set, {n_pool_each:,} backgrounds per substitution ({n_pool:,} draws in all).
    Those changes are not clade-defining, so they show what a background change does to that
    substitution by chance.

    The shift is drawn on the {scale}: {definition}. This is the scale of the categorical Jacobian of
    Zhang and Ovchinnikov (2024), and the only one measured on here. The choice does not rest on which
    scale gives the most significant pairs. It rests on room to move: a plain change in probability is
    limited where the model is already sure, and {conserved_site_note(sites)}.

    Panel A: the shift for every pair, rows the substitution scored and columns the one already on
    the sequence; outlined cells have q below {Q_THRESHOLD:g}. Panel B: all {n_cells} pairs (lines) against
    the pooled null (grey), log counts. Panel C: each substitution against its own null; grey bands hold 50,
    90 and 98% of that null. Positions are mature
    numbering: HA1 residues keep their position, HA2 residues count from the HA2 start; add {SIGNAL_PEPTIDE}
    for numbering that includes the signal peptide.

    {stands} The largest shift is {top.background} then {top.focal} ({top.delta:+.2f}).
    Of the {n_cells} pairs, {counts['n_q_pooled']} have q below {Q_THRESHOLD:g} against all null draws and
    {counts['n_q_studentised']} against draws close to the pair in sequence separation and 3D distance. A p-value is the share of
    null draws at least as extreme (with a +1 correction), doubled for two sides; q adjusts for the {n_cells}
    pairs tested. Clade substitutions can sit on one branch of the tree, so the pairs are not independent
    evidence, and the "close" test uses structure information that is not shipped.
    """


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    return parse_figure_args(__doc__.split("\n\n")[0], argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    guide: Guide = args.guide
    # One gauge, dropped in at the door: everything below this line is single-scale.
    cells = select_gauge(guide.read_table("epistasis_cells", args.results), "cells")
    draws = select_gauge(guide.read_table("epistasis_null_draws", args.results), "null draws")
    sites = guide.read_table("epistasis_focal_sites", args.results)
    validate_inputs(cells, draws, sites)

    recomputed = recompute_placement(cells, draws)
    gaps = check_placement(recomputed)
    check_null_summaries(cells, draws)

    fig = draw(cells, draws, sites)
    stem = STEM
    paths = style.save_figure(fig, args.out / stem)

    order = list(sites["focal"])
    write_values(recomputed.drop(columns=["q_stud_re"]).assign(gauge=GAUGE), args.out, stem)
    write_values(null_bands(draws, order), args.out, f"{stem}_null_bands")
    write_caption(caption(cells, draws, sites), args.out, stem)
    print(f"self-check: p-values reproduce from the draws (worst gap {gaps['p']:.1e}, one draw is {1 / (int(cells['null_n'].iloc[0]) + 1):.1e})")
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
