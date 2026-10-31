"""Figure SJ (stem ``sJ``): do two scans of the same protein agree, as the strains drift apart?

    python figures/make_sJ_dms_concordance.py --guide guides/h3n2_13lineage.yaml

Each point is a pair of deep mutational scanning (DMS) datasets of one protein measured on two
strains. One question is asked: do the two scans rank mutations alike. The two panels place the
pair by how far apart the two strains are -- as a count of amino-acid substitutions (A) and as a
percentage of the protein (B).

Reads the guide's ``dms_cross_lineage_pairs`` table. The agreement itself is computed offline from
the raw scan tables (they live outside this repository); what is recomputed here is what the table
lets us check: the percentage divergence from the substitution count and protein length and the
range of every coefficient.

Colour carries the trait category, drawn from the house categorical palette (see README, "Colour
plan"). That palette holds only warm and violet hues, so nothing here can be read as the blue/green
model pair, which means base and fine-tuned throughout the rest of the set -- and no model appears in
this figure at all. Marker shape carries the viral species, and nothing else: one channel, two
shapes. Marker faces are semi-transparent so overlapping points stay readable, with the edge drawn
at full strength so a single point still reads cleanly. The assay (the ``system`` column) is no
longer drawn on the panels; it stays in the values table beside every point.

Nothing on a panel carries a vertical extent. The floor-censoring sensitivity range -- the same
coefficient recomputed without the variants piled at the assay detection floor -- used to be drawn
as a bracket on the two pairs that qualify, but an I-bar is the visual grammar of a confidence
interval, and a reader takes it as one: as sampling uncertainty on two points and, by omission,
as precision on the other twenty-four. It is a sensitivity range, not an interval, so it is
recorded per pair in the values table (``y_no_floor``, ``censored``) and stated in the caption
instead of drawn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from mutacc import style
from mutacc.guide import Guide, REPO_ROOT, load_guide
from mutacc.io import write_caption, write_values
from mutacc.placement import Candidate, place_existing

STEM = "sJ_dms_concordance"

#: (agreement column, its floor-censoring sensitivity column, the question it answers, y label).
#: The question is stated in the caption, not as a panel title: both panels answer it.
ROWS = [
    ("spearman_rho", "spearman_rho_no_floor", "Do the two scans rank mutations alike?",
     "Rank agreement (Spearman ρ)"),
]
#: (x column, x label, tick values, tick suffix).
COLUMNS = [
    ("aa_distance", "Amino-acid substitutions between the two strains", (1, 2, 5, 10, 20, 50, 100, 200, 500), ""),
    ("pct_divergence", "Divergence between the two strains (% of the protein)", (0.5, 1, 2, 5, 10, 20, 50), "%"),
]

REQUIRED = ["virus", "system", "sample_1", "sample_2", "metric_category", "aa_distance", "focal_protein_length",
            "pct_divergence", "n_variants", "n_sites", "spearman_rho", "frac_at_floor",
            "spearman_rho_no_floor", "indel_count", "join_key", "filter_mode"]

SPECIES_MARKERS = ("^", "o")   # one per viral species, by sorted name: Influenza A, SARS-CoV-2
FACE_ALPHA = 0.55              # marker interiors, so a stack of points reads as a stack
#: A pair with more than this fraction of variants sitting at the assay floor in both scans has its
#: floor-censoring sensitivity recorded: the floor pile-up inflates the correlation, and dropping
#: those variants over-corrects. Recorded in the values table and the caption; not drawn.
FLOOR_BRACKET = 0.05
JITTER = 0.07          # points at one x are offset by up to this fraction so they do not overprint
SEED = 42              # the archive's jitter seed
DIVERGENCE_TOL = 1e-6
MARKER_SIZE = 5.2


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def validate(table: pd.DataFrame) -> None:
    """Fail on a table that would draw a misleading panel."""
    if table.empty:
        raise ValueError("dms_cross_lineage_pairs is empty")
    missing = [c for c in REQUIRED if c not in table.columns]
    if missing:
        raise ValueError(f"dms_cross_lineage_pairs lacks column(s) {missing}")
    for col in ("aa_distance", "pct_divergence", "spearman_rho"):
        if not np.isfinite(table[col].to_numpy(dtype=float)).all():
            raise ValueError(f"missing or non-finite values in {col!r}")
    if (table["aa_distance"] <= 0).any() or (table["pct_divergence"] <= 0).any():
        raise ValueError("distances must be positive to sit on a log axis")
    expected = table["aa_distance"] / table["focal_protein_length"] * 100.0
    worst = float((expected - table["pct_divergence"]).abs().max())
    if worst > DIVERGENCE_TOL:
        raise ValueError(f"pct_divergence is not aa_distance / focal_protein_length (worst gap {worst:.2e})")
    for col in ("spearman_rho", "spearman_rho_no_floor"):
        vals = table[col].dropna()
        if ((vals < -1) | (vals > 1)).any():
            raise ValueError(f"{col} outside [-1, 1]")
    n_species = species_of(table["virus"]).nunique()
    if n_species > len(SPECIES_MARKERS):
        raise ValueError(f"{n_species} viral species but only {len(SPECIES_MARKERS)} marker shapes")


def species_of(virus) -> pd.Series:
    """Viral species from the ``virus`` label: everything before a parenthesised qualifier.

    ``Influenza A (H3N2)`` and ``Influenza A (cross-subtype)`` are one species and so share a shape;
    the subtype distinction lives in the ``system`` column of the values table, not on the panel.
    """
    return pd.Series(virus).astype(str).str.split("(").str[0].str.strip()


def species_markers(table: pd.DataFrame) -> dict[str, str]:
    """Species -> marker, by sorted name, so filtering rows never reshapes the ones that remain."""
    return dict(zip(sorted(species_of(table["virus"]).unique()), SPECIES_MARKERS))


def trait_colours(table: pd.DataFrame) -> dict[str, str]:
    """Trait category -> colour, by sorted name, on the same rule as the fills."""
    return style.colours_for_categories(table["metric_category"])


# ---------------------------------------------------------------------------
# Points per panel
# ---------------------------------------------------------------------------

def panel_points(table: pd.DataFrame, y_col: str, bracket_col: str | None, x_col: str, *, seed: int = SEED) -> pd.DataFrame:
    """The rows one panel draws: true x, jittered x, y, and the floor-censoring sensitivity value
    where one applies. Only ``y`` is drawn; ``y_no_floor`` and ``censored`` go to the values table."""
    sub = table[table[y_col].notna()].copy()
    if sub.empty:
        raise ValueError(f"no rows with a value for {y_col!r}")
    sub["species"] = species_of(sub["virus"]).to_numpy()
    rng = np.random.default_rng(seed)
    sub["x"] = sub[x_col].to_numpy(float)
    sub["x_plotted"] = sub["x"] * rng.uniform(1 - JITTER, 1 + JITTER, size=len(sub))
    sub["y"] = sub[y_col].to_numpy(float)
    if bracket_col is None:
        sub["y_no_floor"] = np.nan
        sub["censored"] = False
    else:
        sub["censored"] = (sub["frac_at_floor"] > FLOOR_BRACKET) & sub[bracket_col].notna()
        sub["y_no_floor"] = np.where(sub["censored"], sub[bracket_col], np.nan)
    keep = ["virus", "species", "system", "metric_category", "sample_1", "sample_2", "filter_mode",
            "x", "x_plotted", "y", "y_no_floor", "censored"]
    return sub[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def marker_kwargs(colour: str) -> dict:
    """Marker keywords for a trait colour: a translucent face inside a full-strength edge.

    The edge keeps a lone point crisp; the face lets a stack of points show through itself.
    """
    return {"markerfacecolor": to_rgba(colour, FACE_ALPHA), "markeredgecolor": colour,
            "markeredgewidth": 0.9}


def _draw_points(ax, pts: pd.DataFrame, markers: dict, colours: dict) -> None:
    """One marker per pair, and nothing else: no panel mark has a vertical extent."""
    for r in pts.itertuples():
        ax.plot([r.x_plotted], [r.y], linestyle="none", marker=markers[r.species], markersize=MARKER_SIZE,
                zorder=4, **marker_kwargs(colours[r.metric_category]))


def _pixels(ax, x, y) -> np.ndarray:
    return ax.transData.transform(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))


def _x_limits(table: pd.DataFrame, x_col: str) -> tuple[float, float]:
    return float(table[x_col].min()) / 1.7, float(table[x_col].max()) * 1.7


def draw(table: pd.DataFrame):
    validate(table)
    markers = species_markers(table)
    colours = trait_colours(table)
    n_rows, n_cols = len(ROWS), len(COLUMNS)
    fig = style.figure(style.WIDTH_DOUBLE, 2.6 * n_rows + 0.9)
    axes = fig.subplots(n_rows, n_cols, sharex="col", squeeze=False)

    drawn = []
    for i, (y_col, bracket_col, _question, ylabel) in enumerate(ROWS):
        for j, (x_col, xlabel, ticks, suffix) in enumerate(COLUMNS):
            ax = axes[i][j]
            pts = panel_points(table, y_col, bracket_col, x_col)
            ax.set_xscale("log")
            ax.minorticks_off()
            lo, hi = _x_limits(table, x_col)
            ax.set_xlim(lo, hi)
            shown = [t for t in ticks if lo <= t <= hi]
            ax.set_xticks(shown)
            ax.set_xticklabels([f"{t:g}{suffix}" for t in shown])
            _draw_points(ax, pts, markers, colours)

            lows = float(pts["y"].min())
            span = 1.0 - lows
            ax.set_ylim(lows - 0.08 * span, 1.0 + 0.16 * span)
            # The two panels answer the same question of the same points, so the question is not
            # repeated as a panel title: the shared y label states it and the caption asks it.
            style.finalise_axes(ax, ylabel=ylabel if j == 0 else "", xlabel=xlabel if i == n_rows - 1 else "")
            drawn.append((ax, pts, y_col))

    style.panel_letters([ax for ax, _, _ in drawn])
    handles = [Line2D([], [], linestyle="none", marker=m, markersize=MARKER_SIZE, label=name,
                      **marker_kwargs(style.INK)) for name, m in markers.items()]
    handles += [Line2D([], [], linestyle="none", marker="o", markersize=MARKER_SIZE, label=name,
                       **marker_kwargs(colour)) for name, colour in colours.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, handlelength=1.4, columnspacing=1.4,
               fontsize=6.5)
    fig.canvas.draw()

    scores = []
    for ax, pts, y_col in drawn:
        obstacles = np.repeat(_pixels(ax, pts["x_plotted"], np.r_[pts["y"]]), 40, axis=0)
        placed: list = []
        style.reference_line(ax, 1.0, "perfect agreement = 1")
        label = next(t for t in ax.texts if t.get_text() == "perfect agreement = 1")
        coords = ("axes fraction", "data")
        cands = [Candidate((x, 1.0), (dx, 2), ha, "bottom", coords) for x, dx, ha in ((1.0, -2, "right"), (0.0, 2, "left"))]
        scores.append(place_existing(ax, label, cands, obstacles=obstacles, placed=placed))
    fig.label_scores = scores
    return fig


# ---------------------------------------------------------------------------
# Values and caption
# ---------------------------------------------------------------------------

def values_table(table: pd.DataFrame) -> pd.DataFrame:
    frames = []
    letter = iter("ABCDEFGHIJKLMNOP")
    for y_col, bracket_col, _question, _ in ROWS:
        for x_col, *_ in COLUMNS:
            pts = panel_points(table, y_col, bracket_col, x_col)
            pts.insert(0, "panel", next(letter))
            pts.insert(1, "measure", y_col)
            pts.insert(2, "x_axis", x_col)
            frames.append(pts)
    return pd.concat(frames, ignore_index=True)


def caption(table: pd.DataFrame) -> str:
    n = len(table)
    systems = list(dict.fromkeys(table["system"]))
    per_system = "; ".join(
        f"{s} ({int((table['system'] == s).sum())} pair{'s' if (table['system'] == s).sum() != 1 else ''})"
        for s in systems
    )
    censored = table[(table["frac_at_floor"] > FLOOR_BRACKET)]
    n_modes = table["filter_mode"].nunique()
    binding = any("binding" in c.lower() for c in table["metric_category"].unique())
    structural = sorted({s for s, j in zip(table["system"], table["join_key"]) if "struct" in j.lower()})
    indel_rows = table[table["indel_count"] > 0]
    lo_rho, hi_rho = table["spearman_rho"].min(), table["spearman_rho"].max()
    structural_text = (
        f" {', '.join(structural)} pairs are joined by a structural rather than a sequence alignment and, per the "
        "analysis notes, cross cell line as well as subtype, so they are not like-for-like with the other systems."
        if structural else ""
    )
    floor_text = (
        f" For {len(censored)} pairs, more than {FLOOR_BRACKET:.0%} of variants sit at the assay floor in both scans "
        f"(rank agreement {', '.join(f'{v:.2f}' for v in sorted(censored['spearman_rho']))} as plotted, falling to "
        f"{', '.join(f'{v:.2f}' for v in sorted(censored['spearman_rho_no_floor']))} with those variants dropped). "
        "The pile-up inflates the plotted value and dropping the floor variants removes the most deleterious "
        "mutations and over-corrects, so the true value lies between. That range is not drawn: it is a sensitivity "
        "to a filtering choice, not sampling uncertainty, and an interval on two points would be read as both. It is "
        "recorded per pair in the values table."
        if len(censored) else ""
    )
    binding_text = (
        " Binding traits measure affinity for a receptor, not fitness, so their agreement is not a fitness agreement."
        if binding else ""
    )
    return f"""
    **Agreement between deep mutational scans of one protein, against how far apart
    the scanned strains are.** Deep mutational scanning (DMS) measures the effect of nearly every
    single amino-acid change in a protein. Each point is a pair of DMS datasets of the same protein
    measured on two strains ({n} pairs: {per_system}). Both panels show the Spearman rank
    correlation of the measured effects of the mutations the two scans share. (A) against the number
    of amino-acid substitutions between the two parental strains; (B) against that number as a
    percentage of the protein (log scales; the same points, re-spaced). Marker shape is the viral
    species and colour is the measured trait, as in the legend; marker faces are semi-transparent so
    that overlapping points remain readable. The assay each pair comes from (the systems listed
    above) is not drawn on the panels; it is recorded for every point in the accompanying values
    table. No model is plotted here, and the colours are drawn from the house categorical palette
    rather than the blue and green that mean base and fine-tuned elsewhere in this set.
    Points at the same x are offset by up to {JITTER:.0%} so they do not overprint. Rank agreement
    ranges from {lo_rho:.2f} to {hi_rho:.2f}.

    What is not shown.{floor_text}{structural_text}{binding_text} Rank agreement is the only measure
    plotted. The source table also holds the Pearson correlation of the same effects and the mean
    per-site Jensen-Shannon divergence between the two scans' amino-acid preference profiles; both
    tell the same story as the rank correlation and are left out to keep the figure to one question.
    The
    synonymous-variant filter is not the same across systems ({n_modes} filter modes, recorded per pair in the
    source table), which moves the coefficients slightly. Substitution counts exclude insertions and deletions, which are recorded
    separately ({len(indel_rows)} pairs have at least one). The agreement of each pair is computed
    offline from the raw scan tables; the figure shows no per-site or regional breakdown, and no mark
    on either panel carries a vertical extent -- there are no uncertainty intervals anywhere in it.
    """


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--guide", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output")
    args = parser.parse_args(argv)
    args.guide = load_guide(args.guide)
    if args.results is None:
        args.results = args.guide.results_dir
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    table = args.guide.read_table("dms_cross_lineage_pairs", args.results)
    fig = draw(table)
    paths = style.save_figure(fig, args.out / STEM)
    write_values(values_table(table), args.out, STEM)
    write_caption(caption(table), args.out, STEM)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
