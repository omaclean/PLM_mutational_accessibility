"""Figure 4: does weighting in accessibility bring a model closer to measured fitness?

    python figures/make_fig04.py --guide guides/h3n2_13lineage.yaml
    python figures/make_fig04.py --guide guides/h3n2_13lineage.yaml

Deep mutational scanning (DMS) measures the effect of nearly every single amino-acid
replacement in the lab, so unlike observed sequence frequencies it does not depend on how
easily a replacement arises. For each scan phenotype this script rank-correlates the measured
values with ``log(model probability) + alpha * log(accessibility)`` over a grid of alpha, for
the base and the fine-tuned model of that scan's family. The model is scored on the scan's own
reference sequence; that is done offline (it needs a GPU) and its output ships in
``dms_scores``. Everything downstream is recomputed here.

Reads the guide's ``dms_scores``, ``dms_scores_masked``, ``dms_peak_summary`` and
``dms_alpha_curves`` tables (plus ``dms_spectrum_*``). Panels, their labels and their order come
from ``dms_peak_summary``; model identity and colour come from the table's role column.

The model probability is the **masked marginal**: the scored site is replaced by the mask token
and predicted from its context. ``mutacc/masked_dms.py`` says why, and why the choice is not a
convention -- the wild-type (unmasked) estimator this figure used until 19 Oct 2026 is not
neutral between a base and a fine-tuned model, and it penalised the fine-tune for having learned
the family. ``--scoring unmasked`` redraws the older version; Figure SO shows both.

One run writes two figures, because the panels are split by virus: the influenza panels are
Figure 4 and the SARS-CoV-2 panels a supplementary figure of their own (S18). The sweep still
runs over every arm in the archive's order, so the split does not move the published
bootstrap band.

Two guards carry over from the source analysis and are tested. Both run on the **unmasked** arm,
which is recomputed on every run for exactly this purpose, because the archive's curves and
flags were themselves computed unmasked and are the only fixed point available:

* the recomputed unmasked curves must reproduce the archive's own curves to 1e-6, or this raises;
* an optimum is marked only where the archive flags the curve interpretable (best weight not at
  the end of the range, and some signal at alpha = 0). The flag is recomputed here and must agree
  with the shipped summary.

The masked arm that is drawn has no archive to check against -- it is new -- so the gate above is
what rules out a tokeniser, a checkpoint or an off-by-one masquerading as the effect of masking.
The flags the drawn figure uses are recomputed from the masked curves themselves; a curve can
legitimately gain or lose an interior optimum under the other estimator, and the values table
carries both that flag and the archive's for comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from mutacc import style
from mutacc.dms_sweep import (
    WEAK_SIGNAL_RHO, accessibility_table, curve_flags, log_accessibility, site_bootstrap_band,
    sweep_curve,
)
from mutacc.guide import Guide, REPO_ROOT, load_guide
from mutacc.io import write_caption, write_values
from mutacc.masked_dms import add_scoring_argument, apply_masked
from mutacc.placement import Candidate, curve_obstacles, place_existing, place_label

STEM = "fig04_alpha_vs_dms"
STEM_SC2 = "sL_dms_sc2"

#: Panels are drawn one virus at a time. Influenza is the paper's subject and keeps the main
#: figure; the SARS-CoV-2 panels are a cross-virus check and go to their own supplementary figure.
MAIN_VIRUS = "IAV"
#: virus tag -> stem. The nested key is a leftover from the symmetrised-spectrum variant
#: that was dropped on 14 Oct 2026; one entry now, kept so adding another is a one-liner.
VIRUS_STEMS = {
    "IAV": {"published": STEM},
    "SC2": {"published": STEM_SC2},
}
ALPHA_LIMIT = 1.0
ALPHA_STEP = 0.01
N_BOOT = 200
BAND_EVERY = 5
#: The archive's seed. The bootstrap generator is consumed arm by arm in the order of
#: ``dms_peak_summary``, which is what reproduces the published band exactly.
BAND_SEED = 20260923
#: An arm with fewer scored replacements than this is a broken join, not a small scan.
MIN_ROWS = 200
SELF_CHECK_TOL = 1e-6
N_COLUMNS = 3
ZERO_LABEL = "no relationship = 0"
#: Trait substring marking a receptor-binding panel; these are grouped at the end of the figure.
RECEPTOR_TRAIT = "binding"

#: Where, along a curve, a direct label may sit if neither end is free.
LABEL_ALPHAS = (0.6, -0.6, 0.35, -0.35, 0.85, -0.85, 0.2, -0.2, 0.1, -0.1)
#: Points between a curve and its label, tried in turn.
LABEL_GAPS = (5, 10, 18, 28)
#: How a label is anchored at a spot, tried in turn. Right of the point first: the best weight is
#: always close to alpha = 0, so a centred label there straddles the dashed alpha = 0 line, and
#: clearing it rightwards keeps the label beside its own dot instead of exiling it to a panel edge.
H_ANCHORS = (("left", 3), ("center", 0), ("right", -3))
#: Where a label goes when its own optimum is too close to the alpha = 0 line to sit beside it.
#: Just clear of the line on the RIGHT, reading rightwards. Every best weight here is at or just
#: below zero, so falling back leftwards would drop the base model's label onto the fine-tuned
#: curve, where a reader has only the colour to tell which line it belongs to.
CLEAR_OF_ZERO_ALPHA = 0.06

REQUIRED_SCORES = ["dataset", "model_name", "spectrum_matrix", "site", "ref_codon", "aa", "log_plm", "log_acc"]
REQUIRED_SUMMARY = ["dataset", "organism", "trait", "panel_label", "citation", "model_role", "model_name",
                    "model_label", "spectrum", "n_variants", "interpretable", "virus_tag"]

ROLE_NAMES = {"base": "Base model", "finetune": "Virus-specific fine-tuned model"}
SPECTRUM_TABLE_PREFIX = "dms_spectrum_"


class SelfCheckError(ValueError):
    """The recomputed sweep does not reproduce the archive's own curves."""


def alpha_grid() -> np.ndarray:
    return np.round(np.arange(-ALPHA_LIMIT, ALPHA_LIMIT + ALPHA_STEP / 2, ALPHA_STEP), 6)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def validate_inputs(scores: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Fail loudly on a table that would draw a blank or wrong panel."""
    for name, frame, cols in (("dms_scores", scores, REQUIRED_SCORES), ("dms_peak_summary", summary, REQUIRED_SUMMARY)):
        if frame.empty:
            raise ValueError(f"{name} is empty")
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} lacks column(s) {missing}")
    for col in ("log_plm", "log_acc"):
        bad = int((~np.isfinite(scores[col].to_numpy(dtype=float))).sum())
        if bad:
            raise ValueError(f"{bad} missing or non-finite values in {col!r}; a NaN would silently drop replacements")
    unknown_roles = set(summary["model_role"]) - set(style.ROLE_COLOURS)
    if unknown_roles:
        raise ValueError(f"dms_peak_summary has unknown model role(s) {sorted(unknown_roles)}")


def published_arms(summary: pd.DataFrame, spectrum: str = "published") -> pd.DataFrame:
    """One row per (panel, model), in the archive's order, for one spectrum variant."""
    arms = summary[summary["spectrum"] == spectrum].reset_index(drop=True)
    if arms.empty:
        raise ValueError(f"dms_peak_summary has no rows for the {spectrum!r} spectrum")
    if arms.duplicated(["dataset", "trait", "model_name"]).any():
        raise ValueError("dms_peak_summary has duplicate (dataset, trait, model) rows")
    return arms


def load_spectra(guide: Guide, results: Path, scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    spectra = {}
    for name in scores["spectrum_matrix"].unique():
        spectra[name] = guide.read_table(SPECTRUM_TABLE_PREFIX + name, results).set_index("from_base")
    return spectra


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def compute_curves(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    n_boot: int = N_BOOT,
    seed: int = BAND_SEED,
    band_every: int = BAND_EVERY,
    log_acc: np.ndarray | None = None,
    spectrum: str = "published",
    max_arms: int | None = None,
) -> pd.DataFrame:
    """Long table: one row per (panel, model, alpha) with rho, its site-bootstrap band and flags.

    ``log_acc`` replaces the shipped accessibility. The symmetrised-spectrum variant that used
    this (published as S17) was dropped on 14 Oct 2026: it moved no curve by more than 0.013 in
    rho, so it was a second figure saying the same thing. Symmetric-versus-directional is
    reported in S4 instead. The hook is kept because it costs nothing and the comparison is
    still reproducible from ``mutacc.dms_sweep.accessibility_table(reversible=True)``.
    ``max_arms`` stops early, for tests; the bootstrap generator is consumed in arm order, so
    the first arms are identical whether or not later ones are run.
    """
    validate_inputs(scores, summary)
    arms = published_arms(summary, spectrum)
    if max_arms is not None:
        arms = arms.iloc[:max_arms]
    alphas = alpha_grid()
    rng = np.random.default_rng(seed)
    acc_column = scores["log_acc"].to_numpy(float) if log_acc is None else np.asarray(log_acc, float)
    rows = []
    for panel, arm in enumerate(arms.itertuples()):
        mask = ((scores["dataset"] == arm.dataset) & (scores["model_name"] == arm.model_name)).to_numpy()
        if arm.trait not in scores.columns:
            raise ValueError(f"dms_scores has no column for trait {arm.trait!r}")
        y_all = scores[arm.trait].to_numpy(float)
        keep = mask & np.isfinite(y_all)
        if keep.sum() < MIN_ROWS:
            raise ValueError(
                f"{arm.dataset}/{arm.model_name}/{arm.trait}: only {int(keep.sum())} scored replacements "
                f"(need {MIN_ROWS}); the join between scores and phenotype is empty or broken"
            )
        log_plm = scores["log_plm"].to_numpy(float)[keep]
        la = acc_column[keep]
        y = y_all[keep]
        sites = scores["site"].to_numpy()[keep]
        rho = sweep_curve(log_plm, la, y, alphas)
        # Drawn on |rho| (see curve_flags); the archive's own largest-rho rule is kept beside it
        # so the guard against the shipped summary still has something to compare.
        flags = curve_flags(rho, alphas, absolute=True)
        archive_rule = curve_flags(rho, alphas)
        if n_boot > 0:
            lo, hi = site_bootstrap_band(log_plm, la, y, sites, alphas, rng, n_boot=n_boot, band_every=band_every)
        else:
            lo = hi = np.full(len(alphas), np.nan)
        for j, a in enumerate(alphas):
            rows.append({
                "dataset": arm.dataset, "trait": arm.trait, "panel_label": arm.panel_label,
                "organism": arm.organism, "virus_tag": arm.virus_tag, "citation": arm.citation,
                "model_name": arm.model_name, "model_label": arm.model_label, "model_role": arm.model_role,
                "alpha": a, "rho": rho[j], "ci_lo": lo[j], "ci_hi": hi[j],
                "n": int(keep.sum()), "n_sites": int(len(np.unique(sites))), **flags,
                "interpretable_archive_rule": archive_rule["interpretable"],
                "interpretable_in_archive": bool(arm.interpretable),
            })
    return pd.DataFrame(rows)


def check_flags(curves: pd.DataFrame) -> None:
    """Under the archive's own rule, the recomputed flag must agree with the shipped summary.

    The figure marks optima on |rho| instead, so its own ``interpretable`` column legitimately
    differs from the shipped one; comparing that would turn the guard into noise. The archive's
    largest-rho rule is recomputed alongside and checked here, so a change in the underlying
    curves is still caught.
    """
    bad = curves[curves["interpretable_archive_rule"] != curves["interpretable_in_archive"]]
    if not bad.empty:
        names = sorted(set(zip(bad["dataset"], bad["trait"], bad["model_name"])))
        raise SelfCheckError(f"interpretable flag differs from the shipped summary for {names}")


def self_check(curves: pd.DataFrame, reference: pd.DataFrame, *, tol: float = SELF_CHECK_TOL) -> float:
    """Every recomputed curve against the archive's own, or raise. Returns the worst difference.

    ``reference`` has columns dataset, model, trait, alpha, rho (the shipped ``dms_alpha_curves``).
    A curve absent from the reference raises: silence would turn the check into a rubber stamp.
    """
    needed = {"dataset", "model", "trait", "alpha", "rho"}
    if not needed <= set(reference.columns):
        raise ValueError(f"reference curves lack column(s) {sorted(needed - set(reference.columns))}")
    ref = reference.assign(alpha=np.round(reference["alpha"].astype(float), 6))
    ref = ref.set_index(["dataset", "model", "trait", "alpha"])["rho"]
    if ref.index.duplicated().any():
        raise ValueError("reference curves have duplicate (dataset, model, trait, alpha) rows")
    worst = 0.0
    for (dataset, model, trait), g in curves.groupby(["dataset", "model_name", "trait"], sort=False):
        try:
            expected = ref.loc[[(dataset, model, trait, a) for a in np.round(g["alpha"], 6)]].to_numpy()
        except KeyError:
            raise SelfCheckError(f"{dataset}/{model}/{trait} is absent from the reference curves") from None
        diff = float(np.nanmax(np.abs(expected - g["rho"].to_numpy())))
        if not np.isfinite(diff):
            raise SelfCheckError(f"{dataset}/{model}/{trait}: no finite values to compare")
        worst = max(worst, diff)
    if worst > tol:
        raise SelfCheckError(f"recomputed sweep differs from the archive's curves by {worst:.2e} (tolerance {tol:g})")
    return worst


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def split_by_virus(curves: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """The curves grouped by virus, the main figure's virus first.

    Only the drawing is split. The sweep and its site bootstrap still run over every arm in the
    archive's order, so the published band is unchanged by which panels are drawn together.
    """
    tags = list(dict.fromkeys(curves["virus_tag"]))
    unknown = [t for t in tags if t not in VIRUS_STEMS]
    if unknown:
        raise ValueError(f"no figure stem for virus tag(s) {unknown}; add them to VIRUS_STEMS")
    order = [MAIN_VIRUS] + [t for t in tags if t != MAIN_VIRUS]
    return [(t, curves[curves["virus_tag"] == t]) for t in order if t in tags]


def panel_order(curves: pd.DataFrame) -> list[tuple[str, str]]:
    """Panels in the archive's order, except that the receptor-binding ones go last.

    The ACE2 panels measure binding rather than fitness and are the only ones read against a
    negative correlation, so they sit together at the end instead of being interleaved with the
    fitness panels. Drawing order only: the sweep and its bootstrap still run in the archive's
    order, which is what reproduces the published band.
    """
    panels = list(dict.fromkeys(zip(curves["dataset"], curves["trait"])))
    binding = [p for p in panels if RECEPTOR_TRAIT in p[1].lower()]
    return [p for p in panels if p not in binding] + binding


def panel_title(first: pd.Series) -> str:
    """Bold virus tag, then what the panel measures.

    The tag comes from the table, like every other label here; mathtext gives the bold, as the
    house style has no TeX. A title line holds one weight, so only the tag is emphasised.

    The source is not in this string: it is drawn under the title by ``panel_subheading``.
    With the citation in brackets on the same line, the five-panel SARS-CoV-2 layout ran its
    headings into the next panel's letter, which is the one thing a heading must not do.
    """
    return f"$\\bf{{{first['virus_tag']}}}$  {first['panel_label']}"


def panel_subheading(first: pd.Series) -> str:
    """The source, on its own line under the title."""
    return str(first["citation"])


def _signed(value: float) -> str:
    return f"{value:+.2f}".replace("-", "\N{MINUS SIGN}")


#: A hollow marker needs the interior extremum to be at least this many times stronger than
#: the model alone. Without it, a curve whose peak simply sits at alpha = 0 -- no weighting
#: helps, which is the opposite of a finding -- would get a marker at alpha = 0.00 and read
#: as though something had been found there.
SHOWN_BELOW_FLOOR_GAIN = 1.5

#: Points of room above the axes for the citation subheading, which is the lower of the two
#: heading lines.
PANEL_SUBHEADING_PAD_PT = 3.0

#: Type sizes for this figure. It is a 2x2 grid at double-column width, so its text can sit
#: well above the house defaults (8 pt labels, 7 pt ticks) and still have room; at the
#: defaults the panels read as mostly white space with small print in them.
SUPLABEL_PT = 10.0
#: The y label carries the statistic, so it is set a little larger than the x label.
SUPYLABEL_PT = 10.5
PANEL_TITLE_PT = 9.0
CITATION_PT = 7.5
DIRECT_LABEL_PT = 7.5
TICK_PT = 9.0
LEGEND_PT = 8.5

#: Legend entries per row. A single row of all of them runs past the figure's 180 mm width,
#: and savefig's tight box then grows the saved figure to hold the overflow -- which scales
#: the panels down by the same proportion the moment the figure is placed at column width.
#: Wrapping keeps the saved page at the width the panels were laid out for.
LEGEND_COLUMNS = 3

#: Legend handle length, in font-size units. The alpha = 0 rule is dashed on a 11.2 pt period
#: (style.ALPHA_ZERO_RULE, scaled by its line width), so the default 1.4 handle is barely one
#: dash and the entry reads as a stray stroke rather than as the dashed line it stands for.
#: This fits two whole periods. Lengthening the handle keeps the key honest: the swatch is
#: the same line as the one on the panels, where re-dashing it just for the legend would not.
LEGEND_HANDLE_LEN = 2.6

#: What the citation line occupies, title baseline to subheading baseline. The title and the
#: panel letter are both raised by this much so that they share the upper line. It tracks
#: the title size: at 9 pt type a 9 pt line sets the citation on the title's descenders.
CITATION_LINE_PT = 11.0

#: Points of room above the axes for the panel title. The panel letter is placed with the same
#: value, which puts it on the title's own baseline instead of floating in the white above it.
PANEL_HEADING_PAD_PT = PANEL_SUBHEADING_PAD_PT + CITATION_LINE_PT


def _tick_text(value: float, places: int) -> str:
    """Fixed-decimal tick text carrying matplotlib's own minus sign (U+2212), not a hyphen."""
    return f"{value:.{places}f}".replace("-", "−")


def _half_step_label(value: float, _pos) -> str:
    """Label the 0.5 steps of the alpha axis to one decimal, blank the 0.25 ones between."""
    return _tick_text(value, 1) if abs(value * 2 - round(value * 2)) < 1e-9 else ""


#: Candidate y tick steps, coarsest first. The y axis is labelled to one decimal place, so
#: every step is a multiple of 0.1: a finer ladder would print two ticks with the same text.
Y_TICK_STEPS = (1.0, 0.5, 0.2, 0.1)

#: Roughly how many labelled y ticks a panel should carry.
Y_TICK_TARGET = 4


def _y_tick_step(lo: float, hi: float) -> float:
    """The coarsest step that still puts about ``Y_TICK_TARGET`` labelled ticks on the axis."""
    span = hi - lo
    for step in Y_TICK_STEPS:
        if span / step >= Y_TICK_TARGET - 1:
            return step
    return Y_TICK_STEPS[-1]


def _shown_below_floor(g: pd.DataFrame) -> bool:
    """True for a real interior optimum whose model-alone signal is under the floor.

    Drawn hollow and labelled as such: the reader can see the feature without the figure
    claiming it as a result, and WEAK_SIGNAL_RHO stays where it is for every other curve.
    """
    if bool(g["interpretable"].iloc[0]) or bool(g["boundary"].iloc[0]):
        return False
    return abs(float(g["rho_best"].iloc[0])) >= SHOWN_BELOW_FLOOR_GAIN * abs(float(g["rho_alpha0"].iloc[0]))


#: Panel C's fine-tuned label, dropped this far in data units below the top of the panel.
FIG4C_GREEN_DROP = 0.06

#: The top of panel C's base label, in data units: under both curves, above the zero line.
FIG4C_BLUE_TOP = 0.045

#: The top of panel D's fine-tuned label, in data units. Its own curve is still descending
#: through this part of the panel, so a label hung just under the curve at alpha = 0 runs
#: into it further right; this clears the descent and leaves the label beside alpha = 0.
FIG4D_GREEN_TOP = 0.17

#: Panel letters whose "no relationship = 0" reads better under the zero line than over it.
ZERO_LABEL_BELOW = {"C"}


def _inset_top(ax, drop: float) -> float:
    """Data y just under the top of the panel, then ``drop`` data units lower again."""
    lo, hi = ax.get_ylim()
    return hi - 0.02 * (hi - lo) - drop


def _curve_point(g: pd.DataFrame, alpha_target: float) -> tuple[float, float]:
    """The curve's own (alpha, rho) nearest ``alpha_target``, to hang a label off."""
    alpha = g["alpha"].to_numpy()
    i = int(np.argmin(np.abs(alpha - alpha_target)))
    return float(alpha[i]), float(g["rho"].to_numpy()[i])


#: First-choice label positions, keyed by (panel letter, model role). Every automatic
#: candidate is anchored to a point on the curve, so the solver can only offer a label a seat
#: beside its own line; it has no way to express "under both curves" or "the empty corner".
#: Where a panel reads better with the label somewhere the curve does not go, name the spot
#: here. Each entry takes the axes, because these are measured against the data rather than
#: against the panel box. They are tried first and the automatic candidates still follow, so
#: a corner that stops being empty loses to a clean automatic placement instead of stranding
#: the label on top of the data. Keyed by letter, not by dataset: the script must not name
#: the values in its own tables (tests/test_fig04.py), and a panel letter is the figure's
#: own coordinate rather than something read out of the data.
LABEL_CORNERS = {
    # The fine-tuned label sat in the middle of the panel, in the gap between the two curves,
    # which is where a reader is trying to look.
    ("C", "finetune"): lambda ax, g: [
        Candidate((0.015, _inset_top(ax, FIG4C_GREEN_DROP)), (0, 0), "left", "top",
                  ("axes fraction", "data")),
    ],
    # The base label sat against the right edge, two thirds of the panel away from the dot
    # whose weight it quotes. Bring it in towards alpha = 0 and tuck it under both curves, in
    # the clear band between them and the zero line.
    ("C", "base"): lambda ax, g: [
        Candidate((0.55, FIG4C_BLUE_TOP), (0, 0), "left", "top", ("axes fraction", "data")),
    ],
    # The fine-tuned label sat out at the right edge, far from the dot whose weight it
    # quotes. Bring it back beside alpha = 0 and hang it under its own curve, which is
    # where this panel has room. Anchored to the curve, so it follows if the curve moves.
    ("D", "finetune"): lambda ax, g: [
        Candidate((0.52, FIG4D_GREEN_TOP), (0, 0), "left", "top", ("axes fraction", "data")),
        Candidate(_curve_point(g, CLEAR_OF_ZERO_ALPHA), (3, -5), "left", "top"),
    ],
}


def _arm_candidates(g: pd.DataFrame, above_first: bool) -> list[Candidate]:
    """Where a curve's direct label may go: close to the curve first, ends before the middle."""
    alpha, rho = g["alpha"].to_numpy(), g["rho"].to_numpy()
    spots = []
    # Prefer sitting beside the marked optimum, which is what the label is about: it quotes
    # the best weight, and a reader should not have to trace the curve from the panel edge
    # to find the dot that weight refers to. It is only a preference -- place_label still
    # moves the label if the optimum is crowded.
    if bool(g["interpretable"].iloc[0]) or _shown_below_floor(g):
        best = int(np.argmin(np.abs(alpha - float(g["best_alpha"].iloc[0]))))
        clear = int(np.argmin(np.abs(alpha - CLEAR_OF_ZERO_ALPHA)))
        # Beside the dot reading rightwards; then clear of the alpha = 0 line, still rightwards;
        # only then back across the dot, which is what sends the label left.
        spots += [(best, "left", 3), (clear, "left", 3)]
        spots += [(best, ha, dx) for ha, dx in H_ANCHORS if ha != "left"]
    spots += [(-1, "right", -2), (0, "left", 2)]
    for target in LABEL_ALPHAS:
        i = int(np.argmin(np.abs(alpha - target)))
        spots += [(i, ha, dx) for ha, dx in H_ANCHORS]
    out = []
    for gap in LABEL_GAPS:
        verticals = [(gap, "bottom"), (-gap, "top")] if above_first else [(-gap, "top"), (gap, "bottom")]
        for i, ha, dx in spots:
            for dy, va in verticals:
                out.append(Candidate((float(alpha[i]), float(rho[i])), (dx, dy), ha, va))
    return out


def _zero_candidates(below_first: bool = False) -> list[Candidate]:
    coords = ("axes fraction", "data")
    verticals = ((-2, "top"), (2, "bottom")) if below_first else ((2, "bottom"), (-2, "top"))
    return [Candidate((x, 0.0), (dx, dy), ha, va, coords)
            for x, dx, ha in ((1.0, -2, "right"), (0.0, 2, "left"))
            for dy, va in verticals]


def _style_panel(ax, group: pd.DataFrame) -> list[pd.DataFrame]:
    """Curves, bands, optima and the zero line. Text goes on afterwards, once layout is final."""
    arms = [g.sort_values("alpha") for _, g in group.groupby("model_name", sort=False)]
    for g in arms:
        colour = style.colour_for_role(g["model_role"].iloc[0])
        ax.fill_between(g["alpha"], g["ci_lo"], g["ci_hi"], color=colour, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(g["alpha"], g["rho"], color=colour, zorder=3)
        # A filled dot is an optimum the figure stands behind. A curve that has a genuine
        # interior extremum but too little signal at alpha = 0 to call it one gets a hollow
        # marker instead: the reader can see the feature without the panel claiming it, and
        # the weak-signal rule stays where it is for every other curve.
        interpretable = bool(g["interpretable"].iloc[0])
        if interpretable or _shown_below_floor(g):
            # The series colour stays on the artist either way; only the marker face is
            # hollowed, so a colour audit still sees a role colour here.
            ax.plot([float(g["best_alpha"].iloc[0])], [float(g["rho_best"].iloc[0])], marker="o",
                    markersize=4.2, color=colour,
                    markerfacecolor=colour if interpretable else style.SURFACE,
                    markeredgecolor=style.SURFACE if interpretable else colour,
                    markeredgewidth=0.7 if interpretable else 1.0,
                    zorder=4, linestyle="none")
    # alpha = 0 is the raw PLM score, the comparison every curve is read against, so the line
    # runs the full height of the panel and sits above the bands rather than behind them.
    ax.axvline(0.0, color=style.INK_MUTED, zorder=3.5, **style.ALPHA_ZERO_RULE)
    ax.margins(y=0.14)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -0.02 * (hi - lo)), max(hi, 0.02 * (hi - lo)))
    ax.set_xlim(group["alpha"].min(), group["alpha"].max())
    # One decimal place on every panel, so the ladder reads the same across the figure. The
    # step has to be a multiple of 0.1 to match: the 0.05 ladder the auto-locator chose for
    # the two shallow panels would print "0.2" twice.
    ax.yaxis.set_major_locator(MultipleLocator(_y_tick_step(*ax.get_ylim())))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _tick_text(v, 1)))
    # Label every other gridline. At this figure's tick size the full 0.25 ladder runs its
    # labels into each other, and blanking the odd ones leaves the gridlines exactly where
    # they were -- thinning the locator instead would thin the grid with it.
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(FuncFormatter(_half_step_label))
    style.reference_line(ax, 0.0, ZERO_LABEL)
    return arms



def _role_legend_label(curves: pd.DataFrame, role: str) -> str:
    """Name the colour's role.

    The legend used to list the checkpoints behind each colour, but every panel title carries its
    virus in bold and every curve is labelled with its own model, so the list only repeated what
    the panel already said.
    """
    return ROLE_NAMES.get(role, role)


def _label_panel(ax, arms: list[pd.DataFrame], letter: str) -> list[int]:
    """Direct labels (with the best weight, or "no optimum marked") and the zero-line label, placed clear of the rest.

    Returns the placement scores (0 = clean) so a test can assert nothing collided.
    """
    zero_label = next(txt for txt in ax.texts if txt.get_text() == ZERO_LABEL)
    curves = [(g["alpha"], g["rho"]) for g in arms]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    zero_line = (np.linspace(x0, x1, 200), np.zeros(200))
    # The dashed alpha = 0 line is drawn the full height of every panel and is the comparison the
    # whole figure is read against, so a label must not sit on it. It is vertical, which
    # curve_obstacles cannot densify (it interpolates y over x), so its points are taken directly.
    alpha_line_pts = ax.transData.transform(np.column_stack([np.zeros(200), np.linspace(y0, y1, 200)]))
    dots = [g for g in arms if bool(g["interpretable"].iloc[0]) or _shown_below_floor(g)]
    dot_pts = ax.transData.transform(np.array([[float(g["best_alpha"].iloc[0]), float(g["rho_best"].iloc[0])]
                                                for g in dots])) if dots else np.empty((0, 2))
    curve_pts = np.vstack([curve_obstacles(ax, curves), np.repeat(dot_pts, 40, axis=0), alpha_line_pts])
    with_zero = np.vstack([curve_pts, curve_obstacles(ax, [zero_line])])
    placed: list = []
    scores = []

    ends = sorted(arms, key=lambda g: float(g["rho"].iloc[-1]), reverse=True)
    for k, g in enumerate(ends):
        text = str(g["model_label"].iloc[0])
        if bool(g["interpretable"].iloc[0]):
            text += f"\nbest α = {_signed(float(g['best_alpha'].iloc[0]))}"
        elif _shown_below_floor(g):
            text += f"\nα = {_signed(float(g['best_alpha'].iloc[0]))} (below signal floor)"
        else:
            text += "\nno optimum marked"
        # The label wears its own curve's colour. These are direct labels sitting next to
        # the line they name, so the colour is the thing that ties them together; in ink
        # the reader has to match them by reading the legend instead. Both role colours
        # clear the 3:1 contrast floor against the surface (see mutacc/style.py).
        corner = LABEL_CORNERS.get((letter, str(g["model_role"].iloc[0])))
        first = corner(ax, g) if corner else []
        _, s = place_label(ax, text, first + _arm_candidates(g, above_first=(k == 0)), obstacles=with_zero,
                           placed=placed, fontsize=DIRECT_LABEL_PT,
                           color=style.colour_for_role(str(g["model_role"].iloc[0])),
                           linespacing=1.15)
        scores.append(s)
    scores.append(place_existing(ax, zero_label, _zero_candidates(letter in ZERO_LABEL_BELOW),
                                 obstacles=curve_pts, placed=placed))

    return scores


def column_count(n_panels: int) -> int:
    """The widest layout up to ``N_COLUMNS`` that leaves no blank cell.

    Since the panels are split by virus, a grid can end with one lonely panel on its own row;
    four panels read better as 2x2 than as 3 + 1.
    """
    for ncol in range(min(N_COLUMNS, n_panels), 1, -1):
        if n_panels % ncol == 0:
            return ncol
    return min(N_COLUMNS, n_panels)


def draw(curves: pd.DataFrame):
    if curves.empty:
        raise ValueError("no curves to draw")
    panels = panel_order(curves)
    ncol = column_count(len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig = style.figure(style.WIDTH_DOUBLE, 2.85 * nrow + 0.5)
    axes = np.atleast_1d(fig.subplots(nrow, ncol, squeeze=False)).ravel()
    drawn, panel_arms = [], []
    for ax, (dataset, trait) in zip(axes, panels):
        group = curves[(curves["dataset"] == dataset) & (curves["trait"] == trait)]
        panel_arms.append(_style_panel(ax, group))
        first = group.iloc[0]
        # Two heading lines: what the panel measures, then where the measurements came from.
        ax.set_title(panel_title(first), loc="left", fontsize=PANEL_TITLE_PT, pad=PANEL_HEADING_PAD_PT)
        ax.annotate(panel_subheading(first), xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, PANEL_SUBHEADING_PAD_PT), textcoords="offset points",
                    ha="left", va="baseline", fontsize=CITATION_PT, color=style.INK_MUTED)
        ax.tick_params(labelsize=TICK_PT)
        drawn.append(ax)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    style.panel_letters(drawn, pad_points=PANEL_HEADING_PAD_PT)

    fig.supxlabel(style.ALPHA_SWEEP_XLABEL, fontsize=SUPLABEL_PT)
    # The statistic is set bold so it reads first; mathtext is the only way to weight part of
    # a label. It is checked for both rendering and PDF text extraction (tests/test_fig04.py).
    fig.supylabel(r"$\mathbf{Spearman\ \rho}$, PLM vs DMS measured effect", fontsize=SUPYLABEL_PT)

    roles = list(dict.fromkeys(curves["model_role"]))
    handles = [Line2D([], [], color=style.colour_for_role(r), label=_role_legend_label(curves, r))
               for r in roles]
    # A legend entry must name something that is on the page: no dot when no optimum is marked,
    # no band when the bootstrap was switched off.
    if bool(curves["interpretable"].any()):
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=4.2, color=style.INK_MUTED,
                              label="Best weight"))
    # A hollow marker is a different claim from a filled one, so it needs its own entry --
    # and the entry only appears when one is actually drawn, same rule as the rest.
    if any(_shown_below_floor(g) for _, g in curves.groupby(["dataset", "trait", "model_name"])):
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=4.2,
                              color=style.INK_MUTED, markerfacecolor=style.SURFACE,
                              markeredgecolor=style.INK_MUTED, markeredgewidth=1.0,
                              label="Optimum below the signal floor"))
    handles.append(Line2D([], [], color=style.INK_MUTED, label="α = 0 (raw PLM score)",
                          **style.ALPHA_ZERO_RULE))
    if bool(curves["ci_lo"].notna().any()):
        handles.append(Patch(facecolor=style.NULL_GREY, alpha=0.3, linewidth=0, label="95% interval over DMS sites"))
    style.legend_above(fig, handles=handles, ncol=min(len(handles), LEGEND_COLUMNS),
                       handlelength=LEGEND_HANDLE_LEN, columnspacing=1.2, fontsize=LEGEND_PT)
    fig.canvas.draw()                      # layout must be final before text is measured
    scores = [s for i, (ax, arms) in enumerate(zip(drawn, panel_arms))
              for s in _label_panel(ax, arms, chr(ord("A") + i))]
    fig.label_scores = scores              # 0 = placed clear of everything
    return fig


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------

def _arm_table(curves: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "trait", "panel_label", "organism", "virus_tag", "citation", "model_name", "model_label",
            "model_role", "n", "n_sites", "best_alpha", "rho_best", "rho_alpha0", "interpretable",
            "boundary", "weak_signal"]
    return curves[cols].drop_duplicates(["dataset", "trait", "model_name"]).reset_index(drop=True)


def _letters(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


#: How the caption describes the estimator. The reader needs to know that the residue being
#: scored was hidden from the model, because the alternative is the obvious reading and it is
#: the one that makes the numbers look like a leak.
SCORING_CLAUSE = {
    "masked": ("with the scored site replaced by the mask token so that the residue actually "
               "there cannot inform its own prediction"),
    "unmasked": ("read from a single forward pass of the intact sequence, in which the model "
                 "does see the residue it is scoring (the wild-type marginal)"),
}


def caption(curves: pd.DataFrame, *, n_boot: int, scoring: str = "masked") -> str:
    scoring_clause = SCORING_CLAUSE[scoring]
    arms = _arm_table(curves)
    panels = panel_order(curves)
    letters = {p: chr(ord("A") + i) for i, p in enumerate(panels)}
    arms["letter"] = [letters[(d, t)] for d, t in zip(arms["dataset"], arms["trait"])]

    sources = arms.drop_duplicates("dataset")
    source_text = "; ".join(f"{r.dataset} ({r.organism}, {r.citation})" for r in sources.itertuples())
    n_arms = len(arms)
    flagged = arms[~arms["interpretable"]]
    n_edge = int(flagged["boundary"].sum())
    n_weak = int((flagged["weak_signal"] & ~flagged["boundary"]).sum())
    binding = sorted({r.letter for r in arms[arms["rho_alpha0"] < 0].itertuples()
                      if RECEPTOR_TRAIT in r.trait.lower()})

    ok = arms[arms["interpretable"]]
    ranges = []
    for role in ("base", "finetune"):
        s = ok.loc[ok["model_role"] == role, "best_alpha"]
        if len(s):
            ranges.append(f"{_signed(s.min())} to {_signed(s.max())} for {ROLE_NAMES[role].lower()}s")

    # Panels where both curves have an optimum: descriptive, in-sample.
    size = ok.assign(abs_alpha0=ok["rho_alpha0"].abs(), abs_best=ok["rho_best"].abs())
    pair = size.pivot_table(index=["dataset", "trait"], columns="model_role",
                            values=["abs_alpha0", "abs_best"], aggfunc="first")
    both = pair.dropna()
    comparison = ""
    if len(both) and {"base", "finetune"} <= set(ok["model_role"]):
        gain = both["abs_best"] - both["abs_alpha0"]
        base_ahead = int((both["abs_alpha0"]["base"] > both["abs_alpha0"]["finetune"]).sum())
        comparison = (
            f" In the {len(both)} panels where both curves have a marked optimum, the base model correlates more "
            f"strongly than the fine-tuned model at α = 0 in {base_ahead}, and moving to the best weight raises the "
            f"correlation in size by a median of {gain['base'].median():.3f} for base models and "
            f"{gain['finetune'].median():.3f} for fine-tuned models (in sample)."
        )

    spectrum_sentence = (
        "Accessibility is the probability, under the nucleotide change spectrum measured for that virus, "
        "that the reference codon mutates to a codon for the replacement's amino acid. "
    )
    receptor = (
        f" Panels {_letters(binding)} measure receptor binding rather than fitness. Binding is not what the models "
        "track and it is not maximised in these backgrounds, so the correlation there runs the other way: the "
        "replacements a model ranks highly are, on average, slightly affinity-lowering. The curves are read in size, "
        "like the rest, and the direction is a property of the assay rather than a model failure."
        if binding else ""
    )

    return f"""
    **Measured fitness effects against a language model's score, as the weight on
    mutational accessibility changes.**
    Deep mutational scanning (DMS) builds nearly every single amino-acid replacement in a
    protein and measures its effect in the laboratory, so the measurement does not depend on
    how easily a replacement arises by mutation. Each panel is one measured phenotype: {source_text}.
    Every replacement is scored by a protein language model (PLM) run on the scan's own reference
    sequence, {scoring_clause}, and combined with accessibility as log(model probability) + α ×
    log(accessibility), where α is the weight on mutational accessibility: 0 is the model alone
    and negative values divide accessibility out. {spectrum_sentence}The vertical axis is the Spearman rank
    correlation between that score and the measured effect over all replacements scored
    ({arms['n'].min():,} to {arms['n'].max():,} per curve, at {arms['n_sites'].min():,} to
    {arms['n_sites'].max():,} sites; 1 = same ranking, 0 = no relationship).

    Colour marks the model's role. Each panel compares a base (untuned) model with the same
    architecture fine-tuned on that virus, so absolute correlations are not comparable between
    model families. Dots mark each curve's best weight -- the α at which the correlation is
    largest in size, so that a panel whose phenotype runs opposite to fitness is read at its
    strongest relationship rather than where it is diluted closest to zero -- with its value in
    the label. A hollow marker means something weaker: the curve has a real interior optimum,
    at least half again as strong as the model alone, but the model alone is below the
    {WEAK_SIGNAL_RHO:g} floor, so the weight is shown without being claimed as a result. Bands are
    95% percentile intervals from a cluster bootstrap over DMS sites ({n_boot} resamples),
    because replacements at one site share a backbone and are not independent; the band is
    computed on every fifth weight and interpolated between. The best weight is chosen on the
    data it is scored on, so the height at the dot is optimistic by an unknown amount.{comparison}

    What is not shown. No optimum is marked on {len(flagged)} of {n_arms} curves: {n_edge} because the
    largest correlation lies at the end of the swept range and {n_weak} more because the correlation at
    α = 0 is below {WEAK_SIGNAL_RHO:g} in size, leaving no signal for the weight to improve. The
    curves are still drawn.{receptor} Where optima are marked, the best weight ranges from
    {' and from '.join(ranges)}.
    """


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--guide", required=True, type=Path, help="guide YAML file")
    parser.add_argument("--results", type=Path, default=None, help="results directory (default: the guide's)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output", help="output directory")
    parser.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap resamples over DMS sites")
    add_scoring_argument(parser)
    args = parser.parse_args(argv)
    args.guide = load_guide(args.guide)
    if args.results is None:
        args.results = args.guide.results_dir
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    guide, results = args.guide, args.results
    scores = guide.read_table("dms_scores", results)
    summary = guide.read_table("dms_peak_summary", results)

    # The gate, always, whichever arm is drawn. The archive's curves and interpretable flags were
    # computed from the wild-type (unmasked) probability, so that arm is the only one with a
    # fixed point to be checked against; run it first and fail here rather than draw a masked
    # figure whose difference from the published one could be anything. No bootstrap: the gate
    # compares rho only, and the band is where nearly all the time goes.
    gate_curves = compute_curves(scores, summary, n_boot=0, spectrum="published")
    check_flags(gate_curves)
    ref = guide.read_table("dms_alpha_curves", results)
    worst = self_check(gate_curves, ref)
    print(f"[gate] recomputed unmasked curves vs the archive's: max |diff| {worst:.2e}")

    if args.scoring == "masked":
        scores = apply_masked(scores, guide.read_table("dms_scores_masked", results))
    curves = compute_curves(scores, summary, n_boot=args.boot, spectrum="published")
    if args.scoring == "unmasked":
        check_flags(curves)

    written = []
    for virus, sub in split_by_virus(curves):
        stem = VIRUS_STEMS[virus]["published"]
        fig = draw(sub)
        paths = style.save_figure(fig, args.out / stem)
        write_values(sub.drop(columns=["interpretable_in_archive"]), args.out, stem)
        write_caption(caption(sub, n_boot=args.boot, scoring=args.scoring), args.out, stem)
        written += [p.name for p in paths]
    print("wrote", ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
