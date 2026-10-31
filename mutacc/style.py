"""The single source of visual truth for every manuscript figure.

Every figure script imports from here and sets nothing of its own. No script may
call ``plt.rcParams.update``, ``plt.style.use``, ``savefig`` directly, or write a
colour literal.

Why this module exists
----------------------
Before it, the repository held 258 distinct hex literals, 79 distinct figure
sizes, four ``savefig`` dpi values, six spellings of "white", and four competing
base-vs-fine-tune colour schemes, two of which mapped the hues to the opposite
roles.

Two design choices are deliberate and should not be undone:

**No seaborn.** Several scripts used ``sns.color_palette("viridis", 2)`` behind a
bare ``try/except``. That call *trims* the ramp and returns blue/green, while the
hand-written fallback used the raw viridis endpoints -- dark purple and yellow. On
any interpreter without seaborn the same model silently changed colour, with no
error. Depending on nothing removes the failure mode rather than patching it.

**Okabe-Ito, not viridis, for categorical roles.** The previously published pair
(``#31688e`` / ``#35b779``) fails two checks: ``#31688e`` is below the chroma
floor and reads as grey, and ``#35b779`` sits at 2.49:1 against white, under the
3:1 contrast floor. The repository already worked around that by giving the two
colours different alphas. Okabe-Ito passes the lightness band, chroma floor,
colour-vision-deficiency separation, normal-vision separation and contrast checks
with no workaround.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.transforms import ScaledTranslation

__all__ = [
    "BASE", "FINETUNE", "ACCESS", "BIOCHEM", "NULL_GREY",
    "INK", "INK_MUTED", "GRID", "SURFACE", "SURFACE_SET_ASIDE", "MEASURED",
    "SEQUENTIAL", "DIVERGING",
    "ROLE_COLOURS", "colour_for_role",
    "CATEGORICAL", "colours_for_categories",
    "WIDTH_SINGLE", "WIDTH_DOUBLE", "DPI",
    "use_house_style", "figure", "panel_letter", "panel_letters",
    "reference_line", "save_figure", "finalise_axes", "panel_title",
    "TAGLINE_SIZE", "TITLE_PAD", "AXIS_FLOOR_PAD", "cut_axis_below",
]


# ---------------------------------------------------------------------------
# Colour. Okabe & Ito (2008), the colour-vision-deficiency-safe qualitative set.
#
# Assigned by ROLE, never by position in a list. A filter that drops a series
# must not repaint the survivors, so never index these by loop counter.
# ---------------------------------------------------------------------------

BASE = "#0072b2"       # base / untuned model          (blue)
FINETUNE = "#009e73"   # fine-tuned model              (bluish green)
ACCESS = "#d55e00"     # mutational accessibility      (vermillion)
BIOCHEM = "#cc79a7"    # Grantham / BLOSUM62           (reddish purple)
NULL_GREY = "#999999"  # nulls, chance lines, references
BASE_SECOND = "#6a3d9a"  # a second untuned model, when two are compared (dark violet; also pair with a marker shape)

# ---------------------------------------------------------------------------
# The two-palette rule. See README.md, "Colour plan".
#
# 1. BASE and FINETUNE are the *model pair*. Blue means untuned and green means
#    fine-tuned in every figure in the set, and neither hue is ever spent on
#    anything else.
# 2. Everything that is not a model -- assay type, structure, antigenic site,
#    virus -- is coloured from CATEGORICAL below. It holds only warm and violet
#    hues, so nothing drawn from it can be read as the model pair.
#
# Why not ColorBrewer Dark2: its first entry, #1b9e77, is dE 3.2 from FINETUNE
# in normal vision. It is the same teal-green, so a Dark2 series would collide
# with "fine-tuned" on its very first colour.
#
# Every entry clears the 3:1 contrast floor against white, sits at least dE 42
# from both BASE and FINETUNE, is at least dE 42 from its neighbours here, and
# is separable from them in greyscale. tests/test_style.py pins all four.
# ---------------------------------------------------------------------------

#: Non-model categories, in order. Index by sorted category name, never by loop
#: counter, so dropping a category cannot repaint the ones that remain.
CATEGORICAL = (
    "#d55e00",  # vermillion
    "#cc79a7",  # reddish purple
    "#6a3d9a",  # violet
)
# A fourth colour must clear the same four tests. The obvious candidate, brick
# #8c564b, fails: it sits 0.040 from violet in luminance, under the 0.045
# greyscale floor, so the two merge in a mono print. Pick one deliberately
# rather than appending and hoping.

#: Role name -> colour. The only sanctioned way to colour a series.
ROLE_COLOURS = {
    "base": BASE,
    "base_second": BASE_SECOND,
    "finetune": FINETUNE,
    "accessibility": ACCESS,
    "biochemistry": BIOCHEM,
    "null": NULL_GREY,
}

# Text and furniture wear ink, never a series colour.
INK = "#1a1a1a"
INK_MUTED = "#5c5c5c"
GRID = "#d9d9d9"
SURFACE = "#ffffff"
#: A panel that is drawn but set aside, shaded so the reader sees it was not counted. Neutral
#: and lighter than GRID, so a curve and its gridlines stay legible on top of it, and grey
#: rather than tinted so it cannot be read as a status colour.
SURFACE_SET_ASIDE = "#f2f2f2"

#: The measured quantity itself, where a panel overlays it on model scores: the phenotype a
#: deep mutational scan reports, not anything a model produced. Named here rather than written
#: into a figure because the no-literals rule has no exception, and kept at the value Figure SM
#: already shipped so that giving it a name changes no published panel. It is deliberately not
#: in CATEGORICAL: that tuple is for interchangeable categories, and this is one fixed series.
MEASURED = "#5c3566"

#: Magnitude. One hue, light to dark. Never a rainbow.
SEQUENTIAL = "viridis"

#: Polarity. Two hues with a neutral grey midpoint; always centre on zero.
DIVERGING = "RdBu_r"


def colour_for_role(role: str) -> str:
    """Return the colour for a named role.

    Raises rather than falling back, so a typo surfaces as an error instead of
    a silently miscoloured figure.
    """
    try:
        return ROLE_COLOURS[role]
    except KeyError:
        raise KeyError(
            f"unknown plot role {role!r}; "
            f"known roles are {sorted(ROLE_COLOURS)}. "
            "Add a role here rather than passing a colour literal."
        ) from None


def colours_for_categories(names) -> dict[str, str]:
    """Map non-model category names to colours, by sorted name.

    Sorting rather than taking input order means a filter that drops one
    category cannot repaint the survivors -- the same rule the role colours
    follow. Raises if there are more categories than the palette holds, because
    silently cycling would give two groups the same colour.
    """
    unique = sorted(set(names))
    if len(unique) > len(CATEGORICAL):
        raise ValueError(
            f"{len(unique)} categories but the categorical palette has "
            f"{len(CATEGORICAL)} colours: {unique}. Encode some of them with "
            "marker shape or fill, or widen CATEGORICAL deliberately."
        )
    return dict(zip(unique, CATEGORICAL))


# ---------------------------------------------------------------------------
# Geometry. MBE column widths.
# ---------------------------------------------------------------------------

WIDTH_SINGLE = 3.46   # inches (88 mm)
WIDTH_DOUBLE = 7.09   # inches (180 mm)
DPI = 400             # one value, everywhere

#: Native size and text sizes of a panel that a composite places in one cell. Every panel script
#: draws at this size, so the composite scales them all by the same factor and an axis label is
#: the same size in every panel. (A frozen raster panel keeps whatever it was drawn with.)
PANEL_SIZE = (4.8, 3.05)   # inches
PANEL_LABEL_PT = 11.5      # axis titles
PANEL_TICK_PT = 11.0       # tick labels
#: The axes box inside a panel, as figure fractions (left, bottom, width, height). The same box in
#: every panel, saved without cropping (``save_panel``), puts every panel's axes, tick labels and
#: titles at the same place and size once the composite scales the panels alike. A cropped save
#: would size each page to its own contents, which is what made the panels' margins and text
#: disagree.
PANEL_AXES = (0.225, 0.174, 0.76, 0.813)
#: Tick length and the gap from tick to its label, in points. Shorter and closer than the
#: house default, because in a panel this furniture is scaled down with the page and the
#: space it was holding read as a band of white between the plot and its axis title.
PANEL_TICK_LEN_PT = 3.0
PANEL_TICK_PAD_PT = 2.0
#: Where the axis titles sit, in axes coordinates, so they do not move with the width of a
#: panel's tick labels ("10^-14" is wider than "0.65"): x of the y title, y of the x title.
PANEL_YLABEL_X = -0.205
PANEL_XLABEL_Y = -0.123


def use_house_style() -> None:
    """Install the house rcParams. Idempotent; safe to call more than once."""
    matplotlib.use("Agg", force=False)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,

        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # Recessive furniture: the data is the figure.
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,

        "lines.linewidth": 1.4,
        "lines.markersize": 4,

        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def panel_axes(fig: Figure) -> Axes:
    """The one axes of a composite panel, in the shared box ``PANEL_AXES``."""
    fig.set_layout_engine("none")      # the box is fixed; a layout engine would move it
    return fig.add_axes(PANEL_AXES)


def place_panel_labels(ax: Axes) -> None:
    """Pin both axis titles to the shared positions and sizes. Call after setting the labels."""
    ax.xaxis.label.set_size(PANEL_LABEL_PT)
    ax.yaxis.label.set_size(PANEL_LABEL_PT)
    ax.tick_params(axis="both", which="major",
                   length=PANEL_TICK_LEN_PT, pad=PANEL_TICK_PAD_PT)
    ax.yaxis.set_label_coords(PANEL_YLABEL_X, 0.5)
    ax.xaxis.set_label_coords(0.5, PANEL_XLABEL_Y)


def figure(width: float = WIDTH_SINGLE, height: float = 2.6, **kwargs) -> Figure:
    """Create a figure at a sanctioned column width."""
    use_house_style()
    return plt.figure(figsize=(width, height), **kwargs)


# ---------------------------------------------------------------------------
# Annotation helpers. These exist so the legibility rules are the easy path.
# ---------------------------------------------------------------------------

def panel_letter(ax: Axes, letter: str, *, dx: float = -0.12, dy: float = 1.06,
                 pad_points: float | None = None) -> None:
    """Place a bold panel letter outside the axes.

    Captions reference panels by letter, so every multi-panel figure needs these.
    The published Figure 3 had none while its own README described "panel B".

    ``pad_points`` puts the letter that many points above the axes instead of at ``dy``,
    on the same baseline as a title set with the same pad. Use it to keep the letter in
    line with the panel heading rather than floating in a band of white above it.
    """
    transform = ax.transAxes
    y, va = dy, "bottom"
    if pad_points is not None:
        transform = ax.transAxes + ScaledTranslation(
            0.0, pad_points / 72.0, ax.figure.dpi_scale_trans)
        y, va = 1.0, "baseline"
    ax.text(
        dx, y, letter,
        transform=transform,
        fontsize=10, fontweight="bold", color=INK,
        ha="left", va=va,
    )


def panel_letters(axes, *, start: str = "A", **kwargs) -> None:
    """Letter a sequence of axes A, B, C, ... in the order given."""
    first = ord(start)
    for offset, ax in enumerate(axes):
        panel_letter(ax, chr(first + offset), **kwargs)


def reference_line(
    ax: Axes,
    value: float,
    label: str,
    *,
    axis: str = "y",
    label_loc: str = "right",
    label_side: str = "above",
    **kwargs,
) -> None:
    """Draw a labelled reference line.

    ``label`` is required, not optional. An unlabelled dashed line asks the
    reader to guess; "chance = 0.5" tells them. Reference lines are always
    neutral grey -- spending a status colour such as red on a neutral annotation
    makes a genuine warning unreadable later.

    ``label_loc`` moves the label to the other end when a data point sits where
    it would otherwise land, and ``label_side`` moves it across the line, for the
    case where both ends of the band on one side are already occupied. Collisions
    are not detected automatically, which is why every figure still has to be
    rendered and looked at.
    """
    if not label:
        raise ValueError(
            "reference_line requires a label: an unlabelled rule is not readable. "
            "Pass e.g. label='chance = 0.5'."
        )
    if label_loc not in {"left", "right"}:
        raise ValueError(f"label_loc must be 'left' or 'right', got {label_loc!r}")
    if label_side not in {"above", "below"}:
        raise ValueError(f"label_side must be 'above' or 'below', got {label_side!r}")

    draw = ax.axhline if axis == "y" else ax.axvline
    line_kwargs = {"color": NULL_GREY, "linestyle": "--", "linewidth": 0.8, "zorder": 1}
    line_kwargs.update(kwargs)
    draw(value, **line_kwargs)

    at_right = label_loc == "right"
    above = label_side == "above"
    if axis == "y":
        ax.annotate(
            label, xy=(1.0 if at_right else 0.0, value),
            xycoords=("axes fraction", "data"),
            xytext=(-2 if at_right else 2, 2 if above else -2), textcoords="offset points",
            ha="right" if at_right else "left", va="bottom" if above else "top",
            fontsize=6, color=INK_MUTED,
        )
    else:
        # On a vertical rule "above"/"below" read as the right and left of the line.
        ax.annotate(
            label, xy=(value, 1.0), xycoords=("data", "axes fraction"),
            xytext=(2 if above else -2, -2), textcoords="offset points",
            ha="left" if above else "right", va="top",
            fontsize=6, color=INK_MUTED,
        )


TAGLINE_SIZE = 6.5     # points; deliberately smaller than the title it sits under
TITLE_PAD = 12         # points of room above the axes for a one-line tagline
TAGLINE_LINE_PAD = 8   # further points of room for each extra tagline line


#: The swept weight is the one symbol the reader has to carry across Figures 3 and 4,
#: so it is set bold where the label introduces it. ``\mathbf`` keeps the upright shape
#: of the plain glyph; ``\boldsymbol`` would slant it and stop matching the alphas drawn
#: on the panels. The words are shared so the two figures cannot drift apart.
ALPHA_SWEEP_XLABEL = (
    r"Weight on mutational accessibility ($\mathbf{\alpha}$);"
    "  0 = model alone,  negative = accessibility divided out"
)

#: Inches of padding constrained_layout leaves around the axes. It is also what it spends
#: on the band between an "outside" legend and the panel headings, and at the default
#: 0.042 that band reads as a gap in the figure. ``legend_above`` tightens it. The value
#: is the smallest that still leaves the legend clear of a two-line tagline; below about
#: 0.016 the two begin to touch, which is why this is a named constant and not a literal.
LEGEND_BAND_PAD = 0.02


def legend_above(fig: Figure, *args, pad: float = LEGEND_BAND_PAD, **kwargs):
    """Figure legend in a row above the panels, close enough to read as their key.

    ``loc="outside upper center"`` asks constrained_layout for the row; the pad it leaves
    under that row is the engine's ``h_pad``, so the gap is closed by setting that rather
    than by moving the legend once the layout is final. Everything else about the legend
    is the caller's, and the layout still owns the geometry.
    """
    engine = fig.get_layout_engine()
    if not isinstance(engine, ConstrainedLayoutEngine):
        raise ValueError(
            "legend_above sets the constrained layout's pad, so the figure needs that "
            f"engine, not {type(engine).__name__}; use style.figure()"
        )
    legend = fig.legend(*args, loc="outside upper center", **kwargs)
    engine.set(h_pad=pad)
    return legend


#: The alpha = 0 rule, shared by the two sweep figures so their weight and dash rhythm
#: cannot drift apart. Heavier than an ordinary reference line because alpha = 0 is the
#: value the whole figure is read against. The colour is deliberately NOT shared: each
#: figure keeps the grey it already had.
ALPHA_ZERO_RULE = MappingProxyType({"linewidth": 1.6, "linestyle": (0, (5, 2))})


AXIS_FLOOR_PAD = 0.04  # fraction of the plotted range left below a cut


def cut_axis_below(ax: Axes, level: float) -> None:
    """Stop the vertical axis just under ``level`` (the no-relationship or chance value).

    Below it a score is anti-predictive, which says nothing about how well a model
    predicts. Call after plotting so the autoscaled top is kept. The caption must say the
    axis is cut, and the values table must keep what was cut.
    """
    top = ax.get_ylim()[1]
    ax.set_ylim(level - AXIS_FLOOR_PAD * (top - level), top)


def panel_title(
    ax: Axes,
    title: str,
    tagline: str = "",
    *,
    title_size: float | None = None,
    tagline_size: float | None = None,
) -> None:
    """Title that names the metric exactly, with a small plain-words tagline above the axes.

    The tagline says in ordinary language what is being asked; it never replaces the
    metric name, because a reader who knows the metric needs to see which one it is.

    A tagline may carry newlines. Wrapping it is better than letting one long line set the
    width of the whole figure, so the title is pushed up by however many lines there are.
    """
    if not tagline:
        ax.set_title(title, loc="left", fontsize=title_size)
        return
    extra = TAGLINE_LINE_PAD * (tagline.count("\n"))
    ax.set_title(title, loc="left", pad=TITLE_PAD + extra, fontsize=title_size)
    ax.text(
        0.0, 1.015, tagline, transform=ax.transAxes,
        fontsize=tagline_size if tagline_size is not None else TAGLINE_SIZE,
        color=INK_MUTED, va="bottom", ha="left",
    )


def finalise_axes(
    ax: Axes,
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    labelsize: float | None = None,
) -> None:
    """Apply labels, rejecting bare column names.

    Axis labels are read by people who are neither ML practitioners nor
    evolutionary geneticists. ``plm_prob`` means nothing to them; "Model
    probability of this replacement" does.
    """
    for text, which in ((xlabel, "x"), (ylabel, "y")):
        if text and _looks_like_a_column_name(text):
            raise ValueError(
                f"{which}label {text!r} looks like a column name. "
                "Use plain words describing what the reader is seeing."
            )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=labelsize)
    if title:
        ax.set_title(title, loc="left", fontsize=labelsize)


def _looks_like_a_column_name(text: str) -> bool:
    """True for snake_case or dotted identifiers with no spaces."""
    stripped = text.strip()
    if " " in stripped:
        return False
    return ("_" in stripped or "." in stripped) and stripped.replace(
        "_", ""
    ).replace(".", "").isalnum()


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------

def save_panel(fig: Figure, path: str | Path, *, close: bool = True,
               suffixes: tuple[str, ...] = (".pdf", ".png")) -> list[Path]:
    """Save a composite panel at its full canvas, not cropped to its contents.

    ``save_figure`` crops to the ink, so two panels with the same axes box come out as pages of
    different size and the composite scales their text differently. A panel saved here keeps the
    page it was drawn on, so every panel built on ``PANEL_AXES`` lines up with every other.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unknown = set(suffixes) - {".pdf", ".png"}
    if unknown:
        raise ValueError(f"save_panel writes .pdf and .png, not {sorted(unknown)}")
    written = []
    for suffix in suffixes:
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=DPI, bbox_inches=fig.bbox_inches, facecolor=SURFACE)   # the whole canvas; None would fall back to the house "tight"
        written.append(target)
    if close:
        plt.close(fig)
    return written


def save_figure(fig: Figure, path: str | Path, *, close: bool = True,
                suffixes: tuple[str, ...] = (".pdf", ".png")) -> list[Path]:
    """Save a figure as PDF and PNG with the same stem.

    MBE wants vector for submission and a raster is wanted for review, so both
    are always written and neither can drift from the other. ``rasterise_figures.sh``
    used to back-fill PNGs from PDFs but skipped any PDF whose script had already
    written its own PNG, so a figure's final resolution depended on which path
    produced it. Writing both here removes that ambiguity.

    ``suffixes`` narrows that, and exists for one case: a panel drawn only to be fed to
    ``figures/panelise.py``. It is not a figure, nobody reviews it on its own, and a PNG
    of it sitting in ``output/`` beside the composite invites someone to use the wrong
    file. Pass ``(".pdf",)`` there. Everything that is a figure writes both.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unknown = set(suffixes) - {".pdf", ".png"}
    if unknown:
        raise ValueError(f"save_figure writes .pdf and .png, not {sorted(unknown)}")
    written = []
    for suffix in suffixes:
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
        written.append(target)
    if close:
        plt.close(fig)
    return written
