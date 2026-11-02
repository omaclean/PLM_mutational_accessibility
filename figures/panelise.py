"""Assemble a multi-panel composite figure from finished panel files.

    python figures/panelise.py --layout guides/composites/fig01.yaml --out output/

Used for the manuscript composites that are put together from panels drawn
elsewhere (a tree that cannot be regenerated, ranked-mutation plots, heatmaps).
It replaces the old ``panelise.ipynb``, which had every path hard-coded, two
different definitions of the row builder, and panel letters placed at
hand-typed coordinates.

What the layout file controls
-----------------------------
Everything comes from the layout YAML (see ``guides/composites/``). Paths are
relative to the repository root. The page is ``width`` wide (the standard double
column by default); each row spans that width and its height follows from the
panels' own aspect ratios, so nothing is stretched. PDF panels stay vector.

Panel letters are never given coordinates. Each letter sits at the top-left
corner of its panel's rectangle plus a fixed margin. A panel that holds two
sub-plots (for example one file with a B and a C plot side by side) lists
``letter_x``: fractions of the panel width at which each letter goes.

Mistakes raise ``LayoutError`` before anything is drawn: a missing file, a repeated
panel key, a key used in no row or twice, an absolute or ``..`` path, or a letter
list that does not match the number of letter slots.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import yaml
from PIL import Image

from mutacc import style

# PyMuPDF is imported *after* matplotlib and the house style, and that order matters.
# PyMuPDF ships its own native libraries and loading it first pulls in the system
# libstdc++, which on this machine is older than the one Pillow's libLerc needs; Pillow
# then fails to import with a GLIBCXX version error. Importing matplotlib first means
# Pillow is already loaded and the clash never arises. Do not move this to the top.
try:
    import fitz  # PyMuPDF
except ImportError:         # reported by _require_pymupdf() when a composite is actually built
    fitz = None

__all__ = ["LayoutError", "Layout", "Panel", "Row", "load_layout", "validate",
           "compose", "render", "main"]

#: Panel letter size in points. Matches ``style.panel_letter`` (10 pt bold).
LETTER_PT = 10.0
#: House-style header drawn by this script, for panels whose own title is cropped off.
#: A frozen panel carries whatever title its author gave it, centred, in its own font and
#: at whatever size the panel's scaling leaves it; Figure 2's four panels disagreed three
#: ways and panel 2C's was truncated mid-word in the source image. Cropping the baked-in
#: title and drawing one here is the only way to make them match, because two of the
#: frozen panels have no generator (data/frozen_panels/README.md).
TITLE_PT = 8.5
TAGLINE_PT = 7.5
#: Points between the tagline and the top of the panel content.
HEADER_GAP_PT = 3.5
#: Points kept clear at the right of a header, so a title never touches the page edge.
HEADER_RIGHT_MARGIN_PT = 2.0
#: Gap between a panel's top-left corner and its letter, in points.
LETTER_MARGIN_PT = 6.0
#: Cap height of DejaVu Sans Bold as a fraction of the font size. Puts the letter's
#: top edge (not its baseline) at the margin.
_CAP_HEIGHT = 0.729
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_FONT_NAME = "dejavu-bold"


class LayoutError(ValueError):
    """The layout file is inconsistent. Raised before any drawing."""


def _require_pymupdf() -> None:
    """Fail with an instruction, not an ImportError traceback, when PyMuPDF is missing."""
    if fitz is None:
        raise SystemExit(
            "panelise.py needs PyMuPDF to place panels (pip install pymupdf, or use environment.yml). "
            "Nothing else in this repository needs it."
        )


@dataclass
class Panel:
    key: str
    file: str
    letter_x: list[float] = field(default_factory=lambda: [0.0])
    crop_top: float = 0.0       # points of a PDF panel's own page to cut off the top
    crop_top_frac: float = 0.0   # fraction of the panel's own height to cut off the top
    crop_left_frac: float = 0.0  # fraction of its own width to cut off the left
    crop_right_frac: float = 0.0 # fraction of its own width to cut off the right
    title: str = ""             # drawn here in the house style, above the panel
    tagline: str = ""           # smaller, muted, under the title
    #: Draw the header *over* the panel's own top-left instead of in a band above it.
    #: The band is what makes a header cost page height; an inset one costs none. Only
    #: safe where the panel's top-left corner is empty, which is a property of the
    #: artwork, so it is opt-in per panel rather than a layout-wide setting.
    header_inset: bool = False


@dataclass
class Row:
    keys: list[str]
    weights: list[float] | None = None
    width_fraction: float = 1.0


@dataclass
class Layout:
    name: str
    panels: list[Panel]
    rows: list[Row]
    width: str | float = "double"
    letters: list[str] | None = None
    letter_pt: float = LETTER_PT
    letter_margin_pt: float = LETTER_MARGIN_PT
    title_pt: float = TITLE_PT
    tagline_pt: float = TAGLINE_PT
    #: Points between the tagline and the panel content. Layout-settable because a
    #: composite that has to fit a page cannot always afford the default.
    header_gap_pt: float = HEADER_GAP_PT
    gap_pt: float = 0.0
    root: Path = ROOT

    @property
    def page_width_pt(self) -> float:
        if self.width == "double":
            return style.WIDTH_DOUBLE * 72
        if self.width == "single":
            return style.WIDTH_SINGLE * 72
        return float(self.width)

    def panel(self, key: str) -> Panel:
        return next(p for p in self.panels if p.key == key)

    def ordered_panels(self) -> list[Panel]:
        """Panels in reading order: row by row, left to right."""
        return [self.panel(k) for row in self.rows for k in row.keys]


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_layout(path: str | Path, *, root: Path = ROOT) -> Layout:
    """Read a layout YAML. Does not touch the panel files; call :func:`validate`."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = root / path
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as err:
        raise LayoutError(f"cannot read layout {path}: {err}") from None
    if not isinstance(raw, dict):
        raise LayoutError(f"{path}: layout must be a mapping")

    for required in ("name", "panels", "rows"):
        if required not in raw:
            raise LayoutError(f"{path}: missing required field {required!r}")

    if not isinstance(raw["panels"], list) or not isinstance(raw["rows"], list):
        raise LayoutError(f"{path}: 'panels' and 'rows' must be lists")
    panels = []
    for entry in raw["panels"]:
        if not isinstance(entry, dict) or "key" not in entry or "file" not in entry:
            raise LayoutError(f"{path}: every panel needs 'key' and 'file', got {entry!r}")
        extra = set(entry) - {"key", "file", "letter_x", "crop_top", "crop_top_frac",
                              "crop_left_frac", "crop_right_frac", "title", "tagline",
                              "header_inset"}
        if extra:
            raise LayoutError(f"{path}: panel {entry['key']!r} has unknown fields {sorted(extra)}")
        if "letter_x" in entry and not isinstance(entry["letter_x"], list):
            raise LayoutError(f"{path}: panel {entry['key']!r}: letter_x must be a list of fractions")
        panels.append(Panel(
            key=str(entry["key"]),
            file=str(entry["file"]),
            letter_x=[float(x) for x in entry.get("letter_x", [0.0])],
            crop_top=float(entry.get("crop_top", 0.0)),
            crop_top_frac=float(entry.get("crop_top_frac", 0.0)),
            crop_left_frac=float(entry.get("crop_left_frac", 0.0)),
            crop_right_frac=float(entry.get("crop_right_frac", 0.0)),
            title=str(entry.get("title", "")),
            tagline=str(entry.get("tagline", "")),
            header_inset=bool(entry.get("header_inset", False)),
        ))

    rows = []
    for entry in raw["rows"]:
        if isinstance(entry, list):
            rows.append(Row(keys=[str(k) for k in entry]))
        elif isinstance(entry, dict) and "panels" in entry:
            weights = entry.get("weights")
            rows.append(Row(
                keys=[str(k) for k in entry["panels"]],
                weights=[float(w) for w in weights] if weights else None,
                width_fraction=float(entry.get("width_fraction", 1.0)),
            ))
        else:
            raise LayoutError(f"{path}: a row is a list of keys or a mapping with 'panels'; got {entry!r}")

    letters = raw.get("letters")
    return Layout(
        name=str(raw["name"]),
        panels=panels,
        rows=rows,
        width=raw.get("width", "double"),
        letters=[str(x) for x in letters] if letters is not None else None,
        letter_pt=float(raw.get("letter_pt", LETTER_PT)),
        letter_margin_pt=float(raw.get("letter_margin_pt", LETTER_MARGIN_PT)),
        title_pt=float(raw.get("title_pt", TITLE_PT)),
        tagline_pt=float(raw.get("tagline_pt", TAGLINE_PT)),
        header_gap_pt=float(raw.get("header_gap_pt", HEADER_GAP_PT)),
        gap_pt=float(raw.get("gap_pt", 0.0)),
        root=root,
    )


def resolve(layout: Layout, panel: Panel) -> Path:
    """Absolute path of a panel file. Layout paths must be relative to the repo root."""
    rel = Path(panel.file)
    if rel.is_absolute():
        raise LayoutError(f"panel {panel.key!r}: path must be relative to the repository root, got {panel.file!r}")
    if ".." in rel.parts:
        raise LayoutError(f"panel {panel.key!r}: path may not contain '..': {panel.file!r}")
    return layout.root / rel


def validate(layout: Layout) -> None:
    """Raise :class:`LayoutError` if the layout cannot be drawn faithfully."""
    keys = [p.key for p in layout.panels]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise LayoutError(f"duplicate panel key(s): {dupes}")

    if not layout.rows:
        raise LayoutError("layout has no rows")

    used = [k for row in layout.rows for k in row.keys]
    unknown = sorted(set(used) - set(keys))
    if unknown:
        raise LayoutError(f"rows refer to undeclared panel key(s): {unknown}")
    repeated = sorted({k for k in used if used.count(k) > 1})
    if repeated:
        raise LayoutError(f"panel key(s) placed more than once: {repeated}")
    unplaced = sorted(set(keys) - set(used))
    if unplaced:
        raise LayoutError(f"panel(s) declared but not placed in any row: {unplaced}")

    for row in layout.rows:
        if not row.keys:
            raise LayoutError("a row has no panels")
        if row.weights is not None and (
            len(row.weights) != len(row.keys) or any(not w > 0 for w in row.weights)
        ):
            raise LayoutError(f"row {row.keys}: weights must be positive, one per panel")
        if not 0 < row.width_fraction <= 1:
            raise LayoutError(f"row {row.keys}: width_fraction must be in (0, 1]")

    for panel in layout.panels:
        path = resolve(layout, panel)
        if not path.is_file():
            raise LayoutError(f"panel {panel.key!r}: input file not found: {panel.file}")
        if path.suffix.lower() != ".pdf" and path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise LayoutError(f"panel {panel.key!r}: unsupported file type {path.suffix!r}")
        if not panel.letter_x or any(not 0 <= x < 1 for x in panel.letter_x):
            raise LayoutError(f"panel {panel.key!r}: letter_x must be fractions in [0, 1)")
        if panel.header_inset and not panel.title:
            raise LayoutError(f"panel {panel.key!r}: header_inset with no title to inset")
        if panel.crop_top < 0:
            raise LayoutError(f"panel {panel.key!r}: crop_top must be >= 0")
        if not 0 <= panel.crop_top_frac < 1:
            raise LayoutError(f"panel {panel.key!r}: crop_top_frac must be in [0, 1)")
        for name in ("crop_left_frac", "crop_right_frac"):
            if not 0 <= getattr(panel, name) < 1:
                raise LayoutError(f"panel {panel.key!r}: {name} must be in [0, 1)")
        if panel.crop_left_frac + panel.crop_right_frac >= 1:
            raise LayoutError(
                f"panel {panel.key!r}: the side crops remove the whole panel "
                f"({panel.crop_left_frac} + {panel.crop_right_frac})"
            )
        if panel.tagline and not panel.title:
            raise LayoutError(f"panel {panel.key!r}: a tagline needs a title above it")

    slots = sum(len(p.letter_x) for p in layout.ordered_panels())
    if layout.letters is not None and len(layout.letters) != slots:
        raise LayoutError(
            f"letter/panel count mismatch: {len(layout.letters)} letters "
            f"{layout.letters} for {slots} letter slots"
        )
    try:
        page_w = layout.page_width_pt
    except (TypeError, ValueError):
        raise LayoutError(f"width must be 'double', 'single' or a number of points, got {layout.width!r}") from None
    if not page_w > 0:
        raise LayoutError("width must be positive")
    if not layout.gap_pt >= 0:
        raise LayoutError(f"gap_pt must be >= 0, got {layout.gap_pt}")
    for row in layout.rows:
        if layout.gap_pt * (len(row.keys) - 1) >= page_w * row.width_fraction:
            raise LayoutError(f"row {row.keys}: the gaps alone are wider than the row")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _natural_size(layout: Layout, panel: Panel) -> tuple[float, float]:
    """Width and height of a panel's *content* (points for PDF, pixels for images).

    Both crops are applied here, so the aspect ratio the layout works from is the one
    actually drawn and a cropped panel is not stretched to fill its old shape.
    """
    path = resolve(layout, panel)
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            rect = doc[0].rect
        full_w, full_h = rect.width, rect.height
    else:
        with Image.open(path) as img:
            full_w, full_h = float(img.width), float(img.height)
    height = full_h - panel.crop_top - panel.crop_top_frac * full_h
    width = full_w * (1.0 - panel.crop_left_frac - panel.crop_right_frac)
    if height <= 0 or width <= 0:
        raise LayoutError(f"panel {panel.key!r}: the crop removes the whole panel")
    return width, height


def header_height(layout: Layout, panel: Panel) -> float:
    """Vertical space this panel's drawn header needs, in points. Zero if it has none.

    An inset header also reserves nothing: it is drawn over the panel's own top-left
    corner, so it costs page height only where the artwork would have been.
    """
    if not panel.title or panel.header_inset:
        return 0.0
    h = layout.letter_margin_pt + layout.title_pt * 1.12
    if panel.tagline:
        h += layout.tagline_pt * 1.30
    return h + layout.header_gap_pt


def plan(layout: Layout) -> tuple[float, dict[str, fitz.Rect]]:
    """Return the page height and each panel's *cell* rectangle on the page.

    A cell is the drawn header plus the panel content beneath it. Aspect ratios are
    honoured on the content, so adding a header makes the page taller rather than
    squashing the plot. Within a row every panel's content shares one top edge, taken
    from the tallest header in the row, so the plots line up even if one panel has a
    tagline and another does not. Use :func:`content_rect` to get the drawing area.
    """
    _require_pymupdf()
    validate(layout)
    page_w = layout.page_width_pt
    rects: dict[str, fitz.Rect] = {}
    y = 0.0
    for row in layout.rows:
        aspects = []
        for key in row.keys:
            w, h = _natural_size(layout, layout.panel(key))
            aspects.append(w / h)
        n = len(row.keys)
        row_w = page_w * row.width_fraction
        usable = row_w - layout.gap_pt * (n - 1)
        row_header = max(header_height(layout, layout.panel(k)) for k in row.keys)
        if row.weights is None:
            # Equal heights: widths follow the aspect ratios and fill the row.
            height = usable / sum(aspects)
            widths = [a * height for a in aspects]
            heights = [height] * n
        else:
            total = sum(row.weights)
            widths = [usable * wt / total for wt in row.weights]
            heights = [w / a for w, a in zip(widths, aspects)]
            height = max(heights)
        x = (page_w - row_w) / 2
        for key, w, h in zip(row.keys, widths, heights):
            rects[key] = fitz.Rect(x, y, x + w, y + row_header + h)
            x += w + layout.gap_pt
        y += row_header + height
    return y, rects


def content_rect(layout: Layout, panel: Panel, cell: fitz.Rect, row_header: float) -> fitz.Rect:
    """The part of a cell the panel file is drawn into, below any header."""
    return fitz.Rect(cell.x0, cell.y0 + row_header, cell.x1, cell.y1)


def _row_headers(layout: Layout) -> dict[str, float]:
    """The header band height applied to each panel, shared across its row."""
    out = {}
    for row in layout.rows:
        tallest = max(header_height(layout, layout.panel(k)) for k in row.keys)
        for key in row.keys:
            out[key] = tallest
    return out


def letter_positions(layout: Layout, rects: dict[str, fitz.Rect]) -> list[tuple[str, fitz.Point]]:
    """Letter and baseline origin for each slot, derived from the panel rectangles."""
    slots = [(p, fx) for p in layout.ordered_panels() for fx in p.letter_x]
    letters = layout.letters or [chr(ord("A") + i) for i in range(len(slots))]
    out = []
    for letter, (panel, frac) in zip(letters, slots):
        rect = rects[panel.key]
        x = rect.x0 + frac * rect.width + layout.letter_margin_pt
        y = rect.y0 + layout.letter_margin_pt + _CAP_HEIGHT * layout.letter_pt
        out.append((letter, fitz.Point(x, y)))
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _bold_font_file() -> Path | None:
    """DejaVu Sans Bold, which ships inside matplotlib (and is the house font)."""
    candidate = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return candidate if candidate.is_file() else None


def _cropped_png(path: Path, panel: Panel) -> bytes:
    """The image with its top crop removed, as PNG bytes.

    Done with Pillow rather than ``fitz.Pixmap``: building a sub-pixmap by hand got the
    origin and the copy rectangle into different coordinate systems and silently produced
    a smeared, wrongly shaped panel rather than an error.
    """
    with Image.open(path) as img:
        cut = int(round(panel.crop_top + panel.crop_top_frac * img.height))
        left = int(round(panel.crop_left_frac * img.width))
        right = img.width - int(round(panel.crop_right_frac * img.width))
        out = (img.crop((left, cut, right, img.height))
               if (cut or left or right != img.width) else img.copy())
        if out.mode not in ("RGB", "RGBA", "L"):
            out = out.convert("RGB")
        buf = BytesIO()
        out.save(buf, format="PNG")
    return buf.getvalue()


def _regular_font_file() -> Path | None:
    """DejaVu Sans, the house body font. Panel titles are regular weight, like matplotlib's."""
    candidate = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return candidate if candidate.is_file() else None


def _rgb(hex_colour: str) -> tuple[float, float, float]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _ink_rgb() -> tuple[float, float, float]:
    return _rgb(style.INK)


def _header_text_width(text: str, size: float) -> float:
    """Width of a header string in points, in the font the header is drawn with."""
    font_file = _regular_font_file()
    font = fitz.Font(fontfile=str(font_file)) if font_file else fitz.Font("helv")
    return font.text_length(text, fontsize=size)


def _draw_header(page, layout: Layout, panel: Panel, cell: fitz.Rect) -> None:
    """Left-aligned title with a smaller muted tagline, matching ``style.panel_title``.

    A header that does not fit raises rather than being drawn off the edge. The panel this
    feature exists to fix was a frozen image whose own title ran past the right edge and was
    saved truncated mid-word, and ``insert_text`` would reproduce that failure silently.
    """
    if not panel.title:
        return
    font_file = _regular_font_file()
    kwargs = ({"fontname": "dejavu", "fontfile": str(font_file)} if font_file
              else {"fontname": "helv"})
    # Start clear of the panel letter and align the title's cap top with the letter's,
    # which is what the letter's own baseline formula does.
    x = cell.x0 + layout.letter_margin_pt + layout.letter_pt + 2.0
    available = cell.x1 - x - HEADER_RIGHT_MARGIN_PT
    for text, size in ((panel.title, layout.title_pt), (panel.tagline, layout.tagline_pt)):
        if not text:
            continue
        over = _header_text_width(text, size) - available
        if over > 0:
            raise LayoutError(
                f"panel {panel.key!r}: header text is {over:.1f} pt too wide for the panel "
                f"({available:.1f} pt available at {size:g} pt): {text!r}. Shorten it, or "
                f"lower title_pt/tagline_pt in the layout."
            )
    baseline = cell.y0 + layout.letter_margin_pt + _CAP_HEIGHT * layout.title_pt
    page.insert_text(fitz.Point(x, baseline), panel.title,
                     fontsize=layout.title_pt, color=_ink_rgb(), **kwargs)
    if panel.tagline:
        page.insert_text(fitz.Point(x, baseline + layout.title_pt * 1.12), panel.tagline,
                         fontsize=layout.tagline_pt, color=_rgb(style.INK_MUTED), **kwargs)


def compose(layout: Layout) -> fitz.Document:
    """Build the composite as a one-page PDF document (vector where inputs are)."""
    page_h, rects = plan(layout)
    doc = fitz.open()
    page = doc.new_page(width=layout.page_width_pt, height=page_h)

    headers = _row_headers(layout)
    for panel in layout.ordered_panels():
        cell = rects[panel.key]
        rect = content_rect(layout, panel, cell, headers[panel.key])
        path = resolve(layout, panel)
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as src:
                src_rect = src[0].rect
                cut = panel.crop_top + panel.crop_top_frac * src_rect.height
                clip = fitz.Rect(src_rect.x0 + panel.crop_left_frac * src_rect.width,
                                 src_rect.y0 + cut,
                                 src_rect.x1 - panel.crop_right_frac * src_rect.width,
                                 src_rect.y1)
                page.show_pdf_page(rect, src, 0, clip=clip)
        else:
            page.insert_image(rect, stream=_cropped_png(path, panel))
        _draw_header(page, layout, panel, cell)

    font_file = _bold_font_file()
    kwargs = {"fontname": _FONT_NAME, "fontfile": str(font_file)} if font_file else {"fontname": "hebo"}
    for letter, origin in letter_positions(layout, rects):
        page.insert_text(origin, letter, fontsize=layout.letter_pt, color=_ink_rgb(), **kwargs)
    return doc


def render(layout: Layout, out_dir: str | Path) -> list[Path]:
    """Write ``<name>.pdf`` and ``<name>.png`` (at ``style.DPI``) into ``out_dir``."""
    _require_pymupdf()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = compose(layout)
    try:
        doc.subset_fonts()  # embed only the glyphs used; needs fontTools, skip if absent
    except Exception:  # noqa: BLE001 - subsetting is an optimisation, not a requirement
        pass
    pdf_path = out_dir / f"{layout.name}.pdf"
    png_path = out_dir / f"{layout.name}.png"
    doc.save(pdf_path, deflate=True, garbage=3)
    doc[0].get_pixmap(dpi=style.DPI, alpha=False).save(png_path)
    doc.close()
    return [pdf_path, png_path]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layout", required=True, help="layout YAML, e.g. guides/composites/fig01.yaml")
    parser.add_argument("--out", default="output", help="output directory (default: output/)")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="directory the layout's panel paths are relative to (default: the repository). "
                             "tools/build_figures.py points it at a staging tree so a scratch build places "
                             "its own panels")
    args = parser.parse_args(argv)
    _require_pymupdf()
    layout = load_layout(args.layout, root=args.root)
    paths = render(layout, args.out)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
