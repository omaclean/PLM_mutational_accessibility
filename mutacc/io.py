"""Companion outputs written next to every figure.

A figure is never written alone. It comes with the table of values it plots (so a
golden-value test can pin the numbers, and a reader can check a point) and a
caption in plain words (so methods prose lives in the caption, not as a grey
footnote inside the image).
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pandas as pd

from mutacc.numbering import load_numbering

__all__ = ["write_values", "write_caption", "caption_label"]

#: A caption that opens by naming itself. The number a figure carries is the
#: manuscript's and lives in ``guides/manuscript/figure_numbers.yaml``; a literal
#: here is how "Figure S18" survived the draft moving that figure (now Figure SL).
_SELF_NAMED = re.compile(r"^\s*(\*\*)?\s*(supplementary\s+)?(figure|table)\b[^.]*\.", re.I)


def write_values(frame: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    """Write the plotted values as ``<stem>_values.tsv``. Refuses an empty frame."""
    if frame.empty:
        raise ValueError(f"{stem}: refusing to write an empty values table")
    path = Path(out_dir) / f"{stem}_values.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, float_format="%.6g")
    return path


def caption_label(stem: str, panel: str | None = None) -> str:
    """``'Figure SH'`` for a stem, ``'Figure 2D'`` for panel D of it.

    Supplementary labels are letters and main-text ones digits; both come from the map
    and neither is ever typed into caption text.

    Read from the numbering map, never typed. A stem the map does not carry raises,
    which is the point: a new figure has no manuscript number until it is given one
    in one place.
    """
    return f"{load_numbering().label_for(stem)}{panel or ''}"


def write_caption(text: str, out_dir: Path, stem: str, *, panel: str | None = None,
                  banner: str | None = None) -> Path:
    """Write ``<stem>_caption.md``, numbered from the map, dedented and wrapped to 90 columns.

    The caption opens with the manuscript label -- "**Figure SH. ...**" -- taken from
    ``guides/manuscript/figure_numbers.yaml``. The caption *text* must therefore not name
    itself, and a text that does raises: that is how the figures came to carry three
    different self-descriptions ("Supplementary figure.", "Figure S18.", "Supplementary
    Figure S16.") and two numbers the draft had since moved.

    ``panel`` appends a panel letter for a figure drawn as one panel of a composite
    ("Figure 2D"). ``banner`` is a warning paragraph that must come *before* the label --
    ``make_sK_branch_codons.py``'s "SYNTHETIC FIXTURE ... do not quote them", which has to be the first
    thing a reader sees and would be buried by a figure number in front of it.
    Hyphens are never broken on: the default would split "virus-specific"
    or "SARS-CoV-2" across lines whenever one lands at the right column, which reads as a
    typo and survives into anything that re-joins the text.
    """
    body = textwrap.dedent(text).strip()
    if _SELF_NAMED.match(body):
        opening = body[:60].replace("\n", " ")
        raise ValueError(
            f"{stem}: the caption text names its own number ({opening!r}). The number comes "
            f"from guides/manuscript/figure_numbers.yaml -- drop it from the text."
        )
    # A retired figure has no manuscript number and must not be given one; its
    # caption opens with its title, like any unnumbered panel.
    if not load_numbering()[stem].retired:
        label = caption_label(stem, panel)
        body = f"**{label}. {body[2:]}" if body.startswith("**") else f"{label}. {body}"
    if banner:
        body = f"{textwrap.dedent(banner).strip()}\n\n{body}"
    path = Path(out_dir) / f"{stem}_caption.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = [textwrap.fill(" ".join(p.split()), 90, break_on_hyphens=False) for p in body.split("\n\n")]
    path.write_text("\n\n".join(paragraphs) + "\n")
    return path
