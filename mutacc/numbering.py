"""Figure numbering: the manuscript label for a content stem.

A figure is identified everywhere in this repository by its *stem* -- the name its
output files carry (``sH_contact_recovery``). What the manuscript calls it is a
separate, far less stable thing, and it lives in one file,
``guides/manuscript/figure_numbers.yaml``. Nothing else may hard-code a figure number, so
re-ordering the supplement edits that file and nothing else.

Three kinds of entry::

    numbered    has a ``label``: "Figure 3", "Figure SH".
    supporting  has a ``parent``: a panel or a side table of a numbered figure,
                and takes that figure's label.
    retired     ``retired: true``: the code is kept, the manuscript slot is gone.

Main-text figures are numbered ("Figure 3"); supplementary ones are *lettered*
("Figure SH", "Table SZ"). The letter is a name rather than a position, so the
supplement can be re-ordered in the draft without any caption, document or test
changing; the manuscript's final S1..SN run is applied once at submission, from
the order of first citation. ``number`` still returns an integer for either
form -- A is 1, O is 15 -- so ordering and the no-gaps check are unchanged.

Reading it::

    from mutacc.numbering import load_numbering

    numbering = load_numbering()
    numbering.label_for("sH_contact_recovery")   # 'Figure SH'
    numbering.caption_prefix("sH_contact_recovery")  # 'Figure SH.'

``label_for`` raises on a stem the map does not know, which is deliberate: a
figure script that is not in the map has no number, and that must be noticed when
it is written rather than when the manuscript is assembled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from mutacc.guide import REPO_ROOT

__all__ = [
    "NUMBERING_PATH", "NumberingError", "FigureNumber", "Numbering", "load_numbering",
]

#: The map itself. One file, by design. It sits under ``guides/manuscript/``
#: rather than ``guides/`` because everything directly in ``guides/`` is globbed
#: and parsed as a *run* guide by ``mutacc.guide.load_guide``.
NUMBERING_PATH = REPO_ROOT / "guides" / "manuscript" / "figure_numbers.yaml"

#: Top-level sections, each holding ``stem: {...}``.
SECTIONS = ("main", "supplementary")

_ENTRY_KEYS = {"label", "parent", "retired", "note"}
_MAIN_LABEL = re.compile(r"^Figure (\d+)$")
#: Supplementary figures are *lettered*, not numbered: "Figure SA" .. "Figure SO". See the
#: module docstring for why, and ``guides/manuscript/figure_numbers.yaml`` for the set.
_SUPP_LABEL = re.compile(r"^Figure S([A-Z])$")
#: Supplementary *tables*. A stem that the draft presents as a table rather than a figure
#: carries this instead, and takes a letter of its own: making a table must not leave a hole
#: in the figure letters, and the two sequences are independent in the draft.
_SUPP_TABLE_LABEL = re.compile(r"^Table S([A-Z])$")
#: label pattern -> what the entry is, per section.
_PATTERNS = {
    "main": (("figure", _MAIN_LABEL),),
    "supplementary": (("figure", _SUPP_LABEL), ("table", _SUPP_TABLE_LABEL)),
}
#: Sequences held to a 1..N run with no gaps. A dropped figure must be noticed, so the
#: figure letters run A..O unbroken. The supplementary *tables* are left out on purpose:
#: their letters name them rather than order them (Z is parked at the end of the alphabet
#: so the one shipped table does not move whenever a figure is added), and holding one
#: table to "must be A" would only force a rename for no reader's benefit.
_CONTIGUOUS = {("main", "figure"), ("supplementary", "figure")}


def _ordinal(token: str) -> int:
    """The position a label's token names: ``"3"`` -> 3, ``"A"`` -> 1, ``"O"`` -> 15."""
    return int(token) if token.isdigit() else ord(token) - ord("A") + 1


class NumberingError(ValueError):
    """The map is malformed, or a stem was asked for that it does not carry."""


@dataclass(frozen=True)
class FigureNumber:
    """One entry. Exactly one of ``label``, ``parent`` and ``retired`` is set."""

    stem: str
    section: str
    label: str | None = None
    parent: str | None = None
    retired: bool = False
    note: str | None = None

    @property
    def is_numbered(self) -> bool:
        return self.label is not None

    @property
    def is_supporting(self) -> bool:
        return self.parent is not None

    @property
    def kind(self) -> str | None:
        """``"figure"`` or ``"table"``, from the shape of the label."""
        if self.label is None:
            return None
        for kind, pattern in _PATTERNS.get(self.section, ()):
            if pattern.match(self.label):
                return kind
        return None

    @property
    def number(self) -> int | None:
        """Where the label sits: 3 for "Figure 3", 8 for "Figure SH", 26 for "Table SZ".

        Main-text labels carry a digit and supplementary ones a letter; both reduce to
        an integer here so that ordering, the duplicate check and the no-gaps check do
        not have to care which form a section uses.
        """
        if self.label is None:
            return None
        for _, pattern in _PATTERNS.get(self.section, ()):
            match = pattern.match(self.label)
            if match:
                return _ordinal(match.group(1))
        return None


@dataclass(frozen=True)
class Numbering:
    """The whole map, validated on load."""

    entries: Mapping[str, FigureNumber]
    path: Path

    def __getitem__(self, stem: str) -> FigureNumber:
        """The entry for a stem, or for the figure a derived stem is a variant of.

        A script may build a stem at run time for a *condition* of the same figure:
        ``sK_branch_codons_SYNTHETIC_FIXTURE`` is figure S11 drawn on planted data,
        ``sG_epistasis_null_wt`` is S7 under a different gauge. These are the same
        manuscript figure and take its number; they are not separate entries, or the
        map would have to list every gauge anyone ever passes.

        The fallback is unambiguous because ``test_output_provenance.py`` already
        forbids one declared stem from being a prefix of another.
        """
        if stem in self.entries:
            return self.entries[stem]
        variants = [s for s in self.entries if stem.startswith(s + "_")]
        if variants:
            return self.entries[max(variants, key=len)]
        raise NumberingError(
            f"{stem!r} is not in {self.path.name}. Every figure stem needs an entry "
            f"there: a label, a parent it supports, or retired: true."
        )

    def __contains__(self, stem: str) -> bool:
        """Exact entries only: used to ask whether the map *lists* a stem."""
        return stem in self.entries

    def label_for(self, stem: str) -> str:
        """The manuscript label for a stem, following ``parent`` for a panel.

        Raises for a retired stem: it has no number to print.
        """
        entry = self[stem]
        if entry.parent is not None:
            return self.label_for(entry.parent)
        if entry.retired:
            raise NumberingError(
                f"{stem!r} is retired ({entry.note or 'no reason recorded'}) and has no "
                f"manuscript number."
            )
        assert entry.label is not None  # guaranteed by _validate
        return entry.label

    def caption_prefix(self, stem: str) -> str:
        """``'Figure SH.'`` -- what a caption opens with."""
        return f"{self.label_for(stem)}."

    def stems(self, section: str | None = None) -> list[str]:
        return sorted(s for s, e in self.entries.items() if section in (None, e.section))

    def numbered(self, section: str | None = None) -> dict[str, FigureNumber]:
        """Stems that carry a label, in manuscript order."""
        chosen = [
            (e.number, s, e) for s, e in self.entries.items()
            if e.is_numbered and section in (None, e.section)
        ]
        return {s: e for _, s, e in sorted(chosen)}


def _validate(entries: dict[str, FigureNumber], path: Path) -> None:
    labels: dict[str, str] = {}
    for stem, entry in sorted(entries.items()):
        kinds = [k for k, v in (("label", entry.label), ("parent", entry.parent),
                                ("retired", entry.retired or None)) if v]
        if len(kinds) != 1:
            raise NumberingError(
                f"{path.name}: {stem} sets {kinds or 'nothing'}; it needs exactly one of "
                f"label, parent, retired."
            )
        if entry.parent is not None:
            if entry.parent not in entries:
                raise NumberingError(f"{path.name}: {stem} has parent {entry.parent!r}, which is not in the map")
            if entries[entry.parent].parent is not None:
                raise NumberingError(f"{path.name}: {stem}'s parent {entry.parent} is itself a panel; nest no deeper")
        if entry.label is not None:
            if entry.kind is None:
                shapes = " or ".join(
                    {"main": ["Figure N"], "supplementary": ["Figure SX", "Table SX"]}[entry.section]
                )
                raise NumberingError(
                    f"{path.name}: {stem}'s label {entry.label!r} is not of the form {shapes}"
                )
            if entry.label in labels:
                raise NumberingError(
                    f"{path.name}: {entry.label!r} is claimed by both {labels[entry.label]} and {stem}"
                )
            labels[entry.label] = stem

    # A gap means a figure was dropped without re-numbering, which is the exact
    # drift this file exists to stop.
    # Figures and tables are counted apart: moving a stem from the figure set to the table set
    # must close the hole it leaves behind, and must not be judged against the figures' run.
    # Supplementary tables are not held to a run at all (see ``_CONTIGUOUS``).
    for section in SECTIONS:
        for kind in ("figure", "table"):
            if (section, kind) not in _CONTIGUOUS:
                continue
            numbers = sorted(
                e.number for e in entries.values()
                if e.section == section and e.is_numbered and e.kind == kind
            )
            if not numbers:
                continue
            expected = list(range(1, len(numbers) + 1))
            if numbers != expected:
                run = "A..N" if section == "supplementary" else "1..N"
                raise NumberingError(
                    f"{path.name}: {section} {kind} numbering is {numbers}, expected "
                    f"{expected} -- it must run {run} with no gaps and no repeats."
                )


def load_numbering(path: Path | str | None = None) -> Numbering:
    """Read and validate ``guides/manuscript/figure_numbers.yaml`` (or another map)."""
    path = Path(path) if path is not None else NUMBERING_PATH
    if not path.exists():
        raise NumberingError(f"no figure numbering map at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise NumberingError(f"{path.name}: expected a mapping of sections, got {type(raw).__name__}")
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise NumberingError(f"{path.name}: unknown section(s) {sorted(unknown)}; expected {list(SECTIONS)}")

    entries: dict[str, FigureNumber] = {}
    for section in SECTIONS:
        block = raw.get(section) or {}
        if not isinstance(block, dict):
            raise NumberingError(f"{path.name}: section {section} must be a mapping of stem -> entry")
        for stem, body in block.items():
            body = body or {}
            if not isinstance(body, dict):
                raise NumberingError(f"{path.name}: entry {stem} must be a mapping, got {type(body).__name__}")
            extra = set(body) - _ENTRY_KEYS
            if extra:
                raise NumberingError(f"{path.name}: {stem} has unknown key(s) {sorted(extra)}")
            if stem in entries:
                raise NumberingError(f"{path.name}: {stem} appears in more than one section")
            entries[stem] = FigureNumber(
                stem=stem,
                section=section,
                label=body.get("label"),
                parent=body.get("parent"),
                retired=bool(body.get("retired", False)),
                note=body.get("note"),
            )

    _validate(entries, path)
    return Numbering(entries=entries, path=path)
