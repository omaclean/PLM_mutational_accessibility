"""Guide files: the one place that says what an analysis run is made of.

A guide names the lineages, their root sequences, the models being compared and
the derived tables a run produced. Every figure script takes ``--guide`` and draws
what it finds, so nothing below ``figures/`` may hard-code a lineage set, a model
name, a virus or a count. Whether a given lineage set is *scientifically sound* is
a separate question (see ``LINEAGE_VALIDITY.md``); this module only checks that a
guide is well formed and that its files exist.

Paths in a guide are relative. ``root`` and ``alignment`` resolve against the
repository root, table names against ``results_dir``. Absolute paths and ``..``
are rejected, because the old guides pointed into another repository and could
not run anywhere else.

A guide is YAML::

    name: my_run
    virus: influenza A/H3N2 HA
    reference: A/Darwin/6/2021
    results_dir: results/my_run
    models:
      - {key: raw_model, label: Base model (ESM-2), role: base}
    lineages:
      - {id: C.1, root: data/roots/C.1.nt.fa, root_type: reconstructed}
    tables:
      alpha_sweep_by_lineage: alpha_sweep_by_lineage.tsv.gz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import yaml

from mutacc.style import ROLE_COLOURS

__all__ = [
    "REPO_ROOT", "GuideError", "Guide", "Lineage", "Model",
    "load_guide", "parse_figure_args", "read_fasta_headers", "root_header_problems",
]

#: The clean_repo directory. Every relative path in a guide resolves from here.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where a script that draws one cell of a composite writes its render.
#:
#: These are intermediates: ``panelise`` reads them, places them in the composite, and nothing
#: else ever looks at them again. They used to be written beside the finished figures in
#: ``output/``, which left four fig02 panels and one fig01 panel sitting next to the two
#: composites built from them -- five files a reader of ``output/`` has to know to ignore.
#: ``build/`` is gitignored, so they are now rebuilt rather than published, and
#: ``tools/build_figures.py`` clears this directory after a full build.
#:
#: What does NOT move: each panel's ``_values.tsv`` and ``_caption.md``. Those are the data
#: record a rebuild is checked against (``tools/build_figures.py``, ``check_build``), so they
#: stay under version control in ``output/`` with every other figure's.
PANELS_DIR = REPO_ROOT / "build" / "panels"

ROOT_TYPES = ("reconstructed", "observed", "unverified")

_TOP_KEYS = {
    "name", "description", "virus", "reference", "results_dir",
    "corpus_cutoff", "models", "lineages", "tables", "notes", "primary_lineage",
}
_REQUIRED_TOP = {"name", "virus", "reference", "results_dir", "models", "lineages", "tables"}
_LINEAGE_KEYS = {"id", "root", "root_type", "alignment"}
_MODEL_KEYS = {"key", "label", "role"}


class GuideError(ValueError):
    """A guide is malformed. The message names the file and the field."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.safe_load`` that refuses a mapping with a repeated key.

    PyYAML keeps the last of two equal keys without a word, so a guide with two ``lineages:``
    blocks (or two entries of one table name) would silently drop the first.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                dup = key in seen
            except TypeError:       # an unhashable key; the parent raises a clearer error
                break
            if dup:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep)


@dataclass(frozen=True)
class Lineage:
    id: str
    root: Path
    root_type: str
    alignment: Path | None = None


@dataclass(frozen=True)
class Model:
    key: str      # value of the ``model`` column in the results tables
    label: str    # what a reader sees; never a checkpoint id
    role: str     # a key of ``mutacc.style.ROLE_COLOURS``


@dataclass(frozen=True)
class Guide:
    name: str
    source: Path
    virus: str
    reference: str
    results_dir: Path
    models: tuple[Model, ...]
    lineages: tuple[Lineage, ...]
    tables: Mapping[str, Path]
    description: str = ""
    corpus_cutoff: str | None = None
    notes: str = ""
    #: The one lineage the single-lineage figures analyse, when the set is not to choose it
    #: itself. ``mutacc.strata.primary_lineage`` falls back to its own rule when this is None.
    primary_lineage: str | None = None
    warnings: tuple[str, ...] = field(default=())

    @property
    def n_lineages(self) -> int:
        """Counted from the guide, never typed into a figure."""
        return len(self.lineages)

    @property
    def lineage_ids(self) -> list[str]:
        return [lin.id for lin in self.lineages]

    def model(self, key: str) -> Model:
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(f"{self.source.name}: no model {key!r}; has {[m.key for m in self.models]}")

    def models_with_role(self, role: str) -> list[Model]:
        return [m for m in self.models if m.role == role]

    def table_path(self, name: str, results: Path | None = None) -> Path:
        """Path of a named table, with a message that lists what exists."""
        try:
            rel = self.tables[name]
        except KeyError:
            raise KeyError(
                f"{self.source.name}: no table {name!r}; this guide declares {sorted(self.tables)}"
            ) from None
        return (Path(results) if results is not None else self.results_dir) / rel

    def read_table(self, name: str, results: Path | None = None, **kwargs) -> pd.DataFrame:
        """Read a named table. Empty tables raise: an empty join must not become a blank panel."""
        path = self.table_path(name, results)
        sep = "," if ".csv" in path.suffixes else "\t"
        frame = pd.read_csv(path, sep=sep, **kwargs)
        if frame.empty:
            raise GuideError(f"table {name!r} ({path.name}) is empty")
        return frame

    def check_lineages(self, observed: Iterable[str], where: str) -> None:
        """Raise unless a table holds exactly the guide's lineages.

        Figure titles and captions quote ``n_lineages`` from the guide; a table with more or
        fewer lineages would make them claim a count the data do not have.
        """
        got, want = {str(x) for x in observed}, set(self.lineage_ids)
        if got != want:
            raise GuideError(
                f"{where}: lineages differ from the guide {self.source.name!r}: "
                f"only in table {sorted(got - want)}, only in guide {sorted(want - got)}"
            )

    def missing_files(self, results: Path | None = None) -> list[Path]:
        """Every file the guide names that is not on disk."""
        wanted = [self.table_path(t, results) for t in self.tables]
        for lin in self.lineages:
            wanted.append(lin.root)
        return [p for p in wanted if not p.is_file()]


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def _fail(source: Path, message: str) -> GuideError:
    return GuideError(f"{source}: {message}")


def _relative(source: Path, where: str, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise _fail(source, f"{where} must be a non-empty string path")
    path = Path(value)
    if path.is_absolute() or value.startswith("~"):
        raise _fail(source, f"{where} is absolute ({value!r}); guides use repo-relative paths only")
    if ".." in path.parts:
        raise _fail(source, f"{where} contains '..' ({value!r}); paths must stay inside the repo")
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise _fail(source, f"{where} ({value!r}) resolves outside the repository (a symlink?)")
    return path


def _check_keys(source: Path, where: str, got: Iterable[str], allowed: set[str], required: set[str]) -> None:
    got = set(got)
    unknown = got - allowed
    if unknown:
        raise _fail(source, f"{where}: unknown key(s) {sorted(map(str, unknown))}; allowed {sorted(allowed)}")
    missing = required - got
    if missing:
        raise _fail(source, f"{where}: missing required key(s) {sorted(missing)}")


def load_guide(path: str | Path) -> Guide:
    """Load and validate a guide. Fails loudly; never guesses a default."""
    source = Path(path)
    if not source.is_file():
        raise GuideError(f"guide not found: {source}")
    try:
        raw = yaml.load(source.read_text(), Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise _fail(source, f"not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise _fail(source, "top level must be a mapping")

    _check_keys(source, "guide", raw, _TOP_KEYS, _REQUIRED_TOP)

    for key in ("name", "virus", "reference"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise _fail(source, f"{key} must be a non-empty string")

    results_dir = REPO_ROOT / _relative(source, "results_dir", raw["results_dir"])

    models = _load_models(source, raw["models"])
    lineages = _load_lineages(source, raw["lineages"])

    tables_raw = raw["tables"]
    if not isinstance(tables_raw, dict) or not tables_raw:
        raise _fail(source, "tables must be a non-empty mapping of name -> file")
    tables = {
        str(k): _relative(source, f"tables.{k}", v) for k, v in tables_raw.items()
    }

    cutoff = raw.get("corpus_cutoff")
    if cutoff is not None:
        cutoff = str(cutoff)
        try:
            pd.Timestamp(cutoff)
        except ValueError:
            raise _fail(source, f"corpus_cutoff {cutoff!r} is not a date") from None

    primary = raw.get("primary_lineage")
    if primary is not None:
        primary = str(primary)
        known = {lin.id: lin for lin in lineages}
        if primary not in known:
            raise _fail(source, f"primary_lineage {primary!r} is not one of the lineages "
                                f"{sorted(known)}")
        if known[primary].root_type == "unverified":
            raise _fail(source, f"primary_lineage {primary!r} has an unverified root; the "
                                "single-lineage figures rest on it, so it needs a known provenance")

    warnings = tuple(
        f"lineage {lin.id}: root_type is 'unverified'" for lin in lineages if lin.root_type == "unverified"
    )

    return Guide(
        name=raw["name"],
        source=source,
        primary_lineage=primary,
        virus=raw["virus"],
        reference=raw["reference"],
        results_dir=results_dir,
        models=models,
        lineages=lineages,
        tables=tables,
        description=str(raw.get("description", "")).strip(),
        corpus_cutoff=cutoff,
        notes=str(raw.get("notes", "")).strip(),
        warnings=warnings,
    )


def _load_models(source: Path, raw: object) -> tuple[Model, ...]:
    if not isinstance(raw, list) or not raw:
        raise _fail(source, "models must be a non-empty list")
    out, seen = [], set()
    for i, item in enumerate(raw):
        where = f"models[{i}]"
        if not isinstance(item, dict):
            raise _fail(source, f"{where} must be a mapping")
        _check_keys(source, where, item, _MODEL_KEYS, _MODEL_KEYS)
        for field_name in _MODEL_KEYS:
            if not isinstance(item[field_name], str) or not item[field_name].strip():
                raise _fail(source, f"{where}.{field_name} must be a non-empty string, got {item[field_name]!r}")
        if item["role"] not in ROLE_COLOURS:
            raise _fail(source, f"{where}.role {item['role']!r} unknown; known roles {sorted(ROLE_COLOURS)}")
        if item["key"] in seen:
            raise _fail(source, f"{where}: duplicate model key {item['key']!r}")
        seen.add(item["key"])
        out.append(Model(key=str(item["key"]), label=str(item["label"]), role=item["role"]))
    return tuple(out)


def _load_lineages(source: Path, raw: object) -> tuple[Lineage, ...]:
    if not isinstance(raw, list) or not raw:
        raise _fail(source, "lineages must be a non-empty list")
    out, seen = [], set()
    for i, item in enumerate(raw):
        where = f"lineages[{i}]"
        if not isinstance(item, dict):
            raise _fail(source, f"{where} must be a mapping")
        _check_keys(source, where, item, _LINEAGE_KEYS, {"id", "root", "root_type"})
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise _fail(source, f"{where}.id must be a non-empty string (quote it in YAML), got {item['id']!r}")
        lid = item["id"]
        if lid in seen:
            raise _fail(source, f"{where}: duplicate lineage id {lid!r}")
        seen.add(lid)
        if not isinstance(item["root_type"], str) or item["root_type"] not in ROOT_TYPES:
            raise _fail(source, f"{where}.root_type {item['root_type']!r} must be one of {ROOT_TYPES}")
        aln = item.get("alignment")
        out.append(Lineage(
            id=lid,
            root=REPO_ROOT / _relative(source, f"{where}.root", item["root"]),
            root_type=item["root_type"],
            alignment=None if aln is None else REPO_ROOT / _relative(source, f"{where}.alignment", aln),
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Root provenance checks (used by tests/test_data_integrity.py)
# ---------------------------------------------------------------------------

def read_fasta_headers(path: Path) -> list[str]:
    """Header lines of a FASTA, without the leading '>'."""
    return [ln[1:].strip() for ln in Path(path).read_text().splitlines() if ln.startswith(">")]


def root_header_problems(guide: Guide) -> list[str]:
    """Disagreements between a guide and the root FASTA headers it points at.

    Two checks, both born from the 9 Oct lineage-root audit:

    * the lineage named in the header must be the lineage the guide files it under
      (``K.nt.fa`` once said ``J.2.4.1``);
    * a root the guide calls ``reconstructed`` must not carry an accession. A header
      that looks like a tip accession on an inferred sequence misleads the reader.

    The label is read from the *last* ``|`` field, or the first when the header
    marks itself ``nextstrain_clade_root``.
    """
    problems = []
    for lin in guide.lineages:
        headers = read_fasta_headers(lin.root)
        if len(headers) != 1:
            problems.append(f"{lin.id}: root FASTA has {len(headers)} records, expected 1")
            continue
        fields = headers[0].split("|")
        label = fields[0] if "nextstrain_clade_root" in fields else fields[-1]
        if label != lin.id and label != lin.id.replace("_int", ""):
            problems.append(f"{lin.id}: header names lineage {label!r} ({headers[0][:60]})")
        has_accession = any(f.startswith(("EPI", "EPI_ISL")) for f in fields)
        if lin.root_type == "reconstructed" and has_accession:
            problems.append(f"{lin.id}: guide says reconstructed but the header carries an accession")
    return problems


# ---------------------------------------------------------------------------
# Command line shared by every figure script
# ---------------------------------------------------------------------------

def parse_figure_args(description: str, argv: Sequence[str] | None = None, *,
                      extra=None) -> argparse.Namespace:
    """``--guide G [--results R] [--out O] [--panels-out P]``, guide loaded as ``args.guide``.

    ``extra`` is called with the parser before it reads ``argv``, for a figure that takes an
    option of its own (``--scoring``, say). It keeps such a figure on the shared parser rather
    than having it rebuild the four common arguments and drift from them.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--guide", required=True, type=Path, help="guide YAML file")
    parser.add_argument("--results", type=Path, default=None,
                        help="results directory (default: the guide's results_dir)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output",
                        help="output directory for PDF + PNG")
    parser.add_argument("--panels-out", type=Path, default=PANELS_DIR,
                        help="where a script that draws a composite panel writes its PDF + PNG "
                             f"(default: {PANELS_DIR.relative_to(REPO_ROOT).as_posix()}/)")
    if extra is not None:
        extra(parser)
    args = parser.parse_args(argv)
    args.guide = load_guide(args.guide)
    if args.results is None:
        args.results = args.guide.results_dir
    return args
