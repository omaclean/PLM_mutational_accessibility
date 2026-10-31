"""Shared pieces for the exporters: the Export record and the MANIFEST writer.

Every table under ``results/`` carries a MANIFEST row saying where it came from, so a
shipped number can always be traced back. The writer lives here because three exporters
need it and two had grown their own copy, which is how two manifests end up with
different ideas of what a row is.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

__all__ = ["Export", "MANIFEST_COLUMNS", "Writer", "append_manifest", "sha256"]


@dataclass(frozen=True)
class Export:
    set_name: str
    source: str                 # relative to the exploration repo root
    dest: str                   # file name under results/<set_name>/
    drop_columns: tuple[str, ...] = field(default=())
    why: str = ""
    #: Optional frame -> frame hook, run after drop_columns (e.g. to keep only the
    #: columns a figure reads). Must be deterministic.
    transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


MANIFEST_COLUMNS = ["file", "source", "source_sha256_16", "rows", "columns", "dropped", "used_for"]


def sha256(path: Path) -> str:
    """First 16 hex digits of a file's SHA-256, enough to spot a changed source."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Writer:
    """Write gzipped TSVs under one results directory and collect their MANIFEST rows."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def add(self, name: str, frame: pd.DataFrame, source: str, why: str,
            sha: str = "", dropped: str = "") -> None:
        if frame.empty:
            raise ValueError(f"{name}: refusing to write an empty table")
        frame.to_csv(self.directory / name, sep="\t", index=False,
                     compression="gzip", float_format="%.12g")
        self.rows.append({"file": name, "source": source, "source_sha256_16": sha,
                          "rows": len(frame), "columns": frame.shape[1],
                          "dropped": dropped, "used_for": why})
        print(f"{self.directory.name}/{name}: {len(frame)} rows x {frame.shape[1]} cols")

    def write_manifest(self) -> None:
        append_manifest(self.directory / "MANIFEST.tsv", self.rows)


def append_manifest(path: Path, rows: list[dict]) -> None:
    """Add rows to a MANIFEST in a single write.

    Rows for files already listed are replaced, which is a re-run and the only case that
    rewrites the file. Otherwise the rows are appended, so an exporter never clobbers rows
    another exporter added for its own tables.
    """
    path = Path(path)
    if not rows:
        return
    for row in rows:
        missing = set(MANIFEST_COLUMNS) - set(row)
        if missing:
            raise ValueError(f"manifest row for {row.get('file')!r} is missing {sorted(missing)}")
        for column, value in row.items():
            # A tab or newline in a prose field would silently shift every later column.
            if isinstance(value, str) and ("\t" in value or "\n" in value):
                raise ValueError(f"manifest field {column!r} for {row['file']!r} contains a tab or newline")

    new = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).astype(str)
    text = new.to_csv(sep="\t", index=False, header=False)
    if not path.exists():
        path.write_text("\t".join(MANIFEST_COLUMNS) + "\n" + text)
        return

    old = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if old["file"].isin(new["file"]).any():
        keep = old.loc[~old["file"].isin(new["file"])]
        merged = pd.concat([keep, new], ignore_index=True).sort_values("file")
        tmp = path.with_suffix(".tmp")
        merged.to_csv(tmp, sep="\t", index=False)
        os.replace(tmp, path)
        return
    with path.open("a") as handle:   # one append; never rewrites rows other exporters added
        handle.write(("" if path.read_bytes().endswith(b"\n") else "\n") + text)
