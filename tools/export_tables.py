"""Copy derived tables out of the exploration repo into ``results/`` as gzipped TSV.

This is the only place that knows where the exploration archive keeps things. It is
a one-off bridge, not part of the figure engine: figures read ``results/`` and
nothing else.

Output is gzipped TSV rather than parquet on purpose. Only one interpreter on the
authoring machine can read parquet and a different one runs the tests, so shipping
parquet would mean no single environment could both draw the figures and test
them. Gzipped TSV needs nothing beyond pandas.

Every table written gets a row in ``results/<set>/MANIFEST.tsv`` recording where
it came from (relative to the exploration repo), a hash of the source, and its
shape, so a shipped table can always be traced back.

    python tools/export_tables.py --source-repo ../  --set h3n2_13lineage
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_base import Export  # noqa: E402

CLEAN = Path(__file__).resolve().parents[1]


def _discover() -> list[Export]:
    """Collect EXPORTS from every tools/exports/*.py (one file per figure)."""
    found: list[Export] = []
    for path in sorted((Path(__file__).parent / "exports").glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"_exports_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        # dataclass() resolves annotations through sys.modules
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        found.extend(getattr(module, "EXPORTS", []))
    return found


EXPORTS = _discover()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run(source_repo: Path, set_name: str, only: list[str] | None = None) -> int:
    chosen = [e for e in EXPORTS if e.set_name == set_name and (not only or e.dest in only)]
    if not chosen:
        print(f"no exports registered for set {set_name!r}", file=sys.stderr)
        return 1
    out_dir = CLEAN / "results" / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "MANIFEST.tsv"
    rows = []
    if manifest_path.exists():
        rows = pd.read_csv(manifest_path, sep="\t").to_dict("records")
    rows = [r for r in rows if r["file"] not in {e.dest for e in chosen}]

    for exp in chosen:
        src = source_repo / exp.source
        if not src.is_file():
            print(f"missing source: {exp.source}", file=sys.stderr)
            return 1
        if src.suffix == ".parquet":
            frame = pd.read_parquet(src)  # needs pyarrow: the PRESCOTT env on the authoring machine
        else:
            frame = pd.read_csv(src, sep="," if src.suffix == ".csv" else "\t")
        frame = frame.drop(columns=[c for c in exp.drop_columns if c in frame.columns])
        if exp.transform is not None:
            # A transform may take (frame) or (frame, source_repo) when it must read a second file.
            takes_repo = len(inspect.signature(exp.transform).parameters) >= 2
            frame = exp.transform(frame, source_repo) if takes_repo else exp.transform(frame)
        frame.to_csv(out_dir / exp.dest, sep="\t", index=False, compression="gzip")
        rows.append({
            "file": exp.dest, "source": exp.source, "source_sha256_16": sha256(src),
            "rows": len(frame), "columns": frame.shape[1],
            "dropped": ",".join(exp.drop_columns), "used_for": exp.why,
        })
        print(f"{exp.dest}: {len(frame)} rows x {frame.shape[1]} cols")
    pd.DataFrame(rows).sort_values("file").to_csv(manifest_path, sep="\t", index=False)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-repo", required=True, type=Path)
    ap.add_argument("--set", dest="set_name", required=True)
    ap.add_argument("--only", nargs="+", help="export just these destination file names")
    ns = ap.parse_args()
    sys.exit(run(ns.source_repo.resolve(), ns.set_name, ns.only))
