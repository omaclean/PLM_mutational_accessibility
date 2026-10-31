"""Run the symmetric-versus-directional test battery and write its tables.

    python tools/run_symmetry_battery.py --guide guides/h3n2_13lineage.yaml --spectrum influenza_h3n2

Reads ``replacement_scores``, ``replacement_properties`` and the lineage roots named by the guide, and
a spectrum from ``data/spectra/``. Scores are the log probability of the guide's base and fine-tuned
models plus four controls that contain no nucleotide information (BLOSUM62, Grantham, hydropathy and
volume similarity). Writes ``results/<run>/symmetry_<spectrum>_<test>.tsv.gz``.

This is analysis, not drawing: figures read the tables it writes. Everything is seeded.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from mutacc import accessibility as acc
from mutacc import symmetry as sym
from mutacc.guide import REPO_ROOT, Guide, load_guide

CONTROLS = ["blosum62", "neg_grantham", "neg_hydropathy", "neg_volume"]
SEED = 20261009


def build_frame(guide: Guide, results: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """One row per (lineage, position, aa) with a log-probability column per model role plus controls."""
    scores = guide.read_table("replacement_scores", results)
    props = guide.read_table("replacement_properties", results)
    key = ["lineage", "position", "aa"]
    wide = None
    model_cols = []
    for m in guide.models:
        col = f"plm_{m.role}"
        if col in model_cols:
            raise ValueError(f"two models share the role {m.role!r}; the battery needs one base and one fine-tune")
        sub = scores.loc[scores["model"] == m.key, key + ["ref_aa", "nt_steps", "plm_prob", "mut_prob", "blosum62", "neg_grantham"]]
        sub = sub.rename(columns={"plm_prob": col})
        sub[col] = np.log10(sub[col].clip(lower=1e-300))
        wide = sub if wide is None else wide.merge(sub[key + [col]], on=key, validate="one_to_one")
        model_cols.append(col)
    wide = wide.merge(props, on=key, validate="one_to_one")
    codons = {}
    for lin in guide.lineages:
        positions = np.sort(wide.loc[wide["lineage"] == lin.id, "position"].unique())
        codons.update({(lin.id, int(p)): c for p, c in zip(positions, acc.reference_codons(lin.root, positions))})
    wide["ref_codon"] = [codons[(l, int(p))] for l, p in zip(wide["lineage"], wide["position"])]
    return wide, model_cols + CONTROLS


def save(frame: pd.DataFrame, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.tsv.gz"
    frame.to_csv(path, sep="\t", index=False, float_format="%.8g", compression="gzip")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--guide", required=True, type=Path)
    ap.add_argument("--spectrum", required=True, help="file stem in data/spectra/, e.g. influenza_h3n2")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-shuffle", type=int, default=500)
    ap.add_argument("--tests", nargs="+", default=["orientation", "null", "flip", "direction", "planted", "robustness"])
    ap.add_argument("--row-sum", type=float, default=None, help="rescale a relative spectrum to this mean row sum")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the spectrum (human: 1.2e-8)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    guide = load_guide(args.guide)
    out_dir = args.out or guide.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    q = acc.load_spectrum(REPO_ROOT / "data" / "spectra" / f"{args.spectrum}.csv") * args.scale
    frame, score_names = build_frame(guide)
    units = sym.units_from_frame(frame, score_names)
    tag = f"symmetry_{args.spectrum}"
    t0 = time.time()
    written = []

    def done(name, frame_):
        written.append(save(frame_, out_dir, f"{tag}_{name}"))
        print(f"{name:12s} {len(frame_):6d} rows  {time.time() - t0:6.1f}s", flush=True)

    if "orientation" in args.tests:
        done("orientation", sym.orientation_battery(units, q, score_names, n_boot=args.n_boot, seed=SEED,
                                                    row_sum=args.row_sum))
    if "null" in args.tests:
        done("null", sym.null_battery(units, q, score_names, n_shuffle=args.n_shuffle, seed=SEED,
                                      row_sum=args.row_sum))
    if "flip" in args.tests:
        done("flip", sym.flip_null(units, q, score_names, row_sum=args.row_sum))
    if "direction" in args.tests:
        done("direction", sym.direction_contrast(units, q, score_names, n_boot=args.n_boot, seed=SEED,
                                                 row_sum=args.row_sum))
    if "planted" in args.tests:
        done("planted", sym.planted_signal(units, q, CONTROLS[:2], row_sum=args.row_sum))
    if "robustness" in args.tests:
        rr = sym.rank_robustness(units, q, score_names, seed=SEED, row_sum=args.row_sum)
        done("robustness", rr)
        written.append(save(rr.attrs["distinct_values"], out_dir, f"{tag}_distinct_values"))

    manifest = out_dir / "MANIFEST.tsv"
    rows = pd.read_csv(manifest, sep="\t").to_dict("records") if manifest.exists() else []
    names = {p.name for p in written}
    rows = [r for r in rows if r["file"] not in names]
    for p in written:
        n = len(pd.read_csv(p, sep="\t"))
        rows.append({"file": p.name, "source": f"generated by tools/run_symmetry_battery.py (spectrum={args.spectrum}, "
                     f"n_boot={args.n_boot}, n_shuffle={args.n_shuffle}, seed={SEED})",
                     "source_sha256_16": "", "rows": n, "columns": len(pd.read_csv(p, sep="\t", nrows=1).columns),
                     "dropped": "", "used_for": "Symmetric-vs-directional tests (SYMMETRY_TESTS.md)"})
    pd.DataFrame(rows).sort_values("file").to_csv(manifest, sep="\t", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
