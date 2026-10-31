"""The symmetry battery on simulated sequences, where the generating spectrum and landscape are known.

    python tools/run_symmetry_sim.py --spectrum sars_cov_2
    python tools/run_symmetry_sim.py --spectrum human --scale 1.2e-8
    python tools/run_symmetry_sim.py --spectrum influenza_h3n2
    python tools/run_symmetry_sim.py --spectrum jc69 --row-sum 3.5e-4

Sequences were evolved from one founder under a known site-wise fitness landscape and, in separate
runs, four spectra: human germline, SARS-CoV-2, influenza H3N2, and flat JC69 (negative control).
ESM-C 600M was scored on the founder before fine-tuning and after fine-tuning on each arm. If a test
cannot recover the spectrum a model was *trained on*, it cannot be trusted on real data; if it can,
the real-data result is informative. JC69 controls for false-positive detection of direction where
no directional bias exists.

Scores: the five models (base, human, SC2, H3N2, JC69 fine-tunes), the alignment profile of each evolved
set (selection and mutation together), the true landscape (selection only, no mutation at all), and
the four amino-acid controls. The extra test here is ``partial``: correlation with accessibility
**holding the true landscape fixed**, which is possible because the true landscape is known.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mutacc import accessibility as acc
from mutacc import symmetry as sym
from mutacc.guide import REPO_ROOT, load_guide

SEED = 20261009
MODELS = ["plm_base", "plm_human_ft", "plm_sc2_ft", "plm_h3n2_ft", "plm_jc69_ft"]
PROFILES = ["pssm_human", "pssm_sc2", "pssm_h3n2", "pssm_jc69"]
SELECTION = ["ground_truth"]
CONTROLS = ["blosum62", "neg_grantham", "neg_hydropathy", "neg_volume"]


def pair_controls(guide_path: Path) -> pd.DataFrame:
    """(ref_aa, aa) -> the four control scores, taken from the real-data tables (nothing typed in)."""
    g = load_guide(guide_path)
    s = g.read_table("replacement_scores")
    s = s[s["model"] == g.models[0].key]
    p = g.read_table("replacement_properties")
    m = s.merge(p, on=["lineage", "position", "aa"], validate="one_to_one")
    pairs = m.groupby(["ref_aa", "aa"])[CONTROLS].agg(["min", "max"])
    for c in CONTROLS:
        if not np.allclose(pairs[(c, "min")], pairs[(c, "max")]):
            raise ValueError(f"control {c!r} is not a function of the amino-acid pair alone")
    return m.groupby(["ref_aa", "aa"])[CONTROLS].first().reset_index()


def build_unit(table: Path, controls: pd.DataFrame) -> tuple[list[sym.Unit], pd.DataFrame]:
    d = pd.read_csv(table, sep="\t")
    d = d.merge(controls, on=["ref_aa", "aa"], how="left", validate="many_to_one")
    if d[CONTROLS].isna().any().any():
        raise ValueError("some simulated replacements have no control score (amino-acid pair absent from real data)")
    for c in MODELS + PROFILES + SELECTION:
        d[c] = np.log10(d[c].clip(lower=1e-300))
    d["lineage"] = "sim"
    d["position"] = d["site"]
    units = sym.units_from_frame(d, MODELS + PROFILES + SELECTION + CONTROLS)
    return units, d


def partial_orientation(unit: sym.Unit, q: np.ndarray, scores: list[str], control: str,
                        n_boot: int, seed: int, row_sum: float | None = None) -> pd.DataFrame:
    """Spearman partial correlation with each orientation of the spectrum, holding ``control`` fixed.

    Within each stratum: rank everything, regress the control's rank out of both the score and the
    target, correlate the residuals. Reported with the paired site-bootstrap difference direct - variant.
    """
    variants = sym.orientation_variants(q)
    names = list(variants)
    targets = {n: sym.accessibility_for(unit, variants[n], row_sum=row_sum) for n in names}
    draws = [None] + sym._draws([unit], n_boot, seed)
    out = {}
    for di, draw in enumerate(draws):
        rows = unit.rows(draw)
        for s in (1, 2, 3):
            r = rows[unit.steps[rows] == s]
            ctl = rankdata(unit.scores[control][r]); ctl = (ctl - ctl.mean()) / np.linalg.norm(ctl - ctl.mean())

            def resid(v):
                z = rankdata(v); z = z - z.mean()
                return z - (z @ ctl) * ctl

            rt = {n: resid(targets[n][r]) for n in names}
            for sc in scores:
                x = resid(unit.scores[sc][r])
                for n in names:
                    out.setdefault((sc, n, s), []).append(
                        float(x @ rt[n] / np.sqrt((x @ x) * (rt[n] @ rt[n]))))
    rows = []
    for (sc, n, s), v in out.items():
        point, boot = v[0], np.array(v[1:])
        rows.append(dict(kind="rho", score=sc, target=n, versus="", stratum=s, value=point,
                         lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
        if n != "direct":
            diff = np.array(out[(sc, "direct", s)][1:]) - boot
            rows.append(dict(kind="delta", score=sc, target="direct", versus=n, stratum=s,
                             value=out[(sc, "direct", s)][0] - point,
                             lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5)))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spectrum", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the spectrum (human: 1.2e-8)")
    ap.add_argument("--row-sum", type=float, default=None, help="rescale a relative spectrum to this mean row sum")
    ap.add_argument("--guide", type=Path, default=REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
    ap.add_argument("--table", type=Path, default=REPO_ROOT / "results" / "sim_v3_star90" / "replacement_scores.tsv.gz")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-shuffle", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out_dir = args.out or args.table.parent
    q = acc.load_spectrum(REPO_ROOT / "data" / "spectra" / f"{args.spectrum}.csv") * args.scale
    units, frame = build_unit(args.table, pair_controls(args.guide))
    names = MODELS + PROFILES + SELECTION + CONTROLS
    tag = f"symmetry_{args.spectrum}"
    written = []

    def save(df, name):
        p = out_dir / f"{tag}_{name}.tsv.gz"
        df.to_csv(p, sep="\t", index=False, float_format="%.8g", compression="gzip")
        written.append(p)
        print(f"{name:12s} {len(df):5d} rows", flush=True)

    save(sym.orientation_battery(units, q, names, n_boot=args.n_boot, seed=SEED, row_sum=args.row_sum), "orientation")
    save(sym.null_battery(units, q, names, n_shuffle=args.n_shuffle, seed=SEED, row_sum=args.row_sum), "null")
    save(sym.flip_null(units, q, names, row_sum=args.row_sum), "flip")
    if sym.favoured_classes(q)[0]:
        save(sym.direction_contrast(units, q, names, n_boot=args.n_boot, seed=SEED, row_sum=args.row_sum), "direction")
    save(partial_orientation(units[0], q, MODELS + PROFILES + CONTROLS, "ground_truth", args.n_boot, SEED, row_sum=args.row_sum),
         "partial_given_truth")

    manifest = out_dir / "MANIFEST.tsv"
    rows = pd.read_csv(manifest, sep="\t").to_dict("records") if manifest.exists() else []
    have = {p.name for p in written}
    rows = [r for r in rows if r["file"] not in have]
    for p in written:
        t = pd.read_csv(p, sep="\t")
        rows.append({"file": p.name, "source": f"generated by tools/run_symmetry_sim.py (spectrum={args.spectrum}, "
                     f"scale={args.scale:g}, n_boot={args.n_boot}, seed={SEED})", "source_sha256_16": "",
                     "rows": len(t), "columns": t.shape[1], "dropped": "",
                     "used_for": "Symmetric-vs-directional tests with known truth (SYMMETRY_TESTS.md)"})
    pd.DataFrame(rows).sort_values("file").to_csv(manifest, sep="\t", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
