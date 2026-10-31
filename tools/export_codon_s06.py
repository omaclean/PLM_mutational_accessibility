"""Derive the tables behind the codon-learning figure and S6, from the shipped scores.

Unlike ``export_tables.py`` this does not read the exploration archive. It reads
``replacement_scores`` and the lineage roots a guide already ships, runs the bootstraps and
permutation nulls once, and writes the results as gzipped TSV next to the scores, with a row per
file in the set's ``MANIFEST.tsv``. The figure scripts read these tables (and recompute every
point estimate from the scores as a check), so the slow resampling never runs when a figure is
drawn.

    python tools/export_codon_s06.py --set h3n2_13lineage --tables codon sG
    python tools/export_codon_s06.py --set h3n2_5lineage  --tables codon

Tables (``codon``), per set:
    codon_within_pair_summary     one row per model: pooled AUROC, site-bootstrap interval, null
    codon_within_pair_difference  fine-tuned minus base, with a paired site-bootstrap interval
    codon_within_pair_by_pair     one row per scored replacement and model
    codon_within_pair_by_lineage  the statistic inside each lineage on its own
    codon_within_pair_robustness  drop the strongest replacements; drop six-codon amino acids
    codon_strand_c_summary        within-residue correlation, interval and null, per model
    codon_strand_c_per_aa         the correlation within each amino acid
Tables (``sG``):
    spectrum_swap_strata          correlation with accessibility under each spectrum, with paired differences
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_base import append_manifest  # noqa: E402
from mutacc import accessibility as acc  # noqa: E402
from mutacc import codon_learning as cl  # noqa: E402
from mutacc import spectrum_swap as sw  # noqa: E402
from mutacc.guide import Guide, load_guide  # noqa: E402

SEED = 20261009
STRAND_C_SEED = 20260928            # the pre-specified seed of the original strand C test
DEFAULTS = {"n_boot": 1000, "n_perm": 1000, "n_perm_c": 10000, "n_boot_s07": 300}

#: S6 conditions: (key, label, spectrum file stem, transform, reference). Labels are what a reader sees.
#: ``h1n1``, ``ecoli``, ``yeast`` and the reversed influenza spectrum are not in the archive's a35.
S07_CONDITIONS = [
    ("h3n2", "Influenza H3N2 (Pauly et al.)", "influenza_h3n2", None, True),
    ("h3n2_symmetrised", "Influenza H3N2 (Pauly GTR-nondirectional)", "influenza_h3n2", "symmetrise", False),
    ("h3n2_reversed", "Influenza H3N2, every direction reversed", "influenza_h3n2", "transpose", False),
    ("panha_fourfold_gtr", "Influenza pan-HA (fourfold GTR)", "panha_fourfold_gtr", None, False),
    ("h1n1", "Influenza H1N1", "influenza_h1n1", None, False),
    ("sars_cov_2", "SARS-CoV-2", "sars_cov_2", None, False),
    ("human", "Human germline", "human", None, False),
    ("ecoli", "E. coli", "ecoli", None, False),
    ("yeast", "Yeast", "yeast", None, False),
    ("k80", "Transitions twice as fast (K80)", "k80_kappa2", None, False),
    ("jc69", "All changes equal (JC69)", "jc69", None, False),
]
SERIES_LABEL = {"neg_grantham": "Grantham similarity (control)", "blosum62": "BLOSUM62 (comparator)"}
PROPERTIES = ["blosum62", "neg_grantham"]


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def _roles(guide: Guide) -> tuple[str, str]:
    base, ft = guide.models_with_role("base"), guide.models_with_role("finetune")
    if len(base) != 1 or len(ft) != 1:
        raise ValueError(f"{guide.source.name}: need exactly one base and one finetune model")
    return base[0].key, ft[0].key


def _roots(guide: Guide) -> dict[str, Path]:
    return {lin.id: lin.root for lin in guide.lineages}


def _interval(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise ValueError("too few finite bootstrap draws")
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Codon tables
# ---------------------------------------------------------------------------

def codon_tables(guide: Guide, scores: pd.DataFrame, *, n_boot: int, n_perm: int, n_perm_c: int,
                 spectrum: Path, symmetrise: bool) -> dict[str, pd.DataFrame]:
    return cl.derive_codon_tables(guide, scores, n_boot=n_boot, n_perm=n_perm, n_perm_c=n_perm_c,
                                  spectrum=spectrum, symmetrise=symmetrise, seed=SEED,
                                  strand_c_seed=STRAND_C_SEED)


# ---------------------------------------------------------------------------
# S6
# ---------------------------------------------------------------------------

def s06_tables(guide: Guide, scores: pd.DataFrame, *, n_boot: int) -> dict[str, pd.DataFrame]:
    import json
    base, ft = _roles(guide)
    conditions = []
    for key, label, stem, transform, reference in S07_CONDITIONS:
        if stem.endswith("_gtr"):
            fit_path = REPO / "data" / f"{stem}.json"
            fit = json.loads(fit_path.read_text())
            bases = "ACGT"
            pi = np.array([fit["base_frequencies"][b] for b in bases], dtype=float)
            R = np.zeros((4, 4), dtype=float)
            for k, v in fit["exchangeabilities"].items():
                b1, b2 = k.split("-")
                i, j = bases.index(b1), bases.index(b2)
                R[i, j] = R[j, i] = float(v)
            q = R * pi[None, :]
            np.fill_diagonal(q, 0.0)
        else:
            q = acc.load_spectrum(REPO / "data" / "spectra" / f"{stem}.csv")
            if transform:
                q = getattr(acc, transform)(q)
        conditions.append(sw.Condition(key, label, q, reference))
    table = sw.swap_table(scores, [base, ft], PROPERTIES, _roots(guide), conditions, n_boot=n_boot, seed=SEED)
    label = {**{m.key: m.label for m in guide.models}, **SERIES_LABEL}
    table["series_label"] = table["series"].map(label)
    table["role"] = [guide.model(s).role if k == "model" else "biochemistry" for s, k in zip(table["series"], table["kind"])]
    return {"spectrum_swap_strata": table}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path, source_note: str, used_for: dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, frame in tables.items():
        if frame.empty:
            raise ValueError(f"{name}: refusing to write an empty table")
        frame.to_csv(out_dir / f"{name}.tsv.gz", sep="\t", index=False, compression="gzip")
        rows.append({"file": f"{name}.tsv.gz", "source": source_note, "source_sha256_16": "",
                     "rows": len(frame), "columns": frame.shape[1], "dropped": "", "used_for": used_for[name]})
        print(f"{name}: {len(frame)} rows x {frame.shape[1]} cols")
    append_manifest(out_dir / "MANIFEST.tsv", rows)


# append_manifest is shared with the other exporters so every MANIFEST is written the
# same way; see tools/export_base.py.


USED = {
    "codon_within_pair_summary": "Codon-learning figure and S10: pooled within-replacement AUROC, site-bootstrap interval, permutation null",
    "codon_within_pair_difference": "Codon-learning figure: fine-tuned minus base, paired site bootstrap",
    "codon_within_pair_by_pair": "Codon-learning figure and S10: AUROC of each scored replacement",
    "codon_within_pair_by_lineage": "Codon-learning figure: the statistic inside each lineage on its own",
    "codon_within_pair_robustness": "Codon-learning figure: dropping the strongest replacements",
    "codon_strand_c_summary": "Codon-learning figure: within-residue correlation with codon persistence and its null",
    "codon_strand_c_per_aa": "Codon-learning figure: the within-residue correlation for each amino acid",
    "spectrum_swap_strata": "S6: correlation with accessibility under each spectrum, with paired differences",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", dest="set_name", required=True, help="a guide name under guides/, e.g. h3n2_13lineage")
    ap.add_argument("--tables", nargs="+", choices=["codon", "sG"], default=["codon", "sG"])
    ap.add_argument("--out-dir", type=Path, default=None, help="default: the guide's results_dir")
    ap.add_argument("--spectrum", default="influenza_h3n2",
                    help="spectrum stem under data/spectra used for strand C (symmetrised unless --directional)")
    ap.add_argument("--directional", action="store_true", help="use the spectrum as given for strand C")
    ap.add_argument("--n-boot", type=int, default=DEFAULTS["n_boot"])
    ap.add_argument("--n-perm", type=int, default=DEFAULTS["n_perm"])
    ap.add_argument("--n-perm-c", type=int, default=DEFAULTS["n_perm_c"])
    ap.add_argument("--n-boot-sG", dest="n_boot_s07", type=int, default=DEFAULTS["n_boot_s07"])
    args = ap.parse_args(argv)

    guide = load_guide(REPO / "guides" / f"{args.set_name}.yaml")
    out_dir = args.out_dir or guide.results_dir
    score_path = guide.table_path("replacement_scores")
    scores = guide.read_table("replacement_scores")
    note = (f"generated by tools/export_codon_s06.py from replacement_scores (sha {sha16(score_path)}) and the guide's roots; "
            f"n_boot={args.n_boot}, n_perm={args.n_perm}, seed={SEED}")
    if "codon" in args.tables:
        spectrum = REPO / "data" / "spectra" / f"{args.spectrum}.csv"
        tables = codon_tables(guide, scores, n_boot=args.n_boot, n_perm=args.n_perm, n_perm_c=args.n_perm_c,
                              spectrum=spectrum, symmetrise=not args.directional)
        write_tables(tables, out_dir, note, USED)
    if "sG" in args.tables:
        tables = s06_tables(guide, scores, n_boot=args.n_boot_s07)
        write_tables(tables, out_dir, note.replace(f"n_boot={args.n_boot}", f"n_boot={args.n_boot_s07}"), USED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
