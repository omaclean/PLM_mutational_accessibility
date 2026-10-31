"""Build replacement_scores_masked.tsv.gz from the masked profiles.

Swaps plm_prob in replacement_scores.tsv.gz with the masked-marginal
probabilities from output/unmasked_check/fig03_13lineage/*masked_profile.csv.
All other columns (mut_prob, nt_steps, blosum62, neg_grantham, ref_aa) remain identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dev" / "masked_fig02"))
from score_masked_profile import read_profile


def build_masked_replacement_scores(
    scores_path: Path, profiles_dir: Path, out_path: Path
) -> Path:
    print(f"Reading {scores_path}...", flush=True)
    scores = pd.read_csv(scores_path, sep="\t")
    masked_plm = np.empty(len(scores), dtype=float)

    for (model, lineage), idx in scores.groupby(["model", "lineage"]).groups.items():
        prof_file = profiles_dir / f"{lineage}__{model}__masked_profile.csv"
        if not prof_file.exists():
            raise FileNotFoundError(f"Missing masked profile: {prof_file}")
        _, prof = read_profile(prof_file)
        row_map = {aa: i for i, aa in enumerate(prof.index)}
        vals = prof.to_numpy(dtype=float)

        sub = scores.iloc[idx]
        pos_idx = sub["position"].to_numpy(int) - 1
        aa_idx = [row_map[a] for a in sub["aa"]]
        masked_plm[idx] = vals[aa_idx, pos_idx]

    out_df = scores.copy()
    out_df["plm_prob"] = masked_plm

    print(f"Writing {out_path}...", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"Done. Wrote {len(out_df):,} rows.", flush=True)
    return out_path


if __name__ == "__main__":
    scores_file = REPO / "results" / "h3n2_13lineage" / "replacement_scores.tsv.gz"
    prof_dir = REPO / "output" / "unmasked_check" / "fig03_13lineage"
    out_file = REPO / "results" / "h3n2_13lineage" / "replacement_scores_masked.tsv.gz"
    build_masked_replacement_scores(scores_file, prof_dir, out_file)
