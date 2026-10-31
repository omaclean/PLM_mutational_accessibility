"""Table behind S2: which replacements were observed in each lineage.

``replacement_scores`` (Figure 2) holds the model and accessibility score of every possible
replacement but not whether it was seen in sequences. S2 needs that label to compute PR-AUC and
partial AUROC from the scores. ``obs_present`` is 1 where the replacement was observed in the
lineage's sequences, and does not depend on the model, so one row per (lineage, position, aa) is
kept; the transform refuses to run if the two models disagree.

Source: ``results/nextstrain_13lin_dense/tables/combined_long_table.csv``.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_base import Export  # noqa: E402

KEY = ["lineage", "position", "aa"]


def observed_only(frame: pd.DataFrame) -> pd.DataFrame:
    per_model = frame.groupby(KEY)["obs_present"].nunique()
    if (per_model > 1).any():
        raise ValueError(f"{int((per_model > 1).sum())} replacements are observed under one model and not another")
    if not set(frame["obs_present"].unique()) <= {0, 1}:
        raise ValueError("obs_present is not 0/1")
    out = frame.drop_duplicates(KEY)[KEY + ["obs_present"]].copy()
    out["obs_present"] = out["obs_present"].astype(int)
    return out.sort_values(KEY).reset_index(drop=True)


EXPORTS = [
    Export(
        "h3n2_13lineage",
        "results/nextstrain_13lin_dense/tables/combined_long_table.csv",
        "replacement_observed.tsv.gz",
        transform=observed_only,
        why="S2: whether each replacement was observed in each lineage",
    ),
]
