"""Table behind the symmetry tests: two amino-acid property controls that contain no nucleotide information.

``replacement_scores`` already carries BLOSUM62 and negative Grantham distance. The direction tests also
use negative hydropathy distance (Kyte and Doolittle 1982) and negative volume distance (Grantham 1974),
so that no conclusion rests on one control. Both are functions of the amino-acid pair alone, so one row
per (lineage, position, aa) is kept.

Source: ``tmp_explore/out/annotated_table_13lin.parquet`` (columns ``hydropathy_d`` and ``d_volume``,
absolute differences). They are negated so that, like BLOSUM62, higher means more exchangeable.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_base import Export  # noqa: E402

KEY = ["lineage", "position", "aa"]


def properties(frame: pd.DataFrame) -> pd.DataFrame:
    one = frame.drop_duplicates(KEY)
    if one[["hydropathy_d", "d_volume"]].isna().any().any():
        raise ValueError("hydropathy or volume distance missing for some replacements")
    out = one[KEY].copy()
    out["neg_hydropathy"] = -one["hydropathy_d"].to_numpy()
    out["neg_volume"] = -one["d_volume"].to_numpy()
    return out.sort_values(KEY).reset_index(drop=True)


EXPORTS = [
    Export(
        "h3n2_13lineage",
        "tmp_explore/out/annotated_table_13lin.parquet",
        "replacement_properties.tsv.gz",
        transform=properties,
        why="Symmetry tests: hydropathy and volume similarity controls (no nucleotide content)",
    ),
]
