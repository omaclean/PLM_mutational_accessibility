"""Tables behind Figure 2 (and S4): model probability against mutational accessibility.

One table, ``replacement_scores``, shipped for both lineage sets. One row per
(model, lineage, site, replacement amino acid). The figures read the same schema from
whichever guide they are given, so the strata panel and the asymmetry panel can come from
different sets without either script knowing which.

Columns kept (everything else in the annotated archive table is dropped to keep the file small):

* ``model, lineage, position, ref_aa, aa``  identify the row. ``position`` is the HA0 (full-length)
  numbering the PLM was scored in; Figure 2 never uses it except as the unit that is resampled.
* ``plm_prob``      the model's probability of the replacement amino acid at that site.
* ``mut_prob``      mutational accessibility: probability of the replacement under the nucleotide
                    mutation spectrum, given the lineage's reference codon.
* ``nt_steps``      minimum nucleotide changes from the reference codon to any codon for ``aa``
                    (the codon-distance stratum).
* ``blosum62, neg_grantham``  amino-acid-pair scores that know nothing about codons.
                    ``neg_grantham`` is the Grantham distance with its sign flipped, so that for
                    both columns "higher = more exchangeable".

``plm_prob`` is rounded to 7 significant digits (below the precision of the fp32 model; no ties are
created, checked in tests). ``mut_prob`` and ``neg_grantham`` are NOT rounded: accessibility takes a few
hundred distinct values per stratum that differ in the 8th-10th digit, and rounding merges them into ties
and moves the published correlations in the third decimal.

Sources (exploration archive): ``tmp_explore/out/annotated_table.parquet`` (five-lineage run) and
``annotated_table_13lin.parquet``. Both were checked against the run's own
``combined_long_table.csv`` on every (model, lineage, position, aa) key: no rows missing or added,
``plm_prob`` and ``mut_prob`` equal to within floating-point noise.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_base import Export  # noqa: E402

KEEP = ["model", "lineage", "position", "ref_aa", "aa", "plm_prob", "mut_prob",
        "nt_steps", "blosum62", "neg_grantham"]
SIG = ["plm_prob"]


def slim(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[KEEP].copy()
    if out[KEEP].isna().any().any():
        raise ValueError("annotated table has missing values in the columns Figure 2 reads")
    for col in SIG:
        out[col] = out[col].map(lambda v: float(f"{v:.7g}"))
    return out.sort_values(["model", "lineage", "position", "aa"]).reset_index(drop=True)


EXPORTS = [
    Export(
        "h3n2_5lineage",
        "tmp_explore/out/annotated_table.parquet",
        "replacement_scores.tsv.gz",
        why="Fig 2 and S4: PLM probability vs accessibility within codon-distance strata",
        transform=slim,
    ),
    Export(
        "h3n2_13lineage",
        "tmp_explore/out/annotated_table_13lin.parquet",
        "replacement_scores.tsv.gz",
        why="Fig 2 and S4 on the larger lineage set; Fig 2 asymmetry panel",
        transform=slim,
    ),
]
