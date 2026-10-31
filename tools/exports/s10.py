"""Table behind S10: agreement between two DMS scans of one protein against how far apart the strains are.

One row per pair of scans (26 in the archive, not the 23 its README says: the three cross-subtype
rows were added after the README). Nothing here depends on the lineage set; it is shipped under
every set so a guide can name it.

Columns kept: what identifies the pair, the two agreement measures and their floor-censoring
brackets, the preference-profile distance (JSD), and the coordinate/filter provenance columns the
caption quotes. ``pearson_r_wt_identical``, ``spearman_rho_wt_identical`` and the ``*_divergent``
columns are dropped: no panel reads them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_base import Export  # noqa: E402

KEEP = [
    "virus", "system", "sample_1", "sample_2", "metric_category", "metric_detail",
    "aa_distance", "focal_protein_length", "pct_divergence", "indel_count",
    "n_variants", "n_sites", "spearman_rho", "pearson_r",
    "frac_at_floor", "spearman_rho_no_floor", "pearson_r_no_floor",
    "jsd_mean", "n_sites_jsd", "join_key", "filter_mode", "wt_agreement",
]


def _keep(frame):
    return frame[KEEP]


EXPORTS = [
    Export(
        "h3n2_13lineage", "results/dms/tables/dms_cross_lineage_pairs.tsv", "dms_cross_lineage_pairs.tsv.gz",
        transform=_keep,
        why="S10: cross-lineage DMS concordance (Spearman, Pearson, JSD) against amino-acid distance",
    ),
]
