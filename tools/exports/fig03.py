"""Tables behind Figure 3 (and S2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_base import Export  # noqa: E402

EXPORTS = [
    Export(
        "h3n2_13lineage",
        "results/nextstrain_13lin_dense/tables/alpha_sweep_fit_metrics_BY_LINEAGE.tsv",
        "alpha_sweep_by_lineage.tsv.gz",
        # The old slope-constrained logistic fit floored AUROC at exactly 0.5
        # wherever the score was anti-predictive. The fixed column keeps the plain
        # name; the clamped one is dropped so no figure can read it by accident.
        drop_columns=("site_logistic_auroc_constrained_fit", "site_logistic_pr_auc_constrained_fit"),
        why="Fig 3 and S2: metrics per lineage across the 101-point alpha grid",
    ),
]
