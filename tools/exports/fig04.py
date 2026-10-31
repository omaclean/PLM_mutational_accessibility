"""Tables behind Figure 4: measured DMS fitness against the alpha-weighted score.

Five tables, all from ``results/dms/`` in the exploration archive:

* ``dms_scores``            one row per (dataset, model, site, replacement): the model score,
                            accessibility, nucleotide-step class (``nt_steps``, read by S12) and the measured traits. Trimmed to the columns the
                            figure reads; the PLM profiles themselves need a GPU and are not shipped.
* ``dms_peak_summary``      a31's own per-curve summary: panel labels, citations, optimum, and the
                            ``interpretable`` flag the figure must respect.
* ``dms_alpha_curves``      a31's own rho-vs-alpha curves. The figure's self-check: it recomputes
                            every curve and must agree to 1e-6.
* ``dms_alpha_bands_*``     the published curves with their site-bootstrap band, for the
                            published and the reversible-spectrum variants.
* ``dms_spectrum_*``        the two 4x4 nucleotide matrices the reversible variant symmetrises.

The model *key* in a31 (``ESM2_650M_base``) is not the key Figure 3's tables use for the same
untuned model (``ESM2_650M_HA80_raw``), and the guide only names two models, so roles and
reader-facing labels for the four DMS models are attached here, in the bridge, and the figure reads
them from the table.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_base import Export  # noqa: E402

SET = "h3n2_13lineage"

# a31 key -> (prose name used by every a31 summary table, role, reader-facing label)
MODELS = {
    "ESM2_650M_base": ("ESM-2 650M base", "base", "Base (ESM-2)"),
    "ESM2_650M_HA80": ("HA-80 fine-tuned", "finetune", "Fine-tuned (IAV)"),
    "ESMC_600M_base": ("ESM-C 600M base", "base", "Base (ESM-C)"),
    "ESMC_600M_SC2ft": ("SC2 fine-tuned", "finetune", "Fine-tuned (SC2)"),
}
BY_PROSE = {prose: (role, label) for prose, role, label in MODELS.values()}

# Short virus names for Figure 4's panel titles. The figure groups panels by virus and will
# not invent an abbreviation of its own, so an organism added here must get a tag here too.
VIRUS_TAGS = {"Influenza H3N2": "IAV", "SARS-CoV-2": "SC2"}

# a31 scores each scan against the spectrum of the virus it came from.
SPECTRUM_OF_DATASET = {"MA22": "influenza_h3n2", "Perth09": "influenza_h3n2",
                       "BA2": "sars_cov_2", "XBB15": "sars_cov_2"}

TRAITS = ["MDCKSIAT1 cell entry", "sera escape", "pH stability", "preference ratio",
          "spike mediated entry", "ACE2 binding", "human sera escape"]


def _scores(frame: pd.DataFrame) -> pd.DataFrame:
    unknown = set(frame["model"]) - set(MODELS)
    if unknown:
        raise ValueError(f"model(s) with no label in tools/exports/fig04.py: {sorted(unknown)}")
    missing_ds = set(frame["dataset"]) - set(SPECTRUM_OF_DATASET)
    if missing_ds:
        raise ValueError(f"dataset(s) with no spectrum in tools/exports/fig04.py: {sorted(missing_ds)}")
    out = frame.copy()
    out["model_name"] = out["model"].map(lambda k: MODELS[k][0])
    out["spectrum_matrix"] = out["dataset"].map(SPECTRUM_OF_DATASET)
    cols = ["dataset", "model_name", "spectrum_matrix", "site", "ref_codon", "aa", "nt_steps", "log_plm", "log_acc"] + TRAITS
    # row order is kept: the bootstrap draws are consumed in table order.
    return out[cols]


def _peak_summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    unknown = set(out["model_name"]) - set(BY_PROSE)
    if unknown:
        raise ValueError(f"model name(s) with no label in tools/exports/fig04.py: {sorted(unknown)}")
    # a31 abbreviates one panel label ("pref. ratio"); readers get the full words.
    out["panel_label"] = out["panel_label"].str.replace("pref. ratio", "preference ratio", regex=False)
    out["model_role"] = out["model_name"].map(lambda n: BY_PROSE[n][0])
    out["model_label"] = out["model_name"].map(lambda n: BY_PROSE[n][1])
    unknown_org = set(out["organism"]) - set(VIRUS_TAGS)
    if unknown_org:
        raise ValueError(f"organism(s) with no short tag in tools/exports/fig04.py: {sorted(unknown_org)}")
    out["virus_tag"] = out["organism"].map(VIRUS_TAGS)
    return out.drop(columns=["model_role_label"])


def _bands(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(columns={"model_label": "model_name"})
    keep = ["dataset", "model_name", "trait", "alpha", "rho", "ci_lo", "ci_hi", "n", "n_sites"]
    return out[keep]


def _within_stratum(frame: pd.DataFrame) -> pd.DataFrame:
    """a31's within-stratum table, keyed by the prose model name every other DMS table uses."""
    out = frame.rename(columns={"model_label": "model_name"})
    return out[["dataset", "trait", "model_name", "stratum", "n", "n_sites", "rho", "ci_lo", "ci_hi", "n_boot"]]


EXPORTS = [
    Export(
        SET, "results/dms/a31_dms_joined.parquet", "dms_scores.tsv.gz",
        transform=_scores,
        why="Fig 4: model score, accessibility and measured trait per DMS replacement",
    ),
    Export(
        SET, "results/dms/tables/a31_dms_peak_summary.tsv", "dms_peak_summary.tsv.gz",
        transform=_peak_summary,
        why="Fig 4: panel labels, citations, optimum and interpretable flag per curve",
    ),
    Export(
        SET, "results/dms/tables/a31_alpha_curves.tsv", "dms_alpha_curves.tsv.gz",
        why="Fig 4 self-check: a31's own rho-vs-alpha curves, recomputed to 1e-6",
    ),
    Export(
        SET, "results/dms/tables/a31_alpha_vs_dms_curves.tsv", "dms_alpha_bands_published.tsv.gz",
        transform=_bands,
        why="Fig 4: published curves with site-bootstrap band (band reproduction test)",
    ),
    Export(
        SET, "results/dms/tables/a31_alpha_vs_dms_curves_reversible.tsv", "dms_alpha_bands_reversible.tsv.gz",
        transform=_bands,
        why="Fig 4 reversible variant: published curves with band (reproduction test)",
    ),
    Export(
        SET, "data/mutation_spectra/matrices_4x4/influenza_h3n2.csv", "dms_spectrum_influenza_h3n2.tsv.gz",
        why="Fig 4 reversible variant: 4x4 nucleotide matrix, influenza A/H3N2",
    ),
    Export(
        SET, "data/mutation_spectra/matrices_4x4/sars_cov_2.csv", "dms_spectrum_sars_cov_2.tsv.gz",
        why="Fig 4 reversible variant: 4x4 nucleotide matrix, SARS-CoV-2",
    ),
    Export(
        SET, "results/dms/tables/a31_within_stratum.tsv", "dms_within_stratum.tsv.gz",
        transform=_within_stratum,
        why="S12: published within-step-stratum Spearman and site-bootstrap interval (reproduction check)",
    ),
]
