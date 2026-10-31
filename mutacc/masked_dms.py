"""Masked marginals for the DMS figures.

Since 19 Oct 2026 every DMS comparison in this manuscript -- Figure 4, the SARS-CoV-2
supplementary panels and the within-stratum figure -- reads the model's probability for a
replacement from a **masked marginal**: the site being scored is replaced by the mask token and
predicted from its context, so the residue actually there cannot inform its own prediction.

The earlier version used the **wild-type (unmasked) marginal**, read from a single forward pass
of the intact reference in which the model does see that residue. That estimator is not neutral
between the models being compared. A model fine-tuned on one protein family can largely copy a
residue it is shown, which squeezes the probability it has left for the *alternatives* -- and it
is the ranking of those alternatives that every one of these figures scores. An untuned model
has less to copy, so scoring both unmasked flatters the untuned arm. The gap is not small: on
Figure 4's cell-entry panel masking lifts the fine-tune by 0.061 in rho at alpha = 0 against
0.010 for the base, which is most of the apparent deficit the unmasked panel showed.

What the published figures are drawn from is therefore the masked column, and the unmasked one
is kept beside it for two jobs that still need it:

* **the gate.** The archive's own curves were computed unmasked, so reproducing them is only
  possible from the unmasked column. Every figure that switched still recomputes its unmasked
  arm and checks it against the archive before drawing the masked one. That is what keeps
  "masking is the only thing that changed" a measured statement rather than an assumption.
* **Figure SO**, which draws both estimators side by side and is the record of the choice.

The probabilities themselves need a GPU and are produced offline, one forward pass per site per
model, by ``dev/masked_dms/embed_masked_profile.py`` (ESM-2, influenza backbones) and
``dev/masked_dms/embed_masked_profile_esmc.py`` (ESM-C, SARS-CoV-2 backbones). Their output is
collapsed to one shipped table by ``tools/build_masked_dms_scores.py`` so that nothing in
``figures/`` depends on a GPU or on the 20 x L profiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: The 20 canonical residues, in the row order the profile CSVs are written with.
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

#: (dataset, ``model_name`` as ``dms_scores`` spells it) -> the checkpoint label the profile
#: files are named by. ``model_name`` is the reader-facing prose name attached in
#: ``tools/exports/fig04.py``; the profile file is named for the checkpoint that produced it.
PROFILE_LABEL = {
    ("MA22", "ESM-2 650M base"): "ESM2_650M_base",
    ("MA22", "HA-80 fine-tuned"): "ESM2_650M_HA80",
    ("Perth09", "ESM-2 650M base"): "ESM2_650M_base",
    ("Perth09", "HA-80 fine-tuned"): "ESM2_650M_HA80",
    ("BA2", "ESM-C 600M base"): "ESMC_600M_base",
    ("BA2", "SC2 fine-tuned"): "ESMC_600M_SC2ft",
    ("XBB15", "ESM-C 600M base"): "ESMC_600M_base",
    ("XBB15", "SC2 fine-tuned"): "ESMC_600M_SC2ft",
}

#: log10 of a cached probability is a float32 round trip, so an unmasked arm recomputed from a
#: profile can differ from the shipped ``log_plm`` in about the sixth decimal. Anything larger
#: means the profile and the table are not describing the same scoring run, and no comparison
#: between them is like for like.
UNMASKED_TOL = 1e-4

#: The column ``dms_scores_masked`` carries the masked value in.
MASKED_COLUMN = "log_plm_masked"


#: The two estimators, in the order ``--scoring`` offers them. ``masked`` is what the
#: manuscript reports; ``unmasked`` redraws the version published before 19 Oct 2026.
SCORING_CHOICES = ("masked", "unmasked")


def add_scoring_argument(parser) -> None:
    """Add ``--scoring`` to a figure's parser, worded the same way everywhere.

    One definition rather than one per switched figure: the wording is what tells a reader
    running the script which arm they are about to draw, and three copies of it would drift.
    """
    parser.add_argument("--scoring", choices=SCORING_CHOICES, default=SCORING_CHOICES[0],
                        help="which model probability to draw: the masked marginal the "
                             "manuscript reports (default), or the wild-type marginal used "
                             "before 19 Oct 2026")


def profile_log_plm(profiles: Path, dataset: str, label: str, kind: str,
                    sites: np.ndarray, aas: np.ndarray) -> np.ndarray:
    """log10 probability of each (site, replacement), read out of a 20 x L profile CSV.

    log10, not log: that is the scale ``log_plm`` is on. The sweep is a rank correlation, so the
    base of the log cannot move rho on its own -- but it scales ``log_plm`` against ``log_acc``,
    and alpha is exactly that trade-off, so a natural log here would silently rescale the alpha
    axis against the published panel.
    """
    path = Path(profiles) / f"{dataset}__{label}__{kind}_profile.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; the masked profiles are built on a GPU by "
            f"dev/masked_dms/embed_masked_profile*.py")
    prof = pd.read_csv(path, index_col=0)
    prof.columns = prof.columns.astype(int)
    prof = prof.loc[AA20]
    if list(prof.columns) != list(range(1, prof.shape[1] + 1)):
        raise ValueError(f"{path} is not 1-indexed by site")
    row = {a: i for i, a in enumerate(prof.index)}
    unknown = sorted(set(aas) - set(row))
    if unknown:
        raise ValueError(f"replacement(s) {unknown} are not rows of {path}")
    if sites.min() < 1 or sites.max() > prof.shape[1]:
        raise ValueError(
            f"sites {sites.min()}-{sites.max()} fall outside the {prof.shape[1]}-site {path.name}")
    return np.log10(prof.to_numpy())[[row[a] for a in aas], sites - 1]


def build_masked_column(scores: pd.DataFrame, profiles: Path) -> tuple[pd.DataFrame, float]:
    """``dataset``/``model_name``/``site``/``aa`` plus the masked log10 probability.

    Returns the table and the worst disagreement between the *unmasked* profile and the shipped
    ``log_plm``, which is the gate: if the recomputed unmasked arm does not reproduce what the
    published figures were drawn from, then any difference the masked arm shows could be a
    tokeniser, a checkpoint or an off-by-one instead of masking.
    """
    masked = np.full(len(scores), np.nan)
    log_plm = scores["log_plm"].to_numpy(float)
    worst = 0.0
    for (dataset, model_name), label in PROFILE_LABEL.items():
        keep = ((scores["dataset"] == dataset) & (scores["model_name"] == model_name)).to_numpy()
        if not keep.any():
            raise ValueError(f"dms_scores has no rows for {dataset}/{model_name}")
        sites = scores["site"].to_numpy(int)[keep]
        aas = scores["aa"].to_numpy()[keep]
        check = profile_log_plm(profiles, dataset, label, "unmasked", sites, aas)
        worst = max(worst, float(np.max(np.abs(check - log_plm[keep]))))
        masked[keep] = profile_log_plm(profiles, dataset, label, "masked", sites, aas)
    if worst > UNMASKED_TOL:
        raise ValueError(
            f"the recomputed unmasked score differs from the shipped log_plm by {worst:.3g} "
            f"(> {UNMASKED_TOL:g}); the masked comparison would not be like for like")
    if not np.isfinite(masked).all():
        missing = scores.loc[~np.isfinite(masked), ["dataset", "model_name"]].drop_duplicates()
        raise ValueError(
            f"no masked profile covers {missing.to_dict('records')}; add them to PROFILE_LABEL")
    out = scores[["dataset", "model_name", "site", "aa"]].copy()
    out[MASKED_COLUMN] = masked
    return out, worst


def apply_masked(scores: pd.DataFrame, masked: pd.DataFrame) -> pd.DataFrame:
    """``scores`` with ``log_plm`` replaced by the masked marginal, row order preserved.

    Row order matters and is asserted rather than trusted: the site bootstrap consumes its
    random draws arm by arm in table order, so a reordered join would move every band in the
    figure while leaving the curves themselves correct -- a difference that would look like a
    result.
    """
    key = ["dataset", "model_name", "site", "aa"]
    missing = [c for c in key + [MASKED_COLUMN] if c not in masked.columns]
    if missing:
        raise ValueError(f"dms_scores_masked lacks column(s) {missing}")
    if masked.duplicated(key).any():
        raise ValueError("dms_scores_masked has duplicate (dataset, model, site, replacement) rows")
    merged = scores.merge(masked, on=key, how="left", sort=False)
    if len(merged) != len(scores):
        raise ValueError(
            f"the masked join changed the row count ({len(scores)} -> {len(merged)}); "
            f"dms_scores_masked does not key one-to-one against dms_scores")
    absent = merged[MASKED_COLUMN].isna()
    if absent.any():
        gaps = merged.loc[absent, ["dataset", "model_name"]].drop_duplicates()
        raise ValueError(
            f"{int(absent.sum())} replacement(s) have no masked probability, in "
            f"{gaps.to_dict('records')}; dms_scores_masked is stale against dms_scores")
    out = scores.copy()
    out["log_plm"] = merged[MASKED_COLUMN].to_numpy()
    return out
