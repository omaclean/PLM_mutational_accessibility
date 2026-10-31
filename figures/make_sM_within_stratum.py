"""Figure SM (stem ``sM``): does the model's link to measured fitness survive inside every
nucleotide-step stratum?

    python figures/make_sM_within_stratum.py --guide guides/h3n2_13lineage.yaml

Mutational accessibility and the model score are both shaped by the genetic code: a
replacement one nucleotide change away is more accessible and usually more favoured by the
model. A model-vs-fitness correlation could therefore be that shared dependence. Within one
stratum every replacement is the same number of nucleotide steps away, so a correlation that
survives there is not an artefact of mutational supply.

Reads ``dms_scores`` (one row per scanned replacement and model), ``dms_scores_masked``,
``dms_within_stratum`` (the published correlations and their site-bootstrap intervals) and
``dms_peak_summary`` (panel names, citations, model roles, the ``interpretable`` flag).

The model probability is the **masked marginal**; ``mutacc/masked_dms.py`` says why, and why
the choice is not a convention. ``--scoring unmasked`` redraws the version published before
19 Oct 2026, which is the arm ``dms_within_stratum`` holds. Because that table is the only
fixed point there is, both of the checks below run on the unmasked arm whichever arm is drawn
-- and the masked bars, having no shipped table, carry intervals recomputed here.

Two checks the old figure never had:

* every plotted correlation is recomputed from ``dms_scores`` and must match the shipped
  table to 1e-9, so the table cannot drift from the scores;
* the old S12 headline, that fine-tuning differs only for some assays, was read off
  non-overlapping intervals. Overlap is not a test. Here the difference (fine-tuned minus
  base) gets its own interval from a bootstrap that resamples SITES and applies the same
  resample to both models, which is the pairing that matters: both models are scored on
  the same replacements.

The strip along the bottom asks a second question: once a stratum has fixed how many
nucleotide changes a replacement needs, is there any accessibility weighting left to do? The
alpha sweep of figure 4 is refitted inside each stratum and the spread of the resulting best
alpha is drawn as a violin, one per group. The numbers behind it, including bootstrap
intervals, go to ``sM_peak_alpha_by_stratum``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, rankdata, spearmanr

from mutacc import style
from mutacc.dms_sweep import MARKED_GAIN, curve_flags, curve_has_signal, sweep_curve
from mutacc.guide import Guide, parse_figure_args
from mutacc.io import write_caption, write_values
from mutacc.masked_dms import add_scoring_argument, apply_masked

STEM = "sM_within_stratum"
ALPHA_STEM = "sM_peak_alpha_by_stratum"
STRATA = (1, 2, 3)
SELF_CHECK_TOL = 1e-9
N_BOOT = 400
SEED = 20261009
ROLE_ORDER = ("base", "finetune")
ROLE_LABEL = {"base": "Base model", "finetune": "Fine-tuned model"}
BAR_WIDTH = 0.38
FITNESS_COLOUR = style.MEASURED   # the measured phenotype, not a model series
NOT_FITNESS_NOTE = "measures receptor binding, not fitness"

# --- distance groups across panels A-J ------------------------------------------------
POOLED = "pooled"
GROUPS = (POOLED, 1, 2, 3)
GROUP_LABEL = {POOLED: "pooled", 1: "1 nt", 2: "2 nt", 3: "3 nt"}
ALPHA_LIMIT = 1.0
ALPHA_STEP = 0.01
#: The bootstrap runs on a coarser grid. The optimum of a resampled curve moves by far more
#: than 0.02, so a finer grid there would buy nothing and cost twice the time.
BOOT_STEP = 0.02
ALPHA_N_BOOT = 200
ALPHA_SEED = 20261013
#: Fewer scored replacements than this is a broken join, not a thin stratum.
MIN_ROWS = 100
TEST_STEM = "sM_dms_stratum_tests"
#: Significance of the measured phenotype across strata. The omnibus is Kruskal-Wallis over the
#: three strata only -- the pooled group is those same replacements again, so including it would
#: compare a set with itself. Pairs are Holm-corrected within a panel, three of them.
TEST_ALPHA = 0.05
STAR_LEVELS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
#: Fewer than this in a stratum and the rank tests are not worth reporting for that panel.
MIN_TEST_N = 10
VIOLIN_JITTER = 0.055


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def panels(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, trait) panel, in the order the companion figure draws them."""
    pub = summary.loc[summary["spectrum"] == "published"]
    cols = ["dataset", "trait", "virus_tag", "panel_label", "citation"]
    out = pub[cols].drop_duplicates(["dataset", "trait"]).reset_index(drop=True)
    # Binding assays are not fitness: many replacements raise binding. (The summary's own
    # ``interpretable`` column is a different thing, whether a curve's optimum is meaningful.)
    out["binding"] = out["trait"].str.lower().str.contains("binding")
    return out


def role_of_model(summary: pd.DataFrame) -> dict[tuple[str, str], str]:
    pub = summary.loc[summary["spectrum"] == "published"]
    return {(r.dataset, r.model_name): r.model_role for r in pub.drop_duplicates(["dataset", "model_name"]).itertuples()}


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def recompute(scores: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Spearman(model score, measured effect) per panel, model and stratum, from the scores."""
    rows = []
    for p in panel.itertuples():
        sub = scores.loc[scores["dataset"] == p.dataset]
        for model, m in sub.groupby("model_name", sort=False):
            for stratum in STRATA:
                cell = m.loc[(m["nt_steps"] == stratum) & m["log_plm"].notna() & m[p.trait].notna()]
                if len(cell) < 3:
                    continue
                rows.append({"dataset": p.dataset, "trait": p.trait, "model_name": model, "stratum": stratum,
                             "n_recomputed": len(cell), "n_sites_recomputed": cell["site"].nunique(),
                             "rho_recomputed": _rho(cell["log_plm"], cell[p.trait])})
    return pd.DataFrame(rows)


def check_against_table(recomputed: pd.DataFrame, table: pd.DataFrame, tol: float = SELF_CHECK_TOL) -> float:
    """Largest disagreement between recomputed and shipped correlations. Raises above ``tol``."""
    key = ["dataset", "trait", "model_name", "stratum"]
    merged = table.merge(recomputed, on=key, how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError(f"{int((merged['_merge'] != 'both').sum())} (assay, model, stratum) cells differ between the "
                         "shipped table and the scores")
    gap = float((merged["rho"] - merged["rho_recomputed"]).abs().max())
    if gap > tol:
        raise ValueError(f"shipped within-stratum table disagrees with the scores by {gap:.2e} (tolerance {tol:g})")
    if not (merged["n"] == merged["n_recomputed"]).all() or not (merged["n_sites"] == merged["n_sites_recomputed"]).all():
        raise ValueError("shipped within-stratum table has different replacement or site counts from the scores")
    return gap


def recompute_with_band(scores: pd.DataFrame, panel: pd.DataFrame, *,
                        n_boot: int = N_BOOT, seed: int = SEED,
                        groups: tuple = GROUPS) -> pd.DataFrame:
    """The same correlations as ``recompute``, with a site bootstrap interval of their own.

    The shipped ``dms_within_stratum`` table carries the archive's intervals, and the archive
    scored the model unmasked. Under the masked estimator there is no such table to read, so the
    panel's bars and whiskers are both computed here.

    The resampling is the one the paired difference uses: SITES with replacement, because
    replacements at one site share a backbone and are not independent. It is *not* the archive's
    own draw, so these intervals are not expected to reproduce the shipped ones to the last
    decimal even where the point estimates agree; the caption says the interval is recomputed.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in panel.itertuples():
        sub = scores.loc[scores["dataset"] == p.dataset]
        for model, m in sub.groupby("model_name", sort=False):
            for group in groups:
                c = m if group == POOLED else m.loc[m["nt_steps"] == group]
                cell = c.loc[c["log_plm"].notna() & c[p.trait].notna(),
                             ["site", "log_plm", p.trait]]
                if len(cell) < 3:
                    continue
                x = cell["log_plm"].to_numpy(float)
                y = cell[p.trait].to_numpy(float)
                sites = cell["site"].to_numpy()
                order = np.argsort(sites, kind="stable")
                _, start = np.unique(sites[order], return_index=True)
                group_splits = np.split(order, start[1:])
                draws = np.empty(n_boot)
                for b in range(n_boot):
                    idx = np.concatenate([group_splits[i] for i in rng.integers(0, len(group_splits), len(group_splits))])
                    draws[b] = _pearson(rankdata(x[idx]), rankdata(y[idx]))
                lo, hi = np.percentile(draws, [2.5, 97.5])
                rows.append({"dataset": p.dataset, "trait": p.trait, "model_name": model,
                             "stratum": group, "n": len(cell), "n_sites": len(group_splits),
                             "rho": _rho(x, y), "ci_lo": float(lo), "ci_hi": float(hi),
                             "n_boot": n_boot})
    return pd.DataFrame(rows)


def dms_norm_by_stratum(scores: pd.DataFrame, panel: pd.DataFrame,
                        groups: tuple = GROUPS) -> pd.DataFrame:
    """Mean and SEM of normalized phenotype (z-score) per panel and distance group."""
    rows = []
    for p in panel.itertuples():
        sub = scores.loc[scores["dataset"] == p.dataset]
        unique_muts = sub.drop_duplicates(["site", "aa"]).copy()
        y_all = unique_muts.loc[unique_muts[p.trait].notna(), p.trait].to_numpy(float)
        y_mean = np.mean(y_all)
        y_std = np.std(y_all) if np.std(y_all) > 0 else 1.0

        for group in groups:
            c = unique_muts if group == POOLED else unique_muts.loc[unique_muts["nt_steps"] == group]
            y_grp = c.loc[c[p.trait].notna(), p.trait].to_numpy(float)
            if len(y_grp) == 0:
                continue
            z_grp = (y_grp - y_mean) / y_std
            rows.append({
                "dataset": p.dataset, "trait": p.trait, "group": str(group),
                "mean_z": float(np.mean(z_grp)),
                "sem_z": float(np.std(z_grp) / np.sqrt(len(z_grp))),
            })
    return pd.DataFrame(rows)


def _stars(p: float) -> str:
    """``*``, ``**`` or ``***`` for a p-value, empty above ``TEST_ALPHA``."""
    if not np.isfinite(p):
        return ""
    for threshold, mark in STAR_LEVELS:
        if p < threshold:
            return mark
    return ""


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, monotone and capped at 1."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    # Holm steps DOWN the sorted p-values: the smallest is multiplied by m, the next by m-1,
    # and the running maximum keeps the result monotone. (Multiplying the largest by m instead
    # is Benjamini-Hochberg's shape run backwards, and it is not this correction.)
    weights = len(p) - np.arange(len(p))
    out = np.empty_like(p)
    out[order] = np.minimum(np.maximum.accumulate(p[order] * weights), 1.0)
    return out


def dms_stratum_tests(scores: pd.DataFrame, panel: pd.DataFrame,
                      strata: tuple = STRATA) -> pd.DataFrame:
    """Does the measured phenotype itself differ across the nucleotide-step strata?

    The purple curve is a mean, and a reader cannot see from three means with SEM whiskers
    whether they are drawn from the same distribution. Kruskal-Wallis asks that question on the
    ranks of the measured effect, which is the right footing: DMS scales differ between assays
    and several are bounded, so a test on the raw values would be testing the scale.

    The omnibus runs over the strata only. The pooled group is the union of those same
    replacements, so putting it in would compare each stratum against itself.

    Where the omnibus is significant, the three pairwise Mann-Whitney tests say *which* stratum
    carries it. They are Holm-corrected within a panel. A stratum is flagged -- the star drawn
    over its marker -- only when it separates from BOTH of the others, which is what "the one
    that is different" has to mean with three groups; a panel whose strata simply rank in order
    has no single odd one out and is left unflagged, with the omnibus star still drawn.

    One row per test: ``test`` is ``kruskal`` (``group_b`` empty) or ``mannwhitney``.
    """
    rows = []
    for p in panel.itertuples():
        sub = scores.loc[scores["dataset"] == p.dataset]
        # The same de-duplication the purple curve uses: a replacement is measured once, but
        # appears in the scores once per model.
        unique_muts = sub.drop_duplicates(["site", "aa"])
        measured = unique_muts.loc[unique_muts[p.trait].notna()]
        by_stratum = {s: measured.loc[measured["nt_steps"] == s, p.trait].to_numpy(float)
                      for s in strata}
        if any(len(v) < MIN_TEST_N for v in by_stratum.values()):
            continue
        stat, p_kw = kruskal(*(by_stratum[s] for s in strata))
        base = {"dataset": p.dataset, "trait": p.trait, "panel_label": p.panel_label,
                "binding": bool(p.binding)}

        pairs = [(a, b) for i, a in enumerate(strata) for b in strata[i + 1:]]
        pair_p = np.array([mannwhitneyu(by_stratum[a], by_stratum[b],
                                        alternative="two-sided").pvalue for a, b in pairs])
        pair_holm = _holm(pair_p) if p_kw < TEST_ALPHA else np.full(len(pairs), np.nan)

        flagged = []
        if p_kw < TEST_ALPHA:
            for s in strata:
                others = [i for i, (a, b) in enumerate(pairs) if s in (a, b)]
                if all(pair_holm[i] < TEST_ALPHA for i in others):
                    flagged.append(s)
        rows.append({**base, "test": "kruskal", "group_a": "|".join(str(s) for s in strata),
                     "group_b": "", "n_a": int(sum(len(v) for v in by_stratum.values())), "n_b": 0,
                     "statistic": float(stat), "p_value": float(p_kw), "p_holm": float(p_kw),
                     "stars": _stars(p_kw),
                     "flagged_groups": "|".join(str(s) for s in flagged)})
        for (a, b), raw, adj in zip(pairs, pair_p, pair_holm):
            u = mannwhitneyu(by_stratum[a], by_stratum[b], alternative="two-sided")
            rows.append({**base, "test": "mannwhitney", "group_a": str(a), "group_b": str(b),
                         "n_a": len(by_stratum[a]), "n_b": len(by_stratum[b]),
                         "statistic": float(u.statistic), "p_value": float(raw),
                         "p_holm": float(adj), "stars": _stars(adj),
                         "flagged_groups": "|".join(str(s) for s in flagged)})
    return pd.DataFrame(rows)


def stratum_stars(tests: pd.DataFrame, dataset: str, trait: str) -> tuple[str, dict]:
    """``(omnibus stars, {stratum: stars})`` for one panel; the stars the panel draws."""
    cell = tests.loc[(tests["dataset"] == dataset) & (tests["trait"] == trait)]
    if cell.empty:
        return "", {}
    kw = cell.loc[cell["test"] == "kruskal"].iloc[0]
    if not kw["stars"]:
        return "", {}
    flagged = [int(s) for s in kw["flagged_groups"].split("|") if s]
    pairs = cell.loc[cell["test"] == "mannwhitney"]
    marks = {}
    for s in flagged:
        involved = pairs.loc[(pairs["group_a"] == str(s)) | (pairs["group_b"] == str(s))]
        # The star over a stratum is its WEAKER separation: it is flagged for differing from
        # both of the others, so the claim is only as strong as the lesser of the two.
        marks[s] = _stars(float(involved["p_holm"].max()))
    return str(kw["stars"]), marks


# ---------------------------------------------------------------------------
# Paired difference
# ---------------------------------------------------------------------------

def paired_difference(scores: pd.DataFrame, panel: pd.DataFrame, roles: dict, *,
                      n_boot: int = N_BOOT, seed: int = SEED,
                      groups: tuple = GROUPS) -> pd.DataFrame:
    """Fine-tuned minus base Spearman, per panel and stratum, with a paired site bootstrap.

    Both models score the same replacements, so each bootstrap draw resamples sites once and
    applies that draw to both. This removes the variation the two models share, which is why
    two overlapping interval bars say little about whether their difference is zero.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in panel.itertuples():
        sub = scores.loc[scores["dataset"] == p.dataset]
        by_role = {}
        for model, m in sub.groupby("model_name", sort=False):
            by_role[roles[(p.dataset, model)]] = m
        if set(by_role) != set(ROLE_ORDER):
            raise ValueError(f"{p.dataset}: expected one base and one fine-tuned model, found {sorted(by_role)}")
        for group in groups:
            cells = {}
            for role, m in by_role.items():
                c = m if group == POOLED else m.loc[m["nt_steps"] == group]
                c = c.loc[c["log_plm"].notna() & c[p.trait].notna(),
                          ["site", "aa", "log_plm", p.trait]]
                cells[role] = c.set_index(["site", "aa"])
            joint = cells["base"].index.intersection(cells["finetune"].index)
            if len(joint) < 3:
                continue
            if len(joint) != len(cells["base"]) or len(joint) != len(cells["finetune"]):
                raise ValueError(f"{p.dataset}/{p.trait}/{group}: the two models do not score the same replacements")
            base, ft = cells["base"].loc[joint], cells["finetune"].loc[joint]
            y = base[p.trait].to_numpy()
            if not np.allclose(y, ft[p.trait].to_numpy()):
                raise ValueError(f"{p.dataset}/{p.trait}/{group}: measured effects differ between the models")
            sites = joint.get_level_values("site").to_numpy()
            order = np.argsort(sites, kind="stable")
            uniq, start = np.unique(sites[order], return_index=True)
            group_splits = np.split(order, start[1:])
            xb, xf = base["log_plm"].to_numpy(), ft["log_plm"].to_numpy()
            draws = np.empty(n_boot)
            for b in range(n_boot):
                idx = np.concatenate([group_splits[i] for i in rng.integers(0, len(group_splits), len(group_splits))])
                ry = rankdata(y[idx])
                draws[b] = _pearson(rankdata(xf[idx]), ry) - _pearson(rankdata(xb[idx]), ry)
            point = _rho(xf, y) - _rho(xb, y)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            rows.append({"dataset": p.dataset, "trait": p.trait, "panel_label": p.panel_label, "stratum": group,
                         "n": len(joint), "n_sites": len(group_splits), "difference": point, "ci_lo": lo, "ci_hi": hi,
                         "n_boot": n_boot, "binding": bool(p.binding)})
    return pd.DataFrame(rows)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


# ---------------------------------------------------------------------------
# The best accessibility weight, refitted inside each stratum
# ---------------------------------------------------------------------------

def alpha_grid(step: float = ALPHA_STEP) -> np.ndarray:
    return np.round(np.arange(-ALPHA_LIMIT, ALPHA_LIMIT + step / 2, step), 6)


def _boot_alpha_ci(c: pd.DataFrame, trait: str, rng: np.random.Generator, *,
                   n_boot: int) -> tuple[float, float]:
    """Percentile interval for the best alpha, resampling SITES with replacement.

    The interval is over the LOCATION of the optimum, so it is wide wherever the curve is
    flat near its top. That is the honest reading: a flat curve does not place alpha.
    """
    grid = alpha_grid(BOOT_STEP)
    uniq, inverse = np.unique(c["site"].to_numpy(), return_inverse=True)
    blocks = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    x, a, y = c["log_plm"].to_numpy(), c["log_acc"].to_numpy(), c[trait].to_numpy()
    best = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(uniq), len(uniq))])
        rho = sweep_curve(x[idx], a[idx], y[idx], grid)
        best[b] = grid[int(np.nanargmax(rho))] if np.isfinite(rho).any() else np.nan
    best = best[np.isfinite(best)]
    if best.size < 2:
        return float("nan"), float("nan")
    return float(np.percentile(best, 2.5)), float(np.percentile(best, 97.5))


def alpha_by_stratum(scores: pd.DataFrame, panel: pd.DataFrame, roles: dict, *,
                     n_boot: int = ALPHA_N_BOOT, seed: int = ALPHA_SEED,
                     groups: tuple = GROUPS) -> pd.DataFrame:
    """Refit ``log_plm + alpha * log_acc`` against the measured effect inside each group.

    One row per (panel, model, group), where group is all replacements together or one
    nucleotide-step stratum. Binding assays are fitted and kept, so the table shows why they
    are set aside, but they get no bootstrap: their curves have no optimum to place.

    ``groups`` and ``n_boot`` narrow the work for the gate, which only needs the pooled
    point estimate: the full refit is the slowest thing in this script and the masked arm
    would otherwise pay for it twice.
    """
    rng = np.random.default_rng(seed)
    grid = alpha_grid()
    rows = []
    for p in panel.itertuples():
        sub = scores.loc[scores["dataset"] == p.dataset]
        for model in sub["model_name"].unique():
            m = sub.loc[sub["model_name"] == model]
            for group in groups:
                c = m if group == POOLED else m.loc[m["nt_steps"] == group]
                c = c.loc[c["log_plm"].notna() & c["log_acc"].notna() & c[p.trait].notna()]
                if len(c) < MIN_ROWS:
                    continue
                rho = sweep_curve(c["log_plm"].to_numpy(), c["log_acc"].to_numpy(),
                                  c[p.trait].to_numpy(), grid)
                f = curve_flags(rho, grid)
                sd = float(np.std(c["log_acc"].to_numpy()))
                lo, hi = (float("nan"), float("nan")) if p.binding else \
                    _boot_alpha_ci(c, p.trait, rng, n_boot=n_boot)
                rows.append({
                    "dataset": p.dataset, "trait": p.trait, "panel_label": p.panel_label,
                    "model_name": model, "role": roles[(p.dataset, model)],
                    "group": str(group), "n": len(c), "n_sites": int(c["site"].nunique()),
                    "sd_log_acc": sd, "peak_alpha": f["best_alpha"],
                    "peak_alpha_ci_lo": lo, "peak_alpha_ci_hi": hi,
                    # alpha is in units of 1 / log-accessibility and that scale shrinks inside
                    # a stratum, so raw alpha is NOT comparable across groups. alpha* x SD is
                    # the shift the weighting actually applies, and that one is comparable.
                    "scaled_weight": f["best_alpha"] * sd,
                    "rho_alpha0": f["rho_alpha0"], "rho_peak": f["rho_best"],
                    "gain": f["rho_best"] - f["rho_alpha0"], "boundary": f["boundary"],
                    "weak_signal": f["weak_signal"], "interpretable": f["interpretable"],
                    # ``interpretable`` is curve_flags' own rule and is kept for comparison, but
                    # it is not what this figure counts on; see ``violin_arms``.
                    "max_abs_rho": float(np.nanmax(np.abs(rho))),
                    "has_signal": bool(curve_has_signal(rho)),
                    "readable": bool((not p.binding)
                                     and (curve_has_signal(rho)
                                          or f["rho_best"] - f["rho_alpha0"] >= MARKED_GAIN)),
                    "n_boot": 0 if p.binding else n_boot, "binding": bool(p.binding),
                })
    return pd.DataFrame(rows)


def check_pooled_alpha(alpha: pd.DataFrame, summary: pd.DataFrame,
                       tol: float = ALPHA_STEP + 1e-9) -> float:
    """The refitted pooled optimum must reproduce the shipped ``peak_alpha``, to one grid step."""
    pub = summary.loc[summary["spectrum"] == "published"]
    key = ["dataset", "trait", "model_name"]
    mine = alpha.loc[alpha["group"] == POOLED, key + ["peak_alpha"]]
    merged = pub[key + ["peak_alpha"]].merge(mine, on=key, how="outer", indicator=True,
                                             suffixes=("_shipped", "_here"))
    if (merged["_merge"] != "both").any():
        raise ValueError(f"{int((merged['_merge'] != 'both').sum())} (assay, model) arms differ "
                         "between dms_peak_summary and the pooled refit")
    gap = float((merged["peak_alpha_shipped"] - merged["peak_alpha_here"]).abs().max())
    if gap > tol:
        raise ValueError(f"refitted pooled optimum differs from the shipped peak_alpha by "
                         f"{gap:.3f} (tolerance {tol:g})")
    return gap


def violin_arms(alpha: pd.DataFrame) -> pd.DataFrame:
    """The arms this panel draws and counts: every assay that measures a phenotype running
    *with* fitness. Only receptor binding is left out.

    Until 22 Oct 2026 this also applied ``curve_flags``' ``interpretable``, which dropped two
    further kinds of fit, and both drops were wrong here:

    * an optimum at the **end of the swept range** was discarded as a bound rather than a
      turning point. Where the curve runs plainly to that edge it is a result, and discarding
      it removed the largest single gain in the one-change stratum.
    * **weakness was judged at alpha = 0 only**, so a curve that merely passes through a
      shallow point there and peaks elsewhere was called empty.

    Together those made the denominator differ between strata (13, 12, 13, 13) for a reason no
    reader of the count could see, and they were not neutral: a fit pinned at +1 is evidence
    *against* a negative weight, and those were the ones removed.

    Nothing is dropped now. The ``readable`` column still marks the arms whose sweep found
    nothing -- no correlation anywhere and no gain from moving alpha -- but those are kept and
    counted, because their alpha* sits at or beside zero and "no preference" is the honest
    reading of them, not an absent one. A violin point at zero says exactly that; it is only on
    a *curve* that drawing a marker would assert an optimum that is not there, which is why
    Figure SP suppresses those two markers and this panel does not.
    """
    return alpha.loc[~alpha["binding"]]


def summarise(diff: pd.DataFrame, table: pd.DataFrame, panel: pd.DataFrame, roles: dict) -> dict:
    """Counts the caption quotes, over the assays that measure fitness (not receptor binding)."""
    d = diff.loc[~diff["binding"]]
    excl = (d["ci_lo"] > 0) | (d["ci_hi"] < 0)
    fit = panel.loc[~panel["binding"], ["dataset", "trait"]]
    bars = table.merge(fit, on=["dataset", "trait"])
    bars["role"] = [roles[(a, m)] for a, m in zip(bars["dataset"], bars["model_name"])]
    above = {role: int((g["ci_lo"] > 0).sum()) for role, g in bars.groupby("role")}
    return {
        "cells": len(d), "excluding_zero": int(excl.sum()),
        "ft_higher": int((d["ci_lo"] > 0).sum()), "ft_lower": int((d["ci_hi"] < 0).sum()),
        "bars": {role: int(len(g)) for role, g in bars.groupby("role")}, "bars_above_zero": above,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_alpha_strip(ax, alpha: pd.DataFrame) -> None:
    """Violins of the best alpha per group and model role, over fitness arms with a readable optimum.

    Two violins per distance group, one for each PLM role, colour-coded:
    base model on the left, fine-tuned model on the right.
    """
    arms = violin_arms(alpha)
    rng = np.random.default_rng(ALPHA_SEED)
    offsets = {"base": -0.18, "finetune": 0.18}
    violin_width = 0.30
    jitter = 0.04

    for i, g in enumerate(GROUPS):
        sub = arms.loc[arms["group"] == str(g)]
        for role in ROLE_ORDER:
            v = sub.loc[sub["role"] == role, "peak_alpha"].to_numpy(float)
            pos = i + offsets[role]
            colour = style.colour_for_role(role)
            if len(v) >= 3:
                parts = ax.violinplot([v], positions=[pos], widths=violin_width,
                                      showextrema=False, showmedians=False)
                for body in parts["bodies"]:
                    body.set_facecolor(colour)
                    body.set_edgecolor(colour)
                    body.set_linewidth(0.8)
                    body.set_alpha(0.35)
            if len(v):
                ax.scatter(pos + rng.uniform(-jitter, jitter, len(v)), v, s=9,
                           color=colour, linewidth=0, alpha=0.9, zorder=3)
                med = float(np.median(v))
                ax.plot([pos - 0.09, pos + 0.09], [med, med], color=style.INK, linewidth=1.2, zorder=4)

    ax.axhline(0.0, color=style.INK_MUTED, linewidth=0.6, zorder=1)
    counts = arms.groupby("group")["peak_alpha"].size()
    sd = arms.groupby("group")["sd_log_acc"].median()
    x = np.arange(len(GROUPS), dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{GROUP_LABEL[g]}\n{int(counts.get(str(g), 0))} arms, SD {sd.get(str(g), float('nan')):.1f}"
                        for g in GROUPS])
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("α*", fontsize=7)
    # pad leaves room for the panel letter, which sits above the left corner of this axes.
    ax.set_title("Best accessibility weight α* when the sweep is refitted inside one stratum "
                 "(receptor-binding assays excluded; base vs fine-tuned)", loc="left", fontsize=7, pad=9)


def draw(table: pd.DataFrame, panel: pd.DataFrame, roles: dict, alpha: pd.DataFrame | None = None,
         dms_norm: pd.DataFrame | None = None, dms_tests: pd.DataFrame | None = None):
    ncol = 3
    nrow = int(np.ceil(len(panel) / ncol))
    strip = alpha is not None and not violin_arms(alpha).empty
    fig = style.figure(style.WIDTH_DOUBLE, 2.10 * nrow + 0.4 + (1.50 if strip else 0.0))
    if strip:
        # No hspace here: the house style lays out with constrained layout, which spaces rows
        # from the labels they actually carry. A gridspec hspace on top of it is applied as a
        # fraction of the mean axes height and blew the grid apart.
        gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.80])
        axes = np.asarray([fig.add_subplot(gs[r, c]) for r in range(nrow) for c in range(ncol)])
        ax_strip = fig.add_subplot(gs[nrow, :])
    else:
        axes = np.asarray(fig.subplots(nrow, ncol, sharey=False)).reshape(-1)
        ax_strip = None
    x = np.arange(len(GROUPS), dtype=float)
    handles = {}
    for idx, (ax, p) in enumerate(zip(axes, panel.itertuples())):
        col_idx = idx % ncol
        cell = table.loc[(table["dataset"] == p.dataset) & (table["trait"] == p.trait)].copy()
        cell["role"] = [roles[(p.dataset, m)] for m in cell["model_name"]]
        cell["stratum_str"] = cell["stratum"].astype(str)
        for j, role in enumerate(ROLE_ORDER):
            c = cell.loc[cell["role"] == role].set_index("stratum_str").reindex([str(g) for g in GROUPS])
            off = (j - 0.5) * BAR_WIDTH
            colour = style.colour_for_role(role)
            rho = c["rho"].to_numpy(float)
            err = np.vstack([rho - c["ci_lo"].to_numpy(float), c["ci_hi"].to_numpy(float) - rho])
            bars = ax.bar(x + off, rho, BAR_WIDTH * 0.92, color=colour, linewidth=0, zorder=2)
            ax.errorbar(x + off, rho, yerr=err, fmt="none", ecolor=style.INK, elinewidth=0.8, capsize=1.8,
                        capthick=0.8, zorder=3)
            handles.setdefault(role, bars[0])
        ax.axhline(0.0, color=style.INK_MUTED, linewidth=0.6, zorder=1)
        n_by = cell.groupby("stratum_str")["n"].max().reindex([str(g) for g in GROUPS])
        ax.set_xticks(x)
        labels = [
            GROUP_LABEL[g] if g == POOLED else f"{GROUP_LABEL[g]}\nn={int(n_by.loc[str(g)]):,}"
            for g in GROUPS
        ]
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_xlim(-0.55, len(GROUPS) - 0.45)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Spearman ρ" if col_idx == 0 else "", fontsize=6.8)
        # The virus tag leads the header in bold: the grid mixes influenza and SARS-CoV-2
        # panels, and which virus a panel is about is the first thing a reader needs. It goes
        # inline before the lineage rather than on its own line, which would cost a line of
        # white space in every panel. Mathtext is the only way to bold part of a title.
        ax.set_title(f"$\\bf{{{p.virus_tag}}}$ {p.panel_label}\n{p.citation}", loc="left", fontsize=6.8)
        if p.binding:
            ax.text(0.0, 1.0, NOT_FITNESS_NOTE, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=5.5, color=style.INK_MUTED)

        if dms_norm is not None:
            ax2 = ax.twinx()
            norm_cell = dms_norm.loc[(dms_norm["dataset"] == p.dataset) & (dms_norm["trait"] == p.trait)].set_index("group")
            mz = norm_cell.reindex([str(g) for g in GROUPS])["mean_z"].to_numpy(float)
            sem = norm_cell.reindex([str(g) for g in GROUPS])["sem_z"].to_numpy(float)
            line = ax2.errorbar(x, mz, yerr=sem, fmt="o--", color=FITNESS_COLOUR, markersize=3.2,
                                linewidth=1.0, elinewidth=0.8, capsize=1.8, capthick=0.8,
                                alpha=0.9, zorder=4)
            ax2.set_ylim(-0.25, 0.25)
            ax2.grid(visible=False)
            handles.setdefault("fitness_line", line[0])

            if dms_tests is not None:
                omnibus, marks = stratum_stars(dms_tests, p.dataset, p.trait)
                if omnibus:
                    ax2.text(0.98, 0.95, f"KW {omnibus}", transform=ax2.transAxes,
                             ha="right", va="top", fontsize=5.5, color=FITNESS_COLOUR, zorder=6,
                             bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.75))
                for g, mark in marks.items():
                    i = GROUPS.index(g)
                    # Above the marker and its whisker, and kept inside the axes: the curve
                    # runs off this scale in the steepest panels and a star drawn at the
                    # marker would then sit outside the frame.
                    y = min(mz[i] + sem[i] + 0.035, 0.205)
                    ax2.text(x[i], y, mark, ha="center", va="bottom", fontsize=7,
                             color=FITNESS_COLOUR, zorder=6)

            if col_idx == ncol - 1:
                ax2.spines["right"].set_visible(True)
                ax2.spines["right"].set_color(FITNESS_COLOUR)
                ax2.spines["right"].set_linewidth(0.6)
                ax2.set_ylabel("Mean DMS (z-score)", fontsize=6.5, color=FITNESS_COLOUR)
                ax2.set_yticks([-0.2, 0.0, 0.2])
                ax2.tick_params(axis="y", labelcolor=FITNESS_COLOUR, labelsize=5.5, length=2, pad=1.5)
            else:
                ax2.spines["right"].set_visible(False)
                ax2.set_yticks([])
                ax2.tick_params(axis="y", labelcolor="none", length=0)

    for ax in axes[len(panel):]:
        ax.set_visible(False)
    fig.supylabel("Rank correlation of model score with measured effect (Spearman ρ)", fontsize=8)
    fig.supxlabel("Replacements grouped by the minimum nucleotide changes needed to make them",
                  fontsize=8)
    leg_handles = [handles[r] for r in ROLE_ORDER]
    leg_labels = [ROLE_LABEL[r] for r in ROLE_ORDER]
    if "fitness_line" in handles:
        leg_handles.append(handles["fitness_line"])
        leg_labels.append("Mean DMS fitness (z-score ± SEM)")
    fig.legend(leg_handles, leg_labels, loc="outside upper center", ncol=len(leg_handles), fontsize=6.8)
    style.panel_letters(axes[: len(panel)], dx=-0.18, dy=1.20)
    if ax_strip is not None:
        draw_alpha_strip(ax_strip, alpha)
        # A full-width axes needs a much smaller nudge than a one-third-width one; the grid's
        # dx would put the letter outside the figure.
        style.panel_letters([ax_strip], start=chr(ord("A") + len(panel)), dx=-0.045, dy=1.30)
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.02, wspace=0.02, hspace=0.04)
    return fig


def summarise_alpha(alpha: pd.DataFrame) -> dict:
    """Counts for the alpha sentence, over the fitness arms with a readable optimum."""
    arms = violin_arms(alpha)
    strata = arms.loc[arms["group"] != POOLED]
    pooled = arms.loc[arms["group"] == POOLED]
    dropped = alpha.loc[alpha["binding"]]
    out = {
        "n_strata": len(strata), "n_pooled": len(pooled),
        "neg_strata": int((strata["peak_alpha"] < 0).sum()),
        "neg_pooled": int((pooled["peak_alpha"] < 0).sum()),
        "excl_zero_neg": int((strata["peak_alpha_ci_hi"] < 0).sum()),
        "median_stratum_gain": float(strata["gain"].median()),
        "max_stratum_gain": float(strata["gain"].max()),
        "median_pooled_scaled": float(pooled["scaled_weight"].median()),
        "median_stratum_scaled": float(strata["scaled_weight"].median()),
        "sd_ratio": float(pooled["sd_log_acc"].median() / strata["sd_log_acc"].median()),
        "n_binding_arms": int(len(dropped)),
        "binding_at_edge": int(dropped["boundary"].sum()),
        "binding_assays": sorted(dropped["panel_label"].unique()),
        "n_unreadable_fitness": int((~alpha["binding"] & ~alpha["readable"]).sum()),
        "n_kept_at_edge": int(violin_arms(alpha)["boundary"].sum()),
    }
    by_group = {}
    for g in GROUPS:
        s = arms.loc[arms["group"] == str(g)]
        by_group[str(g)] = {"n": len(s), "negative": int((s["peak_alpha"] < 0).sum()),
                            "median": float(s["peak_alpha"].median()) if len(s) else float("nan")}
    out["by_group"] = by_group
    # The pooled fit separates the two models completely, so one median over both lands in the
    # empty gap between them. Raw alpha is readable here because every pooled arm shares a
    # comparable spread of log(accessibility); only ACROSS groups does alpha need x SD.
    out["pooled_by_role"] = {
        role: {"n": len(s), "negative": int((s["peak_alpha"] < 0).sum()),
               "lo": float(s["peak_alpha"].min()), "hi": float(s["peak_alpha"].max())}
        for role, s in pooled.groupby("role")}
    return out


#: What the caption says the model probability is. The estimator is not a detail a reader can
#: infer from the panel, and the two arms of this figure differ by nothing else.
SCORING_SENTENCE = {
    "masked": ("The model probability is the masked marginal: the scored site is replaced by "
               "the mask token and predicted from its context, so the residue actually there "
               "cannot inform its own prediction."),
    "unmasked": ("The model probability is the wild-type (unmasked) marginal, read from one "
                 "forward pass of the intact sequence, in which the model does see the residue "
                 "it is scoring."),
}


def summarise_tests(tests: pd.DataFrame) -> dict:
    """Counts for the sentence about the purple stars."""
    kw = tests.loc[tests["test"] == "kruskal"]
    sig = kw.loc[kw["p_value"] < TEST_ALPHA]
    flagged = sig.loc[sig["flagged_groups"] != ""]
    return {"n_panels": len(kw), "n_sig": len(sig), "n_flagged": len(flagged),
            "panels_flagged": sorted(flagged["panel_label"].unique())}


def tests_sentence(tests: pd.DataFrame) -> str:
    """The sentence the purple stars need, or nothing if no panel carries one."""
    t = summarise_tests(tests)
    if not t["n_sig"]:
        return ""
    return (f" The purple stars test that curve itself: three means with SEM whiskers do not show "
            f"whether the strata are drawn from the same distribution. A Kruskal-Wallis test asks "
            f"whether the measured effect differs across the three nucleotide-step strata. It is a "
            f"rank test because DMS scales differ between assays and several are bounded, so a "
            f"test on the raw values would partly be testing the scale; and it is run over the "
            f"strata only, the pooled group being the union of those same replacements. Each "
            f"replacement is counted once, not once per model. The test is significant in "
            f"{t['n_sig']} of {t['n_panels']} panels, marked \"KW\" at the top right of the panel. "
            f"Where it is, three pairwise Mann-Whitney tests, Holm-corrected within the panel, "
            f"say which stratum carries it, and a star is drawn over a stratum only where it "
            f"separates from both of the others -- which is what a single odd group out can mean "
            f"when there are three ({t['n_flagged']} panels have one; a panel whose strata simply "
            f"rank in order keeps its \"KW\" mark and gets no star). Stars are * p < 0.05, "
            f"** p < 0.01, *** p < 0.001, and the star over a stratum carries the weaker of its "
            f"two corrected p-values, since the claim is only as strong as its lesser half. Every "
            f"statistic and p-value, corrected and raw, is in {TEST_STEM}_values.tsv.")


def caption(panel: pd.DataFrame, table: pd.DataFrame, diff: pd.DataFrame, roles: dict, n_boot: int,
            alpha: pd.DataFrame | None = None, scoring: str = "masked",
            dms_tests: pd.DataFrame | None = None) -> str:
    c = summarise(diff, table, panel, roles)
    sites = f"{int(table['n_sites'].min()):,} to {int(table['n_sites'].max()):,}"
    ns = f"{int(table['n'].min()):,} to {int(table['n'].max()):,}"
    above = c["bars_above_zero"]
    bars = c["bars"]
    tail = "" if alpha is None else alpha_paragraph(summarise_alpha(alpha), len(panel))
    stars = "" if dms_tests is None or dms_tests.empty else tests_sentence(dms_tests)
    return f"""
    **Model score against measured fitness, within each
    nucleotide-step stratum.**
    Mutational accessibility and the model score are both shaped by the genetic code, so a
    correlation with measured fitness could be that shared dependence. In each panel the
    replacements are split by the minimum number of nucleotide changes needed to make them (1,
    2 or 3), so within a group every replacement is equally easy to reach. Bars are the
    Spearman rank correlation between the model score (the model alone, no accessibility
    weighting) and the effect measured by deep mutational scanning (DMS), for a base model and
    the same model fine-tuned on that virus. {SCORING_SENTENCE[scoring]} Each group holds {ns}
    replacements at {sites} sites.
    Error bars are 95% percentile intervals from a bootstrap that resamples DMS sites, not
    replacements, because replacements at one site share a backbone and are not independent.

    In the assays that measure fitness, the interval lies wholly above zero for
    {above.get('base', 0)} of {bars.get('base', 0)} base-model bars and
    {above.get('finetune', 0)} of {bars.get('finetune', 0)} fine-tuned-model bars.
    Overlapping error bars do not show that two models are alike: across the {c['cells']}
    (assay, stratum) cells, the difference between fine-tuned and base correlation
    (resampling the same sites for both models, {n_boot} resamples) excludes zero in
    {c['excluding_zero']}, with the fine-tuned model higher in {c['ft_higher']} and lower in
    {c['ft_lower']}. Those differences are in the accompanying table. Panels marked as receptor
    binding measure binding rather than fitness, and many replacements raise binding, so the
    correlation there is a property of the assay and not a model failure; they are excluded
    from these counts. The overlaid purple curve and markers show the mean experimental phenotype
    (normalized as a z-score relative to all assayed variants in each dataset, ± SEM) across
    distance groups (right axis). In cell entry and growth preference assays, functional viability
    decays sharply as codon distance increases, matching the drop in PLM prediction accuracy;
    in immune escape assays, the phenotype remains flat.{stars}{tail}
    """


def alpha_paragraph(a: dict, n_panels: int) -> str:
    """The paragraph for the violin strip. Every number comes from the refit table."""
    g = a["by_group"]
    # Strata only. The pooled group is not a within-stratum arm and listing it here once made
    # the parenthetical contradict the count it was explaining.
    per_group = "; ".join(f"{g[str(k)]['negative']} of {g[str(k)]['n']} at {GROUP_LABEL[k]}"
                          for k in STRATA if g[str(k)]["n"])
    assays = ", ".join(a["binding_assays"])
    ft = a["pooled_by_role"]["finetune"]
    bs = a["pooled_by_role"]["base"]
    return f"""

    (**{chr(ord('A') + n_panels)}**) The accessibility weight refitted inside one stratum. The
    score of figure 4, log(model probability) + α log(accessibility), was swept over α and
    rank-correlated with the measured effect again, separately within each group of
    replacements. Each point is one assay-and-model arm, the violin its distribution, the bar
    its median. Fitted across all replacements at once, a negative α is favoured in
    {a['neg_pooled']} of {a['n_pooled']} arms. Fitted inside a single stratum it is still
    favoured in {a['neg_strata']} of {a['n_strata']} ({per_group}), and the interval
    excludes zero in {a['excl_zero_neg']}. So down-weighting accessible replacements still helps
    once the number of nucleotide changes is held fixed, and the weighting is not purely a
    restatement of that count — but only at one and two changes. At three it is a coin flip
    ({g['3']['negative']} of {g['3']['n']}, median α* {g['3']['median']:+.2f}), where the
    replacements are rarest and the measured effects weakest. The help is small throughout: the
    median gain in ρ over α = 0 is {a['median_stratum_gain']:.3f} (largest
    {a['max_stratum_gain']:.3f}).

    Two cautions. α is not comparable across the groups on the axis as drawn. Fixing the
    number of changes removes most of the spread in log(accessibility) — pooled it is about
    {a['sd_ratio']:.0f} times wider than within one stratum (the SD under each tick) — so the
    same real amount of down-weighting needs a correspondingly larger α. On a scale-matched
    footing, α* × SD, the median weight is {a['median_pooled_scaled']:.2f} pooled against
    {a['median_stratum_scaled']:.2f} within strata, so most of the pooled effect is the step
    count after all. Second, the {a['n_binding_arms']} receptor-binding arms ({assays}) are
    left out of this panel and of every count above. Those assays measure a phenotype that runs
    *against* fitness, so their correlations are negative across the whole sweep and the largest
    one falls wherever the model is diluted hardest towards zero: {a['binding_at_edge']} of them
    put the optimum hard against the end of the swept range, which is where α runs when there is
    no peak to find. No optimum is defined for them, and reading one would be meaningless. Every
    arm, binding included, is in the accompanying table, and Figure SP draws every curve.

    Nothing else is set aside, which is why every group above has the same denominator: an arm
    counted in one stratum is counted in all of them. An optimum at the end of the swept range
    is kept ({a['n_kept_at_edge']} of the arms counted above) — where a curve runs plainly to
    that edge it is a result, though a bound rather than a turning point. So are the
    {a['n_unreadable_fitness']} arms whose sweep found nothing at all, no correlation anywhere
    and no gain from moving α; their α* sits beside zero and counts as no preference, which can
    only weaken the tallies above, never strengthen them. Figure SP draws every curve behind
    these counts.

    Pooled, the two models separate completely: all {ft['n']} fine-tuned arms prefer a negative
    α, between {ft['lo']:+.2f} and {ft['hi']:+.2f}, while the {bs['n']} base arms fall between
    {bs['lo']:+.2f} and {bs['hi']:+.2f}, only {bs['negative']} of them negative. The ranges do
    not overlap, so an arm's model can be read off its preferred weight alone. Dividing
    accessibility back out helps where the model absorbed it, and the fine-tuned model is the
    one that did.
    """


def main(argv=None) -> int:
    args = parse_figure_args(__doc__.split("\n\n")[0], argv, extra=add_scoring_argument)
    guide: Guide = args.guide
    scores = guide.read_table("dms_scores", args.results)
    table = guide.read_table("dms_within_stratum", args.results)
    summary = guide.read_table("dms_peak_summary", args.results)

    panel = panels(summary)
    roles = role_of_model(summary)
    # Both gates run on the unmasked arm, whichever arm is drawn: the shipped table and
    # dms_peak_summary were computed from the wild-type (unmasked) probability, so that arm is
    # the only one with a fixed point to check against. Fail here rather than draw a masked
    # panel whose difference from the published one could be anything.
    gap = check_against_table(recompute(scores, panel), table)
    # The pooled group only, and no bootstrap: that is all check_pooled_alpha reads, and the
    # refit is the slowest thing here. The drawn arm's own refit follows.
    alpha = alpha_by_stratum(scores, panel, roles, n_boot=0, groups=(POOLED,))
    alpha_gap = check_pooled_alpha(alpha, summary)

    dms_norm = dms_norm_by_stratum(scores, panel)
    # The measured phenotype is a property of the assay, not of how the model was scored, so
    # these run once on the unmasked frame and are the same in both arms.
    dms_tests = dms_stratum_tests(scores, panel)

    if args.scoring == "masked":
        scores = apply_masked(scores, guide.read_table("dms_scores_masked", args.results))
        # No archive table exists for this arm, so the bars and their intervals are both
        # recomputed. See recompute_with_band for what the interval is and is not.
        table = recompute_with_band(scores, panel, groups=GROUPS)
    else:
        pooled_band = recompute_with_band(scores, panel, groups=(POOLED,))
        table = pd.concat([pooled_band, table], ignore_index=True)
    # Everything below is the arm that is drawn. The pooled fit above was the gate.
    alpha = alpha_by_stratum(scores, panel, roles)
    diff = paired_difference(scores, panel, roles, groups=GROUPS)

    fig = draw(table, panel, roles, alpha, dms_norm=dms_norm, dms_tests=dms_tests)
    paths = style.save_figure(fig, args.out / STEM)

    out = table.copy()
    out["role"] = [roles[(d, m)] for d, m in zip(out["dataset"], out["model_name"])]
    write_values(out, args.out, STEM)
    write_values(diff, args.out, "sM_paired_difference")
    write_values(alpha, args.out, ALPHA_STEM)
    write_values(dms_norm, args.out, "sM_dms_fitness_by_stratum")
    write_values(dms_tests, args.out, TEST_STEM)
    write_caption(caption(panel, table, diff, roles, N_BOOT, alpha, scoring=args.scoring,
                          dms_tests=dms_tests),
                  args.out, STEM)
    print(f"[gate] shipped correlations agree with the unmasked scores to {gap:.1e}")
    print(f"self-check: pooled alpha refit matches dms_peak_summary to {alpha_gap:.3f}")
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
