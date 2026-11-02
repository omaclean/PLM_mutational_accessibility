"""How well the deep mutational scans pin down the weight they select.

No plotting, no paths. Used by ``tools/export_sW_loo_alpha.py``.

Figure 4 sweeps the weight alpha over a grid, marks the alpha at which each curve's rank
correlation with a measured phenotype is largest, and quotes that alpha in the panel. The
alpha is read off the same replacements the correlation is reported on, which is the DMS
half of Reviewer 2's objection: "the alpha parameter is selected on the same data used to
report performance" (Reviewer 2, uncertainty row).

The band already drawn on those panels does not answer it. ``dms_sweep.site_bootstrap_band``
resamples sites at a **fixed** alpha and reports how precisely rho is known at each weight --
vertical uncertainty. The selected weight is never re-chosen inside a replicate, so the band
says nothing about how well determined the *location* of the optimum is. A curve with a broad
plateau and a curve with a sharp peak can carry the same band.

What this module does
---------------------

For one panel and one model, draw ``n_boot`` replicates resampling SITES with replacement --
the same independent unit, the same blocks and the same draws as the band
(``dms_sweep.site_blocks`` and one ``rng.integers`` call per replicate) -- recompute the whole
rho-versus-alpha curve on the resampled replacements, and take **that replicate's** argmax.
The spread of those argmaxes is the answer: the 2.5th and 97.5th percentiles of alpha*, its
median, and beside them the point estimate from the unresampled data.

The argmax is ``dms_sweep.curve_flags``'s, including its ``absolute`` option, because that is
the rule the shipped panels and ``output/sY_dms_alpha_optima.tsv`` are marked with. A second
selection rule here would produce an interval around a weight no figure reports.

What the interval is, and is not
--------------------------------

It is a **percentile interval of a non-smooth statistic**. An argmax jumps: a replicate that
tips one grid point above its neighbour moves alpha* by a whole step, and the bootstrap
distribution can be irregular, lumpy on the 0.01 grid and multi-modal where a curve has two
comparable optima. Nothing here corrects for that, and none of the usual coverage arguments
(which assume a smooth, asymptotically normal functional) apply. Read it as the range of
weights these sites would have selected, not as a confidence interval with a coverage
guarantee.

It is also expected to be **wide wherever the curve is a plateau**, which is the honest
result and is not smoothed away. The interval is not narrowed, the curve is not smoothed
before the argmax is taken, and no estimator is swapped in to make a flat panel look sharp:
a weight the data does not pin down is a finding about the data.

Boundary pile-up
----------------

A replicate whose argmax lands at the end of the swept range has not found a peak; it has
found whichever end dilutes the model hardest. ``curve_flags`` already calls that
``boundary``, and the same test is applied to every replicate, so the two fractions returned
beside the interval -- ``frac_boundary_low`` and ``frac_boundary_high`` -- say how much of the
distribution is piled against an edge. **A panel whose replicates mostly pile up at an edge
has no interior optimum and its interval must not be read as one.** The point estimate's own
``boundary``, ``weak_signal`` and ``interpretable`` flags are carried through unchanged for
the same reason: they are what says whether the row is readable at all.

Why there is no p-value here
----------------------------

Deliberately absent, as in ``mutacc.loo_alpha``, and for a reason of its own. There is no null
hypothesis in sight: the question is how precisely one selected weight is located, and an
interval answers it directly. A p-value would have to be manufactured against some null --
"alpha* = 0", say -- and on these data that null is about a quantity the sweep does not test:
every panel's curve is continuous through alpha = 0, so a weight near zero is a small effect,
not an absent one. Worse, the panels are not independent of each other (two models score the
same replacements, and the four influenza panels share two reference backbones), so any
collection of such p-values would be correlated by construction. The interval, the median, the
point estimate and the boundary fractions are the result, and they are what should be quoted.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from mutacc.dms_sweep import curve_flags, site_blocks, sweep_curve

__all__ = [
    "ALPHA_STAR", "DMS_METRIC", "N_BOOT", "SEED", "LEVEL", "MIN_SITES", "MIN_ROWS",
    "Arm", "iter_arms", "alpha_star_bootstrap", "arm_row", "alpha_star_table",
]

#: Value of the ``scope`` column on the rows this module writes, which is how a reader tells
#: them from the leave-one-lineage-out rows sharing the same file.
ALPHA_STAR = "dms_alpha_star"

#: Value of the ``metric`` column: the sweep's own quantity, a Spearman rank correlation
#: between the weighted score and the laboratory measurement.
DMS_METRIC = "dms_spearman_rho"

#: Replicates per (panel, model). Each one re-ranks every replacement at all 201 weights, which
#: costs about a third of a second on the larger scans, so this is minutes per arm rather than
#: the seconds the band costs -- the band gets away with 200 replicates on every fifth weight
#: because a percentile *envelope* is smooth, and an argmax distribution is not. A thousand
#: puts the 2.5th and 97.5th percentiles at the 25th and 975th order statistics, which is
#: enough for them to stop moving by a grid step between seeds; 200 was not.
N_BOOT = 1000

#: The seed each arm's generator is built from. See :func:`arm_row` for why it is combined with
#: the arm's position rather than shared across arms.
SEED = 20261021

LEVEL = 0.95

#: Fewer independent sites than this cannot carry an interval, and a replicate drawn from one
#: or two of them is the same data every time. Raised rather than returned as NaN: a DMS panel
#: with two sites is a broken join, not a small scan.
MIN_SITES = 3

#: An arm with fewer scored replacements than this is a broken join, not a small scan. The
#: same guard, and the same number, as ``figures/make_fig04.py``; it is repeated rather than
#: imported because nothing under ``mutacc/`` may depend on ``figures/``.
MIN_ROWS = 200

#: Columns of ``dms_peak_summary`` this module needs. ``figures/make_fig04.py`` validates the
#: same table more thoroughly before drawing from it; what is checked here is only what is read.
REQUIRED_SUMMARY = ("dataset", "organism", "trait", "panel_label", "citation", "model_role",
                    "model_name", "model_label", "spectrum", "virus_tag")


@dataclass(frozen=True)
class Arm:
    """One panel scored by one model: the replacements, their sites, and who they belong to.

    ``order`` is the arm's position in ``dms_peak_summary``, which is the archive's order and
    the order the figure's own bootstrap consumes its draws in. It is carried because it seeds
    this module's generator (see :func:`arm_row`), so it is part of the result, not a label.
    """

    order: int
    dataset: str
    trait: str
    panel_label: str
    organism: str
    virus_tag: str
    citation: str
    model_name: str
    model_label: str
    model_role: str
    log_plm: np.ndarray
    log_acc: np.ndarray
    y: np.ndarray
    sites: np.ndarray


def iter_arms(scores: pd.DataFrame, summary: pd.DataFrame, *, spectrum: str = "published") -> list[Arm]:
    """Every (panel, model) of one spectrum variant, in the order ``dms_peak_summary`` holds.

    The join is the one ``figures/make_fig04.compute_curves`` makes: the rows of ``dms_scores``
    for that dataset and model, keeping only the replacements the panel's trait was measured
    on. ``scores`` is expected to carry whichever model probability is being reported -- pass
    the frame ``mutacc.masked_dms.apply_masked`` returns to get the masked marginal the
    manuscript quotes, as the exporter does.

    A missing trait column, an empty join or a join under :data:`MIN_ROWS` raises here rather
    than producing a row with an interval computed from nothing.
    """
    missing = [c for c in REQUIRED_SUMMARY if c not in summary.columns]
    if missing:
        raise ValueError(f"dms_peak_summary lacks column(s) {missing}")
    arms = summary[summary["spectrum"] == spectrum].reset_index(drop=True)
    if arms.empty:
        raise ValueError(f"dms_peak_summary has no rows for the {spectrum!r} spectrum")
    if arms.duplicated(["dataset", "trait", "model_name"]).any():
        raise ValueError("dms_peak_summary has duplicate (dataset, trait, model) rows")

    out = []
    for order, arm in enumerate(arms.itertuples()):
        if arm.trait not in scores.columns:
            raise ValueError(f"dms_scores has no column for trait {arm.trait!r}")
        mask = ((scores["dataset"] == arm.dataset) & (scores["model_name"] == arm.model_name)).to_numpy()
        y_all = scores[arm.trait].to_numpy(float)
        keep = mask & np.isfinite(y_all)
        if keep.sum() < MIN_ROWS:
            raise ValueError(
                f"{arm.dataset}/{arm.model_name}/{arm.trait}: only {int(keep.sum())} scored "
                f"replacements (need {MIN_ROWS}); the join between scores and phenotype is "
                f"empty or broken"
            )
        out.append(Arm(
            order=order, dataset=str(arm.dataset), trait=str(arm.trait),
            panel_label=str(arm.panel_label), organism=str(arm.organism),
            virus_tag=str(arm.virus_tag), citation=str(arm.citation),
            model_name=str(arm.model_name), model_label=str(arm.model_label),
            model_role=str(arm.model_role),
            log_plm=scores["log_plm"].to_numpy(float)[keep],
            log_acc=scores["log_acc"].to_numpy(float)[keep],
            y=y_all[keep],
            sites=scores["site"].to_numpy()[keep],
        ))
    return out


def alpha_star_bootstrap(
    log_plm: np.ndarray,
    log_acc: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    alphas: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int,
    absolute: bool = True,
    level: float = LEVEL,
) -> dict:
    """The selected weight, its percentile interval over resampled sites, and the edge counts.

    The point estimate and its three flags come from ``curve_flags`` on the unresampled data,
    so they are exactly what the figure marks and what ``sY_dms_alpha_optima.tsv`` reports.
    Each replicate then resamples sites with replacement, **recomputes the whole curve**, and
    is put through the same ``curve_flags``: nothing here re-reads the full-data optimum, which
    is the one way this could be made to look precise without being so.

    Returns ``alpha_star`` (the point estimate), ``rho_best`` and ``rho_alpha0`` beside it, the
    flags ``boundary``, ``weak_signal`` and ``interpretable``, then ``alpha_star_median``,
    ``ci_lo`` and ``ci_hi`` over the replicates, the two edge fractions
    ``frac_boundary_low`` and ``frac_boundary_high``, and the replicate counts ``n_boot`` and
    ``n_boot_used``.

    The interval is a percentile interval of a non-smooth statistic: see the module docstring
    for what that does and does not license. Replicates are *not* smoothed, trimmed or
    re-centred, so a plateau comes back wide.

    Refusals, all of them loud:

    * ``n_boot`` below 2 -- two percentiles cannot be taken from one number;
    * fewer than :data:`MIN_SITES` distinct sites;
    * a curve with no finite value (``curve_flags``'s own refusal), or a **constant** one.
      A flat curve has no optimum: ``argmax`` would hand back the first grid point, which on
      this grid is the far negative edge, and the row would read as a boundary result instead
      of as the degenerate input it is.
    """
    if n_boot < 2:
        raise ValueError(f"n_boot must be at least 2 to take a percentile interval, got {n_boot}")
    alphas = np.asarray(alphas, float)
    uniq, blocks = site_blocks(sites)
    if len(uniq) < MIN_SITES:
        raise ValueError(f"{len(uniq)} distinct site(s); a site bootstrap needs at least {MIN_SITES}")

    rho = sweep_curve(log_plm, log_acc, y, alphas)
    flags = curve_flags(rho, alphas, absolute=absolute)
    if _is_flat(rho):
        raise ValueError("the sweep curve is constant; it has no optimum to locate")

    stars = np.full(n_boot, np.nan)
    boundary = np.zeros(n_boot, bool)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([blocks[i] for i in pick])
        replicate = sweep_curve(log_plm[idx], log_acc[idx], y[idx], alphas)
        if not np.isfinite(replicate).any() or _is_flat(replicate):
            continue
        rep_flags = curve_flags(replicate, alphas, absolute=absolute)
        stars[b] = rep_flags["best_alpha"]
        boundary[b] = rep_flags["boundary"]
    used = np.isfinite(stars)
    if int(used.sum()) < 2:
        raise ValueError(
            f"only {int(used.sum())} of {n_boot} replicates gave a curve with an optimum; "
            f"there is nothing to take percentiles of"
        )
    drawn, at_edge = stars[used], boundary[used]
    tail = (1.0 - level) / 2.0 * 100.0
    return {
        "alpha_star": flags["best_alpha"],
        "rho_best": flags["rho_best"],
        "rho_alpha0": flags["rho_alpha0"],
        "boundary": flags["boundary"],
        "weak_signal": flags["weak_signal"],
        "interpretable": flags["interpretable"],
        "alpha_star_median": float(np.median(drawn)),
        "ci_lo": float(np.percentile(drawn, tail)),
        "ci_hi": float(np.percentile(drawn, 100.0 - tail)),
        "frac_boundary_low": float(np.mean(at_edge & (drawn < 0))),
        "frac_boundary_high": float(np.mean(at_edge & (drawn > 0))),
        "n_boot": int(n_boot),
        "n_boot_used": int(used.sum()),
    }


def _is_flat(curve: np.ndarray) -> bool:
    """True for a curve whose finite values are all the same number."""
    finite = curve[np.isfinite(curve)]
    return bool(finite.size and finite.min() == finite.max())


def arm_row(
    arm: Arm,
    alphas: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    level: float = LEVEL,
    absolute: bool = True,
) -> dict:
    """One arm's bootstrap, as a row: its identity, its selected weight and the interval.

    **The generator is built per arm, from ``[seed, arm.order]``.** ``site_bootstrap_band``
    instead threads one generator through every arm in table order, which is what reproduces
    the published band but also means no arm can be recomputed on its own. Here each arm is
    independent of which others were run, so a single panel can be checked, re-run or
    recomputed in a worker process and still give the shipped numbers. The arm's position in
    ``dms_peak_summary`` is part of its seed, so reordering that table would reseed the
    bootstrap; that is intended, since it would also reorder the figure's own band.
    """
    rng = np.random.default_rng([seed, arm.order])
    result = alpha_star_bootstrap(arm.log_plm, arm.log_acc, arm.y, arm.sites, alphas, rng,
                                  n_boot=n_boot, absolute=absolute, level=level)
    return {
        "panel_label": arm.panel_label,
        "dataset": arm.dataset,
        "trait": arm.trait,
        "organism": arm.organism,
        "virus_tag": arm.virus_tag,
        "citation": arm.citation,
        "model": arm.model_name,
        "model_label": arm.model_label,
        "model_role": arm.model_role,
        "n": int(len(arm.y)),
        "n_sites": int(len(np.unique(arm.sites))),
        **result,
    }


def alpha_star_table(
    arms: Sequence[Arm],
    alphas: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    level: float = LEVEL,
    absolute: bool = True,
    runner: Callable[..., Iterable[dict]] = map,
) -> pd.DataFrame:
    """One row per arm, in the order given.

    ``runner`` is ``map`` by default and may be a process pool's ``map``: every arm is seeded
    from its own position (:func:`arm_row`), so the table does not depend on how the work was
    spread or on what order the results came back in. The pool belongs to the caller --
    nothing under ``mutacc/`` starts processes of its own.
    """
    if not len(arms):
        raise ValueError("no arms to bootstrap")
    work = functools.partial(arm_row, alphas=np.asarray(alphas, float), n_boot=n_boot,
                             seed=seed, level=level, absolute=absolute)
    return pd.DataFrame(list(runner(work, arms)))
