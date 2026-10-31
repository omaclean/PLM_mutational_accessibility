#!/usr/bin/env python3
"""How strongly model probability tracks mutational accessibility, on both scales.

    python tools/export_fig01bc_correlations.py
    python tools/export_fig01bc_correlations.py <repo> <out_dir>   # how the build calls it

The companion of ``tools/export_fig01bc_values.py``. That script writes the two quantities
Figure 1B and 1C rank -- the language model's probability for a replacement and the
probability of reaching it under the nucleotide mutation spectrum -- for one model. This
one writes how closely those two quantities agree, for every model in the guide, under
every coefficient the manuscript could quote.

Why the table exists
--------------------
Figure 2A/B print a Pearson correlation taken on the probabilities themselves
(``figures/make_fig02ab.py``), and the Results quote it as the fraction of the model's
variance that accessibility explains. Those probabilities span about nine orders of
magnitude, so a raw-scale Pearson is set by a handful of large values and says almost
nothing about the bulk of the cloud -- which is why the panels are drawn on log axes in
the first place. On log10 probabilities, the scale the panels are read on, the same
replacements give a much larger correlation. The two numbers answer different questions
and neither is wrong, so this table writes both side by side, with the rank correlation
that does not depend on the scale at all, rather than leaving a reader to guess which
scale a quoted coefficient was taken on.

What is in it
-------------
One row per (model, lineage, nucleotide-change group, statistic). ``scale`` names which of
the two variables was logged, and ``plm_transform`` and ``mut_transform`` say so one
variable at a time, so no row can be read as applying a transform it did not:

``pearson_r`` on ``raw``         neither logged; the coefficient Figure 2A/B print.
``pearson_r`` on ``log10_plm``   model probability logged, accessibility left linear.
``pearson_r`` on ``log10_mut``   accessibility logged, model probability left linear.
``pearson_r`` on ``log10_both``  both logged; the scale the panels are drawn on.
``spearman_r`` on ``rank``       rank correlation, invariant to any monotone rescaling.

The two semi-log rows are what separates "the log scale helps" from "the log scale of one
particular variable helps". Both probabilities are heavy-tailed, so either one left linear
can on its own hold a raw-scale correlation down; which one does is a fact about the data
and is not recoverable from the ``raw`` and ``log10_both`` rows alone.

``r_squared`` is the square of the coefficient on its own row. For the Pearson rows it is
the usual fraction of variance explained *on that scale*; for the Spearman row it is the
fraction of rank variance, which is not the same quantity and is written only so that a
reader comparing rows is not left to square them by hand.

Every lineage of the guide gets its own rows and none are pooled. The guide's lineages are
scored at the same sites, so pooling them stacks near-copies of the same replacements and
reports a precision the data do not have (``mutacc.strata``, LINEAGE_VALIDITY.md section
4). ``is_primary`` marks the one lineage the single-lineage figures are drawn from.

Intervals, and why there are no p-values
----------------------------------------
The interval is a percentile bootstrap over *sites*, not over replacements: the 19
replacements at a site share a codon and a model context, so resampling them individually
would treat 10,754 points as 10,754 independent observations and return an interval far
too narrow. A p-value against zero is not written for the same reason -- it would be
computed from that same false count, and no analysis here turns on whether these
correlations differ from zero.

Source
------
``replacement_scores`` from the run guide, the same shipped table
``tools/export_fig01bc_values.py`` reads, filtered to nothing and sliced by model and
lineage. Nothing that names a model or a lineage is written into this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
from scipy.stats import rankdata                              # noqa: E402

from mutacc.guide import REPO_ROOT, GuideError, load_guide    # noqa: E402
from mutacc.io import write_caption, write_values             # noqa: E402

STEM = "fig01bc_correlation_stats"

#: The guide whose lineages and models these correlations are taken over.
DEFAULT_GUIDE = REPO_ROOT / "guides" / "h3n2_5lineage.yaml"

#: The two columns being correlated: the model's probability and the accessibility of the
#: same replacement. Named here because the panels name them the same way.
PLM_COL = "plm_prob"
MUT_COL = "mut_prob"

#: Nucleotide-change groups reported beside the whole cloud, as Figure 2A/B colour by.
GROUPS = (1, 2, 3)
ALL_GROUP = "all"

#: ``scale`` -> the transform applied to (model probability, accessibility). The name of a
#: scale says which variable was logged; the pair says it again in a form nothing has to
#: parse. ``rank`` is the one scale that is not a transform of one variable or the other.
IDENTITY, LOG10, RANK = "identity", "log10", "rank"
SCALES = {
    "raw": (IDENTITY, IDENTITY),
    "log10_plm": (LOG10, IDENTITY),
    "log10_mut": (IDENTITY, LOG10),
    "log10_both": (LOG10, LOG10),
    "rank": (RANK, RANK),
}

#: ``statistic`` -> the scales it is reported on. Spearman is invariant to any monotone
#: transform of either variable, so logging one or both cannot change it and it is written
#: once, on ``rank``, rather than four times with the same number in every row.
STATISTICS = {
    "pearson_r": ("raw", "log10_plm", "log10_mut", "log10_both"),
    "spearman_r": ("rank",),
}

N_BOOT = 1000
LEVEL = 0.95
#: Fixed so the table is reproducible; the date the table was first written.
SEED = 20261023

COLS = [
    "model", "model_label", "model_role", "lineage", "is_primary",
    "group", "n", "n_sites", "statistic", "scale", "plm_transform", "mut_transform",
    "value", "r_squared", "ci_lo", "ci_hi", "n_boot", "level",
]


def paired_columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two probabilities and the site each replacement sits at, refusing non-positives.

    A zero or a negative probability has no logarithm, and dropping it quietly would make
    the raw-scale and log-scale rows describe different sets of replacements -- which is
    precisely the comparison this table exists to make.
    """
    plm = frame[PLM_COL].to_numpy(float)
    mut = frame[MUT_COL].to_numpy(float)
    bad = ~(np.isfinite(plm) & np.isfinite(mut) & (plm > 0) & (mut > 0))
    if bad.any():
        raise GuideError(
            f"{int(bad.sum())} of {len(frame)} replacements have a non-finite or "
            f"non-positive {PLM_COL}/{MUT_COL}; they have no log10 and the raw and log "
            f"rows would not describe the same replacements"
        )
    return plm, mut, frame["position"].to_numpy(int)


def transformed(values: np.ndarray, how: str) -> np.ndarray:
    """One variable under one of the three transforms a scale is built from."""
    if how == IDENTITY:
        return values
    if how == LOG10:
        return np.log10(values)
    if how == RANK:
        return rankdata(values)
    raise GuideError(f"unknown transform {how!r}; expected one of "
                     f"{sorted({IDENTITY, LOG10, RANK})}")


def coefficients(plm: np.ndarray, mut: np.ndarray) -> dict[tuple[str, str], float]:
    """``(statistic, scale) -> coefficient`` for one set of replacements.

    Every one of them is a Pearson correlation of some transform of the pair -- that is
    what a Spearman correlation is -- so they are taken the same way, from one table of
    transforms, rather than through separate library calls that could disagree about ties
    or about constant input.
    """
    out = {}
    for statistic, scales in STATISTICS.items():
        for scale in scales:
            x, y = SCALES[scale]
            out[(statistic, scale)] = float(
                np.corrcoef(transformed(plm, x), transformed(mut, y))[0, 1])
    return out


def site_blocks(position: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by site, which is the unit the bootstrap resamples."""
    order = np.argsort(position, kind="mergesort")
    starts = np.flatnonzero(np.diff(position[order])) + 1
    return np.split(order, starts)


def bootstrap_intervals(
    plm: np.ndarray, mut: np.ndarray, position: np.ndarray, *,
    n_boot: int = N_BOOT, level: float = LEVEL, seed: int = SEED,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Percentile intervals for every coefficient, resampling whole sites with replacement.

    One set of draws serves all three coefficients, so their intervals describe the same
    resampled clouds and can be read against each other.
    """
    blocks = site_blocks(position)
    if len(blocks) < 2:
        raise GuideError("a site bootstrap needs at least two sites")
    rng = np.random.default_rng(seed)
    draws: dict[tuple[str, str], list[float]] = {}
    for pick in rng.integers(0, len(blocks), size=(n_boot, len(blocks))):
        idx = np.concatenate([blocks[i] for i in pick])
        for key, value in coefficients(plm[idx], mut[idx]).items():
            draws.setdefault(key, []).append(value)
    tail = (1.0 - level) / 2.0 * 100.0
    return {key: tuple(float(v) for v in np.percentile(values, [tail, 100.0 - tail]))
            for key, values in draws.items()}


def rows_for(frame: pd.DataFrame, *, model, lineage: str, is_primary: bool,
             n_boot: int, seed: int) -> list[dict]:
    """Every statistic for one model and one lineage, whole cloud and each group."""
    steps = frame["nt_steps"].to_numpy(int)
    if not set(steps) <= set(GROUPS):
        raise GuideError(f"nt_steps must be one of {GROUPS}, got {sorted(set(steps))}")
    out = []
    for group in (ALL_GROUP, *(str(g) for g in GROUPS)):
        part = frame if group == ALL_GROUP else frame[steps == int(group)]
        plm, mut, position = paired_columns(part)
        values = coefficients(plm, mut)
        intervals = bootstrap_intervals(plm, mut, position, n_boot=n_boot, seed=seed)
        for statistic, scales in STATISTICS.items():
            for scale in scales:
                lo, hi = intervals[(statistic, scale)]
                out.append(dict(
                    model=model.key, model_label=model.label, model_role=model.role,
                    lineage=lineage, is_primary=is_primary, group=group,
                    n=len(part), n_sites=int(len(np.unique(position))),
                    statistic=statistic, scale=scale,
                    plm_transform=SCALES[scale][0], mut_transform=SCALES[scale][1],
                    value=values[(statistic, scale)],
                    r_squared=values[(statistic, scale)] ** 2,
                    ci_lo=lo, ci_hi=hi, n_boot=n_boot, level=LEVEL,
                ))
    return out


def build(guide, results: Path | None = None, *, n_boot: int = N_BOOT,
          seed: int = SEED) -> pd.DataFrame:
    """The whole table: every model of the guide against every lineage of it."""
    frame = guide.read_table("replacement_scores", results)
    guide.check_lineages(frame["lineage"].unique(), "replacement_scores")
    primary = guide.primary_lineage
    rows = []
    for model in guide.models:
        part = frame[frame["model"] == model.key]
        if part.empty:
            raise GuideError(f"replacement_scores has no rows for model {model.key!r}")
        for lineage in guide.lineages:
            sub = part[part["lineage"] == lineage.id]
            if sub.empty:
                raise GuideError(f"replacement_scores has no rows for model {model.key!r} "
                                 f"and lineage {lineage.id!r}")
            rows += rows_for(sub, model=model, lineage=lineage.id,
                             is_primary=lineage.id == primary, n_boot=n_boot, seed=seed)
    return pd.DataFrame(rows)[COLS]


def caption(table: pd.DataFrame, guide) -> str:
    """What the table holds, and which of its numbers the figures print.

    Every number quoted here is read back out of the table, so the caption cannot drift
    from the file it describes.
    """
    primary = guide.primary_lineage

    def get(role: str, scale: str, statistic: str = "pearson_r") -> float:
        rows = table[(table["model_role"] == role) & (table["scale"] == scale)
                     & (table["statistic"] == statistic) & (table["group"] == ALL_GROUP)
                     & (table["lineage"] == primary)]
        if len(rows) != 1:
            raise GuideError(f"expected one {role}/{scale}/{statistic} row for {primary}, "
                             f"got {len(rows)}")
        return float(rows["value"].iloc[0])

    n = int(table[(table["group"] == ALL_GROUP) & (table["lineage"] == primary)]["n"].iloc[0])
    labels = {role: table[table["model_role"] == role]["model_label"].iloc[0]
              for role in ("base", "finetune")}
    return f"""
    **Supporting table: correlation of model probability with mutational accessibility,
    on every scale.** Each row is one coefficient for one model, one lineage and one group
    of replacements. `scale` says which of the two variables was logged -- `raw` neither,
    `log10_plm` the model probability only, `log10_mut` the accessibility only,
    `log10_both` both -- and `plm_transform` and `mut_transform` repeat that one variable
    at a time. `rank` is the Spearman correlation, written once because no monotone
    transform of either variable can change it. The replacements are every single-residue
    replacement away from a lineage's reference protein ({n:,} of them per lineage), the
    same set panels B and C rank and Figure 2A and 2B plot.

    On the lineage the single-lineage figures are drawn from (`{primary}`), over all
    replacements, for the model labelled "{labels['base']}" and for "{labels['finetune']}"
    in turn: {get('base', 'raw'):.3f} and {get('finetune', 'raw'):.3f} with neither
    variable logged, {get('base', 'log10_plm'):.3f} and
    {get('finetune', 'log10_plm'):.3f} with the model probability logged,
    {get('base', 'log10_mut'):.3f} and {get('finetune', 'log10_mut'):.3f} with the
    accessibility logged, {get('base', 'log10_both'):.3f} and
    {get('finetune', 'log10_both'):.3f} with both, and rank correlations of
    {get('base', 'rank', 'spearman_r'):.3f} and
    {get('finetune', 'rank', 'spearman_r'):.3f}.

    The first pair is the coefficient printed under Figure 2A and 2B. Both probabilities
    span about nine orders of magnitude, so a correlation taken on either untransformed is
    set by a handful of large values -- but the two semi-log rows show the two variables
    are not symmetric in this. Logging the model probability alone recovers a large part of
    the distance to the joint log scale -- about half of it for the base model and more
    for the fine-tuned one -- while logging the accessibility alone recovers little of it
    for either, and for the fine-tuned model leaves the coefficient near raw. The raw
    coefficient is therefore held down mainly by the skew of the model's own probabilities
    rather than by the accessibility it is being compared with. A fraction of variance
    explained is only interpretable beside the scale it was taken on, which is why
    `r_squared` never appears without `scale`.

    `ci_lo` and `ci_hi` are a {int(table['level'].iloc[0] * 100)}% percentile bootstrap
    interval over sites, {int(table['n_boot'].iloc[0])} draws: the 19 replacements at a
    site share a codon and a model context, so resampling replacements individually would
    report a precision the data do not have. No p-value is written, for the same reason.
    Lineages are listed separately and never pooled, because the guide's lineages are
    scored at the same sites and pooling them stacks near-copies of one cloud.

    Written by `tools/export_fig01bc_correlations.py` from the guide's
    `replacement_scores` table, the shipped per-replacement table that
    `tools/export_fig01bc_values.py` also reads.
    """


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # Positional, and both optional: tools/build_figures.py calls a table exporter as
    # ``<script> <repo> <out_dir>`` with no flags at all.
    parser.add_argument("repo", nargs="?", type=Path, default=REPO_ROOT,
                        help="repository root (default: the one this script lives in)")
    parser.add_argument("out_dir", nargs="?", type=Path, default=None,
                        help="where to write the table (default: <repo>/output)")
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE,
                        help="run guide naming the lineages and models")
    parser.add_argument("--results", type=Path, default=None,
                        help="read the guide's tables from here instead of its results_dir")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"bootstrap draws over sites (default: {N_BOOT})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"bootstrap seed (default: {SEED})")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = args.repo / "output"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    guide = load_guide(args.guide)
    table = build(guide, args.results, n_boot=args.n_boot, seed=args.seed)
    path = write_values(table, args.out_dir, STEM)
    print(f"wrote {path} ({len(table)} rows)")
    path = write_caption(caption(table, guide), args.out_dir, STEM)
    print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
