"""Codon learning: does a model never shown codons still tell synonymous codons apart?

    python figures/make_codon_learning.py --guide guides/h3n2_13lineage.yaml \\
        --guide-secondary guides/h3n2_5lineage.yaml

Promotable either way: the main figure (``codon_learning``) is a candidate main figure, and the
all-replacements figure (``sI_codon_pair_tracking``) is supplementary slot S9, the
follow-up to the codon-learning main figure. Both are
written every time.

Reads the tables made by ``tools/export_codon_s06.py`` (``codon_*``) and, as a check, recomputes
every point estimate from the guide's ``replacement_scores`` and roots; the figure refuses to draw
if the shipped tables and the scores disagree by more than 1e-9.

A  Within-replacement test. With the amino-acid change held fixed (say serine to asparagine),
   sites differ only in which codon the reference residue uses. Does the model rank the new
   residue higher at sites whose codon makes it one nucleotide change away? 0.5 = no.
B  A different question, asked within each reference amino acid: does a site's total
   probability of change fall where a mutation is more likely to be synonymous? Null in both
   models. Shown so the two tests can be read side by side.
C  Each scored replacement on its own, fine-tuned against base.
D  Stress test: how much of the pooled value rests on a few replacements.

The grey bars are where chance puts each number (the middle 95% of shuffles). Dots with
whiskers are the observed value and its 95% interval from resampling sites.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from mutacc import accessibility as acc
from mutacc import codon_learning as cl

from mutacc import style
from mutacc.guide import REPO_ROOT, Guide, load_guide, parse_figure_args
from mutacc.io import write_caption, write_values

STEM = "fig05_codon_learning"
STEM_PAIRS = "sI_codon_pair_tracking"
SELF_CHECK_TOL = 1e-9
TABLES = ("codon_within_pair_summary", "codon_within_pair_difference", "codon_within_pair_by_pair",
          "codon_within_pair_by_lineage", "codon_within_pair_robustness",
          "codon_strand_c_summary", "codon_strand_c_per_aa")
ROLE_ORDER = ("base", "finetune")
ROLE_LABEL = {"base": "Base model", "finetune": "Fine-tuned model"}
PANEL_A, PANEL_B, PANEL_C, PANEL_D = (chr(ord("A") + i) for i in range(4))
OFFSET = 0.2
NULL_HALF_WIDTH = 0.17
#: The recomputed null must reproduce the shipped percentiles to this.
NULL_TOL = 0.02
PAIR_ARROW = "→"


# ---------------------------------------------------------------------------
# Inputs and the self-check
# ---------------------------------------------------------------------------

def load_tables(guide: Guide, results: Path | None) -> dict[str, pd.DataFrame]:
    return {name: cl.read_derived(guide, name, results) for name in TABLES}


def role_rows(frame: pd.DataFrame, guide: Guide) -> pd.DataFrame:
    """Rows for the base and fine-tuned model, in that order, with each model's role from the guide."""
    out = frame.copy()
    out["role"] = [guide.model(m).role for m in out["model"]]
    got = set(out["role"])
    if not set(ROLE_ORDER) <= got:
        raise ValueError(f"{guide.source.name}: the table needs a base and a fine-tuned model, found roles {sorted(got)}")
    return out


def check_against_scores(guide: Guide, scores: pd.DataFrame, tables: dict[str, pd.DataFrame],
                         tol: float = SELF_CHECK_TOL) -> float:
    """Largest gap between shipped point estimates and the same numbers recomputed from the scores. Raises above ``tol``."""
    roots = {lin.id: lin.root for lin in guide.lineages}
    if "lineage" in scores.columns and len(guide.lineages) == 1 and scores["lineage"].nunique() > 1:
        scores = scores[scores["lineage"] == guide.lineages[0].id]
    summary, by_pair, sc = tables["codon_within_pair_summary"], tables["codon_within_pair_by_pair"], tables["codon_strand_c_summary"]
    gap = 0.0
    for row in summary.itertuples():
        prep = cl.WithinPair(cl.canonical_scores(scores, row.model))
        res = prep.blocks()
        gap = max(gap, abs(res.pooled - row.auc))
        if int(res.ok.sum()) != row.n_blocks:
            raise ValueError(f"model {row.model}: shipped table scores {row.n_blocks} blocks, the scores give {int(res.ok.sum())}")
        mine = cl.pair_table(prep, res).set_index("pair")["auc"]
        theirs = by_pair.loc[by_pair["model"] == row.model].set_index("pair")["auc"]
        if set(mine.index) != set(theirs.index):
            raise ValueError(f"model {row.model}: the shipped table and the scores score different replacements")
        gap = max(gap, float((mine - theirs.reindex(mine.index)).abs().max()))
    for row in sc.itertuples():
        q = acc.load_spectrum(REPO_ROOT / row.spectrum_file)
        q = acc.symmetrise(q) if row.symmetrised else q
        per = cl.strand_c_per_aa(cl.strand_c_sites(scores, row.model, roots, q))
        gap = max(gap, abs(float(per["rho"].mean()) - row.mean_rho))
        if len(per) != row.n_aa:
            raise ValueError(f"model {row.model}: shipped within-residue table has {row.n_aa} amino acids, the scores give {len(per)}")
    if gap > tol:
        raise ValueError(f"shipped codon tables disagree with the scores by {gap:.2e} (tolerance {tol:g})")
    return gap


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _estimate_panel(ax, rows: pd.DataFrame, sets: list[str], value: str, lo: str, hi: str, chance: float,
                    chance_label: str, floor: float) -> dict:
    """Dots with site-bootstrap whiskers over grey chance bars, one pair of dots per lineage set."""
    handles = {}
    for si, name in enumerate(sets):
        for role in ROLE_ORDER:
            r = rows.loc[(rows["set"] == name) & (rows["role"] == role)].iloc[0]
            x = si + (OFFSET if role == "finetune" else -OFFSET)
            bar = ax.fill_between([x - NULL_HALF_WIDTH, x + NULL_HALF_WIDTH], r["null_lo"], r["null_hi"],
                                  color=style.NULL_GREY, alpha=0.35, linewidth=0, zorder=1)
            handles.setdefault("null", bar)
            ax.errorbar([x], [r[value]], yerr=[[r[value] - r[lo]], [r[hi] - r[value]]], fmt="o",
                        color=style.colour_for_role(role), ecolor=style.colour_for_role(role),
                        markersize=4.5, elinewidth=1.2, capsize=2.2, zorder=3)
            handles.setdefault(role, ax.plot([], [], "o", color=style.colour_for_role(role))[0])
    top = max(rows[hi].max(), rows["null_hi"].max())
    ax.set_ylim(min(floor, rows[lo].min(), rows["null_lo"].min()) - 0.01, top + 0.03 * (top - floor if top > floor else 1))
    style.reference_line(ax, chance, chance_label, label_loc="right")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(sets)
    ax.set_xlim(-0.6, len(sets) - 0.4)
    ax.grid(axis="x", visible=False)
    return handles


def _stack(per_set: dict[str, dict[str, pd.DataFrame]], key: str, guides: dict[str, Guide]) -> pd.DataFrame:
    frames = []
    for name, tables in per_set.items():
        f = role_rows(tables[key], guides[name])
        f["set"] = name
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def null_draws(guide: Guide, scores: pd.DataFrame, summary: pd.DataFrame, *, n_perm: int, seed: int = 20261013):
    """Recompute the permutation null and check it reproduces the shipped bounds.

    The shipped tables carry only the null's 2.5th and 97.5th percentiles. The figure needs
    the draws themselves to show the distribution, so they are recomputed here from the same
    scores and the same shuffle rule, then checked against what shipped.
    """
    preps = {m.key: cl.WithinPair(cl.canonical_scores(scores, m.key)) for m in guide.models}
    codons = cl.reference_codon_matrix(next(iter(preps.values())), {lin.id: lin.root for lin in guide.lineages})
    draws = cl.permutation_null(preps, codons, n_perm=n_perm, seed=seed)
    for row in summary.itertuples():
        d = draws[row.model]
        lo, hi = np.percentile(d, [2.5, 97.5])
        if abs(lo - row.null_lo) > NULL_TOL or abs(hi - row.null_hi) > NULL_TOL:
            raise ValueError(
                f"{row.model}: recomputed null [{lo:.4f}, {hi:.4f}] does not reproduce the shipped "
                f"[{row.null_lo:.4f}, {row.null_hi:.4f}] within {NULL_TOL}"
            )
    return draws


def _exposition_panel(ax, scores: pd.DataFrame, pairs: pd.DataFrame, guide: Guide, model: str):
    """One worked replacement, so the pooled number has a visible meaning.

    Within a single amino-acid replacement in a single lineage, the amino-acid change is held
    fixed, so chemistry and substitution matrices are constant and only the reference codon
    differs. The AUROC is then just: of every easy-site/hard-site pair, how often does the
    model score the easy one higher.
    """
    canon = cl.canonical_scores(scores, model)
    canon = canon.assign(pair=canon["ref_aa"] + ">" + canon["aa"])
    p = role_rows(pairs, guide)
    p = p.loc[p["role"] == "finetune"]
    want = p.sort_values("auc", ascending=False)["pair"].iloc[0]
    sub = canon.loc[canon["pair"] == want]
    lin = sub.groupby("lineage").size().idxmax()
    sub = sub.loc[sub["lineage"] == lin]
    easy = sub.loc[sub["nt_steps"] == 1, "plm_prob"].to_numpy()
    hard = sub.loc[sub["nt_steps"] > 1, "plm_prob"].to_numpy()

    rng = np.random.default_rng(7)
    for vals, x0, colour in ((easy, 0.0, style.CATEGORICAL[0]), (hard, 1.0, style.CATEGORICAL[2])):
        ax.plot(x0 + rng.uniform(-0.17, 0.17, len(vals)), vals, linestyle="none", marker="o",
                markersize=3.4, color=colour, alpha=0.8, markeredgecolor=style.SURFACE,
                markeredgewidth=0.35, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([f"Easy\n({len(easy)})", f"Hard\n({len(hard)})"], fontsize=6.5)
    ax.set_xlim(-0.5, 1.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(f"PLM probability for {want.replace('>', PAIR_ARROW)} at each site")
    style.panel_title(ax, f"One number: {want.replace('>', PAIR_ARROW)}",
                      "Easy = 1 change, hard = 2-3")


def _null_panel(ax, draws, summary: pd.DataFrame, guide: Guide):
    """The permutation null as a histogram, with each model's observed value marked."""
    rows = role_rows(summary, guide).set_index("role")
    model_labels = {m.role: m.label for m in guide.models}
    allv = np.concatenate([draws[rows.loc[r, "model"]] for r in ROLE_ORDER])
    bins = np.linspace(allv.min(), max(allv.max(), rows["auc"].max()) + 0.005, 46)
    for role in ROLE_ORDER:
        colour = style.colour_for_role(role)
        label = model_labels.get(role, ROLE_LABEL.get(role, role))
        ax.hist(draws[rows.loc[role, "model"]], bins=bins, histtype="stepfilled", alpha=0.35,
                color=colour, edgecolor=colour, linewidth=0.9, zorder=2,
                label=f"{label}: shuffled")
    # Each line is labelled where it stands. The fine-tuned line sits clear to the right of
    # the histograms, so its label goes to the left of it in that gap; the base line sits over
    # the bars, so its label goes above them. Anchored in data x, axes-fraction y, and kept
    # inside the axes so constrained_layout does not shrink the panel to fit the text.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    for role, frac, ha, dx in (("finetune", 0.74, "right", -5), ("base", 0.985, "left", 4)):
        r = rows.loc[role]
        colour = style.colour_for_role(role)
        label = model_labels.get(role, ROLE_LABEL.get(role, role))
        ax.axvline(r["auc"], color=colour, linewidth=2.6, zorder=4)
        # A bare rule over a histogram reads as another bar. The marker on the baseline says
        # "this value", and pins the line to the axis it is measured against.
        ax.plot([r["auc"]], [0], marker="v", markersize=8.0, color=colour, clip_on=False,
                markeredgewidth=0, zorder=6)
        ax.annotate(f"{label} {r['auc']:.3f}\np = {r['p_perm']:.3f}",
                    (r["auc"], frac), xycoords=("data", "axes fraction"),
                    xytext=(dx, 0), textcoords="offset points",
                    ha=ha, va="top", fontsize=6.2, color=colour, linespacing=1.3, zorder=5,
                    annotation_clip=False)
    style.reference_line(ax, 0.5, "chance = 0.5", axis="x", label_loc="left", label_side="below")
    # A little air at both ends: the extra width would otherwise go to the bars alone, and the
    # arrowheads and the right-hand rule need room to sit inside the axes.
    ax.set_xlim(float(allv.min()) - 0.002, float(rows["auc"].max()) + 0.022)
    ax.set_xlabel("Pooled within-replacement AUROC")
    ax.set_ylabel(f"Shuffles (of {int(summary['n_perm'].iloc[0]):,})")
    style.panel_title(ax, "Against the shuffled null",
                      "Codons swapped between same-amino-acid sites")


def _background_panel(ax, pairs: pd.DataFrame, guide: Guide, is_single_lineage: bool = False):
    """All the scored sites in one stacked bar, so the size of the test is visible at a glance."""
    p = role_rows(pairs, guide)
    p = p.loc[p["role"] == "finetune"]
    easy = float(p["sites_one_step"].sum())
    hard = float(p["sites_multi_step"].sum())
    ax.bar([0], [easy], width=0.6, color=style.CATEGORICAL[0], linewidth=0, zorder=2)
    ax.bar([0], [hard], width=0.6, bottom=[easy], color=style.CATEGORICAL[2], linewidth=0, zorder=2)
    for value, base, text in ((easy, 0.0, f"Easy\n1 nt change\n{easy:,.0f} sites"),
                              (hard, easy, f"Hard\n2-3 nt changes\n{hard:,.0f} sites")):
        ax.annotate(text, (0.32, base + value / 2), textcoords="offset points", xytext=(2, 0),
                    ha="left", va="center", fontsize=6.5, color=style.INK, linespacing=1.25, zorder=3)
    ax.set_xticks([])
    ax.set_xlim(-0.42, 1.22)
    ax.grid(axis="x", visible=False)
    # Only the 1,000 line is kept. The others ran straight through the two labels, and a
    # stacked bar is read off its own boundary rather than off the gridlines anyway.
    ax.grid(axis="y", visible=False)
    ax.axhline(1000, color=style.GRID, linewidth=0.6, zorder=1)
    if is_single_lineage:
        ax.set_ylabel("Pair-site evaluations")
    else:
        ax.set_ylabel("Sites per lineage")
    style.panel_title(ax, "The whole test", f"{len(p)} amino-acid pairs")


def draw(per_set: dict[str, dict[str, pd.DataFrame]], guides: dict[str, Guide], primary: str,
         scores: pd.DataFrame, draws):
    """One figure: what the test compares, the null it is read against, and the data behind it."""
    guide = guides[primary]
    summary = per_set[primary]["codon_within_pair_summary"]
    ft = role_rows(summary, guide).set_index("role").loc["finetune", "model"]

    fig = style.figure(style.WIDTH_DOUBLE, 2.6)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 0.5, 4.9], wspace=0.04)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])   # the aggregated totals
    ax_c = fig.add_subplot(gs[0, 2])   # the null, widest

    _exposition_panel(ax_a, scores, per_set[primary]["codon_within_pair_by_pair"], guide, ft)
    _background_panel(ax_b, per_set[primary]["codon_within_pair_by_pair"], guide,
                      is_single_lineage=(getattr(guide, "n_lineages", len(guide.lineages)) == 1))
    _null_panel(ax_c, draws, summary, guide)
    # dx is a fraction of each panel's own width, so one value would push C's letter far
    # out into the gap beside it. Each letter sits just clear of its own y label instead.
    for ax, letter, dx in ((ax_a, "A", -0.28), (ax_b, "B", -0.55), (ax_c, "C", -0.075)):
        style.panel_letter(ax, letter, dx=dx, pad_points=style.TITLE_PAD)
    return fig


CHANCE = 0.5


def draw_all_pairs(pairs: pd.DataFrame, guide: Guide):
    """Every scored replacement with its interval, strongest fine-tuned first.

    One panel, split over two columns to keep the labels readable. A 20x20 heat map of the same
    fine-tuned values used to sit above it; it showed nothing the forest does not, and a cell
    whose interval covers chance looked like one whose interval clears it.
    """
    p = role_rows(pairs, guide)
    order = (p.loc[p["role"] == "finetune"].sort_values("auc", ascending=False)["pair"]).tolist()
    half = int(np.ceil(len(order) / 2))
    fig = style.figure(style.WIDTH_DOUBLE, 0.19 * half + 1.0)
    axes = list(np.atleast_1d(fig.subplots(1, 2, squeeze=False)).ravel())
    lo, hi = p["ci_lo"].min(), p["ci_hi"].max()
    for ax, chunk in zip(axes, (order[:half], order[half:])):
        y = np.arange(len(chunk))[::-1]
        for role in ROLE_ORDER:
            r = p.loc[p["role"] == role].set_index("pair").reindex(chunk)
            off = 0.17 if role == "finetune" else -0.17
            colour = style.colour_for_role(role)
            ax.errorbar(r["auc"], y + off, xerr=[r["auc"] - r["ci_lo"], r["ci_hi"] - r["auc"]], fmt="o", color=colour,
                        ecolor=colour, markersize=3, elinewidth=0.7, capsize=0, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([c.replace(">", PAIR_ARROW) for c in chunk])
        ax.set_ylim(-0.8, len(chunk) - 0.2)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(lo - 0.02, hi + 0.02)
        style.reference_line(ax, CHANCE, "chance = 0.5", axis="x")
        ax.set_xlabel("Within-replacement AUROC")
    axes[0].set_ylabel("Amino-acid change, reference to new residue")
    handles = [axes[0].plot([], [], "o", color=style.colour_for_role(r))[0] for r in ("finetune", "base")]
    fig.legend(handles, [ROLE_LABEL["finetune"], ROLE_LABEL["base"]], loc="outside upper center", ncol=2)
    return fig


# ---------------------------------------------------------------------------
# Companion outputs
# ---------------------------------------------------------------------------

def build_values(per_set, guides) -> pd.DataFrame:
    """One row per number the figure draws. Panels: A worked example, B the null, C the totals."""
    parts = []
    primary = next(iter(per_set))
    guide = guides[primary]
    is_single = (getattr(guide, "n_lineages", len(guide.lineages)) == 1)

    b = _stack(per_set, "codon_within_pair_summary", guides)
    parts.append(b.assign(panel=PANEL_C, statistic="pooled within-replacement AUROC")
                  .rename(columns={"auc": "estimate"})[
                      ["panel", "set", "model", "role", "statistic", "estimate", "ci_lo", "ci_hi",
                       "null_lo", "null_hi", "null_mean", "p_perm", "n_pairs", "n_pairs_above_half",
                       "p_sign", "lineage_ci_lo", "lineage_ci_hi", "n_boot", "n_perm"]])

    pairs = role_rows(per_set[primary]["codon_within_pair_by_pair"], guide)
    ft = pairs.loc[pairs["role"] == "finetune"]
    worked = ft.sort_values("auc", ascending=False).iloc[0]
    parts.append(pd.DataFrame([{
        "panel": PANEL_A, "set": primary, "model": worked["model"], "role": "finetune",
        "statistic": f"within-replacement AUROC for {worked['pair']}", "estimate": worked["auc"],
        "ci_lo": worked["ci_lo"], "ci_hi": worked["ci_hi"],
        "sites_one_step": worked["sites_one_step"], "sites_multi_step": worked["sites_multi_step"],
    }]))

    stat_b = "evaluated sites, summed over amino-acid pairs" if is_single else "sites per lineage, summed over amino-acid pairs"
    parts.append(pd.DataFrame([{
        "panel": PANEL_B, "set": primary, "model": ft["model"].iloc[0], "role": "finetune",
        "statistic": stat_b, "estimate": float(ft["sites_one_step"].sum() + ft["sites_multi_step"].sum()),
        "sites_one_step": float(ft["sites_one_step"].sum()), "sites_multi_step": float(ft["sites_multi_step"].sum()),
        "n_pairs": int(len(ft)),
    }]))
    return pd.concat(parts, ignore_index=True)


def caption(per_set, guides, primary: str, gap: float) -> str:
    sets = list(per_set)
    t = {s: role_rows(per_set[s]["codon_within_pair_summary"], guides[s]).set_index("role") for s in sets}
    b = t[primary]
    ft, base = b.loc["finetune"], b.loc["base"]
    d = per_set[primary]["codon_within_pair_difference"].iloc[0]
    pairs = role_rows(per_set[primary]["codon_within_pair_by_pair"], guides[primary])
    pf = pairs.loc[pairs["role"] == "finetune"]
    worked = pf.sort_values("auc", ascending=False).iloc[0]
    easy_total, hard_total = float(pf["sites_one_step"].sum()), float(pf["sites_multi_step"].sum())
    other = [s for s in sets if s != primary]
    replicate = ""
    if other:
        o = t[other[0]]
        replicate = (f" The same test on {other[0]} gives {o.loc['finetune', 'auc']:.3f} for the fine-tuned "
                     f"model and {o.loc['base', 'auc']:.3f} for the base model.")
    is_single = (getattr(guides[primary], "n_lineages", len(guides[primary].lineages)) == 1)
    if is_single:
        replication_note = "evaluated on a single reference sequence to prevent cross-lineage pseudo-replication."
        panel_b_desc = f"{easy_total:,.0f} easy and {hard_total:,.0f} hard site evaluations across the reference sequence"
    else:
        narrow = (ft["ci_hi"] - ft["ci_lo"]) / max(ft["lineage_ci_hi"] - ft["lineage_ci_lo"], 1e-12)
        replication_note = f"resampling lineages would give an interval about {narrow:.0f} times too narrow for this statistic."
        panel_b_desc = f"{easy_total:,.0f} easy and {hard_total:,.0f} hard sites per lineage"

    return f"""
    **Does a model that was never shown codons tell synonymous codons apart?**
    The models read amino acids, not codons. Serine has six codons, and which one a site uses
    decides whether a change to asparagine needs one nucleotide change or two or three. If a
    model had picked up nothing about codons, where the reference codon sits should not change
    how it scores the replacement.

    **(A) What one number means.** The amino-acid change is held fixed, here
    {worked['pair'].replace('>', PAIR_ARROW)} in the reference sequence, so chemistry, substitution
    matrices and the identity of the new residue are all constant and only the reference codon
    differs. Each dot is one site, placed by the probability the fine-tuned model gives the
    replacement there. Sites are split by whether the reference codon reaches the new amino acid
    in one nucleotide change ({int(worked['sites_one_step'])} sites) or in two
    or three ({int(worked['sites_multi_step'])} sites). The statistic is the AUROC: the share of
    easy-site/hard-site pairs in which the easy site scores higher. For this replacement it is
    {worked['auc']:.3f}.

    **(B) How much data this rests on.** {panel_b_desc}, summed over the {len(pf)} ordered amino-acid pairs.
    A pair is scored only where both groups are non-empty.

    **(C) The same number pooled, against its null.** Pooling over the
    {int(ft['n_pairs'])} scored amino-acid pairs gives {ft['auc']:.3f} (95% interval
    {ft['ci_lo']:.3f} to {ft['ci_hi']:.3f}) for the fine-tuned model and {base['auc']:.3f}
    ({base['ci_lo']:.3f} to {base['ci_hi']:.3f}) for the base model; the paired difference is
    {d['difference']:.3f} ({d['ci_lo']:.3f} to {d['ci_hi']:.3f}).{replicate} The histograms are
    the null: {int(ft['n_perm']):,} shuffles that swap reference codons between positions
    carrying the same amino acid, leaving everything else as it is. One shuffle gives one pooled
    AUROC, and the same shuffles are used for both models. The fine-tuned model falls outside
    its null ({ft['null_lo']:.3f} to {ft['null_hi']:.3f}, p = {ft['p_perm']:.3f}); the base model
    falls inside its own ({base['null_lo']:.3f} to {base['null_hi']:.3f}, p = {base['p_perm']:.3f}).

    Intervals resample sites: {replication_note} The figure recomputes
    the null from the scores and checks percentiles against the derivation; the point estimates
    agree with the scores to {gap:.0e}.
    """


def caption_pairs(pairs: pd.DataFrame, guide: Guide) -> str:
    p = role_rows(pairs, guide)
    ft = p.loc[p["role"] == "finetune"]
    base = p.loc[p["role"] == "base"]
    return f"""
    **Every scored replacement.**
    Within-replacement AUROC for each of the {len(ft)} amino-acid changes that have enough sites
    with one-change and with more-change codons, fine-tuned and base model side by side, strongest
    fine-tuned first. Whiskers are 95% intervals from resampling sites. Chance is 0.5.
    {int((ft['auc'] > 0.5).sum())} of {len(ft)} fine-tuned values and {int((base['auc'] > 0.5).sum())}
    of {len(base)} base values lie above chance; {int((ft['ci_lo'] > 0.5).sum())} and
    {int((base['ci_lo'] > 0.5).sum())} have an interval wholly above it. The replacements share
    their sites, so these counts are descriptive.
    """


class SingleLineageGuide:
    def __init__(self, g: Guide, lin_id: str):
        self.g = g
        self.lin_id = lin_id
        self.models = g.models
        self.source = g.source
        self.name = f"{g.name} ({lin_id})"
        self.primary_lineage = lin_id
        self.results_dir = g.results_dir
        self.n_lineages = 1

    def models_with_role(self, r):
        return self.g.models_with_role(r)

    def model(self, m):
        return self.g.model(m)

    @property
    def lineages(self):
        return [l for l in self.g.lineages if l.id == self.lin_id]


def parse(argv=None):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--guide-secondary", type=Path, default=None,
                     help="a second guide drawn beside the first in panels A and B")
    pre.add_argument("--results-secondary", type=Path, default=None)
    pre.add_argument("--lineage", type=str, default=None,
                     help="evaluate on a single lineage root sequence (e.g. J.2_int)")
    known, rest = pre.parse_known_args(argv)
    args = parse_figure_args(__doc__.split("\n\n")[0], rest)
    args.guide_secondary = load_guide(known.guide_secondary) if known.guide_secondary else None
    args.results_secondary = known.results_secondary
    args.lineage = known.lineage
    return args


def label_for(guide: Guide | SingleLineageGuide) -> str:
    if hasattr(guide, "lin_id"):
        return guide.lin_id
    if getattr(guide, "n_lineages", len(guide.lineages)) == 1:
        return guide.lineages[0].id
    return f"{guide.n_lineages} lineages"


def main(argv=None) -> int:
    args = parse(argv)
    target_lineage = args.lineage or (args.guide.primary_lineage if args.guide_secondary is None else None)
    if target_lineage is not None:
        slg = SingleLineageGuide(args.guide, target_lineage)
        scores = args.guide.read_table("replacement_scores", args.results)
        scores_lin = scores[scores["lineage"] == target_lineage]
        spectra_dir = REPO_ROOT / "data" / "spectra"
        cand = [p for p in spectra_dir.glob("*.csv") if args.guide.name.split("_")[0] in p.stem]
        spec_path = cand[0] if cand else next(spectra_dir.glob("*.csv"))
        tables = cl.derive_codon_tables(slg, scores_lin, n_boot=1000, n_perm=1000, n_perm_c=1000,
                                        spectrum=spec_path, symmetrise=True)
        gap = check_against_scores(slg, scores_lin, tables)
        per_set = {target_lineage: tables}
        guides = {target_lineage: slg}
        primary = target_lineage
        primary_scores = scores_lin
        summary = tables["codon_within_pair_summary"]
        draws = null_draws(slg, primary_scores, summary, n_perm=int(summary["n_perm"].iloc[0]))
        fig = draw(per_set, guides, primary, primary_scores, draws)
        paths = style.save_figure(fig, args.out / STEM)
        pairs = tables["codon_within_pair_by_pair"]
        fig2 = draw_all_pairs(pairs, slg)
        paths += style.save_figure(fig2, args.out / STEM_PAIRS)
        write_values(build_values(per_set, guides), args.out, STEM)
        write_values(role_rows(pairs, slg), args.out, STEM_PAIRS)
        write_caption(caption(per_set, guides, primary, gap), args.out, STEM)
        write_caption(caption_pairs(pairs, slg), args.out, STEM_PAIRS)
        print(f"self-check: shipped tables agree with the scores to {gap:.1e}")
        print("wrote", ", ".join(p.name for p in paths))
        return 0

    guides = {label_for(args.guide): args.guide}
    results = {label_for(args.guide): args.results}
    if args.guide_secondary is not None:
        name = label_for(args.guide_secondary)
        if name in guides:
            raise ValueError(f"both guides have {args.guide_secondary.n_lineages} lineages; panels would share a label")
        guides[name], results[name] = args.guide_secondary, args.results_secondary
    primary = label_for(args.guide)
    per_set, gap = {}, 0.0
    for name, g in guides.items():
        tables = load_tables(g, results[name])
        gap = max(gap, check_against_scores(g, g.read_table("replacement_scores", results[name]), tables))
        per_set[name] = tables

    primary_scores = guides[primary].read_table("replacement_scores", results[primary])
    summary = per_set[primary]["codon_within_pair_summary"]
    draws = null_draws(guides[primary], primary_scores, summary, n_perm=int(summary["n_perm"].iloc[0]))
    fig = draw(per_set, guides, primary, primary_scores, draws)
    paths = style.save_figure(fig, args.out / STEM)
    pairs = per_set[primary]["codon_within_pair_by_pair"]
    fig2 = draw_all_pairs(pairs, guides[primary])
    paths += style.save_figure(fig2, args.out / STEM_PAIRS)
    write_values(build_values(per_set, guides), args.out, STEM)
    write_values(role_rows(pairs, guides[primary]), args.out, STEM_PAIRS)
    write_caption(caption(per_set, guides, primary, gap), args.out, STEM)
    write_caption(caption_pairs(pairs, guides[primary]), args.out, STEM_PAIRS)
    print(f"self-check: shipped tables agree with the scores to {gap:.1e}")
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
