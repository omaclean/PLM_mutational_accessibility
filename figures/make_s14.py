"""S14: does the lineage set the analysis rests on support the conclusions drawn from it?

    python figures/make_s14.py --guide guides/h3n2_13lineage.yaml --guide-compare guides/h3n2_5lineage.yaml

Two checks, one per panel. ``--guide`` is the main lineage set; ``--guide-compare`` (optional) is a
second set panel A is also drawn on, so a result can be seen to depend on the set or not. Every
panel states how many lineages it used, counted from its guide.

Trimmed from four panels to two on 13 Oct 2026. The dropped panels were the codon-persistence
null and the resampling-unit comparison; both are still computed by the audit scripts and their
tables still ship, so they can be restored from ``dev/lineage_validity/`` if wanted.

A  Within-replacement AUROC. With the amino-acid replacement held fixed, does the model put the
   target higher where the reference codon is one nucleotide step away?
B  Does the model's total probability of leaving a residue track how often a mutation at that
   codon is silent? Compared with a permutation null.
C  Does the fine-tuned model's advantage over the base model survive on replacements that never
   occur in its fine-tuning corpus?
D  The same AUROC with an interval from resampling lineages and from resampling sites.

Reads tables built by ``tools/export_s14_s08.py`` from tables the repository already ships. Before
drawing, the point estimates are recomputed from ``replacement_scores`` (and, for C and D,
``replacement_observed`` and ``corpus_membership``) and must equal the shipped tables to 1e-9, so a
table cannot drift from the scores it came from. Intervals are not recomputed here (two thousand
resamples each); ``tests/test_s14.py`` rebuilds them from the archive's seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd

from mutacc import audit, style
from mutacc.guide import Guide, load_guide
from mutacc.io import write_caption, write_values

STEM = "s14_lineage_set_validity"
SELF_CHECK_TOL = 1e-9
ROLE_ORDER = ("finetune", "base")
ROLE_LABEL = {"base": "Base model", "finetune": "Fine-tuned model"}
DODGE = 0.2
UNIT_ORDER = ("lineage", "site")
UNIT_LABEL = {"lineage": "Resampling\nlineages", "site": "Resampling\nsites"}
VALUE_COLUMNS = ["panel", "series", "group", "measure", "value", "lo", "hi", "n", "n_lineages", "guide"]
PANEL_LETTERS = [chr(ord("A") + i) for i in range(2)]
Y_PAD = 0.12  # fraction of the plotted range left around the data


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def role_key(guide: Guide) -> dict[str, str]:
    """``{role: model key}`` for the one base and one fine-tuned model of a guide."""
    out = {}
    for role in ROLE_ORDER:
        found = guide.models_with_role(role)
        if len(found) != 1:
            raise ValueError(f"{guide.source.name}: expected one {role!r} model, found {len(found)}")
        out[role] = found[0].key
    return out


def by_role(table: pd.DataFrame, guide: Guide, what: str) -> pd.DataFrame:
    """Index a per-model table by role, refusing missing, extra or repeated models."""
    keys = role_key(guide)
    if table["model"].duplicated().any():
        raise ValueError(f"{what}: a model appears twice")
    unknown = set(table["model"]) - set(keys.values())
    if unknown:
        raise ValueError(f"{what}: models {sorted(unknown)} are not in the guide")
    out = table.set_index("model")
    missing = [m for m in keys.values() if m not in out.index]
    if missing:
        raise ValueError(f"{what}: no row for model(s) {missing}")
    return out.rename(index={v: k for k, v in keys.items()})


def set_label(guide: Guide) -> str:
    return f"{guide.n_lineages} lineages"


def check_point_estimates(guide: Guide, results: Path, tables: dict) -> dict[str, float]:
    """Recompute every point estimate from the scores and compare with the shipped tables.

    Returns the largest disagreement per check; raises if any exceeds ``SELF_CHECK_TOL``.
    """
    scores = guide.read_table("replacement_scores", results)
    keys = role_key(guide)
    gaps = {}
    within = tables["within"].set_index("model")
    for role, key in keys.items():
        r = audit.within_replacement(scores.loc[scores["model"] == key], n_boot=0)
        gaps[f"within_replacement/{role}"] = abs(r["auc"] - float(within.loc[key, "auc"]))
        if int(within.loc[key, "n_pairs_scored"]) != r["n_pairs_scored"]:
            raise ValueError(f"{guide.name}: shipped pair count differs from the scores for {role}")
    if "interval" in tables:
        observed = guide.read_table("replacement_observed", results)
        interval = tables["interval"].set_index("model")
        for role, key in keys.items():
            mean = audit.per_lineage_auroc(scores.loc[scores["model"] == key], observed).mean()
            gaps[f"interval_unit/{role}"] = abs(mean - float(interval.loc[key, "mean_auroc"]))
        membership = guide.read_table("corpus_membership", results)
        split = audit.corpus_split(scores, observed, membership, finetune=keys["finetune"], base=keys["base"], n_boot=0)
        shipped = tables["split"].set_index("in_corpus")
        for row in split.itertuples():
            for col in ("auroc_finetune", "auroc_base", "advantage"):
                gaps[f"corpus_split/{row.in_corpus}/{col}"] = abs(getattr(row, col) - float(shipped.loc[row.in_corpus, col]))
            if int(shipped.loc[row.in_corpus, "n_cells"]) != row.n_cells:
                raise ValueError(f"{guide.name}: shipped cell count differs from the scores")
    worst = max(gaps.values())
    if worst > SELF_CHECK_TOL:
        bad = {k: v for k, v in gaps.items() if v > SELF_CHECK_TOL}
        raise ValueError(f"{guide.name}: shipped tables disagree with the scores: {bad}")
    return gaps


def load_tables(guide: Guide, results: Path, *, full: bool) -> dict:
    tables = {"within": guide.read_table("audit_within_replacement", results),
              "persistence": guide.read_table("audit_persistence", results)}
    if full:
        tables["interval"] = guide.read_table("audit_interval_unit", results)
        tables["split"] = guide.read_table("audit_corpus_split", results)
    return tables


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _limits(ax, values, *, include=None):
    values = np.asarray(values, dtype=float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if include is not None:
        lo, hi = min(lo, include), max(hi, include)
    pad = Y_PAD * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def _dodged(ax, sets, frames, value, lo, hi, guides):
    """One point per (set, model) with an interval bar. Returns legend handles by role."""
    handles = {}
    x = np.arange(len(sets), dtype=float)
    for j, role in enumerate(ROLE_ORDER):
        off = (j - 0.5) * 2 * DODGE
        vals = np.array([frames[i].loc[role, value] for i in range(len(sets))], dtype=float)
        err = np.vstack([vals - [frames[i].loc[role, lo] for i in range(len(sets))],
                         [frames[i].loc[role, hi] for i in range(len(sets))] - vals])
        h = ax.errorbar(x + off, vals, yerr=err, fmt="o", color=style.colour_for_role(role), capsize=2.5,
                        elinewidth=1.2, markersize=4.5, zorder=3)
        handles[role] = h
    ax.set_xticks(x)
    ax.set_xticklabels([set_label(g) for g in guides])
    ax.set_xlim(-0.6, len(sets) - 0.4)
    ax.grid(axis="x", visible=False)
    return handles


def draw(guides: list[Guide], tables: list[dict], main_tables: dict):
    """``guides`` and ``tables`` are ordered by lineage count; the last is the main set."""
    fig = style.figure(style.WIDTH_DOUBLE, 3.0)
    axA, axC = fig.subplots(1, 2)
    sets = range(len(guides))

    # A: within-replacement AUROC --------------------------------------------------------------
    fr = [by_role(t["within"], g, "audit_within_replacement") for g, t in zip(guides, tables)]
    handles = _dodged(axA, sets, fr, "auc", "ci_lo", "ci_hi", guides)
    style.reference_line(axA, 0.5, "chance = 0.5", label_loc="left")
    _limits(axA, np.concatenate([[f.loc[r, c] for r in ROLE_ORDER] for f in fr for c in ("ci_lo", "ci_hi")]), include=0.5)
    style.finalise_axes(axA, ylabel="Within-replacement AUROC")
    style.panel_title(axA, "Codon distance, replacement held fixed",
                      "Do one-step changes rank higher? 95% site interval")
    axA.legend([handles[r] for r in ROLE_ORDER], [ROLE_LABEL[r] for r in ROLE_ORDER], loc="upper left",
               fontsize=6, ncol=2, borderaxespad=0.2, columnspacing=1.0)

    # C: corpus split -------------------------------------------------------------------------
    split = main_tables["split"].sort_values("in_corpus").reset_index(drop=True)
    if list(split["in_corpus"].astype(bool)) != [False, True]:
        raise ValueError("audit_corpus_split must hold exactly one row for absent and one for present replacements")
    xc = np.arange(2)
    adv = split["advantage"].to_numpy(float)
    axC.errorbar(xc, adv, yerr=np.vstack([adv - split["ci_lo"], split["ci_hi"] - adv]), fmt="o",
                 color=style.colour_for_role("finetune"), elinewidth=1.2, capsize=3, markersize=5, zorder=3)
    axC.set_xlim(-0.6, 1.6)
    for xi, r in zip(xc, split.itertuples()):
        axC.annotate(f"{r.advantage:+.3f}", (xi, r.ci_hi), textcoords="offset points", xytext=(0, 3),
                     ha="center", va="bottom", fontsize=6.5, color=style.INK)
    axC.set_xticks(xc)
    axC.set_xticklabels([f"{'Absent' if not r.in_corpus else 'Present'}\n"
                         f"{int(r.n_observed):,}/{int(r.n_cells):,} seen" for r in split.itertuples()])
    style.reference_line(axC, 0.0, "no advantage", label_loc="left")
    axC.grid(axis="x", visible=False)
    _limits(axC, np.r_[split["ci_lo"], split["ci_hi"], 0.0], include=0.0)
    style.finalise_axes(axC, ylabel="AUROC of fine-tuned minus base")
    style.panel_title(axC, "Advantage is not memorisation",
                      f"By presence in the fine-tuning corpus ({set_label(guides[-1])})")

    style.panel_letters([axA, axC], dx=-0.17, dy=1.12)
    return fig


# ---------------------------------------------------------------------------
# Values and caption
# ---------------------------------------------------------------------------

def build_values(guides, tables, main_tables) -> pd.DataFrame:
    rows = []

    def add(panel, series, group, measure, value, lo, hi, n, guide):
        rows.append({"panel": panel, "series": series, "group": group, "measure": measure, "value": value,
                     "lo": lo, "hi": hi, "n": n, "n_lineages": guide.n_lineages, "guide": guide.name})

    for g, t in zip(guides, tables):
        w = by_role(t["within"], g, "within")
        for role in ROLE_ORDER:
            add("A", role, set_label(g), "within-replacement AUROC", w.loc[role, "auc"], w.loc[role, "ci_lo"],
                w.loc[role, "ci_hi"], w.loc[role, "n_pairs_scored"], g)
    g = guides[-1]
    for r in main_tables["split"].itertuples():
        add("B", "advantage", "present" if r.in_corpus else "absent", "AUROC fine-tuned minus base", r.advantage,
            r.ci_lo, r.ci_hi, r.n_observed, g)
    return pd.DataFrame(rows, columns=VALUE_COLUMNS)


def _clear_of(lo, hi, value):
    return bool(lo > value or hi < value)


def caption(guides, tables, main_tables) -> str:
    g = guides[-1]
    sets = " and ".join(set_label(x) for x in guides)
    w = [by_role(t["within"], x, "within") for x, t in zip(guides, tables)]
    ft_clear = sum(_clear_of(f.loc["finetune", "ci_lo"], f.loc["finetune", "ci_hi"], 0.5) for f in w)
    bs_clear = sum(_clear_of(f.loc["base", "ci_lo"], f.loc["base", "ci_hi"], 0.5) for f in w)
    split = main_tables["split"].set_index("in_corpus")
    absent, present = split.loc[False], split.loc[True]
    a_text = "; ".join(
        f"{set_label(x)}: fine-tuned {f.loc['finetune', 'auc']:.3f} "
        f"[{f.loc['finetune', 'ci_lo']:.3f}, {f.loc['finetune', 'ci_hi']:.3f}], base {f.loc['base', 'auc']:.3f} "
        f"[{f.loc['base', 'ci_lo']:.3f}, {f.loc['base', 'ci_hi']:.3f}]" for x, f in zip(guides, w))
    return f"""
    **Does the lineage set support the conclusions drawn from it?**
    Four checks on the lineage sets ({sets}) behind the main figures. Lineages here are disjoint sets of
    sequences, but they share most of their sites, so a result seen once per lineage is not seen
    independently. AUROC is the area under the receiver-operating-characteristic curve: the chance that a
    case of interest ranks above one that is not; 0.5 is chance.

    (A) With the amino-acid replacement held fixed (for example alanine to valine), does the model rank
    the replacement higher at sites where the reference codon is one nucleotide step away than where it is
    two or three? Inside one replacement every amino-acid quantity, including chemistry and substitution
    matrices, is constant, so only codon information is left. Values are weighted means of per-replacement
    AUROCs; bars are 95% intervals from resampling sites, because a site recurs in every lineage. {a_text}.
    The fine-tuned model's interval excludes chance on {ft_clear} of {len(w)} sets and the base model's on
    {bs_clear} of {len(w)}.

    (B) On the {set_label(g)} set, the fine-tuned model's AUROC advantage over the base model for predicting
    which replacements occur, split by whether each replacement is present in the fine-tuning corpus. Cells
    are positions by amino acids; a cell counts as observed if any lineage observed it. Advantage on
    replacements absent from the corpus: {absent['advantage']:+.3f} [{absent['ci_lo']:+.3f}, {absent['ci_hi']:+.3f}]
    ({int(absent['n_observed'])} observed); present: {present['advantage']:+.3f}
    [{present['ci_lo']:+.3f}, {present['ci_hi']:+.3f}] ({int(present['n_observed'])} observed). The
    absent-group estimate rests on fewer observed replacements and is the noisier one. Intervals resample
    sites.

    """


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--guide", required=True, type=Path, help="main guide YAML (all four panels)")
    parser.add_argument("--guide-compare", type=Path, default=None,
                        help="a second guide; panels A and B are also drawn on it")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    args = parser.parse_args(argv)
    args.guide = load_guide(args.guide)
    args.guide_compare = None if args.guide_compare is None else load_guide(args.guide_compare)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    main_guide = args.guide
    guides = [main_guide] if args.guide_compare is None else [args.guide_compare, main_guide]
    guides.sort(key=lambda g: g.n_lineages)
    if len({g.name for g in guides}) != len(guides):
        raise ValueError("--guide and --guide-compare name the same run")
    guides = [g for g in guides if g.name != main_guide.name] + [main_guide]  # main set last
    main_tables = load_tables(main_guide, main_guide.results_dir, full=True)
    tables = [main_tables if g is main_guide else load_tables(g, g.results_dir, full=False) for g in guides]

    worst = 0.0
    for g, t in zip(guides, tables):
        worst = max(worst, max(check_point_estimates(g, g.results_dir, t).values()))
    fig = draw(guides, tables, main_tables)
    paths = style.save_figure(fig, args.out / STEM)
    write_values(build_values(guides, tables, main_tables), args.out, STEM)
    write_caption(caption(guides, tables, main_tables), args.out, STEM)
    print(f"self-check: shipped point estimates agree with the scores to {worst:.1e}")
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
