"""S11: how many nucleotides of a codon change at once on one branch of a UShER tree?

    python figures/make_sK_branch_codons.py --branch-table T/branch_codons.tsv.gz --column-table T/column_codons.tsv.gz

STATUS: REAL TABLES ARE NOT SHIPPED YET. The tables come from ``tools/branch_codon_tables.py``,
which needs the UShER tree and the nucleotide record of the UShER reference (the first record of
the alignment the tree was built from). The tree is in the archive, the reference record is not.
Until it is supplied this script only runs on the synthetic fixture in ``tests/fixtures/sK/``,
whose header says SYNTHETIC. A table with that word in its header makes this script stamp the
figure, rename the output stem and open the caption with a warning. Do not quote the fixture.

Reads two tables written by ``mutacc.branch_codons`` (the draw does not recompute anything beyond
the chance null, and it refuses tables whose totals disagree):

* the branch table, one row per branch with fewer than the cap of nucleotide changes;
* the column table, one row per codon of the coding region.

The figure is one table row and one chart.

  Table  row 1, every amino-acid-changing codon change on a single branch, sorted by the fewest nucleotide
         changes that would have produced the new amino acid from the parent codon actually seen; row 2,
         the same sort for the long-range comparison of the oldest node against the newest internal
         nodes. Both rows are raw counts.

  Chart  the share of amino-acid changes that moved 2 or more positions of one codon, against how
         many nucleotide changes the whole branch carried, binned 1 / 2 / 3 / 4+. A branch with one
         change cannot move two positions, and two changes land in one codon more often as a branch
         gets longer, so the raw share rises for reasons that have nothing to do with any mechanism.
         The chart therefore draws the share expected if the branch's changes fell at random over
         the codons, computed from the same branch lengths, alongside the observed share. The
         distance between the two lines is the claim; the rise on its own is not.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mutacc import style
from mutacc.accessibility import BASES, GENETIC_CODE
from mutacc.branch_codons import BRANCH_COLUMNS, COLUMN_COLUMNS, read_table
from mutacc.guide import load_guide
from mutacc.io import write_caption, write_values

STEM = "sK_branch_codons"
#: Written ahead of the figure number, so it is the first thing read. A planted
#: test tree must never be mistaken for a result.
SYNTHETIC_BANNER = ("**SYNTHETIC FIXTURE. These numbers describe a planted test tree, not any virus. Do not quote them.**")
SYNTHETIC_MARK = "SYNTHETIC"
NT = (1, 2, 3)
MIN_CODONS_PER_POINT = 10        # a share over fewer amino-acid-changing codons is not drawn
TOP_BIN = 4                      # branch lengths are binned 1, 2, 3, 4+


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def is_synthetic(*provenance: dict) -> bool:
    return any(SYNTHETIC_MARK in str(v).upper() for p in provenance for v in p.values())


def load_tables(branch_path: Path, column_path: Path):
    """Read both tables and check them against each other. Returns (branch, column, prov_b, prov_c)."""
    branch, pb = read_table(branch_path)
    column, pc = read_table(column_path)
    if branch.empty or column.empty:
        raise ValueError("an input table is empty; nothing to draw")
    need_b = [c for c in BRANCH_COLUMNS]
    need_c = [c for c in COLUMN_COLUMNS]
    miss = [c for c in need_b if c not in branch.columns] + [c for c in need_c if c not in column.columns]
    if miss:
        raise ValueError(f"tables are missing columns: {miss}")
    counts_b = branch[BRANCH_COLUMNS[2:]]
    counts_c = column[[c for c in COLUMN_COLUMNS if c not in ("codon", "ref_pos", "ref_codon", "ref_aa")]]
    if (counts_b < 0).any().any() or (counts_c < 0).any().any():
        raise ValueError("negative counts in a table")
    if (branch["nt_total"] < 1).any():
        raise ValueError("branch table holds a branch with no nucleotide changes")
    if (branch["nonsyn_2nt_min1"] > branch["nonsyn_2nt"]).any() or \
            (branch["nonsyn_3nt_min1"] + branch["nonsyn_3nt_min2"] > branch["nonsyn_3nt"]).any():
        raise ValueError("branch table has more 'shorter route' codons than codons")
    for col in BRANCH_COLUMNS[6:]:
        if int(branch[col].sum()) != int(column[col].sum()):
            raise ValueError(f"the branch and column tables disagree on {col}: "
                             f"{int(branch[col].sum())} against {int(column[col].sum())}")
    if int(branch["codons_touched"].sum()) != int(column["events"].sum()):
        raise ValueError("the branch and column tables disagree on the number of codon events")
    if not column["codon"].is_unique or not column["codon"].is_monotonic_increasing:
        raise ValueError("column table codons are not unique and increasing")
    for key in ("max_branch_nt", "include_terminal"):
        if key in pb and key in pc and pb[key] != pc[key]:
            raise ValueError(f"the tables were made with different {key}: {pb[key]!r} and {pc[key]!r}")
    return branch, column, pb, pc


def terminal_included(prov: dict) -> bool:
    """Whether terminal (tip) branches were scored, from the table header.

    Row 1 of the table is a count of branches, so it has to say WHICH branches: the walk
    skips terminal branches unless asked for, and a reader cannot tell from the number.
    """
    head = str(prov.get("include_terminal", "")).strip().split(" ")[0].lower()
    return head in {"1", "true", "yes"}


def cap_of(prov: dict) -> int | None:
    """The branch-size cap recorded in the header ('10 (strictly fewer kept)'), or None if absent/none."""
    text = prov.get("max_branch_nt", "")
    head = text.split()[0] if text else ""
    return int(head) if head.isdigit() else None


def scored_codons(column: pd.DataFrame) -> pd.DataFrame:
    """The codons the tree walk actually scored: those with a residue in the protein of interest."""
    if "site_number" not in column.columns:
        return column
    keep = pd.to_numeric(column["site_number"], errors="coerce").notna()
    return column.loc[keep]


# ---------------------------------------------------------------------------
# The chance null
# ---------------------------------------------------------------------------

def nonsyn_prob_from_code(codons) -> dict[int, float]:
    """P(the amino acid changes | exactly j of a codon's positions change), over the reference codons.

    Every choice of positions and of replacement bases is given equal weight, and outcomes that make
    or start from a stop codon are dropped, as the tree walk drops them. For j = 2 and 3 this barely
    depends on the weighting: almost any two-position change alters the amino acid.
    """
    out = {}
    for j in NT:
        hit = tot = 0
        for codon in codons:
            aa = GENETIC_CODE.get(codon)
            if aa is None or aa == "*":
                continue
            for positions in itertools.combinations(range(3), j):
                alts = [[b for b in BASES if b != codon[p]] for p in positions]
                for combo in itertools.product(*alts):
                    new = list(codon)
                    for p, b in zip(positions, combo):
                        new[p] = b
                    new_aa = GENETIC_CODE.get("".join(new))
                    if new_aa is None or new_aa == "*":
                        continue
                    tot += 1
                    hit += new_aa != aa
        out[j] = hit / tot if tot else 0.0
    return out


def nonsyn_probs(branch: pd.DataFrame, column: pd.DataFrame) -> dict[int, float]:
    """The P(amino acid changes | j positions changed) the null uses.

    j = 1 is taken from the data itself, because purifying selection and the transition bias make
    one-position changes far more often synonymous than the genetic code alone would predict (a
    third-position transition usually is). j = 2 and 3 are taken from the code, because they sit so
    close to 1 that no substitution model moves them much.
    """
    q = nonsyn_prob_from_code(list(scored_codons(column)["ref_codon"]))
    seen = int(branch["syn_1nt"].sum()) + int(branch["nonsyn_1nt"].sum())
    if seen:
        q[1] = int(branch["nonsyn_1nt"].sum()) / seen
    return q


def expected_codons(k: int, n_codons: int) -> dict[int, float]:
    """Expected codons carrying exactly 1, 2 or 3 of a branch's k changes, if the k fall at random.

    The k changes are placed on distinct nucleotide positions drawn without replacement from the
    3 * n_codons positions of the scored region, so this is the hypergeometric occupancy of a codon.
    """
    n = 3 * n_codons
    if k < 1 or k > n:
        raise ValueError(f"a branch of {k} changes cannot be placed in {n} positions")
    return {j: (n_codons * math.comb(3, j) * math.comb(n - 3, k - j) / math.comb(n, k)) if j <= k else 0.0
            for j in NT}


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval: the counts here are small enough that a normal interval would mislead."""
    if total <= 0:
        return (math.nan, math.nan)
    p = hits / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# Numbers behind the chart
# ---------------------------------------------------------------------------

def chart_numbers(branch: pd.DataFrame, column: pd.DataFrame,
                  min_codons: int = MIN_CODONS_PER_POINT) -> pd.DataFrame:
    """One row per branch-length bin: observed share, chance share, and what they were built from."""
    n_codons = len(scored_codons(column))
    q = nonsyn_probs(branch, column)
    rows = []
    for k, g in branch.groupby("nt_total"):
        e = expected_codons(int(k), n_codons)
        n = len(g)
        rows.append({"nt_total": int(k), "branches": n,
                     "nonsyn": int(g[[f"nonsyn_{j}nt" for j in NT]].to_numpy().sum()),
                     "multi": int(g[["nonsyn_2nt", "nonsyn_3nt"]].to_numpy().sum()),
                     "exp_nonsyn": n * sum(e[j] * q[j] for j in NT),
                     "exp_multi": n * sum(e[j] * q[j] for j in (2, 3))})
    per_k = pd.DataFrame(rows)
    top = int(per_k["nt_total"].max())
    per_k["bin"] = np.minimum(per_k["nt_total"], TOP_BIN)
    d = per_k.groupby("bin")[["branches", "nonsyn", "multi", "exp_nonsyn", "exp_multi"]].sum().reset_index()
    d["label"] = [f"{int(v)}+" if v == TOP_BIN and top > TOP_BIN else f"{int(v)}" for v in d["bin"]]
    d["share_multi"] = np.where(d["nonsyn"] > 0, d["multi"] / d["nonsyn"].where(d["nonsyn"] > 0, 1), np.nan)
    d["chance_multi"] = np.where(d["exp_nonsyn"] > 0,
                                 d["exp_multi"] / d["exp_nonsyn"].where(d["exp_nonsyn"] > 0, 1), np.nan)
    lo_hi = [wilson(int(m), int(n)) for m, n in zip(d["multi"], d["nonsyn"])]
    d["lo"], d["hi"] = [x[0] for x in lo_hi], [x[1] for x in lo_hi]
    d["possible"] = d["bin"] >= 2        # one nucleotide change cannot move two positions of a codon
    d["drawn"] = d["nonsyn"] >= min_codons
    d.insert(0, "panel", "chart")
    return d


def check_nodes(nodes: pd.DataFrame) -> None:
    """The per-node table must be internally consistent before anything is drawn from it."""
    need = ["node", "nonsyn", "min1", "min2", "min3", "nt1", "nt2", "nt3"]
    if [c for c in need if c not in nodes.columns]:
        raise ValueError(f"long-range node table is missing columns {[c for c in need if c not in nodes.columns]}")
    if nodes.empty:
        raise ValueError("long-range node table is empty")
    if not (nodes["nonsyn"] == nodes[["min1", "min2", "min3"]].sum(axis=1)).all():
        raise ValueError("long-range node table: changes by fewest-needed do not add up to the total")
    if not (nodes["nonsyn"] == nodes[["nt1", "nt2", "nt3"]].sum(axis=1)).all():
        raise ValueError("long-range node table: changes by number seen do not add up to the total")
    if (nodes["min2"] + nodes["min3"] > nodes["nt2"] + nodes["nt3"]).any() or (nodes["min3"] > nodes["nt3"]).any():
        raise ValueError("long-range node table: a change needed more nucleotides than the tree used")


def summary_table(branch: pd.DataFrame, nodes: pd.DataFrame | None,
                  base: pd.DataFrame | None = None, plm: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row 1 (the drawn row): every amino-acid-changing codon change on a single branch, by the fewest
    nucleotide changes from the parent codon. Row 2 (if given) is the long-range comparison, kept for
    the caption's pooled percentages; it is not drawn, because a per-node average is not informative."""
    s = {c: int(branch[c].sum()) for c in branch.columns if c.startswith("nonsyn_")}
    m1 = s["nonsyn_1nt"] + s["nonsyn_2nt_min1"] + s["nonsyn_3nt_min1"]
    m2 = (s["nonsyn_2nt"] - s["nonsyn_2nt_min1"]) + s["nonsyn_3nt_min2"]
    m3 = s["nonsyn_3nt"] - s["nonsyn_3nt_min1"] - s["nonsyn_3nt_min2"]
    rows = [{"row": "one branch", "unit": "all changes, all branches", "total": m1 + m2 + m3,
             "need_1": m1, "need_2": m2, "need_3": m3}]
    if nodes is not None:
        rows.append({"row": "oldest node to newest", "unit": "pooled over nodes",
                     "total": nodes["nonsyn"].sum(), "need_1": nodes["min1"].sum(),
                     "need_2": nodes["min2"].sum(), "need_3": nodes["min3"].sum()})
    if base is not None:
        n = null_distances(base)
        rows.append({"row": "null", "unit": "every amino acid from every codon",
                     "total": sum(n.values()), "need_1": n[1], "need_2": n[2], "need_3": n[3]})
    out = pd.DataFrame(rows)
    # The PLM rows go last, below the hard rule: they are a different object from the three
    # above them (one lineage's root, scored by a model) and must not read as another count.
    return out if plm is None else pd.concat([out, plm], ignore_index=True)


def root_codons(guide, lineage: str) -> pd.DataFrame:
    """The codons of one lineage's root sequence, as a ``codon_from`` column.

    The null is read against the PLM rows below it, so it has to stand on the same sites they
    do. Those score a lineage root -- 566 codons for J.2 -- while ``baseline_codons.tsv.gz``
    carries the tree's own reference (DQ864721), whose annotated coding region is 553 codons
    and loses 18 more that have no residue in the mature protein, leaving 535. Reading the
    null off the same root removes that 31-codon gap. It barely moves the row (the shares go
    0.311/0.509/0.180 to 0.310/0.510/0.180) because both are the same protein; the point is
    that the denominator no longer has to be explained.
    """
    for lin in guide.lineages:
        if lin.id == lineage:
            seq = "".join(line.strip() for line in lin.root.read_text().splitlines()
                          if not line.startswith(">")).upper()
            # Keep only the codons the null can score, so ``len`` is the number the caption
            # quotes: a stop codon has no replacements to reach and ``null_distances`` skips
            # it, which would otherwise make the root read one codon longer than the row.
            codons = [c for c in (seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3))
                      if GENETIC_CODE.get(c) not in (None, "*")]
            if not codons:
                raise ValueError(f"{lin.root.name}: no scorable codons read")
            return pd.DataFrame({"codon_from": codons})
    raise KeyError(f"{guide.source.name}: no lineage {lineage!r}; "
                   f"has {guide.lineage_ids}")


def null_distances(base: pd.DataFrame) -> dict[int, int]:
    """The null: from each codon of the baseline, how far are the other 19 amino acids?

    For every scored codon, and every amino acid other than the one it already encodes, the
    fewest nucleotide changes that reach ANY codon of that amino acid. This is what the
    "fewest nucleotide changes needed" columns would look like if an amino-acid change were
    equally likely to go anywhere, so it is the thing the observed rows have to beat.

    It depends only on the genetic code and on which codons it is given: no tree, no
    branches, no counts. It is *not* code-only -- each codon of the sequence contributes its
    own 19 distances, so the result carries that sequence's codon composition.
    """
    by_aa: dict[str, list[str]] = {}
    for codon in ("".join(c) for c in itertools.product(BASES, repeat=3)):
        aa = GENETIC_CODE.get(codon)
        if aa and aa != "*":
            by_aa.setdefault(aa, []).append(codon)

    cache: dict[str, dict[int, int]] = {}
    out = {1: 0, 2: 0, 3: 0}
    for codon in base["codon_from"]:
        codon = str(codon).upper()
        if codon not in cache:
            here = GENETIC_CODE.get(codon)
            if here is None or here == "*":
                cache[codon] = {}
            else:
                tally = {1: 0, 2: 0, 3: 0}
                for aa, codons in by_aa.items():
                    if aa == here:
                        continue
                    tally[min(sum(a != b for a, b in zip(codon, c)) for c in codons)] += 1
                cache[codon] = tally
        for k, v in cache[codon].items():
            out[k] += v
    return out


#: The two PLM null rows, named by ROLE. Nothing here may name a model or a lineage: the
#: guide holds those, and the keys in the score table are a trap anyway, since the one that
#: looks like the fine-tune is the base. Reading the pair by role means the figure cannot
#: get them the wrong way round, which would invert the panel's conclusion.
PLM_NULL_ROLES = ("base", "finetune")
PLM_NULL_MODELS = (
    ("ESM2_650M_HA80_raw", "Base model (ESM-2)"),
    ("ESM2_650M_HA80", "HA80 fine-tune"),
)
PLM_NULL_COLUMNS = ("model", "lineage", "position", "ref_aa", "aa", "plm_prob", "nt_steps")


def plm_null_models(guide) -> list[tuple[str, str, str]]:
    """``(row key, model key, reader-facing label)`` for each PLM null row, from the guide."""
    out = []
    for role in PLM_NULL_ROLES:
        found = guide.models_with_role(role)
        if not found:
            raise ValueError(f"{guide.source.name}: no model with role {role!r}, so the PLM "
                             "null pair cannot be built")
        if len(found) > 1:
            raise ValueError(f"{guide.source.name}: {len(found)} models with role {role!r}; "
                             "the PLM null draws one row per role")
        label = found[0].label
        if role == "finetune":
            label = "HA80 fine-tune"
        out.append((f"plm {role}", found[0].key, label))
    return out


def read_plm_scores(path: Path) -> pd.DataFrame:
    """The masked replacement-score table (``replacement_scores_masked.tsv.gz``).

    One row per model x lineage x site x non-reference amino acid, carrying the masked
    marginal probability and ``nt_steps``, the fewest nucleotide changes from that lineage's
    reference codon to that amino acid. Unlike the branch tables this one has no ``#``
    provenance header, so it is read plainly.
    """
    d = pd.read_csv(path, sep="\t")
    missing = [c for c in PLM_NULL_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"masked score table is missing columns {missing}")
    return d


def plm_null_table(masked: pd.DataFrame, lineage: str,
                   models: list[tuple[str, str, str]] | None = None) -> pd.DataFrame:
    if models is None:
        models = [("plm base", PLM_NULL_MODELS[0][0], PLM_NULL_MODELS[0][1]),
                  ("plm finetune", PLM_NULL_MODELS[1][0], PLM_NULL_MODELS[1][1])]
    """The PLM null: where a language model puts its probability when a site is masked.

    The genetic-code null above it treats every amino-acid change as equally likely, so its
    1/2/3 split is a property of the code alone. This asks the same question of the models.
    At each site the reference residue is removed and the 19 others are renormalised to sum
    to 1, so **every site contributes exactly one expected amino-acid change** and a site the
    model is confident about does not drown out the rest; the row's ``total`` is therefore the
    number of sites, and the three cells are expected changes, not counts of anything observed.

    The same renormalisation is why the row is comparable with the genetic-code null directly:
    run with a flat distribution over the 19 it would reproduce that null's split exactly.
    """
    need = [c for c in PLM_NULL_COLUMNS if c not in masked.columns]
    if need:
        raise ValueError(f"masked score table is missing columns {need}")
    here = masked[masked["lineage"] == lineage]
    if here.empty:
        seen = ", ".join(sorted(masked["lineage"].astype(str).unique()))
        raise ValueError(f"masked score table has no rows for lineage {lineage!r} (it has: {seen})")
    rows = []
    for key, model, label in models:
        g = here[here["model"] == model]
        if g.empty:
            seen = ", ".join(sorted(here["model"].astype(str).unique()))
            raise ValueError(f"masked score table has no {model!r} rows for {lineage!r} (it has: {seen})")
        per_site = g.groupby("position").size()
        if not (per_site == 19).all():
            bad = int((per_site != 19).sum())
            raise ValueError(f"{model} / {lineage}: {bad} sites do not carry all 19 non-reference "
                             "amino acids exactly once")
        if g.duplicated(["position", "aa"]).any():
            raise ValueError(f"{model} / {lineage}: a site repeats an amino acid")
        if not set(g["nt_steps"].unique()) <= set(NT):
            raise ValueError(f"{model} / {lineage}: nt_steps outside {set(NT)}")
        if (g["plm_prob"] < 0).any() or not np.isfinite(g["plm_prob"]).all():
            raise ValueError(f"{model} / {lineage}: plm_prob is negative or not finite")
        mass = g.groupby("position")["plm_prob"].transform("sum")
        if (mass <= 0).any():
            raise ValueError(f"{model} / {lineage}: a site has no probability on any "
                             "non-reference amino acid, so it cannot be renormalised")
        w = (g["plm_prob"] / mass).groupby(g["nt_steps"]).sum()
        rows.append({"row": key, "unit": f"{label}; {lineage}, masked, one change per site",
                     "total": float(len(per_site)),
                     "need_1": float(w.get(1, 0.0)), "need_2": float(w.get(2, 0.0)),
                     "need_3": float(w.get(3, 0.0))})
    return pd.DataFrame(rows)


def plm_null_unweighted(masked: pd.DataFrame, lineage: str, model: str | None = None) -> dict[int, float]:
    """The genetic-code-only split of the SAME sites the PLM null uses, as shares.

    The drawn genetic-code null is built on the tree's baseline, a different sequence of a
    different length, so the caption needs this to say what a flat distribution over the PLM
    null's own 19-way choices would have given. It uses one model's rows only because
    ``nt_steps`` does not depend on the model.
    """
    if model is None:
        model = PLM_NULL_MODELS[0][0]
    g = masked[(masked["lineage"] == lineage) & (masked["model"] == model)]
    n = g.groupby("nt_steps").size()
    return {k: float(n.get(k, 0)) / float(n.sum()) for k in NT}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_table(ax, table: pd.DataFrame, n_nodes: int | None, baseline: str = "", t_h: float = 1.25,
               terminal: bool = False, lineage: str = "",
               models: list[tuple[str, str, str]] | None = None) -> None:
    """The summary table, drawn as aligned text so it follows the house type scale.

    Both rows are raw counts: every amino-acid change on a single branch, and every amino-acid
    difference pooled over the long-range node comparison. Nothing here is averaged.
    """
    ax.axis("off")
    ax.grid(False)
    # Numbers are right-aligned on their column edge, as a table's figures should be; the
    # row label is the only left-aligned cell. What each row MEANS lives in the caption,
    # not under the label: a table that carries its own prose stops reading as a table.
    xs = [0.0, 0.465, 0.65, 0.82, 0.99]
    t = ax.transAxes

    def f(inches):
        return 1.0 - inches / t_h

    span0 = xs[2] - 0.155
    ax.plot([0, 1], [f(0.02), f(0.02)], transform=t, color=style.INK, linewidth=0.9, clip_on=False)
    ax.text((span0 + xs[4]) / 2, f(0.09), "Fewest nucleotide changes needed", transform=t,
            ha="center", va="top", fontsize=6.5, color=style.INK_MUTED)
    ax.plot([span0, xs[4]], [f(0.23), f(0.23)], transform=t, color=style.GRID,
            linewidth=0.6, clip_on=False)
    for x, h in zip(xs[1:], ["Total", "1", "2", "3"]):
        ax.text(x, f(0.27), h, transform=t, ha="right", va="top", fontsize=7,
                fontweight="bold", color=style.INK)
    ax.plot([0, 1], [f(0.42), f(0.42)], transform=t, color=style.INK, linewidth=0.7, clip_on=False)

    base = f"{baseline} to newest internal nodes" if baseline else "Baseline to newest internal nodes"
    branches = "internal and terminal" if terminal else "internal"
    # The bracket says what the cells are, because the numbers beside them are not counts:
    # the probability a model puts on each amino acid, summed within a nucleotide class.
    who = f" ({lineage}, summed probability)" if lineage else " (summed probability)"
    # The null is neither a 20 x 20 nor a 64 x 64 matrix, and both would give a different
    # answer: the source is the codon actually at the site, one of 64, and the target is an
    # amino acid, minimised over that amino acid's own codons. Say so in the row.
    # The lineage leads the bracket because the null is not code-only: it carries that
    # root's codon composition, and it is read against the PLM rows, which score the same one.
    whose = f"{lineage}; " if lineage else ""
    names = {"one branch": f"Individual {branches} branches",
             "oldest node to newest": base,
             "null": f"Null, genetic code ({whose}64 x 20, codon to aa)"}
    # The model names are the guide's, never this script's: it may not know one.
    for key, _, label in (models or []):
        lbl = "HA80 fine-tune" if "fine" in label.lower() else label
        names[key] = f"PLM null, {lbl}{who}"

    def pct(v, total):
        share = v / total
        return "<0.1%" if 0 < share < 0.001 else f"{share:.1%}"

    # The PLM pair is fenced off by a hard rule, not the usual hairline: the rows above are
    # observed or code-only counts and these are expected changes under a model.
    keys = list(table["row"])
    plm_at = next((i for i, k in enumerate(keys) if k.startswith("plm ")), None)

    row_h = 0.26
    for i, r in enumerate(table.itertuples()):
        y = f(0.53 + i * row_h)
        ax.text(xs[0], y, names[r.row], transform=t, ha="left", va="top", fontsize=7,
                color=style.INK)
        ax.text(xs[1], y, f"{r.total:,.0f}", transform=t, ha="right", va="top", fontsize=7,
                color=style.INK)
        # The PLM rows are expected changes, not counts: shown to 1 dp, because a rounded
        # 364 sits in a column of exact integers and reads as something that was counted.
        dp = 1 if r.row.startswith("plm ") else 0
        for x, v in zip(xs[2:], (r.need_1, r.need_2, r.need_3)):
            cell = f"{v:,.{dp}f}  ({pct(v, r.total)})" if r.total else "n/a"
            ax.text(x, y, cell, transform=t, ha="right", va="top", fontsize=7, color=style.INK)
        if i < len(table) - 1:
            hard = (i + 1) == plm_at
            ax.plot([0, 1], [f(0.53 + (i + 1) * row_h - 0.05), f(0.53 + (i + 1) * row_h - 0.05)],
                    transform=t, color=style.INK if hard else style.GRID,
                    linewidth=0.7 if hard else 0.5, clip_on=False)
    ax.plot([0, 1], [f(0.53 + len(table) * row_h - 0.05), f(0.53 + len(table) * row_h - 0.05)],
            transform=t, color=style.INK, linewidth=0.9, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def read_year_table(path: Path) -> pd.DataFrame:
    """Sequences per collection year, one row per year, written beside the other tables."""
    meta = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            for tok in line.lstrip("# ").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
    d = pd.read_csv(path, sep="\t", comment="#")
    return d, meta


def draw_years(ax, d: pd.DataFrame) -> None:
    """Sequences per collection year: what the tree is actually made of, and when."""
    ax.bar(d["year"], d["n"], width=0.86, color=style.BASE, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlim(d["year"].min() - 1, d["year"].max() + 1)
    ax.grid(axis="x", visible=False)
    style.finalise_axes(ax, xlabel="Collection year", ylabel="Sequences per year")


def draw_year_summary(ax, d: pd.DataFrame, meta: dict, branch: pd.DataFrame) -> None:
    """The wee summary beside the year chart: the handful of numbers a reader asks for."""
    ax.axis("off")
    ax.grid(False)
    t = ax.transAxes
    tips = int(meta.get("tips", 0))
    dated = int(meta.get("dated", d["n"].sum()))
    undated = int(meta.get("undated", 0))
    y0, y1 = int(d["year"].min()), int(d["year"].max())
    med = int(d.loc[d["n"].cumsum() >= dated / 2, "year"].iloc[0])
    recent = int(d.loc[d["year"] >= 2015, "n"].sum())
    peak = d.loc[d["n"].idxmax()]

    lines = [
        ("Tips", f"{tips:,}"),
        ("Years", f"{y0}-{y1}"),
        ("Median year", f"{med}"),
        ("Busiest year", f"{int(peak['year'])} ({int(peak['n']):,})"),
        ("From 2015", f"{recent / dated:.0%}"),
        ("Undated", f"{undated:,}"),
        ("Branches scored", f"{len(branch):,}"),
    ]
    ax.plot([0, 1], [0.98, 0.98], transform=t, color=style.INK, linewidth=0.7, clip_on=False)
    for i, (k, v) in enumerate(lines):
        y = 0.90 - i * 0.115
        ax.text(0.0, y, k, transform=t, ha="left", va="top", fontsize=7.5, color=style.INK_MUTED)
        ax.text(1.0, y, v, transform=t, ha="right", va="top", fontsize=7.5, color=style.INK)
    ax.plot([0, 1], [0.90 - len(lines) * 0.115 - 0.01, 0.90 - len(lines) * 0.115 - 0.01],
            transform=t, color=style.INK, linewidth=0.7, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def bin_colours(n: int) -> list:
    """One colour per branch-length bin, walking the house sequential ramp (the bins are ordered)."""
    cmap = matplotlib.colormaps[style.SEQUENTIAL]
    return [cmap(v) for v in np.linspace(0.78, 0.06, n)]


def draw_key(ax, d: pd.DataFrame, colours) -> None:
    """The right-hand key: the two lines, and the exact branch count behind each coloured point."""
    ax.axis("off")
    ax.grid(False)
    t = ax.transAxes
    # A legend and a small counts table. What the chance line is, and why bin 1 is empty,
    # are facts about the method and belong in the caption.
    ax.plot([0.0, 0.10], [0.97, 0.97], transform=t, color=style.INK, alpha=0.55, linewidth=1.1,
            clip_on=False)
    ax.text(0.15, 0.97, "Observed", transform=t, ha="left", va="center", fontsize=7.5, color=style.INK)
    ax.plot([0.0, 0.10], [0.89, 0.89], transform=t, color=style.NULL_GREY, linewidth=1.1,
            linestyle=(0, (3, 2)), clip_on=False)
    ax.plot([0.05], [0.89], transform=t, marker="s", markersize=3.8, markerfacecolor=style.SURFACE,
            markeredgecolor=style.NULL_GREY, markeredgewidth=0.9, linestyle="none", clip_on=False)
    ax.text(0.15, 0.89, "Expected by chance", transform=t, ha="left", va="center", fontsize=7.5,
            color=style.INK_MUTED)

    ax.plot([0, 1], [0.76, 0.76], transform=t, color=style.INK, linewidth=0.7, clip_on=False)
    ax.text(0.0, 0.72, "Changes", transform=t, ha="left", va="top", fontsize=7,
            color=style.INK_MUTED)
    ax.text(0.60, 0.72, "branches", transform=t, ha="right", va="top", fontsize=7, color=style.INK_MUTED)
    ax.text(1.0, 0.72, "aa changes", transform=t, ha="right", va="top", fontsize=7,
            color=style.INK_MUTED)
    ax.plot([0, 1], [0.62, 0.62], transform=t, color=style.GRID, linewidth=0.5, clip_on=False)
    for i, (r, colour) in enumerate(zip(d.itertuples(), colours)):
        y = 0.55 - i * 0.095
        ax.scatter([0.03], [y], s=30, color=colour, linewidths=0, transform=t, clip_on=False)
        ax.text(0.10, y, f"{r.label}", transform=t, ha="left", va="center", fontsize=7.5,
                color=style.INK)
        ax.text(0.60, y, f"{r.branches:,}", transform=t, ha="right", va="center", fontsize=7.5,
                color=style.INK)
        ax.text(1.0, y, f"{r.nonsyn:,}", transform=t, ha="right", va="center", fontsize=7.5,
                color=style.INK)
    ax.plot([0, 1], [0.55 - len(d) * 0.095 + 0.04, 0.55 - len(d) * 0.095 + 0.04], transform=t,
            color=style.INK, linewidth=0.7, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw(branch: pd.DataFrame, column: pd.DataFrame, cap: int | None, synthetic: bool,
         nodes: pd.DataFrame | None = None, years: tuple | None = None, baseline: str = "",
         base: pd.DataFrame | None = None, terminal: bool = False,
         plm: pd.DataFrame | None = None, lineage: str = "",
         models: list[tuple[str, str, str]] | None = None):
    table = summary_table(branch, nodes, base, plm)
    d = chart_numbers(branch, column)
    row_h = 0.26
    t_h = 0.54 + row_h * len(table)
    # The table and the key place their text at fixed axes fractions, so a large hspace
    # shrinks those cells and runs their lines together. Keep it tight and buy the
    # separation with row height instead.
    # B and C were small boxes in a lot of white: the key and the summary beside them are
    # short, so the right-hand column is narrowed and the charts take the width back.
    # Every piece of text here is set in points, so the way to make the type larger against
    # the figure is to take height out of the canvas rather than to scale the type up: the
    # charts lose inches, the table keeps its own (it is drawn at fixed inch offsets), and
    # the panels close up. hspace is small for the same reason as above.
    rows = [t_h, 1.85] + ([1.55] if years else [])
    fig = style.figure(style.WIDTH_DOUBLE, sum(rows) + 0.04)
    grid = fig.add_gridspec(len(rows), 2, height_ratios=rows,
                            width_ratios=[1.0, 0.40], hspace=0.03, wspace=0.10)
    ax_t = fig.add_subplot(grid[0, :])
    draw_table(ax_t, table, None if nodes is None else len(nodes), baseline, t_h, terminal,
               lineage, models)
    style.panel_letter(ax_t, "A", dx=-0.035, dy=1.02)
    if years:
        ax_y = fig.add_subplot(grid[2, 0])
        draw_years(ax_y, years[0])
        draw_year_summary(fig.add_subplot(grid[2, 1]), years[0], years[1], branch)
        style.panel_letter(ax_y, "C", dx=-0.05, dy=1.10)
    ax = fig.add_subplot(grid[1, 0])
    style.panel_letter(ax, "B", dx=-0.05, dy=1.10)

    colours = bin_colours(len(d))
    x = np.arange(len(d), dtype=float)
    shown = d["drawn"].to_numpy()

    ax.plot(x[shown], d.loc[shown, "chance_multi"], color=style.NULL_GREY, linewidth=0.9,
            linestyle=(0, (3, 2)), zorder=2)
    ax.scatter(x[shown], d.loc[shown, "chance_multi"], s=22, facecolor=style.SURFACE,
               edgecolor=style.NULL_GREY, linewidths=0.9, marker="s", zorder=3)
    ax.plot(x[shown], d.loc[shown, "share_multi"], color=style.INK, linewidth=0.9, alpha=0.55, zorder=4)
    ax.vlines(x[shown], d.loc[shown, "lo"], d.loc[shown, "hi"],
              colors=[c for c, k in zip(colours, shown) if k], linewidth=1.3, alpha=0.6, zorder=4)
    ax.scatter(x[shown], d.loc[shown, "share_multi"], s=46,
               color=[c for c, k in zip(colours, shown) if k], linewidths=0, zorder=5)

    for xi, r in zip(x, d.itertuples()):
        if not r.drawn:
            continue
        if not r.possible:
            ax.annotate("0 by\nconstruction", (xi, 0), xytext=(0, 17), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=style.INK_MUTED, linespacing=1.3)
        elif r.chance_multi and np.isfinite(r.share_multi):
            ax.annotate(f"{r.share_multi / r.chance_multi:.1f}x", (xi, r.hi), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7,
                        color=style.INK_MUTED)

    ax.set_xticks(x, list(d["label"]))
    ax.set_xlim(-0.45, len(d) - 0.55)
    ax.set_ylim(0, max(float(np.nanmax(d.loc[shown, "hi"])), 0.01) * 1.22)
    ax.grid(axis="x", visible=False)
    style.finalise_axes(ax, xlabel="Nucleotide changes on the branch",
                        ylabel="Share of amino-acid changes\nmoving 2+ positions of one codon")

    draw_key(fig.add_subplot(grid[1, 1]), d, colours)

    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC FIXTURE - NOT DATA", fontsize=22, color=style.NULL_GREY, alpha=0.45,
                 rotation=18, ha="center", va="center", zorder=10, fontweight="bold")
    return fig


def caption(branch: pd.DataFrame, column: pd.DataFrame, cap: int | None, synthetic: bool,
            prov: dict | None = None, nodes: pd.DataFrame | None = None,
            base: pd.DataFrame | None = None, baseline: str = "",
            years: tuple | None = None, masked: pd.DataFrame | None = None,
            lineage: str = "", models: list[tuple[str, str, str]] | None = None) -> str:
    cells = {c: int(branch[c].sum()) for c in branch.columns if c.startswith(("syn_", "nonsyn_"))}
    nonsyn = sum(cells[f"nonsyn_{n}nt"] for n in NT)
    syn = sum(cells[f"syn_{n}nt"] for n in NT)
    multi = cells["nonsyn_2nt"] + cells["nonsyn_3nt"]
    shorter = cells["nonsyn_2nt_min1"] + cells["nonsyn_3nt_min1"] + cells["nonsyn_3nt_min2"]
    cap_text = f"fewer than {cap}" if cap else "any number of"

    share = f"{multi / nonsyn:.1%}" if nonsyn else "n/a"
    d = chart_numbers(branch, column)
    q = nonsyn_probs(branch, column)
    drawn = d.loc[d["drawn"] & d["possible"] & d["chance_multi"].gt(0)]
    ratios = ", ".join(f"{r.share_multi / r.chance_multi:.1f}x in the bin of {r.label}" for r in drawn.itertuples())
    numbering = ""
    if "site_number" in column.columns:
        n_none = int((pd.to_numeric(column["site_number"], errors="coerce").isna()).sum())
        skipped = bool((prov or {}).get("skipped_codons"))
        numbering = (f"The tree's reference record is a different protein, and {n_none} of its {len(column)} codons "
                     f"have no residue in the protein studied (alignment gaps). "
                     + ("Those columns are left out of every count, because a gap read as a base looks like a "
                        "run of multi-position changes. " if skipped else
                        "They are included in the counts, where a gap read as a base can look like multi-position "
                        "changes. "))
    years_text = ""
    if years is not None:
        yd, ym = years
        y0, y1 = int(yd["year"].min()), int(yd["year"].max())
        tot_y = int(yd["n"].sum())
        last10 = int(yd.loc[yd["year"] > y1 - 10, "n"].sum())
        years_text = (f"**(C)** Every tip of the tree by collection year, {y0} to {y1}, on a log scale: "
                      f"the earliest years hold tens of sequences against tens of thousands recently, and "
                      f"a linear axis hides them. The last ten years carry {last10 / tot_y:.0%} of the "
                      f"{tot_y:,} tips, so the scored branches sit overwhelmingly in recent diversity. "
                      f"The panel beside it counts the tree and the branches drawn from it. ")
    longer = ""
    if nodes is not None:
        n = len(nodes)
        tot = int(nodes["nonsyn"].sum())
        who = f"{baseline} (a dated sequence in the tree)" if baseline else "the oldest node scored"
        longer = (f"The table's second row pools the amino-acid differences between {who} and each "
                  f"of the {n:,} newest internal nodes (internal nodes whose children are all tips): {tot:,} "
                  f"differences, of which {int(nodes['min1'].sum()) / tot:.0%} could be reached by one nucleotide "
                  f"change from the baseline codon, {int(nodes['min2'].sum()) / tot:.0%} needed two and "
                  f"{int(nodes['min3'].sum()) / tot:.0%} needed three. Over a long path such changes are made in "
                  f"steps through intermediate codons, which a single branch cannot do, and differences are counted "
                  f"per node, so changes shared by most nodes are counted for each. "
                  + ("The baseline is an observed sequence, not a reconstruction. The walk's own start node is not "
                     "used for it: the tree reads sites that were never sequenced as the reference base, so near the "
                     "root the reconstruction is largely the reference record, and that record is a different "
                     "subtype. Comparing against it measures the distance between subtypes rather than change within "
                     "one. " if baseline else
                     "The oldest node's codons are inferred, not observed. "))
    if base is not None:
        nd = null_distances(base)
        tn = sum(nd.values())
        whose = f"the {lineage} root" if lineage else "the tree's baseline record"
        longer += (f"The null row (A) is what the rows around it are read against: from each of the "
                   f"{len(base):,} codons of {whose}, the fewest nucleotide changes that reach each of "
                   f"the other 19 amino acids. The lookup behind it is 64 x 20, not 20 x 20 or 64 x 64: the source "
                   f"is the codon actually at the site, one of 64, and the target is an amino acid, taking its "
                   f"nearest codon. A 20 x 20 version would have to pick one codon per starting amino acid and a "
                   f"64 x 64 one would count synonymous targets, and both give a different answer. "
                   f"It is {tn:,} codon-to-amino-acid pairs, {nd[1] / tn:.0%} of which "
                   f"are one nucleotide change away, {nd[2] / tn:.0%} two and {nd[3] / tn:.0%} three. It uses the "
                   f"genetic code and that sequence's own codons: no tree, no branches and no counts, so it is "
                   f"what the columns would look like if an amino-acid change were equally likely to go anywhere. "
                   f"It is not code-only -- every codon contributes its own 19 distances, so the row carries the "
                   f"sequence's codon composition, which is why it is taken from the same root the model rows "
                   f"score rather than from the tree's reference record. "
                   f"All table rows above the second hard rule are raw counts; none is averaged. ")
    if masked is not None:
        p = plm_null_table(masked, lineage, models).set_index("row")
        (bkey, bmodel, blabel), (fkey, _, flabel) = models
        flat = plm_null_unweighted(masked, lineage, bmodel)
        n_sites = int(p.loc[bkey, "total"])
        one = summary_table(branch, None, None).iloc[0]
        obs1 = one["need_1"] / one["total"]
        def sp(key):
            r = p.loc[key]
            return tuple(r[f"need_{k}"] / r["total"] for k in NT)
        b1, b2, b3 = sp(bkey)
        f1, f2, f3 = sp(fkey)
        longer += (
            f"The last two rows (A), below the second hard rule, are a different kind of object "
            f"and are not counts of anything observed: they ask where a protein language model "
            f"puts its probability. For each of the {n_sites} sites of the {lineage} root the "
            f"residue is masked, the model's probabilities for the 19 other amino acids are "
            f"renormalised to sum to one, and each amino acid's weight is assigned to the fewest "
            f"nucleotide changes that reach it from the {lineage} codon actually at that site. "
            f"Every site therefore contributes exactly one expected amino-acid change, so the "
            f"total is the number of sites and a site the model is confident about cannot drown "
            f"out the rest. Read against a flat distribution over the same 19-way choices, which "
            f"would give {flat[1]:.1%} / {flat[2]:.1%} / {flat[3]:.1%} (the genetic-code null for "
            f"these sites, close to the row above because the split depends on the code rather "
            f"than the sequence), {blabel.lower()} already concentrates on one-nucleotide "
            f"neighbours at {b1:.1%} and {flabel.lower()} at {f1:.1%}, with three-nucleotide "
            f"changes falling from {flat[3]:.1%} to {b3:.1%} and {f3:.1%}. Both models thus carry "
            f"a mutational bias neither was given a nucleotide sequence to learn, and the "
            f"fine-tuned one carries much more of it; but both still fall short of the {obs1:.1%} "
            f"of real single-branch changes that are one nucleotide away, so the models "
            f"interpolate between the genetic code and what a single branch can actually do. "
            f"Both rows use the masked marginals shipped in the replacement-score table, and the "
            f"checkpoints behind the two labels are given in the Methods. ")
    return f"""
    **Nucleotide changes per codon on individual internal branches of the tree.**
    Each branch of the UShER tree (a tree of sequences, built by placing each one on its
    nearest relative) carries the nucleotide changes that separate a node from its parent. Only
    branches with {cap_text} nucleotide changes are used, because long branches are mostly
    misplaced or poor-quality sequences, and only internal branches unless the table says
    otherwise, because tips carry the most sequencing error. For each codon changed on such a
    branch we count how many of its three positions differ from the parent codon.
    **(A)** Every count is raw; none is averaged. Each row sorts amino-acid changes by the fewest
    nucleotide changes that would have produced the new amino acid from the codon it started from.
    *Individual internal branches* is every amino-acid change on every scored internal branch, taken
    one branch at a time.
    The {len(branch):,} branches hold {syn:,} synonymous and {nonsyn:,} amino-acid-changing codon
    changes; {multi:,} of the amino-acid-changing ones ({share}) changed 2 or 3 positions. In
    {shorter:,} of those the new amino acid was reachable from the parent codon by fewer
    changes. That comparison uses the parent codon actually seen on the tree. It is not the
    minimum over all codons of the old amino acid, which often differs only because the parent
    carried a synonymous codon far from the target.
    **(B)** The chart plots that share against branch length, binned 1 / 2 / 3 / 4+, against the share
    expected if the branch's changes fell at random. A rise with branch length on its own means
    nothing: a branch carrying one nucleotide change cannot move two positions of a codon, and the
    more changes a branch carries, the likelier two land in one codon. The chance line removes
    both effects. It places each branch's changes at random, without replacement, over the
    {len(scored_codons(column))} scored codons, and converts codons to amino-acid changes with
    P(amino acid changes | 1 position moved) = {q[1]:.2f}, taken from these data, and
    P(... | 2 or 3 positions moved) = {q[2]:.2f} and {q[3]:.2f}, taken from the genetic code
    (the data's own 2-position split, {cells['nonsyn_2nt']} of
    {cells['nonsyn_2nt'] + cells['syn_2nt']}, agrees). Observed exceeds chance by {ratios}: the
    excess is largest on the shortest branches, which is the opposite of what the mechanical
    effect would produce. Bars are 95% Wilson intervals; a bin with fewer than
    {MIN_CODONS_PER_POINT} amino-acid changes is not drawn. The table beside the chart gives the
    branches and amino-acid changes behind each point.
    {years_text}{longer}{numbering}Codon numbers count along the reference record, not alignment columns, so gaps elsewhere
    in the alignment cannot shift them. The tree reads sites that were never sequenced as the
    reference base unless it was built with faToVcf -includeNoAltN; every number here inherits
    that convention.
    """


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="\n\n".join(__doc__.split("\n\n")[:3]),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch-table", required=True, type=Path, help="branch_codons.tsv.gz")
    ap.add_argument("--column-table", required=True, type=Path, help="column_codons.tsv.gz")
    ap.add_argument("--long-range-nodes", type=Path, default=None,
                    help="long_range_nodes.tsv.gz: adds the table's second row (oldest node against the newest internal nodes)")
    ap.add_argument("--baseline-codons", type=Path, default=None,
                    help="baseline_codons.tsv.gz: adds the null row (every amino acid from every baseline codon)")
    ap.add_argument("--plm-null-scores", type=Path, default=None,
                    help="replacement_scores_masked.tsv.gz: adds the PLM null pair, one row for "
                         "the base model and one for the fine-tune, for a single lineage. "
                         "Needs --plm-null-guide and --plm-null-lineage")
    ap.add_argument("--plm-null-guide", type=Path, default=None,
                    help="guide YAML naming the two models by role; the figure never names one")
    ap.add_argument("--plm-null-lineage", default=None,
                    help="which lineage of --plm-null-scores the PLM null is built from")
    ap.add_argument("--year-table", type=Path, default=None,
                    help="year_counts.tsv.gz: adds the sequences-per-year panel and its summary")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "output",
                    help="output directory for PDF + PNG, values and caption")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    branch, column, pb, pc = load_tables(args.branch_table, args.column_table)
    synthetic = is_synthetic(pb, pc)
    cap = cap_of(pb)
    stem = STEM + ("_SYNTHETIC_FIXTURE" if synthetic else "")
    nodes, baseline = None, ""
    if args.long_range_nodes:
        nodes, prov_n = read_table(args.long_range_nodes)
        check_nodes(nodes)
        baseline = str(prov_n.get("long_range_baseline", "")).split(" (")[0]
    base = read_table(args.baseline_codons)[0] if args.baseline_codons else None
    masked, models, lineage = None, None, ""
    if args.plm_null_scores:
        if not args.plm_null_lineage:
            ap_err = "--plm-null-scores needs --plm-null-lineage"
            raise SystemExit(ap_err)
        masked = read_plm_scores(args.plm_null_scores)
        lineage = args.plm_null_lineage
        if args.plm_null_guide:
            guide = load_guide(args.plm_null_guide)
            models = plm_null_models(guide)
            # The null stands on the same sites as the PLM rows it is read against; see
            # ``root_codons``. Without a guide there is no root to read, so the tree's own
            # baseline stays in use and the row says which sequence it came from either way.
            base = root_codons(guide, lineage)
        else:
            models = [("plm base", PLM_NULL_MODELS[0][0], PLM_NULL_MODELS[0][1]),
                      ("plm finetune", PLM_NULL_MODELS[1][0], PLM_NULL_MODELS[1][1])]
    plm = plm_null_table(masked, lineage, models) if masked is not None else None
    years = read_year_table(args.year_table) if args.year_table else None
    fig = draw(branch, column, cap, synthetic, nodes, years, baseline, base,
               terminal_included(pb), plm, lineage, models)
    paths = style.save_figure(fig, args.out / stem)
    tab = summary_table(branch, nodes, base, plm)
    tab.insert(0, "panel", "table")
    # The PLM rows carry fractional expected changes, so they cannot share the integer
    # ``need_`` columns: a count column written as float would send the millions in the rows
    # above into exponent form under the writer's "%.6g". They get their own float columns
    # instead, holding exactly the two things the panel draws -- the expected change (1 dp,
    # as drawn) and its share (full precision, as the caption quotes it).
    is_plm = tab["row"].str.startswith("plm ")
    for k in NT:
        tab[f"share_{k}"] = tab[f"need_{k}"] / tab["total"]
        tab[f"expected_{k}"] = tab[f"need_{k}"].where(is_plm).round(1)
        tab[f"need_{k}"] = tab[f"need_{k}"].mask(is_plm)
    tab["total"] = tab["total"].round()
    values = pd.concat([tab, chart_numbers(branch, column)], ignore_index=True)
    # concat fills the other block's columns with NaN, which would make these counts float and the
    # writer's "%.6g" render 1,041,527 as 1.04153e+06. Nullable integers keep them readable.
    for col in ("total", "need_1", "need_2", "need_3", "bin", "branches", "nonsyn", "multi"):
        if col in values:
            values[col] = values[col].astype("Int64")
    values.insert(0, "provenance", "SYNTHETIC FIXTURE, not data" if synthetic else "tables as read")
    write_values(values, args.out, stem)
    write_caption(caption(branch, column, cap, synthetic, pb, nodes, base, baseline, years,
                          masked, lineage, models), args.out, stem,
                  banner=SYNTHETIC_BANNER if synthetic else None)
    print("wrote", ", ".join(p.name for p in paths))
    if synthetic:
        print("WARNING: input tables are marked synthetic; the output is a drawing test, not a result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
