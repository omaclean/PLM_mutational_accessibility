"""How many times each observed amino-acid replacement arose, on a UShER tree.

The lineage panel (Table SZ) says how many distinct amino-acid replacements each lineage
carries. It does not say whether a replacement happened once and spread, or happened
several times independently. This module answers that from the tree, and the distinction
matters for the manuscript: a replacement that arose repeatedly is evidence about what
mutation and selection can reach, whereas a replacement counted once in a large clade may
be one ancient event that every descendant inherited.

Definitions
-----------
Everything is counted inside one lineage's subtree, against that lineage's own root
protein, so the unit matches ``replacement_observed.tsv.gz`` exactly.

*Replacement*
    a (position, amino acid) pair where the amino acid differs from the lineage root's
    residue at that position. Position is 1-based along the root's own 566-residue
    protein, the frame every shipped lineage table uses.

*Origin*
    a branch of the subtree whose parent codon at that position encodes something other
    than the replacement's amino acid and whose child codon encodes it. Synonymous
    changes are not origins; neither is a codon change into or out of a stop, nor one
    through a codon holding a base other than A, C, G or T. Two origins of one
    replacement therefore always sit on two different branches, and neither branch is on
    the path from the root to the other with the state unbroken -- if it were, the state
    would have had to revert in between, which is itself a second event.

*Homoplasy*
    a replacement with two or more origins. The count of these, and their share of the
    replacements the tree can place, are the two numbers Table SZ reports.

Two processes make a replacement recurrent, and they are told apart by ``n_routes``, the
number of distinct parent-codon-to-child-codon changes across a replacement's origins:

``n_routes == 1``
    the same nucleotide change, in different parts of the tree. Recurrence here rests on
    the topology: the tree says these are separate branches.
``n_routes >= 2``
    the same amino acid reached by different nucleotide changes. This needs no topology
    to believe -- two different codon changes cannot be one event -- so it is the
    conservative evidence, and it is a strict subset of the above by construction.

The two are a partition of the homoplasies, not two overlapping sets, and the reported
count combines them.

What limits the answer
----------------------
Two coverage gaps are counted rather than hidden, because both shrink the denominator and
neither is visible in the output table otherwise.

* **Columns.** UShER positions are counted along its reference record, which for the
  segment 4 tree is ``DQ864721`` -- not an H3 sequence. 535 of the root protein's 566
  positions have a reference codon; positions 1-26 (the signal peptide, and the first ten
  residues of mature HA1) and 562-566 do not, and no replacement there can be placed.
  ``numbering_from_column_table`` builds that map from the shipped S11 column table, whose
  own numbering was checked against the conserved residues Y98, W153, H183 and Y195.
* **Sequences.** A lineage's sub-alignment and the tree are not the same sequence set. A
  replacement observed in a sequence the tree does not hold has no origin to find, so it
  is reported as unplaced rather than as a single origin.

Nothing here reads a file or runs a program; ``tools/export_homoplasy.py`` does the input
and output. The walk itself is ``mutacc.branch_codons.walk_branches``, used with no
branch-length ceiling and with terminal branches included: a replacement seen in one
sequence lives on a terminal branch, so excluding those would drop most of the
replacements the panel counts. ``origins_frame`` keeps the internal-branch counts beside
the full ones so that sensitivity can be reported.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from mutacc.branch_codons import CodonEvent, WalkResult

__all__ = [
    "SIGNAL_PEPTIDE", "Origin", "numbering_from_column_table", "collect_origins",
    "origins_frame", "summarise", "SUMMARY_COLUMNS",
    "p_origin_visible", "p_two_or_more", "rarefied_counts",
    "rarefy_frame", "RAREFIED_COLUMNS", "placed_frames",
]

#: Residues of the signal peptide, which the mature-protein numbering counts from 1 after.
#: A mature site number ``s`` is position ``s + SIGNAL_PEPTIDE`` of the root protein.
SIGNAL_PEPTIDE = 16


@dataclass(frozen=True)
class Origin:
    """One branch on which one replacement arose."""
    position: int          # 1-based along the lineage root protein
    aa: str                # the amino acid gained
    branch: str            # child node of the branch
    codon_from: str
    codon_to: str
    terminal: bool

    @property
    def route(self) -> tuple[str, str]:
        return self.codon_from, self.codon_to


def numbering_from_column_table(column: pd.DataFrame) -> dict[int, int]:
    """``{reference codon index: root-protein position}`` from an S11 column table.

    The table's ``site_number`` is a mature-protein residue number, counted over the
    columns at which the H3 consensus has a residue; it is empty where the reference
    carries a residue H3 lacks, and those codons are left out of the map. Positions are
    shifted by the signal peptide, which the mature numbering does not count.
    """
    for needed in ("codon", "site_number"):
        if needed not in column.columns:
            raise ValueError(f"the column table has no {needed!r} column")
    numbered = column.dropna(subset=["site_number"])
    mapping = {int(c): int(s) + SIGNAL_PEPTIDE
               for c, s in zip(numbered["codon"], numbered["site_number"])}
    if len(mapping) != len(numbered):
        raise ValueError("the column table repeats a codon index")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("two reference codons map to one root-protein position")
    return mapping


def collect_origins(events: Iterable[CodonEvent], numbering: Mapping[int, int],
                    root_protein: str) -> list[Origin]:
    """Keep the codon events that gained an amino acid the lineage root does not carry.

    An event is kept when it changed the amino acid (``category == "nonsyn"``, which
    already excludes stops and ambiguous codons), its codon has a root-protein position,
    and the amino acid gained differs from the root's residue there. The last test is what
    makes the unit a *replacement*: a branch reverting to the root residue is a change, but
    it is not one of the replacements the panel counts.
    """
    out: list[Origin] = []
    for event in events:
        if event.category != "nonsyn":
            continue
        position = numbering.get(event.codon)
        if position is None:
            continue
        if not 1 <= position <= len(root_protein):
            raise ValueError(f"position {position} is outside the "
                             f"{len(root_protein)}-residue root protein")
        if event.aa_to == root_protein[position - 1]:
            continue
        out.append(Origin(position, event.aa_to, event.branch,
                          event.codon_from, event.codon_to, event.terminal))
    return out


def origins_frame(result: WalkResult, numbering: Mapping[int, int],
                  root_protein: str,
                  n_tip: Mapping[str, int] | None = None) -> pd.DataFrame:
    """One row per replacement the tree places, with how many times it arose.

    Columns: ``position``, ``aa``, ``n_origins``, ``n_routes``, ``n_internal_origins``
    (origins on branches that do not end in a tip) and ``routes`` (the distinct codon
    changes, as ``GAT>AAT`` strings, so a reader can check a route by eye).

    Given ``n_tip`` -- the lineage tips at or below every node, the same mapping the walk
    was driven with -- a ``supports`` column is added, holding each origin's descendant
    tip count in descending order as a comma-separated string. That is what
    :func:`rarefied_counts` needs, and it is the only input it needs: the tree is not
    walked again.
    """
    origins = collect_origins(result.events, numbering, root_protein)
    grouped: dict[tuple[int, str], list[Origin]] = defaultdict(list)
    for origin in origins:
        grouped[(origin.position, origin.aa)].append(origin)

    rows = []
    for (position, aa), group in sorted(grouped.items()):
        routes = sorted({o.route for o in group})
        row = {
            "position": position,
            "aa": aa,
            "n_origins": len(group),
            "n_routes": len(routes),
            "n_internal_origins": sum(not o.terminal for o in group),
            "routes": ",".join(f"{a}>{b}" for a, b in routes),
        }
        if n_tip is not None:
            row["supports"] = ",".join(
                str(n) for n in sorted((n_tip[o.branch] for o in group), reverse=True))
        rows.append(row)
    columns = ["position", "aa", "n_origins", "n_routes", "n_internal_origins", "routes"]
    if n_tip is not None:
        columns.append("supports")
    return pd.DataFrame(rows, columns=columns)


def p_origin_visible(support: int, n_total: int, n_target: int) -> float:
    """Chance an origin is still seen when ``n_target`` of ``n_total`` tips are sampled.

    An origin is visible in a subsample exactly when at least one of its ``support``
    descendant tips is drawn, so the complement is the hypergeometric probability of
    drawing none of them::

        P(none) = C(N-s, n) / C(N, n) = prod_{j=0}^{s-1} (N-n-j) / (N-j)

    The product form is used because it has ``support`` terms rather than ``n_target``,
    and most origins have very few descendants. It is exact in floating point for the
    sizes here and needs no large binomials.
    """
    if not 0 <= n_target <= n_total:
        raise ValueError(f"cannot sample {n_target} of {n_total} tips")
    if support <= 0:
        raise ValueError("an origin with no descendant tips is not an origin")
    if support > n_total:
        raise ValueError(f"an origin supported by {support} tips in a {n_total}-tip lineage")
    if support > n_total - n_target:
        return 1.0                      # too many tips carry it to miss them all
    p_none = 1.0
    for j in range(support):
        p_none *= (n_total - n_target - j) / (n_total - j)
    return 1.0 - p_none


def p_two_or_more(probabilities: Sequence[float]) -> float:
    """Chance that two or more independent events happen -- a Poisson binomial tail.

    The origins of one replacement are separate branches, so whether each is sampled is
    independent given the subsample size. Only the 0, 1 and 2-or-more cells are needed,
    so the convolution is truncated at two and the cost is linear in the origin count.
    """
    distribution = [1.0, 0.0, 0.0]      # P(0), P(1), P(>=2)
    for p in probabilities:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{p} is not a probability")
        distribution = [distribution[0] * (1 - p),
                        distribution[0] * p + distribution[1] * (1 - p),
                        distribution[2] + distribution[1] * p]
    return distribution[2]


def rarefied_counts(supports: Iterable[Sequence[int]], n_total: int,
                    n_target: int) -> tuple[float, float]:
    """``(replacements still placed, of them homoplasies)`` at a common sampling depth.

    Table SZ's homoplasy share rises with sampling depth -- Spearman 0.96 against
    sequence count across the thirteen lineages -- because every extra sequence is another
    chance to catch a recurrence. That is a species-accumulation problem, and rarefying
    every lineage to one depth is its standard answer. Rarefying to the smallest lineage
    (778 tips) drops that correlation to 0.23.

    Both counts are expectations, not a draw: the probabilities above are exact, so no
    subsampling is simulated and the answer has no Monte Carlo error. ``supports`` is one
    sequence of descendant tip counts per replacement, which is the ``supports`` column
    :func:`origins_frame` writes.
    """
    placed = homoplasies = 0.0
    for group in supports:
        visible = [p_origin_visible(s, n_total, n_target) for s in group]
        none = 1.0
        for p in visible:
            none *= (1 - p)
        placed += 1 - none
        homoplasies += p_two_or_more(visible)
    return placed, homoplasies


#: One line per summary column, the single source for the docs and the exporter's header.
SUMMARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("lineage", "the lineage"),
    ("n_observed", "replacements with obs_present = 1 in the shipped observed table"),
    ("n_covered_positions", "root-protein positions the UShER reference has a codon for"),
    ("n_observed_covered", "those at a position the UShER reference has a codon for"),
    ("n_placed", "those the tree places at least one origin for"),
    ("n_unplaced", "observed and covered, but with no origin in the tree"),
    ("n_homoplasy", "placed replacements with two or more origins"),
    ("n_homoplasy_one_route", "homoplasies whose origins all used the same codon change"),
    ("n_homoplasy_multi_route", "homoplasies reached by two or more different codon changes"),
    ("n_homoplasy_internal", "homoplasies with two or more origins on internal branches alone"),
    ("n_origins_total", "origins summed over the placed replacements"),
    ("pct_multi_origin", "n_homoplasy as a percentage of n_placed"),
    ("max_origins", "most origins any one replacement has"),
    ("n_tree_tips", "lineage sequences the tree holds"),
    ("n_unseen", "placed replacements the observed table does not mark as present"),
)


#: One line per rarefied column, the single source for the docs and the exporter's header.
RAREFIED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("lineage", "the lineage"),
    ("rarefied_to", "tips every lineage is rarefied to: the smallest lineage's tip count"),
    ("rarefied_n_placed", "replacements expected to remain placed at that depth"),
    ("rarefied_n_homoplasy", "of those, the number expected to still have two or more origins"),
    ("rarefied_pct_multi_origin", "the second as a percentage of the first"),
)


def rarefy_frame(per_lineage: Mapping[str, pd.DataFrame], tree_tips: Mapping[str, int],
                 n_target: int | None = None) -> pd.DataFrame:
    """``pct_multi_origin`` recomputed at one sampling depth for every lineage.

    The shipped ``pct_multi_origin`` cannot be read across lineages: it tracks sequence
    count at Spearman 0.96, because a deeper sample catches more recurrences of the same
    replacement. This column can be, and it is the one to quote when lineages are
    compared. ``n_target`` defaults to the smallest lineage's tip count, the deepest
    common depth available; the lineage that defines it is returned unchanged, which is
    the arithmetic check that the rarefaction is doing nothing at full depth.

    Needs the ``supports`` column, so the frames must come from an
    :func:`origins_frame` that was given ``n_tip``.
    """
    if n_target is None:
        n_target = min(tree_tips[lineage] for lineage in per_lineage)
    rows = []
    for lineage in sorted(per_lineage):
        frame = per_lineage[lineage]
        if "supports" not in frame.columns:
            raise ValueError(f"{lineage}: origins_frame was not given n_tip, so there are "
                             "no per-origin supports to rarefy")
        total = tree_tips[lineage]
        if n_target > total:
            raise ValueError(f"cannot rarefy {lineage} to {n_target} tips: it has {total}")
        supports = [[int(x) for x in str(cell).split(",")] for cell in frame["supports"]]
        placed, homoplasies = rarefied_counts(supports, total, n_target)
        rows.append({
            "lineage": lineage,
            "rarefied_to": n_target,
            "rarefied_n_placed": placed,
            "rarefied_n_homoplasy": homoplasies,
            "rarefied_pct_multi_origin": 100.0 * homoplasies / placed if placed else float("nan"),
        })
    return pd.DataFrame(rows, columns=[name for name, _ in RAREFIED_COLUMNS])


def placed_frames(per_lineage: Mapping[str, pd.DataFrame], observed: pd.DataFrame,
                  covered_positions: Mapping[str, Sequence[int]] | Sequence[int],
                  ) -> dict[str, pd.DataFrame]:
    """Each lineage's frame cut to the replacements Table SZ's denominator counts.

    A replacement counts only when the observed table marks it present *and* its position
    has a UShER reference codon. :func:`summarise` and :func:`rarefy_frame` have to agree
    about that set or their two percentages are taken over different denominators -- which
    is exactly the defect this function exists to prevent: the rarefied share of the
    smallest lineage must come back equal to its raw share, and it cannot if the two are
    counting different replacements.
    """
    seen = observed[observed["obs_present"] == 1]
    present = {lineage: set(zip(group["position"], group["aa"]))
               for lineage, group in seen.groupby("lineage")}
    out = {}
    for lineage, frame in per_lineage.items():
        if isinstance(covered_positions, Mapping):
            covered = set(covered_positions[lineage])
        else:
            covered = set(covered_positions)
        keep = {(p, a) for p, a in present.get(lineage, set()) if p in covered}
        out[lineage] = frame[[key in keep for key in zip(frame["position"], frame["aa"])]]
    return out


def summarise(per_lineage: Mapping[str, pd.DataFrame], observed: pd.DataFrame,
              covered_positions: Mapping[str, Sequence[int]] | Sequence[int],
              tree_tips: Mapping[str, int]) -> pd.DataFrame:
    """Fold the per-lineage origin frames into one row per lineage.

    ``observed`` is the shipped ``replacement_observed`` table. ``covered_positions`` is
    either one sequence of positions for every lineage or one per lineage. The denominator
    for ``pct_multi_origin`` is ``n_placed``, not ``n_observed``: a replacement the tree
    cannot place is not evidence either way, and dividing by the full observed count would
    quietly read every coverage gap as a single-origin replacement.
    """
    seen = observed[observed["obs_present"] == 1]
    present = {lineage: set(zip(group["position"], group["aa"]))
               for lineage, group in seen.groupby("lineage")}
    kept = placed_frames(per_lineage, observed, covered_positions)

    rows = []
    for lineage in sorted(per_lineage):
        frame = per_lineage[lineage]
        if isinstance(covered_positions, Mapping):
            covered = set(covered_positions[lineage])
        else:
            covered = set(covered_positions)
        observed_here = present.get(lineage, set())
        observed_covered = {(p, a) for p, a in observed_here if p in covered}

        placed_keys = set(zip(frame["position"], frame["aa"]))
        placed = kept[lineage]
        homoplasy = placed[placed["n_origins"] >= 2]
        rows.append({
            "lineage": lineage,
            "n_observed": len(observed_here),
            "n_covered_positions": len(covered),
            "n_observed_covered": len(observed_covered),
            "n_placed": len(placed),
            "n_unplaced": len(observed_covered - placed_keys),
            "n_homoplasy": len(homoplasy),
            "n_homoplasy_one_route": int((homoplasy["n_routes"] == 1).sum()),
            "n_homoplasy_multi_route": int((homoplasy["n_routes"] >= 2).sum()),
            "n_homoplasy_internal": int((homoplasy["n_internal_origins"] >= 2).sum()),
            "n_origins_total": int(placed["n_origins"].sum()),
            "pct_multi_origin": 100.0 * len(homoplasy) / len(placed) if len(placed) else float("nan"),
            "max_origins": int(placed["n_origins"].max()) if len(placed) else 0,
            "n_tree_tips": int(tree_tips[lineage]),
            "n_unseen": len(placed_keys - observed_here),
        })
    frame = pd.DataFrame(rows, columns=[name for name, _ in SUMMARY_COLUMNS])
    bad = frame["n_homoplasy_one_route"] + frame["n_homoplasy_multi_route"] != frame["n_homoplasy"]
    if bad.any():
        raise ValueError("the one-route and multi-route homoplasies do not partition the homoplasies")
    return frame
