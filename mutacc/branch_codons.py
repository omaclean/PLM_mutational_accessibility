"""Nucleotide changes per codon, per branch, on a UShER tree (supplementary figure S11).

Question: when a codon changes on one branch of a tree, how many of its three nucleotides
changed, and how many did the genetic code force? Two nucleotide changes in one codon on one
branch are interesting because a single change would often have done the same job.

This module is pure: no file is read, no program is run. ``tools/branch_codon_tables.py`` does
the input and output and calls these functions. It replaces the archive's
``scripts/branch_codon_distance.py``, which never produced output (``docs/notes/PLAN.md`` section 3.2).

What goes in
------------
* a tree (Newick, as ``matUtils extract -t`` writes it, all tips expanded);
* the mutations UShER placed on each branch (``matUtils extract -A``: ``node: A27G,T99C``);
* the reference sequence: the FIRST record of the alignment UShER was built from. UShER
  positions are 1-based positions along that record with its gap columns skipped, so codon
  numbers here are counted along the reference record, never along alignment columns. A gap
  block elsewhere in the alignment therefore cannot shift them. The frame is asserted
  (``check_frame``) rather than assumed;
* the target tips.

Trap: ``faToVcf`` needs ``-includeNoAltN`` when the UShER tree is built. Without it UShER reads
sites that were never sequenced (N or gap) as the reference base, so a codon can describe sites
nobody observed. That cannot be repaired here: it is fixed when the tree is built, and every
number below inherits the convention. The reference is checked against the VCF and against every
mutation's "from" base, which catches a wrong reference or coordinate system, not this.

What comes out
--------------
Per branch, per codon changed on it, one *event* (``CodonEvent``). Events are folded into two
tables and are only written out on request.

Branch table (``branch_table``): one row per branch with FEWER THAN ``max_nt`` nucleotide
changes (strictly fewer; the archive kept ``<=``). Branches with no changes are not rows.
Column table (``column_table``): one row per codon of the coding region, with how many times
the column changed by 1, 2 and 3 nucleotides over the whole tree. Recurrence is collapsed there
by design: a column that changed 167 independent times is one row with counts, not 167 rows.

Definitions (``COLUMN_DOCS`` holds one line per column and is the single source for the docs)
---------------------------------------------------------------------------------------------
``n_nt``
    Positions at which the parent and child codon differ (net change, 1 to 3).
``n_mut``
    Mutation records UShER put in that codon on that branch. Differs from ``n_nt`` when two
    records hit one position or a position is changed and changed back.
``n_min_from_parent``
    Fewest nucleotide changes that turn the PARENT codon into any codon of the new amino acid.
    Always at most ``n_nt``. This is the headline comparator: ``n_nt > n_min_from_parent`` means
    a shorter route existed from where the branch started.
``n_min_code``
    Fewest changes between any codon of the old amino acid and any codon of the new one, which
    ignores which synonymous codon the parent had. It is at most ``n_min_from_parent`` and is
    kept for comparison with the archive only. ``n_nt > n_min_code`` usually means the parent
    carried a synonymous codon far from the target, not that the branch took a long route.
``isolated``
    Every mutation record on the branch lies in this codon (``branch_nt_total == n_mut``).
``terminal``
    The branch ends in a tip. Terminal branches are skipped unless asked for, because tips carry
    the most sequencing and alignment error; the column exists only when they are included.

Categories of a codon changed on a branch: ``syn`` (same amino acid), ``nonsyn`` (different),
``dropped_stop`` (parent or child codon is a stop), ``dropped_ambiguous`` (a codon with a base
other than A, C, G or T), ``reverted`` (touched, but the net result equals the parent).
"""

from __future__ import annotations

import gzip
import io
import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

import pandas as pd
from Bio.Data import CodonTable
from Bio.Seq import Seq

__all__ = [
    "BASES", "CODON2AA", "AA2CODONS", "MIN_AA_DISTANCE", "FrameError", "InputError",
    "hamming", "min_distance_from_codon", "Tree", "parse_newick", "parse_mutation_paths",
    "mrca",
    "first_fasta_record", "ungap", "check_frame", "check_reference_against_vcf", "Cds",
    "count_below", "main_clade_and_start", "CodonEvent", "WalkResult", "walk_branches",
    "events_frame", "LongRange", "compare_to_start", "branch_table", "column_table", "COLUMN_DOCS", "BRANCH_COLUMNS",
    "COLUMN_COLUMNS", "write_table", "read_table", "summary_text",
]

BASES = "ACGT"
GAP_CHARS = "-."

# ---------------------------------------------------------------------------
# Genetic code, taken from Biopython (standard table), never typed in.
# ---------------------------------------------------------------------------

_TABLE = CodonTable.unambiguous_dna_by_id[1]
CODON2AA: dict[str, str] = {
    "".join(c): str(Seq("".join(c)).translate()) for c in itertools.product(BASES, repeat=3)
}
AA2CODONS: dict[str, list[str]] = {}
for _codon, _aa in CODON2AA.items():
    AA2CODONS.setdefault(_aa, []).append(_codon)
assert sorted(c for c in CODON2AA if CODON2AA[c] == "*") == sorted(_TABLE.stop_codons)
AMINO_ACIDS = sorted(a for a in AA2CODONS if a != "*")


def hamming(a: str, b: str) -> int:
    """Positions at which two equal-length strings differ."""
    if len(a) != len(b):
        raise ValueError(f"hamming needs equal lengths, got {len(a)} and {len(b)}")
    return sum(x != y for x, y in zip(a, b))


#: Fewest nucleotide changes between any codon of one amino acid and any of another.
MIN_AA_DISTANCE: dict[tuple[str, str], int] = {
    (a, b): 0 if a == b else min(hamming(x, y) for x in AA2CODONS[a] for y in AA2CODONS[b])
    for a, b in itertools.product(AMINO_ACIDS, repeat=2)
}


def min_distance_from_codon(codon: str, aa_to: str) -> int:
    """Fewest nucleotide changes from this exact codon to any codon of ``aa_to``."""
    return min(hamming(codon, c) for c in AA2CODONS[aa_to])


class InputError(ValueError):
    """An input file is malformed or inconsistent with another input."""


class FrameError(InputError):
    """The reference record does not read as an open reading frame."""


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

@dataclass
class Tree:
    children: dict[str, list[str]]
    parent: dict[str, str]
    root: str
    preorder: list[str]

    @property
    def tips(self) -> set[str]:
        return {n for n, k in self.children.items() if not k}


_NEWICK_TOKEN = re.compile(r"\(|\)|,|;|:[^(),;]*|'(?:[^']|'')*'|[^(),:;'\s][^(),:;]*")


def parse_newick(text: str) -> Tree:
    """Iterative Newick reader (trees here have tens of thousands of tips; no recursion).

    Branch lengths are ignored. Duplicate names, unbalanced brackets, an unlabelled tip, an
    empty string and text after the closing ``;`` are all errors. An unlabelled internal node
    is named ``_u<N>``.
    """
    text = "".join(text.split()) if "'" not in text else text.strip()
    if not text:
        raise InputError("empty tree")
    children: dict[str, list[str]] = {}
    parent: dict[str, str] = {}
    stack: list[list[str]] = []
    root: str | None = None
    closed: list[str] | None = None      # children of an internal node just closed by ')'
    n_unlabelled = 0
    done = False

    def new_name(name: str) -> str:
        if name in children:
            raise InputError(f"duplicate node name {name!r} in the tree")
        return name

    def place(name: str, kids: list[str]) -> None:
        nonlocal root
        children[name] = kids
        for kid in kids:
            parent[kid] = name
        if stack:
            stack[-1].append(name)
        else:
            root = name

    pos = 0
    tokens = []
    while pos < len(text):
        m = _NEWICK_TOKEN.match(text, pos)
        if not m:
            raise InputError(f"cannot read the tree at character {pos}: {text[pos:pos + 20]!r}")
        tokens.append(m.group(0))
        pos = m.end()
    for tok in tokens:
        if done:
            raise InputError("text after the closing ';' of the tree")
        if tok.startswith(":"):
            continue
        if closed is not None:
            kids, closed = closed, None
            if tok in "(),;":
                n_unlabelled += 1
                name = new_name(f"_u{n_unlabelled}")
                place(name, kids)
                # fall through to handle this punctuation token
            else:
                place(new_name(tok.strip("'")), kids)
                continue
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if not stack:
                raise InputError("unbalanced ')' in the tree")
            closed = stack.pop()
            if not closed:
                raise InputError("empty '()' in the tree")
        elif tok == ",":
            if not stack:
                raise InputError("',' outside brackets in the tree")
        elif tok == ";":
            done = True
        else:
            name = new_name(tok.strip("'"))
            children[name] = []
            if stack:
                stack[-1].append(name)
            else:
                root = name
    if stack:
        raise InputError("unbalanced '(' in the tree")
    if closed is not None:   # '(...)' with no label and no ';'
        n_unlabelled += 1
        place(new_name(f"_u{n_unlabelled}"), closed)
    if root is None:
        raise InputError("no tree found")
    # a node only becomes a child via stack lists, so a leaf name that is also a parent is caught above
    preorder, todo = [], [root]
    while todo:
        node = todo.pop()
        preorder.append(node)
        todo.extend(reversed(children[node]))
    if len(preorder) != len(children):
        raise InputError("the tree has nodes that are not connected to its root")
    return Tree(children, parent, root, preorder)


def count_below(tree: Tree, members: Iterable[str]) -> dict[str, int]:
    """Number of ``members`` among the tips at or below every node."""
    members = set(members)
    n: dict[str, int] = {}
    for node in reversed(tree.preorder):
        kids = tree.children[node]
        n[node] = sum(n[k] for k in kids) if kids else int(node in members)
    return n


def mrca(tree: Tree, tips: Iterable[str], n_tip: Mapping[str, int] | None = None) -> str:
    """The lowest node holding every one of ``tips``.

    Unlike ``main_clade_and_start`` this makes no purity judgement: it is the common
    ancestor and nothing else, so a clade that is not monophyletic simply returns a node
    with foreign tips below it. Pass ``n_tip`` to reuse a ``count_below`` already done.
    """
    tips = set(tips)
    if not tips:
        raise InputError("no tips to take a common ancestor of")
    missing = tips - tree.tips
    if missing:
        raise InputError(f"{len(missing)} of the tips are not in the tree, "
                         f"first {sorted(missing)[0]!r}")
    n_tip = count_below(tree, tips) if n_tip is None else n_tip
    node = tree.root
    while True:
        below = [k for k in tree.children[node] if n_tip[k] == len(tips)]
        if not below:
            return node
        node = below[0]


def main_clade_and_start(tree: Tree, all_tips: set[str], tips: set[str], purity: float):
    """Pick the node the walk starts from.

    The main clade is found by following, from the root, the child holding most ``all_tips``
    until at least ``purity`` of its tips are in that set (a few mislabelled tips elsewhere would
    otherwise drag the common ancestor to the root). The start is then the lowest node below it
    that still holds every one of ``tips``. Returns ``(clade, start, size, n_all, n_tip)``.
    """
    if not 0 < purity <= 1:
        raise ValueError(f"purity must be in (0, 1], got {purity}")
    if not tips:
        raise InputError("no target tips")
    size = count_below(tree, tree.tips)
    n_all = count_below(tree, all_tips)
    n_tip = count_below(tree, tips)
    clade = tree.root
    while n_all[clade] / size[clade] < purity and tree.children[clade]:
        clade = max(tree.children[clade], key=lambda k: n_all[k])   # first of equals: tree order
    start = clade
    while True:
        nxt = [k for k in tree.children[start] if n_tip[k] == n_tip[start]]
        if not nxt:
            break
        start = nxt[0]
    return clade, start, size, n_all, n_tip


# ---------------------------------------------------------------------------
# Mutations and the reference
# ---------------------------------------------------------------------------

_MUT = re.compile(r"([ACGT])(\d+)([ACGT])")


def parse_mutation_paths(lines: Iterable[str], tree: Tree | None = None) -> dict[str, list[tuple[str, int, str]]]:
    """``matUtils extract -A`` lines -> ``{node: [(from, 1-based position, to), ...]}``.

    Nodes with no mutations are left out. Record order within a node is kept, because two
    records at one position are applied in that order. An unreadable record, a repeated node
    line, a position below 1, or (given ``tree``) a node the tree does not have is an error:
    dropping records silently would make the tables look cleaner than the tree is.
    """
    out: dict[str, list[tuple[str, int, str]]] = {}
    seen: set[str] = set()
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        node, sep, text = line.partition(":")
        node = node.strip()
        if not sep or not node:
            raise InputError(f"mutation line without 'node: mutations': {line[:60]!r}")
        if node in seen:
            raise InputError(f"node {node!r} appears twice in the mutation file")
        seen.add(node)
        if tree is not None and node not in tree.children:
            raise InputError(f"mutation file names {node!r}, which is not in the tree")
        recs = []
        for tok in (t.strip() for t in text.split(",")):
            if not tok:
                continue
            m = _MUT.fullmatch(tok)
            if not m or int(m[2]) < 1:
                raise InputError(f"cannot read mutation {tok!r} on {node!r}; expected like A27G")
            if m[1] == m[3]:
                raise InputError(f"mutation {tok!r} on {node!r} changes nothing")
            recs.append((m[1], int(m[2]), m[3]))
        if recs:
            out[node] = recs
    return out


def first_fasta_record(text_lines: Iterable[str]) -> tuple[str, str]:
    """(name, sequence) of the first FASTA record; nothing else is read."""
    name, buf = None, []
    for line in text_lines:
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                break
            name = line[1:].strip()
        elif name is not None:
            buf.append(line)
        elif line:
            raise InputError("alignment does not start with a FASTA header")
    if name is None:
        raise InputError("no FASTA record found")
    seq = "".join(buf).upper()
    if not seq:
        raise InputError(f"first record {name!r} is empty")
    return name, seq


def ungap(seq: str) -> str:
    """The reference record without its gap columns: the coordinate system UShER uses."""
    return "".join(ch for ch in seq.upper() if ch not in GAP_CHARS)


@dataclass(frozen=True)
class Cds:
    """The coding region as 1-based inclusive positions along the reference record."""
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def n_codons(self) -> int:
        return self.length // 3


def check_frame(ref: str, start: int = 1, end: int | None = None) -> Cds:
    """Assert that ``ref[start-1:end]`` is an open reading frame, and return it as a ``Cds``.

    Fails unless the length is a multiple of three and the translation has no stop before its
    last codon (a final stop is allowed). The message names the offset that does read cleanly
    when there is one, which is usually a 5' UTR the caller forgot to skip.
    """
    ref = ref.upper()
    end = len(ref) if end is None else end
    if start < 1 or end > len(ref) or end < start:
        raise FrameError(f"coding region {start}..{end} does not fit the {len(ref)}-base reference record")
    cds = Cds(start, end)
    if cds.length % 3:
        raise FrameError(f"coding region {start}..{end} is {cds.length} bases, not a multiple of 3")
    stops = [i for i in range(cds.n_codons - 1) if CODON2AA.get(ref[start - 1 + 3 * i:start + 2 + 3 * i]) == "*"]
    if stops:
        hint = ""
        for off in (1, 2, 0):
            seg = ref[start - 1 + off:end]
            seg = seg[: len(seg) - len(seg) % 3]
            if not any(CODON2AA.get(seg[i:i + 3]) == "*" for i in range(0, len(seg) - 3, 3)):
                hint = f"; starting at {start + off} reads without an internal stop"
                break
        raise FrameError(f"reference record has a stop codon at codon {stops[0] + 1} "
                         f"(positions {start + 3 * stops[0]}..{start + 3 * stops[0] + 2}) inside {start}..{end}{hint}")
    return cds


def check_reference_against_vcf(ref: str, vcf_lines: Iterable[str]) -> int:
    """Every VCF REF base must equal the reference record at that position. Returns sites checked.

    This is what shows that UShER positions count the reference record's own bases. A different
    coordinate system, or the wrong alignment, disagrees at most sites. Any disagreement is an
    error: a tolerance would let a shifted frame through.
    """
    n = 0
    bad = []
    for line in vcf_lines:
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 4:
            raise InputError(f"VCF line with fewer than 4 columns: {line[:60]!r}")
        pos = int(f[1])
        if not 1 <= pos <= len(ref):
            raise InputError(f"VCF position {pos} is outside the {len(ref)}-base reference record")
        n += 1
        if ref[pos - 1] != f[3].upper():
            bad.append((pos, f[3], ref[pos - 1]))
    if n == 0:
        raise InputError("the VCF has no sites to check the reference against")
    if bad:
        raise InputError(f"reference record disagrees with the VCF REF base at {len(bad)} of {n} sites "
                         f"(first: position {bad[0][0]} VCF {bad[0][1]} record {bad[0][2]}); wrong "
                         "alignment, or UShER positions are not counted along the first record")
    return n


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

CATEGORIES = ("syn", "nonsyn", "dropped_stop", "dropped_ambiguous", "reverted")


@dataclass(frozen=True)
class CodonEvent:
    branch: str            # child node of the branch
    parent: str
    codon: int             # 1-based, along the reference record's coding region
    codon_from: str
    codon_to: str
    category: str
    n_mut: int
    n_nt: int
    aa_from: str
    aa_to: str
    n_min_code: int | None
    n_min_from_parent: int | None
    isolated: bool
    branch_nt_total: int
    terminal: bool


@dataclass
class BranchRecord:
    branch: str
    parent: str
    terminal: bool
    n_target_tips: int
    nt_total: int
    nt_in_cds: int


@dataclass
class WalkResult:
    events: list[CodonEvent] = field(default_factory=list)
    branches: list[BranchRecord] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)


def _apply(state: list[str], node: str, recs: Sequence[tuple[str, int, str]], undo: list | None) -> None:
    for frm, pos, to in recs:
        p = pos - 1
        if p >= len(state):
            raise InputError(f"mutation {frm}{pos}{to} on {node!r} lies beyond the {len(state)}-base reference record")
        if state[p] != frm:
            raise InputError(f"mutation {frm}{pos}{to} on {node!r}: tracked sequence has {state[p]!r} there. "
                             "The reference record is not the one UShER used, or positions are counted differently")
        if undo is not None:
            undo.append((p, state[p]))
        state[p] = to


def _classify(before: str, after: str) -> tuple[str, str, str, int | None, int | None, int]:
    """(category, aa_from, aa_to, n_min_code, n_min_from_parent, n_nt) for one codon change."""
    if before == after:
        return "reverted", CODON2AA.get(before, "?"), CODON2AA.get(after, "?"), None, None, 0
    n_nt = hamming(before, after)
    a1, a2 = CODON2AA.get(before), CODON2AA.get(after)
    if a1 is None or a2 is None:
        return "dropped_ambiguous", a1 or "?", a2 or "?", None, None, n_nt
    if a1 == "*" or a2 == "*":
        return "dropped_stop", a1, a2, None, None, n_nt
    if a1 == a2:
        return "syn", a1, a2, 0, 0, n_nt
    return "nonsyn", a1, a2, MIN_AA_DISTANCE[(a1, a2)], min_distance_from_codon(before, a2), n_nt


def walk_branches(tree: Tree, muts: Mapping[str, Sequence[tuple[str, int, str]]], ref: str, cds: Cds,
                  start: str, n_tip: Mapping[str, int], *, max_nt: int | None = 10,
                  include_terminal: bool = False, skip_codons: Collection[int] = ()) -> WalkResult:
    """Walk every branch below ``start`` that leads to at least one target tip.

    The start node's own branch is ancestral, not scored: its mutations (and those above it,
    root included) are applied to the reference so every scored branch has the right parent
    sequence. A branch is scored when it has more than 0 and strictly fewer than ``max_nt``
    mutation records (``max_nt`` of None or 0 means no upper limit) and, unless
    ``include_terminal``, does not end in a tip. Every mutation's "from" base is checked
    against the tracked sequence; a mismatch raises ``InputError``.

    ``ref`` is the reference record without gap columns. Positions outside the coding region
    count toward ``nt_total`` and ``stats['mutations_outside_cds']`` but make no events.
    ``skip_codons`` (1-based codon numbers) are treated the same way, and are also counted in
    ``stats['mutations_in_skipped_codons']``: their mutations still change the tracked sequence,
    they just never become events. Used for columns where the protein of interest has no residue,
    so that an alignment gap read as a base cannot show up as a run of multi-position changes.
    """
    skip = frozenset(skip_codons)
    if start not in tree.children:
        raise InputError(f"start node {start!r} is not in the tree")
    state = list(ref.upper())
    path = []
    node = start
    while True:
        path.append(node)
        if node == tree.root:
            break
        node = tree.parent[node]
    stats: Counter = Counter()
    for node in reversed(path):
        recs = muts.get(node, ())
        stats["ancestral_mutations"] += len(recs)
        _apply(state, node, recs, None)

    result = WalkResult(stats=stats)
    c0 = cds.start - 1
    undo: dict[str, list[tuple[int, str]]] = {}
    todo: list[tuple[str, str | None, bool]] = [(start, None, False)]
    while todo:
        node, par, leaving = todo.pop()
        if leaving:
            for p, old in reversed(undo.pop(node)):   # reversed: two records at one position
                state[p] = old
            continue
        todo.append((node, None, True))
        todo.extend((k, node, False) for k in reversed(tree.children[node]) if n_tip[k] > 0)
        if par is None:
            undo[node] = []
            continue
        recs = muts.get(node, ())
        by_codon: dict[int, list[int]] = {}
        outside = skipped_here = 0
        for _, pos, _ in recs:
            p = pos - 1
            if c0 <= p < cds.end and (p - c0) // 3 + 1 not in skip:
                by_codon.setdefault((p - c0) // 3, []).append(p)
            else:
                outside += 1
                if c0 <= p < cds.end:
                    skipped_here += 1
        before = {c: "".join(state[c0 + 3 * c:c0 + 3 * c + 3]) for c in by_codon}
        undo[node] = []
        _apply(state, node, recs, undo[node])
        terminal = not tree.children[node]
        stats["branches_walked"] += 1
        nt_total = len(recs)
        if terminal and not include_terminal:
            stats["skipped_terminal"] += 1
            continue
        if max_nt and nt_total >= max_nt:
            stats["skipped_long"] += 1
            continue
        if nt_total == 0:
            stats["skipped_zero_nt"] += 1
            continue
        stats["kept"] += 1
        stats["mutations_outside_cds"] += outside - skipped_here
        stats["mutations_in_skipped_codons"] += skipped_here
        result.branches.append(BranchRecord(node, par, terminal, n_tip[node], nt_total, nt_total - outside))
        for c in sorted(by_codon):
            after = "".join(state[c0 + 3 * c:c0 + 3 * c + 3])
            cat, a1, a2, n_code, n_par, n_nt = _classify(before[c], after)
            n_mut = len(by_codon[c])
            result.events.append(CodonEvent(node, par, c + 1, before[c], after, cat, n_mut, n_nt, a1, a2,
                                            n_code, n_par, nt_total == n_mut, nt_total, terminal))
    return result


# ---------------------------------------------------------------------------
# The long comparison: oldest scored node against the newest internal nodes
# ---------------------------------------------------------------------------

@dataclass
class LongRange:
    """Net amino-acid differences between the start node and every newest internal node."""
    n_nodes: int
    per_node: pd.DataFrame      # one row per newest internal node
    changes: pd.DataFrame       # one row per distinct (codon, new amino acid) seen in at least one such node
    baseline: pd.DataFrame      # the baseline's own codons: one row per scored codon


def compare_to_start(tree: Tree, muts: Mapping[str, Sequence[tuple[str, int, str]]], ref: str, cds: Cds,
                     start: str, n_tip: Mapping[str, int], *, skip_codons: Collection[int] = (),
                     baseline: str | None = None) -> LongRange:
    """Compare each *newest internal node* with the start node, codon by codon.

    A newest internal node is an internal node below ``start`` all of whose children are tips, and that
    leads to at least one target tip. For every codon whose amino acid differs from the start node's, record the
    fewest nucleotide changes that turn the START codon into any codon of the new amino acid
    (``n_min``): the question is how many changes the amino acid change needed, not how many the tree used.
    Observed differences between the two codons are ``n_nt``. Codons in ``skip_codons`` are left out, and
    codons that are stops or ambiguous at either end are counted apart.

    Mutations on every branch are applied, whatever the branch's length, because the state of a node is
    the sum of its path. The branch-length cap of ``walk_branches`` is about which branches are scored,
    not which are believed.
    """
    if start not in tree.children:
        raise InputError(f"start node {start!r} is not in the tree")
    if baseline is not None and baseline not in tree.children:
        raise InputError(f"baseline node {baseline!r} is not in the tree")
    skip = frozenset(skip_codons)
    c0, n_codons = cds.start - 1, cds.n_codons

    def state_at(target: str) -> list[str]:
        """The reference record with every mutation from the root down to ``target`` applied."""
        s = list(ref.upper())
        path, node = [], target
        while True:
            path.append(node)
            if node == tree.root:
                break
            node = tree.parent[node]
        for node in reversed(path):
            _apply(s, node, muts.get(node, ()), None)
        return s

    # The baseline is what every node is compared against; the walk still descends from
    # ``start``, so ``state`` has to begin at ``start`` whatever the baseline is. Without a
    # baseline the two are the same node and this is the original behaviour.
    #
    # They are separable because the start node's reconstruction is NOT a usable ancestor:
    # UShER reads sites that were never sequenced as the reference base, so near the root
    # the reconstruction is largely the reference record itself. When that record is a
    # different subtype, comparing against it measures the distance between subtypes rather
    # than drift within one. Pass a real dated sequence as ``baseline`` instead.
    anc_state = state_at(baseline) if baseline is not None else state_at(start)
    anc = ["".join(anc_state[c0 + 3 * c:c0 + 3 * c + 3]) for c in range(n_codons)]
    state = state_at(start)
    live = [c for c in range(n_codons) if c + 1 not in skip]

    node_rows, seen = [], {}
    undo: dict[str, list[tuple[int, str]]] = {}
    todo: list[tuple[str, bool]] = [(start, False)]
    while todo:
        node, leaving = todo.pop()
        if leaving:
            for p, old in reversed(undo.pop(node)):
                state[p] = old
            continue
        todo.append((node, True))
        kids = [k for k in tree.children[node] if n_tip[k] > 0]
        todo.extend((k, False) for k in reversed(kids))
        undo[node] = []
        if node != start:
            _apply(state, node, muts.get(node, ()), undo[node])
        if node == start or not tree.children[node] or any(tree.children[k] for k in tree.children[node]):
            continue
        row = Counter()
        for c in live:
            now = "".join(state[c0 + 3 * c:c0 + 3 * c + 3])
            if now == anc[c]:
                continue
            cat, a1, a2, n_code, n_par, n_nt = _classify(anc[c], now)
            if cat != "nonsyn":
                row[cat] += 1
                continue
            row["nonsyn"] += 1
            row[f"min{n_par}"] += 1
            row[f"nt{n_nt}"] += 1
            key = (c + 1, a2)
            hit = seen.setdefault(key, {"codon": c + 1, "codon_from": anc[c], "aa_from": a1, "aa_to": a2,
                                        "n_min": n_par, "n_min_code": n_code, "nodes": 0, "n_nt_least": n_nt})
            hit["nodes"] += 1
            hit["n_nt_least"] = min(hit["n_nt_least"], n_nt)
        node_rows.append({"node": node, "n_target_tips": n_tip[node], **row})
    cols = ["node", "n_target_tips", "nonsyn", "min1", "min2", "min3", "nt1", "nt2", "nt3",
            "syn", "dropped_stop", "dropped_ambiguous", "reverted"]
    per_node = pd.DataFrame(node_rows).reindex(columns=cols).fillna(0)
    per_node[cols[1:]] = per_node[cols[1:]].astype(int)
    ch_cols = ["codon", "codon_from", "aa_from", "aa_to", "n_min", "n_min_code", "n_nt_least", "nodes"]
    changes = pd.DataFrame(sorted(seen.values(), key=lambda r: (r["codon"], r["aa_to"])), columns=ch_cols)
    base = pd.DataFrame([{"codon": c + 1, "codon_from": anc[c], "aa_from": CODON2AA.get(anc[c], "X")}
                         for c in live])
    return LongRange(len(per_node), per_node, changes, base)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_CLASS_COUNTS = [f"{cls}_{n}nt" for cls in ("syn", "nonsyn") for n in (1, 2, 3)]
_EXCESS = ["nonsyn_2nt_min1", "nonsyn_3nt_min1", "nonsyn_3nt_min2"]
_OTHER = ["dropped_stop", "dropped_ambiguous", "reverted"]

COLUMN_DOCS: dict[str, str] = {
    "branch": "Child node of the branch (the node UShER names; a tip name for a terminal branch).",
    "parent": "Parent node of the branch.",
    "terminal": "1 if the branch ends in a tip (column present only when terminal branches are included).",
    "n_target_tips": "Target tips at or below the branch.",
    "nt_total": "Nucleotide mutation records UShER put on the branch, in or out of the coding region.",
    "nt_in_cds": "Of those, how many lie inside the coding region and in a codon that is not skipped.",
    "codons_touched": "Codons of the coding region with at least one mutation record on the branch.",
    "syn_1nt": "Codons whose amino acid did not change, with 1 position differing from the parent codon.",
    "syn_2nt": "As syn_1nt, 2 positions differing.",
    "syn_3nt": "As syn_1nt, 3 positions differing.",
    "nonsyn_1nt": "Codons whose amino acid changed, with 1 position differing from the parent codon.",
    "nonsyn_2nt": "As nonsyn_1nt, 2 positions differing.",
    "nonsyn_3nt": "As nonsyn_1nt, 3 positions differing.",
    "nonsyn_2nt_min1": "Of nonsyn_2nt, those where a single change from the parent codon would have given the same new amino acid.",
    "nonsyn_3nt_min1": "Of nonsyn_3nt, those where one change from the parent codon would have sufficed.",
    "nonsyn_3nt_min2": "Of nonsyn_3nt, those where two changes from the parent codon would have sufficed. "
                       "Cells not listed (n equals the minimum) are nonsyn_Nnt minus these.",
    "dropped_stop": "Codons changed on the branch where the parent or child codon is a stop; not scored.",
    "dropped_ambiguous": "Codons changed on the branch where a codon holds a base other than A, C, G or T; not scored.",
    "reverted": "Codons touched by two or more records that cancel, so child equals parent; not scored.",
    "codon": "Codon number along the reference record's coding region, starting at 1.",
    "site_number": "codon minus the user-supplied numbering offset (for example mature-protein numbering); "
                   "column present only when an offset was given.",
    "ref_pos": "Position of the codon's first base along the reference record, starting at 1.",
    "ref_codon": "The reference record's codon here (not the codon of any ancestor).",
    "ref_aa": "Its amino acid; X when the codon holds a base other than A, C, G or T.",
    "events": "Branch-codon events of any category at this column; each is one branch on which the column changed.",
    "n_distinct_aa_changes": "Distinct (old, new) amino-acid pairs among the nonsyn events at this column.",
}

BRANCH_COLUMNS = ["branch", "parent", "n_target_tips", "nt_total", "nt_in_cds", "codons_touched",
                  *_CLASS_COUNTS, *_EXCESS, *_OTHER]
COLUMN_COLUMNS = ["codon", "ref_pos", "ref_codon", "ref_aa", "events", *_CLASS_COUNTS, *_EXCESS, *_OTHER,
                  "n_distinct_aa_changes"]


def _count_cells(events: Iterable[CodonEvent]) -> Counter:
    c: Counter = Counter()
    for e in events:
        if e.category in ("syn", "nonsyn"):
            c[f"{e.category}_{e.n_nt}nt"] += 1
            if e.category == "nonsyn" and e.n_nt > e.n_min_from_parent:
                c[f"nonsyn_{e.n_nt}nt_min{e.n_min_from_parent}"] += 1
        else:
            c[e.category] += 1
    return c


def branch_table(result: WalkResult, *, include_terminal: bool = False) -> pd.DataFrame:
    """One row per scored branch, in walk order."""
    by_branch: dict[str, list[CodonEvent]] = {}
    for e in result.events:
        by_branch.setdefault(e.branch, []).append(e)
    rows = []
    for b in result.branches:
        ev = by_branch.get(b.branch, [])
        row = {"branch": b.branch, "parent": b.parent, "n_target_tips": b.n_target_tips,
               "nt_total": b.nt_total, "nt_in_cds": b.nt_in_cds, "codons_touched": len(ev)}
        cells = _count_cells(ev)
        row.update({k: cells.get(k, 0) for k in [*_CLASS_COUNTS, *_EXCESS, *_OTHER]})
        if include_terminal:
            row["terminal"] = int(b.terminal)
        rows.append(row)
    cols = list(BRANCH_COLUMNS)
    if include_terminal:
        cols.insert(2, "terminal")
    return pd.DataFrame(rows, columns=cols)


def column_table(result: WalkResult, ref: str, cds: Cds, *, numbering_offset: int | None = None) -> pd.DataFrame:
    """One row per codon of the coding region, zeros included, so a row's absence never means zero."""
    by_codon: dict[int, list[CodonEvent]] = {}
    for e in result.events:
        by_codon.setdefault(e.codon, []).append(e)
    ref = ref.upper()
    rows = []
    for c in range(1, cds.n_codons + 1):
        first = cds.start - 1 + 3 * (c - 1)
        codon = ref[first:first + 3]
        ev = by_codon.get(c, [])
        cells = _count_cells(ev)
        row = {"codon": c, "ref_pos": first + 1, "ref_codon": codon, "ref_aa": CODON2AA.get(codon, "X"),
               "events": len(ev)}
        row.update({k: cells.get(k, 0) for k in [*_CLASS_COUNTS, *_EXCESS, *_OTHER]})
        row["n_distinct_aa_changes"] = len({(e.aa_from, e.aa_to) for e in ev if e.category == "nonsyn"})
        if numbering_offset is not None:
            row["site_number"] = c - numbering_offset
        rows.append(row)
    cols = list(COLUMN_COLUMNS)
    if numbering_offset is not None:
        cols.insert(1, "site_number")
    return pd.DataFrame(rows, columns=cols)


def events_frame(result: WalkResult, *, include_terminal: bool = False,
                 numbering_offset: int | None = None) -> pd.DataFrame:
    """Every branch-codon event as a row (for ``--dump-events``). Large; not read by the figure."""
    cols = ["branch", "parent", "codon", "codon_from", "codon_to", "category", "aa_from", "aa_to", "n_mut",
            "n_nt", "n_min_from_parent", "n_min_code", "isolated", "branch_nt_total"]
    df = pd.DataFrame([[getattr(e, c) for c in cols] for e in result.events], columns=cols)
    df["isolated"] = df["isolated"].astype(int)
    if numbering_offset is not None:
        df.insert(3, "site_number", df["codon"] - numbering_offset)
    if include_terminal:
        df["terminal"] = [int(e.terminal) for e in result.events]
    return df


# ---------------------------------------------------------------------------
# Tables on disk: gzipped TSV with leading "# key: value" provenance lines, byte-reproducible
# ---------------------------------------------------------------------------

def write_table(df: pd.DataFrame, path: str | Path, provenance: Mapping[str, str] | None = None) -> Path:
    """Write a TSV (gzipped when the name ends .gz) with leading ``# key: value`` lines.

    The gzip header carries no timestamp or file name, so the same table is the same bytes.
    Provenance values must be single-line and must not hold an absolute path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ""
    for k, v in (provenance or {}).items():
        v = str(v)
        if "\n" in v or "\n" in str(k):
            raise ValueError("provenance entries must be single-line")
        if v.startswith("/") or re.search(r"(^|\s)/(home|data|software|tmp)\b", v):
            raise ValueError(f"provenance {k!r} looks like an absolute path: {v!r}")
        head += f"# {k}: {v}\n"
    body = head + df.to_csv(sep="\t", index=False, lineterminator="\n")
    raw = body.encode()
    if path.suffix == ".gz":
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as fh:
            fh.write(raw)
        raw = buf.getvalue()
    path.write_bytes(raw)
    return path


def read_table(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a table written by ``write_table``: (frame, provenance)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    prov: dict[str, str] = {}
    n_head = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            k, _, v = line[1:].partition(":")
            prov[k.strip()] = v.strip()
            n_head += 1
    df = pd.read_csv(path, sep="\t", skiprows=n_head, dtype={"branch": str, "parent": str, "ref_codon": str},
                     keep_default_na=False)
    return df, prov


def summary_text(result: WalkResult, branch: pd.DataFrame, column: pd.DataFrame, header: Sequence[str] = ()) -> str:
    """Plain-text run summary with the counts every number in the figure caption comes from."""
    s = result.stats
    cells = {k: int(branch[k].sum()) for k in [*_CLASS_COUNTS, *_EXCESS, *_OTHER]} if len(branch) else {}
    out = list(header)
    out += [
        f"branches walked: {s['branches_walked']}; scored: {s['kept']}; skipped terminal: {s['skipped_terminal']}; "
        f"skipped for too many changes: {s['skipped_long']}; skipped with no changes: {s['skipped_zero_nt']}",
        f"ancestral mutations applied, not scored: {s['ancestral_mutations']}",
        f"mutation records outside the coding region (scored branches): {s['mutations_outside_cds']}",
        f"mutation records in skipped codons (scored branches): {s['mutations_in_skipped_codons']}",
        "",
        "codons changed on scored branches, by number of positions differing from the parent codon (1 / 2 / 3):",
        f"  synonymous : {[cells.get(f'syn_{n}nt', 0) for n in (1, 2, 3)]}",
        f"  amino acid changing : {[cells.get(f'nonsyn_{n}nt', 0) for n in (1, 2, 3)]}",
        "  amino acid changing where a shorter route existed from the parent codon: "
        f"2 nt where 1 would do {cells.get('nonsyn_2nt_min1', 0)}; 3 nt where 1 would do "
        f"{cells.get('nonsyn_3nt_min1', 0)}; 3 nt where 2 would do {cells.get('nonsyn_3nt_min2', 0)}",
        f"  dropped (stop) {cells.get('dropped_stop', 0)}; dropped (ambiguous) {cells.get('dropped_ambiguous', 0)}; "
        f"reverted {cells.get('reverted', 0)}",
        f"codon columns: {len(column)}; columns with an amino-acid change: {int((column['nonsyn_1nt'] + column['nonsyn_2nt'] + column['nonsyn_3nt'] > 0).sum()) if len(column) else 0}",
    ]
    return "\n".join(out) + "\n"
