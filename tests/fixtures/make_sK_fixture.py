"""Regenerate tests/fixtures/sK/*.tsv.gz: SYNTHETIC tables that exercise the S11 drawing path.

    python tests/fixtures/make_sK_fixture.py [outdir]

A seeded random tree and random mutations on a random reference, run through the same
``mutacc.branch_codons`` functions the real tables use. The tables carry ``# note: SYNTHETIC``.
They describe nothing about any virus and must never be moved under results/.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mutacc import branch_codons as bc  # noqa: E402

SEED = 12
N_CODONS = 90
N_TIPS = 500
NOTE = "SYNTHETIC FIXTURE: random tree and random mutations, not data"


def make(seed: int = SEED):
    rng = random.Random(seed)
    while True:
        ref = "".join(rng.choice("ACGT") for _ in range(3 * N_CODONS))
        if all(bc.CODON2AA[ref[i:i + 3]] != "*" for i in range(0, len(ref), 3)):
            break
    hot = [rng.randrange(N_CODONS) for _ in range(6)]      # codons that attract repeated changes
    nodes = ["n0"]
    kids = {"n0": []}
    parent = {}
    tips = []
    while len(tips) + sum(1 for n in nodes if kids[n]) < 2 * N_TIPS:    # grow by splitting random tips
        leaves = [n for n in nodes if not kids[n]]
        n = rng.choice(leaves)
        for _ in range(2):
            c = f"n{len(nodes)}"
            nodes.append(c)
            kids[c] = []
            kids[n].append(c)
            parent[c] = n
        if len([m for m in nodes if not kids[m]]) >= N_TIPS:
            break
    seq = {"n0": list(ref)}
    muts = {}
    for n in nodes[1:]:
        s = list(seq[parent[n]])
        recs = []
        k = rng.choice([0, 1, 1, 1, 2, 2, 3, 4, 6, 9, 12])
        for _ in range(k):
            codon = rng.choice(hot) if rng.random() < 0.3 else rng.randrange(N_CODONS)
            p = 3 * codon + rng.randrange(3)
            if rng.random() < 0.45:                          # a second hit in the same codon
                p = 3 * codon + rng.randrange(3)
            to = rng.choice([b for b in "ACGT" if b != s[p]])
            recs.append((s[p], p + 1, to))
            s[p] = to
        if recs:
            muts[n] = recs
        seq[n] = s
    leaves = [n for n in nodes if not kids[n]]
    return ref, nodes, kids, muts, leaves


def newick(kids, node="n0"):
    def rec(n):
        return n if not kids[n] else "(" + ",".join(rec(k) for k in kids[n]) + ")" + n
    return rec(node) + ";"


def tables(seed: int = SEED):
    ref, nodes, kids, muts, leaves = make(seed)
    tree = bc.parse_newick(newick(kids))
    parsed = {n: r for n, r in muts.items()}
    tips = set(leaves)
    cds = bc.check_frame(ref)
    clade, start, _, _, n_tip = bc.main_clade_and_start(tree, tips, tips, 0.95)
    res = bc.walk_branches(tree, parsed, ref, cds, start, n_tip, max_nt=10)
    return bc.branch_table(res), bc.column_table(res, ref, cds)


def main(out: Path) -> None:
    b, c = tables()
    prov = {"tool": "tests/fixtures/make_sK_fixture.py", "note": NOTE, "max_branch_nt": "10 (strictly fewer kept)",
            "include_terminal": 0}
    bc.write_table(b, out / "branch_codons_SYNTHETIC.tsv.gz", prov)
    bc.write_table(c, out / "column_codons_SYNTHETIC.tsv.gz", prov)
    print(f"{len(b)} branches, {len(c)} codons")


if __name__ == "__main__":
    # ``sK`` is this generator's own fixture directory. It read ``s12`` until 20 Oct 2026,
    # which was a directory that never existed: running it with no argument wrote the tables
    # somewhere nothing reads them, and the shipped fixture looked current when it was not.
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "sK")
