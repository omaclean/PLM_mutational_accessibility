#!/usr/bin/env python3
"""Check Table SX's higher-order claim: how many of the nine K replacements co-occur.

    python tools/check_sX_corpus_cooccurrence.py --corpus <ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.fas>

The table shipped for Table SX is pairwise: ``k_corpus_leakage_pairs.tsv.gz`` says how
often two derived residues occur in the same corpus sequence, and nothing more. The draft
caption also claims **"No single sequence carries more than five of the nine mutations
together"**, which is an order-9 statement that no shipped table can support and that
``tests/test_sX_corpus_leakage.py`` therefore cannot pin. This script computes it.

It goes back to the same public corpus ``tools/export_sX_corpus_leakage.py`` uses -- the
HA-80 fine-tuning set, NCBI Influenza Virus Database pulled 11 Apr 2024, clustered at 99%
identity and filtered, 8,099 sequences, collection dates 1902 to 2015-02-27 -- and reuses
that module's alignment so the two routes cannot drift apart. The corpus is not shipped
here (4.6 MB of third-party sequence) but is public:

    https://raw.githubusercontent.com/spyros-lytras/plm_entropy/main/\
IAV_HA_pLMs/evotuning/training_data/ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.fas

``--meta`` is optional. The per-sequence carriage count needs only the residues, so the
script runs on the FASTA alone; pass the metadata CSV to additionally break the answer
down by serotype and host.

Exit status is 0 when every pinned expectation holds and 1 otherwise, so this can be run
as a pre-submission gate. What it pins, and where each number comes from:

  8,099 sequences                      Table SX caption, and tests/test_sX_corpus_leakage.py
  all nine replacements present        Table SX caption
  minimum 70 occurrences (HA1:T328A)   Results, "at least 70 of the 8,099 sequences"
  HA1:T328A absent from human H3N2     Results (needs --meta)
  36 unordered pairs, 33 co-occurring  Table SX caption
  the three that never co-occur        Table SX caption
  max carried together                 THE CLAIM UNDER TEST -- printed, and compared
                                       against --expect-max (default 5)
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Imported bare rather than as ``tools.export_sX_corpus_leakage``, which is how
# tests/ reaches it: the two spellings would otherwise load the module twice and
# REPLACEMENTS would be two objects rather than one.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_sX_corpus_leakage import (  # noqa: E402
    NO_STATE,
    REPLACEMENTS,
    backbone_protein,
    check_backbone,
    read_fasta,
)

#: Expectations carried by the manuscript, checked rather than assumed.
EXPECT_SEQUENCES = 8099
EXPECT_MIN_COUNT = 70
EXPECT_MIN_LABEL = "HA1:T328A"
EXPECT_PAIRS = 36
EXPECT_PAIRS_COOCCURRING = 33
EXPECT_NEVER_COOCCUR = frozenset(
    {
        frozenset({"HA1:T135K", "HA1:N158D"}),
        frozenset({"HA1:T135K", "HA1:Q173R"}),
        frozenset({"HA1:Q173R", "HA1:T328A"}),
    }
)
EXPECT_ABSENT_FROM_HUMAN_H3N2 = "HA1:T328A"


def align_states(corpus: Path, protein: str) -> list[dict[str, str]]:
    """One dict per corpus sequence: replacement label -> the residue facing that site.

    Lifted from ``export_sX_corpus_leakage.corpus_states`` minus the metadata join, so
    the alignment settings stay identical: a global BLASTp-scored pairwise alignment
    with end gaps free, because corpus sequences are variously truncated at both ends
    and charging those truncations would drag the far end out of frame.
    """
    from Bio import Align

    aligner = Align.PairwiseAligner(
        scoring="blastp", mode="global", open_gap_score=-11, extend_gap_score=-1
    )
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0

    rows = []
    for name, sequence in read_fasta(corpus):
        alignment = aligner.align(protein, sequence)[0]
        backbone_index, query_index = alignment.indices
        facing = {}
        for b, q in zip(backbone_index, query_index):
            if b >= 0:
                facing[b] = sequence[q] if q >= 0 else "-"
        row = {"name": name}
        row.update(
            {label: facing.get(site - 1, "-") for label, site, _a, _d in REPLACEMENTS}
        )
        rows.append(row)
    if not rows:
        raise ValueError(f"no sequences read from {corpus}")
    return rows


def carried(row: dict[str, str]) -> frozenset[str]:
    """The replacements whose *derived* residue this sequence carries.

    A site with no state (gap or ambiguity) is not carriage. That makes the count a
    floor per sequence, which is the conservative direction for a claim phrased as
    "no sequence carries more than N".
    """
    return frozenset(
        label
        for label, _site, _ancestral, derived in REPLACEMENTS
        if row[label] == derived and row[label] not in NO_STATE
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="HA-80 corpus FASTA")
    parser.add_argument("--meta", type=Path, help="optional metadata CSV, for the host/serotype breakdown")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/roots/h3n2_13lineage/J.2.nt.fa"),
        help="J.2 root, whose translated precursor defines the nine sites",
    )
    parser.add_argument(
        "--expect-max",
        type=int,
        default=5,
        help="the manuscript's claimed maximum carried together (default 5)",
    )
    args = parser.parse_args(argv)

    protein = backbone_protein(args.root)
    check_backbone(protein)
    print(f"backbone: {len(protein)} aa, all nine ancestral residues confirmed\n")

    rows = align_states(args.corpus, protein)
    sets = [carried(row) for row in rows]
    failures: list[str] = []

    def check(ok: bool, message: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")
        if not ok:
            failures.append(message)

    print(f"corpus: {len(rows)} sequences")
    check(len(rows) == EXPECT_SEQUENCES, f"sequence count is {EXPECT_SEQUENCES}")

    print("\nper-replacement carriage")
    counts = {}
    for label, _site, _ancestral, _derived in REPLACEMENTS:
        counts[label] = sum(1 for s in sets if label in s)
        print(f"    {label:<12} {counts[label]:>5}")
    check(all(n > 0 for n in counts.values()), "all nine replacements occur in the corpus")
    low = min(counts, key=counts.get)
    check(
        counts[low] >= EXPECT_MIN_COUNT,
        f"rarest replacement occurs at least {EXPECT_MIN_COUNT} times "
        f"(rarest is {low} at {counts[low]})",
    )
    check(low == EXPECT_MIN_LABEL, f"the rarest replacement is {EXPECT_MIN_LABEL}")

    print("\npairwise co-occurrence")
    pairs = list(itertools.combinations([r[0] for r in REPLACEMENTS], 2))
    never = {frozenset(p) for p in pairs if not any(set(p) <= s for s in sets)}
    check(len(pairs) == EXPECT_PAIRS, f"{EXPECT_PAIRS} unordered pairs")
    check(
        len(pairs) - len(never) == EXPECT_PAIRS_COOCCURRING,
        f"{EXPECT_PAIRS_COOCCURRING} pairs co-occur in at least one sequence",
    )
    for pair in sorted(never, key=sorted):
        print(f"    never together: {' + '.join(sorted(pair))}")
    check(never == EXPECT_NEVER_COOCCUR, "the never-co-occurring pairs are the three named")

    print("\nhigher-order carriage  <-- the claim under test")
    histogram = {k: sum(1 for s in sets if len(s) == k) for k in range(len(REPLACEMENTS) + 1)}
    for k, n in histogram.items():
        if n:
            print(f"    {k} of 9 carried: {n:>5} sequence{'s' if n != 1 else ''}")
    observed_max = max(len(s) for s in sets)
    print(f"\n    maximum carried together: {observed_max}")
    check(
        observed_max <= args.expect_max,
        f'"no single sequence carries more than {args.expect_max} of the nine together"',
    )

    top = sorted(((len(s), rows[i]["name"], s) for i, s in enumerate(sets)), reverse=True)[:5]
    print("\n    the five sequences carrying most:")
    for size, name, s in top:
        print(f"      {name:<14} {size}  {', '.join(sorted(s))}")

    if args.meta:
        import csv

        print("\nhuman H3N2 subset")
        records = {r["name"]: r for r in csv.DictReader(args.meta.open())}
        human = [
            s
            for row, s in zip(rows, sets)
            if records.get(row["name"], {}).get("serotype") == "H3N2"
            and records.get(row["name"], {}).get("host") == "Human"
        ]
        print(f"    {len(human)} sequences")
        absent = EXPECT_ABSENT_FROM_HUMAN_H3N2
        check(
            not any(absent in s for s in human),
            f"{absent} does not occur in human H3N2 corpus sequences",
        )
        if human:
            print(f"    maximum carried together in human H3N2: {max(len(s) for s in human)}")
    else:
        print("\nhuman H3N2 subset skipped (pass --meta to include it)")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
