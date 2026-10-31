"""Mature H3 residue numbers for the codons of the S11 reference record.

    export IAV_DB=<...>/IAV_DB_genbank_n_gisaid_rebuild/db_build/iav-db.db
    python tools/h3_numbering_from_db.py --db $IAV_DB --ref-fasta ref_aligned.fasta \
        --tips tips.txt --out numbering.tsv

The S11 reference record (the UShER reference) is not an H3 sequence, and it has gaps against the
alignment, so its codon index is not an H3 number. Every sequence in the database is aligned onto
one shared set of columns, so this reads the aligned rows of a fixed random sample of the tree's
GenBank tips, takes the majority codon per column, and counts the columns where the sample has a
residue. Counting starts at 1 on the first column, which holds the first residue of the mature
protein (checked below, not assumed).

Checks that stop the run
  * the sample's consensus must carry the conserved receptor-binding residues Y98, W153, H183, Y195
    (mature H3 numbering) at the numbers this counting gives them;
  * every row read must have the same width as the reference's aligned row.

Output: one row per codon of the reference record: ``codon`` (record codon index), ``alignment_codon``,
``site_number`` (mature H3 number, empty where the sample has a gap, i.e. the reference carries a
residue H3 lacks), ``consensus_aa``, ``ref_aa``. Database reads are single-row lookups on the indexed
key, read-only.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq

LANDMARKS = {98: "Y", 153: "W", 183: "H", 195: "Y"}
GAP = set("-.")


def codon_columns(aligned: str) -> list[str]:
    return [aligned[i:i + 3] for i in range(0, len(aligned) - len(aligned) % 3, 3)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--ref-fasta", required=True, type=Path, help="aligned row of the UShER reference, one record")
    ap.add_argument("--tips", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--segment", default="4")
    ap.add_argument("--n", type=int, default=300, help="GenBank tips sampled (default 300)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    ref = "".join(args.ref_fasta.read_text().split("\n")[1:]).strip().upper()
    ref_cols = codon_columns(ref)
    tips = [t for t in args.tips.read_text().split() if not t.startswith("EPI")]
    rng = random.Random(args.seed)
    sample = rng.sample(tips, min(args.n, len(tips)))

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=60)
    rows = []
    for acc in sample:
        hit = con.execute("select alignment from sequence_alignment where primary_accession=? and segment=?",
                          (acc, args.segment)).fetchone()
        if hit is None:
            continue
        aln = hit[0].upper()
        if len(aln) != len(ref):
            sys.exit(f"{acc}: aligned row is {len(aln)} wide, reference is {len(ref)}")
        rows.append(codon_columns(aln))
    if len(rows) < 0.8 * len(sample):
        sys.exit(f"only {len(rows)} of {len(sample)} sampled tips have an aligned row")

    consensus, present = [], []
    for j in range(len(ref_cols)):
        codons = [r[j] for r in rows]
        ok = [c for c in codons if not (set(c) & GAP) and set(c) <= set("ACGT")]
        gapped = sum(1 for c in codons if set(c) & GAP)
        has = gapped < 0.5 * len(codons) and bool(ok)
        present.append(has)
        consensus.append(Counter(ok).most_common(1)[0][0] if has else None)

    number, k = {}, 0
    for j, has in enumerate(present):
        if has:
            k += 1
            number[j] = k
    aa = {j: str(Seq(c).translate()) for j, c in enumerate(consensus) if c}
    by_number = {number[j]: aa[j] for j in number}
    bad = {n: (want, by_number.get(n)) for n, want in LANDMARKS.items() if by_number.get(n) != want}
    if bad:
        sys.exit(f"landmark residues not where mature H3 numbering puts them (number: wanted, found): {bad}")

    out, rec_codon, in_codon = [], 0, 0
    for j, c in enumerate(ref_cols):
        if set(c) & GAP:
            if set(c) - GAP:
                sys.exit(f"alignment column {j + 1}: reference codon is partly gapped; frame not preserved")
            continue
        rec_codon += 1
        out.append({"codon": rec_codon, "alignment_codon": j + 1, "site_number": number.get(j),
                    "consensus_aa": aa.get(j), "ref_aa": str(Seq(c).translate())})
    df = pd.DataFrame(out)
    df["site_number"] = df["site_number"].astype("Int64")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"{len(rows)} tips read; {len(number)} H3 residues numbered; "
          f"{int(df['site_number'].isna().sum())} of {len(df)} reference codons have no H3 residue; "
          f"landmarks {sorted(LANDMARKS)} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
