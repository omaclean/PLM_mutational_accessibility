"""Count how many times each observed amino-acid replacement arose, per lineage.

    python tools/export_homoplasy.py \
        --nwk <export>/seg4.nwk --paths <export>/seg4_paths.txt --vcf <export>/seg4_ref.vcf \
        --ref <export>/ref_DQ864721.ungapped.fa \
        --accessions dev/lineage_panel/clade_accessions.tsv \
        --roots data/roots/h3n2_13lineage \
        --columns results/s11_h3n2_ha_tree/column_codons.tsv.gz \
        --observed results/h3n2_13lineage/replacement_observed.tsv.gz \
        --out-dir results/h3n2_13lineage

The tree inputs come from one ``matUtils extract`` on the segment 4 mutation-annotated
tree, run once over the **whole** tree rather than per lineage::

    matUtils extract -i <seg4 usher.pb> -t seg4.nwk -A seg4_paths.txt -v seg4_ref.vcf -n

Pruning to one lineage's tips first would be wrong for this question. ``matUtils`` collapses
the paths it prunes away and merges their mutations onto the branch that survives, so two
changes at one codon on two different branches can become one branch carrying both -- which
is exactly the distinction being counted. Walking the full tree and descending only into
branches that lead to a lineage member (``n_tip``) keeps every branch as UShER placed it.

``--ref`` is the first record of the alignment UShER was built from, with its gap columns
removed. For the segment 4 tree that is ``DQ864721``, which is **not an H3**: its codon
index is not an H3 residue number, which is why ``--columns`` is needed to translate it.
Every one of the reference's bases is checked against the VCF before anything is counted;
a disagreement means the wrong reference or the wrong coordinate system, and stops the run.

**Eight of the thirteen lineages are paraphyletic, and that is by construction.** A sequence
is assigned to the most derived clade it falls in, so ``E`` means "clade E but not E.2", and
``J.2`` means "J.2 but not J.2.4 or K". The common ancestor of such a set is the whole clade's
ancestor, which is why ``n_mrca_tips`` is far larger than the lineage for E, G.1, G.1.1, G.1.3,
G.2, J, J.2 and J.2.4 -- exactly the eight with a descendant lineage in the panel -- and close
to it for the five terminal ones (C.1, D, E.2, G.2.2, K). It is not a misassignment, and it
does not widen the walk: ``walk_branches`` descends only into children with a lineage member
below them, so a branch leading solely into a descendant lineage is never scored. A branch
ancestral to both the lineage and a descendant lineage *is* scored, which is right -- the
lineage's own sequences carry what happened on it.

Two tables are written:

``homoplasy_by_lineage.tsv.gz``  one row per lineage, the counts Table SZ reports.
``homoplasy_replacements.tsv.gz``  one row per placed replacement, with its origin count,
                                   its distinct codon routes, and whether the observed
                                   table marks it present. This is the table to read when
                                   a lineage's number looks surprising.

Neither holds an accession or a sequence: the per-lineage tip lists are inputs, not output.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mutacc import branch_codons as bc
from mutacc import homoplasy as hp
from tools.export_base import Writer, sha256

BY_LINEAGE = "homoplasy_by_lineage.tsv.gz"
REPLACEMENTS = "homoplasy_replacements.tsv.gz"


def read_accessions(path: Path) -> dict[str, list[str]]:
    with Path(path).open() as handle:
        rows = [row for row in csv.reader(handle, delimiter="\t") if len(row) >= 2]
    by: dict[str, list[str]] = collections.defaultdict(list)
    seen: set[str] = set()
    for accession, lineage in ((row[0], row[1]) for row in rows):
        if accession in seen:
            raise ValueError(f"{path}: {accession} appears in more than one lineage")
        seen.add(accession)
        by[lineage].append(accession)
    return dict(by)


def read_roots(directory: Path) -> dict[str, str]:
    """``{lineage: root protein}`` from the shipped nucleotide root records."""
    from Bio.Seq import Seq
    roots = {}
    for fasta in sorted(Path(directory).glob("*.nt.fa")):
        lineage = fasta.name[: -len(".nt.fa")]
        nt = "".join(line.strip() for line in fasta.read_text().splitlines()[1:]).upper()
        if len(nt) % 3:
            raise ValueError(f"{fasta}: {len(nt)} bases is not a multiple of 3")
        protein = str(Seq(nt).translate())
        if not protein.endswith("*") or protein.count("*") != 1:
            raise ValueError(f"{fasta}: the root does not read as one open reading frame")
        roots[lineage] = protein[:-1]
    if not roots:
        raise ValueError(f"{directory}: no *.nt.fa root records")
    lengths = {len(p) for p in roots.values()}
    if len(lengths) != 1:
        raise ValueError(f"the roots are not all one length: {sorted(lengths)}")
    return roots


def build(nwk: Path, paths: Path, vcf: Path, ref_fasta: Path, accessions: Path,
          roots_dir: Path, columns: Path, observed: Path):
    ref = bc.ungap(bc.first_fasta_record(Path(ref_fasta).read_text().splitlines())[1]).upper()
    cds = bc.check_frame(ref)
    checked = bc.check_reference_against_vcf(ref, Path(vcf).read_text().splitlines())
    print(f"reference record {len(ref)} bases, coding region {cds.start}..{cds.end} "
          f"({cds.n_codons} codons), {checked} VCF sites agree")

    numbering = hp.numbering_from_column_table(pd.read_csv(columns, sep="\t", comment="#"))
    covered = sorted(numbering.values())
    print(f"{len(numbering)} of {cds.n_codons} reference codons carry a root-protein "
          f"position; {len(covered)} of the 566 positions are covered")

    tree = bc.parse_newick(Path(nwk).read_text())
    muts = bc.parse_mutation_paths(Path(paths).read_text().splitlines(), tree)
    print(f"tree: {len(tree.children):,} nodes, {len(tree.tips):,} tips, "
          f"{sum(len(v) for v in muts.values()):,} mutation records")

    roots = read_roots(roots_dir)
    by_lineage = read_accessions(accessions)
    seen_table = pd.read_csv(observed, sep="\t")

    per_lineage, replacement_rows, tree_tips, diagnostics = {}, [], {}, []
    for lineage in sorted(roots):
        members = by_lineage.get(lineage)
        if not members:
            raise ValueError(f"{lineage}: no accessions, but a root record ships for it")
        tips = set(members) & tree.tips
        if not tips:
            raise ValueError(f"{lineage}: none of its {len(members)} accessions are in the tree")
        n_tip = bc.count_below(tree, tips)
        size = bc.count_below(tree, tree.tips)
        start = bc.mrca(tree, tips, n_tip)
        result = bc.walk_branches(tree, muts, ref, cds, start, n_tip,
                                  max_nt=None, include_terminal=True)
        frame = hp.origins_frame(result, numbering, roots[lineage], n_tip)
        per_lineage[lineage] = frame
        tree_tips[lineage] = len(tips)
        frame_out = frame.copy()
        frame_out.insert(0, "lineage", lineage)
        replacement_rows.append(frame_out)
        diagnostics.append({
            "lineage": lineage,
            "n_accessions": len(members),
            "n_mrca_tips": size[start],
            "n_branches": result.stats["branches_walked"],
        })
        print(f"  {lineage:6s} {len(tips):6,} tips  MRCA {start:>12s} holds "
              f"{size[start]:7,} tips ({len(tips) / size[start]:.3f} of them this lineage)  "
              f"{result.stats['branches_walked']:7,} branches  "
              f"{len(frame):5,} replacements placed")

    summary = hp.summarise(per_lineage, seen_table, covered, tree_tips)
    diagnostic_frame = pd.DataFrame(diagnostics)
    summary = summary.merge(diagnostic_frame[["lineage", "n_accessions", "n_mrca_tips",
                                              "n_branches"]],
                            on="lineage", validate="one_to_one")
    # Rarefy the same replacements the percentage above is taken over, or the smallest
    # lineage would not come back unchanged and the two columns would not be comparable.
    summary = summary.merge(
        hp.rarefy_frame(hp.placed_frames(per_lineage, seen_table, covered), tree_tips),
        on="lineage", validate="one_to_one")

    replacements = pd.concat(replacement_rows, ignore_index=True)
    present = {(l, p, a) for l, p, a in
               zip(seen_table.loc[seen_table["obs_present"] == 1, "lineage"],
                   seen_table.loc[seen_table["obs_present"] == 1, "position"],
                   seen_table.loc[seen_table["obs_present"] == 1, "aa"])}
    replacements["observed"] = [int((l, p, a) in present) for l, p, a in
                                zip(replacements["lineage"], replacements["position"],
                                    replacements["aa"])]
    return summary, replacements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nwk", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--vcf", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--accessions", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tree-name", default="seg4 treepatch repair usher.pb 24 Sep 2026",
                        help="what to record in the MANIFEST as the tree these came from")
    args = parser.parse_args(argv)

    summary, replacements = build(args.nwk, args.paths, args.vcf, args.ref, args.accessions,
                                  args.roots, args.columns, args.observed)
    print()
    print(summary.to_string(index=False))

    writer = Writer(args.out_dir)
    source = f"tools/export_homoplasy.py on {args.tree_name}"
    writer.add(BY_LINEAGE, summary, source,
               "Table SZ columns 'Homoplasies' and 'Multiple origins'",
               sha=sha256(args.nwk))
    writer.add(REPLACEMENTS, replacements, source,
               "per-replacement origin counts behind the Table SZ homoplasy columns",
               sha=sha256(args.nwk))
    writer.write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
