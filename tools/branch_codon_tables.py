"""Build the two S11 tables (per branch, per codon column) from a UShER tree.

    python tools/branch_codon_tables.py --pb tree.pb --aln aln.fasta --tips tips.txt --outdir out/

Inputs
  --pb         the UShER tree (.pb). ``matUtils`` must be on PATH; it is run to export the
               tree, the mutations on every branch and a no-genotype VCF.
  --aln        the alignment UShER was built from. Its FIRST record must be the UShER
               reference; only that record is read.
  --tips       target tip names, one per line (the branches walked are those leading to them).
  --all-tips   optional wider group (for example every year) used only to pick the main clade.

Instead of --pb, ``--nwk`` and ``--paths`` (and ``--vcf`` to check the reference) take the files
``matUtils extract -t`` / ``-A`` / ``-v ... -n`` wrote, so a run does not need matUtils.

Outputs in --outdir: ``long_range_nodes.tsv.gz`` and ``long_range_changes.tsv.gz`` (the oldest scored node
against each newest internal node: amino-acid differences by the fewest nucleotide changes they needed, and the
distinct changes seen), ``branch_codons.tsv.gz`` (one row per branch with fewer than
--max-branch-nt changes), ``column_codons.tsv.gz`` (one row per codon of the coding region),
``summary.txt``, and with --dump-events ``events.tsv.gz`` (every branch-codon event; large).
Tables begin with ``# key: value`` provenance lines (file names only, no paths). Column
definitions are in ``mutacc/branch_codons.py`` (COLUMN_DOCS) and ``figures/README.md``.

Threads: matUtils gets --threads (default 2, at most 14). Nothing else is parallel.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from mutacc import branch_codons as bc  # noqa: E402

MAX_THREADS = 14


def run_matutils(pb: Path, workdir: Path, threads: int) -> tuple[Path, Path, Path]:
    """Export tree, per-node mutations and reference bases with matUtils (skipped if present)."""
    workdir.mkdir(parents=True, exist_ok=True)
    base = ["matUtils", "extract", "-i", str(pb), "-T", str(threads), "-d", str(workdir)]
    nwk, paths, vcf = workdir / "pb.nwk", workdir / "pb_paths.txt", workdir / "pb_ref.vcf"
    try:
        if not (nwk.exists() and paths.exists()):
            subprocess.run(base + ["-t", nwk.name, "-A", paths.name], check=True, capture_output=True)
        if not vcf.exists():
            subprocess.run(base + ["-v", vcf.name, "-n"], check=True, capture_output=True)
    except FileNotFoundError:
        raise SystemExit("matUtils is not on PATH; pass --nwk/--paths/--vcf instead of --pb") from None
    return nwk, paths, vcf


def read_names(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pb", type=Path, help="UShER tree; matUtils is run on it")
    ap.add_argument("--nwk", type=Path, help="Newick from matUtils extract -t (instead of --pb)")
    ap.add_argument("--paths", type=Path, help="mutations per node from matUtils extract -A (instead of --pb)")
    ap.add_argument("--vcf", type=Path, help="no-genotype VCF from matUtils extract -v -n (checks the reference)")
    ap.add_argument("--aln", type=Path, required=True, help="alignment whose FIRST record is the UShER reference")
    ap.add_argument("--tips", type=Path, required=True)
    ap.add_argument("--all-tips", type=Path)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--max-branch-nt", type=int, default=10,
                    help="keep branches with STRICTLY FEWER nucleotide changes than this; 0 = no limit (default 10)")
    ap.add_argument("--include-terminal", action="store_true", help="also score terminal (tip) branches")
    ap.add_argument("--purity", type=float, default=0.95, help="main-clade purity (default 0.95)")
    ap.add_argument("--cds-start", type=int, default=1,
                    help="first base of the coding region along the reference record, 1-based (default 1)")
    ap.add_argument("--cds-end", type=int, default=None, help="last base, inclusive (default: end of the record)")
    ap.add_argument("--numbering-offset", type=int, default=None,
                    help="add a site_number = codon - OFFSET column (for example the signal-peptide length for "
                         "mature numbering). Not applied unless given; check it against the reference.")
    ap.add_argument("--numbering-table", type=Path, default=None,
                    help="TSV with columns codon and site_number (empty where the reference codon has no residue in "
                         "the protein of interest), from tools/h3_numbering_from_db.py. Adds a site_number column; "
                         "use it instead of --numbering-offset when the reference is a different subtype.")
    ap.add_argument("--skip-unnumbered", action="store_true",
                    help="with --numbering-table: leave out codons that have no site_number, i.e. columns where the "
                         "protein of interest has no residue. Their mutations still change the tracked sequence.")
    ap.add_argument("--baseline", default=None,
                    help="tip (or node) whose reconstructed sequence the long-range row compares against, "
                         "instead of the walk's start node. The start node's reconstruction is largely the "
                         "reference record wherever sites were never sequenced, so when the reference is a "
                         "different subtype it is not a usable ancestor. Give a real dated accession.")
    ap.add_argument("--require-all-tips", action="store_true", help="fail if any --tips name is not in the tree")
    ap.add_argument("--note", default=None,
                    help="free single-line note stored in the table header (a synthetic input must say so here)")
    ap.add_argument("--dump-events", action="store_true", help="also write events.tsv.gz")
    ap.add_argument("--threads", type=int, default=2, help=f"matUtils threads, at most {MAX_THREADS}")
    return ap


def run(args: argparse.Namespace) -> dict:
    if args.threads < 1 or args.threads > MAX_THREADS:
        raise SystemExit(f"--threads must be between 1 and {MAX_THREADS}")
    if args.max_branch_nt < 0:
        raise SystemExit("--max-branch-nt must be 0 (no limit) or positive")
    if args.pb:
        nwk, paths, vcf = run_matutils(args.pb, args.outdir / "matutils_export", args.threads)
    elif args.nwk and args.paths:
        nwk, paths, vcf = args.nwk, args.paths, args.vcf
    else:
        raise SystemExit("give --pb, or both --nwk and --paths")

    tree = bc.parse_newick(nwk.read_text())
    muts = bc.parse_mutation_paths(paths.read_text().splitlines(), tree)
    with open(args.aln) as fh:
        ref_name, ref_aln = bc.first_fasta_record(fh)
    ref = bc.ungap(ref_aln)
    cds = bc.check_frame(ref, args.cds_start, args.cds_end)
    n_vcf = bc.check_reference_against_vcf(ref, vcf.read_text().splitlines()) if vcf else None

    in_tree = tree.tips
    tips_all = read_names(args.tips)
    tips = tips_all & in_tree
    if not tips:
        raise SystemExit("none of the --tips names is a tip of the tree")
    if args.require_all_tips and tips != tips_all:
        raise SystemExit(f"{len(tips_all - tips)} --tips names are not in the tree (first: {sorted(tips_all - tips)[0]})")
    wide = (read_names(args.all_tips) & in_tree) if args.all_tips else tips
    wide |= tips
    clade, start, size, n_all, n_tip = bc.main_clade_and_start(tree, wide, tips, args.purity)

    max_nt = args.max_branch_nt or None
    num = None
    skip: set[int] = set()
    if args.numbering_table:
        num = pd.read_csv(args.numbering_table, sep="\t")
        if len(num) != cds.n_codons or not (num["codon"].to_numpy() == range(1, cds.n_codons + 1)).all():
            raise SystemExit("--numbering-table does not list the same codons as the coding region")
        if args.skip_unnumbered:
            skip = set(num.loc[num["site_number"].isna(), "codon"].astype(int))
    elif args.skip_unnumbered:
        raise SystemExit("--skip-unnumbered needs --numbering-table")
    res = bc.walk_branches(tree, muts, ref, cds, start, n_tip, max_nt=max_nt, include_terminal=args.include_terminal,
                           skip_codons=skip)
    if args.baseline and args.baseline not in tree.children:
        raise SystemExit(f"--baseline {args.baseline} is not a tip or node of the tree")
    lr = bc.compare_to_start(tree, muts, ref, cds, start, n_tip, skip_codons=skip,
                             baseline=args.baseline)
    branch = bc.branch_table(res, include_terminal=args.include_terminal)
    column = bc.column_table(res, ref, cds, numbering_offset=args.numbering_offset)
    if num is not None:
        column.insert(1, "site_number", num["site_number"].astype("Int64"))
        if "consensus_aa" in num.columns:       # the residue of the protein of interest, not of the reference
            column.insert(2, "consensus_aa", num["consensus_aa"].fillna(""))
    if branch.empty:
        raise SystemExit("no branch passed the filters; check --tips, --max-branch-nt and --include-terminal")

    prov = {
        "tool": "tools/branch_codon_tables.py",
        "tree": (args.pb or args.nwk).name,
        "reference_record": ref_name,
        "target_tips_in_tree": len(tips),
        "target_tips_not_in_tree": len(tips_all - tips),
        "max_branch_nt": f"{args.max_branch_nt} (strictly fewer kept)" if args.max_branch_nt else "no limit",
        "include_terminal": int(args.include_terminal),
        "coding_region": f"{cds.start}..{cds.end} of the reference record",
        "codon_numbering": "along the reference record's coding region (not alignment columns)",
    }
    if skip:
        prov["skipped_codons"] = f"{len(skip)} codons with no residue in the protein of interest"
    if args.numbering_table:
        prov["site_number"] = f"mature protein numbering from {args.numbering_table.name}"
    prov["long_range_baseline"] = args.baseline or f"{start} (the walk's start node)"
    if args.note:
        prov["note"] = args.note
    args.outdir.mkdir(parents=True, exist_ok=True)
    bc.write_table(branch, args.outdir / "branch_codons.tsv.gz", prov)
    bc.write_table(column, args.outdir / "column_codons.tsv.gz", prov)
    changes = lr.changes
    if num is not None:
        changes = changes.merge(num[["codon", "site_number"]], on="codon", how="left")
        changes["site_number"] = changes["site_number"].astype("Int64")
    bc.write_table(lr.per_node, args.outdir / "long_range_nodes.tsv.gz", prov)
    bc.write_table(changes, args.outdir / "long_range_changes.tsv.gz", prov)
    bc.write_table(lr.baseline, args.outdir / "baseline_codons.tsv.gz", prov)
    if args.dump_events:
        bc.write_table(bc.events_frame(res, include_terminal=args.include_terminal,
                                       numbering_offset=args.numbering_offset),
                       args.outdir / "events.tsv.gz", prov)
    header = [
        f"target tips in tree: {len(tips)} (not in tree: {len(tips_all - tips)})",
        f"main clade {clade}: {size[clade]} tips, {n_all[clade] / size[clade]:.1%} in the clade-defining set; "
        f"walk starts at {start}; {len(tips) - n_tip[start]} target tips outside it are excluded",
        f"reference record {ref_name}: {len(ref)} bases, coding region {cds.start}..{cds.end} ({cds.n_codons} codons), "
        + (f"checked against {n_vcf} VCF sites" if n_vcf else "NOT checked against a VCF (none given)"),
    ]
    pn = lr.per_node
    header.append(
        f"long comparison: baseline {args.baseline or start} (walk from {start}) against {lr.n_nodes} newest internal nodes; per node on average "
        f"{pn['nonsyn'].mean():.1f} amino-acid differences, of which {pn['min1'].mean():.1f} need 1 nucleotide change, "
        f"{pn['min2'].mean():.1f} need 2 and {pn['min3'].mean():.1f} need 3 (fewest changes from the start codon); "
        f"{len(lr.changes)} distinct (site, new amino acid) changes in all")
    (args.outdir / "summary.txt").write_text(bc.summary_text(res, branch, column, header))
    return {"branch": branch, "column": column, "result": res, "start": start}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = run(args)
    print((args.outdir / "summary.txt").read_text())
    print(f"wrote {len(out['branch'])} branch rows and {len(out['column'])} codon rows to {args.outdir.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
