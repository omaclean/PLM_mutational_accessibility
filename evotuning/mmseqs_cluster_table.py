#!/usr/bin/env python
"""Cluster FASTA sequences with mmseqs and export per-threshold cluster assignments."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

DEFAULT_CLUSTER_IDENTITIES = (0.90, 0.95, 0.99, 0.9995)
DEFAULT_COVERAGE_MODE = 0
DEFAULT_COVERAGE_THRESHOLD = 0.0  # 0.0 = no coverage requirement (mmseqs default)
DEFAULT_MIN_LENGTH = 900
# The tree builder fills an all-pairs distance matrix in Python, so its cost is
# quadratic in the number of cluster representatives. On a corpus with ~1M
# representatives that never completes, and it runs after the expensive mmseqs
# clustering, so an unguarded attempt throws away the whole job.
MAX_TREE_REPRESENTATIVES = 400
SummaryValue = str | int | float
SummaryRow = dict[str, SummaryValue]


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    sequence_lines: list[str] = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(sequence_lines)))
                name = line[1:].split()[0]
                sequence_lines = []
            else:
                sequence_lines.append(line)

    if name is not None:
        records.append((name, "".join(sequence_lines)))

    return records


def normalise_sequence(sequence: str) -> str:
    cleaned = sequence.strip().upper().replace(" ", "")
    if cleaned.endswith("*"):
        cleaned = cleaned[:-1]
    return cleaned


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds: list[float] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part.endswith("%"):
            threshold = float(part[:-1]) / 100.0
        else:
            threshold = float(part)
            if threshold > 1:
                threshold /= 100.0
        if threshold <= 0 or threshold > 1:
            raise argparse.ArgumentTypeError(f"Invalid clustering threshold: {raw_part}")
        thresholds.append(threshold)

    if not thresholds:
        raise argparse.ArgumentTypeError("At least one clustering threshold is required.")
    return tuple(thresholds)


def threshold_percent_label(threshold: float) -> str:
    percent = threshold * 100
    return f"{percent:.4f}".rstrip("0").rstrip(".")


def threshold_column_name(threshold: float) -> str:
    return f"cluster_{threshold_percent_label(threshold).replace('.', '_')}pct"


def threshold_file_label(threshold: float) -> str:
    return threshold_percent_label(threshold).replace(".", "_")


def write_clean_fasta(records: list[tuple[str, str]], output_path: Path) -> None:
    with output_path.open("w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def filter_records_by_min_length(
    records: list[tuple[str, str]],
    min_length: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (kept, removed) partitioned by sequence length >= min_length."""
    kept = [(name, seq) for name, seq in records if len(seq) >= min_length]
    removed = [(name, seq) for name, seq in records if len(seq) < min_length]
    return kept, removed


def validate_records(records: list[tuple[str, str]]) -> None:
    if not records:
        raise ValueError("No FASTA records were found.")

    # Counter rather than list.count() in a loop: the latter is quadratic and
    # does not return on a million-record corpus.
    name_counts = Counter(name for name, _sequence in records)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        duplicates = ", ".join(duplicate_names[:5])
        raise ValueError(f"FASTA IDs must be unique. Duplicate IDs found: {duplicates}")

    empty_names = [name for name, sequence in records if not sequence]
    if empty_names:
        empties = ", ".join(empty_names[:5])
        raise ValueError(f"Found empty sequences after cleaning for IDs: {empties}")


def parse_cluster_assignments(tsv_path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with tsv_path.open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            representative, member = line.split("\t", maxsplit=1)
            assignments[member] = representative
            assignments.setdefault(representative, representative)
    return assignments


def compute_cluster_sizes(assignments: dict[str, str]) -> list[int]:
    cluster_sizes: dict[str, int] = {}
    for representative in assignments.values():
        cluster_sizes[representative] = cluster_sizes.get(representative, 0) + 1
    return sorted(cluster_sizes.values(), reverse=True)


def sequence_distance(sequence_a: str, sequence_b: str) -> int:
    overlap = min(len(sequence_a), len(sequence_b))
    mismatches = sum(a != b for a, b in zip(sequence_a[:overlap], sequence_b[:overlap]))
    mismatches += abs(len(sequence_a) - len(sequence_b))
    return mismatches


def get_cluster_representative_records(
    records: list[tuple[str, str]],
    assignments: dict[str, str],
) -> list[tuple[str, str]]:
    seen_representatives: set[str] = set()
    representatives: list[tuple[str, str]] = []
    for name, sequence in records:
        representative = assignments[name]
        if representative != name or representative in seen_representatives:
            continue
        seen_representatives.add(representative)
        representatives.append((name, sequence))
    return representatives


def run_mmseqs_clustering(
    input_fasta: Path,
    thresholds: tuple[float, ...],
    coverage_mode: int,
    working_dir: Path,
    min_length: int = DEFAULT_MIN_LENGTH,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[list[tuple[str, str]], dict[float, dict[str, str]], list[tuple[str, str]]]:
    mmseqs_path = shutil.which("mmseqs")
    if not mmseqs_path:
        raise RuntimeError("`mmseqs` is not installed or not on PATH.")

    raw_records = read_fasta(input_fasta)
    cleaned_records = [(name, normalise_sequence(sequence)) for name, sequence in raw_records]

    if min_length > 0:
        cleaned_records, removed_records = filter_records_by_min_length(cleaned_records, min_length)
        print(
            f"[filter] Kept {len(cleaned_records)} sequences >= {min_length} AA; "
            f"removed {len(removed_records)} sequences below the minimum length."
        )
    else:
        removed_records = []

    validate_records(cleaned_records)

    clean_fasta = working_dir / "clean_input.fasta"
    write_clean_fasta(cleaned_records, clean_fasta)

    input_db = working_dir / "input_db"
    subprocess.run([mmseqs_path, "createdb", str(clean_fasta), str(input_db)], check=True)

    expected_ids = {name for name, _sequence in cleaned_records}
    results: dict[float, dict[str, str]] = {}

    for threshold in thresholds:
        label = threshold_file_label(threshold)
        cluster_db = working_dir / f"cluster_{label}"
        tmp_dir = working_dir / f"tmp_{label}"
        tsv_path = working_dir / f"cluster_{label}.tsv"

        subprocess.run(
            [
                mmseqs_path,
                "cluster",
                str(input_db),
                str(cluster_db),
                str(tmp_dir),
                "--min-seq-id",
                str(threshold),
                "--cov-mode",
                str(coverage_mode),
                "-c",
                str(coverage_threshold),
            ],
            check=True,
        )
        subprocess.run(
            [mmseqs_path, "createtsv", str(input_db), str(input_db), str(cluster_db), str(tsv_path)],
            check=True,
        )

        assignments = parse_cluster_assignments(tsv_path)
        missing_ids = sorted(expected_ids - set(assignments))
        unexpected_ids = sorted(set(assignments) - expected_ids)
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise RuntimeError(f"mmseqs did not emit assignments for all sequences at {threshold_percent_label(threshold)}% identity: {preview}")
        if unexpected_ids:
            preview = ", ".join(unexpected_ids[:5])
            raise RuntimeError(f"mmseqs emitted unknown sequence IDs at {threshold_percent_label(threshold)}% identity: {preview}")
        results[threshold] = assignments

    return cleaned_records, results, removed_records


def write_assignment_csv(
    records: list[tuple[str, str]],
    threshold_assignments: dict[float, dict[str, str]],
    output_csv: Path,
) -> None:
    fieldnames = ["fasta_id", *(threshold_column_name(threshold) for threshold in threshold_assignments)]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, _sequence in records:
            row = {"fasta_id": name}
            for threshold, assignments in threshold_assignments.items():
                row[threshold_column_name(threshold)] = assignments[name]
            writer.writerow(row)


def summarise_assignments(threshold_assignments: dict[float, dict[str, str]]) -> list[SummaryRow]:
    summary_rows: list[SummaryRow] = []
    for threshold, assignments in threshold_assignments.items():
        cluster_sizes = compute_cluster_sizes(assignments)
        singleton_clusters = sum(size == 1 for size in cluster_sizes)
        summary_rows.append(
            {
                "threshold": str(threshold),
                "threshold_percent": threshold_percent_label(threshold),
                "cluster_count": len(set(assignments.values())),
                "sequence_count": len(assignments),
                "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "mean_cluster_size": round(statistics.fmean(cluster_sizes), 6) if cluster_sizes else 0.0,
                "median_cluster_size": statistics.median(cluster_sizes) if cluster_sizes else 0.0,
                "singleton_clusters": singleton_clusters,
                "singleton_fraction": round(singleton_clusters / len(cluster_sizes), 6) if cluster_sizes else 0.0,
            }
        )
    return summary_rows


def write_summary_csv(summary_rows: list[SummaryRow], summary_csv: Path) -> None:
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "threshold_percent",
                "cluster_count",
                "sequence_count",
                "largest_cluster_size",
                "mean_cluster_size",
                "median_cluster_size",
                "singleton_clusters",
                "singleton_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def write_cluster_plots(
    threshold_assignments: dict[float, dict[str, str]],
    summary_rows: list[SummaryRow],
    output_dir: Path,
) -> list[Path]:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    thresholds = list(threshold_assignments)
    cluster_sizes_by_threshold = {
        threshold: compute_cluster_sizes(assignments)
        for threshold, assignments in threshold_assignments.items()
    }

    if cluster_sizes_by_threshold:
        figure_count = len(cluster_sizes_by_threshold)
        columns = min(2, figure_count)
        rows = math.ceil(figure_count / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 4.5 * rows))
        axes_list = axes.flat if hasattr(axes, "flat") else [axes]

        for axis, threshold in zip(axes_list, thresholds):
            cluster_sizes = cluster_sizes_by_threshold[threshold]
            bins = range(1, max(cluster_sizes) + 2) if cluster_sizes else [1, 2]
            axis.hist(cluster_sizes, bins=bins, align="left", edgecolor="black", alpha=0.8)
            axis.set_title(f"Cluster sizes at {threshold_percent_label(threshold)}% identity")
            axis.set_xlabel("Sequences per cluster")
            axis.set_ylabel("Number of clusters")
            axis.set_yscale("log")
            axis.grid(True, alpha=0.3)

        for axis in list(axes_list)[len(thresholds):]:
            axis.axis("off")

        fig.tight_layout()
        histogram_path = output_dir / "cluster_size_histograms.png"
        fig.savefig(str(histogram_path), dpi=150)
        plt.close(fig)
        written_paths.append(histogram_path)

        fig, axis = plt.subplots(figsize=(8, 5))
        for threshold in thresholds:
            cluster_sizes = cluster_sizes_by_threshold[threshold]
            ranks = list(range(1, len(cluster_sizes) + 1))
            axis.plot(ranks, cluster_sizes, marker="o", linewidth=1.5, label=f"{threshold_percent_label(threshold)}%")
        axis.set_title("Cluster size rank curves")
        axis.set_xlabel("Cluster rank")
        axis.set_ylabel("Sequences per cluster")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend(title="Identity")
        fig.tight_layout()
        rank_path = output_dir / "cluster_size_rank_curves.png"
        fig.savefig(str(rank_path), dpi=150)
        plt.close(fig)
        written_paths.append(rank_path)

    if summary_rows:
        threshold_labels = [str(row["threshold_percent"]) for row in summary_rows]
        cluster_counts = [int(row["cluster_count"]) for row in summary_rows]
        largest_cluster_sizes = [int(row["largest_cluster_size"]) for row in summary_rows]
        singleton_fractions = [float(row["singleton_fraction"]) for row in summary_rows]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        axes[0].plot(threshold_labels, cluster_counts, marker="o")
        axes[0].set_ylabel("Cluster count")
        axes[0].set_title("Clustering summary across identity thresholds")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(threshold_labels, largest_cluster_sizes, marker="o", color="tab:orange")
        axes[1].set_ylabel("Largest cluster")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(threshold_labels, singleton_fractions, marker="o", color="tab:green")
        axes[2].set_ylabel("Singleton fraction")
        axes[2].set_xlabel("Identity threshold (%)")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        overview_path = output_dir / "clustering_threshold_overview.png"
        fig.savefig(str(overview_path), dpi=150)
        plt.close(fig)
        written_paths.append(overview_path)

    return written_paths


def write_quick_tree_outputs(
    records: list[tuple[str, str]],
    threshold_assignments: dict[float, dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    try:
        phylo = importlib.import_module("Bio.Phylo")
        newick = importlib.import_module("Bio.Phylo.Newick")
        tree_construction = importlib.import_module("Bio.Phylo.TreeConstruction")
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    for threshold, assignments in threshold_assignments.items():
        representatives = get_cluster_representative_records(records, assignments)
        threshold_label = threshold_file_label(threshold)
        newick_path = output_dir / f"tree_{threshold_label}.newick"
        image_path = output_dir / f"tree_{threshold_label}.png"

        if not representatives:
            continue

        if len(representatives) > MAX_TREE_REPRESENTATIVES:
            print(
                f"Skipping tree for {threshold_percent_label(threshold)}% identity: "
                f"{len(representatives)} representatives exceeds the {MAX_TREE_REPRESENTATIVES} "
                "limit for all-pairs distances.",
                flush=True,
            )
            continue

        if len(representatives) == 1:
            tree = newick.Tree(root=newick.Clade(name=representatives[0][0]), rooted=True)
        else:
            names = [name for name, _sequence in representatives]
            matrix: list[list[float]] = []
            for row_index, (_row_name, row_sequence) in enumerate(representatives):
                distances = [0.0]
                for column_index in range(row_index):
                    column_sequence = representatives[column_index][1]
                    distances.append(float(sequence_distance(row_sequence, column_sequence)))
                matrix.append(distances)

            distance_matrix = tree_construction.DistanceMatrix(names=names, matrix=matrix)
            constructor = tree_construction.DistanceTreeConstructor()
            tree = constructor.nj(distance_matrix)

        phylo.write(tree, str(newick_path), "newick")
        written_paths.append(newick_path)

        figure_height = max(4, 0.35 * len(representatives) + 1.5)
        fig = plt.figure(figsize=(10, figure_height))
        axis = fig.add_subplot(1, 1, 1)
        phylo.draw(tree, axes=axis, do_show=False)
        axis.set_title(f"Quick tree at {threshold_percent_label(threshold)}% identity")
        fig.tight_layout()
        fig.savefig(str(image_path), dpi=150)
        plt.close(fig)
        written_paths.append(image_path)

    return written_paths

# python comb_old_new.fa --input-fasta /projects/u6dr/OM/PLM_Out/cluster_outputs/cluster_assignments.csv --output-csv /projects/u6dr/OM/PLM_Outputs/cluster_outputs/cluster_summary.csv --cluster-thresholds 0.9,0.95,0.99,0.9995 --coverage-mode 0
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster FASTA sequences with mmseqs and export cluster assignments.")
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path for the main per-sequence output table. This CSV will contain one row per FASTA ID and one cluster-assignment column per threshold, such as `cluster_90pct`, `cluster_95pct`, `cluster_99pct`, and `cluster_99_95pct`.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path for the summary CSV. This file contains one row per threshold with aggregate statistics such as cluster count, largest cluster size, mean cluster size, and singleton fraction. If omitted, the script writes `<output-csv-stem>_summary.csv` next to `--output-csv`.",
    )
    parser.add_argument(
        "--cluster-thresholds",
        type=parse_thresholds,
        default=DEFAULT_CLUSTER_IDENTITIES,
        help="Comma-separated identity thresholds to run in mmseqs. You can pass decimals like `0.9,0.95,0.99,0.9995`, percentages like `90,95,99,99.95`, or explicit percent strings like `90%%,95%%,99%%,99.95%%`. The script will create one cluster-assignment column, summary row, tree, and set of plots for each threshold.",
    )
    parser.add_argument("--coverage-mode", type=int, default=DEFAULT_COVERAGE_MODE, help="Coverage mode to pass to mmseqs. See mmseqs documentation for details:)")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help=(
            "Minimum fraction of the sequence that must be covered by the alignment for two sequences "
            "to be clustered together (mmseqs -c flag). Default 0.0 = no coverage requirement, so "
            "partial sequences cluster with their full-length relatives as long as identity is met. "
            "Set to e.g. 0.8 to require 80%% coverage (behaviour depends on --coverage-mode)."
        ),
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help=(
            f"Minimum sequence length in amino acids (default: {DEFAULT_MIN_LENGTH}). "
            "Sequences shorter than this are excluded from clustering and the output CSV. "
            "Set to 0 to disable filtering."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    input_fasta = args.input_fasta.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    summary_csv = args.summary_csv.expanduser().resolve() if args.summary_csv else output_csv.with_name(f"{output_csv.stem}_summary.csv")
    min_length = args.min_length

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="mmseqs_cluster_table_", dir=str(output_csv.parent)) as tmp_dir_name:
        working_dir = Path(tmp_dir_name)
        records, threshold_assignments, removed_records = run_mmseqs_clustering(
            input_fasta=input_fasta,
            thresholds=args.cluster_thresholds,
            coverage_mode=args.coverage_mode,
            working_dir=working_dir,
            min_length=min_length,
            coverage_threshold=args.coverage_threshold,
        )

    # Write filtered FASTA (kept sequences only) next to the input file
    if min_length > 0:
        filtered_fasta = input_fasta.with_name(input_fasta.stem + "_filtered.fa")
        write_clean_fasta(records, filtered_fasta)
        print(f"Wrote filtered FASTA ({len(records)} sequences) to {filtered_fasta}")

    write_assignment_csv(records, threshold_assignments, output_csv)
    summary_rows = summarise_assignments(threshold_assignments)
    write_summary_csv(summary_rows, summary_csv)
    plot_dir = output_csv.with_name(f"{output_csv.stem}_plots")
    written_plot_paths = write_cluster_plots(threshold_assignments, summary_rows, plot_dir)
    tree_dir = output_csv.with_name(f"{output_csv.stem}_trees")
    written_tree_paths = write_quick_tree_outputs(records, threshold_assignments, tree_dir)

    print(f"Wrote assignments to {output_csv}")
    print(f"Wrote summary to {summary_csv}")
    if written_plot_paths:
        print(f"Wrote plots to {plot_dir}")
    if written_tree_paths:
        print(f"Wrote trees to {tree_dir}")
    for row in summary_rows:
        print(f"{row['threshold_percent']}% identity: {row['cluster_count']} clusters across {row['sequence_count']} sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
