"""Bring the S3 viral-panel scores and sequences out of the exploration archive.

    python tools/export_viral_panel.py --source ../tmp_explore/IAV_vs_SC2_criss_cross \
                                       --cellular ../tmp_explore/cellular_phage_panel

``--cellular`` adds the eukaryotic, bacterial and phage comparison panel, fetched and scored the same
way, so the viral scores can be read against something. It is optional: without it the export is the
original viral-only panel and the figure is unchanged.

Reads the panel the scorer used (``viral_panel.faa`` and ``viral_panel.tsv``, pulled from UniProt) and the
two all-site score files (``viral_panel_allsites_<model>.tsv.gz``, one row per masked position). Writes

* ``data/viral_panel/viral_panel.faa`` and ``viral_panel_sequences.tsv``: every amino-acid sequence
  scored, with accession, protein name, organism and length. These are public UniProt records.
* ``results/viral_panel/viral_panel_sites.tsv.gz``: the per-position scores, as scored.
* ``results/viral_panel/viral_panel_summary.tsv.gz``: one row per (model, protein) with the headline
  number (mean probability given to the true amino acid), the top-1 hit rate, and a composition baseline.

The composition baseline is what a model would score by ignoring the sequence context entirely and
guessing amino acids in the same proportions as the protein itself contains: the sum over amino acids
of the squared frequency. It is the honest floor for "how much does context help".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_base import Writer, sha256  # noqa: E402

CLEAN = Path(__file__).resolve().parents[1]
MODEL_FILES = {"esm2": ("ESM-2 650M base", "base"), "esmc": ("ESM-C 600M base", "base_second")}
#: Display order, top to bottom. The three viral groups are the original S3 panel; the three
#: cellular/phage groups are the comparison added beside it, so a reader can see whether a viral
#: score is low for a protein or low for the model. They come last so the viral panel still reads
#: as the subject of the figure.
GROUPS = ("Influenza viruses", "Coronaviruses", "Other viruses", "Eukaryotes", "Bacteria", "Phages")
CELLULAR_GROUPS = ("Eukaryotes", "Bacteria", "Phages")
PREFIX = {"IAV ": "Influenza A ", "IBV ": "Influenza B ", "ICV ": "Influenza C "}


def display_label(label: str) -> str:
    for short, long in PREFIX.items():
        if label.startswith(short):
            return long + label[len(short):]
    return label


def group_of(organism: str) -> str:
    low = organism.lower()
    if low.startswith("influenza"):
        return GROUPS[0]
    if "coronavirus" in low:
        return GROUPS[1]
    return GROUPS[2]


def read_faa(path: Path) -> dict[str, str]:
    out = {}
    for block in path.read_text().split(">")[1:]:
        head, *rest = block.strip().split("\n")
        out[head.split("|")[0].replace("_", " ")] = "".join(rest)
    return out


def read_panel(src: Path, stem: str, group_from_table: bool) -> pd.DataFrame:
    """One panel's table with its sequences attached.

    ``group_from_table`` keeps the group the fetch script assigned (the cellular and phage panel,
    where the organism string does not tell you whether a host is a eukaryote). The viral panel
    instead derives its group from the organism, as it always has.
    """
    seqs = read_faa(src / f"{stem}.faa")
    meta = pd.read_csv(src / f"{stem}.tsv", sep="\t")
    meta["sequence"] = meta["label"].map(seqs)
    if meta["sequence"].isna().any():
        raise SystemExit(f"a protein in {stem}.tsv has no sequence in {stem}.faa")
    meta["group"] = meta["group"] if group_from_table else meta["organism"].map(group_of)
    unknown = set(meta["group"]) - set(GROUPS)
    if unknown:
        raise SystemExit(f"{stem}: group(s) {sorted(unknown)} have no place in the display order")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, type=Path, help="directory holding the panel and score files")
    ap.add_argument("--cellular", type=Path, default=None,
                    help="directory holding the cellular/phage comparison panel (cellular_panel.*); "
                         "omit it and only the viral panel is exported")
    args = ap.parse_args(argv)
    src = args.source.resolve()
    cell = args.cellular.resolve() if args.cellular else None

    panels = [(src, "viral_panel", False)]
    if cell:
        panels.append((cell, "cellular_panel", True))
    meta = pd.concat([read_panel(d, stem, from_table) for d, stem, from_table in panels], ignore_index=True)
    if meta["label"].duplicated().any() or meta["accession"].duplicated().any():
        raise SystemExit("the two panels share a label or an accession")
    # The fetch script keeps only the 20 standard amino acids, so a UniProt entry holding unknown residues (X) is
    # scored with those residues removed. Record how many, and refuse anything else (an unexplained difference).
    meta = meta.rename(columns={"length": "length_uniprot"})
    meta["length"] = meta["sequence"].str.len()
    meta["unknown_residues_removed"] = meta["length_uniprot"] - meta["length"]
    if (meta["unknown_residues_removed"] < 0).any():
        raise SystemExit("a sequence is longer than the length UniProt reported")
    if set("".join(meta["sequence"])) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise SystemExit("a sequence holds something other than the 20 standard amino acids")
    for r in meta[meta["unknown_residues_removed"] > 0].itertuples():
        print(f"note: {r.label} ({r.accession}): {r.unknown_residues_removed} unknown residue(s) removed before scoring")
    meta["display_label"] = meta["label"].map(display_label)
    meta["group_order"] = meta["group"].map(GROUPS.index)

    data_dir = CLEAN / "data" / "viral_panel"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "viral_panel.faa").write_text(
        "".join((d / f"{stem}.faa").read_text() for d, stem, _ in panels))
    meta[["label", "display_label", "group", "accession", "protein_name", "organism", "length",
          "length_uniprot", "unknown_residues_removed", "sequence"]].to_csv(data_dir / "viral_panel_sequences.tsv", sep="\t", index=False)
    print(f"data/viral_panel: {len(meta)} sequences, {int(meta['length'].sum()):,} residues")

    sites, summary = [], []
    sources = []
    for key, (model, role) in MODEL_FILES.items():
        parts = []
        for directory, stem, _ in panels:
            path = directory / f"{stem}_allsites_{key}.tsv.gz"
            if not path.is_file():
                raise SystemExit(f"missing {path.name}; run score_panel_allsites.py for {key} first")
            sources.append(path)
            part = pd.read_csv(path, sep="\t")
            if set(part["model"]) != {model}:
                raise SystemExit(f"{path.name} holds {sorted(set(part['model']))}, expected {model!r}")
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        n = df.groupby("label").size()
        expect = meta.set_index("label")["length"]
        if not n.reindex(expect.index).eq(expect).all():
            raise SystemExit(f"{path.name}: a protein does not have one score per residue")
        if not df["p_wt"].between(0, 1).all():
            raise SystemExit(f"{path.name}: probabilities outside [0, 1]")
        sites.append(df)
        g = df.groupby("label", sort=False)
        s = pd.DataFrame({"mean_p_wt": g["p_wt"].mean(), "median_p_wt": g["p_wt"].median(),
                          "top1": g["top1_correct"].mean(), "n_sites": g.size()}).reset_index()
        s.insert(0, "model", model)
        s.insert(1, "role", role)
        summary.append(s)
    sites = pd.concat(sites, ignore_index=True)
    summary = pd.concat(summary, ignore_index=True)

    freq = {r.label: pd.Series(list(r.sequence)).value_counts(normalize=True) for r in meta.itertuples()}
    base = pd.Series({k: float((v ** 2).sum()) for k, v in freq.items()}, name="composition_baseline")
    summary = summary.merge(meta[["label", "display_label", "group", "group_order", "accession", "length"]], on="label")
    summary = summary.merge(base, left_on="label", right_index=True)

    w = Writer(CLEAN / "results" / "viral_panel")
    why_sites = "S3: probability given to every masked residue, per model"
    why_sum = "S3: headline numbers per model and protein"
    where = "tmp_explore/IAV_vs_SC2_criss_cross/viral_panel_allsites_<model>.tsv.gz"
    if cell:
        where += " + tmp_explore/cellular_phage_panel/cellular_panel_allsites_<model>.tsv.gz"
    w.add("viral_panel_sites.tsv.gz", sites, where, why_sites, sha=",".join(sha256(p) for p in sources))
    w.add("viral_panel_summary.tsv.gz", summary, "derived from viral_panel_sites.tsv.gz and the panel table",
          why_sum)
    w.write_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
