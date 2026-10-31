"""Build the SARS-CoV-2 spike table that S1 reads, and the root FASTAs it needs.

    python tools/export_s01.py [--archive ..]

S1 asks Figure 2's question of a second virus: does a model's probability for a replacement
track how easily mutation supplies it, once the number of nucleotide changes is held fixed?

Inputs, all in the exploration archive under ``tmp_explore/``:

* ``sc2/<lineage>.nt.txt`` and ``.prot.txt``  the spike coding sequence and its translation;
* ``esmc_matrices/<lineage>__<model>.csv``  the model's 20 x length probability matrix. The
  reference protein is stored in the header row (after a leading ``sequence`` cell), and is
  checked against the translation here.

Two ESM-C 600M models were scored on all five spike lineages (Wuhan-Hu-1, JN.1, XEC, LP.8.1,
NB.1.8.1), by ``IAV_vs_SC2_criss_cross/score_esmc.py``:

    ESMC_600M_base    ESM-C 600M, no fine-tuning
    ESMC_600M_SC2ft   the same model fine-tuned on SARS-CoV-2 spike, checkpoint
                      ``my_SC2_finetunes/magma_ESMC_600M_99_95perc`` (``esmc_weights.pt``)

S1 originally used the ESM-2 650M spike fine-tune (``magma_esm-2_650M_95perc``) and an ESM-2
model fine-tuned on influenza HA. Both were the wrong comparison: this paper's spike
fine-tune is the ESM-C one, and there is no ESM-C influenza fine-tune scored on spike, so the
foreign-protein arm is dropped rather than mixed in from a different backbone.

Accessibility uses the SARS-CoV-2 spectrum ``data/spectra/sars_cov_2.csv`` *as given*, which is
how the archive's own pipeline used it. ``blosum62`` and ``neg_grantham`` are properties of the
amino-acid pair alone, so they are looked up from the shipped influenza table rather than
recomputed from a second source.

Output: ``results/sars_cov_2_spike/replacement_scores.tsv.gz`` in the same schema as the
influenza tables, plus ``data/roots/sars_cov_2_spike/<lineage>.nt.fa``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_base import Writer, sha256          # noqa: E402
from mutacc import accessibility as acc         # noqa: E402

LINEAGES = ["Wuhan-Hu-1_Reference", "JN.1", "XEC", "LP.8.1", "NB.1.8.1"]
MODELS = ["ESMC_600M_base", "ESMC_600M_SC2ft"]
AAS = list("ACDEFGHIKLMNPQRSTVWY")
SET = "sars_cov_2_spike"
SIG = 7                      # plm_prob significant digits, as for the influenza tables


def read_matrix(path: Path) -> tuple[str, pd.DataFrame]:
    """The reference protein and the 20 x length probability matrix of one scored file."""
    header = path.read_text().split("\n", 1)[0].rstrip("\r").split(",")
    protein = "".join(header[1:])
    frame = pd.read_csv(path, header=None, skiprows=1, index_col=0).astype(float)
    frame.columns = range(frame.shape[1])
    # Rows are labelled and may come in any order (the pipeline writes A R N D ...), so look
    # them up by letter; only require that all twenty are there exactly once.
    if sorted(frame.index) != sorted(AAS):
        raise ValueError(f"{path.name}: amino-acid rows are {list(frame.index)}, expected the twenty in {AAS}")
    if frame.shape[1] != len(protein):
        raise ValueError(f"{path.name}: {frame.shape[1]} columns for a {len(protein)}-residue protein")
    return protein, frame


def pair_properties(flu_table: Path) -> pd.DataFrame:
    """blosum62 and neg_grantham for every ordered amino-acid pair, from the influenza table."""
    cols = ["ref_aa", "aa", "blosum62", "neg_grantham"]
    pairs = pd.read_csv(flu_table, sep="\t", usecols=cols).drop_duplicates()
    if pairs.duplicated(["ref_aa", "aa"]).any():
        raise ValueError("a pair's BLOSUM62 or Grantham value differs between rows of the influenza table")
    if len(pairs) != 380:
        raise ValueError(f"expected 380 ordered pairs in the influenza table, found {len(pairs)}")
    return pairs


def lineage_rows(archive: Path, lineage: str, q: np.ndarray) -> pd.DataFrame:
    sc2 = archive / "tmp_explore" / "sc2"
    matrices = archive / "tmp_explore" / "esmc_matrices"
    nt = (sc2 / f"{lineage}.nt.txt").read_text().strip().upper()
    protein = (sc2 / f"{lineage}.prot.txt").read_text().strip()
    translated = "".join(acc.GENETIC_CODE[nt[i:i + 3]] for i in range(0, len(nt) - 3, 3))
    if translated != protein:
        raise ValueError(f"{lineage}: the nucleotide sequence does not translate to the stored protein")
    if len(nt) != 3 * len(protein) + 3 or acc.GENETIC_CODE[nt[-3:]] != "*":
        raise ValueError(f"{lineage}: expected the coding sequence plus one stop codon")

    mats = {}
    for model in MODELS:
        seq, frame = read_matrix(matrices / f"{lineage}__{model}.csv")
        if seq != protein:
            raise ValueError(f"{lineage} {model}: the scored protein differs from the translation")
        mats[model] = frame

    rows = []
    for i, ref_aa in enumerate(protein):
        codon = nt[3 * i: 3 * i + 3]
        if ref_aa not in AAS or set(codon) - set("ACGT"):
            continue                                # unknown residue: nothing to score
        for aa in AAS:
            if aa == ref_aa:
                continue
            rows.append((lineage, i + 1, ref_aa, aa, codon))
    base = pd.DataFrame(rows, columns=["lineage", "position", "ref_aa", "aa", "codon"])
    base["mut_prob"] = acc.accessibility(base["codon"], base["aa"], q)
    base["nt_steps"] = [acc.nucleotide_steps(c, a) for c, a in zip(base["codon"], base["aa"])]
    out = []
    for model in MODELS:
        part = base.copy()
        part["model"] = model
        part["plm_prob"] = [float(mats[model].at[a, p - 1]) for a, p in zip(part["aa"], part["position"])]
        out.append(part)
    return pd.concat(out, ignore_index=True)


def write_roots(archive: Path) -> None:
    folder = CLEAN / "data" / "roots" / SET
    folder.mkdir(parents=True, exist_ok=True)
    for lineage in LINEAGES:
        nt = (archive / "tmp_explore" / "sc2" / f"{lineage}.nt.txt").read_text().strip().upper()
        (folder / f"{lineage}.nt.fa").write_text(f">spike|{lineage}\n{nt}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", type=Path, default=CLEAN.parent,
                    help="the exploration archive (default: the folder above clean_repo)")
    args = ap.parse_args(argv)
    archive = args.archive.resolve()

    q = acc.load_spectrum(CLEAN / "data" / "spectra" / "sars_cov_2.csv")
    frames = [lineage_rows(archive, lin, q) for lin in LINEAGES]
    table = pd.concat(frames, ignore_index=True)

    props = pair_properties(CLEAN / "results" / "h3n2_13lineage" / "replacement_scores.tsv.gz")
    table = table.merge(props, on=["ref_aa", "aa"], how="left", validate="many_to_one")
    if table[["blosum62", "neg_grantham"]].isna().any().any():
        raise ValueError("an amino-acid pair has no BLOSUM62 or Grantham value")
    table["plm_prob"] = table["plm_prob"].map(lambda v: float(f"{v:.{SIG}g}"))
    keep = ["model", "lineage", "position", "ref_aa", "aa", "plm_prob", "mut_prob",
            "nt_steps", "blosum62", "neg_grantham"]
    table = table[keep].sort_values(["model", "lineage", "position", "aa"]).reset_index(drop=True)

    write_roots(archive)
    writer = Writer(CLEAN / "results" / SET)
    source = "tmp_explore/sc2/<lineage>.{nt,prot}.txt and esmc_matrices/<lineage>__<model>.csv"
    writer.add("replacement_scores.tsv.gz", table, source,
               "S1: spike, model probability vs accessibility within codon-distance strata",
               sha=sha256(archive / "tmp_explore" / "esmc_matrices" / "JN.1__ESMC_600M_SC2ft.csv"))
    writer.write_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
