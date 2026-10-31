"""Table behind the simulation arm of the symmetry tests.

Sequences were evolved from one synthetic 700-residue founder under a known nucleotide spectrum and a
known site-wise fitness landscape, changing only the spectrum: Human Germline, SARS-CoV-2,
Influenza H3N2 (Pauly et al. 2017), and JC69 (flat, no direction). ESM-C 600M was fine-tuned on each.
Every model was re-scored directly on the founder itself, eliminating the tip-representative divergence
flaw from earlier scorings (where tip representatives differed from founder at 9-11% of sites).

One row per (site, replacement): 700 x 19 = 13,300 rows. The reference codon comes from the founder,
so accessibility can be recomputed under any spectrum.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_base import Export  # noqa: E402
from mutacc import accessibility as acc  # noqa: E402

FOUNDER = "dev/simulated_phylogenies/v3_star90/founder.fna"
FITNESS = "tmp_explore/gpu_sim_arms/v3_jc69_star90/fitness_landscape_v2.csv"
ARMS = ["human", "sc2", "h3n2", "jc69"]


def sim_table(frame: pd.DataFrame, source_repo: Path) -> pd.DataFrame:
    from Bio import SeqIO
    from Bio.Data import CodonTable

    seq = str(SeqIO.read(source_repo / FOUNDER, "fasta").seq).upper()
    table = CodonTable.unambiguous_dna_by_id[1].forward_table
    codons = {i + 1: seq[3 * i : 3 * i + 3] for i in range(len(seq) // 3)}
    wt_aas = {i + 1: table[codons[i + 1]] for i in range(len(seq) // 3)}

    base = frame[frame["model"] == "founder_base"].copy()

    out = pd.DataFrame()
    out["site"] = base["position"]
    out["ref_codon"] = out["site"].map(codons)
    out["ref_aa"] = out["site"].map(wt_aas)
    out["aa"] = base["aa"]
    out["nt_steps"] = [acc.nucleotide_steps(c, a) for c, a in zip(out["ref_codon"], out["aa"])]
    out["plm_base"] = base["plm_prob"].values

    score_dir = source_repo / "tmp_explore/gpu_sim_arms/score"
    for arm in ARMS:
        df_arm = pd.read_csv(score_dir / f"founder_{arm}/tables/combined_long_table.csv")
        ft = df_arm[df_arm["model"] == f"founder_{arm}"]
        if not (ft["position"].values == out["site"].values).all() or not (ft["aa"].values == out["aa"].values).all():
            raise ValueError(f"arm {arm} rows do not align with base table")
        out[f"plm_{arm}_ft"] = ft["plm_prob"].values
        out[f"pssm_{arm}"] = ft["obs_freq"].values

    fl = pd.read_csv(source_repo / FITNESS)
    fl_melt = fl.melt(id_vars=["site", "founder_aa"], var_name="aa", value_name="ground_truth")
    out = out.merge(fl_melt[["site", "aa", "ground_truth"]], on=["site", "aa"], how="left", validate="one_to_one")

    q_human = acc.load_spectrum(source_repo / "clean_repo/data/spectra/human.csv") * 1.2e-8
    q_sc2 = acc.load_spectrum(source_repo / "clean_repo/data/spectra/sars_cov_2.csv")
    q_h3n2 = acc.load_spectrum(source_repo / "clean_repo/data/spectra/influenza_h3n2.csv")
    q_jc69 = acc.load_spectrum(source_repo / "clean_repo/data/spectra/jc69.csv")

    out["mut_prob_Human_Germline"] = acc.accessibility(out["ref_codon"], out["aa"], q_human)
    out["mut_prob_SC2_Directional"] = acc.accessibility(out["ref_codon"], out["aa"], q_sc2)
    out["mut_prob_H3N2"] = acc.accessibility(out["ref_codon"], out["aa"], q_h3n2)
    out["mut_prob_JC69"] = acc.accessibility(out["ref_codon"], out["aa"], q_jc69, row_sum=3.5e-4)

    cols = [
        "site", "ref_codon", "ref_aa", "aa", "nt_steps",
        "plm_base", "plm_human_ft", "plm_sc2_ft", "plm_h3n2_ft", "plm_jc69_ft",
        "ground_truth", "pssm_human", "pssm_sc2", "pssm_h3n2", "pssm_jc69",
        "mut_prob_SC2_Directional", "mut_prob_Human_Germline", "mut_prob_H3N2", "mut_prob_JC69",
    ]
    return out[cols].sort_values(["site", "aa"]).reset_index(drop=True)


EXPORTS = [
    Export(
        "sim_v3_star90",
        "tmp_explore/gpu_sim_arms/score/founder_base/tables/combined_long_table.csv",
        "replacement_scores.tsv.gz",
        transform=sim_table,
        why="Symmetry tests with known truth: ESM-C base and fine-tuned on sequences evolved under known spectra (scored on founder)",
    ),
]
