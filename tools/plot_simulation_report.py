#!/usr/bin/env python3
"""Generate high-resolution figures for the Simulation Control Progress Report."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import SeqIO
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
REPO_CLEAN = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_CLEAN / "reports/assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASES = ["A", "C", "G", "T"]
SUB_CLASSES = [f"{b1}->{b2}" for b1 in BASES for b2 in BASES if b1 != b2]

def count_mutations(founder_fna: Path, dataset_fna: Path, max_seqs: int = 5000):
    founder_seq = str(SeqIO.read(founder_fna, "fasta").seq).upper()
    counts = {k: 0 for k in SUB_CLASSES}
    total = 0
    records = list(SeqIO.parse(dataset_fna, "fasta"))[:max_seqs]
    for rec in records:
        seq = str(rec.seq).upper()
        for b_ref, b_obs in zip(founder_seq, seq):
            if b_ref != b_obs and b_ref in BASES and b_obs in BASES:
                counts[f"{b_ref}->{b_obs}"] += 1
                total += 1
    pcts = {k: (v / total * 100.0) if total > 0 else 0.0 for k, v in counts.items()}
    return pcts, total

def generate_four_spectra():
    """Plot 1: Empirical 12-Class Mutation Spectra Across All 4 Simulation Arms."""
    founder_fna = REPO / "dev/simulated_phylogenies/v3_star90/founder.fna"
    human_fna = REPO / "dev/simulated_phylogenies/v3_star90/v3_star90.fna"
    sc2_fna = REPO / "dev/simulated_phylogenies/v3_sc2_star90/v3_sc2_star90.fna"
    h3n2_fna = REPO / "dev/simulated_phylogenies/v3_h3n2_star90/v3_h3n2_star90.fna"
    jc69_fna = REPO / "tmp_explore/gpu_sim_arms/v3_jc69_star90/v3_jc69_star90.fna"

    h_pcts, _ = count_mutations(founder_fna, human_fna)
    s_pcts, _ = count_mutations(founder_fna, sc2_fna)
    ia_pcts, _ = count_mutations(founder_fna, h3n2_fna)
    jc_pcts, _ = count_mutations(founder_fna, jc69_fna)

    rows = []
    for sub in SUB_CLASSES:
        rows.append({"Mutation": sub, "Spectrum": "Human Germline", "Percentage": h_pcts[sub]})
        rows.append({"Mutation": sub, "Spectrum": "SARS-CoV-2 (SC2)", "Percentage": s_pcts[sub]})
        rows.append({"Mutation": sub, "Spectrum": "Influenza A (H3N2)", "Percentage": ia_pcts[sub]})
        rows.append({"Mutation": sub, "Spectrum": "JC69 (Flat Control)", "Percentage": jc_pcts[sub]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(15, 6))
    palette = ["#457b9d", "#e63946", "#2a9d8f", "#6c757d"]
    
    # Custom grouped barplot
    spectra = ["Human Germline", "SARS-CoV-2 (SC2)", "Influenza A (H3N2)", "JC69 (Flat Control)"]
    x = np.arange(len(SUB_CLASSES))
    width = 0.20
    
    for idx, (spec, col) in enumerate(zip(spectra, palette)):
        sub_df = df[df["Spectrum"] == spec]
        vals = [sub_df[sub_df["Mutation"] == m]["Percentage"].values[0] for m in SUB_CLASSES]
        rects = ax.bar(x + (idx - 1.5) * width, vals, width, label=spec, color=col)
        for rect in rects:
            h = rect.get_height()
            if h > 15.0:
                ax.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points', fontweight='bold')

    ax.set_title("Empirical 12-Class Substitution Spectra in Matched Evolutionary Simulations (4 Arms)\n"
                 "Identical 700-AA Founder, Shared Fitness Landscape (β=2.0); Human vs. SC2 vs. H3N2 vs. JC69 Flat Control",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Observed Percentage of All Nucleotide Mutations (%)", fontsize=11)
    ax.set_xlabel("Nucleotide Mutation Class", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(SUB_CLASSES, rotation=35, ha='right', fontsize=10)
    ax.axhline(100.0/12.0, color="#6c757d", linestyle=":", lw=1.2, label="Equal Rate (8.33%)")
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax.legend(title="Simulation Arm", loc="upper right", frameon=True, fontsize=10)
    ax.set_ylim(0, 43)

    plt.tight_layout()
    png_path = OUT_DIR / "plot1_four_spectra.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated {png_path}")

def generate_symmetry_battery_plot():
    """Plot 2: Directional Symmetry Battery on Founder (Flip Null & Direction Contrast)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Read flip data
    sc2_flip = pd.read_csv(REPO / "clean_repo/results/sim_v3_star90/symmetry_sars_cov_2_flip.tsv.gz", sep="\t")
    h3n2_flip = pd.read_csv(REPO / "clean_repo/results/sim_v3_star90/symmetry_influenza_h3n2_flip.tsv.gz", sep="\t")
    
    models = ["plm_base", "plm_human_ft", "plm_sc2_ft", "plm_h3n2_ft", "plm_jc69_ft"]
    model_labels = ["Base ESM-C", "Human FT", "SC2 FT", "H3N2 FT", "JC69 (Flat) FT"]
    colors = ["#1d3557", "#457b9d", "#e63946", "#2a9d8f", "#6c757d"]

    # Panel A: Flip Null Rank across Strata for SC2 vs H3N2 spectrum
    x = np.arange(len(models))
    width = 0.25

    # SC2 spectrum: 1-nt, 2-nt, 3-nt
    sc2_s1 = [sc2_flip[(sc2_flip["score"] == m) & (sc2_flip["stratum"] == 1)]["pct_flip"].values[0] for m in models]
    sc2_s2 = [sc2_flip[(sc2_flip["score"] == m) & (sc2_flip["stratum"] == 2)]["pct_flip"].values[0] for m in models]
    sc2_s3 = [sc2_flip[(sc2_flip["score"] == m) & (sc2_flip["stratum"] == 3)]["pct_flip"].values[0] for m in models]

    b1 = ax1.bar(x - width, sc2_s1, width, label="1 nt (SC2 matrix)", color="#fca5a5")
    b2 = ax1.bar(x, sc2_s2, width, label="2 nt (SC2 matrix)", color="#ef4444")
    b3 = ax1.bar(x + width, sc2_s3, width, label="3 nt (SC2 matrix)", color="#991b1b")

    ax1.axhline(0.05, color="black", linestyle="--", lw=1.2, label="p = 0.05 Threshold")
    ax1.axhline(0.50, color="gray", linestyle=":", lw=1, label="Chance (p = 0.50)")
    ax1.set_title("A: Flip-Null Test Under SARS-CoV-2 Spectrum\n(True Direction Rank / 64 Reciprocal Flips; Lower = Better)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Flip-Null Percentile (p-value)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=10)
    ax1.set_ylim(0, 0.65)
    ax1.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax1.legend(loc="upper right", fontsize=8.5)

    # Annotate SC2 FT rank (1st or 2nd)
    ax1.annotate("Rank 1/64 (p=0.016)\nRank 2/64 (p=0.031)", xy=(2, 0.031), xytext=(2, 0.25),
                 ha='center', fontsize=9, fontweight='bold', color="#e63946",
                 arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.5))

    # Panel B: Direction Contrast (% of Truth)
    sc2_dir = pd.read_csv(REPO / "clean_repo/results/sim_v3_star90/symmetry_sars_cov_2_direction.tsv.gz", sep="\t")
    h3n2_dir = pd.read_csv(REPO / "clean_repo/results/sim_v3_star90/symmetry_influenza_h3n2_direction.tsv.gz", sep="\t")

    sc2_pct = [sc2_dir[sc2_dir["score"] == m]["fraction_of_truth"].values[0] * 100 for m in models]
    h3n2_pct = [h3n2_dir[h3n2_dir["score"] == m]["fraction_of_truth"].values[0] * 100 for m in models]

    ax2.bar(x - width/2, sc2_pct, width, label="Under SARS-CoV-2 Spectrum", color="#e63946")
    ax2.bar(x + width/2, h3n2_pct, width, label="Under Influenza H3N2 Spectrum", color="#2a9d8f")
    ax2.axhline(0, color="0.4", lw=0.8)
    ax2.set_title("B: Direction Contrast Recovery (% of True Contrast)\n(Favoured vs. Disfavoured Rate Contrast Recovered by Model)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Contrast as % of Ground-Truth Spectrum", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax2.legend(loc="upper left", fontsize=9)

    for i in range(len(models)):
        val_s = sc2_pct[i]
        ax2.annotate(f"{val_s:.1f}%", (x[i] - width/2, max(val_s, 0) + 1), ha='center', fontsize=8, color="#e63946", fontweight='bold')
        val_h = h3n2_pct[i]
        ax2.annotate(f"{val_h:.1f}%", (x[i] + width/2, max(val_h, 0) + 1), ha='center', fontsize=8, color="#2a9d8f")

    plt.tight_layout()
    png_path = OUT_DIR / "plot2_founder_battery.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated {png_path}")

def generate_partial_spearman_plot():
    """Plot 3: Partial Spearman Correlation Holding True Fitness Fixed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sc2_part = pd.read_csv(REPO / "clean_repo/results/sim_v3_star90/symmetry_sars_cov_2_partial_given_truth.tsv.gz", sep="\t")
    h3n2_part = pd.read_csv(REPO / "clean_repo/results/sim_v3_star90/symmetry_influenza_h3n2_partial_given_truth.tsv.gz", sep="\t")

    models = ["plm_base", "plm_human_ft", "plm_sc2_ft", "plm_h3n2_ft", "plm_jc69_ft"]
    model_labels = ["Base ESM-C", "Human FT", "SC2 FT", "H3N2 FT", "JC69 (Flat) FT"]

    # Filter to direct rho holding ground_truth fixed
    def get_rhos(df, stratum):
        sub = df[(df["kind"] == "rho") & (df["target"] == "direct") & (df["stratum"] == stratum)]
        return [sub[sub["score"] == m]["value"].values[0] for m in models], \
               [sub[sub["score"] == m]["lo"].values[0] for m in models], \
               [sub[sub["score"] == m]["hi"].values[0] for m in models]

    x = np.arange(len(models))
    width = 0.25

    # Panel A: Under SARS-CoV-2 Accessibility
    r1, lo1, hi1 = get_rhos(sc2_part, 1)
    r2, lo2, hi2 = get_rhos(sc2_part, 2)
    r3, lo3, hi3 = get_rhos(sc2_part, 3)

    ax1.bar(x - width, r1, width, label="1 nt stratum", color="#93c5fd")
    ax1.bar(x, r2, width, label="2 nt stratum", color="#2563eb")
    ax1.bar(x + width, r3, width, label="3 nt stratum", color="#1e3a8a")
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.set_title("A: Partial Correlation with SARS-CoV-2 Accessibility\n(Holding Ground-Truth Fitness Fixed; Founder Background)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Partial Spearman ρ (Score vs. Mut_Prob | Truth)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=10)
    ax1.set_ylim(-0.08, 0.32)
    ax1.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax1.legend(loc="upper left", fontsize=8.5)

    # Annotate SC2 FT 2-nt peak
    ax1.annotate("SC2 FT Peak:\n+0.256 [0.231, 0.286]\n(Beats Base +0.174)", xy=(2, 0.256), xytext=(2, 0.14),
                 ha='center', fontsize=8.5, fontweight='bold', color="#1e3a8a",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", ec="#2563eb", lw=1))

    # Panel B: Under Influenza H3N2 Accessibility
    r1_h, _, _ = get_rhos(h3n2_part, 1)
    r2_h, _, _ = get_rhos(h3n2_part, 2)
    r3_h, _, _ = get_rhos(h3n2_part, 3)

    ax2.bar(x - width, r1_h, width, label="1 nt stratum", color="#a7f3d0")
    ax2.bar(x, r2_h, width, label="2 nt stratum", color="#059669")
    ax2.bar(x + width, r3_h, width, label="3 nt stratum", color="#064e3b")
    ax2.axhline(0, color="0.4", lw=0.8)
    ax2.set_title("B: Partial Correlation with Influenza H3N2 Accessibility\n(Holding Ground-Truth Fitness Fixed; Founder Background)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Partial Spearman ρ (Score vs. Mut_Prob | Truth)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=10)
    ax2.set_ylim(-0.16, 0.25)
    ax2.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax2.legend(loc="upper left", fontsize=8.5)

    # Annotate H3N2 FT and JC69 FT
    ax2.annotate("H3N2 FT Specific:\n+0.175 (1 nt), +0.174 (2 nt)\n(Dominates Base +0.052 / -0.077)",
                 xy=(3, 0.174), xytext=(3, 0.04),
                 ha='center', fontsize=8.5, fontweight='bold', color="#064e3b",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ecfdf5", ec="#059669", lw=1))

    ax2.annotate("JC69 Flat FT:\n-0.0005 (1 nt)\n-0.0274 (2 nt)\nZero bias!",
                 xy=(4, -0.0005), xytext=(4, -0.11),
                 ha='center', fontsize=8.5, fontweight='bold', color="#6c757d",
                 arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1))

    plt.tight_layout()
    png_path = OUT_DIR / "plot3_partial_given_truth.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated {png_path}")

def generate_founder_fitness_recovery():
    """Plot 4: Raw vs Debiased Correlation with Ground-Truth Fitness on Founder."""
    scores_path = REPO / "clean_repo/results/sim_v3_star90/replacement_scores.tsv.gz"
    df = pd.read_csv(scores_path, sep="\t")

    # Models to test
    models = ["plm_base", "plm_human_ft", "plm_sc2_ft", "plm_h3n2_ft", "plm_jc69_ft"]
    model_labels = ["Base ESM-C", "Human FT", "SC2 FT", "H3N2 FT", "JC69 (Flat) FT"]
    
    # Compute raw spearman rho with ground truth
    raw_rhos = []
    deb_rhos = []
    alpha = -0.075

    for m in models:
        rho_raw, _ = spearmanr(df[m], df["ground_truth"])
        raw_rhos.append(rho_raw)
        
        # Debias using the model's matched spectrum accessibility (or SC2 accessibility for base/flat)
        if m == "plm_human_ft":
            mut_col = "mut_prob_Human_Germline"
        elif m == "plm_sc2_ft":
            mut_col = "mut_prob_SC2_Directional"
        elif m == "plm_h3n2_ft":
            mut_col = "mut_prob_H3N2"
        else:
            mut_col = "mut_prob_SC2_Directional"

        debiased = df[m] - alpha * np.log(np.clip(df[mut_col], 1e-300, 1.0))
        rho_deb, _ = spearmanr(debiased, df["ground_truth"])
        deb_rhos.append(rho_deb)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, raw_rhos, width, label="Raw Model Score on Founder (α = 0)", color="#93c5fd")
    rects2 = ax.bar(x + width/2, debiased_rhos if 'debiased_rhos' in locals() else deb_rhos, width,
                    label="Debiased with Matched Spectrum (α = -0.075)", color="#1d3557")

    ax.set_title("Ground-Truth Fitness Recovery on Founder Background Across All 5 Models\n"
                 "Raw PLM Probabilities vs. Debiased Scores with Matched Mutational Spectrum",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Spearman ρ with Ground-Truth Fitness (π_i,a)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylim(0, 0.45)
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax.legend(loc="upper left", fontsize=9.5)

    for i in range(len(models)):
        ax.annotate(f"{raw_rhos[i]:.3f}", (x[i] - width/2, raw_rhos[i] + 0.005), ha='center', fontsize=8.5)
        diff = deb_rhos[i] - raw_rhos[i]
        diff_str = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
        ax.annotate(f"{deb_rhos[i]:.3f}\n({diff_str})", (x[i] + width/2, deb_rhos[i] + 0.005),
                    ha='center', fontsize=8.5, fontweight='bold', color="#1d3557")

    plt.tight_layout()
    png_path = OUT_DIR / "plot4_founder_fitness_recovery.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated {png_path}")

def copy_archive_figures():
    """Copy the archive S25-S28 figures to the assets folder."""
    import shutil
    src_dir = REPO / "figures/key_revision_figures/supplementary"
    figs = [
        "S25_simulation_control_sc2_vs_human_crossover.png",
        "S26_directional_vs_symmetric_mutation_specificity.png",
        "S27_pssm_alignment_vs_simulation_fitness.png",
        "S28_plm_probability_vs_simulation_fitness_debiasing.png"
    ]
    for f in figs:
        src = src_dir / f
        dst = OUT_DIR / f
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied {f} to {dst}")

def main():
    print("=== Generating Simulation Control Report Plots ===")
    copy_archive_figures()
    generate_four_spectra()
    generate_symmetry_battery_plot()
    generate_partial_spearman_plot()
    generate_founder_fitness_recovery()
    print("All plots generated in:", OUT_DIR)

if __name__ == "__main__":
    main()
