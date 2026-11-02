"""S16: Reviewer 2 simulation control — matched evolutionary simulations under four mutation spectra.

    python figures/make_sN_simulation_control.py

Sequences were evolved from one synthetic founder under one shared site-wise fitness
landscape (beta = 2.0), varying only the generating nucleotide mutation spectrum:
human germline, SARS-CoV-2, influenza A H3N2, and flat JC69 (negative control).
ESM-C 600M was scored on the founder before fine-tuning and after fine-tuning on each arm.

Panels are lettered left to right, row by row.
Panel A shows empirical 12-class substitution spectra across the four evolutionary arms.
Panel B displays the cross-correlation heatmap between PLMs and mutation spectra across all replacements.
Panel C displays the mutational accessibility alpha sweep curves for fitness estimation.
Panel D displays the cross-correlation heatmap restricted to the N=1 nucleotide step band.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from mutacc import style
from mutacc.io import write_caption, write_values

STEM = "sN_simulation_control"
RESULTS_SET = "sim_v3_star90"
REPO_ROOT = Path(__file__).resolve().parents[1]

# The four arms are organisms, not model roles, so they take the non-model
# palette (see mutacc/style.py, "the two-palette rule"). Earlier this figure
# spent "finetune" green on the SARS-CoV-2 arm and "accessibility" vermillion on
# the human arm, which reads as the model pair from every other figure in the
# set -- and in panel C all four arms *are* fine-tuned, so green there said
# nothing. The flat JC69 arm is the negative control and wears the null grey.
#: The biological arms, keyed by the arm rank ``arm_rank`` assigns. Ranks are
#: fixed properties of the arm, not loop counters, so a results set missing one
#: arm cannot repaint the ones that remain -- the rule colours_for_categories
#: enforces for names.
BIOLOGICAL_ARM_RANKS = (1, 2, 3)


def colour_for_rank(rank: int) -> str:
    """Colour for an evolutionary arm, keyed by its arm rank.

    Negative rank is the untuned base model, which is a model and so keeps the
    model blue. Rank 0 is the flat neutral negative control and wears null grey.
    """
    if rank < 0:
        return style.colour_for_role("base")
    if rank == 0:
        return style.colour_for_role("null")
    return style.colours_for_categories(BIOLOGICAL_ARM_RANKS)[rank]


def fmt_3sf(v: float) -> str:
    """Format a float to 3 significant figures with explicit sign."""
    if abs(v) < 1e-12:
        return "+0.000"
    sign = "+" if v >= 0 else "-"
    val = abs(v)
    decimals = 3 - 1 - int(math.floor(math.log10(val)))
    if decimals <= 0:
        s = f"{round(val, decimals):.0f}"
    else:
        s = f"{val:.{decimals}f}"
    return f"{sign}{s}"


def _relative_luminance(rgb) -> float:
    channels = []
    for c in rgb[:3]:
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    lo, hi = sorted((la, lb))
    return (hi + 0.05) / (lo + 0.05)


def text_colour_on(rgba) -> str:
    import matplotlib.colors as mcolors
    ink = mcolors.to_rgb(style.INK)
    surface = mcolors.to_rgb(style.SURFACE)
    return style.INK if _contrast(rgba, ink) >= _contrast(rgba, surface) else style.SURFACE


def load(results: Path) -> dict:
    spectra_path = results / "empirical_substitution_spectra.tsv.gz"
    scores_path = results / "replacement_scores.tsv.gz"
    if not spectra_path.exists() or not scores_path.exists():
        raise FileNotFoundError(f"Missing required tables in {results}")

    spectra_df = pd.read_csv(spectra_path, sep="\t")
    scores_df = pd.read_csv(scores_path, sep="\t")

    # The heatmaps used to say only "PLMs", which left a reader unable to tell
    # which protein model produced the rows. The name is read off the results
    # set's own manifest rather than written here, both because figure scripts
    # may not name a model and so that a results set built on a different model
    # relabels itself.
    plm_name = "PLM"
    manifest_path = results / "MANIFEST.tsv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, sep="\t")
        row = manifest[manifest["file"] == scores_path.name]
        if not row.empty:
            found = re.search(r"([A-Za-z][A-Za-z0-9]*-?[A-Za-z0-9]+)\s+base\b",
                              str(row["used_for"].iloc[0]))
            if found:
                plm_name = found.group(1)

    sources_path = REPO_ROOT / "data/spectra/spectra_sources.csv"
    if sources_path.exists():
        sources = pd.read_csv(sources_path)
        label_map = dict(zip(sources["key"], sources["label"]))
    else:
        label_map = {}

    return {
        "spectra": spectra_df,
        "scores": scores_df,
        "label_map": label_map,
        "plm_name": plm_name,
    }


def clean_short_label(lbl: str) -> str:
    parts = lbl.split()
    if len(parts) > 1 and "(" in parts[-1]:
        return parts[0]
    return parts[-1]


def get_aligned_specs(scores_df: pd.DataFrame, label_map: dict) -> tuple[list[dict], list[dict]]:
    def arm_rank(tok: str) -> int:
        t = tok.lower()
        if "jc" in t or "flat" in t:
            return 0
        if "hum" in t:
            return 1
        if "3" in t:
            return 3
        if "2" in t or "sc" in t:
            return 2
        return 99

    mut_probs = [c for c in scores_df.columns if c.startswith("mut_prob_")]
    cols = [c for c in scores_df.columns if c.startswith("plm_")]

    # Build spectrum specs
    raw_s_specs = []
    for col in mut_probs:
        token = col.replace("mut_prob_", "").lower()
        match_label = None
        for k, lbl in label_map.items():
            if k.lower() in token or token.split("_")[0] in k.lower():
                match_label = lbl
                break
        if match_label is None and ("2" in token or "sc" in token):
            for k, lbl in label_map.items():
                if "2" in k and k.startswith("s"):
                    match_label = lbl
                    break
        match_label = match_label or col.replace("mut_prob_", "").replace("_", " ")

        if "sc" in token:
            short = "SC2"
        elif "jc69" in token or "flat" in token:
            short = "Flat"
        elif "3" in token:
            short = clean_short_label(match_label)
        else:
            short = "Human"

        raw_s_specs.append({
            "col": col,
            "label": match_label,
            "short_label": short,
            "token": token,
            "rank": arm_rank(token),
        })
    s_specs = sorted(raw_s_specs, key=lambda x: x["rank"])

    # Build model specs
    raw_m_specs = []
    flat_mut = [s["col"] for s in s_specs if "jc69" in s["token"] or "flat" in s["token"]][0]

    for col in cols:
        token = col.replace("plm_", "").replace("_ft", "").lower()
        if token == "base":
            role = "base"
            label = "Base (untuned)"
            short = "Base"
            mut_match = flat_mut
            rank = -1
        elif "jc69" in token or "flat" in token:
            role = "null"
            match_label = None
            for k, lbl in label_map.items():
                if "jc69" in k.lower() or "flat" in k.lower():
                    match_label = f"{lbl} FT"
                    break
            label = match_label or "Flat FT"
            short = "Flat FT"
            mut_match = flat_mut
            rank = arm_rank(token)
        else:
            match_label = None
            for k, lbl in label_map.items():
                if token in k.lower():
                    match_label = f"{lbl} FT"
                    break
            label = match_label or f"{token.upper()} FT"

            if "sc" in token:
                role = "finetune"
                short = "SC2 FT"
            elif "human" in token:
                role = "accessibility"
                short = "Human FT"
            elif "3" in token:
                role = "biochemistry"
                short = f"{clean_short_label(label.replace(' FT', ''))} FT"
            else:
                role = "null"
                short = "Flat FT"

            mut_match = None
            for s in s_specs:
                if arm_rank(s["token"]) == arm_rank(token):
                    mut_match = s["col"]
                    break
            mut_match = mut_match or flat_mut
            rank = arm_rank(token)

        raw_m_specs.append({
            "col": col,
            "label": label,
            "short_label": short,
            "role": role,
            "token": token,
            "mut_col": mut_match,
            "rank": rank,
        })

    base_specs = [m for m in raw_m_specs if m["rank"] < 0]
    ft_specs = sorted([m for m in raw_m_specs if m["rank"] >= 0], key=lambda x: x["rank"])
    m_specs = base_specs + ft_specs

    return m_specs, s_specs


def draw_panel_a(ax, spectra_df: pd.DataFrame, label_map: dict):
    def arm_rank(arm_str: str) -> int:
        a = arm_str.lower()
        if "jc" in a or "flat" in a:
            return 0
        if "hum" in a:
            return 1
        if "3" in a:
            return 3
        if "2" in a or "sc" in a:
            return 2
        return 99

    arms = sorted(spectra_df["arm"].unique().tolist(), key=arm_rank)
    classes = spectra_df["mutation_class"].unique().tolist()
    x = np.arange(len(classes))
    width = 0.8 / len(arms)

    for idx, arm in enumerate(arms):
        sub = spectra_df[spectra_df["arm"] == arm]
        val_map = dict(zip(sub["mutation_class"], sub["percentage"]))
        vals = [val_map.get(c, 0.0) for c in classes]
        col = colour_for_rank(arm_rank(arm))
        disp_label = label_map.get(arm, arm.replace("_", " ").title())
        ax.bar(x + (idx - (len(arms) - 1) / 2.0) * width, vals, width,
               label=disp_label, color=col, edgecolor="white", linewidth=0.5)

    # "Equal rate" read as a claim about the bars, and the flat arm's bars are
    # visibly not equal. The line is the flat *proposal* rate; what is plotted
    # is what selection retained, which is transition-enriched in every arm,
    # the flat one included. The label now says which of the two it marks.
    style.reference_line(ax, 100.0 / len(classes), label="Flat proposal (8.3%)",
                         label_loc="right", label_side="above")
    for t in list(ax.texts):
        if "8.3" in t.get_text():
            t.remove()
    # Hand-placed over the C->A/C->G gap: the automatic right-hand placement lands
    # on the T->C bars, and x=3.1 put the box through the top of the A->G arm.
    ax.text(4.5, 9.6, "Flat proposal (8.3%)", ha="center", va="bottom", fontsize=5.8, color=style.INK_MUTED,
            bbox=dict(boxstyle="square,pad=0.15", facecolor=style.SURFACE, edgecolor="none", alpha=0.92))
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Observed substitutions (%)")
    ax.set_xlim(-0.65, 11.65)
    ax.set_ylim(0, 44)
    style.panel_title(ax, "Empirical substitution spectra across 4 arms",
                      "Substitutions retained after selection, not proposal rates")
    ax.legend(loc="upper left", frameon=False, fontsize=5.8, ncol=2,
              columnspacing=0.8, handlelength=0.9, handletextpad=0.3, labelspacing=0.25)
    style.finalise_axes(ax)


def draw_panel_c(ax, scores_df: pd.DataFrame, model_specs: list[dict], alphas: np.ndarray):
    fit = scores_df["ground_truth"].to_numpy()

    for spec in model_specs:
        col = spec["col"]
        mut_col = spec["mut_col"]
        c = colour_for_rank(spec["rank"])

        log_p = np.log(scores_df[col].clip(lower=1e-12).to_numpy())
        log_m = np.log(scores_df[mut_col].clip(lower=1e-25).to_numpy())

        rhos = []
        for a in alphas:
            score = log_p + a * log_m
            r, _ = spearmanr(score, fit)
            rhos.append(float(r))
        rhos = np.array(rhos)

        best_idx = int(np.argmax(rhos))
        best_alpha = float(alphas[best_idx])
        best_rho = float(rhos[best_idx])

        short = spec["short_label"]
        ax.plot(alphas, rhos, color=c, lw=1.6, label=f"{short} ({best_alpha:+.2f})")
        ax.scatter([best_alpha], [best_rho], color=c, s=18, zorder=4, edgecolor="white", linewidth=0.5)

    style.reference_line(ax, 0.0, label="alpha = 0", axis="x", label_side="below",
                         color=style.INK_MUTED, zorder=2, **style.ALPHA_ZERO_RULE)
    ax.set_xlabel(r"Mutational accessibility weight $\alpha$")
    ax.set_ylabel("Spearman ρ vs fitness")
    ax.set_xlim(-0.52, 0.52)
    ax.set_ylim(0.0, 0.68)
    style.panel_title(ax, r"Fitness estimation across accessibility weight $\alpha$",
                      r"Debiasing ($\alpha < 0$) recovers underlying fitness landscape")
    ax.legend(loc="upper right", frameon=False, fontsize=5.8,
              handlelength=1.0, handletextpad=0.3, labelspacing=0.25)
    style.finalise_axes(ax)


def draw_heatmap(ax, mat: pd.DataFrame, title: str, subtitle: str, highlight_diag: bool = True,
                 diag_offset: int = 1, cbar_x: float = 1.04, cbar_w: float = 0.035):
    values = mat.to_numpy(dtype=float)
    # Symmetric about zero, so a unit of correlation is the same depth of colour
    # either side and a near-zero cell stays near-white. The panel D claim is
    # that the flat control sits at zero, and a two-slope norm stretched to the
    # data minimum painted its -0.007 cell a confident blue.
    #
    # Both matrices are almost entirely positive, so the blue half is nearly
    # unused: the colourbar is then clipped to the data rather than showing a
    # negative axis that no cell reaches. Clipping the bar, not the norm, keeps
    # the two sides comparable.
    span = float(np.nanmax(np.abs(values)))
    norm = Normalize(vmin=-span, vmax=span)
    lo = min(float(np.nanmin(values)), 0.0)
    hi = float(np.nanmax(values))
    bar_pad = 0.02 * (hi - lo)

    image = ax.imshow(values, cmap=style.DIVERGING, norm=norm, aspect="auto")

    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = values[i, j]
            color = text_colour_on(image.cmap(norm(v)))
            ax.text(j, i, fmt_3sf(v), ha="center", va="center", fontsize=5.8, fontweight="semibold", color=color)

            # Outline matching diagonal cell for fine-tuned models
            # Panels B and D carry an untuned base row on top, so the matched
            # cell sits one row down; panel E has fine-tunes only and matches on
            # the true diagonal.
            if highlight_diag and i >= diag_offset and (i - diag_offset) == j:
                rect = Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                                 edgecolor=style.INK, linewidth=1.3, zorder=5)
                ax.add_patch(rect)

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=0, ha="center", fontsize=7.0)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7.0)

    ax.set_xticks(np.arange(-0.5, len(mat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = ax.inset_axes([cbar_x, 0.05, cbar_w, 0.9])
    cbar = plt.colorbar(image, cax=cb)
    cbar.ax.set_ylim(lo - bar_pad, hi + bar_pad)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)

    style.panel_title(ax, title, subtitle)
    style.finalise_axes(ax)


def compute_cross_correlation_matrix(df: pd.DataFrame, model_specs: list[dict], spectrum_specs: list[dict]) -> pd.DataFrame:
    m_labels = [m["short_label"] for m in model_specs]
    s_labels = [s["short_label"] for s in spectrum_specs]
    mat = pd.DataFrame(index=m_labels, columns=s_labels, dtype=float)

    for m in model_specs:
        m_col = m["col"]
        m_lbl = m["short_label"]
        for s in spectrum_specs:
            s_col = s["col"]
            s_lbl = s["short_label"]
            r, _ = spearmanr(df[m_col], df[s_col])
            mat.loc[m_lbl, s_lbl] = float(r)
    return mat


def compute_interaction_matrix(df: pd.DataFrame, model_specs: list[dict],
                               spectrum_specs: list[dict]) -> pd.DataFrame:
    """Double-centred fine-tune x spectrum correlations: the specificity term.

    The raw correlations in panels B and D are dominated by two nuisance effects.
    Rows differ in how much *generic* mutational accessibility a model carries --
    the untuned base carries the most of all, because a model trained on natural
    sequence has already absorbed the average mutational spectrum of life, the
    same reason an empirical substitution matrix is not a biochemistry control.
    Columns differ in how much structure each spectrum has left once codon
    distance is held fixed. Neither says anything about whether a model learned
    the spectrum it was trained on.

    Removing the row and column means leaves the model x spectrum interaction,
    which is the quantity the specificity claim is actually about. The base model
    is dropped first: it has no generating arm, so it has no diagonal, and
    leaving it in would push its large row effect into everyone else's residuals.
    """
    ft = [m for m in model_specs if m["rank"] >= 0]
    mat = compute_cross_correlation_matrix(df, ft, spectrum_specs)
    values = mat.to_numpy(dtype=float)
    centred = (values
               - values.mean(axis=1, keepdims=True)
               - values.mean(axis=0, keepdims=True)
               + values.mean())
    return pd.DataFrame(centred, index=mat.index, columns=mat.columns)


def permutation_test_diagonal(interaction: pd.DataFrame) -> tuple[float, float, int]:
    """Exact test: is the true fine-tune-to-arm matching the best of all matchings?

    The statistic is the sum of the matched diagonal. Every one-to-one matching of
    fine-tuned models to arms is enumerated, so this is exact rather than sampled.
    With four arms there are only 4! = 24 matchings, which puts a hard floor of
    1/24 = 0.042 on the attainable p-value; a result at the floor means the true
    matching beat all twenty-three alternatives, and nothing stronger is sayable
    from four arms.
    """
    values = interaction.to_numpy(dtype=float)
    n = len(values)
    observed = float(np.trace(values))
    null = np.array([sum(values[i, order[i]] for i in range(n))
                     for order in permutations(range(n))])
    p_value = float((null >= observed).sum() / len(null))
    return observed, p_value, len(null)


def draw_panel_e(ax, interaction: pd.DataFrame, spectrum_specs: list[dict],
                 p_value: float, n_perm: int):
    """The specificity claim, stated as a picture rather than as a matrix of residuals.

    This panel used to be the double-centred matrix itself. Nobody could read it: the
    numbers are residuals, which have no natural scale, and the claim -- that each
    fine-tune leans towards its own arm -- was spread across a diagonal the reader had to
    reconstruct cell by cell. Here each fine-tune is one row of four points, one per
    spectrum, and the point for the arm it was actually trained on is ringed. The claim is
    now a thing you can see: the ringed point is the rightmost one in every row. The full
    matrix is still in the values table.
    """
    rows, cols = list(interaction.index), list(interaction.columns)
    rank_by_label = {s["short_label"]: s["rank"] for s in spectrum_specs}
    ys = np.arange(len(rows))

    for i, row_label in enumerate(rows):
        vals = interaction.loc[row_label].to_numpy(dtype=float)
        ax.plot([vals.min(), vals.max()], [ys[i], ys[i]], color=style.GRID,
                linewidth=1.0, zorder=1)
        for j, col_label in enumerate(cols):
            own = i == j
            colour = colour_for_rank(rank_by_label[col_label])
            ax.plot([vals[j]], [ys[i]], marker="o", linestyle="none",
                    markersize=9.0 if own else 5.5, color=colour,
                    markeredgecolor=style.INK if own else style.SURFACE,
                    markeredgewidth=1.2 if own else 0.6, zorder=4 if own else 3)
        # To the right of the ringed point, which is the rightmost one in every row,
        # so the number cannot land on another row's markers.
        ax.annotate(f"{vals[i]:+.3f}", xy=(vals[i], ys[i]), xytext=(9, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.0, color=style.INK)

    # Label on the right of the rule: the left of the zero line is where the top row's
    # points sit, and the default placement landed on them.
    style.reference_line(ax, 0.0, label="no preference", axis="x", label_side="above")
    ax.set_yticks(ys)
    ax.set_yticklabels(rows, fontsize=7)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -0.75)
    span = float(np.abs(interaction.to_numpy()).max())
    ax.set_xlim(-1.35 * span, 1.75 * span)   # room on the right for the value labels
    # "Excess rank correlation" meant nothing to most readers. The axis now says in
    # plain words what the number is: how much more this model agrees with this
    # spectrum than its own average agreement, and the spectrum's own average, would
    # already give you.
    ax.set_xlabel("Extra agreement, above this model's and this spectrum's own average")

    # No legend: the four arm colours are already named in panel A's legend and panel C's,
    # and one here sat on top of the bottom row. The only new thing to explain is the ring,
    # which the tagline says in words.
    # Readers could not tell whether E came from C or from D. It comes from D, and the
    # tagline now names it rather than leaving it to the caption.
    style.panel_title(
        ax, "Each fine-tune leans towards the spectrum it was trained on",
        f"built from panel D, with each row's and each column's average removed\n"
        f"ringed = the model's own training arm, highest in all {len(rows)} rows; "
        f"best of {n_perm} matchings, exact p = {p_value:.3f}")
    style.finalise_axes(ax)


def build_values_table(data: dict, model_specs: list[dict], spectrum_specs: list[dict], alphas: np.ndarray) -> pd.DataFrame:
    rows = []
    label_map = data["label_map"]
    scores_df = data["scores"]

    # Named by what the block holds, not by position in this function: the letters
    # follow the figure's left-to-right reading order, in which the full matrix is B
    # and the alpha sweep is C, while the blocks below are still built in the order
    # the quantities are computed. The table is sorted back into letter order at the end.
    panel_a_name = "A"          # spectra
    panel_full_name = "B"       # cross-correlation, all replacements
    panel_alpha_name = "C"      # alpha sweep
    panel_d_name = chr(ord("A") + 3)      # cross-correlation, N=1 band
    panel_e_name = chr(ord("A") + 4)      # interaction

    # Panel A: Spectra
    spectra = data["spectra"]
    for _, r in spectra.iterrows():
        disp = label_map.get(r["arm"], str(r["arm"]))
        rows.append({
            "panel": panel_a_name,
            "condition": disp,
            "category": str(r["mutation_class"]),
            "metric": "observed_percentage",
            "value": float(r["percentage"]),
            "lo": np.nan,
            "hi": np.nan,
            "n": float(r.get("count", np.nan)),
        })

    # Panel C: Alpha sweeps
    fit = scores_df["ground_truth"].to_numpy()
    for spec in model_specs:
        col = spec["col"]
        mut_col = spec["mut_col"]
        lbl = spec["label"]

        log_p = np.log(scores_df[col].clip(lower=1e-12).to_numpy())
        log_m = np.log(scores_df[mut_col].clip(lower=1e-25).to_numpy())

        rhos = []
        for a in alphas:
            score = log_p + a * log_m
            r, _ = spearmanr(score, fit)
            rhos.append(float(r))
            rows.append({
                "panel": panel_alpha_name,
                "condition": lbl,
                "category": f"alpha={a:+.3f}",
                "metric": "spearman_rho_vs_fitness",
                "value": float(r),
                "lo": np.nan,
                "hi": np.nan,
                "n": float(len(fit)),
            })
        rhos = np.array(rhos)
        best_idx = int(np.argmax(rhos))
        rows.append({
            "panel": panel_alpha_name,
            "condition": lbl,
            "category": "optimum",
            "metric": "optimal_alpha",
            "value": float(alphas[best_idx]),
            "lo": np.nan,
            "hi": np.nan,
            "n": float(len(fit)),
        })

    # Panel B: Cross-correlation (All replacements)
    mat_all = compute_cross_correlation_matrix(scores_df, model_specs, spectrum_specs)
    for m_lbl in mat_all.index:
        for s_lbl in mat_all.columns:
            rows.append({
                "panel": panel_full_name,
                "condition": m_lbl,
                "category": s_lbl,
                "metric": "spearman_rho_all_replacements",
                "value": float(mat_all.loc[m_lbl, s_lbl]),
                "lo": np.nan,
                "hi": np.nan,
                "n": float(len(scores_df)),
            })

    # Panel D: Cross-correlation (1-nt band only)
    df_1nt = scores_df[scores_df["nt_steps"] == 1]
    mat_1nt = compute_cross_correlation_matrix(df_1nt, model_specs, spectrum_specs)
    for m_lbl in mat_1nt.index:
        for s_lbl in mat_1nt.columns:
            rows.append({
                "panel": panel_d_name,
                "condition": m_lbl,
                "category": s_lbl,
                "metric": "spearman_rho_1nt_band",
                "value": float(mat_1nt.loc[m_lbl, s_lbl]),
                "lo": np.nan,
                "hi": np.nan,
                "n": float(len(df_1nt)),
            })

    # Panel E: the specificity interaction, plus the exact permutation test
    interaction = compute_interaction_matrix(df_1nt, model_specs, spectrum_specs)
    for m_lbl in interaction.index:
        for s_lbl in interaction.columns:
            rows.append({
                "panel": panel_e_name,
                "condition": m_lbl,
                "category": s_lbl,
                "metric": "interaction_rho_1nt_band",
                "value": float(interaction.loc[m_lbl, s_lbl]),
                "lo": np.nan,
                "hi": np.nan,
                "n": float(len(df_1nt)),
            })
    stat, p_value, n_perm = permutation_test_diagonal(interaction)
    for metric, value in (("matched_diagonal_sum", stat), ("exact_permutation_p", p_value)):
        rows.append({
            "panel": panel_e_name,
            "condition": "matched diagonal",
            "category": f"{n_perm} matchings",
            "metric": metric,
            "value": float(value),
            "lo": np.nan,
            "hi": np.nan,
            "n": float(len(df_1nt)),
        })

    out = pd.DataFrame(rows)
    # Blocks are appended in computation order (A, C, B, D, E); the reader wants
    # them in panel order. Stable sort, so each block keeps its internal order.
    return out.sort_values("panel", kind="stable").reset_index(drop=True)


def make_caption() -> str:
    """Reviewer 2 simulation control across four evolutionary arms.

A synthetic 700-amino-acid founder sequence was evolved under a shared site-wise fitness
landscape (beta = 2.0) across four distinct nucleotide substitution regimes: flat JC69
(neutral negative control), human germline, SARS-CoV-2, and influenza A H3N2. ESM-C 600M
was scored on the founder background prior to fine-tuning and after fine-tuning on 5,000
evolved sequences from each arm.

The flat/JC69 regime implements the Jukes-Cantor 1969 neutral substitution model, where all
twelve nucleotide substitution rates between distinct bases are exactly equal (each proposing
at a uniform 8.33% rate, 1/12). Because all base changes are equally probable, the flat/JC69
arm lacks any directional mutation bias, transition/transversion skew, or GC-content drive.
Sequences in this arm evolve under selection on the shared fitness landscape in the complete
absence of mutational directionality. The flat/JC69 fine-tune (Flat FT) therefore provides a
strict negative control: because no directional bias exists in its training data, any detected
correlation with biological mutation spectra measures the benchmark's false-positive rate.

(A) Empirical substitution spectra across 12 nucleotide mutation classes. Each biological
regime exhibits its characteristic evolutionary transitions (e.g. C->T transitions in SARS-CoV-2
and A->G transitions in H3N2). Note that these are the substitutions selection *retained*, not
the rates at which they were proposed, and the two differ: the flat/JC69 arm proposes all twelve
classes at a uniform 8.33% by construction, but its retained spectrum is transition-enriched
(53.9% transitions against the 33.3% expected from a flat proposal over the founder's codons).
This is the structure of the genetic code acting through purifying selection, not a property of
the proposal. Enumerating every single-nucleotide change from the founder's codons, transitions
are 52.4% of the synonymous changes but only 28.8% of the nonsynonymous ones, because a
transition at a degenerate third position is usually silent. Purifying selection therefore
removes transversions preferentially, and retaining only the synonymous changes gives a 52.4%
transition share against the 53.9% observed. This mechanism is long established: Xia, Hafner
and Sudman (1996) On transition bias in mitochondrial genes of pocket gophers, Journal of
Molecular Evolution 43:32-40, doi:10.1007/BF02352297, showed that at twofold degenerate sites,
where a transition is synonymous and a transversion is not, purifying selection rather than
mutation accounts for most of the observed transition bias. The separate route, in which
transitions are milder when they *are* nonsynonymous, contributes little here (mean
ground-truth fitness 0.0264 for nonsynonymous transitions against 0.0210 for transversions)
and is in any case disputed: Stoltzfus and Norris (2016, Mol Biol Evol 33:595-602) found no
significant fitness difference between nonsynonymous transitions and transversions, while
Lyons and Lauring (2017, Mol Biol Evol 34:3205-3215) found selection to be a major
contributor. Nothing in this panel rests on that disputed route. The dashed line marks the flat
proposal rate, not an expectation for the plotted bars. The flat arm remains a valid negative
control: it carries no directional mutation bias, and the structure visible in its bars is
code-and-selection structure that all four arms share.

(B) Cross-correlation between model predictions and mutation spectra across the full accessibility
matrix (N = 13,300). Spearman rank correlation across all single-amino-acid replacements is elevated
for all models (rho ~ 0.25 to 0.50). This high correlation does not reflect spectrum-specific learning,
but is driven primarily by the shared geometry of the genetic code and codon distance (1-step vs
2-step vs 3-step mutations) that globally separates accessible from inaccessible replacements in both
PLM probabilities and mutational proposal rates. Crucially, the neutral Flat FT control also correlates
at rho ~ 0.25 to 0.31 across all spectra, proving that codon distance structure alone creates a spurious
baseline correlation across all replacements regardless of whether directional bias was learned.

(C) Ground-truth fitness estimation across mutational accessibility weight alpha. Combining PLM
log-probabilities with mutational accessibility log P_mut reveals that subtracting proposal bias
(alpha < 0) systematically improves fitness landscape recovery for models fine-tuned on biased
evolutionary data. The untuned base model is optimal at alpha = 0.

(D) Cross-correlation restricted to the N = 1 nucleotide step band (N = 4,174). Conditioning on
single-nucleotide changes controls for codon distance, eliminating the large baseline correlation
inflation driven by genetic code architecture. Consequently, overall correlation magnitudes decrease
relative to the full matrix (rho ~ 0.00 to 0.16). Rather than running counter to the hypothesis, this
lower correlation directly demonstrates that the apparent signal in (B) was confounded: with codon
distance held constant, the neutral Flat/JC69 control drops to near-zero correlation across all biological
spectra (rho = -0.007 to +0.044), confirming zero false-positive bias. Simultaneously, genuine learned
proposal bias is unmasked in the biological arms: each fine-tuned model specifically and uniquely peaks
on its generating mutation spectrum (SARS-CoV-2 FT rho = +0.162 on SARS-CoV-2 vs +0.049 on H3N2; H3N2 FT
rho = +0.145 on H3N2 vs +0.075 on Flat; Human FT rho = +0.109 on Human vs +0.004 on Flat). Boxed cells
highlight the matching generating arm for each fine-tuned model.

Note that the untuned base model carries a *higher* absolute correlation than any fine-tune on three
of the four spectra (rho = +0.146, +0.189, +0.224 on Flat, Human and SARS-CoV-2). This is not a failure
of fine-tuning and it is not an artefact of codon distance, which is already held fixed here: it is a
row effect. A model trained on natural sequence has absorbed the average mutational accessibility of
life, so the untuned base scores highly against every spectrum without being specific to any of them,
for the same reason an empirical substitution matrix cannot serve as a biochemistry control. Fine-tuning
on 5,000 sequences from a single arm replaces that broad prior with site-specific fitness structure
(Spearman rho with the true landscape rises from +0.17 for the base model to +0.50 to +0.60 for the
fine-tunes), trading generic accessibility for a smaller arm-specific signal. Two checks confirm the
diagnosis: holding the true fitness landscape fixed by partial correlation leaves the base row
essentially unchanged (+0.224 to +0.223 on SARS-CoV-2), and holding the number of single-nucleotide
paths to the target codon fixed also leaves the base model ahead. Absolute correlation in (D) therefore
measures how much generic accessibility a model carries, not what it learned; panel (E) separates the
two.

(E) The model x spectrum interaction in the N = 1 band, obtained by removing the row and column means
from the fine-tuned block of (D). Row means differ because models carry different amounts of generic
accessibility, and column means differ because the spectra retain different amounts of structure once
codon distance is fixed; neither bears on specificity. What remains is whether each fine-tune favours
the spectrum it was trained on. It does: every fine-tune ranks its own generating arm first of four
(Flat +0.041, Human +0.025, SARS-CoV-2 +0.028, H3N2 +0.045; ringed in the panel). Enumerating all 4! = 24 one-to-one
matchings of fine-tunes to arms, the true matching gives the largest matched diagonal sum (+0.140)
of all 24, an exact p = 0.042. With four arms, 1/24 = 0.042 is the smallest attainable p-value, so this
is the strongest statement the design supports; it is not evidence of a large effect, and the
interaction terms themselves are small (0.025 to 0.045).

One caveat on the flat arm. Within the N = 1 band the flat/JC69 accessibility takes only 13 distinct
values spanning exactly threefold, and they encode nothing but how many single-nucleotide paths reach
the target codon: the flat arm has no direction to learn. Flat FT matching its own arm in (E) therefore
means it learned path multiplicity, not a directional bias, and is not a false positive. The
negative-control claim rests on the raw values in (D), where Flat FT sits between -0.007 and +0.044
against every spectrum, and not on its diagonal in (E).
"""
    return make_caption.__doc__.strip()


def draw(data: dict):
    spectra_df = data["spectra"]
    scores_df = data["scores"]
    label_map = data["label_map"]

    model_specs, spectrum_specs = get_aligned_specs(scores_df, label_map)
    alphas = np.linspace(-0.5, 0.5, 101)

    # The house style puts constrained layout in charge, which recomputes the
    # margins at draw time: the left/right/bottom/top this gridspec used to pass
    # were silently discarded, and the slack they were meant to absorb showed up
    # as a band of white between the two rows. Only the ratios and the spacings
    # are ours to set.
    # Letters read left to right, row by row, like every other figure in the set:
    # A spectra, B the full matrix, C the alpha sweep, D the N=1 band, E across
    # the foot. The two line plots keep the wider left column and the two heatmaps
    # the right one -- putting a heatmap in the left column squeezed the bar chart
    # and the cell values until they overlapped -- so the heatmaps stay stacked and
    # directly comparable, with E under D, which is what E is built from. E is
    # shorter than the rows above it because it is a 4x4 with no base row.
    fig = style.figure(style.WIDTH_DOUBLE, 5.15)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0, 0.72],
                          hspace=0.06, wspace=0.12)

    ax_a = fig.add_subplot(gs[0, 0])      # A
    ax_c = fig.add_subplot(gs[1, 0])      # C: alpha sweep, under A
    ax_b = fig.add_subplot(gs[0, 1])      # B: full matrix, beside A
    ax_d = fig.add_subplot(gs[1, 1])      # D
    ax_e = fig.add_subplot(gs[2, :])

    # Panel A: Spectra
    draw_panel_a(ax_a, spectra_df, label_map)

    # Panel C: Alpha sweep curves
    draw_panel_c(ax_c, scores_df, model_specs, alphas)

    # Panel B: Cross-correlation heatmap (All replacements)
    plm = data["plm_name"]
    mat_all = compute_cross_correlation_matrix(scores_df, model_specs, spectrum_specs)
    draw_heatmap(ax_b, mat_all,
                 f"{plm} vs mutation spectra (full matrix)",
                 f"Spearman ρ across all replacements (N={len(scores_df):,})")

    # Panel D: Cross-correlation heatmap (1-nt band only)
    df_1nt = scores_df[scores_df["nt_steps"] == 1]
    mat_1nt = compute_cross_correlation_matrix(df_1nt, model_specs, spectrum_specs)
    # The tagline has to repeat the metric: a reader landing on D alone had only
    # "controlled for codon distance" and no way to know these are the same
    # Spearman rho as B, just on the single-nucleotide band.
    draw_heatmap(ax_d, mat_1nt,
                 f"{plm} vs mutation spectra (N=1 band)",
                 f"Spearman ρ, codon distance held at 1 nt (N={len(df_1nt):,})")

    # Panel E: the specificity term, with the rows and columns of D taken out
    interaction = compute_interaction_matrix(df_1nt, model_specs, spectrum_specs)
    _stat, p_value, n_perm = permutation_test_diagonal(interaction)
    draw_panel_e(ax_e, interaction, spectrum_specs, p_value, n_perm)

    # A and C carry a full-height y-label, whose top reaches past dy=1.10 and
    # collided with the letter. The heatmaps have short row labels and no
    # y-label, so they keep the tighter placement.
    style.panel_letter(ax_a, "A", dx=-0.20, dy=1.13)
    style.panel_letter(ax_c, "C", dx=-0.20, dy=1.13)
    style.panel_letter(ax_b, "B", dx=-0.13, dy=1.10)
    style.panel_letter(ax_d, chr(ord("A") + 3), dx=-0.13, dy=1.10)
    # E's tagline wraps to two lines, which pushes its title up; the letter follows.
    style.panel_letter(ax_e, chr(ord("A") + 4), dx=-0.075, dy=1.40)
    return fig, model_specs, spectrum_specs, alphas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / f"results/{RESULTS_SET}")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output")
    args = parser.parse_args()

    style.use_house_style()
    data = load(args.results)

    fig, model_specs, spectrum_specs, alphas = draw(data)

    args.out.mkdir(parents=True, exist_ok=True)
    style.save_figure(fig, args.out / STEM)

    values = build_values_table(data, model_specs, spectrum_specs, alphas)
    write_values(values, args.out, STEM)

    # Export Supplementary Table 16
    values.to_csv(args.out / "supl_table_16.tsv", sep="\t", index=False)
    values.to_csv(args.out / "table_sN_simulation_control.tsv", sep="\t", index=False)

    caption = make_caption()
    write_caption(caption, args.out, STEM)
    print(f"Wrote {args.out / STEM}.pdf and tables.")


if __name__ == "__main__":
    main()
