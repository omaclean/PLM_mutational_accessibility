"""S13: where in the protein does the model's coupling signal recover real 3D contacts, and does fine-tuning help there?

    python figures/make_sH_contact_recovery.py --guide guides/h3n2_13lineage.yaml

The model is scanned substitution by substitution; how strongly a change at one position moves the
predictions at another gives a coupling score for every pair of positions, on the zero-sum scale of
Zhang and Ovchinnikov (2024), corrected for average product and symmetrised. Pairs at least six
positions apart are ranked by that score and compared with real contacts (C-beta within 8 Angstrom
in a reference structure, minimum over the three protomers). AUROC is the chance that a true contact
outranks a non-contact: 0.5 is chance, 1 is perfect.

Reads ``contact_strata_benchmark`` (AUROC per structural stratum, per structure),
``contact_residue_accuracy`` (the same per residue), ``contact_positional_accuracy`` (the archive's
own moving averages), ``contact_regions`` (domain boundaries) and ``contact_surface_vs_core`` (an
earlier table, used only to cross-check). ``epistasis_focal_sites`` marks the clade sites. ``antigenic_sites`` and
``antigenic_residue_accuracy`` add one more stratum, the published antigenic residues, taken as a
single class rather than the five named sites and on the zero-sum gauge only (this was
figure S8, which is a negative result and is folded in here rather than standing alone).

Checks on every run: the moving averages are recomputed from the per-residue table and must match
the archive's to its printed precision; every interval flag, difference and base rate is recomputed;
the regions must tile the sequence exactly and agree with the HA1/HA2 label of every focal site; the
earlier surface-vs-core table must agree with the strata table.

New here, not in the archive: the archive's interval resamples PAIRS, but a residue sits in many
pairs, so pairs are not independent and that interval is too narrow. ``*_resid`` columns resample
residues instead (see tools/export_s07_s13.py). Both are drawn.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mutacc import antigenic, style
from mutacc.guide import Guide, GuideError, parse_figure_args
from mutacc.io import write_caption, write_values

STEM = "sH_contact_recovery"
WINDOW = 25
MIN_PERIODS = 8              # as in the archive: a window with fewer than 8 residues is blank
PRINT_TOL = 5.1e-5           # the archive prints four decimals
FEW_CONTACTS = 30            # strata with fewer contacts than this are flagged as thin
SIGNAL_PEPTIDE = 16
SEP_MIN, CONTACT_A = 6, 8     # provenance: pairs at least this far apart; contact = C-beta within this many Angstrom
STRATA_COLUMNS = {"structure", "category", "n_pairs", "n_contacts", "base_rate", "auroc_base", "auroc_ft", "delta",
                  "delta_lo", "delta_hi", "crosses_zero", "delta_lo_resid", "delta_hi_resid", "crosses_zero_resid"}
RESIDUE_COLUMNS = {"structure", "res_id", "modelled", "n_contacts", "auroc_ft", "auroc_base", "delta"}
MARKERS = ("o", "s", "^", "v")
#: Antigenic residues against the rest of the domain. Taken from the house categorical palette,
#: not the base/fine-tune pair -- every point in that panel is already a fine-tuned-minus-base
#: difference, so blue/green would be misread as the two models.
ANTIGENIC_COLOURS = (style.CATEGORICAL[0], style.CATEGORICAL[1])
ANTIGENIC_LABEL = {antigenic.STRATUM_ANTIGENIC: "Antigenic residues",
                   antigenic.STRATUM_REST: "Rest of the domain"}
LINESTYLES = ("-", ":", "--", "-.")


# ---------------------------------------------------------------------------
# Inputs and checks
# ---------------------------------------------------------------------------

def structures_of(strata: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(strata["structure"]))


def validate_strata(strata: pd.DataFrame) -> None:
    missing = STRATA_COLUMNS - set(strata.columns)
    if missing:
        raise ValueError(f"strata table lacks columns {sorted(missing)}")
    if strata.duplicated(["structure", "category"]).any():
        raise ValueError("duplicate (structure, category) rows in the strata table")
    numeric = ["auroc_base", "auroc_ft", "delta", "n_pairs", "n_contacts", "delta_lo", "delta_hi",
               "delta_lo_resid", "delta_hi_resid", "base_rate"]
    if strata[numeric].isna().any().any():
        raise ValueError("NaN in a strata AUROC, interval or count")
    if not strata[["auroc_base", "auroc_ft"]].apply(lambda c: c.between(0, 1)).all().all():
        raise ValueError("an AUROC outside [0, 1]")
    if (strata["n_contacts"] > strata["n_pairs"]).any() or (strata["n_contacts"] <= 0).any():
        raise ValueError("contact counts are not between 1 and the number of pairs")
    cats = strata.groupby("structure")["category"].apply(set)
    if cats.map(frozenset).nunique() > 1:
        # a stratum can drop out of a structure for being too sparse; a wholly different list cannot
        if not set.intersection(*cats.tolist()):
            raise ValueError("structures share no strata")
    for side in ("", "_resid"):
        bad = (strata[f"delta_lo{side}"] > strata[f"delta_hi{side}"]).any()
        if bad:
            raise ValueError(f"an interval has its low end above its high end ({side or 'pair'} bootstrap)")


def check_strata(strata: pd.DataFrame, surface_core: pd.DataFrame | None = None) -> dict[str, float]:
    """Recompute every derived column of the strata table and compare. Raises ValueError."""
    validate_strata(strata)
    gaps = {
        "delta": float((strata["auroc_ft"] - strata["auroc_base"] - strata["delta"]).abs().max()),
        "base_rate": float((100 * strata["n_contacts"] / strata["n_pairs"] - strata["base_rate"]).abs().max()),
    }
    if gaps["delta"] > 1.1e-4:
        raise ValueError(f"delta is not fine-tuned minus base AUROC (off by {gaps['delta']:.2e})")
    if gaps["base_rate"] > 1.1e-4:
        raise ValueError(f"base rate is not contacts over pairs (off by {gaps['base_rate']:.2e})")
    # intervals are rounded to four decimals, so a flag may disagree only when an end is within rounding of zero
    for side, flag in (("", "crosses_zero"), ("_resid", "crosses_zero_resid")):
        lo, hi = strata[f"delta_lo{side}"], strata[f"delta_hi{side}"]
        want = (lo < 0) & (hi > 0)
        edge = (lo.abs() < 1e-4) | (hi.abs() < 1e-4)
        if ((want != strata[flag].astype(bool)) & ~edge).any():
            raise ValueError(f"{flag} disagrees with the interval it describes")
    if surface_core is not None:
        first = strata[strata["structure"] == strata["structure"].iloc[0]].set_index("category")
        sc = surface_core.set_index("category")
        shared = first.index.intersection(sc.index)
        if len(shared) == 0:
            raise ValueError("surface-vs-core table shares no strata with the strata table")
        for col in ("n_pairs", "n_contacts", "auroc_base", "auroc_ft"):
            diff = (first.loc[shared, col].astype(float) - sc.loc[shared, col].astype(float)).abs()
            if diff.isna().any():
                raise ValueError(f"surface-vs-core table has missing {col} values")
            gap = float(diff.max())
            tol = 0.0 if col.startswith("n_") else PRINT_TOL * 1.01
            if gap > tol:
                raise ValueError(f"surface-vs-core table disagrees with the strata table on {col} by {gap:.2e}")
            gaps[f"surface_core_{col}"] = gap
    return gaps


def validate_residues(residues: pd.DataFrame) -> None:
    missing = RESIDUE_COLUMNS - set(residues.columns)
    if missing:
        raise ValueError(f"residue table lacks columns {sorted(missing)}")
    for name, g in residues.groupby("structure"):
        ids = g["res_id"].to_numpy()
        if len(ids) != len(set(ids)) or ids.min() != 1 or ids.max() != len(ids):
            raise ValueError(f"{name}: residue ids are not 1..N without repeats")
        scored = g["auroc_ft"].notna()
        if (scored & ~g["modelled"].astype(bool)).any():
            raise ValueError(f"{name}: a residue not resolved in the structure has an AUROC")
        for col in ("auroc_ft", "auroc_base"):
            if not g[col].dropna().between(0, 1).all():
                raise ValueError(f"{name}: {col} outside [0, 1]")
        if ((g["auroc_ft"] - g["auroc_base"] - g["delta"]).abs().max() > 1.1e-4):
            raise ValueError(f"{name}: per-residue delta is not fine-tuned minus base")
        if (g.loc[scored, "n_contacts"] <= 0).any():
            raise ValueError(f"{name}: a residue has an AUROC but no contacts")
    if len(set(residues.groupby("structure").size())) != 1:
        raise ValueError("structures cover different numbers of residues")


def moving_average(values: pd.Series, window: int = WINDOW, min_periods: int = MIN_PERIODS) -> pd.Series:
    """Centred moving average over residue positions, blank where fewer than ``min_periods`` are scored."""
    return values.rolling(window, center=True, min_periods=min_periods).mean()


def positional_series(residues: pd.DataFrame) -> pd.DataFrame:
    """Moving averages of the per-residue AUROCs, one column per structure and model."""
    out = None
    for name, g in residues.groupby("structure", sort=False):
        g = g.sort_values("res_id").set_index("res_id")
        cols = {f"{name}_ft": moving_average(g["auroc_ft"]), f"{name}_base": moving_average(g["auroc_base"]),
                f"{name}_delta": moving_average(g["delta"])}
        frame = pd.DataFrame(cols)
        out = frame if out is None else out.join(frame)
    return out.reset_index()


def check_positional(mine: pd.DataFrame, shipped: pd.DataFrame, tol: float = PRINT_TOL) -> float:
    """Largest gap between the recomputed and archive moving averages. NaNs must fall in the same places."""
    shared = [c for c in shipped.columns if c != "res_id"]
    missing = set(shared) - set(mine.columns)
    if missing:
        raise ValueError(f"cannot recompute moving averages for {sorted(missing)}")
    if len(mine) != len(shipped) or not (mine["res_id"].to_numpy() == shipped["res_id"].to_numpy()).all():
        raise ValueError("moving-average tables cover different residues")
    worst = 0.0
    for c in shared:
        a, b = mine[c].to_numpy(float), shipped[c].to_numpy(float)
        if (np.isnan(a) != np.isnan(b)).any():
            raise ValueError(f"{c}: blank positions differ from the archive's")
        if (~np.isnan(a)).any():
            worst = max(worst, float(np.nanmax(np.abs(a - b))))
    if worst > tol:
        raise ValueError(f"moving averages differ from the archive's by {worst:.2e} (tolerance {tol:.1e})")
    return worst


def check_regions(regions: pd.DataFrame, sites: pd.DataFrame, n_residues: int) -> int:
    """Domains tile 1..N exactly; the HA2 start implied by them matches every focal site's HA1/HA2 label.

    Returns the first HA2 position. Everything is in mature numbering, so there is no 16-residue
    offset to apply: a site labelled HA1 must lie before the HA2 start, and HA2 position 1 is the
    HA2 start itself.
    """
    if not {"kind", "name", "start", "end"} <= set(regions.columns):
        raise ValueError("region table lacks kind/name/start/end")
    dom = regions[regions["kind"] == "domain"].sort_values("start")
    if dom.empty:
        raise ValueError("no domain regions")
    if (dom["start"] > dom["end"]).any():
        raise ValueError("a region ends before it starts")
    if int(dom["start"].iloc[0]) != 1 or int(dom["end"].iloc[-1]) != n_residues:
        raise ValueError(f"domains run {int(dom['start'].iloc[0])}-{int(dom['end'].iloc[-1])}, not 1-{n_residues}")
    if (dom["start"].to_numpy()[1:] != dom["end"].to_numpy()[:-1] + 1).any():
        raise ValueError("domains overlap or leave a gap")
    ha2 = dom[dom["name"].str.startswith("HA2")]
    if ha2.empty:
        raise ValueError("no HA2 region")
    ha2_start = int(ha2["start"].min())
    if (dom[dom["name"].str.startswith("HA1")]["end"].max() != ha2_start - 1):
        raise ValueError("HA1 does not end where HA2 starts")
    for r in sites.itertuples():
        in_ha2 = r.seq_pos >= ha2_start
        if (r.region == "HA2") != in_ha2:
            raise ValueError(f"{r.focal}: labelled {r.region} but position {r.seq_pos} is "
                             f"{'after' if in_ha2 else 'before'} the HA2 start {ha2_start}")
        if r.region == "HA2" and r.region_pos != r.seq_pos - ha2_start + 1:
            raise ValueError(f"{r.focal}: HA2 position {r.region_pos} is not {r.seq_pos} - {ha2_start} + 1")
    return ha2_start


def strata_summary(strata: pd.DataFrame) -> pd.DataFrame:
    """Per structure: how many strata the fine-tuned model clearly beats, clearly trails, or neither, under each interval."""
    rows = []
    for name, g in strata.groupby("structure", sort=False):
        row = {"structure": name, "n_strata": len(g)}
        for side, tag in (("", "pair"), ("_resid", "resid")):
            lo, hi = g[f"delta_lo{side}"], g[f"delta_hi{side}"]
            row[f"n_better_{tag}"] = int((lo > 0).sum())
            row[f"n_worse_{tag}"] = int((hi < 0).sum())
            row[f"n_unclear_{tag}"] = int(((lo <= 0) & (hi >= 0)).sum())
        row["n_thin"] = int((g["n_contacts"] < FEW_CONTACTS).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Antigenic residues as one stratum (was figure S8)
# ---------------------------------------------------------------------------

def antigenic_stratum(sites: pd.DataFrame, res: pd.DataFrame) -> dict:
    """Per-residue gain at antigenic residues against the rest of the domain, on the zero-sum gauge.

    The five published sites are pooled into one class on purpose. Split by site the panel asked
    five uncorrected questions at once and one site came out above the rest in both structures,
    which is the kind of thing five looks produce; as one class the comparison is a single test.
    """
    site_of = antigenic.load_sites(sites)
    d = antigenic.annotate_residues(res, site_of).reset_index(drop=True)
    rng = np.random.default_rng(antigenic.SEED)
    groups = antigenic.group_summary(d, rng)
    rows = []
    for struct in sorted(d["structure"].unique()):
        sub = d.loc[d["structure"] == struct]
        c = antigenic.contrast(sub.loc[sub["antigenic"], "delta"], sub.loc[~sub["antigenic"], "delta"])
        rows.append({"structure": struct, "comparison": "antigenic minus rest", **c})
    return {"groups": groups, "contrasts": pd.DataFrame(rows), "label": antigenic.GAUGE_LABEL,
            "n_sites": len(set(site_of.values())), "structures": sorted(d["structure"].unique())}


def antigenic_values(anti: dict) -> pd.DataFrame:
    # The gauge is a constant now, not a column of the input, so it is written back in: a values
    # file that does not say which scale it is on cannot be checked against anything later.
    return pd.concat([anti["groups"].assign(table="group"),
                      anti["contrasts"].assign(table="contrast")],
                     ignore_index=True, sort=False).assign(gauge=antigenic.GAUGE)


def antigenic_difference(anti: dict, structure: str) -> pd.Series:
    c = anti["contrasts"]
    row = c.loc[c["structure"] == structure]
    if len(row) != 1:
        raise ValueError(f"no antigenic contrast for {structure}")
    return row.iloc[0]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _category_labels(cats, strata: pd.DataFrame, structs: list[str]) -> list[str]:
    """Stratum name followed by its true-contact count in each structure, in panel order.

    One count is not enough: the structure panels share these tick labels, but a stratum holds
    different numbers of contacts in each structure (Stalk Core: 119 in 6WXB, 20 in 7ZJ7), so a
    single structure's count would be read off the wrong panel.
    """
    counts = strata.pivot(index="category", columns="structure", values="n_contacts").reindex(cats)
    out = []
    for c in cats:
        ns = " / ".join("—" if pd.isna(counts.loc[c, s]) else f"{int(counts.loc[c, s]):,}" for s in structs)
        out.append(f"{c}  (n = {ns})")
    return out


def draw_antigenic(ax, anti: dict) -> list:
    """Antigenic residues against the rest of the domain, one row per structure.

    Returns the legend handles for the two strata. The per-residue gain is already a
    fine-tuned-minus-base difference, so the line at zero is "fine-tuning helped neither group".
    """
    structs = anti["structures"]
    g = anti["groups"]
    y = np.arange(len(structs))[::-1].astype(float)
    handles = []
    for off, stratum, colour in ((0.16, antigenic.STRATUM_ANTIGENIC, ANTIGENIC_COLOURS[0]),
                                 (-0.16, antigenic.STRATUM_REST, ANTIGENIC_COLOURS[1])):
        r = g[g["stratum"] == stratum].set_index("structure").reindex(structs)
        err = np.vstack([r["mean_delta"] - r["ci_lo"], r["ci_hi"] - r["mean_delta"]])
        bar = ax.errorbar(r["mean_delta"], y + off, xerr=err, fmt="o", color=colour, markersize=3.5,
                          elinewidth=1.1, capsize=2, zorder=3,
                          label=ANTIGENIC_LABEL[stratum])
        handles.append(bar)
    ax.axvline(0, color=style.INK_MUTED, linewidth=0.7, zorder=1)
    ax.annotate("no change", xy=(0, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2),
                textcoords="offset points", fontsize=7, color=style.INK_MUTED, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(structs)
    # The top of the range is empty by design: the legend sits in it, over no data.
    ax.set_ylim(-0.78, len(structs) - 0.4 + 0.52)
    ax.set_xlabel("Per-residue AUROC gain")
    ax.grid(axis="y", visible=False)
    # The contrast goes inside the axes, not into the tick labels: a wide tick label on a narrow
    # panel steals width from every other panel in the row through the constrained layout.
    # The two bars overlapping is not the test, so the test is written out next to them: which
    # structure it belongs to, what the interval is an interval of, and what the p-value came from.
    for yy, struct in zip(y, structs):
        a = antigenic_difference(anti, struct)
        ax.annotate(f"{struct}, antigenic − rest: {a['difference']:+.3f} "
                    f"(95% bootstrap [{a['ci_lo']:+.3f}, {a['ci_hi']:+.3f}]; "
                    f"label-permutation test, {int(a['n_perm']):,} shuffles: p = {a['p_perm']:.2f})",
                    xy=(0.015, yy - 0.42), xycoords=("axes fraction", "data"), fontsize=7,
                    color=style.INK, va="center", ha="left")
    style.panel_title(ax, "Antigenic residues",
                      f"{anti['label']} gauge; the sites pooled as one class\n"
                      f"Bars are 95% bootstrap intervals over residues "
                      f"({antigenic.N_BOOT:,} resamples), not a test of the difference")
    return handles


def draw(strata: pd.DataFrame, residues: pd.DataFrame, regions: pd.DataFrame, sites: pd.DataFrame,
         base_label: str, ft_label: str, max_res: int, anti: dict | None = None):
    structs = structures_of(strata)
    cats = list(dict.fromkeys(strata["category"]))
    y = np.arange(len(cats))[::-1]

    # The antigenic panel gets a row of its own rather than a fourth column in the top row: the
    # leftmost top panel pays for the full-length stratum labels, and a fourth column starves it
    # until the constrained layout collapses every axes in the row to zero width.
    fig = style.figure(style.WIDTH_DOUBLE, 9.1 if anti else 8.6)
    # Two structures and a legend, nothing more: give the row only the height that holds them,
    # or the panel reads as mostly empty next to the dense ones above and below it.
    rows = [1.6] + ([0.58] if anti else []) + [1.0, 0.8]
    gs = fig.add_gridspec(len(rows), 1, height_ratios=rows)
    top = gs[0].subgridspec(1, len(structs) + 1, width_ratios=[1.0] * len(structs) + [1.15])
    ax_s = [fig.add_subplot(top[0, 0])]
    for k in range(1, len(structs)):
        ax_s.append(fig.add_subplot(top[0, k], sharey=ax_s[0], sharex=ax_s[0]))
    ax_d = fig.add_subplot(top[0, len(structs)], sharey=ax_s[0])
    # Its own y axis: the rows are structures, not the strata the top row shares. Full width with
    # its legend and its numbers inside it -- half width left a quarter of the row empty, and the
    # legend in that gap was further from the marks it names than a legend sitting on them.
    ax_a = fig.add_subplot(gs[1]) if anti else None
    ax_p = fig.add_subplot(gs[len(rows) - 2])
    ax_q = fig.add_subplot(gs[len(rows) - 1], sharex=ax_p)

    base_c, ft_c = style.colour_for_role("base"), style.colour_for_role("finetune")

    # --- A, B: AUROC by stratum, base to fine-tuned ---------------------------------------
    for ax, name in zip(ax_s, structs):
        g = strata[strata["structure"] == name].set_index("category").reindex(cats)
        for yy, r in zip(y, g.itertuples()):
            if np.isnan(r.auroc_base):
                continue
            ax.plot([r.auroc_base, r.auroc_ft], [yy, yy], color=style.NULL_GREY, linewidth=1.2, zorder=2)
        ax.scatter(g["auroc_base"], y, s=26, color=base_c, zorder=3, label=base_label, linewidths=0)
        ax.scatter(g["auroc_ft"], y, s=26, color=ft_c, zorder=3, label=ft_label, linewidths=0)
        style.reference_line(ax, 0.5, "chance = 0.5", axis="x")
        ax.set_xlabel("Contact AUROC")
        ax.grid(axis="y", visible=False)
        style.panel_title(ax, name, "Higher: contacts recovered better")
    ax_s[0].set_yticks(y)
    ax_s[0].set_yticklabels(_category_labels(cats, strata, structs), fontsize=7)
    ax_s[0].set_ylabel(f"Structural stratum\n(n = true contacts, {' / '.join(structs)})")
    for ax in ax_s[1:]:
        ax.tick_params(labelleft=False)
    lo = float(strata[["auroc_base", "auroc_ft"]].min().min())
    hi = float(strata[["auroc_base", "auroc_ft"]].max().max())
    ax_s[0].set_xlim(min(0.4, lo - 0.03), min(1.0, hi + 0.03))
    
    # --- C: change on fine-tuning with both intervals -------------------------------------
    n = len(structs)
    offsets = np.linspace(-0.2, 0.2, n) if n > 1 else [0.0]
    for off, name, mk in zip(offsets, structs, MARKERS):
        g = strata[strata["structure"] == name].set_index("category").reindex(cats)
        yy = y + off
        ax_d.hlines(yy, g["delta_lo_resid"], g["delta_hi_resid"], color=style.NULL_GREY, linewidth=2.6, zorder=2)
        ax_d.hlines(yy, g["delta_lo"], g["delta_hi"], color=style.INK, linewidth=0.9, zorder=3)
        ax_d.scatter(g["delta"], yy, marker=mk, s=16, color=style.INK, zorder=4, linewidths=0)
    ax_d.axvline(0, color=style.INK_MUTED, linewidth=0.7, zorder=1)
    ax_d.set_xlabel("AUROC change on fine-tuning")
    ax_d.grid(axis="y", visible=False)
    ax_d.tick_params(labelleft=False)
    handles = [Line2D([], [], marker=mk, linestyle="", color=style.INK, markersize=3.5, label=name)
               for name, mk in zip(structs, MARKERS)]
    handles += [Line2D([], [], color=style.INK, linewidth=0.9, label="95% interval, resampling pairs"),
                Line2D([], [], color=style.NULL_GREY, linewidth=2.6, label="95% interval, resampling residues")]
    style.panel_title(ax_d, "Fine-tuned minus base", "Right of line: fine-tuning helped")

    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=base_c, markersize=4, label=base_label),
                Line2D([], [], marker="o", linestyle="", color=ft_c, markersize=4, label=ft_label)] + handles,
               loc="outside upper center", ncol=3, fontsize=7.5)

    # --- D: antigenic residues against the rest of the domain -----------------------------
    if ax_a is not None:
        ax_a.legend(handles=draw_antigenic(ax_a, anti), loc="upper right", ncol=2,
                    frameon=False, borderaxespad=0.4)

    # --- E: AUROC along the sequence ------------------------------------------------------
    series = positional_series(residues)
    pos = series["res_id"].to_numpy()
    for name, ls in zip(structs, LINESTYLES):
        ax_p.plot(pos, series[f"{name}_base"], color=base_c, linestyle=ls, linewidth=1.3)
        ax_p.plot(pos, series[f"{name}_ft"], color=ft_c, linestyle=ls, linewidth=1.3)
    style.reference_line(ax_p, 0.5, "chance = 0.5", label_loc="right")
    ax_p.set_ylabel(f"Contact AUROC\n({WINDOW}-residue moving average)")
    handles = [Line2D([], [], color=base_c, label=base_label), Line2D([], [], color=ft_c, label=ft_label)]
    handles += [Line2D([], [], color=style.INK_MUTED, linestyle=ls, label=name) for name, ls in zip(structs, LINESTYLES)]
    ax_p.set_ylim(ax_p.get_ylim()[0], 1.08)
    ax_p.legend(handles=handles, loc="upper left", ncol=4, fontsize=7)
    ax_p.tick_params(labelbottom=False)
    ax_p.set_xlim(0.5, max_res + 0.5)

    dom = regions[regions["kind"] == "domain"].sort_values("start")
    for ax in (ax_p, ax_q):
        for k, r in enumerate(dom.itertuples()):
            if k % 2 == 0:
                ax.axvspan(r.start - 0.5, r.end + 0.5, color=style.GRID, alpha=0.45, linewidth=0, zorder=0)
    ha2_start = int(dom[dom["name"].str.startswith("HA2")]["start"].min())
    for ax in (ax_p, ax_q):
        ax.axvline(ha2_start - 0.5, color=style.INK_MUTED, linewidth=0.8, zorder=1)
    for k, r in enumerate(dom.itertuples()):
        ax_p.annotate(r.name, xy=((r.start + r.end) / 2, 1.0), xycoords=("data", "axes fraction"),
                      xytext=(0, 2 + 7 * (k % 2)), textcoords="offset points", ha="center", va="bottom",
                      fontsize=6.5, color=style.INK_MUTED)

    # --- F: change along the sequence -----------------------------------------------------
    for name, ls in zip(structs, LINESTYLES):
        ax_q.plot(pos, series[f"{name}_delta"], color=style.INK, linestyle=ls, linewidth=1.3, label=name)
    ax_q.axhline(0, color=style.NULL_GREY, linewidth=0.8, linestyle="--")
    ax_q.annotate("no change", xy=(1.0, 0), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                  textcoords="offset points", ha="right", va="bottom", fontsize=7, color=style.INK_MUTED)
    ymin = ax_q.get_ylim()[0]
    ax_q.plot(sites["seq_pos"], np.full(len(sites), ymin), marker="|", linestyle="", color=style.INK, markersize=7,
              clip_on=False, label="clade-defining sites")
    ax_q.set_ylabel("AUROC change\n(fine-tuned minus base)")
    ax_q.set_xlabel(f"Residue position, mature numbering (HA1 1 to {ha2_start - 1}; HA2 from {ha2_start}; "
                    f"add {SIGNAL_PEPTIDE} for numbering with the signal peptide)")
    ax_q.legend(loc="upper left", ncol=3, fontsize=7)
    ax_q.grid(axis="x", visible=False)

    row = ax_s + [ax_d] + ([ax_a] if ax_a is not None else [])
    style.panel_letters(row, dx=-0.08, dy=1.2)
    style.panel_letter(ax_p, chr(ord("A") + len(row)), dx=-0.06, dy=1.22)
    style.panel_letter(ax_q, chr(ord("A") + len(row) + 1), dx=-0.06, dy=1.05)
    return fig


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------

def antigenic_paragraph(anti: dict, letter: str) -> str:
    """One sentence describing the antigenic panel. Every number comes from the tables."""
    n_perm = int(anti["contrasts"]["n_perm"].max())
    return (f" Panel {letter}: the same per-residue gain at the published antigenic residues against the "
            f"rest of the same domain, on the {anti['label'].lower()} gauge. The {anti['n_sites']} named sites are "
            f"pooled into one class, so this is a single comparison rather than {anti['n_sites']} uncorrected looks. "
            f"Each bar is a 95% bootstrap interval over residues ({antigenic.N_BOOT:,} resamples) for that group's "
            f"own mean gain; the difference between the groups carries its own 95% bootstrap interval and a "
            f"p-value from permuting the antigenic and rest labels ({n_perm:,} shuffles), because two "
            f"overlapping intervals are not a test of a difference.")


def antigenic_result(anti: dict) -> str:
    """The negative result itself, read off the contrast table."""
    parts = []
    for struct in anti["structures"]:
        r = antigenic_difference(anti, struct)
        g = anti["groups"]
        a = g[(g["structure"] == struct) & (g["stratum"] == antigenic.STRATUM_ANTIGENIC)].iloc[0]
        b = g[(g["structure"] == struct) & (g["stratum"] == antigenic.STRATUM_REST)].iloc[0]
        parts.append(f"{struct}: {a['mean_delta']:+.3f} at {int(a['n_residues'])} antigenic residues against "
                     f"{b['mean_delta']:+.3f} at {int(b['n_residues'])} others (difference {r['difference']:+.3f}, "
                     f"95% bootstrap [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], permutation p = {r['p_perm']:.2f})")
    c = anti["contrasts"]
    clear = int(((c["ci_lo"] > 0) | (c["ci_hi"] < 0)).sum())
    verdict = ("Fine-tuning is not specialised to the antigenic sites: the gain there is about the gain "
               "elsewhere in the same domain" if clear == 0 else
               f"The difference excludes zero in {clear} of {len(c)} structures")
    return (f" {verdict} — {'; '.join(parts)}. Antigenic positions are joined on the mature-chain "
            f"numbering only; neighbouring residues are not independent, so these intervals are if "
            f"anything too narrow.")



def caption(strata: pd.DataFrame, residues: pd.DataFrame, regions: pd.DataFrame, base_label: str, ft_label: str,
            anti: dict | None = None) -> str:
    summ = strata_summary(strata).set_index("structure")
    structs = list(summ.index)
    whole = strata[strata["category"] == strata["category"].iloc[0]].set_index("structure")
    bits = []
    for s in structs:
        r = summ.loc[s]
        w = whole.loc[s]
        bits.append(f"{s}: over all long-range pairs the AUROC goes from {w.auroc_base:.3f} to {w.auroc_ft:.3f} "
                    f"({w.delta:+.3f}); of {int(r.n_strata)} strata the fine-tuned model is clearly ahead in "
                    f"{int(r.n_better_pair)} and clearly behind in {int(r.n_worse_pair)} by the pair interval, "
                    f"but ahead in {int(r.n_better_resid)} and behind in {int(r.n_worse_resid)} once residues are resampled")
    n_thin = int(summ["n_thin"].max())
    ratio = float(((strata["delta_hi_resid"] - strata["delta_lo_resid"]) / (strata["delta_hi"] - strata["delta_lo"])).replace(np.inf, np.nan).mean())
    per = residues.groupby("structure").apply(lambda g: int(g["modelled"].astype(bool).sum()), include_groups=False)
    resolved = "; ".join(f"{s} resolves {n} residues" for s, n in per.items())
    anti_letter = chr(ord("A") + len(structs) + 1)
    pos_letter = chr(ord("A") + len(structs) + (2 if anti else 1))
    delta_letter = chr(ord("A") + len(structs) + (3 if anti else 2))
    anti_panel = "" if anti is None else antigenic_paragraph(anti, anti_letter)
    anti_result = "" if anti is None else antigenic_result(anti)
    return f"""
    **Does the model's coupling signal recover real 3D contacts, and where?**
    Substituting each residue and reading how the model's predictions move at every other position gives a
    coupling score for each pair of positions (the categorical Jacobian of Zhang and Ovchinnikov 2024, on the
    zero-sum scale, average-product corrected and symmetrised). Pairs at least {SEP_MIN} positions apart are ranked
    by that score and compared with real contacts, a C-beta distance under {CONTACT_A} Angstrom in a reference structure
    (minimum over the three protomers). AUROC is the chance that a true contact outranks a non-contact: 0.5 is
    chance. Two reference structures are shown because the answer differs between them: {resolved}.

    Panels A and B: AUROC in each structural stratum for {base_label} and {ft_label}; the
    grey segment joins them. Strata are by burial (core and surface are the lower and upper thirds of
    neighbour counts in that structure) and by domain (head versus stalk, using the boundary the pipeline
    hard-codes). Panel C: the change on fine-tuning with two 95% intervals. The thin interval resamples pairs, as
    the original analysis did. The wide grey one resamples residues, which is the right unit because one
    residue sits in hundreds of pairs and its pairs share its fate; it is new, and on average {ratio:.1f} times as
    wide.{anti_panel} Panel {pos_letter}: AUROC for each residue's own partners, as a {WINDOW}-residue moving
    average along the sequence, blank where fewer than {MIN_PERIODS} residues are resolved. Panel
    {delta_letter}: the same difference. Shading
    alternates by domain; ticks mark the clade-defining sites.

    {'; '.join(bits)}. {n_thin} strata hold fewer than {FEW_CONTACTS} true contacts and should not be read
    individually. The head and stalk boundary is hard-coded and known to be unstable, so the head-stalk
    stratum carries that caveat.{anti_result} Positions are mature numbering throughout; the first residue of the HA2 region is
    position {int(regions[(regions['kind'] == 'domain') & regions['name'].str.startswith('HA2')]['start'].min())}.
    """


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_figure_args(__doc__.split("\n\n")[0], argv)
    guide: Guide = args.guide
    strata = guide.read_table("contact_strata_benchmark", args.results)
    residues = guide.read_table("contact_residue_accuracy", args.results)
    positional = guide.read_table("contact_positional_accuracy", args.results)
    regions = guide.read_table("contact_regions", args.results)
    surface = guide.read_table("contact_surface_vs_core", args.results)
    sites = guide.read_table("epistasis_focal_sites", args.results)
    anti = antigenic_stratum(guide.read_table("antigenic_sites", args.results),
                             guide.read_table("antigenic_residue_accuracy", args.results))

    gaps = check_strata(strata, surface)
    validate_residues(residues)
    n_res = int(residues.groupby("structure")["res_id"].max().iloc[0])
    check_regions(regions, sites, n_res)
    gap = check_positional(positional_series(residues), positional)

    base = guide.models_with_role("base")
    ft = guide.models_with_role("finetune")
    if len(base) != 1 or len(ft) != 1:
        raise GuideError("S13 needs exactly one base and one fine-tuned model in the guide")

    fig = draw(strata, residues, regions, sites, base[0].label, ft[0].label, n_res, anti)
    paths = style.save_figure(fig, args.out / STEM)

    write_values(strata, args.out, STEM)
    write_values(positional_series(residues), args.out, f"{STEM}_along_sequence")
    write_values(strata_summary(strata), args.out, f"{STEM}_summary")
    write_values(antigenic_values(anti), args.out, f"{STEM}_antigenic")
    write_caption(caption(strata, residues, regions, base[0].label, ft[0].label, anti), args.out, STEM)
    print(f"self-check: moving averages agree with the archive to {gap:.1e}; strata agree to {max(gaps.values()):.1e}")
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
