"""Export the derived tables behind S7 (epistasis against a null) and S13 (contact recovery by region).

A one-off bridge from the exploration archive, like ``export_tables.py`` (which it does not edit).
It re-derives the numbers from the archive's own scan store and contact maps, asserts they agree
with the archive's published tables, and only then writes small gzipped TSVs.

    python tools/export_s07_s13.py --source-repo ../ [--set h3n2_13lineage] [--n-boot 400]

What is NOT shipped, and why:

* The 440 MB scan store (``logprob.f32.npy``) and the two 15 MB contact-map archives: too large.
  What ships instead is every number the figures read, the full pooled null (92,692 draws per
  gauge, 4 gauges) so the p-values can be recomputed, and the per-residue accuracy table.
  S7 draws only the zero-sum gauge (``make_sG_epistasis_null.select_gauge`` drops the rest at load), but all
  four are still exported: they are what the archive produced and what the manifest pins. Do
  not narrow ``GAUGES`` to match the figure -- the shipped tables would stop matching their
  manifest rows and the cross-gauge facts in ``tests/test_sG_epistasis_null.py`` would have nothing to read.
* The PDB/mmCIF structures. They are public deposits, but a figure engine should not need them;
  the per-residue table already carries which residues each structure resolves.

Needs the scan store, the contact maps and the two CIFs, so it runs only where the archive lives.
Needs numpy, pandas, scipy and matplotlib.

Manifest rows are appended to ``results/<set>/MANIFEST.tsv`` in one write. A file already listed
there with identical shape and source hash is left alone; one that differs is an error unless
``--replace-manifest`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd

CLEAN = Path(__file__).resolve().parents[1]

GAUGES = ("wt", "logit", "prob", "zerosum")   # the four the archive's null table carries
N_BOOT = 400
SEED = 20261009
SEP_MIN, CONTACT_A = 6, 8.0
L_EXPECTED = 551

# Region boundaries, mature numbering. Derived from the archive's own data (see
# plot_positional_accuracy.py): the fusion peptide is checked against the backbone below.
REGIONS = [
    ("domain", "HA1 N-terminus and esterase domain", 1, 51),
    ("domain", "HA1 head (Cys52 to Cys277)", 52, 277),
    ("domain", "HA1 stem-side subdomain", 278, 329),
    ("domain", "HA2 fusion peptide", 330, 352),
    ("domain", "HA2 stalk", 353, 551),
    ("strata_head_definition", "Head as used for the strata (hard-coded in the pipeline)", 50, 320),
]
HA2_START = 330
FUSION_PEPTIDE_START = "GIFGAIAGFIE"

MANIFEST_COLUMNS = ["file", "source", "source_sha256_16", "rows", "columns", "dropped", "used_for"]


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# S7: epistasis cells and the pooled null
# ---------------------------------------------------------------------------

def export_epistasis(src: Path) -> dict[str, tuple[pd.DataFrame, list[str], str, str]]:
    """Return {file: (frame, sources, dropped, why)} for the S7 tables."""
    sys.path.insert(0, str(src / "scripts" / "figure4_epistasis"))
    nul = load_module(src / "scripts" / "figure4_epistasis" / "fig4_null_analysis.py", "_arch_null")

    store = src / "tmp_explore" / "fig4_null" / "ESM2-HA80"
    manifest, logp, _done, rows = nul.load_store(store)
    backbone = manifest["backbone_sequence"]
    focal_muts, canon = list(manifest["focal_muts"]), list(manifest["focal_canon"])
    L, ha2_start = int(manifest["L"]), int(manifest["ha2_start"])
    if L != L_EXPECTED or ha2_start != HA2_START or len(backbone) != L:
        raise SystemExit(f"store geometry changed: L={L}, ha2_start={ha2_start}, backbone={len(backbone)}")
    if not backbone[HA2_START - 1:].startswith(FUSION_PEPTIDE_START):
        raise SystemExit("HA2 does not start at mature position 330 (0-based index 329) in the backbone")
    if backbone[51] != "C" or backbone[276] != "C":
        raise SystemExit("Cys52 / Cys277 are not where the region table says")

    # --- the archive's cell table (with Calpha/Cbeta geometry covariates) is the reference
    ref = pd.read_csv(src / "results" / "figure4_null" / "a28_fig4_null_cells.tsv", sep="\t")
    ref = ref[ref["gauge"].isin(GAUGES)]

    # --- pooled null: the same nine substitutions on every background that avoids their sites
    focal_pos = {int(m[1:-1]) for m in focal_muts}
    mut_rows = rows[rows.kind == "mutant"]
    pool = mut_rows[~mut_rows.pos1.isin(focal_pos)]
    draws = []
    for g in GAUGES:
        for fm, name in zip(focal_muts, canon):
            j0 = int(fm[1:-1]) - 1
            d = nul.cell_deltas(logp, backbone, j0, fm[-1], g)[pool.row.values]
            draws.append(pd.DataFrame({"gauge": g, "focal": name, "delta": np.sort(d)}))
    draws = pd.concat(draws, ignore_index=True)
    draws["delta"] = draws["delta"].map(lambda x: float(f"{x:.12g}"))

    # --- agree with the archive's table: null summaries and p-values from the rounded draws
    worst_p, worst_stat, n_tied = 0.0, 0.0, 0
    for (g, name), grp in draws.groupby(["gauge", "focal"], sort=False):
        null = grp["delta"].to_numpy()
        cells = ref[(ref.gauge == g) & (ref.focal == name)]
        for r in cells.itertuples():
            up, dn, two, n, floor = nul._emp_p(null, r.delta)
            gap_p = max(abs(up - r.p_pooled_up), abs(dn - r.p_pooled_dn), abs(two - r.p_pooled_two))
            worst_p = max(worst_p, gap_p)
            n_tied += gap_p > 1e-9
            worst_stat = max(worst_stat, abs(np.median(null) - r.null_median), abs(null.std(ddof=1) - r.null_sd))
            if n != r.null_n:
                raise SystemExit(f"{g}/{name}: pool size {n} != archive {r.null_n}")
    # A p-value can move by one tied draw: one logit null draw sits 7e-17 from an observed shift,
    # and which side of it rounds up depends on the numpy build. Anything larger is a real change.
    if worst_stat > 1e-9 or worst_p > 2.5 / (int(len(draws) / (len(GAUGES) * len(canon))) + 1) or n_tied > 3:
        raise SystemExit(f"recomputed null disagrees with the archive's table (p {worst_p:.2e}, stats {worst_stat:.2e}, "
                         f"{n_tied} cells differ)")
    worst = worst_p
    print(f"epistasis: pooled null reproduces the archive's p-values and summaries (worst {worst:.1e})")

    # --- the observed shifts must also be reproducible from the store
    lab = dict(zip(focal_muts, canon))
    for g in GAUGES:
        for fm in focal_muts:
            j0 = int(fm[1:-1]) - 1
            delta = nul.cell_deltas(logp, backbone, j0, fm[-1], g)
            for bm in focal_muts:
                if bm == fm or int(bm[1:-1]) == int(fm[1:-1]):
                    continue
                i0 = int(bm[1:-1]) - 1
                brow = nul.row_index(i0, nul.alt_rank_of(backbone, i0, bm[-1]))
                want = ref[(ref.gauge == g) & (ref.focal == lab[fm]) & (ref.background == lab[bm])]["delta"].iloc[0]
                if abs(delta[brow] - want) > 1e-9:
                    raise SystemExit(f"observed shift {g} {lab[bm]}->{lab[fm]} differs from the archive's table")
    print("epistasis: every observed shift reproduces from the store")

    keep = ["gauge", "exclusion", "background", "focal", "bg_pos", "focal_pos", "separation",
            "focal_p0", "focal_logodds0", "delta", "null_n", "null_median", "null_sd", "null_mad",
            "p_pooled_up", "p_pooled_dn", "p_pooled_two", "p_pooled_floor",
            "delta_studentised", "p_stud_up", "p_stud_two", "p_stud_floor",
            "percentile_pooled", "q_bh_pooled", "q_bh_stud", "resolution_limited"]
    cells = ref[keep].reset_index(drop=True)
    dropped = ",".join(c for c in ref.columns if c not in keep)

    sites = []
    for fm, name in zip(focal_muts, canon):
        pos = int(fm[1:-1])
        wt, alt = fm[0], fm[-1]
        if backbone[pos - 1] != wt:
            raise SystemExit(f"{fm}: backbone residue at {pos} is {backbone[pos - 1]}, not {wt}")
        in_ha2 = pos >= ha2_start
        sites.append({
            "focal": name, "index_label": fm, "seq_pos": pos, "wt": wt, "alt": alt,
            "region": "HA2" if in_ha2 else "HA1",
            "region_pos": pos - ha2_start + 1 if in_ha2 else pos,
            "p_wt0": float(np.exp(logp[0, nul.aa_row_of(wt), pos - 1])),
            "p_alt0": float(np.exp(logp[0, nul.aa_row_of(alt), pos - 1])),
        })
    sites = pd.DataFrame(sites)

    base_src = ["tmp_explore/fig4_null/ESM2-HA80/logprob.f32.npy", "tmp_explore/fig4_null/ESM2-HA80/manifest.json"]
    cells_src = ["results/figure4_null/a28_fig4_null_cells.tsv"]
    return {
        "epistasis_cells.tsv.gz": (cells, cells_src, dropped,
            "S7: 72 directed cells x 4 gauges with null placement; contact/nt1/separation comparators dropped"),
        "epistasis_null_draws.tsv.gz": (draws, base_src,
            "all rows except gauge, focal, delta (sorted; row order is not information)",
            "S7: the pooled null, one value per background that avoids the nine sites; p-values are recomputed from it"),
        "epistasis_focal_sites.tsv.gz": (sites, base_src, "",
            "S7, S13: the nine focal substitutions in mature numbering, with HA1/HA2 position and baseline probabilities"),
    }


# ---------------------------------------------------------------------------
# S13: contact recovery by structural region
# ---------------------------------------------------------------------------

def weighted_auroc(inv: np.ndarray, n_groups: int, w: np.ndarray, contact: np.ndarray) -> float:
    """AUROC with case weights; tied scores share a group and count half."""
    wp = np.bincount(inv, weights=w * contact, minlength=n_groups)
    wn = np.bincount(inv, weights=w * ~contact, minlength=n_groups)
    tp, tn = wp.sum(), wn.sum()
    if tp <= 0 or tn <= 0:
        return np.nan
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (tp * tn))


def residue_bootstrap(v_ft, v_bs, contact, ii, jj, modelled_idx, n_boot, rng):
    """95% interval on the fine-tuned minus baseline AUROC, resampling RESIDUES, not pairs.

    A residue appears in many pairs, so pairs are not independent. Each draw gives every
    modelled residue a resampling count (multinomial); a pair is weighted by the product of
    its two residues' counts. This is the product-weight bootstrap for dyadic data and is, if
    anything, slightly conservative.
    """
    _, inv_ft = np.unique(v_ft, return_inverse=True)
    _, inv_bs = np.unique(v_bs, return_inverse=True)
    g_ft, g_bs = inv_ft.max() + 1, inv_bs.max() + 1
    n_res = len(modelled_idx)
    counts_all = np.zeros(int(max(ii.max(), jj.max(), modelled_idx.max())) + 1)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        counts_all[:] = 0
        counts_all[modelled_idx] = rng.multinomial(n_res, np.full(n_res, 1.0 / n_res))
        w = counts_all[ii] * counts_all[jj]
        if w[contact].sum() <= 0 or w[~contact].sum() <= 0:
            continue
        out[b] = weighted_auroc(inv_ft, g_ft, w, contact) - weighted_auroc(inv_bs, g_bs, w, contact)
    ok = out[~np.isnan(out)]
    if ok.size < n_boot * 0.9:
        return np.nan, np.nan, int(ok.size)
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return float(lo), float(hi), int(ok.size)


def export_contacts(src: Path, n_boot: int) -> dict[str, tuple[pd.DataFrame, list[str], str, str]]:
    dev = src / "dev" / "fig4_gauge"
    sys.path.insert(0, str(src))
    pps = load_module(dev / "plot_strata_per_structure.py", "_arch_strata")
    import dev.fig4_gauge.contact_benchmark as cb  # noqa: E402  (the archive's own distance routine)

    ft = dict(np.load(dev / "jacobian_maps.npz"))
    bs = dict(np.load(dev / "jacobian_maps_baseline.npz"))
    L = ft["zerosum"].shape[0]
    if L != L_EXPECTED:
        raise SystemExit(f"contact maps are {L} wide, expected {L_EXPECTED}")

    rng_archive = np.random.default_rng(0)       # the archive draws from ONE stream, in this order
    pps.RNG = rng_archive
    rng_res = np.random.default_rng(SEED)

    strata, srcs = [], []
    for name, meta in pps.STRUCTURES.items():
        dist, _ = cb.trimer_distances(meta["cif"], L)
        burial, head, modelled = pps.features(dist)
        s_ft, s_bs = pps.prep(ft["zerosum"]), pps.prep(bs["zerosum"])
        ii, jj = np.triu_indices(L, k=SEP_MIN)
        d = dist[ii, jj]
        ok = np.isfinite(d) & np.isfinite(s_ft[ii, jj]) & np.isfinite(s_bs[ii, jj])
        ii, jj, d = ii[ok], jj[ok], d[ok]
        v_ft, v_bs = s_ft[ii, jj], s_bs[ii, jj]
        contact = d < CONTACT_A
        modelled_idx = np.flatnonzero(modelled)

        ref = pd.read_csv(dev / f"strata_benchmark_{name}.tsv", sep="\t", index_col=0)
        for cat, m in pps.masks(ii, jj, burial, head).items():
            if m.sum() < 50 or contact[m].sum() < 5:
                continue
            r = pps.evaluate(v_ft[m], v_bs[m], contact[m])
            for col in ("n_pairs", "n_contacts", "base_rate", "auroc_base", "auroc_ft", "delta",
                        "delta_lo", "delta_hi", "enrich_base", "enrich_ft"):
                if abs(float(r[col]) - float(ref.loc[cat, col])) > 5.1e-5:
                    raise SystemExit(f"{name}/{cat}/{col}: {r[col]} vs archive {ref.loc[cat, col]}")
            lo, hi, n_ok = residue_bootstrap(v_ft[m], v_bs[m], contact[m], ii[m], jj[m], modelled_idx, n_boot, rng_res)
            row = ref.loc[cat].to_dict()
            row.update({"Category": cat, "structure": name, "delta_lo_resid": lo, "delta_hi_resid": hi,
                        "n_boot_resid": n_ok, "n_residues": int(len(modelled_idx))})
            strata.append(row)
        srcs.append(f"dev/fig4_gauge/strata_benchmark_{name}.tsv")
    print("contacts: strata reproduce the archive's tables to rounding (CI included)")
    strata = pd.DataFrame(strata)
    strata["crosses_zero"] = strata["crosses_zero"].astype(bool)
    strata["crosses_zero_resid"] = (strata["delta_lo_resid"] < 0) & (strata["delta_hi_resid"] > 0)
    strata = strata.rename(columns={"Category": "category"})
    cols = ["structure", "category", "n_pairs", "n_contacts", "base_rate", "auroc_base", "auroc_ft", "delta",
            "delta_lo", "delta_hi", "crosses_zero", "enrich_base", "enrich_ft",
            "delta_lo_resid", "delta_hi_resid", "crosses_zero_resid", "n_boot_resid", "n_residues"]
    strata = strata[cols]

    svc = pd.read_csv(dev / "surface_vs_core_benchmark.tsv", sep="\t")
    svc = svc.rename(columns={"Category": "category", "Pairs": "n_pairs", "Contacts": "n_contacts",
                              "Base Rate (%)": "base_rate", "AUROC Base": "auroc_base", "AUROC FT": "auroc_ft",
                              "Delta AUROC": "delta", "Spearman Base": "spearman_base", "Spearman FT": "spearman_ft",
                              "Top5% Enrich Base": "enrich_base", "Top5% Enrich FT": "enrich_ft",
                              "Delta Enrich": "delta_enrich"})

    acc = pd.read_csv(dev / "residue_contact_accuracy.tsv", sep="\t")
    resid = acc[["structure", "res_id", "modelled", "n_contacts", "n_partners",
                 "auroc_ft_zerosum", "auroc_base_zerosum", "delta_auroc_zerosum"]].rename(columns={
        "auroc_ft_zerosum": "auroc_ft", "auroc_base_zerosum": "auroc_base", "delta_auroc_zerosum": "delta"})
    resid["modelled"] = resid["modelled"].astype(bool)

    pos = pd.read_csv(dev / "positional_accuracy.tsv", sep="\t")
    for name in resid["structure"].unique():
        sub = resid[resid.structure == name].set_index("res_id").reindex(range(1, L + 1))
        for model in ("ft", "base"):
            mine = sub[f"auroc_{model}"].rolling(25, center=True, min_periods=8).mean()
            gap = np.nanmax(np.abs(mine.to_numpy() - pos[f"{name}_{model}"].to_numpy()))
            if gap > 5.1e-5:
                raise SystemExit(f"positional accuracy {name} {model}: moving average off by {gap:.2e}")
    print("contacts: moving averages reproduce positional_accuracy.tsv")

    regions = pd.DataFrame(REGIONS, columns=["kind", "name", "start", "end"])
    arch = ["dev/fig4_gauge/contact_benchmark.py"]
    return {
        "contact_strata_benchmark.tsv.gz": (strata, srcs + ["dev/fig4_gauge/jacobian_maps.npz",
                                                            "dev/fig4_gauge/jacobian_maps_baseline.npz"],
            "pair-level scores (recomputed to check, not shipped)",
            "S13: contact AUROC by structural stratum, both structures; archive pair-bootstrap interval plus a new residue-bootstrap interval"),
        "contact_surface_vs_core.tsv.gz": (svc, ["dev/fig4_gauge/surface_vs_core_benchmark.tsv"], "",
            "S13: earlier single-structure table; adds the Spearman columns and cross-checks the strata table"),
        "contact_residue_accuracy.tsv.gz": (resid, ["dev/fig4_gauge/residue_contact_accuracy.tsv"],
            "the other four gauges, prec10 columns",
            "S13: per-residue contact AUROC (zero-sum gauge), both structures; modelled = resolved in that structure"),
        "contact_positional_accuracy.tsv.gz": (pos, ["dev/fig4_gauge/positional_accuracy.tsv"], "",
            "S13: the archive's 25-residue moving averages; the figure recomputes them and must agree to 5.1e-5"),
        "contact_regions.tsv.gz": (regions, ["dev/fig4_gauge/plot_positional_accuracy.py"], "",
            "S13: region boundaries in mature numbering, checked against the backbone"),
    }


# ---------------------------------------------------------------------------
# Writing and manifest
# ---------------------------------------------------------------------------

def write_all(out_dir: Path, src: Path, exports: dict, replace_manifest: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "MANIFEST.tsv"
    existing = pd.read_csv(manifest_path, sep="\t") if manifest_path.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
    new_rows = []
    for fname, (frame, sources, dropped, why) in exports.items():
        for s in sources:
            if not (src / s).is_file():
                raise SystemExit(f"missing source: {s}")
        buf = io.BytesIO()
        frame.to_csv(buf, sep="\t", index=False, compression="gzip", float_format="%.12g")
        (out_dir / fname).write_bytes(buf.getvalue())
        digest = hashlib.sha256(";".join(sha16(src / s) for s in sources).encode()).hexdigest()[:16]
        row = {"file": fname, "source": ";".join(sources), "source_sha256_16": digest, "rows": len(frame),
               "columns": frame.shape[1], "dropped": dropped, "used_for": why}
        print(f"{fname}: {len(frame)} rows x {frame.shape[1]} cols, {len(buf.getvalue()) / 1e6:.2f} MB")
        old = existing[existing["file"] == fname]
        if old.empty:
            new_rows.append(row)
        elif (old.iloc[0][["rows", "columns", "source_sha256_16"]].astype(str).tolist()
              != [str(row["rows"]), str(row["columns"]), row["source_sha256_16"]]):
            if not replace_manifest:
                raise SystemExit(f"{fname}: manifest row differs from this export; pass --replace-manifest")
            existing = existing[existing["file"] != fname]
            new_rows.append(row)
    if replace_manifest and len(existing):
        # replacing is a whole-file rewrite; only done when asked
        pd.concat([existing, pd.DataFrame(new_rows, columns=MANIFEST_COLUMNS)]).sort_values("file").to_csv(
            manifest_path, sep="\t", index=False)
    elif new_rows:
        text = "".join("\t".join(str(r[c]).replace("\t", " ") for c in MANIFEST_COLUMNS) + "\n" for r in new_rows)
        with manifest_path.open("a") as handle:     # one append write
            handle.write(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-repo", required=True, type=Path)
    ap.add_argument("--set", dest="set_name", default="h3n2_13lineage")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--replace-manifest", action="store_true")
    ap.add_argument("--only", choices=["s08", "s14"], default=None)
    args = ap.parse_args(argv)
    src = args.source_repo.resolve()
    exports = {}
    if args.only in (None, "s08"):
        exports.update(export_epistasis(src))
    if args.only in (None, "s14"):
        exports.update(export_contacts(src, args.n_boot))
    write_all(CLEAN / "results" / args.set_name, src, exports, args.replace_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
