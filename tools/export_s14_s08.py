"""Build the tables behind supplementary figure S14 (lineage-set validity audit) and the antigenic
panel of S13.

The antigenic comparison was figure S8 until it was folded into S13: it is a null, so it is now one
panel there, with the five published sites pooled into a single class and only the zero-sum gauge
drawn. This file keeps its old name until the figures are renumbered.

The exported tables still carry all five gauges, and deliberately: they are what the archive
produced, and the manifest ties them to it. Nothing downstream reads the other four -- since
15 Oct 2026 ``mutacc.antigenic.select_gauge`` drops them as the figure loads the table. Do not
narrow the export to match the figure; that would break the provenance chain for no gain.

    python tools/export_s14_s08.py --source-repo ../ [--n-boot 2000] [--only sZ|s09]

Kept apart from ``export_tables.py`` on purpose. Two kinds of table come out:

* **Re-derived.** Every S14 table is computed here from tables the repository already ships
  (``replacement_scores``, ``replacement_observed``, the lineage roots and the spectrum), by
  ``mutacc.audit``, with the seeds the archive used. ``tests/test_s14.py`` rebuilds the cheap
  parts and pins the numbers to the archive's. The one outside input is the fine-tuning corpus
  alignment, which is GISAID-free but large, so only its per-column amino-acid counts ship
  (``corpus_membership``).
* **Copied.** The antigenic per-residue accuracy and pair benchmark come from structure files and
  551 x 551 coupling maps (15 MB each) that are not shipped, so the archive's numbers are reshaped,
  not recomputed. The intervals and contrasts are recomputed from the per-residue table at draw time.

Every table gets a MANIFEST row, appended in one write.
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

from export_base import MANIFEST_COLUMNS, Writer, append_manifest, sha256  # noqa: E402,F401
from mutacc import audit  # noqa: E402
from mutacc.accessibility import BASES, reference_codons  # noqa: E402
from mutacc.guide import load_guide  # noqa: E402

PERSISTENCE_SEED = 20260928
HA1_END = 329  # the archive's manifest puts the start of HA2 at 330
CORPUS = Path("data") / "ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.aln.fas"
SITES = Path("data") / "h3_antigenic_sites.csv"
RESIDUE_ACCURACY = Path("dev") / "fig4_gauge" / "residue_contact_accuracy.tsv"
PAIR_BENCHMARK = Path("dev") / "fig4_gauge" / "antigenic_pair_benchmark.tsv"

GAUGE_LABELS = {
    "prob": "Probability (published)", "logratio": "Log-ratio", "nonwt": "Non-wild-type",
    "wt": "Wild-type reference", "zerosum": "Zero-sum", "KL": "KL divergence", "JSD": "JS divergence",
}
HEADLINE_GAUGE = "zerosum"
RESIDUE_GAUGES = ["prob", "logratio", "nonwt", "wt", "zerosum"]
PAIR_GAUGES = RESIDUE_GAUGES + ["KL", "JSD"]
STRATA = {  # archive label -> (plain label, antigenic residues in the pair)
    "Antigenic - Antigenic": ("both residues antigenic", 2),
    "Antigenic - Other": ("one residue antigenic", 1),
    "Other - Other": ("neither residue antigenic", 0),
    "All long-range": ("all long-range pairs", np.nan),
}

# The Export record, Writer, append_manifest and sha256 are shared with the other
# exporters so every MANIFEST is written the same way; see tools/export_base.py.


def read_alignment(path: Path) -> tuple[str, list[str]]:
    seqs, cur = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    seqs.append("".join(cur))
    return seqs[0], seqs[1:]


# ---------------------------------------------------------------------------
# S14
# ---------------------------------------------------------------------------

def spectrum_rates(path: Path) -> dict:
    q = pd.read_csv(path, sep="\t", index_col=0).loc[list(BASES), list(BASES)].to_numpy(float)
    return audit.reversible_rates(q)


def codon_table(guide) -> pd.DataFrame:
    positions = np.arange(1, 567)
    frames = [pd.DataFrame({"lineage": lin.id, "position": positions,
                            "ref_codon": reference_codons(lin.root, positions)}) for lin in guide.lineages]
    return pd.concat(frames, ignore_index=True)


def model_keys(guide) -> tuple[str, str]:
    ft, bs = guide.models_with_role("finetune"), guide.models_with_role("base")
    if len(ft) != 1 or len(bs) != 1:
        raise ValueError(f"{guide.source.name}: need exactly one fine-tuned and one base model")
    return ft[0].key, bs[0].key


def export_run(guide_path: Path, spectrum: Path, n_boot: int, n_perm: int, extra: bool, source_repo: Path) -> None:
    guide = load_guide(guide_path)
    scores = guide.read_table("replacement_scores")
    out = Writer(guide.results_dir)
    ft, bs = model_keys(guide)
    tag = f"n_boot={n_boot}, seed={audit.SEED}"
    src = f"generated by tools/export_s14_s08.py from replacement_scores.tsv.gz ({tag})"

    within, by_lineage = [], []
    for model in (ft, bs):
        r = audit.within_replacement(scores.loc[scores["model"] == model], n_boot=n_boot)
        within.append({"model": model, "auc": r["auc"], "ci_lo": r["lo"], "ci_hi": r["hi"], "n_boot": n_boot,
                       "seed": audit.SEED, "n_rows": r["n_rows"], "n_lineages": r["n_lineages"],
                       "n_blocks_scored": r["n_blocks_scored"], "n_pairs_scored": r["n_pairs_scored"]})
        by_lineage += [{"model": model, "lineage": k, "auc": v} for k, v in r["per_lineage"].items()]
    out.add("audit_within_replacement.tsv.gz", pd.DataFrame(within), src,
            "S14 panel A: within-replacement AUROC of nucleotide distance, site-clustered interval")
    out.add("audit_within_replacement_by_lineage.tsv.gz", pd.DataFrame(by_lineage), src,
            "S14 panel A: the same statistic scored in each lineage alone")

    rates = spectrum_rates(spectrum)
    site = audit.persistence_site_table(scores, codon_table(guide), rates)
    summary, per_aa = audit.persistence(site, n_perm=n_perm, seed=PERSISTENCE_SEED)
    summary["seed"] = PERSISTENCE_SEED
    psrc = (f"generated by tools/export_s14_s08.py from replacement_scores.tsv.gz, the lineage roots and the "
            f"symmetrised spectrum (n_perm={n_perm}, seed={PERSISTENCE_SEED})")
    out.add("audit_persistence.tsv.gz", summary, psrc,
            "S14 panel B: mean within-residue correlation of synonymous fraction with escape, permutation null")
    out.add("audit_persistence_by_residue.tsv.gz", per_aa, psrc, "S14 panel B: the per-residue correlations")

    if extra:
        observed = guide.read_table("replacement_observed")
        interval = []
        for model in (ft, bs):
            r = audit.interval_unit(scores.loc[scores["model"] == model], observed, n_boot=n_boot)
            interval.append({"model": model, **r, "seed": audit.SEED})
        out.add("audit_interval_unit.tsv.gz", pd.DataFrame(interval),
                f"generated by tools/export_s14_s08.py from replacement_scores.tsv.gz and replacement_observed.tsv.gz ({tag})",
                "S14 panel D: AUROC interval from resampling lineages against resampling sites")

        corpus_path = source_repo / CORPUS
        backbone, training = read_alignment(corpus_path)
        membership = audit.corpus_membership(backbone, training)
        out.add("corpus_membership.tsv.gz", membership, str(CORPUS), "S14 panel C: how often each replacement "
                "occurs in the fine-tuning corpus (per-column counts only; positions are precursor numbering)",
                sha=sha256(corpus_path))
        split = audit.corpus_split(scores, observed, membership, finetune=ft, base=bs, n_boot=n_boot)
        split["seed"] = audit.SEED
        out.add("audit_corpus_split.tsv.gz", split,
                f"generated by tools/export_s14_s08.py from replacement_scores, replacement_observed and "
                f"corpus_membership ({tag})",
                "S14 panel C: fine-tuned minus base AUROC on replacements absent from, and present in, the corpus")
    out.write_manifest()


# ---------------------------------------------------------------------------
# Antigenic sites (S13 panel D; was figure S8)
# ---------------------------------------------------------------------------

def residue_long(path: Path) -> pd.DataFrame:
    wide = pd.read_csv(path, sep="\t")
    wide = wide.loc[wide["modelled"].astype(bool)]
    parts = []
    for gauge in RESIDUE_GAUGES:
        part = wide[["structure", "res_id", "n_contacts", "n_partners"]].copy()
        part["gauge"] = gauge
        part["auroc_finetune"] = wide[f"auroc_ft_{gauge}"]
        part["auroc_base"] = wide[f"auroc_base_{gauge}"]
        part["delta"] = wide[f"delta_auroc_{gauge}"]
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long = long.loc[long["delta"].notna()].copy()
    long["in_domain"] = long["res_id"] <= HA1_END
    long["gauge_label"] = long["gauge"].map(GAUGE_LABELS)
    long["headline"] = long["gauge"] == HEADLINE_GAUGE
    return long.sort_values(["structure", "gauge", "res_id"]).reset_index(drop=True)


def pair_long(path: Path) -> pd.DataFrame:
    wide = pd.read_csv(path, sep="\t")
    rows = []
    for _, r in wide.iterrows():
        label, ends = STRATA[r["Stratum"]]
        for gauge in PAIR_GAUGES:
            rows.append({"stratum": label, "antigenic_ends": ends, "n_pairs": int(r["Pairs"]),
                         "n_contacts": int(r["Contacts"]), "contact_rate_pct": r["Base rate (%)"],
                         "gauge": gauge, "gauge_label": GAUGE_LABELS[gauge], "headline": gauge == HEADLINE_GAUGE,
                         "auroc_finetune": r[f"AUROC FT {gauge}"], "auroc_base": r[f"AUROC Base {gauge}"],
                         "delta": r[f"Delta {gauge}"]})
    return pd.DataFrame(rows)


def export_s09(source_repo: Path) -> None:
    guide = load_guide(CLEAN / "guides" / "h3n2_13lineage.yaml")
    out = Writer(guide.results_dir)
    sites = pd.read_csv(source_repo / SITES)
    out.add("antigenic_sites.tsv.gz", sites, str(SITES),
            "S13 panel D: the antigenic-site residues, in mature (HA1) and precursor (HA0) numbering, joined on HA1 only",
            sha=sha256(source_repo / SITES))
    out.add("antigenic_residue_accuracy.tsv.gz", residue_long(source_repo / RESIDUE_ACCURACY), str(RESIDUE_ACCURACY),
            "S13 panel D: per-residue contact AUROC, fine-tuned and base, per structure and gauge; only the "
            "headline gauge is drawn, the rest keep the bootstrap in the order the archive used",
            sha=sha256(source_repo / RESIDUE_ACCURACY), dropped="precision-at-10 columns, unmodelled residues")
    out.add("antigenic_pair_benchmark.tsv.gz", pair_long(source_repo / PAIR_BENCHMARK), str(PAIR_BENCHMARK),
            "Pair-level contact AUROC by number of antigenic residues in the pair; no figure draws it since "
            "S8 was folded into S13, kept because it is shipped and checked",
            sha=sha256(source_repo / PAIR_BENCHMARK))
    out.write_manifest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-repo", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--only", choices=["sZ", "s09"])
    args = ap.parse_args(argv)
    spectrum = CLEAN / "results" / "h3n2_13lineage" / "dms_spectrum_influenza_h3n2.tsv.gz"
    source = args.source_repo.resolve()
    if args.only in (None, "sZ"):
        export_run(CLEAN / "guides" / "h3n2_13lineage.yaml", spectrum, args.n_boot, args.n_perm, True, source)
        export_run(CLEAN / "guides" / "h3n2_5lineage.yaml", spectrum, args.n_boot, args.n_perm, False, source)
    if args.only in (None, "s09"):
        export_s09(source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
