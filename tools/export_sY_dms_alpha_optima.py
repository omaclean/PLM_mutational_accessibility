#!/usr/bin/env python3
"""Per-panel alpha optima behind the DMS alpha sweep: the table and its caption.

Derived entirely from the shipped figure values tables, which are the same analysis the
archive called S16_dms_alpha_vs_measured_fitness. One row per panel x model.

Both tables are needed. ``make_fig04.py`` writes the influenza panels to
``fig04_alpha_vs_dms_values.tsv`` and the SARS-CoV-2 panels to ``sL_dms_sc2_values.tsv``;
reading only the first gives 8 rows where the table has 18, and drops every SARS-CoV-2 panel
without a word (found by rebuilding the figures into a scratch directory, 16 Oct 2026).

The caption is written here too, from the table just built. It used to be hand-written
beside a second, byte-identical copy of the table under another number, which is how the
repo came to carry two supplementary tables both labelled S16 and a caption calling one of
them S15 (docs/notes/extended_red_team_tests.md A1). Every number the caption quotes is now read off
the rows, so a rebuild on different data writes different sentences instead of false ones,
and the label comes from the numbering map like every other caption's.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mutacc.io import write_caption  # noqa: E402

REPO = Path(sys.argv[1])
#: Optional second argument: read the figure values from, and write the table to, this
#: directory instead of ``<repo>/output`` (``tools/build_figures.py`` builds into a scratch one).
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "output"
SOURCES = [OUT_DIR / "fig04_alpha_vs_dms_values.tsv", OUT_DIR / "sL_dms_sc2_values.tsv"]
STEM = "sY_dms_alpha_optima"
OUT = OUT_DIR / f"{STEM}.tsv"

COLS = [
    "dataset", "trait", "panel_label", "organism", "citation",
    "model_name", "model_label", "model_role",
    "n", "n_sites",
    "alpha_best", "rho_best", "rho_best_ci_lo", "rho_best_ci_hi",
    "rho_alpha0", "rho_alpha0_ci_lo", "rho_alpha0_ci_hi",
    "delta_rho", "delta_rho_in_size", "accessibility_helps",
    "boundary", "weak_signal", "interpretable",
]


def close(a, b):
    return abs(float(a) - float(b)) < 1e-9



def _signed_range(values) -> str:
    """``"-0.21 to -0.13"``, or the single value where there is only one."""
    lo, hi = min(values), max(values)
    return f"{lo:+.2f}" if lo == hi else f"{lo:+.2f} to {hi:+.2f}"


def caption(rows: list[dict]) -> str:
    """The caption, with every quoted number read off ``rows``.

    Nothing here is asserted: if the data stops supporting a sentence, the sentence changes.
    A caption that hard-codes its own findings is the failure this replaces.
    """
    # Receptor binding is not fitness, and many replacements raise it, so a correlation there
    # is a property of the assay: those rows stay in the table and out of the claims, exactly
    # as the figures treat them.
    binding = [r for r in rows if "binding" in r["trait"].lower()]
    good = [r for r in rows
            if r["interpretable"] == "True" and "binding" not in r["trait"].lower()]
    by_role = {role: [r for r in good if r["model_role"] == role] for role in ("base", "finetune")}
    unusable = [r for r in rows if r["interpretable"] != "True"]
    at_edge = sum(1 for r in unusable if r["boundary"] == "True")
    weak = sum(1 for r in unusable if r["weak_signal"] == "True")
    deltas = {role: [float(r["delta_rho"]) for r in grp] for role, grp in by_role.items()}
    alphas = {role: [float(r["alpha_best"]) for r in grp] for role, grp in by_role.items()}
    n_panels = len({(r["dataset"], r["trait"]) for r in rows})
    scans = ", ".join(sorted({r["citation"] for r in rows}))

    helps = (f"Dividing accessibility out improves the fine-tuned models by "
             f"{min(deltas['finetune']):.3f} to {max(deltas['finetune']):.3f} in rho and the base "
             f"models by {min(deltas['base']):.3f} to {max(deltas['base']):.3f}."
             if deltas["base"] and deltas["finetune"] else "")
    sides = (f"On the {len(good)} interpretable fitness rows the fine-tuned models put their best weight at "
             f"{_signed_range(alphas['finetune'])} and the base models at "
             f"{_signed_range(alphas['base'])}. {helps}"
             if deltas["base"] and deltas["finetune"] else
             f"{len(good)} of {len(rows)} rows carry an interpretable optimum.")

    return f"""
    **Best weight on mutational accessibility for each deep mutational scan.** One row for
    each measured phenotype and each model, summarising the {n_panels} panels of the two DMS
    sweep figures ({scans}).

    Every single amino-acid replacement in a scan is scored as log(model probability) + alpha
    x log(accessibility). Alpha is the weight on mutational accessibility: 0 is the model on
    its own, negative values divide accessibility out, positive values add more of it.
    Accessibility is the probability, under the nucleotide change spectrum measured for that
    virus, that the reference codon mutates to a codon for the replacement's amino acid.
    `rho` is the Spearman rank correlation between that score and the laboratory measurement.
    The model is run on each scan's own reference sequence, and its probability for a
    replacement is the masked marginal: the scored site is replaced by the mask token and
    predicted from its context, so the residue actually there cannot inform its own
    prediction. The figures' own supplement shows what the wild-type (unmasked) estimator
    gives instead.

    Columns. `dataset`, `trait` and `panel_label` name the scan and the phenotype measured,
    `panel_label` being the name the figures use; `organism` and `citation` give the virus and
    the source publication; `model_name`, `model_label` and `model_role` name the scoring
    model, each panel pairing a base (untuned) model with the same architecture fine-tuned on
    that virus; `n` and `n_sites` count the replacements scored and the protein sites they sit
    at; `alpha_best` is the alpha giving the highest `rho`, searched over -1.00 to +1.00 in
    steps of 0.01; `rho_best` and `rho_alpha0`, each with a 95% interval, are the correlation
    at `alpha_best` and for the model alone; `delta_rho` is the difference between them, which
    is what reweighting accessibility buys, and `delta_rho_in_size` is that difference taken
    in size. The two differ only where a correlation is negative: the sweep places every
    optimum by the size of the correlation, because a receptor-binding curve that grows more
    negative has a stronger relationship, not a weaker one, and a signed difference reads
    backwards there.

    Three flags. `accessibility_helps` is True where `rho_best` is larger in size than either
    end of the alpha = 0 interval -- a weak screen, not a paired test, since the two values
    come from the same resamples. `boundary` is True where the best alpha sits at the edge of the search
    range, so the curve has no interior optimum and `alpha_best` is not an estimate.
    `weak_signal` is True where the scan carries too little signal to locate an optimum at
    all. `interpretable` is False where either of the last two fires: **read `alpha_best` only
    on rows where it is True.**

    Intervals are 95% percentile intervals from a cluster bootstrap over DMS sites, because
    replacements at one site share a backbone and are not independent. They are bootstrap
    percentiles, not t-intervals.

    **What the table shows.** {sides} The {len(binding)} receptor-binding rows are left out of
    those counts: binding is not fitness, and many replacements raise it, so the sign of a
    correlation there is a property of the assay rather than of the model.

    **{len(unusable)} of {len(rows)} rows are not interpretable** and `alpha_best` must not be
    quoted for them. {weak} carry too little signal at alpha = 0 for a weight to improve on,
    and {at_edge} put the best alpha at the edge of the search range, where the curve rises
    without ever turning over. The two flags are not exclusive, so those counts may overlap.

    **Source.** The shipped values tables of the two DMS sweep figures, which hold the full
    201-point sweep: this table takes the row at each curve's best alpha and the row at alpha
    = 0, and checks both against the `best_alpha`, `rho_best` and `rho_alpha0` columns those
    files carry (it stops if they disagree beyond 1e-9).
    """


def main():
    needed = {"dataset", "trait", "model_name", "alpha", "best_alpha", "rho", "rho_best", "rho_alpha0"}
    rows = []
    for src in SOURCES:
        part = list(csv.DictReader(src.open(), delimiter="\t"))
        if not part or not needed <= set(part[0]):
            raise SystemExit(f"{src} is not a DMS values table: {len(part)} rows, "
                             f"missing columns {sorted(needed - set(part[0] if part else []))}")
        rows += part
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["trait"], r["model_name"]), []).append(r)

    out = []
    for key, grp in groups.items():
        head = grp[0]
        best_alpha = head["best_alpha"]
        at_best = [r for r in grp if close(r["alpha"], best_alpha)]
        at_zero = [r for r in grp if close(r["alpha"], 0.0)]
        if len(at_best) != 1 or len(at_zero) != 1:
            raise SystemExit(f"{key}: expected one row at best alpha and one at 0, "
                             f"got {len(at_best)} and {len(at_zero)}")
        b, z = at_best[0], at_zero[0]

        # Guard: the stored summary columns must agree with the curve itself.
        if not close(b["rho"], head["rho_best"]):
            raise SystemExit(f"{key}: rho at best alpha {b['rho']} != rho_best {head['rho_best']}")
        if not close(z["rho"], head["rho_alpha0"]):
            raise SystemExit(f"{key}: rho at alpha 0 {z['rho']} != rho_alpha0 {head['rho_alpha0']}")

        delta = float(b["rho"]) - float(z["rho"])
        out.append({
            "dataset": head["dataset"], "trait": head["trait"],
            "panel_label": head["panel_label"], "organism": head["organism"],
            "citation": head["citation"], "model_name": head["model_name"],
            "model_label": head["model_label"], "model_role": head["model_role"],
            "n": head["n"], "n_sites": head["n_sites"],
            "alpha_best": f"{float(best_alpha):+.2f}",
            "rho_best": f"{float(b['rho']):.6f}",
            "rho_best_ci_lo": f"{float(b['ci_lo']):.6f}",
            "rho_best_ci_hi": f"{float(b['ci_hi']):.6f}",
            "rho_alpha0": f"{float(z['rho']):.6f}",
            "rho_alpha0_ci_lo": f"{float(z['ci_lo']):.6f}",
            "rho_alpha0_ci_hi": f"{float(z['ci_hi']):.6f}",
            "delta_rho": f"{delta:+.6f}",
            # The sweep picks each optimum by the SIZE of the correlation, because the
            # receptor-binding panels correlate negatively and a curve that grows more
            # negative is a stronger relationship, not a weaker one. A signed difference
            # therefore reads backwards on those rows, so the size is carried beside it and
            # is the column the flag below uses.
            "delta_rho_in_size": f"{abs(float(b['rho'])) - abs(float(z['rho'])):+.6f}",
            # Does reweighting accessibility help, beyond the band at alpha = 0? Compared in
            # size, against whichever end of that band is further from zero.
            "accessibility_helps": str(abs(float(b["rho"]))
                                       > max(abs(float(z["ci_lo"])), abs(float(z["ci_hi"])))),
            "boundary": head["boundary"], "weak_signal": head["weak_signal"],
            "interpretable": head["interpretable"],
        })

    out.sort(key=lambda r: (r["organism"], r["dataset"], r["trait"], r["model_role"]))
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT} ({len(out)} rows)")
    path = write_caption(caption(out), OUT_DIR, STEM)
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
