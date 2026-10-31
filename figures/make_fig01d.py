"""Figure 1, panel D: the codon-to-amino-acid accessibility matrix.

    python figures/make_fig01d.py --spectrum data/spectra/<key>.csv

Replaces the frozen `fig1d_codon_to_aa_heatmap.pdf`, which came from a different repository
and had no generator, so its colours, its scale and its tiny tick labels could not be
touched. Everything here is recomputed from the 4x4 nucleotide spectrum in `data/spectra/`
by `mutacc.accessibility.codon_aa_table`, the same function `tests/test_accessibility.py`
pins against the pipeline's shipped `mut_prob` column.

Why the old panel was unreadable
--------------------------------
The quantity spans **fifteen orders of magnitude**: a replacement one nucleotide away sits
near 1e-4, one three nucleotides away near 1e-15. The frozen panel drew that on a *linear*
scale from 0 to 3.5e-4, so every cell below about 5e-5 -- which is almost all of them --
landed in the bottom few percent of the ramp and came out the same dark purple. The palette
was not really the problem; the linear scale was, and no choice of hue fixes it. On a log
scale the same data separates cleanly into the one-, two- and three-step bands, which is the
structure the panel exists to show.

The codon's own amino acid is left blank. That cell is dominated by the probability of the
codon not changing at all, which is not a replacement and would otherwise be the only thing
visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from mutacc import accessibility as acc
from mutacc import style
from mutacc.guide import PANELS_DIR
from mutacc.io import write_values

STEM = "fig01d_codon_accessibility"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = REPO_ROOT / "data/spectra/spectra_sources.csv"

#: The stop codons' symbol, as the genetic code tables write it.
STOP = "*"

#: Tick label sizes. 64 codons across a 7.09 in figure leaves about 6.4 pt of axis per
#: column once the margins are taken off; rotated 90 degrees a label's horizontal extent is
#: its font size, not its length, so the codons can be read at 6 pt without colliding.
#: The old 3.6 pt was set to match the frozen panel this replaced and was illegible in print.
CODON_LABEL_SIZE = 6.4
AA_LABEL_SIZE = 8.2


def focus_spectrum() -> tuple[Path, str]:
    """Path and display label of the spectrum this manuscript is about.

    Chosen by the ``manuscript_role`` column rather than by filename, because a figure
    script may not name a virus (``tests/test_figure_rules.py``).
    """
    sources = pd.read_csv(SOURCES)
    focus = sources[sources["manuscript_role"] == "focus"]
    if len(focus) != 1:
        raise SystemExit(
            f"{SOURCES.name} must mark exactly one spectrum as the focus, found {len(focus)}"
        )
    key = str(focus["key"].iloc[0])
    return SOURCES.parent / f"{key}.csv", str(focus["label"].iloc[0])


def matrix(spectrum_path: Path) -> pd.DataFrame:
    """64 codons x 21 amino acids (stop included) of replacement probability.

    The stop column is taken as the remainder of each row rather than recomputed, so it
    cannot disagree with the tested table beside it: every codon has to turn into some
    codon, so the twenty amino acids and the stops sum to one.
    """
    q = acc.load_spectrum(spectrum_path)
    table = acc.codon_aa_table(q)
    stop = 1.0 - table.sum(axis=1)
    full = np.column_stack([table, np.clip(stop, 0.0, None)])
    targets = list(acc.AA20) + [STOP]
    frame = pd.DataFrame(full, index=acc.CODONS, columns=targets)
    # Group the codons by what they encode, so the matrix keeps its block structure, with
    # the stop codons last rather than first (where the symbol would sort them).
    order = sorted(acc.CODONS,
                   key=lambda c: (acc.GENETIC_CODE[c] == STOP, acc.GENETIC_CODE[c], c))
    return frame.loc[order]


def draw(mat: pd.DataFrame, label: str):
    values = mat.to_numpy(dtype=float).T          # rows = target amino acid
    codons = list(mat.index)
    targets = list(mat.columns)

    # Blank the codon's own amino acid: that cell is mostly the probability of no change.
    synonymous = np.array([[acc.GENETIC_CODE[c] == t for c in codons] for t in targets])
    shown = np.where(synonymous | (values <= 0), np.nan, values)
    logged = np.log10(np.where(np.isnan(shown), np.nan, shown))

    fig = style.figure(style.WIDTH_DOUBLE, 3.6)
    ax = fig.subplots()
    finite = logged[np.isfinite(logged)]
    norm = Normalize(vmin=float(np.floor(finite.min())), vmax=float(np.ceil(finite.max())))
    image = ax.imshow(logged, cmap=style.SEQUENTIAL, norm=norm, aspect="auto",
                      interpolation="nearest")

    ax.set_xticks(range(len(codons)))
    ax.set_xticklabels(codons, rotation=90, fontsize=CODON_LABEL_SIZE, family="monospace")
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=AA_LABEL_SIZE, family="monospace")
    ax.set_xticks(np.arange(-0.5, len(codons), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=0.3)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=1.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = ax.inset_axes([1.012, 0.0, 0.011, 1.0])
    bar = fig.colorbar(image, cax=cb)
    bar.set_label("log10 probability per site", fontsize=7.7)
    bar.ax.tick_params(labelsize=7.15)
    bar.outline.set_visible(False)

    style.finalise_axes(ax, xlabel="Starting codon", ylabel="Replacement amino acid",
                        labelsize=8.8)
    style.panel_title(ax, f"Mutational accessibility of every replacement ({label})",
                      "log scale: replacements require one, two or three nucleotide changes",
                      title_size=9.7, tagline_size=7.15)
    return fig


def values_table(mat: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for codon in mat.index:
        for target in mat.columns:
            rows.append({
                "panel": chr(68),
                "condition": label,
                "codon": codon,
                "ref_aa": acc.GENETIC_CODE[codon],
                "target_aa": target,
                "synonymous": acc.GENETIC_CODE[codon] == target,
                "nt_steps": acc.nucleotide_steps(codon, target) if target != STOP else np.nan,
                "metric": "accessibility_probability",
                "value": float(mat.loc[codon, target]),
            })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    default_path, default_label = focus_spectrum()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spectrum", type=Path, default=default_path,
                        help="4x4 nucleotide spectrum CSV (default: the manuscript's focus spectrum)")
    parser.add_argument("--label", default=default_label, help="display name for the spectrum")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--panels-out", type=Path, default=PANELS_DIR,
                        help=f"where the panel render goes (default: "
                             f"{PANELS_DIR.relative_to(REPO_ROOT).as_posix()}/)")
    args = parser.parse_args(argv)

    style.use_house_style()
    mat = matrix(args.spectrum)
    fig = draw(mat, args.label)
    paths = style.save_figure(fig, args.panels_out / STEM, suffixes=(".pdf",))
    write_values(values_table(mat, args.label), args.out, STEM)
    print("wrote", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
