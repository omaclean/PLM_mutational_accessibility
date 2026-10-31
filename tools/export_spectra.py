"""Build the spectrum-comparison tables that S4 and S5 read.

    python tools/export_spectra.py

Unlike the other exporters this one needs no exploration archive: every input is already
in the clean repo, under ``data/spectra/``. It turns each 4x4 nucleotide matrix into the
chance that a codon changes into each amino acid, compares every matrix against every
other, and writes three tables to ``results/spectra_codon_space/``.

What the comparison is for
--------------------------
Reviewer 1 asked whether the manuscript's gap between influenza's own mutation matrix and
Kimura 80 says something about *influenza* or merely about *expressivity*: K80 has one free
nucleotide parameter and the influenza matrix has twelve, so a gap could mean "K80 is too
simple" rather than "K80 is not influenza". Separating the two means adding matrices that
are as expressive as influenza's and just as wrong, which is what the measured spectra of
other organisms are.

The statistic is the manuscript's own: the percent of variance in a target table retained
by a predictor table, 100 * (1 - var(residual) / var(target)). It can go negative, which
means the predictor is worse than the target's own mean. Both published numbers are
compress-and-rebuild statistics, so the tables carry that form as well as the plain
matrix-against-matrix one:

    84.8%   influenza's own table, compressed to 20x20 and rebuilt, against itself
    43.6%   K80 compressed and rebuilt, against influenza  (51.0% without compression)

Every matrix is first scaled to the same total rate, so only the shape of the spectrum
counts, never how fast the organism mutates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CLEAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_base import Writer, sha256  # noqa: E402
import spectra_compare as sc  # noqa: E402

SPECTRA_DIR = Path("data") / "spectra"
SOURCES = SPECTRA_DIR / "spectra_sources.csv"
SET_NAME = "spectra_codon_space"

#: The spectrum the manuscript's claim is about is read from the sources table's
#: ``manuscript_role`` column, so no code here names a virus. ``--focus`` overrides it.
FOCUS_DEFAULT = None

WHY = {
    "spectra_pairwise": (
        "S5: every spectrum against every other in codon space, as percent of variance "
        "retained (compressed and uncompressed) and Pearson r"
    ),
    "spectra_generic_stand_in": (
        "S4: mean agreement of each spectrum with all the others, against its number of "
        "free nucleotide parameters; the non-exclusivity test"
    ),
    "spectra_sources": (
        "S4 and S5: display label, kind, free nucleotide parameters and citation per spectrum, "
        "so the figures name nothing themselves"
    ),
}


def ordered_keys(sources: pd.DataFrame, spectra_dir: Path, focus: str) -> list[str]:
    """Focus spectrum first, then the rest by how well they correlate with it.

    Ordering by agreement rather than alphabetically puts the heatmap's structure on its
    diagonal, which is the whole point of the panel.
    """
    keys = list(sources["key"])
    if focus not in keys:
        raise ValueError(f"focus {focus!r} is not one of {keys}")
    tables = {k: sc.codon_table(sc.acc.load_spectrum(spectra_dir / f"{k}.csv")) for k in keys}
    rest = [k for k in keys if k != focus]
    rest.sort(key=lambda k: -sc.retained_and_corr(tables[focus], tables[k])[1])
    return [focus] + rest


def duplicate_organisms(sources: pd.DataFrame) -> list[str]:
    """Keys to leave out of the stand-in table because another key covers the same organism.

    Two subtypes of the same virus would otherwise inflate each other's mean agreement:
    each counts the other as an "other spectrum" while being almost the same measurement.
    """
    if "organism_group" not in sources.columns:
        return []
    keep_first = sources.dropna(subset=["organism_group"]).drop_duplicates("organism_group", keep="first")
    return [k for k in sources["key"]
            if k in set(sources.dropna(subset=["organism_group"])["key"]) - set(keep_first["key"])]


def build(spectra_dir: Path, focus: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    sources = sc.load_sources(spectra_dir / SOURCES.name)
    if "manuscript_role" not in sources.columns:
        raise ValueError(f"{SOURCES}: needs a manuscript_role column naming the focus spectrum")
    sources["manuscript_role"] = sources["manuscript_role"].fillna("")
    if focus is None:
        marked = sources.loc[sources["manuscript_role"] == "focus", "key"]
        if len(marked) != 1:
            raise ValueError(
                f"{SOURCES}: expected exactly one row with manuscript_role 'focus', found {len(marked)}"
            )
        focus = marked.iloc[0]
    order = ordered_keys(sources, spectra_dir, focus)

    pairwise = sc.pairwise(spectra_dir, sources, order)
    labels = sources.set_index("key")["label"]
    pairwise = pairwise.assign(
        target_label=pairwise["target"].map(labels),
        predictor_label=pairwise["predictor"].map(labels),
        is_focus_target=pairwise["target"].eq(focus),
    )

    excluded = duplicate_organisms(sources)
    stand_in = sc.average_table(pairwise, sources, exclude=tuple(excluded))
    stand_in = stand_in.sort_values("mean_corr_to_others", ascending=False).reset_index(drop=True)
    stand_in = stand_in.assign(is_focus=stand_in["key"].eq(focus))

    ordered_sources = sources.set_index("key").loc[order].reset_index()
    ordered_sources = ordered_sources.assign(
        plot_order=range(len(order)),
        excluded_from_stand_in=ordered_sources["key"].isin(excluded),
        is_focus=ordered_sources["key"].eq(focus),
    )
    return pairwise, stand_in, ordered_sources, excluded


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spectra-dir", type=Path, default=CLEAN / SPECTRA_DIR,
                    help="directory of 4x4 matrices plus spectra_sources.csv")
    ap.add_argument("--focus", default=FOCUS_DEFAULT,
                    help="key of the spectrum the claim is about (default: the first of kind 'virus')")
    args = ap.parse_args(argv)

    pairwise, stand_in, sources, excluded = build(args.spectra_dir, args.focus)
    if excluded:
        print("left out of the stand-in table as a repeated organism:", ", ".join(excluded))

    writer = Writer(CLEAN / "results" / SET_NAME)
    source_note = f"computed by tools/export_spectra.py from {SPECTRA_DIR.as_posix()}"
    sha = sha256(args.spectra_dir / SOURCES.name)
    writer.add("spectra_pairwise.tsv.gz", pairwise, source_note, WHY["spectra_pairwise"], sha=sha)
    writer.add("spectra_generic_stand_in.tsv.gz", stand_in, source_note, WHY["spectra_generic_stand_in"], sha=sha)
    writer.add("spectra_sources.tsv.gz", sources, source_note, WHY["spectra_sources"], sha=sha)
    writer.write_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
