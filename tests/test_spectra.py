"""S4 and S5: the spectrum comparison engine, its exported tables, and both figures.

The golden values are the manuscript's own published numbers, which the archive script
``a16_cross_species_matrices.py`` reproduced and which this engine must keep reproducing:
the focus spectrum's self-compression and the comparator's score against it.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "figures"))

import spectra_compare as sc  # noqa: E402
import export_spectra as ex  # noqa: E402
import make_sE_spectra_codon_space  # noqa: E402

SPECTRA_DIR = REPO_ROOT / "data" / "spectra"
RESULTS = REPO_ROOT / "results" / ex.SET_NAME

#: The two numbers printed in the manuscript, and the uncompressed form of the second.
PUBLISHED_SELF_COMPRESSION = 84.8
PUBLISHED_COMPARATOR = 43.6
PUBLISHED_COMPARATOR_UNCOMPRESSED = 51.0


@pytest.fixture(scope="module")
def stand_in():
    """The ranking behind S5 panels B and C.

    It outlived S04, which was withdrawn on 13 Oct 2026; the table went unused until the
    generic-matrix comparison was folded into S5 on 14 Oct 2026.
    """
    return pd.read_csv(RESULTS / "spectra_generic_stand_in.tsv.gz", sep="\t")


@pytest.fixture(scope="module")
def sources():
    return pd.read_csv(RESULTS / "spectra_sources.tsv.gz", sep="\t").sort_values("plot_order")


@pytest.fixture(scope="module")
def pairwise():
    return pd.read_csv(RESULTS / "spectra_pairwise.tsv.gz", sep="\t")


def cell(pairwise, target, predictor, column="retained_compressed"):
    row = pairwise[(pairwise["target"] == target) & (pairwise["predictor"] == predictor)]
    assert len(row) == 1, f"{target}/{predictor}: expected one row, found {len(row)}"
    return float(row[column].iloc[0])


def roled(sources, role):
    keys = sources.loc[sources["manuscript_role"] == role, "key"]
    assert len(keys) == 1, f"expected one {role} spectrum, found {len(keys)}"
    return keys.iloc[0]


# ---------------------------------------------------------------------------
# Golden values
# ---------------------------------------------------------------------------

def test_the_published_numbers_are_reproduced(pairwise, sources):
    focus, comparator = roled(sources, "focus"), roled(sources, "comparator")
    assert cell(pairwise, focus, focus) == pytest.approx(PUBLISHED_SELF_COMPRESSION, abs=0.05)
    assert cell(pairwise, focus, comparator) == pytest.approx(PUBLISHED_COMPARATOR, abs=0.05)
    assert cell(pairwise, focus, comparator, "retained_uncompressed") == pytest.approx(
        PUBLISHED_COMPARATOR_UNCOMPRESSED, abs=0.05)


def test_the_engine_recomputes_the_shipped_table(pairwise, sources):
    """The shipped tables are not stale: recomputing from data/spectra gives them back."""
    order = list(sources["key"])
    fresh = sc.pairwise(SPECTRA_DIR, sc.load_sources(SPECTRA_DIR / "spectra_sources.csv"), order)
    merged = fresh.merge(pairwise, on=["target", "predictor"], suffixes=("_fresh", "_shipped"))
    assert len(merged) == len(order) ** 2
    for column in ("retained_compressed", "retained_uncompressed", "corr"):
        assert np.allclose(merged[f"{column}_fresh"], merged[f"{column}_shipped"], atol=1e-9)


def test_a_spectrum_predicts_itself_best_of_all_after_compression(pairwise, sources):
    """Compression can only lose information, so nothing beats a spectrum's own rebuilt table.

    Not a tautology: an organism measured twice may come close, which is why the check is
    against every spectrum except one sharing the target's organism group.
    """
    groups = dict(zip(sources["key"], sources["organism_group"].fillna("")))
    for key in sources["key"]:
        own = cell(pairwise, key, key)
        for other in sources["key"]:
            if other == key or (groups[key] and groups[key] == groups[other]):
                continue
            assert cell(pairwise, key, other) <= own + 1e-9, f"{other} beat {key} at its own table"


def test_retained_variance_is_not_symmetric_but_correlation_is(pairwise, sources):
    keys = list(sources["key"])
    asymmetric = any(
        abs(cell(pairwise, a, b) - cell(pairwise, b, a)) > 1.0
        for a in keys for b in keys if a != b
    )
    assert asymmetric, "retained variance is scaled by the target's variance; it should differ by direction"
    for a in keys:
        for b in keys:
            assert cell(pairwise, a, b, "corr") == pytest.approx(cell(pairwise, b, a, "corr"), abs=1e-9)


def test_scaling_a_spectrum_does_not_change_any_comparison():
    """Every matrix is rescaled to a common total rate, so an organism's overall speed cannot count."""
    q = sc.acc.load_spectrum(SPECTRA_DIR / "jc69.csv")
    one = sc.codon_table(q)
    ten = sc.codon_table(q * 10.0)
    assert np.allclose(one, ten, rtol=1e-12, atol=0, equal_nan=True)


# ---------------------------------------------------------------------------
# The exported tables
# ---------------------------------------------------------------------------

def test_every_declared_spectrum_appears_in_every_table(sources, pairwise):
    declared = set(pd.read_csv(SPECTRA_DIR / "spectra_sources.csv")["key"])
    assert set(sources["key"]) == declared
    assert set(pairwise["target"]) == declared and set(pairwise["predictor"]) == declared
    assert len(pairwise) == len(declared) ** 2


def test_the_focus_spectrum_sorts_first_and_is_marked_once(sources):
    assert sources["is_focus"].sum() == 1
    assert bool(sources.sort_values("plot_order")["is_focus"].iloc[0])


def test_a_repeated_organism_is_dropped_from_the_stand_in_table(stand_in, sources):
    groups = sources.dropna(subset=["organism_group"])
    assert not groups.empty, "the fixture needs at least one repeated organism to be meaningful"
    for _, rows in groups.groupby("organism_group"):
        assert rows["key"].isin(stand_in["key"]).sum() == 1, "an organism is counted twice"
    assert set(stand_in["key"]) <= set(sources["key"])


def test_the_stand_in_numbers_follow_from_the_pairwise_table(stand_in, pairwise):
    kept = set(stand_in["key"])
    for _, row in stand_in.iterrows():
        others = kept - {row["key"]}
        mine = pairwise[(pairwise["target"] == row["key"]) & pairwise["predictor"].isin(others)]
        assert row["mean_corr_to_others"] == pytest.approx(mine["corr"].mean(), abs=1e-9)
        assert row["min_corr_to_others"] == pytest.approx(mine["corr"].min(), abs=1e-9)
        back = pairwise[(pairwise["predictor"] == row["key"]) & pairwise["target"].isin(others)]
        assert row["mean_retained_predicting_others"] == pytest.approx(
            back["retained_compressed"].mean(), abs=1e-9)
        assert int(row["n_others"]) == len(others)


def test_the_tables_hold_no_missing_numbers(pairwise, stand_in):
    for frame, name in ((pairwise, "pairwise"), (stand_in, "stand_in")):
        numeric = frame.select_dtypes("number")
        assert np.isfinite(numeric.to_numpy()).all(), f"{name} has a non-finite value"


def test_the_manifest_lists_every_shipped_table():
    manifest = pd.read_csv(RESULTS / "MANIFEST.tsv", sep="\t")
    on_disk = {p.name for p in RESULTS.glob("*.tsv.gz")}
    assert on_disk <= set(manifest["file"]), f"unlisted: {sorted(on_disk - set(manifest['file']))}"
    for _, row in manifest[manifest["file"].isin(on_disk)].iterrows():
        frame = pd.read_csv(RESULTS / row["file"], sep="\t")
        assert len(frame) == row["rows"] and frame.shape[1] == row["columns"]
        assert str(row["used_for"]).strip(), f"{row['file']} has no 'used_for' note"


# ---------------------------------------------------------------------------
# Input rejection
# ---------------------------------------------------------------------------

def test_a_sources_table_with_no_focus_is_refused(tmp_path):
    frame = pd.read_csv(SPECTRA_DIR / "spectra_sources.csv")
    frame["manuscript_role"] = ""
    for name in frame["key"]:
        (tmp_path / f"{name}.csv").write_text((SPECTRA_DIR / f"{name}.csv").read_text())
    frame.to_csv(tmp_path / "spectra_sources.csv", index=False)
    with pytest.raises(ValueError, match="focus"):
        ex.build(tmp_path, None)


def test_a_duplicate_key_is_refused(tmp_path):
    frame = pd.read_csv(SPECTRA_DIR / "spectra_sources.csv")
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    doubled.to_csv(tmp_path / "spectra_sources.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        sc.load_sources(tmp_path / "spectra_sources.csv")


def test_an_all_zero_spectrum_is_refused():
    with pytest.raises(ValueError, match="no non-zero rate"):
        sc.codon_table(np.zeros((4, 4)))


def test_a_figure_refuses_a_pairwise_table_that_lost_a_spectrum(tmp_path, pairwise, sources, stand_in):
    short = pairwise[pairwise["target"] != sources["key"].iloc[-1]]
    short.to_csv(tmp_path / "spectra_pairwise.tsv.gz", sep="\t", index=False, compression="gzip")
    sources.to_csv(tmp_path / "spectra_sources.tsv.gz", sep="\t", index=False, compression="gzip")
    stand_in.to_csv(tmp_path / "spectra_generic_stand_in.tsv.gz", sep="\t", index=False, compression="gzip")
    with pytest.raises(ValueError, match="disagree|expected"):
        make_sE_spectra_codon_space.load(tmp_path)


def test_a_figure_refuses_a_ranking_with_uneven_comparisons(tmp_path, pairwise, sources, stand_in):
    """The two rankings are means, so they only compare if every spectrum faced the same others."""
    uneven = stand_in.copy()
    uneven.loc[uneven.index[0], "n_others"] = int(uneven["n_others"].iloc[0]) + 1
    pairwise.to_csv(tmp_path / "spectra_pairwise.tsv.gz", sep="\t", index=False, compression="gzip")
    sources.to_csv(tmp_path / "spectra_sources.tsv.gz", sep="\t", index=False, compression="gzip")
    uneven.to_csv(tmp_path / "spectra_generic_stand_in.tsv.gz", sep="\t", index=False, compression="gzip")
    with pytest.raises(ValueError, match="different\\s+numbers of others"):
        make_sE_spectra_codon_space.load(tmp_path)


def test_the_ranking_panels_name_the_merged_spectrum_from_the_tables(sources, stand_in):
    """The 'counted once' wording is derived, not typed: no virus name lives in the script."""
    merged = make_sE_spectra_codon_space.merged_labels(sources, stand_in)
    assert merged == ["Influenza H1N1"]
    assert "Influenza H1N1" in make_sE_spectra_codon_space.merged_tagline(merged)
    assert make_sE_spectra_codon_space.merged_tagline([]) == "every spectrum counted once"


# ---------------------------------------------------------------------------
# The figures and their captions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module,stem", [(make_sE_spectra_codon_space, make_sE_spectra_codon_space.STEM)])
def test_the_figure_writes_a_pdf_a_png_values_and_a_caption(tmp_path, module, stem):
    assert module.main(["--out", str(tmp_path)]) == 0
    for suffix in (".pdf", ".png", "_values.tsv", "_caption.md"):
        path = tmp_path / f"{stem}{suffix}"
        assert path.is_file() and path.stat().st_size > 0, f"{path.name} missing or empty"


def test_s05_caption_quotes_the_published_numbers(tmp_path, pairwise, sources):
    make_sE_spectra_codon_space.main(["--out", str(tmp_path)])
    text = (tmp_path / f"{make_sE_spectra_codon_space.STEM}_caption.md").read_text()
    assert f"{PUBLISHED_SELF_COMPRESSION:.1f}%" in text
    assert f"{PUBLISHED_COMPARATOR:.1f}%" in text
    assert f"{PUBLISHED_COMPARATOR_UNCOMPRESSED:.1f}%" in text


def test_s05_caption_names_the_comparator_not_merely_the_simplest_matrix(tmp_path, sources, pairwise):
    """The published contrast is with a named matrix; the simplest one is a different spectrum.

    Quoting the published percentage beside the wrong spectrum's name would be a false
    provenance claim, which is the error this test exists to stop coming back.
    """
    make_sE_spectra_codon_space.main(["--out", str(tmp_path)])
    text = (tmp_path / f"{make_sE_spectra_codon_space.STEM}_caption.md").read_text()
    labels = dict(zip(sources["key"], sources["label"]))
    comparator = roled(sources, "comparator")
    simplest = sources.sort_values("free_nt_parameters")["key"].iloc[0]
    assert simplest != comparator, "the fixture no longer distinguishes the two cases"
    published = f"{PUBLISHED_COMPARATOR:.1f}%"
    before = text.split(published)[0]
    assert labels[comparator] in before[-200:], "the published number is not attributed to the comparator"


def test_s05_caption_uses_a_rival_that_supports_the_claim(tmp_path, pairwise, sources):
    """The 'as expressive, still wrong' sentence must cite a spectrum that is in fact wrong.

    Citing the best-scoring equally-expressive spectrum would argue the opposite of the
    sentence it sits in, because a close relative of the focus scores highly.
    """
    make_sE_spectra_codon_space.main(["--out", str(tmp_path)])
    # Only the heatmap's own paragraphs. The panel B/C paragraph names the spectrum that was
    # merged into its subtype, which is an equally expressive one being mentioned for an
    # unrelated reason; scanning the whole caption would read that as the claim below.
    text = (tmp_path / f"{make_sE_spectra_codon_space.STEM}_caption.md").read_text().split("(**B**")[0]
    focus = roled(sources, "focus")
    comparator = roled(sources, "comparator")
    n_free = int(sources.loc[sources["key"] == focus, "free_nt_parameters"].iloc[0])
    rivals = [k for k in sources["key"] if k != focus
              and int(sources.loc[sources["key"] == k, "free_nt_parameters"].iloc[0]) == n_free]
    cited = [k for k in rivals if sources.loc[sources["key"] == k, "label"].iloc[0] in text]
    assert cited, "the caption cites no equally expressive spectrum"
    for key in cited:
        assert cell(pairwise, focus, key) < cell(pairwise, focus, comparator), (
            f"the caption cites {key}, which does better than the comparator it is contrasted with"
        )


@pytest.mark.parametrize("module", [make_sE_spectra_codon_space], ids=["sF"])
def test_every_number_in_the_caption_is_in_the_values_table(tmp_path, module):
    """Caption numbers are recomputed from the data, never typed."""
    import re
    module.main(["--out", str(tmp_path)])
    caption = (tmp_path / f"{module.STEM}_caption.md").read_text()
    # Every values file the figure writes, not just the main one: panels B and C quote the
    # ranking table, which ships beside it.
    pool = set()
    for path in sorted(tmp_path.glob(f"{module.STEM}*_values.tsv")):
        values = pd.read_csv(path, sep="\t")
        for v in values.select_dtypes("number").to_numpy().ravel():
            if np.isfinite(v):
                pool |= {f"{v:.0f}", f"{v:.1f}", f"{v:.2f}", f"{abs(v):.0f}"}
    assert pool, "no values files were written"
    # Percentages and correlations the caption states, skipping the methods constants.
    quoted = re.findall(r"(-?\d+\.\d)%", caption) + re.findall(r"at (\d\.\d\d)", caption)
    assert quoted, "the caption states no numbers"
    missing = [q for q in quoted if q not in pool and q.lstrip("-") not in pool]
    assert not missing, f"caption numbers absent from the values table: {missing}"


def test_s05_picks_text_colour_by_measured_contrast():
    import matplotlib.pyplot as plt
    from mutacc import style
    cmap = plt.get_cmap(style.DIVERGING)
    for level in np.linspace(0.0, 1.0, 21):
        rgba = cmap(level)
        chosen = make_sE_spectra_codon_space.text_colour_on(rgba)
        assert chosen in {style.INK, style.SURFACE}
        import matplotlib.colors as mcolors
        ratio = make_sE_spectra_codon_space._contrast(rgba, mcolors.to_rgb(chosen))
        assert ratio >= 3.0, f"text on {rgba} would sit at {ratio:.2f}:1, under the 3:1 floor"


def test_the_exporter_runs_end_to_end_and_is_deterministic(tmp_path, pairwise):
    """Two runs of the engine agree exactly: nothing here depends on dict order or a clock."""
    sources = sc.load_sources(SPECTRA_DIR / "spectra_sources.csv")
    order = list(sources["key"])
    first = sc.pairwise(SPECTRA_DIR, sources, order)
    second = sc.pairwise(SPECTRA_DIR, sources, order)
    pd.testing.assert_frame_equal(first, second)
