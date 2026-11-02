"""Figure 2 panels A and B: one lineage, every point, the numbers in the subheaders.

Golden values are the shipped ones. The properties that matter are structural: the panels
plot every replacement of the single primary lineage (no pooling, no dropping zeros), the
statistics printed under the panel titles in the composite are the ones the code computes,
and the axis text is the same size as in the other generated panels.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import pearsonr, spearmanr

import make_fig02ab as ab
import make_fig02d as d
from mutacc import style
from mutacc.guide import REPO_ROOT, load_guide

G5 = load_guide(REPO_ROOT / "guides" / "h3n2_5lineage.yaml")
BASE = next(m.key for m in G5.models if m.role == "base")
FT = next(m.key for m in G5.models if m.role == "finetune")
LAYOUT = {p["key"]: p for p in yaml.safe_load((REPO_ROOT / "guides/composites/fig02.yaml").read_text())["panels"]}


@pytest.fixture(scope="module")
def one_lineage():
    frame = G5.read_table("replacement_scores")
    return frame[frame["lineage"].astype(str) == G5.primary_lineage]


def rows_for(frame, model):
    return frame[frame["model"] == model]


def synthetic(n=60):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"plm_prob": 10 ** rng.uniform(-8, 0, n), "mut_prob": 10 ** rng.uniform(-14, -4, n),
                         "nt_steps": rng.integers(1, 4, n)})


# --- the statistics ------------------------------------------------------------------------

def test_golden_correlations_for_the_primary_lineage(one_lineage):
    base = ab.correlations(rows_for(one_lineage, BASE))
    ft = ab.correlations(rows_for(one_lineage, FT))
    assert base["spearman_r"] == pytest.approx(0.2750, abs=5e-4)
    assert ft["spearman_r"] == pytest.approx(0.6549, abs=5e-4)
    assert base["pearson_r"] == pytest.approx(0.0538, abs=5e-4)
    assert ft["pearson_r"] == pytest.approx(0.0454, abs=5e-4)


def test_correlations_agree_with_independent_implementations():
    frame = synthetic()
    got = ab.correlations(frame)
    assert got["spearman_r"] == pytest.approx(spearmanr(frame["plm_prob"], frame["mut_prob"])[0])
    assert got["pearson_r"] == pytest.approx(pearsonr(frame["plm_prob"], frame["mut_prob"])[0])


def test_pearson_is_on_the_probabilities_not_their_logs():
    """The frozen panels' Pearson was on the raw values; a log would give a different number."""
    frame = synthetic()
    raw = ab.correlations(frame)["pearson_r"]
    logged = pearsonr(np.log10(frame["plm_prob"]), np.log10(frame["mut_prob"]))[0]
    assert abs(raw - logged) > 1e-3


def test_subheader_text_has_three_decimals_for_both():
    assert ab.subheader({"spearman_r": 0.27504, "pearson_r": 0.05379}) == "Spearman ρ = 0.275; Pearson r = 0.054"


# --- the composite prints exactly those numbers --------------------------------------------

@pytest.mark.parametrize("key, model", [("2A", BASE), ("2B", FT)])
def test_the_composite_subheader_is_the_computed_statistic(key, model, one_lineage):
    expected = ab.subheader(ab.correlations(rows_for(one_lineage, model)))
    assert LAYOUT[key]["tagline"] == expected, "re-run make_fig02ab.py and copy the subheader into fig02.yaml"


@pytest.mark.parametrize("key", ["2A", "2B"])
def test_the_composite_places_the_generated_panels(key):
    assert LAYOUT[key]["file"] == f"build/panels/fig02{key[1].lower()}_scatter_panel.pdf"
    assert "crop_top_frac" not in LAYOUT[key], "a generated panel has no title of its own to crop"


# --- every point, one lineage --------------------------------------------------------------

def test_the_panel_data_is_every_replacement_of_one_lineage(one_lineage):
    rows = rows_for(one_lineage, BASE)
    assert one_lineage["lineage"].nunique() == 1
    assert len(rows) == 10754
    summary = pd.DataFrame(ab.summary_rows(rows, panel="A", model=BASE, lineage=G5.primary_lineage))
    counts = summary[summary["measure"] == "n_replacements"].set_index("group")["value"]
    assert counts.sum() == len(rows)
    assert counts.to_dict() == {"1": 3334.0, "2": 5487.0, "3": 1933.0}


def test_the_figure_draws_one_scatter_point_per_replacement():
    import matplotlib.pyplot as plt

    frame = synthetic(90)
    fig = ab.draw(frame)
    try:
        (ax,) = fig.axes
        drawn = sum(len(c.get_offsets()) for c in ax.collections)
        assert drawn == len(frame)
        assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("column", ["plm_prob", "mut_prob"])
@pytest.mark.parametrize("bad", [0.0, np.nan, -1.0])
def test_a_value_that_cannot_go_on_a_log_axis_is_an_error_not_a_filter(column, bad):
    frame = synthetic()
    frame.loc[3, column] = bad
    with pytest.raises(ValueError, match="log axes"):
        ab.log_columns(frame)


def test_an_unknown_nucleotide_step_group_is_refused():
    frame = synthetic()
    frame.loc[0, "nt_steps"] = 4
    with pytest.raises(ValueError, match="nt_steps"):
        ab.log_columns(frame)


# --- appearance ----------------------------------------------------------------------------

def test_group_colours_are_the_original_dark2_three_and_distinct():
    assert ab.group_colours() == {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3"}


def test_one_ink_group_is_drawn_last_so_it_is_on_top():
    assert ab.DRAW_ORDER == (3, 2, 1)


def test_the_default_legend_is_inside_the_plot_titled_and_ordered_as_before():
    import matplotlib.pyplot as plt

    fig = ab.draw(synthetic())
    try:
        (ax,) = fig.axes
        legend = ax.get_legend()
        assert legend.get_title().get_text() == "Mut_num"
        assert [t.get_text() for t in legend.get_texts()] == ["1", "2", "3"]
        assert legend._loc == 2, "upper left, inside the axes"
        # It sits on the point cloud, so it is set well under the tick labels: it names three
        # colours already on show and must not compete with the data underneath it.
        size = legend.get_texts()[0].get_fontsize()
        assert size == ab.LEGEND_PT < style.PANEL_TICK_PT
        assert legend.get_title().get_fontsize() == ab.LEGEND_PT
        assert legend.legend_handles[0].get_markersize() == ab.LEGEND_MARKER_PT < 6.0
    finally:
        plt.close(fig)


def test_the_text_variant_has_no_legend_and_coloured_explanatory_text():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    fig = ab.draw(synthetic(), labels="text")
    try:
        (ax,) = fig.axes
        assert ax.get_legend() is None
        found = {t.get_text(): to_hex(t.get_color()) for t in ax.texts}
        colours = ab.group_colours()
        assert found == {words: colours[k] for k, (_, words) in ab.TEXT_LABELS.items()}
    finally:
        plt.close(fig)


def test_text_labels_sit_in_the_gaps_clear_of_every_cloud(one_lineage):
    """Each coloured phrase must be away from the points of every group, in the shipped data."""
    rows = rows_for(one_lineage, BASE)
    lx, ly, group = ab.log_columns(rows)
    half = 0.32          # decades: half the text height at this size
    for k, (log_y, _) in ab.TEXT_LABELS.items():
        near = np.abs(ly - log_y) < half
        in_text_span = lx < np.log10(ab.TEXT_X) + 2.6
        assert not (near & in_text_span).any(), f"label for group {k} overlaps points"


def test_unknown_label_style_is_refused():
    with pytest.raises(ValueError, match="labels"):
        ab.draw(synthetic(), labels="legned")


def test_points_are_bigger_but_the_cloud_keeps_about_the_same_density():
    """Size went up from 3.4 and opacity down, so area x opacity (ink per point) stays close."""
    old_ink = 3.4 * 0.32
    for name, cfg in ab.VARIANTS.items():
        assert cfg["size"] > 3.4, name
        assert 0.10 <= cfg["alpha"] <= 0.32, name
        assert cfg["size"] * cfg["alpha"] < 3.5 * old_ink, name
    assert ab.VARIANTS["text"]["size"] > ab.VARIANTS["legend"]["size"], "the text variant has the bigger points"


def test_the_legend_variant_draws_the_fainter_cloud():
    """Only the ``legend`` variant puts a key on top of the points, so only it has to be read
    through. The ``text`` variant writes in the gaps between the clouds and does not."""
    assert ab.VARIANTS["legend"]["alpha"] < ab.VARIANTS["text"]["alpha"]
    # ...but not so faint that a lone point in the sparse 3-change cloud disappears
    assert ab.VARIANTS["legend"]["alpha"] >= 0.10


def test_both_variants_use_the_same_x_range_and_cut_the_data_closely(one_lineage):
    assert ab.VARIANTS["legend"]["xlim"] == ab.VARIANTS["text"]["xlim"]
    lo, hi = ab.VARIANTS["legend"]["ylim"]
    for model in (BASE, FT):
        lx, ly, _ = ab.log_columns(rows_for(one_lineage, model))
        assert 10 ** ly.min() > lo and 10 ** ly.max() < hi, "every point is inside the axes"
        assert 10 ** lx.min() > ab.VARIANTS["legend"]["xlim"][0]
        assert hi / 10 ** ly.max() < 10 ** 0.8, "no more than ~0.8 decades of headroom"


def test_no_contour_artists_are_drawn():
    """The density contours were tried and removed: only scatter collections remain."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    fig = ab.draw(synthetic())
    try:
        (ax,) = fig.axes
        assert all(isinstance(c, PathCollection) for c in ax.collections)
    finally:
        plt.close(fig)


def test_x_ticks_are_every_two_decades_not_every_one():
    import matplotlib.pyplot as plt

    fig = ab.draw(synthetic())
    try:
        (ax,) = fig.axes
        fig.canvas.draw()
        ticks = [t for t in ax.get_xticks() if ab.VARIANTS['legend']['xlim'][0] <= t <= ab.VARIANTS['legend']['xlim'][1]]
        exponents = np.round(np.log10(ticks)).astype(int)
        assert set(np.diff(exponents)) == {2}
        assert len(ticks) <= 5
    finally:
        plt.close(fig)


def test_the_axes_sit_in_the_shared_panel_box():
    import matplotlib.pyplot as plt

    fig = ab.draw(synthetic())
    try:
        (ax,) = fig.axes
        assert tuple(ax.get_position().bounds) == pytest.approx(style.PANEL_AXES)
    finally:
        plt.close(fig)


def test_both_panels_use_the_same_fixed_axes():
    import matplotlib.pyplot as plt

    limits = []
    for _ in range(2):
        fig = ab.draw(synthetic())
        try:
            (ax,) = fig.axes
            limits.append((ax.get_xlim(), ax.get_ylim()))
        finally:
            plt.close(fig)
    assert limits[0] == limits[1]


# --- the same text in every generated panel ------------------------------------------------

def test_generated_panels_share_one_native_size_and_label_sizes():
    import matplotlib.pyplot as plt

    figs = [ab.draw(synthetic())]
    try:
        assert tuple(figs[0].get_size_inches()) == pytest.approx(style.PANEL_SIZE)
        (ax,) = figs[0].axes
        assert ax.xaxis.label.get_size() == style.PANEL_LABEL_PT
        assert ax.yaxis.label.get_size() == style.PANEL_LABEL_PT
        assert ax.get_xticklabels()[0].get_size() == style.PANEL_TICK_PT
    finally:
        for f in figs:
            plt.close(f)


def test_panel_d_uses_the_same_native_size_and_label_sizes():
    source = (REPO_ROOT / "figures" / "make_fig02d.py").read_text()
    assert "style.figure(*style.PANEL_SIZE)" in source
    assert source.count("style.place_panel_labels(ax)") == 1 and source.count("style.PANEL_TICK_PT") == 2
    assert "style.panel_axes(fig)" in source and "style.save_panel(" in source


def test_panel_d_separates_all_from_the_codon_groups_by_space_not_by_a_rule():
    """The gap stays; the rule that used to sit in it does not.

    A line drawn where no point lies reads as something plotted -- a threshold, a divider, a
    fitted value -- and panel D already separates "all" from groups 1/2/3 by ``GAP`` alone.
    The y = 0 reference is a different line and stays: group 3's base model straddles it.
    """
    source = (REPO_ROOT / "figures" / "make_fig02d.py").read_text()
    assert d.GAP > 0, "the space between 'all' and the codon-distance groups carries the split"
    # ``axvline`` spans the whole panel; the ``ax.vlines`` that remain are the error bars, which
    # run between one estimate's own interval ends and sit on their points.
    assert "axvline" not in source
    assert source.count("style.reference_line(ax, 0.0") == 1


def _composite_spans():
    import matplotlib  # noqa: F401  -- before fitz, or Pillow dies
    fitz = pytest.importorskip("fitz")
    spans = []
    with fitz.open(REPO_ROOT / "output" / "fig02_composite.pdf") as doc:
        for block in doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    spans.append((span["text"].strip(), span["size"], span["bbox"]))
    return spans


AXIS_TITLES = {"PLM probability", "Mutation probability", d.YLABEL, "Nucleotide changes needed", "Epoch"}


def test_the_composites_axis_titles_render_at_one_size_in_all_four_panels():
    spans = [s for s in _composite_spans() if s[0] in AXIS_TITLES]
    found = {s[0] for s in spans}
    assert found == AXIS_TITLES
    assert sum(s[0] == d.YLABEL for s in spans) == 2, "panels C and D carry the same y title"
    sizes = [s[1] for s in spans]
    assert max(sizes) / min(sizes) < 1.03, sorted(set(round(x, 2) for x in sizes))


def test_the_tick_labels_render_at_one_size_in_all_four_panels():
    spans = _composite_spans()
    for text in ("raw", "all", "0.65", "0.8", "10"):
        assert any(s[0] == text for s in spans), text
    sizes = {s[0]: s[1] for s in spans if s[0] in {"raw", "all", "0.65", "0.8"}}
    assert max(sizes.values()) / min(sizes.values()) < 1.03, sizes


def test_y_titles_line_up_down_each_column_and_x_titles_across_each_row():
    spans = _composite_spans()
    left = {s[0] + str(i): s[2] for i, s in enumerate(spans) if s[0] in {"Mutation probability", d.YLABEL}}
    ys = sorted(b[0] for b in left.values())           # x0 of the four rotated y titles
    # columns: A and C share a left edge, B and D share another
    assert abs(ys[0] - ys[1]) < 2.0 and abs(ys[2] - ys[3]) < 2.0, ys
    centre = lambda b: (b[0] + b[2]) / 2
    plm = sorted(centre(b) for t, _, b in spans if t == "PLM probability")
    epoch = [centre(b) for t, _, b in spans if t == "Epoch"]
    nuc = [centre(b) for t, _, b in spans if t == "Nucleotide changes needed"]
    assert len(plm) == 2 and len(epoch) == 1 and len(nuc) == 1
    assert abs(plm[0] - epoch[0]) < 2.0, "A and C x titles are centred on the same line"
    assert abs(plm[1] - nuc[0]) < 2.0, "B and D x titles are centred on the same line"


# --- caption -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def caption_text(one_lineage):
    rows = []
    for panel, model in (("A", BASE), ("B", FT)):
        rows += ab.summary_rows(rows_for(one_lineage, model), panel=panel, model=model, lineage=G5.primary_lineage)
    summary = pd.DataFrame(rows)
    root = ab.root_description(G5, G5.primary_lineage)
    return ab.caption(summary, root, G5), root


def test_the_caption_names_the_root_sequence_file_and_header(caption_text):
    text, root = caption_text
    flat = " ".join(text.split())
    assert root["path"] == "data/roots/h3n2_5lineage/J.2_int.nt.fa"
    assert root["path"] in flat
    assert root["header"].startswith("EPI2990337|HA|") and root["header"] in flat
    assert f"`{G5.primary_lineage}`" in flat


def test_the_caption_says_the_root_is_inferred_not_observed(caption_text):
    text, root = caption_text
    assert root["root_type"] == "reconstructed"
    flat = " ".join(text.split())
    assert "inferred ancestral sequence, not an observed one" in flat
    assert "should not be read as its source" in flat


def test_the_caption_reports_the_values_and_both_correlations(caption_text):
    flat = " ".join(caption_text[0].split())
    for number in ("10,754", "3,334", "5,487", "1,933", "0.275", "0.655", "0.054", "0.045"):
        assert number in flat, number
    assert "Mut_num" in flat and "not their logarithms" in flat


def test_the_caption_does_not_describe_contours_or_pooling(caption_text):
    flat = " ".join(caption_text[0].split()).lower()
    assert "contour" not in flat and "pooled" not in flat


def test_an_observed_root_is_described_as_observed():
    summary = pd.DataFrame([dict(panel=p, series="m", group=g, measure=m, value=1.0, n=1, lineage="L")
                            for p in "AB" for g, m in [("all", "spearman_r"), ("all", "pearson_r"),
                                                       ("1", "n_replacements"), ("2", "n_replacements"),
                                                       ("3", "n_replacements")]])

    class Guide:
        virus = "a virus"
        models = G5.models

    root = dict(lineage="L", path="root.fa", root_type="observed", header="tip|HA|X")
    text = " ".join(ab.caption(summary, root, Guide).split())
    assert "an observed sequence" in text and "inferred" not in text


def test_root_description_reads_the_header_from_the_file_the_guide_names():
    root = ab.root_description(G5, G5.primary_lineage)
    first = (REPO_ROOT / root["path"]).read_text().splitlines()[0]
    assert first == ">" + root["header"]


# --- the shipped outputs -------------------------------------------------------------------

def test_shipped_values_table_matches_what_the_code_computes(one_lineage):
    shipped = pd.read_csv(REPO_ROOT / "output" / "fig02ab_scatter_panels_values.tsv", sep="\t")
    for panel, model in (("A", BASE), ("B", FT)):
        fresh = pd.DataFrame(ab.summary_rows(rows_for(one_lineage, model), panel=panel, model=model,
                                             lineage=G5.primary_lineage))
        old = shipped[shipped["panel"] == panel].reset_index(drop=True)
        assert old[["series", "group", "measure"]].equals(fresh[["series", "group", "measure"]])
        np.testing.assert_allclose(old["value"], fresh["value"], rtol=2e-5)


def test_a_built_panel_is_one_vector_page_with_live_text(tmp_path, one_lineage):
    """The panels are intermediates and are no longer shipped (``build_figures.panels_dir``), so
    this builds them. What it checks is unchanged: one page each, and the axis title as real text
    rather than outlines, because ``panelise`` crops and scales these pages into the composite."""
    import matplotlib  # noqa: F401
    fitz = pytest.importorskip("fitz")

    for name, model in (("fig02a_scatter_panel", BASE), ("fig02b_scatter_panel", FT)):
        style.save_panel(ab.draw(rows_for(one_lineage, model)), tmp_path / name,
                         suffixes=(".pdf",))
        with fitz.open(tmp_path / f"{name}.pdf") as doc:
            assert len(doc) == 1
            assert "PLM probability" in doc[0].get_text()


def test_the_panel_renders_are_not_shipped_in_output():
    """``output/`` is the published figures. A panel render there is a leftover: the composite
    already contains it, and a reader cannot tell it from a finished figure."""
    strays = sorted(p.name for p in (REPO_ROOT / "output").glob("fig02[ab]_scatter_panel.*"))
    assert strays == []
