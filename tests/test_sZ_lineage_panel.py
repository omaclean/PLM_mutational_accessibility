"""Table SZ: the lineage panel -- size, sampling period and variation per lineage.

(It carries the stem ``sZ_lineage_panel`` from when the draft numbered it Figure S15. It
became Table SZ on 19 Oct 2026 and the figure slot it left went to ``sO_masked_marginals``; the stem
is unchanged because stems are not numbers -- see guides/manuscript/figure_numbers.yaml.)

The table is a description of the input data rather than a result, so the tests here are
mostly arithmetic and internal consistency: the counts have to add up, the dates have to
be ordered, and the table has to agree with the guide and with the tables the figures
actually read. The golden numbers are pinned because a silent change in the lineage set
or the date join would otherwise look like nothing at all.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

import make_sZ_lineage_panel as sZ
from mutacc.guide import REPO_ROOT, load_guide

GUIDE = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")
RESULTS = REPO_ROOT / "results" / "h3n2_13lineage"

#: The sequence count per lineage. These are the weights every lineage-mean in the paper
#: carries.
#:
#: Repinned 22 Oct 2026 for the reassignment (docs/notes/LINEAGE_REASSIGNMENT_PLAN.md): the thirteen
#: lineages are now delimited by the *published* rule generalised to the branching tree,
#: not by the nearest-root construction these were first read from on 15 Oct. D and E
#: additionally carry outgroup roots. The previous values were
#: C.1 6082, D 1217, E 1543, E.2 1613, G.1 779, G.1.1 6775, G.1.3 3879, G.2 5460,
#: G.2.2 2525, J 2489, J.2 25841, J.2.4 964, K 19492 (total 78,659).
EXPECTED_SIZES = {
    "C.1": 5821, "D": 1139, "E": 1511, "E.2": 1588, "G.1": 1138, "G.1.1": 12597,
    "G.1.3": 4182, "G.2": 6969, "G.2.2": 2460, "J": 4059, "J.2": 26703, "J.2.4": 776,
    "K": 17718,
}
TOTAL = 86_661


@pytest.fixture(scope="module")
def table():
    return sZ.load(RESULTS)


def date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


# ---------------------------------------------------------------------------
# The shipped table
# ---------------------------------------------------------------------------

def test_the_table_covers_exactly_the_guides_lineages(table):
    sZ.check_against_guide(table, GUIDE)          # raises if they disagree
    assert len(table) == len(GUIDE.lineages)


def test_sequence_counts_are_the_pipelines_own(table):
    assert dict(zip(table["lineage"], table["n_sequences"])) == EXPECTED_SIZES
    assert table["n_sequences"].sum() == TOTAL


def test_a_changed_lineage_set_is_caught(table):
    """The guide check is the thing standing between a dropped lineage and a silent figure."""
    short = table.iloc[1:].copy()
    with pytest.raises(ValueError, match="only in the guide"):
        sZ.check_against_guide(short, GUIDE)


def test_the_date_categories_partition_every_sequence(table):
    counted = (table["n_dated"] + table["n_partial_date"]
               + table["n_implausible_date"] + table["n_undated"])
    assert (counted == table["n_sequences"]).all()
    assert (table["n_dated"] > 0).all()
    assert (table[["n_partial_date", "n_implausible_date", "n_undated"]] >= 0).all().all()


def test_most_sequences_carry_a_usable_date(table):
    """A lineage whose dates mostly failed to join would make its span meaningless."""
    assert (table["pct_dated"] >= 75).all()
    assert table["n_dated"].sum() / table["n_sequences"].sum() > 0.90


# ---------------------------------------------------------------------------
# The dates themselves
# ---------------------------------------------------------------------------

def test_every_span_is_ordered(table):
    for _, row in table.iterrows():
        first, last = date(row["first_date"]), date(row["last_date"])
        low, high = date(row["span_start_p1"]), date(row["span_end_p99"])
        assert first <= low <= date(row["median_date"]) <= high <= last, row["lineage"]
        assert row["span_days"] == (high - low).days


def test_no_lineage_starts_before_the_pre_training_corpus_ends():
    """The 2016 floor, which is the thing actually keeping bad dates out.

    The corpus the model was pre-trained on ends in 2015, so no sequence in this panel
    can legitimately pre-date 2016. G.1.1 and J.2 used to report first dates of 2011 and
    2012 -- source-metadata errors that cleared the old 2010 floor. The earliest real
    lineage, C.1, starts 2016-05-21, so the floor cuts nothing genuine.
    """
    frame = sZ.load(RESULTS)
    for _, row in frame.iterrows():
        assert date(row["first_date"]) >= datetime.date(2016, 1, 1), (
            f"{row['lineage']} starts {row['first_date']}, before the 2015 corpus cutoff"
        )


def test_first_date_is_now_close_enough_to_the_percentile_span_to_quote():
    """With the floor in place no lineage's absolute first date is a decade adrift.

    This is the check that would have caught the original bug. It stays after the fix:
    if a future database rebuild reintroduces a date years before its own lineage, the
    gap between first_date and the 1st percentile is where it shows up.
    """
    frame = sZ.load(RESULTS)
    for _, row in frame.iterrows():
        gap = (date(row["span_start_p1"]) - date(row["first_date"])).days
        assert gap < 365 * 3, (
            f"{row['lineage']}: first_date {row['first_date']} is {gap} days before its "
            "own 1st percentile -- suspect a bad collection date in the source metadata"
        )


def test_the_lineages_are_ordered_by_when_they_were_sampled(table):
    starts = [date(v) for v in table["span_start_p1"]]
    assert starts == sorted(starts)


def test_no_lineage_was_sampled_before_the_fine_tuning_corpus_closed(table):
    """The set exists to avoid the corpus the model was tuned on; the 1st percentile is
    the right place to check that, for the reason the test above gives."""
    cutoff = GUIDE.corpus_cutoff
    cutoff = cutoff if isinstance(cutoff, datetime.date) else date(str(cutoff))
    assert all(date(v) >= cutoff for v in table["span_start_p1"])


# ---------------------------------------------------------------------------
# Diversity columns, against the tables the figures read
# ---------------------------------------------------------------------------

def test_replacement_counts_match_the_observed_table(table):
    observed = GUIDE.read_table("replacement_observed")
    counts = observed[observed["obs_present"] == 1].groupby("lineage").size()
    assert dict(zip(table["lineage"], table["replacements_observed"])) == counts.to_dict()


def test_there_are_at_least_as_many_replacements_as_sites(table):
    """A site can carry more than one alternative amino acid, never fewer than one."""
    assert (table["replacements_observed"] >= table["sites_differing"]).all()


def test_the_largest_lineage_is_thirty_times_the_smallest(table):
    """The imbalance a reviewer asked about; the figure exists partly to show it."""
    ratio = table["n_sequences"].max() / table["n_sequences"].min()
    assert 30 < ratio < 36


# ---------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------

def test_the_figure_draws(table):
    import matplotlib
    matplotlib.use("Agg")
    figure = sZ.draw(table, int(table["n_sequences"].sum()))
    assert figure.get_size_inches()[0] == pytest.approx(7.09, abs=0.01)
    matplotlib.pyplot.close(figure)


def test_every_column_has_a_cell_for_every_lineage(table):
    for key, _header, formatter, _align in sZ.COLUMNS:
        rendered = [formatter(value) for value in table[key]]
        assert len(rendered) == len(table)
        assert all(text.strip() for text in rendered), key


def test_the_caption_states_the_total_and_the_percentile_warning(table):
    text = caption_text(table)
    assert f"{TOTAL:,}" in text
    assert "percentile span is the one to read" in text
    assert "1905" in text                      # the spreadsheet-serial artefact is named
    assert "Top 5% freq threshold" in text


def test_top5_threshold_is_within_expected_frequency_bounds(table):
    """The 95th percentile frequency threshold should range between ~0.25% and ~7%.

    The ceiling was 5% until the 22 Oct 2026 reassignment, when G.1 moved from 3.18% to
    6.62%. That is not a regression. Under the nearest-root construction G.1's top
    replacement sat at **84.7%** frequency -- the signature of a bucket holding two
    sub-populations, where the marker separating them is carried by most but not all
    members. The reassigned G.1 is 100% pure on the independent clade call, its top
    replacement is 57.4%, and the tail below it is even (30%, 27%, 19%, 18%, 17%, 14%,
    12%) rather than falling away. A flatter tail puts the eleventh-ranked frequency --
    which is what this threshold is -- higher. So the bound moved because the lineage
    got cleaner, not because the cut got worse.
    """
    assert "top5_freq_threshold" in table.columns
    assert (table["top5_freq_threshold"] > 0.002).all()
    assert (table["top5_freq_threshold"] < 0.07).all()
    # Check specific pinned values: G.1 is highest (~6.62%), K is lowest (~0.29%)
    g1 = table.loc[table["lineage"] == "G.1", "top5_freq_threshold"].iloc[0]
    k = table.loc[table["lineage"] == "K", "top5_freq_threshold"].iloc[0]
    assert g1 == pytest.approx(0.06620, rel=1e-3)
    assert k == pytest.approx(0.00286, rel=1e-3)


# ---------------------------------------------------------------------------
# The origin columns
# ---------------------------------------------------------------------------
#: Homoplasy count per lineage, from the 24 Sep 2026 segment 4 UShER tree. Pinned because
#: a rebuilt tree, a changed lineage membership file or a shifted numbering would all move
#: these quietly, and the figure would still draw.
#: Repinned 22 Oct 2026: the tree is unchanged (the same 24 Sep segment 4 UShER tree,
#: 780,727 nodes / 572,108 samples), but the lineage membership it is counted over is the
#: reassigned one. Previous values were C.1 279, D 81, E 105, E.2 91, G.1 47, G.1.1 254,
#: G.1.3 155, G.2 272, G.2.2 145, J 156, J.2 747, J.2.4 60, K 646.
EXPECTED_HOMOPLASIES = {
    "C.1": 251, "D": 65, "E": 87, "E.2": 87, "G.1": 47, "G.1.1": 362, "G.1.3": 164,
    "G.2": 301, "G.2.2": 131, "J": 191, "J.2": 684, "J.2.4": 35, "K": 561,
}

#: Root-protein positions the UShER reference record has a codon for. The reference is an
#: H2, so positions 1-26 and 562-566 have none and no replacement there can be placed.
COVERED_POSITIONS = 535


def test_homoplasy_counts_are_the_trees_own(table):
    assert dict(zip(table["lineage"], table["n_homoplasy"])) == EXPECTED_HOMOPLASIES


def test_the_origin_counts_nest_inside_the_observed_counts(table):
    """Every narrowing has to be a narrowing: a homoplasy is a placed replacement, which is
    an observed replacement at a position the tree covers."""
    assert (table["n_homoplasy"] <= table["n_placed"]).all()
    assert (table["n_placed"] <= table["n_observed_covered"]).all()
    assert (table["n_observed_covered"] <= table["replacements_observed"]).all()
    assert (table["n_homoplasy_multi_route"] <= table["n_homoplasy"]).all()
    assert (table["n_placed"] + table["n_unplaced"] == table["n_observed_covered"]).all()


def test_the_percentage_is_the_count_over_what_the_tree_places(table):
    """The denominator is n_placed, not the full observed count. Dividing by the latter
    would read every coverage gap as a replacement that arose exactly once."""
    expected = 100.0 * table["n_homoplasy"] / table["n_placed"]
    assert table["pct_multi_origin"].values == pytest.approx(expected.values)
    assert (table["pct_multi_origin"] > 20).all()
    assert (table["pct_multi_origin"] < 70).all()


def test_the_tree_covers_the_same_positions_for_every_lineage(table):
    assert set(table["n_covered_positions"]) == {COVERED_POSITIONS}


def test_a_homoplasy_has_at_least_two_origins(table):
    """max_origins is the per-lineage maximum, so it must clear 2 wherever one was found."""
    assert (table.loc[table["n_homoplasy"] > 0, "max_origins"] >= 2).all()


def test_a_rebuilt_origin_table_that_disagrees_with_the_panel_is_caught(table):
    """The defect this guards against is the one that put a phantom mutation in Figure
    1B/C: a stored table re-read against newer roots, with nothing comparing the two."""
    stale = table.copy()
    stale.loc[0, "replacements_observed"] = int(stale.loc[0, "replacements_observed"]) + 1
    with pytest.raises(ValueError, match="disagree about how many replacements"):
        sZ.load_homoplasy(RESULTS, stale)


def test_the_dated_column_is_gone_but_the_date_quality_is_still_stated(table):
    """The column was dropped on 21 Oct 2026; the sentence it anchored had to stay, because
    First, Last and Middle 98% still depend on which dates parsed."""
    assert "pct_dated" not in [key for key, _h, _f, _a in sZ.COLUMNS]
    assert "pct_dated" in table.columns          # still computed, still tested above
    text = caption_text(table)
    assert "retained in every replacement, origin and frequency count" in text


def test_the_caption_defines_an_origin_and_names_both_processes(table):
    text = caption_text(table)
    assert "An *origin* is a branch" in text
    assert "two or more different codon changes" in text
    assert "rises with sampling depth" in text   # the size confound must not be buried
    assert "floor on recurrence" in text


# ---------------------------------------------------------------------------
# the combined row at the foot of the table
# ---------------------------------------------------------------------------

@pytest.fixture
def summary(table):
    return sZ.summary_row(table, RESULTS)


def caption_text(frame) -> str:
    """The caption as ``main`` writes it, summary row and pooled share included."""
    pooled = 100.0 * frame["n_homoplasy"].sum() / frame["n_placed"].sum()
    return sZ.caption(frame, GUIDE, sZ.summary_row(frame, RESULTS), pooled)


def test_the_combined_row_sums_the_disjoint_sequence_sets(table, summary):
    # The thirteen lineages are disjoint, so sequences are the one column a plain sum is
    # right for, and it has to come to the total the caption quotes.
    assert summary["n_sequences"] == table["n_sequences"].sum()
    assert summary["n_sequences"] == TOTAL


def test_the_combined_row_spans_the_whole_sampling_period(table, summary):
    assert summary["first_month"] == table["first_month"].min()
    assert summary["last_month"] == table["last_month"].max()
    assert summary["first_month"] <= summary["last_month"]


def test_sites_and_replacements_are_counted_once_not_summed(table, summary):
    # The same replacement is reached in several lineages, so a sum double-counts. The
    # union must be strictly smaller than the sum, and no larger than any single total.
    assert summary["sites_differing"] < table["sites_differing"].sum()
    assert summary["replacements_observed"] < table["replacements_observed"].sum()
    assert summary["sites_differing"] >= table["sites_differing"].max()
    assert summary["replacements_observed"] >= table["replacements_observed"].max()


def test_the_union_counts_are_the_distinct_replacements_in_the_observed_table(summary):
    seen = pd.read_csv(RESULTS / "replacement_observed.tsv.gz", sep="\t")
    seen = seen[seen["obs_present"] == 1]
    assert summary["sites_differing"] == seen["position"].nunique()
    assert summary["replacements_observed"] == len(set(zip(seen["position"], seen["aa"])))


def test_a_replacement_cannot_outnumber_the_sites_it_sits_on(summary):
    assert summary["replacements_observed"] >= summary["sites_differing"]


def test_the_mean_multiple_origin_share_is_below_the_pooled_one(table, summary):
    # This is why the asterisk exists. Pooling weights a lineage by its size and the deep
    # lineages are the recurrent ones, so the pooled share is the larger of the two. If
    # these ever coincide the asterisk is misleading and the caption is wrong.
    homoplasy = pd.read_csv(RESULTS / "homoplasy_by_lineage.tsv.gz", sep="\t")
    pooled = 100.0 * homoplasy["n_homoplasy"].sum() / homoplasy["n_placed"].sum()
    assert summary["pct_multi_origin"] == pytest.approx(table["pct_multi_origin"].mean())
    assert summary["pct_multi_origin"] < pooled


def test_the_combined_homoplasy_count_is_the_sum_of_the_rows(table, summary):
    # Its unit is the lineage-replacement pair, not the replacement: one replacement
    # recurring in three lineages is three recurrences and is counted three times.
    assert summary["n_homoplasy"] == table["n_homoplasy"].sum()
    assert summary["n_homoplasy"] > summary["replacements_observed"]


def test_every_starred_column_is_a_column_the_table_draws():
    assert sZ.ACROSS_ROWS <= {key for key, _, _, _ in sZ.COLUMNS}


def test_the_summary_row_fills_every_drawn_column(summary):
    for key, _header, fmt, _align in sZ.COLUMNS:
        assert key in summary.index
        assert fmt(summary[key])          # every cell formats and is non-empty


def test_the_summary_row_is_not_mistaken_for_a_lineage(table, summary):
    assert summary["lineage"] not in set(table["lineage"])


def test_a_panel_the_observed_table_disagrees_with_is_caught(table):
    wrong = table.copy()
    wrong.loc[0, "replacements_observed"] += 1
    with pytest.raises(ValueError, match="present replacements"):
        sZ.summary_row(wrong, RESULTS)


def test_the_rarefied_share_leaves_the_smallest_lineage_unchanged():
    """The lineage that sets the depth is sampled whole, so rarefying must not move it.

    This is the arithmetic check that the rarefied column is taken over the same
    replacements as the raw one. It caught a real defect: the rarefaction was run over
    every replacement the tree places, while the raw percentage is taken over the ones the
    observed table marks present at a covered position, so the smallest lineage came back
    at 22.60% against its raw 23.38%.
    """
    homoplasy = pd.read_csv(RESULTS / "homoplasy_by_lineage.tsv.gz", sep="\t")
    if "rarefied_pct_multi_origin" not in homoplasy.columns:
        pytest.skip("the shipped origin table predates the rarefied columns")
    smallest = homoplasy.loc[homoplasy["n_tree_tips"].idxmin()]
    assert smallest["rarefied_to"] == smallest["n_tree_tips"]
    assert smallest["rarefied_pct_multi_origin"] == pytest.approx(
        smallest["pct_multi_origin"], abs=1e-6)
    assert smallest["rarefied_n_placed"] == pytest.approx(smallest["n_placed"], abs=1e-6)


def test_rarefying_lowers_every_deeper_lineage():
    """Every lineage but the smallest is subsampled, so its share can only fall."""
    homoplasy = pd.read_csv(RESULTS / "homoplasy_by_lineage.tsv.gz", sep="\t")
    if "rarefied_pct_multi_origin" not in homoplasy.columns:
        pytest.skip("the shipped origin table predates the rarefied columns")
    deeper = homoplasy[homoplasy["n_tree_tips"] > homoplasy["n_tree_tips"].min()]
    assert (deeper["rarefied_pct_multi_origin"] < deeper["pct_multi_origin"]).all()


def test_rarefying_breaks_the_correlation_with_sampling_depth():
    """The raw column tracks depth at Spearman 0.96; that is why it cannot be read across
    lineages, and removing it is the whole purpose of the rarefied one."""
    from scipy import stats
    homoplasy = pd.read_csv(RESULTS / "homoplasy_by_lineage.tsv.gz", sep="\t")
    if "rarefied_pct_multi_origin" not in homoplasy.columns:
        pytest.skip("the shipped origin table predates the rarefied columns")
    raw, _ = stats.spearmanr(homoplasy["n_tree_tips"], homoplasy["pct_multi_origin"])
    rarefied, p = stats.spearmanr(homoplasy["n_tree_tips"],
                                  homoplasy["rarefied_pct_multi_origin"])
    assert raw > 0.9
    assert abs(rarefied) < 0.5 and p > 0.05


