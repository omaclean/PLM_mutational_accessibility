"""S3: recovery of a masked amino acid across a panel of proteins. Synthetic inputs only."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import make_sC_viral_panel as sC


def summary(n_per_group=(2, 2, 3), models=(("model one", "base"), ("model two", "base_second")), seed=0,
            vary_composition=True):
    rng = np.random.default_rng(seed)
    rows = []
    for g, n in enumerate(n_per_group):
        for i in range(n):
            label = f"protein {g}-{i}"
            base = rng.uniform(0.05, 0.9)
            # a different floor per protein, because that is what the real panel has: a test with one
            # shared value cannot tell a per-row tick from a single averaged line
            comp = 0.055 + 0.004 * (g + i) if vary_composition else 0.06
            for m, (name, role) in enumerate(models):
                rows.append({"model": name, "role": role, "label": label, "display_label": label.title(),
                             "group": f"group {g}", "group_order": g, "length": 200 + 10 * i,
                             "mean_p_wt": float(np.clip(base + 0.1 * m * rng.normal(), 0.01, 0.99)),
                             "top1": 0.3, "n_sites": 200 + 10 * i, "composition_baseline": comp})
    return pd.DataFrame(rows)


@pytest.fixture
def results(tmp_path):
    d = tmp_path / "res"
    d.mkdir()
    summary().to_csv(d / "viral_panel_summary.tsv.gz", sep="\t", index=False)
    return d


def test_figure_and_companions_are_written(results, tmp_path):
    out = tmp_path / "out"
    assert sC.main(["--results", str(results), "--out", str(out)]) == 0
    for suffix in (".pdf", ".png", "_values.tsv", "_caption.md"):
        assert (out / f"{sC.STEM}{suffix}").is_file()
    vals = pd.read_csv(out / f"{sC.STEM}_values.tsv", sep="\t")
    assert len(vals) == 14 and vals["mean_p_wt"].between(0, 1).all()


def test_order_is_by_group_then_mean_of_models_not_by_either_model(results):
    info = sC.layout(sC.load(results))
    assert list(info["group_order"]) == sorted(info["group_order"])
    for _, block in info.groupby("group_order"):
        assert list(block["mean_of_models"]) == sorted(block["mean_of_models"], reverse=True)
    assert info["y"].is_monotonic_increasing and info["y"].is_unique


def test_groups_are_separated_by_a_heading_row(results):
    info = sC.layout(sC.load(results))
    steps = info["y"].diff().dropna()
    assert steps.max() > 2 * sC.ROW        # a heading and a gap sit between groups
    assert steps.min() == sC.ROW


@pytest.mark.parametrize("damage,message", [
    (lambda d: d.drop(columns=["composition_baseline"]), "missing columns"),
    (lambda d: d.assign(mean_p_wt=d["mean_p_wt"] + 1), "outside"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "more than one row"),
    (lambda d: d.iloc[1:], "not every protein"),
    (lambda d: d.assign(role="mystery"), "no marker"),
])
def test_bad_inputs_are_refused(tmp_path, damage, message):
    d = tmp_path / "res"
    d.mkdir()
    damage(summary()).to_csv(d / "viral_panel_summary.tsv.gz", sep="\t", index=False)
    with pytest.raises(ValueError, match=message):
        sC.load(d)


def reference_lines(ax):
    """The axvlines, which span the axes in y and so are the only lines with ydata (0, 1)."""
    return [line.get_xdata()[0] for line in ax.lines if tuple(line.get_ydata()) == (0.0, 1.0)]


def composition_ticks(ax):
    for coll in ax.collections:
        if coll.get_label() == sC.COMPOSITION_LABEL:
            return coll.get_offsets()
    raise AssertionError("no composition-baseline ticks were drawn")


def test_the_only_reference_line_is_the_universal_one_in_twenty(results):
    """Nothing on this panel may be drawn as a line unless it is the same for every protein."""
    df = sC.load(results)
    fig, _ = sC.draw(df)
    try:
        assert reference_lines(fig.axes[0]) == [sC.BLIND_GUESS] == [0.05]
    finally:
        plt.close(fig)


def test_the_line_does_not_move_when_the_compositions_do(tmp_path):
    """The old panel drew the mean composition baseline, so it slid about with the panel's contents."""
    seen = set()
    for shift in (0.0, 0.03):
        d = summary()
        d["composition_baseline"] += shift
        fig, _ = sC.draw(d)
        try:
            seen.update(reference_lines(fig.axes[0]))
        finally:
            plt.close(fig)
    assert seen == {sC.BLIND_GUESS}


def test_each_protein_gets_a_tick_at_its_own_composition_baseline(results):
    df = sC.load(results)
    info = sC.layout(df)
    fig, _ = sC.draw(df)
    try:
        ticks = composition_ticks(fig.axes[0])
    finally:
        plt.close(fig)
    assert len(ticks) == len(info)
    want = dict(zip(info["y"], info["composition_baseline"]))
    assert {y: x for x, y in ticks} == pytest.approx(want)
    assert len(set(info["composition_baseline"])) > 1       # the fixture really does vary them


def test_the_floors_are_both_named_for_the_reader(results, tmp_path):
    out = tmp_path / "out"
    assert sC.main(["--results", str(results), "--out", str(out)]) == 0
    text = " ".join((out / f"{sC.STEM}_caption.md").read_text().split())   # the writer rewraps the lines
    assert "one amino acid in twenty" in text
    assert "own composition baseline" in text and "per row" in text


def test_marker_shape_follows_role_not_position(results):
    """Dropping a model must not change the other model's marker or colour."""
    d = sC.load(results)
    only_second = d[d["role"] == "base_second"]
    assert only_second["role"].nunique() == 1
    assert sC.MARKERS["base_second"] != sC.MARKERS["base"]
