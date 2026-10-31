"""Every shipped guide validates, resolves, and a bad guide fails loudly."""

from pathlib import Path

import pytest
import yaml

from mutacc import guide as G
from mutacc.guide import GuideError, load_guide

GUIDES = sorted((G.REPO_ROOT / "guides").glob("*.yaml"))

MINIMAL = {
    "name": "t", "virus": "v", "reference": "r", "results_dir": "results/t",
    "models": [{"key": "m", "label": "M", "role": "base"}],
    "lineages": [{"id": "L", "root": "data/L.fa", "root_type": "reconstructed"}],
    "tables": {"x": "x.tsv.gz"},
}


def write(tmp_path, **overrides):
    doc = {**MINIMAL, **overrides}
    for k in [k for k, v in doc.items() if v is None]:
        del doc[k]
    path = tmp_path / "g.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_there_are_shipped_guides():
    assert GUIDES, "no guides shipped"


@pytest.mark.parametrize("path", GUIDES, ids=lambda p: p.name)
def test_shipped_guide_loads_and_every_file_exists(path):
    g = load_guide(path)
    assert g.missing_files() == []
    assert g.n_lineages == len(g.lineage_ids) == len(set(g.lineage_ids))


@pytest.mark.parametrize("path", GUIDES, ids=lambda p: p.name)
def test_shipped_guide_paths_are_relative(path):
    text = path.read_text()
    assert "/" + "home" not in text and "/" + "Users" not in text


def test_minimal_guide_loads(tmp_path):
    g = load_guide(write(tmp_path))
    assert g.n_lineages == 1 and g.models[0].role == "base"


@pytest.mark.parametrize("override, fragment", [
    ({"results_dir": "/abs/path"}, "absolute"),
    ({"results_dir": "../elsewhere"}, "'..'"),
    ({"tables": {"x": "/abs/x.tsv"}}, "absolute"),
    ({"models": [{"key": "m", "label": "M", "role": "nonsense"}]}, "unknown"),
    ({"models": [{"key": "m", "label": "M", "role": "base"}] * 2}, "duplicate"),
    ({"lineages": [{"id": "L", "root": "a.fa", "root_type": "maybe"}]}, "root_type"),
    ({"lineages": [{"id": "L", "root": "a.fa", "root_type": "observed"}] * 2}, "duplicate"),
    ({"lineages": []}, "non-empty"),
    ({"lineages": [{"id": "L", "root": "a.fa"}]}, "missing"),
    ({"tables": {}}, "non-empty"),
    ({"corpus_cutoff": "not-a-date"}, "not a date"),
    ({"virus": None}, "missing"),
    ({"typo_key": 1}, "unknown key"),
])
def test_bad_guides_fail_loudly(tmp_path, override, fragment):
    with pytest.raises(GuideError, match=fragment):
        load_guide(write(tmp_path, **override))


def test_non_mapping_and_bad_yaml_and_missing_file(tmp_path):
    bad = tmp_path / "b.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(GuideError, match="mapping"):
        load_guide(bad)
    bad.write_text("a: [unclosed")
    with pytest.raises(GuideError, match="YAML"):
        load_guide(bad)
    with pytest.raises(GuideError, match="not found"):
        load_guide(tmp_path / "nope.yaml")


def test_unknown_table_error_lists_what_exists(tmp_path):
    g = load_guide(write(tmp_path))
    with pytest.raises(KeyError, match="declares"):
        g.table_path("missing")


def test_unverified_root_is_a_warning_not_an_error(tmp_path):
    g = load_guide(write(tmp_path, lineages=[{"id": "L", "root": "a.fa", "root_type": "unverified"}]))
    assert g.warnings


def test_empty_table_raises(tmp_path):
    g = load_guide(write(tmp_path))
    (tmp_path / "x.tsv.gz").write_text("")
    with pytest.raises(Exception):
        g.read_table("x", results=tmp_path)
    import gzip
    with gzip.open(tmp_path / "x.tsv.gz", "wt") as fh:
        fh.write("a\tb\n")
    with pytest.raises(GuideError, match="empty"):
        g.read_table("x", results=tmp_path)


def test_parse_figure_args_loads_guide():
    args = G.parse_figure_args("t", ["--guide", str(GUIDES[0])])
    assert args.guide.name and args.results == args.guide.results_dir
