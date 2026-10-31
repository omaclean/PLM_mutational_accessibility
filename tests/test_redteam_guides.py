"""Red team: the guide loader and the composite layout loader.

Bugs these tests pinned (fixed in ``mutacc/guide.py`` and ``figures/panelise.py``):

* a repeated YAML key (two ``lineages:`` blocks, one table named twice) kept the last silently;
* a symlink inside the repo pointing outside it passed the path check;
* ``role: [x]`` raised a bare TypeError, ``label:`` (empty) became the string "None", and an
  unquoted ``id: 1.10`` became the lineage "1.1";
* ``panelise.py`` crashed with an ImportError traceback when PyMuPDF was missing, and accepted a
  negative or row-wide ``gap_pt`` and NaN row weights.
"""

import importlib
import sys
import textwrap

import pytest
import yaml

from mutacc import guide as G
from mutacc.guide import GuideError, load_guide

MINIMAL = {
    "name": "t", "virus": "v", "reference": "r", "results_dir": "results/t",
    "models": [{"key": "m", "label": "M", "role": "base"}],
    "lineages": [{"id": "L", "root": "data/L.fa", "root_type": "reconstructed"}],
    "tables": {"x": "x.tsv.gz"},
}


def write_doc(tmp_path, doc):
    path = tmp_path / "g.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def write_text(tmp_path, text):
    path = tmp_path / "g.yaml"
    path.write_text(textwrap.dedent(text))
    return path


# ---------------------------------------------------------------------------
# Malformed YAML and structure
# ---------------------------------------------------------------------------

BASE_YAML = """\
name: t
virus: v
reference: r
results_dir: results/t
models:
  - {key: m, label: M, role: base}
lineages:
  - {id: A, root: data/A.fa, root_type: observed}
tables:
  x: x.tsv.gz
"""


def test_the_base_yaml_used_below_is_valid(tmp_path):
    assert load_guide(write_text(tmp_path, BASE_YAML)).n_lineages == 1


def test_duplicate_top_level_key_is_refused(tmp_path):
    text = BASE_YAML + "lineages:\n  - {id: B, root: data/B.fa, root_type: observed}\n"
    with pytest.raises(GuideError, match="duplicate key"):
        load_guide(write_text(tmp_path, text))


def test_duplicate_table_name_is_refused(tmp_path):
    text = BASE_YAML + "  x: other.tsv.gz\n"
    with pytest.raises(GuideError, match="duplicate key"):
        load_guide(write_text(tmp_path, text))


def test_duplicate_key_inside_an_inline_mapping_is_refused(tmp_path):
    text = BASE_YAML.replace("{key: m, label: M, role: base}", "{key: m, label: M, role: base, role: finetune}")
    with pytest.raises(GuideError, match="duplicate key"):
        load_guide(write_text(tmp_path, text))


def test_tabs_and_unclosed_structures_are_guide_errors(tmp_path):
    for bad in ("name:\tt\n\t- x", "name: [t", "a: b: c", "{{{", "- - -\n  - :"):
        with pytest.raises(GuideError):
            load_guide(write_text(tmp_path, bad))


def test_empty_file_and_scalar_documents_are_guide_errors(tmp_path):
    for bad in ("", "   \n", "42", "just a string", "null", "[]"):
        with pytest.raises(GuideError):
            load_guide(write_text(tmp_path, bad))


def test_yaml_python_tags_cannot_execute(tmp_path):
    bad = BASE_YAML + "extra: !!python/object/apply:os.system ['echo no']\n"
    with pytest.raises(GuideError):
        load_guide(write_text(tmp_path, bad))


@pytest.mark.parametrize("override", [
    {"models": [{"key": "m", "label": "M", "role": ["base"]}]},        # unhashable role
    {"models": [{"key": "m", "label": None, "role": "base"}]},         # label would print as "None"
    {"models": [{"key": "m", "label": "", "role": "base"}]},
    {"models": [{"key": 7, "label": "M", "role": "base"}]},
    {"models": [{"key": "m", "label": "M", "role": "base", "colour": "red"}]},   # unknown key
    {"models": [{"key": "m", "label": "M"}]},                          # missing role
    {"models": "not a list"},
    {"models": [["m", "M", "base"]]},
    {"lineages": [{"id": 1.1, "root": "a.fa", "root_type": "observed"}]},        # unquoted 1.10
    {"lineages": [{"id": "", "root": "a.fa", "root_type": "observed"}]},
    {"lineages": [{"id": "A", "root": "a.fa", "root_type": ["observed"]}]},
    {"lineages": [{"id": "A", "root": "", "root_type": "observed"}]},
    {"lineages": [{"id": "A", "root": 5, "root_type": "observed"}]},
    {"lineages": "A"},
    {"tables": {"x": ""}},
    {"tables": {"x": 3}},
    {"tables": ["a.tsv"]},
    {"results_dir": ""},
    {"results_dir": 7},
    {"name": ""},
    {"name": ["a"]},
    {"reference": None},
])
def test_wrong_types_and_empties_are_guide_errors_never_bare_exceptions(tmp_path, override):
    doc = {**MINIMAL, **override}
    for k in [k for k, v in doc.items() if v is None]:
        del doc[k]
    with pytest.raises(GuideError):
        load_guide(write_doc(tmp_path, doc))


def test_numeric_table_key_is_kept_as_text_and_unknown_top_key_names_itself(tmp_path):
    g = load_guide(write_doc(tmp_path, {**MINIMAL, "tables": {"x": "x.gz", 2: "y.gz"}}))
    assert set(g.tables) == {"x", "2"}
    with pytest.raises(GuideError, match="unknown key"):
        load_guide(write_doc(tmp_path, {**MINIMAL, 5: "x"}))


def test_corpus_cutoff_accepts_a_date_and_rejects_nonsense(tmp_path):
    g = load_guide(write_text(tmp_path, BASE_YAML + "corpus_cutoff: 2015-01-01\n"))
    assert g.corpus_cutoff == "2015-01-01"
    for bad in ("corpus_cutoff: [2015]", "corpus_cutoff: yesterday-ish", "corpus_cutoff: {a: 1}"):
        with pytest.raises(GuideError):
            load_guide(write_text(tmp_path, BASE_YAML + bad + "\n"))


# ---------------------------------------------------------------------------
# Path traversal and symlink escapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/etc/passwd", "~/x", "~", "../x", "a/../../x", "./../x", "a/b/../../../x", "data/../../x",
])
@pytest.mark.parametrize("where", ["results_dir", "root", "table", "alignment"])
def test_path_traversal_is_refused_everywhere_a_path_is_accepted(tmp_path, path, where):
    doc = {**MINIMAL}
    if where == "results_dir":
        doc["results_dir"] = path
    elif where == "root":
        doc["lineages"] = [{"id": "A", "root": path, "root_type": "observed"}]
    elif where == "table":
        doc["tables"] = {"x": path}
    else:
        doc["lineages"] = [{"id": "A", "root": "a.fa", "root_type": "observed", "alignment": path}]
    with pytest.raises(GuideError):
        load_guide(write_doc(tmp_path, doc))


def test_a_symlink_inside_the_repo_that_points_outside_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.fa").write_text(">x\nATG\n")
    link = G.REPO_ROOT / "data" / "_redteam_link_tmp"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create symlinks here")
    try:
        doc = {**MINIMAL, "lineages": [{"id": "A", "root": "data/_redteam_link_tmp/secret.fa",
                                        "root_type": "observed"}]}
        with pytest.raises(GuideError, match="outside the repository"):
            load_guide(write_doc(tmp_path, doc))
        doc = {**MINIMAL, "results_dir": "data/_redteam_link_tmp"}
        with pytest.raises(GuideError, match="outside the repository"):
            load_guide(write_doc(tmp_path, doc))
    finally:
        link.unlink()


def test_a_symlink_that_stays_inside_the_repo_is_allowed(tmp_path):
    link = G.REPO_ROOT / "data" / "_redteam_inside_tmp"
    try:
        link.symlink_to(G.REPO_ROOT / "data" / "roots", target_is_directory=True)
    except OSError:
        pytest.skip("cannot create symlinks here")
    try:
        doc = {**MINIMAL, "lineages": [{"id": "A", "root": "data/_redteam_inside_tmp/x.fa", "root_type": "observed"}]}
        assert load_guide(write_doc(tmp_path, doc)).n_lineages == 1
    finally:
        link.unlink()


# ---------------------------------------------------------------------------
# Missing files, empty tables, dtypes
# ---------------------------------------------------------------------------

def test_missing_files_names_every_absent_table_and_root(tmp_path):
    g = load_guide(write_doc(tmp_path, MINIMAL))
    missing = {p.name for p in g.missing_files()}
    assert missing == {"x.tsv.gz", "L.fa"}


def test_read_table_refuses_header_only_and_empty_files_and_keeps_dtypes(tmp_path):
    import gzip

    g = load_guide(write_doc(tmp_path, MINIMAL))
    with gzip.open(tmp_path / "x.tsv.gz", "wt") as fh:
        fh.write("lineage\tposition\tp\nG.1.10\t007\t1e-9\nJ\t8\tNaN\n")
    frame = g.read_table("x", results=tmp_path)
    assert list(frame["lineage"]) == ["G.1.10", "J"]          # text keeps its trailing zero
    assert frame["position"].tolist() == [7, 8]                # and the numeric column is numeric
    assert frame["p"].isna().sum() == 1                        # NaN survives; callers must check it
    with gzip.open(tmp_path / "x.tsv.gz", "wt") as fh:
        fh.write("lineage\tposition\n")
    with pytest.raises(GuideError, match="empty"):
        g.read_table("x", results=tmp_path)


def test_unknown_model_and_role_helpers_fail_with_a_list(tmp_path):
    g = load_guide(write_doc(tmp_path, MINIMAL))
    with pytest.raises(KeyError, match="has"):
        g.model("ghost")
    assert g.models_with_role("finetune") == []


def test_two_models_may_share_a_role_but_not_a_key(tmp_path):
    two = {**MINIMAL, "models": [{"key": "a", "label": "A", "role": "base"},
                                 {"key": "b", "label": "B", "role": "base"}]}
    assert len(load_guide(write_doc(tmp_path, two)).models_with_role("base")) == 2


def test_shipped_guides_still_load_after_the_stricter_checks():
    for path in sorted((G.REPO_ROOT / "guides").glob("*.yaml")):
        assert load_guide(path).missing_files() == []


# ---------------------------------------------------------------------------
# panelise: layout loader and the PyMuPDF dependency
# ---------------------------------------------------------------------------

def layout_file(tmp_path, body):
    path = tmp_path / "layout.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_panelise_without_pymupdf_exits_with_an_instruction(monkeypatch, capsys):
    """Block ``fitz``, import the module fresh, and run main(): one clear line, no traceback."""
    monkeypatch.setitem(sys.modules, "fitz", None)          # makes ``import fitz`` raise ImportError
    monkeypatch.delitem(sys.modules, "panelise", raising=False)
    sys.path.insert(0, str(G.REPO_ROOT / "figures"))
    try:
        mod = importlib.import_module("panelise")
        assert mod.fitz is None
        with pytest.raises(SystemExit) as exc:
            mod.main(["--layout", "guides/composites/fig01.yaml", "--out", "/nonexistent-not-written"])
        assert "PyMuPDF" in str(exc.value) and "pymupdf" in str(exc.value)
        with pytest.raises(SystemExit):
            mod.render(object(), "/nonexistent-not-written")
    finally:
        sys.path.remove(str(G.REPO_ROOT / "figures"))
        sys.modules.pop("panelise", None)


@pytest.fixture
def panelise():
    pytest.importorskip("fitz")
    sys.path.insert(0, str(G.REPO_ROOT / "figures"))
    sys.modules.pop("panelise", None)
    mod = importlib.import_module("panelise")
    yield mod
    sys.path.remove(str(G.REPO_ROOT / "figures"))
    sys.modules.pop("panelise", None)


GOOD_LAYOUT = """\
name: t
panels:
  - {key: a, file: data/frozen_panels/PLACEHOLDER}
  - {key: b, file: data/frozen_panels/PLACEHOLDER}
rows:
  - [a, b]
"""


def frozen_pdf(panelise):
    pdfs = sorted((G.REPO_ROOT / "data" / "frozen_panels").glob("*.pdf"))
    if not pdfs:
        pytest.skip("no frozen panel to lay out")
    return pdfs[0].relative_to(G.REPO_ROOT).as_posix()


@pytest.mark.parametrize("extra, fragment", [
    ("gap_pt: -5", "gap_pt"),
    ("gap_pt: .nan", "gap_pt"),
    ("gap_pt: 100000", "gaps alone"),
    ("width: wide", "width"),
    ("width: -10", "positive"),
    ("width: 0", "positive"),
])
def test_layout_geometry_that_cannot_be_drawn_is_a_layout_error(panelise, tmp_path, extra, fragment):
    pdf = frozen_pdf(panelise)
    path = layout_file(tmp_path, GOOD_LAYOUT.replace("data/frozen_panels/PLACEHOLDER", pdf) + extra + "\n")
    layout = panelise.load_layout(path)
    with pytest.raises(panelise.LayoutError, match=fragment):
        panelise.validate(layout)


def test_nan_and_nonpositive_row_weights_are_refused(panelise, tmp_path):
    pdf = frozen_pdf(panelise)
    for weights in ("[1, .nan]", "[1, 0]", "[1, -1]", "[1]"):
        body = GOOD_LAYOUT.replace("data/frozen_panels/PLACEHOLDER", pdf).replace(
            "  - [a, b]", f"  - {{panels: [a, b], weights: {weights}}}")
        layout = panelise.load_layout(layout_file(tmp_path, body))
        with pytest.raises(panelise.LayoutError, match="weights"):
            panelise.validate(layout)


@pytest.mark.parametrize("body, fragment", [
    ("name: t\npanels: x\nrows: []\n", "lists"),
    ("name: t\npanels: []\nrows: 5\n", "lists"),
    ("name: t\npanels:\n  - {key: a, file: f.pdf, letter_x: 0.5}\nrows: [[a]]\n", "list of fractions"),
    ("name: t\npanels:\n  - {key: a, file: f.pdf, bogus: 1}\nrows: [[a]]\n", "unknown fields"),
    ("name: t\npanels:\n  - {file: f.pdf}\nrows: [[a]]\n", "key"),
    ("- a\n- b\n", "mapping"),
])
def test_malformed_layouts_raise_layout_errors_not_type_errors(panelise, tmp_path, body, fragment):
    with pytest.raises(panelise.LayoutError, match=fragment):
        panelise.load_layout(layout_file(tmp_path, body))


@pytest.mark.parametrize("file", ["/etc/hosts", "../x.pdf", "data/../../x.pdf", "data/frozen_panels/nope.pdf"])
def test_layout_paths_cannot_leave_the_repo_or_point_at_nothing(panelise, tmp_path, file):
    body = f"name: t\npanels:\n  - {{key: a, file: '{file}'}}\nrows: [[a]]\n"
    layout = panelise.load_layout(layout_file(tmp_path, body))
    with pytest.raises(panelise.LayoutError):
        panelise.validate(layout)


def test_layout_letter_fractions_must_be_inside_the_panel(panelise, tmp_path):
    pdf = frozen_pdf(panelise)
    for lx in ("[1.0]", "[-0.1]", "[.nan]", "[]"):
        body = f"name: t\npanels:\n  - {{key: a, file: {pdf}, letter_x: {lx}}}\nrows: [[a]]\n"
        layout = panelise.load_layout(layout_file(tmp_path, body))
        with pytest.raises(panelise.LayoutError):
            panelise.validate(layout)
