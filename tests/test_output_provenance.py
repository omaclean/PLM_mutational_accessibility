"""Every file in ``output/`` must be written by exactly one script.

Two failure modes this catches, both of which have already happened here:

*Two producers.* ``make_s15.py`` and ``make_sN_simulation_control.py`` are two generations of the
same simulation-control figure and both write ``table_sN_simulation_control.tsv``
and ``supl_table_16.tsv``. Whichever runs last wins, so the shipped supplementary
table depends on the order the figure set is built in.

*Orphans.* ``output/`` is where the manuscript is assembled from, so a file no
script writes any more is a number that cannot be regenerated and that nothing
will ever update. ``table_s15_dms_alpha_optima.tsv`` and
``table_s15_simulation_control.tsv`` are orphans of a renumbering.

Producers are read out of the source rather than by running anything: each figure
script declares its output stems as ``STEM``-prefixed module constants and writes
any extra tables under a literal name.
"""

from __future__ import annotations

import ast
from collections import defaultdict

import pytest

from mutacc.guide import REPO_ROOT

OUTPUT = REPO_ROOT / "output"
SCRIPT_DIRS = [REPO_ROOT / "figures", REPO_ROOT / "tools", REPO_ROOT / "tools" / "exports"]
SUFFIXES = (".tsv", ".pdf", ".png", ".md")

#: Stems whose files are deliberately not in ``output/``: the script still exists
#: but its figure has been retired to ``dev/retired_figures/``.
RETIRED_STEMS = {
    "fig02_model_vs_accessibility",
    "fig04_alpha_vs_dms_reversible",
    "s17_dms_reversible_sc2",
    "sF_spectrum_delta",
}


def _script_paths():
    seen = {}
    for directory in SCRIPT_DIRS:
        for path in sorted(directory.glob("*.py")):
            if path.name != "__init__.py":
                seen[path.resolve()] = path
    return sorted(seen.values())


SCRIPTS = _script_paths()


#: Calls that write an output file. A bare string anywhere else in a script is a
#: mention -- an exporter naming the figure it feeds, a docstring, a caption --
#: and must not count as a claim to write the file, or every script that refers
#: to a figure would be reported as a second producer of it.
WRITING_CALLS = {"write_values", "write_caption", "save_figure", "to_csv", "savefig"}


def _declared_outputs(path):
    """Output basenames a script claims: ``STEM`` constants and stems or filenames
    passed to a writing call."""
    tree = ast.parse(path.read_text())
    stems, literals = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                # ``STEM``, ``STEM_PAIRS``, ``ALPHA_STEM`` ... all name an output stem.
                if "STEM" in target.id:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        stems.add(node.value.value)
                # ``OUT = REPO / "output" / "sY_dms_alpha_optima.tsv"``.
                # Only destination constants count: a script naming its *input*
                # (``SRC = .../fig04_alpha_vs_dms_values.tsv``) reads that file,
                # it does not write it.
                elif target.id.isupper() and ("OUT" in target.id or "DEST" in target.id):
                    for inner in ast.walk(node.value):
                        if (
                            isinstance(inner, ast.Constant)
                            and isinstance(inner.value, str)
                            and inner.value.endswith(SUFFIXES)
                            and "/" not in inner.value
                        ):
                            literals.add(inner.value)
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name not in WRITING_CALLS:
                continue
            # Walk each argument: the name may be a bare string, or built up as
            # ``args.out / "supl_table_16.tsv"``, which is a BinOp.
            for arg in node.args:
                for inner in ast.walk(arg):
                    if not (isinstance(inner, ast.Constant) and isinstance(inner.value, str)):
                        continue
                    text = inner.value
                    if "/" in text or text == path.name or not text:
                        continue
                    (literals if text.endswith(SUFFIXES) else stems).add(text)
    return stems, literals


def _producer_map():
    """basename in output/ -> list of scripts that write it."""
    produced = defaultdict(list)
    for path in SCRIPTS:
        stems, literals = _declared_outputs(path)
        for stem in stems:
            for name in OUTPUT.glob(f"{stem}.*"):
                produced[name.name].append(path.name)
            for name in OUTPUT.glob(f"{stem}_*"):
                produced[name.name].append(path.name)
        for literal in literals:
            if (OUTPUT / literal).exists():
                produced[literal].append(path.name)
    return {k: sorted(set(v)) for k, v in produced.items()}


PRODUCED = _producer_map()


def test_scripts_and_outputs_were_found():
    """Guard against the AST walk or the globs silently matching nothing."""
    assert len(SCRIPTS) > 20, [p.name for p in SCRIPTS]
    assert len(PRODUCED) > 50, sorted(PRODUCED)


@pytest.mark.parametrize("name", sorted(PRODUCED), ids=lambda n: n)
def test_no_output_file_has_two_producers(name):
    """Two scripts writing one file makes the shipped version depend on run order."""
    producers = PRODUCED[name]
    assert len(producers) == 1, f"output/{name} is written by {producers}"


#: Stems a static read of the source cannot resolve, with the reason.
#: ``sA``'s stem is built from the guide's virus name (figures/make_sA_spike.py:101);
#: the composites are assembled by ``panelise.py`` from the layouts in
#: ``guides/composites/``, one per YAML file.
DYNAMIC_STEMS = {"sA_spike_model_vs_accessibility"} | {
    f"{p.stem}_composite" for p in (REPO_ROOT / "guides" / "composites").glob("*.yaml")
}


def test_every_output_file_has_a_producer():
    """A file nobody writes cannot be regenerated, so it cannot be checked either."""
    on_disk = {p.name for p in OUTPUT.iterdir() if p.is_file() and p.name.endswith(SUFFIXES)}
    orphans = {
        name for name in on_disk - set(PRODUCED)
        if not any(name.startswith(stem) for stem in DYNAMIC_STEMS)
    }
    assert not orphans, f"output/ holds files no script writes: {sorted(orphans)}"


def test_the_dynamic_stem_allowance_is_still_needed():
    """Keep the allow-list honest: a stem that became statically resolvable should
    leave the list rather than sit there hiding a future orphan."""
    unused = {s for s in DYNAMIC_STEMS if any(n.startswith(s) for n in PRODUCED)}
    assert not unused, f"these stems are now resolved statically; drop them: {sorted(unused)}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_declared_stems_are_not_a_prefix_of_another_stem(path):
    """``STEM`` is globbed as ``{stem}_*`` when collecting a figure's values files, so
    one stem being a prefix of another would make a figure claim the other's tables."""
    stems, _ = _declared_outputs(path)
    others = set()
    for other in SCRIPTS:
        if other != path:
            others |= _declared_outputs(other)[0]
    for stem in stems:
        clashing = sorted(o for o in others if o.startswith(stem + "_") or stem.startswith(o + "_"))
        assert not clashing, f"{path.name}: stem {stem!r} overlaps {clashing}"
