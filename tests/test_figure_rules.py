"""The engine's governing rule: nothing below figures/ names a lineage, model, virus or count."""

import ast
import re
from pathlib import Path

import pytest

from mutacc.guide import REPO_ROOT, load_guide

SCRIPTS = sorted((REPO_ROOT / "figures").glob("*.py"))
GUIDES = sorted((REPO_ROOT / "guides").glob("*.yaml"))

VIRUS_WORDS = re.compile(r"h3n2|h1n1|influenza|sars|cov-?2|spike|ha80|esm-?2|esm-?c|esm2_t33", re.I)
COUNT_WORDS = re.compile(r"\b(thirteen|five|13)\b", re.I)
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
FORBIDDEN_CALLS = re.compile(
    r"rcParams\s*(\.update|\[)|plt\.style\.use|\bsavefig\s*\(|import seaborn|from seaborn|sns\."
)


def forbidden_names():
    names = set()
    for g in map(load_guide, GUIDES):
        names |= {lin.id for lin in g.lineages} | {m.key for m in g.models}
    return names


def docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                out.add(id(body[0].value))
    return out


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_literals_in_figure_scripts(path):
    tree = ast.parse(path.read_text())
    skip = docstring_nodes(tree)
    names = forbidden_names()
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        v = node.value
        if isinstance(v, str):
            if v in names:
                problems.append(f"line {node.lineno}: {v!r} is a lineage/model name")
            if VIRUS_WORDS.search(v):
                problems.append(f"line {node.lineno}: {v!r} names a virus or model")
            if COUNT_WORDS.search(v):
                problems.append(f"line {node.lineno}: {v!r} hard-codes a count")
        elif isinstance(v, int) and not isinstance(v, bool) and v == 13:
            problems.append(f"line {node.lineno}: literal 13")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_style_comes_from_the_style_module(path):
    text = path.read_text()
    assert not HEX.search(text), "colour literal in a figure script; use mutacc.style roles"
    assert not FORBIDDEN_CALLS.search(text), "figure scripts must not set rcParams, call savefig or use seaborn"
