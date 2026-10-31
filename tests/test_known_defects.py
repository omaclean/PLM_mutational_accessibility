"""Defects found in the 15 Oct 2026 pre-resubmission audit, pinned.

Each test here states the behaviour the code *should* have. **All of them pass**:
defects 1 and 2 have since been fixed in ``mutacc/`` and these tests now guard
the fixes against regression; defect 3 was never a code fault but an undeclared
choice, and its tests make that choice visible.

An earlier version of this docstring described an ``xfail(strict=True)`` ratchet
for the tests that "fail today". No such marker was ever present -- checked
across ``tests/`` on 15 Oct 2026, the word appears only in prose -- and with the
fixes landed none is wanted: a strict xfail on a passing test fails the suite.
The ratchet the prose was reaching for is what these tests already are. If a new
defect is pinned here *before* it is fixed, mark it ``xfail(strict=True)`` then,
and the marker comes off with the fix.

State of each, verified by reading the modules on 15 Oct 2026:

1. ``strata._additive_residual`` now takes the total about zero (strata.py:323),
   matching its no-intercept design. Fixed.
2. ``metrics._check`` now rejects a non-finite or non-binary outcome
   (metrics.py:88-91). Fixed.
3. ``strata.primary_lineage`` is unchanged and the 13-lineage guide still names
   no primary lineage, so the headline lineage is still the alphabetical
   tie-break. Open, and a decision for the author.

The only number the first fix moved lives in a retired figure
(``dev/retired_figures/fig02_model_vs_accessibility_values.tsv`` still carries
the pre-fix 0.886896). No shipped table, caption or document quotes the additive
share, so nothing in the manuscript needs regenerating for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mutacc import metrics, strata
from mutacc.guide import REPO_ROOT, load_guide
from mutacc.strata import _additive_residual


# ---------------------------------------------------------------------------
# 1. The additive-share R² mixes an uncentred fit with a centred total
# ---------------------------------------------------------------------------
# FIXED. ``asym(A->B)`` is fitted as ``s_A - s_B``, a design that cannot produce
# a constant, so the fit has no intercept. The denominator used to be the centred
# total ``((values - values.mean())**2).sum()``; mixing the two is not a share of
# anything and is unbounded below. It is now the total about zero (strata.py:323).
#
# The two tests below look incompatible -- one demands the share lie in [0, 1],
# the next demands it equal the uncentred definition, which sounds unbounded.
# They are compatible, and only because of the fix: a least-squares fit is an
# orthogonal projection, so the residual sum of squares cannot exceed the total
# *about the same origin*. Against the uncentred total the share is in [0, 1] by
# construction. Against the centred total it was not, which is the defect.
#
# On the shipped 13-lineage table the asymmetries are not zero-mean
# (mut_asym mean -0.690, plm_asym mean -0.260), so the pre-fix values were
# understated: share_by_amino_acid_mut 0.8353 against 0.8490 uncentred, and
# share_by_amino_acid_plm 0.8869 against 0.9080. The direction was conservative
# -- the corrected values support the "most of it is additive" claim more
# strongly. The only place those numbers were ever written out is
# ``figures/make_fig02.py``, a retired figure, so the fix moved nothing that is
# shipped or quoted.

def test_additive_share_is_a_share():
    """With no additive structure and a large offset the share must stay in [0, 1]."""
    rng = np.random.default_rng(0)
    n_aa = 6
    first = rng.integers(0, n_aa, 200)
    second = (first + 1 + rng.integers(0, n_aa - 1, 200)) % n_aa
    values = 5.0 + rng.normal(0, 0.1, 200)      # offset, no s_A - s_B structure
    _, share = _additive_residual(values, first, second, n_aa)
    assert 0.0 <= share <= 1.0, f"share of variance is {share}"


def test_additive_share_matches_the_uncentred_definition():
    """The denominator is the total about zero, matching the no-intercept fit."""
    rng = np.random.default_rng(1)
    n_aa = 5
    first = rng.integers(0, n_aa, 120)
    second = (first + 1 + rng.integers(0, n_aa - 1, 120)) % n_aa
    values = rng.normal(2.0, 1.0, 120)
    resid, share = _additive_residual(values, first, second, n_aa)
    uncentred = 1.0 - (resid ** 2).sum() / (values ** 2).sum()
    assert np.isclose(share, uncentred)


def test_a_purely_additive_asymmetry_is_fully_explained():
    """Built exactly as ``s_A - s_B``, the share must be 1."""
    rng = np.random.default_rng(2)
    n_aa = 7
    scores = rng.normal(0, 1, n_aa)
    first = rng.integers(0, n_aa, 300)
    second = (first + 1 + rng.integers(0, n_aa - 1, 300)) % n_aa
    values = scores[first] - scores[second]
    resid, share = _additive_residual(values, first, second, n_aa)
    assert np.isclose(share, 1.0, atol=1e-9)
    assert np.allclose(resid, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. The ranking metrics validate the score but not the outcome
# ---------------------------------------------------------------------------
# FIXED. ``_check`` used to reject a non-finite *score* but pass any *outcome*
# through ``outcome > 0.5``, which silently read NaN as a negative and 2.0 or -1
# as a label: a single NaN label in a replacement table shifted a published AUROC
# instead of raising. It now rejects both (metrics.py:88-91), and the tests below
# hold that. The last one is the other half of the ratchet -- an over-strict
# check that rejected an ordinary integer or boolean outcome would be just as
# wrong, and would show up as a suite-wide failure rather than a quiet one.

@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param([1.0, np.nan, 0.0, 1.0], id="nan-label"),
        pytest.param([1.0, np.inf, 0.0, 1.0], id="inf-label"),
        pytest.param([1.0, 2.0, -1.0, 0.0], id="out-of-range-label"),
    ],
)
def test_auroc_rejects_an_outcome_that_is_not_zero_or_one(outcome):
    with pytest.raises(ValueError):
        metrics.auroc(np.array([3.0, 2.0, 1.0, 0.0]), np.array(outcome))


def test_auroc_still_accepts_a_clean_binary_outcome():
    """The new validation must not reject the ordinary case, in either dtype."""
    score = np.array([3.0, 2.0, 1.0, 0.0])
    assert np.isclose(metrics.auroc(score, np.array([1.0, 1.0, 0.0, 0.0])), 1.0)
    assert np.isclose(metrics.auroc(score, np.array([1, 1, 0, 0])), 1.0)
    assert np.isclose(metrics.auroc(score, np.array([True, True, False, False])), 1.0)


# ---------------------------------------------------------------------------
# 3. The headline lineage is chosen alphabetically
# ---------------------------------------------------------------------------
# strata.py:82. ``primary_lineage`` prefers a verified root, then the most
# scored replacements, then the sorted identifier. On the 13-lineage set all
# thirteen roots are ``reconstructed`` and all thirteen carry exactly the same
# number of rows, so the first two rules do nothing and the published lineage
# behind Fig 2D and S4 is picked by the alphabetical tie-break alone.
#
# This is defensible, but it is undeclared: the guide names no primary lineage,
# so renaming a lineage or re-sorting the table moves the headline figure to
# different data with no test failing. These tests make that visible; declaring
# ``primary_lineage`` in the guide would make them moot.

GUIDE_13 = load_guide(REPO_ROOT / "guides" / "h3n2_13lineage.yaml")


def test_the_thirteen_lineage_guide_declares_no_primary_lineage():
    assert GUIDE_13.primary_lineage is None, (
        "the guide now names a primary lineage; the tie-break below no longer decides it"
    )


def test_the_published_primary_lineage_is_decided_by_the_tie_break_alone():
    scores = GUIDE_13.read_table("replacement_scores", GUIDE_13.results_dir)
    pairs = [(lin.id, lin.root_type) for lin in GUIDE_13.lineages]

    assert {t for _, t in pairs} == {"reconstructed"}, "rule 1 (root provenance) now separates lineages"
    counts = scores["lineage"].astype(str).value_counts()
    assert counts.nunique() == 1, "rule 2 (most scored replacements) now separates lineages"

    chosen = strata.primary_lineage(scores, pairs, GUIDE_13.primary_lineage)
    assert chosen == sorted(counts.index)[0] == "C.1", (
        f"the headline lineage moved to {chosen!r}; Fig 2D and S4 now describe different data"
    )


# ---------------------------------------------------------------------------
# 4. A caption with no generator is invisible to the provenance test
# ---------------------------------------------------------------------------
# ``test_output_provenance.py`` credits a script with every file whose name
# starts with a stem it declares (``{stem}_*``). So a hand-written
# ``<stem>_caption.md`` beside a figure that genuinely is produced is reported as
# produced too, and nothing notices that no code writes it.
#
# Two are orphaned this way. ``table_s15_dms_alpha_optima_caption.md`` is the one
# A1 found. ``fig02d_strata_panel_caption.md`` is new: ``make_fig02d.py`` draws
# the panel and writes its values table but never calls ``write_caption``, and
# the shipped caption opens "**Figure 2D. ...**" with a number typed by hand --
# the exact drift the numbering map exists to stop.
#
# Marked strict-xfail: it fails today, and whoever gives those two captions a
# generator (or retires them) must take the marker off, so the fix cannot land
# silently and the orphan cannot quietly come back. This is the ratchet the file
# docstring used to claim and did not have.

CAPTION_WRITERS = {"write_caption"}


def _stems_passed_to_write_caption():
    """Stems any script hands to ``write_caption``, resolving module constants.

    A call that passes a loop variable (``make_fig04.py`` writes two figures in a
    loop) names no stem statically; those are returned as ``None`` and treated as
    covering the script's own declared stems.
    """
    import ast

    from test_output_provenance import SCRIPTS, _declared_outputs

    named, dynamic = set(), set()
    for path in SCRIPTS:
        tree = ast.parse(path.read_text())
        consts = {
            t.id: n.value.value
            for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            for t in n.targets if isinstance(t, ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) not in CAPTION_WRITERS:
                continue
            arg = node.args[2] if len(node.args) > 2 else None
            if isinstance(arg, ast.Constant):
                named.add(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in consts:
                named.add(consts[arg.id])
            else:
                dynamic |= _declared_outputs(path)[0]
    return named | dynamic


@pytest.mark.xfail(strict=True, reason="fig02d and table_s15 captions have no generator")
def test_every_shipped_caption_is_written_by_a_script():
    from mutacc.guide import REPO_ROOT

    written = _stems_passed_to_write_caption()
    shipped = {p.name[: -len("_caption.md")] for p in (REPO_ROOT / "output").glob("*_caption.md")}
    orphans = sorted(shipped - written)
    assert not orphans, (
        f"captions nothing writes: {orphans}. A caption with no generator cannot be "
        f"regenerated, so its number and its numbers can drift from the figure's."
    )
