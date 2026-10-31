"""S11 figure: draws from two tables. Only the SYNTHETIC fixture exists; nothing here is a result."""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

import make_sK_branch_codons as sK
from mutacc import branch_codons as bc

FIX = Path(__file__).parent / "fixtures" / "sK"
B, C = FIX / "branch_codons_SYNTHETIC.tsv.gz", FIX / "column_codons_SYNTHETIC.tsv.gz"
sys.path.insert(0, str(FIX.parent))
import make_sK_fixture as mk  # noqa: E402


@pytest.fixture(scope="module")
def tabs():
    return sK.load_tables(B, C)


def test_shipped_fixture_is_exactly_what_the_generator_makes(tmp_path):
    mk.main(tmp_path)
    for n in (B.name, C.name):
        assert (tmp_path / n).read_bytes() == (FIX / n).read_bytes()


def test_fixture_is_marked_synthetic_and_not_under_results():
    _, _, pb, pc = sK.load_tables(B, C)
    assert sK.is_synthetic(pb, pc) and "SYNTHETIC" in B.name
    assert "results" not in B.parts[-4:]
    # real tables are shipped under results/ since 12 Oct; none of them may be marked synthetic
    for path in (Path(__file__).resolve().parents[1] / "results").rglob("*_codons.tsv.gz"):
        _, prov = sK.read_table(path)
        assert not sK.is_synthetic(prov), f"{path.name} is marked synthetic but sits under results/"


def test_chart_numbers_come_from_the_tables(tabs):
    branch, column, *_ = tabs
    d = sK.chart_numbers(branch, column)
    assert d["branches"].sum() == len(branch)
    assert d["nonsyn"].sum() == sum(int(branch[f"nonsyn_{n}nt"].sum()) for n in (1, 2, 3))
    assert d["multi"].sum() == int(branch[["nonsyn_2nt", "nonsyn_3nt"]].to_numpy().sum())
    assert list(d["bin"]) == sorted(d["bin"]) and d["bin"].max() <= sK.TOP_BIN
    # the bin of one nucleotide change cannot hold a multi-position codon
    one = d.loc[d["bin"] == 1]
    assert not one["possible"].any() and (one["multi"] == 0).all() and (one["chance_multi"] == 0).all()


def test_the_chance_null_is_a_proper_occupancy(tabs):
    branch, column, *_ = tabs
    n_codons = len(sK.scored_codons(column))
    for k in (1, 2, 3, 5):
        e = sK.expected_codons(k, n_codons)
        # every change lands in exactly one codon, so the changes accounted for add back to k
        assert sum(j * e[j] for j in (1, 2, 3)) == pytest.approx(float(k))
        assert e[1] >= e[2] >= e[3] >= 0
    assert sK.expected_codons(1, n_codons)[2] == 0        # one change cannot occupy two positions
    with pytest.raises(ValueError):
        sK.expected_codons(0, n_codons)
    with pytest.raises(ValueError):
        sK.expected_codons(3 * n_codons + 1, n_codons)


def test_null_probabilities_take_j1_from_the_data_and_j23_from_the_code(tabs):
    branch, column, *_ = tabs
    q = sK.nonsyn_probs(branch, column)
    seen = int(branch["syn_1nt"].sum()) + int(branch["nonsyn_1nt"].sum())
    assert q[1] == pytest.approx(int(branch["nonsyn_1nt"].sum()) / seen)
    code = sK.nonsyn_prob_from_code(list(sK.scored_codons(column)["ref_codon"]))
    assert q[2] == code[2] and q[3] == code[3]
    # two- and three-position changes alter the amino acid almost regardless of the bases chosen
    assert 0.95 < q[2] <= 1.0 and 0.95 < q[3] <= 1.0


def test_wilson_interval_brackets_the_estimate():
    lo, hi = sK.wilson(0, 100)
    assert lo == 0.0 and 0 < hi < 0.1
    lo, hi = sK.wilson(17, 1904)
    assert lo < 17 / 1904 < hi and 0 <= lo and hi <= 1
    assert all(math.isnan(v) for v in sK.wilson(0, 0))


def test_cap_from_header():
    assert sK.cap_of({"max_branch_nt": "10 (strictly fewer kept)"}) == 10
    assert sK.cap_of({"max_branch_nt": "no limit"}) is None and sK.cap_of({}) is None


def test_inconsistent_tables_are_refused(tmp_path, tabs):
    branch, column, pb, pc = tabs
    bad = column.copy()
    bad.loc[0, "nonsyn_1nt"] += 1
    p = bc.write_table(bad, tmp_path / "c.tsv.gz", pc)
    with pytest.raises(ValueError, match="disagree"):
        sK.load_tables(B, p)
    bb = branch.copy()
    bb.loc[0, "nonsyn_2nt_min1"] = bb.loc[0, "nonsyn_2nt"] + 1
    with pytest.raises(ValueError):
        sK.load_tables(bc.write_table(bb, tmp_path / "b.tsv.gz", pb), C)
    bb = branch.copy()
    bb.loc[0, "nt_total"] = 0
    with pytest.raises(ValueError, match="no nucleotide"):
        sK.load_tables(bc.write_table(bb, tmp_path / "b2.tsv.gz", pb), C)
    with pytest.raises(ValueError, match="missing columns"):
        sK.load_tables(bc.write_table(branch.drop(columns=["reverted"]), tmp_path / "b3.tsv.gz", pb), C)
    with pytest.raises(ValueError, match="different"):
        sK.load_tables(bc.write_table(branch, tmp_path / "b4.tsv.gz", {**pb, "max_branch_nt": "7"}), C)


def test_empty_tables_are_refused(tmp_path, tabs):
    branch, _, pb, _ = tabs
    with pytest.raises(ValueError, match="empty"):
        sK.load_tables(bc.write_table(branch.iloc[0:0], tmp_path / "e.tsv.gz", pb), C)


def test_end_to_end_render_marks_the_output_synthetic(tmp_path):
    assert sK.main(["--branch-table", str(B), "--column-table", str(C), "--out", str(tmp_path)]) == 0
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {f"sK_branch_codons_SYNTHETIC_FIXTURE{x}" for x in (".pdf", ".png", "_values.tsv", "_caption.md")}
    assert (tmp_path / "sK_branch_codons_SYNTHETIC_FIXTURE.png").stat().st_size > 30_000
    cap = (tmp_path / "sK_branch_codons_SYNTHETIC_FIXTURE_caption.md").read_text()
    assert cap.startswith("**SYNTHETIC FIXTURE")
    v = pd.read_csv(tmp_path / "sK_branch_codons_SYNTHETIC_FIXTURE_values.tsv", sep="\t")
    assert (v["provenance"] == "SYNTHETIC FIXTURE, not data").all() and set(v["panel"]) == {"chart", "table"}


def test_a_non_synthetic_table_gets_the_plain_stem_and_no_warning(tmp_path, tabs):
    branch, column, pb, pc = tabs
    clean = {k: v for k, v in pb.items() if k != "note"} | {"tool": "x"}
    clean = {k: v for k, v in clean.items() if "SYNTH" not in v.upper()}
    b = bc.write_table(branch, tmp_path / "b.tsv.gz", clean)
    c = bc.write_table(column, tmp_path / "c.tsv.gz", clean)
    out = tmp_path / "o"
    sK.main(["--branch-table", str(b), "--column-table", str(c), "--out", str(out)])
    assert (out / "sK_branch_codons.png").exists()
    assert not (out / "sK_branch_codons_caption.md").read_text().startswith("**SYNTHETIC")


def test_counts_in_the_values_file_are_written_as_integers(tmp_path, tabs):
    """%.6g would turn a million-odd into 1.04153e+06; the counts must stay readable."""
    branch, column, pb, pc = tabs
    nodes = pd.DataFrame({"node": ["a"], "nonsyn": [1041527], "min1": [833761], "min2": [177936],
                          "min3": [29830], "nt1": [833761], "nt2": [177936], "nt3": [29830]})
    n = bc.write_table(nodes, tmp_path / "n.tsv.gz", pb)
    out = tmp_path / "o"
    sK.main(["--branch-table", str(B), "--column-table", str(C), "--long-range-nodes", str(n),
              "--out", str(out)])
    text = (out / "sK_branch_codons_SYNTHETIC_FIXTURE_values.tsv").read_text()
    assert "e+0" not in text and "1041527" in text


def test_render_is_deterministic(tmp_path):
    for d in ("a", "b"):
        sK.main(["--branch-table", str(B), "--column-table", str(C), "--out", str(tmp_path / d)])
    n = "sK_branch_codons_SYNTHETIC_FIXTURE"
    for ext in ("_values.tsv", "_caption.md"):
        assert (tmp_path / "a" / (n + ext)).read_bytes() == (tmp_path / "b" / (n + ext)).read_bytes()


def test_site_number_column_is_used_when_present(tmp_path, tabs):
    branch, column, pb, pc = tabs
    col = column.copy()
    col.insert(1, "site_number", col["codon"] - 16)
    c = bc.write_table(col, tmp_path / "c.tsv.gz", pc)
    assert sK.main(["--branch-table", str(B), "--column-table", str(c), "--out", str(tmp_path / "o")]) == 0


def test_help_says_real_tables_are_absent(capsys):
    with pytest.raises(SystemExit):
        sK.main(["--help"])
    out = capsys.readouterr().out
    assert "NOT SHIPPED" in out and "SYNTHETIC" in out


# ---------------------------------------------------------------------------
# the shipped real tables (22 Sep 2026 tree). Pinned so a silent change in the engine is noticed.
# ---------------------------------------------------------------------------

REAL = Path(__file__).resolve().parents[1] / "results" / "s11_h3n2_ha_tree"


@pytest.fixture(scope="module")
def real():
    return sK.load_tables(REAL / "branch_codons.tsv.gz", REAL / "column_codons.tsv.gz")


def test_real_tables_pass_the_figure_loader_and_are_pinned(real):
    branch, column, pb, pc = real
    assert len(branch) == 50094 and len(column) == 553
    assert int(branch["nonsyn_1nt"].sum()) == 29751 and int(branch["nonsyn_2nt"].sum()) == 264
    assert int(branch["nonsyn_3nt"].sum()) == 13
    assert (branch["nt_total"] < 10).all()


def test_real_columns_without_an_h3_residue_carry_no_events(real):
    """The 18 reference codons with no H3 residue were left out of the walk; they must read zero."""
    _, column, pb, _ = real
    blank = pd.to_numeric(column["site_number"], errors="coerce").isna()
    assert int(blank.sum()) == 18
    assert (column.loc[blank, "events"] == 0).all()
    assert pb.get("skipped_codons")


def test_real_site_numbers_increase_with_the_codon_and_never_repeat(real):
    _, column, *_ = real
    n = pd.to_numeric(column["site_number"], errors="coerce").dropna()
    assert n.is_monotonic_increasing and n.is_unique


def test_summary_table_rows_add_up_and_second_row_is_optional(tabs):
    branch, *_ = tabs
    one = sK.summary_table(branch, None)
    assert len(one) == 1
    row = one.iloc[0]
    assert row["need_1"] + row["need_2"] + row["need_3"] == row["total"] == sum(
        int(branch[f"nonsyn_{n}nt"].sum()) for n in (1, 2, 3))
    nodes = pd.DataFrame({"node": ["a", "b"], "nonsyn": [10, 20], "min1": [8, 15], "min2": [2, 4], "min3": [0, 1],
                          "nt1": [6, 10], "nt2": [3, 7], "nt3": [1, 3]})
    two = sK.summary_table(branch, nodes)
    assert two.iloc[1]["total"] == 30 and two.iloc[1]["need_2"] == 6   # pooled, not averaged


@pytest.mark.parametrize("edit", [
    lambda d: d.assign(min1=d["min1"] + 1),
    lambda d: d.assign(nt1=d["nt1"] + 1),
    lambda d: d.assign(min3=d["nt3"] + 1, nonsyn=d["nonsyn"] + d["nt3"] + 1 - d["min3"]),
])
def test_long_range_node_table_is_refused_when_inconsistent(edit):
    nodes = pd.DataFrame({"node": ["a"], "nonsyn": [10], "min1": [8], "min2": [2], "min3": [0],
                          "nt1": [6], "nt2": [3], "nt3": [1]})
    sK.check_nodes(nodes)                      # the consistent table passes
    with pytest.raises(ValueError):
        sK.check_nodes(edit(nodes))


def test_real_long_range_table_is_pinned_and_consistent():
    nodes, prov = sK.read_table(REAL / "long_range_nodes.tsv.gz")
    sK.check_nodes(nodes)
    assert len(nodes) == 44965
    tab = sK.summary_table(sK.read_table(REAL / "branch_codons.tsv.gz")[0], nodes)
    assert int(tab.iloc[1]["total"]) == 3563404 and int(tab.iloc[1]["need_2"]) == 537399
    changes, _ = sK.read_table(REAL / "long_range_changes.tsv.gz")
    assert len(changes) == 2869 and set(changes["n_min"]) == {1, 2, 3}
    assert int(changes["nodes"].sum()) == int(nodes["nonsyn"].sum())


# ---------------------------------------------------------------------------
# The PLM null pair (panel A, below the hard rule)
# ---------------------------------------------------------------------------

AA19 = [a for a in "ACDEFGHIKLMNPQRSTVWY" if a != "M"]
MASKED = Path(__file__).resolve().parents[1] / "results" / "h3n2_13lineage" / "replacement_scores_masked.tsv.gz"
ROOT_NT = Path(__file__).resolve().parents[1] / "data" / "roots" / "h3n2_13lineage" / "J.2.nt.fa"


def fake_masked(probs_by_site, steps_by_aa=None, lineage="J.2"):
    """A masked score frame for both models. ``probs_by_site`` maps position -> {aa: prob}."""
    steps_by_aa = steps_by_aa or {a: 1 + i % 3 for i, a in enumerate(AA19)}
    rows = []
    for model, _ in sK.PLM_NULL_MODELS:
        for pos, probs in probs_by_site.items():
            for aa, p in probs.items():
                rows.append({"model": model, "lineage": lineage, "position": pos, "ref_aa": "M",
                             "aa": aa, "plm_prob": p, "mut_prob": p, "nt_steps": steps_by_aa[aa],
                             "blosum62": 0.0, "neg_grantham": 0.0})
    return pd.DataFrame(rows)


def test_plm_null_with_a_flat_distribution_reproduces_the_genetic_code_split():
    """The row's whole claim: it is the genetic-code null re-weighted. Flat weights must give it back."""
    sites = {p: {a: 0.01 for a in AA19} for p in range(1, 6)}
    d = fake_masked(sites)
    tab = sK.plm_null_table(d, "J.2")
    flat = sK.plm_null_unweighted(d, "J.2")
    for _, row in tab.iterrows():
        for k in sK.NT:
            assert row[f"need_{k}"] / row["total"] == pytest.approx(flat[k])


def test_each_site_contributes_exactly_one_expected_change():
    sites = {1: {a: 0.5 if a == "A" else 1e-9 for a in AA19},      # a very confident site
             2: {a: 1.0 for a in AA19}}                             # and a flat one
    tab = sK.plm_null_table(fake_masked(sites), "J.2")
    for _, row in tab.iterrows():
        assert row["total"] == 2
        assert row["need_1"] + row["need_2"] + row["need_3"] == pytest.approx(2.0)


def test_a_confident_site_cannot_outweigh_the_others():
    """Equal weight per site is the chosen pooling; scaling one site's mass must change nothing."""
    sites = {1: {a: 0.9 if a == "A" else 1e-6 for a in AA19},
             2: {a: 1.0 for a in AA19}, 3: {a: 1.0 for a in AA19}}
    a = sK.plm_null_table(fake_masked(sites), "J.2")
    louder = {p: {k: v * (1000 if p == 1 else 1) for k, v in probs.items()}
              for p, probs in sites.items()}
    b = sK.plm_null_table(fake_masked(louder), "J.2")
    for k in sK.NT:
        assert a[f"need_{k}"].to_list() == pytest.approx(b[f"need_{k}"].to_list())


def test_probability_on_one_nucleotide_neighbours_lands_in_the_first_column():
    one = [a for a in AA19 if a in ("A", "C")]
    steps = {a: (1 if a in one else 3) for a in AA19}
    sites = {1: {a: (1.0 if a in one else 1e-12) for a in AA19}}
    tab = sK.plm_null_table(fake_masked(sites, steps), "J.2")
    assert tab["need_1"].to_list() == pytest.approx([1.0, 1.0], abs=1e-6)
    assert tab["need_3"].to_list() == pytest.approx([0.0, 0.0], abs=1e-6)


@pytest.mark.parametrize("edit,msg", [
    (lambda d: d.drop(columns=["nt_steps"]), "missing columns"),
    (lambda d: d[d["lineage"] != "J.2"], "no rows for lineage"),
    (lambda d: d[d["model"] != sK.PLM_NULL_MODELS[1][0]], "rows for"),
    (lambda d: d[~((d["position"] == 1) & (d["aa"] == "A"))], "19 non-reference"),
    (lambda d: d.assign(nt_steps=d["nt_steps"].where(d["aa"] != "A", 0)), "nt_steps outside"),
    (lambda d: d.assign(plm_prob=d["plm_prob"].where(d["aa"] != "A", -1.0)), "negative or not finite"),
    (lambda d: d.assign(plm_prob=0.0), "cannot be renormalised"),
])
def test_a_malformed_masked_table_is_refused(edit, msg):
    sites = {p: {a: 0.01 for a in AA19} for p in (1, 2)}
    good = fake_masked(sites)
    sK.plm_null_table(good, "J.2")                     # the good one passes
    with pytest.raises(ValueError, match=msg):
        sK.plm_null_table(edit(good), "J.2")


def test_a_repeated_amino_acid_is_refused():
    sites = {1: {a: 0.01 for a in AA19}}
    d = fake_masked(sites)
    dup = pd.concat([d, d[d["aa"] == "A"]], ignore_index=True).drop(
        index=d.index[d["aa"] == "C"][0:0])
    # 20 rows on site 1 now; the count check fires before the duplicate check, and both are errors
    with pytest.raises(ValueError):
        sK.plm_null_table(dup, "J.2")


def test_base_and_finetune_are_not_swapped():
    """``ESM2_650M_HA80_raw`` is the BASE model. Swapping the pair would invert the panel."""
    assert sK.PLM_NULL_MODELS[0][0] == "ESM2_650M_HA80_raw"
    assert sK.PLM_NULL_MODELS[1][0] == "ESM2_650M_HA80"
    prov = Path(__file__).resolve().parents[1] / "output" / "unmasked_check" / \
        "fig03_13lineage" / "J.2__provenance.json"
    if prov.exists():
        import json
        p = json.loads(prov.read_text())
        text = json.dumps(p)
        assert "ESM2-HA80" in text, "provenance no longer names the fine-tuned checkpoint"


def test_plm_rows_are_appended_last_and_fenced_by_a_hard_rule(tabs):
    branch, *_ = tabs
    plm = sK.plm_null_table(fake_masked({p: {a: 0.01 for a in AA19} for p in (1, 2)}), "J.2")
    tab = sK.summary_table(branch, None, None, plm)
    assert list(tab["row"])[-2:] == ["plm base", "plm finetune"]
    assert sK.summary_table(branch, None, None).shape[0] == len(tab) - 2
    keys = list(tab["row"])
    assert next(i for i, k in enumerate(keys) if k.startswith("plm ")) == len(keys) - 2


# --- the real shipped table -------------------------------------------------

@pytest.fixture(scope="module")
def masked_real():
    return sK.read_plm_scores(MASKED)


def test_real_masked_table_gives_the_pinned_j2_plm_null(masked_real):
    tab = sK.plm_null_table(masked_real, "J.2").set_index("row")
    assert tab.loc["plm base", "total"] == 566 and tab.loc["plm finetune", "total"] == 566
    share = {k: {r: tab.loc[r, f"need_{k}"] / 566 for r in tab.index} for k in sK.NT}
    assert share[1]["plm base"] == pytest.approx(0.6437, abs=5e-4)
    assert share[2]["plm base"] == pytest.approx(0.3085, abs=5e-4)
    assert share[3]["plm base"] == pytest.approx(0.0478, abs=5e-4)
    assert share[1]["plm finetune"] == pytest.approx(0.8768, abs=5e-4)
    assert share[2]["plm finetune"] == pytest.approx(0.1174, abs=5e-4)
    assert share[3]["plm finetune"] == pytest.approx(0.0058, abs=5e-4)
    # the direction of the claim: the fine-tune is the more nucleotide-biased of the two,
    # and both are far more biased than the genetic code alone
    flat = sK.plm_null_unweighted(masked_real, "J.2")
    assert share[1]["plm finetune"] > share[1]["plm base"] > flat[1]


def test_real_masked_nt_steps_agree_with_the_j2_root_codons(masked_real):
    """The N145S lesson: never trust a stored ref_aa/nt_steps against the live FASTA."""
    from mutacc.accessibility import GENETIC_CODE, nucleotide_steps
    seq = "".join(l.strip() for l in ROOT_NT.read_text().splitlines() if not l.startswith(">"))
    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    j2 = masked_real[(masked_real["lineage"] == "J.2") &
                     (masked_real["model"] == sK.PLM_NULL_MODELS[0][0])]
    assert len(j2) == 566 * 19
    bad_ref = bad_steps = 0
    for r in j2.itertuples():
        codon = codons[r.position - 1]
        if GENETIC_CODE.get(codon) != r.ref_aa:
            bad_ref += 1
            continue
        if nucleotide_steps(codon, r.aa) != r.nt_steps:
            bad_steps += 1
    assert bad_ref == 0 and bad_steps == 0


def test_real_masked_table_carries_every_lineage_for_both_models(masked_real):
    per = masked_real.groupby(["model", "lineage"]).size()
    assert set(per.index.get_level_values("model")) == {m for m, _ in sK.PLM_NULL_MODELS}
    assert (per == per.iloc[0]).all(), "lineages do not all carry the same number of rows"


def test_end_to_end_with_the_plm_null_writes_exact_shares(tmp_path, tabs):
    """The PLM rows are expected changes, so they get float columns of their own; the integer
    count columns must stay integral or the writer's %.6g sends the millions into exponents."""
    sK.main([*("--branch-table", str(B)), *("--column-table", str(C)),
             *("--plm-null-scores", str(MASKED)), *("--plm-null-lineage", "J.2"),
             *("--out", str(tmp_path))])
    out = next(tmp_path.glob("*_values.tsv"))
    v = pd.read_csv(out, sep="\t", keep_default_na=False, na_values=[""])
    plm = v[v["row"].isin(["plm base", "plm finetune"])]
    assert len(plm) == 2
    for _, row in plm.iterrows():
        assert sum(row[f"share_{k}"] for k in sK.NT) == pytest.approx(1.0)
        # the expected changes are the drawn cells, to 1 dp, and they add to the site count
        assert sum(row[f"expected_{k}"] for k in sK.NT) == pytest.approx(row["total"], abs=0.2)
        for k in sK.NT:
            assert pd.isna(row[f"need_{k}"]), "an expected change was written as a count"
            assert row[f"expected_{k}"] == pytest.approx(row[f"share_{k}"] * row["total"], abs=0.05)
        # the share is carried at full precision, not recovered from the 1 dp cell
        assert row["share_1"] != pytest.approx(row["expected_1"] / row["total"], abs=1e-12)
    # the count rows must be untouched by all of that
    counted = v[v["row"].isin(["one branch", "oldest node to newest"])]
    for _, row in counted.iterrows():
        for k in sK.NT:
            assert pd.isna(row[f"expected_{k}"])
            assert float(row[f"need_{k}"]).is_integer()
    text = out.read_text()
    assert "e+0" not in text
    for line in text.splitlines():
        if "\ttable\t" in line and ("\tplm base\t" in line or "\tplm finetune\t" in line):
            continue
        if "\ttable\t" in line:
            for c in line.split("\t")[4:8]:
                assert c and "." not in c, f"count column written as a float: {c!r}"


# ---------------------------------------------------------------------------
# The null's codons (row 3), taken from the same root the PLM rows score
# ---------------------------------------------------------------------------

GUIDE_13 = Path(__file__).resolve().parents[1] / "guides" / "h3n2_13lineage.yaml"


def test_root_codons_reads_the_scorable_codons_of_the_lineage_root():
    """``len`` is what the caption quotes, so the stop codon must not be in it.

    The root is 1,701 nt = 567 codons, the last of which is a stop: it has no replacements
    to reach and ``null_distances`` skips it, so counting it would make the caption claim one
    codon more than the row is built from.
    """
    from mutacc.guide import load_guide
    codons = sK.root_codons(load_guide(GUIDE_13), "J.2")
    assert len(codons) == 566
    assert list(codons.columns) == ["codon_from"]
    aas = {sK.GENETIC_CODE.get(c) for c in codons["codon_from"]}
    assert None not in aas and "*" not in aas, "a stop or unreadable codon reached the null"


def test_root_codons_refuses_a_lineage_the_guide_does_not_have():
    from mutacc.guide import load_guide
    with pytest.raises(KeyError, match="no lineage"):
        sK.root_codons(load_guide(GUIDE_13), "not-a-lineage")


def test_the_null_stands_on_the_same_sites_as_the_plm_rows():
    """Until 23 Oct 2026 the null was built from ``baseline_codons.tsv.gz`` -- the tree's own
    reference DQ864721, 553 annotated codons less 18 with no residue in the mature protein,
    so 535. The PLM rows beside it score the 566-codon J.2 root, and a reader could only
    recover the 31-codon gap by dividing the null's total by 19. Both now use the one root.
    """
    from mutacc.guide import load_guide
    codons = sK.root_codons(load_guide(GUIDE_13), "J.2")
    nd = sK.null_distances(codons)
    assert sum(nd.values()) == 19 * len(codons) == 10754


def test_the_null_matches_the_nt_steps_of_the_scores_the_models_were_run_on():
    """The strongest form of the same check: the null's split by fewest-nucleotide-changes
    must be the split in the scored table itself, because both describe the same root. A
    mismatch means the null is reading a different sequence from the models.
    """
    from mutacc.guide import load_guide
    scores = pd.read_csv(MASKED, sep="\t")
    one = scores[(scores["lineage"] == "J.2")
                 & (scores["model"] == scores["model"].iloc[0])]
    want = one.groupby("nt_steps").size().to_dict()
    nd = sK.null_distances(sK.root_codons(load_guide(GUIDE_13), "J.2"))
    assert nd == {int(k): int(v) for k, v in want.items()}
