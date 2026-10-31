"""The shipped S3 inputs: sequences and per-position scores must agree with each other."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "data" / "viral_panel" / "viral_panel_sequences.tsv"
FAA = ROOT / "data" / "viral_panel" / "viral_panel.faa"
SITES = ROOT / "results" / "viral_panel" / "viral_panel_sites.tsv.gz"
SUMMARY = ROOT / "results" / "viral_panel" / "viral_panel_summary.tsv.gz"
AA = set("ACDEFGHIKLMNPQRSTVWY")


@pytest.fixture(scope="module")
def seqs():
    return pd.read_csv(SEQ, sep="\t")


@pytest.fixture(scope="module")
def sites():
    return pd.read_csv(SITES, sep="\t")


@pytest.fixture(scope="module")
def summary():
    return pd.read_csv(SUMMARY, sep="\t")


#: What the shipped panel is made of. Hard-coded so a half-finished export, or a fetch that
#: silently dropped a protein, fails here instead of quietly shortening the figure.
EXPECTED_GROUPS = {"Influenza viruses": 11, "Coronaviruses": 5, "Other viruses": 16,
                   "Eukaryotes": 3, "Bacteria": 3, "Phages": 3}


def test_the_panel_holds_the_viral_groups_and_the_cellular_comparison(seqs, summary):
    assert seqs.groupby("group").size().to_dict() == EXPECTED_GROUPS
    # the comparison groups must sort below the viral ones, or the figure stops being about viruses
    order = summary.drop_duplicates("group").set_index("group")["group_order"].to_dict()
    viral = [order[g] for g in ("Influenza viruses", "Coronaviruses", "Other viruses")]
    assert max(viral) < min(order[g] for g in ("Eukaryotes", "Bacteria", "Phages"))


def test_sequences_are_standard_residues_with_an_accession_each(seqs):
    assert len(seqs) == sum(EXPECTED_GROUPS.values())
    assert seqs["accession"].is_unique and seqs["label"].is_unique
    assert seqs["accession"].notna().all()
    for s in seqs["sequence"]:
        assert set(s) <= AA
    assert (seqs["sequence"].str.len() == seqs["length"]).all()


def test_only_removed_residues_explain_a_length_difference_from_uniprot(seqs):
    gap = seqs["length_uniprot"] - seqs["length"]
    assert (gap == seqs["unknown_residues_removed"]).all() and (gap >= 0).all()
    assert seqs.loc[seqs["unknown_residues_removed"] > 0, "accession"].tolist() == ["P17005"]


def test_fasta_and_table_hold_the_same_sequences(seqs):
    fasta = {}
    for block in FAA.read_text().split(">")[1:]:
        head, *rest = block.strip().split("\n")
        fasta[head.split("|")[1]] = "".join(rest)
    assert fasta == dict(zip(seqs["accession"], seqs["sequence"]))


def test_every_position_of_every_protein_is_scored_once_by_each_model(seqs, sites):
    assert sites.groupby(["model", "accession"]).size().unstack(0).notna().all().all()
    for (_, acc), block in sites.groupby(["model", "accession"]):
        want = seqs.set_index("accession").loc[acc, "sequence"]
        assert list(block["position"]) == list(range(1, len(want) + 1))
        assert "".join(block["wt"]) == want


def test_probabilities_are_probabilities_and_top1_agrees_with_them(sites):
    assert sites["p_wt"].between(0, 1).all() and not sites["p_wt"].isna().any()
    assert (sites.loc[sites["top1_correct"], "p_wt"] >= 0.05).all()       # a correct top guess is never negligible
    assert (sites["top1"] == sites["wt"]).eq(sites["top1_correct"]).all()
    assert set(sites["top1"]) <= AA


def test_summary_is_exactly_the_mean_of_the_site_table(sites, summary):
    got = sites.groupby(["model", "accession"])["p_wt"].mean()
    s = summary.set_index(["model", "accession"])["mean_p_wt"]
    assert np.allclose(got.reindex(s.index), s, atol=1e-6)
    top = sites.groupby(["model", "accession"])["top1_correct"].mean()
    assert np.allclose(top.reindex(s.index), summary.set_index(["model", "accession"])["top1"], atol=1e-6)


def test_composition_baseline_is_the_sum_of_squared_frequencies(seqs, summary):
    for r in seqs.itertuples():
        f = pd.Series(list(r.sequence)).value_counts(normalize=True)
        want = float((f ** 2).sum())
        got = summary.loc[summary["accession"] == r.accession, "composition_baseline"]
        assert np.allclose(got, want)


def test_two_models_and_each_has_a_distinct_role(summary):
    assert summary["model"].nunique() == 2 and summary.groupby("model")["role"].nunique().eq(1).all()
    assert summary["role"].nunique() == 2
