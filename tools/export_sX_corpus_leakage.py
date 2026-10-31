"""Export the K-lineage / training-corpus overlap tables behind Table SX.

    python tools/export_sX_corpus_leakage.py \
        --corpus  <ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.fas> \
        --meta    <ncbiflu_HA_all_110424_noX_clu99_meta.csv> \
        --root    data/roots/h3n2_13lineage/J.2.nt.fa \
        --results results/h3n2_13lineage

Reviewer question (lines 423 to 425): do any of the nine subclade K replacements, or the
interacting pairs, appear in the pre-2015 training data? Answering it needs the corpus at
*sequence* level, which `corpus_membership.tsv.gz` cannot give -- that table is a marginal
per site, so it says a residue was seen somewhere and never that two were seen together.
This exporter therefore goes back to the corpus FASTA.

The corpus is the public HA-80 fine-tuning set (NCBI Influenza Virus Database, pulled
11 Apr 2024, clustered at 99% identity and filtered; 8,099 sequences, collection dates
1902 to 2015-02-27). It is public, so recomputing from it is reproducible outside the
repo; it is not shipped here because it is 4.6 MB of third-party sequence.

Each corpus sequence is aligned pairwise to the J.2 root's translated precursor rather
than taking a column from a multiple alignment. The corpus spans every HA subtype, whose
indel patterns differ, and a pairwise alignment to the backbone the replacements are
*defined on* is the thing that keeps a site meaning one site. `test_sX_corpus_leakage.py`
checks the per-site counts against the shipped `corpus_membership.tsv.gz`, which is built
from the project MSA, so the two routes have to agree where both are well defined.

Two tables are written:

  k_corpus_leakage.tsv.gz        one row per replacement: how often the derived residue
                                 occurs in the corpus, over all of it, within the H3
                                 subtype whatever the host, and within human H3N2, and
                                 when it was first seen in each.
  k_corpus_leakage_pairs.tsv.gz  one row per unordered pair of the nine: how often both
                                 derived residues occur in the same corpus sequence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: (label, J.2 precursor site, ancestral aa, derived aa), from LINEAGE_DEFINITIONS.md §4.
#: The site is the precursor / alignment coordinate; mature HA1 is site - 16 and mature
#: HA2 is site - 345, which is where "HA1:K2N" and "HA2:S49N" come from.
REPLACEMENTS = (
    ("HA1:K2N", 18, "K", "N"),
    ("HA1:T135K", 151, "T", "K"),
    ("HA1:S144N", 160, "S", "N"),
    ("HA1:N158D", 174, "N", "D"),
    ("HA1:I160K", 176, "I", "K"),
    ("HA1:Q173R", 189, "Q", "R"),
    ("HA1:K189R", 205, "K", "R"),
    ("HA1:T328A", 344, "T", "A"),
    ("HA2:S49N", 394, "S", "N"),
)

#: the ordered pairs that clear the pooled BH cut in Figure SG under the Zhang zero-sum
#: gauge. The pair table is unordered, so these are matched either way round.
INTERACTING_PAIRS = (
    ("HA1:S144N", "HA1:I160K"),
    ("HA1:K189R", "HA1:Q173R"),
    ("HA1:N158D", "HA1:Q173R"),
)

#: residues that carry no state: an alignment gap, or an ambiguity code.
NO_STATE = frozenset("-X*")


def read_fasta(path: Path):
    """Yield ``(name, sequence)``, taking the first whitespace-delimited field as name."""
    name, chunks = None, []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def backbone_protein(root: Path) -> str:
    """The J.2 root's translated precursor, stop codon removed."""
    from Bio.Seq import Seq

    _name, nucleotides = next(iter(read_fasta(root)))
    return str(Seq(nucleotides).translate()).rstrip("*")


def check_backbone(protein: str) -> None:
    """Every replacement's ancestral residue must be what the J.2 backbone carries.

    This is the one guard that catches a numbering slip. The nine sites are quoted in
    three coordinate systems across the manuscript and a frame or offset error would
    otherwise sail through and produce a table of plausible nonsense.
    """
    for label, site, ancestral, _derived in REPLACEMENTS:
        if site > len(protein):
            raise ValueError(
                f"{label}: site {site} is past the end of the {len(protein)} aa backbone"
            )
        found = protein[site - 1]
        if found != ancestral:
            raise ValueError(
                f"{label}: the J.2 backbone carries {found} at precursor site {site}, "
                f"not the {ancestral} the replacement is defined from"
            )


def corpus_states(corpus: Path, meta: Path, protein: str) -> pd.DataFrame:
    """One row per corpus sequence, one column per replacement, holding its residue.

    The residue is read at the backbone's own coordinate: each sequence is aligned to
    the backbone and the column facing the site is taken, or ``-`` where the sequence
    has a gap or does not reach that far.
    """
    from Bio import Align

    records = {row["name"]: row for row in csv.DictReader(meta.open())}

    aligner = Align.PairwiseAligner(
        scoring="blastp", mode="global", open_gap_score=-11, extend_gap_score=-1
    )
    # Corpus sequences are variously truncated at both ends, and charging those
    # truncations as gaps would drag the alignment out of frame at the far end.
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0

    sites = [site for _label, site, _a, _d in REPLACEMENTS]
    labels = [label for label, _s, _a, _d in REPLACEMENTS]

    rows = []
    for name, sequence in read_fasta(corpus):
        alignment = aligner.align(protein, sequence)[0]
        backbone_index, query_index = alignment.indices
        facing = {}
        for b, q in zip(backbone_index, query_index):
            if b >= 0:
                facing[b] = sequence[q] if q >= 0 else "-"
        record = records.get(name, {})
        row = {
            "name": name,
            "date": record.get("date", ""),
            "serotype": record.get("serotype", ""),
            "host": record.get("host", ""),
        }
        row.update({label: facing.get(site - 1, "-") for label, site in zip(labels, sites)})
        rows.append(row)

    if not rows:
        raise ValueError(f"no sequences read from {corpus}")
    return pd.DataFrame(rows)


#: The corpus subset the K lineage descends from, and the name the table carries for
#: it. The name is written into the table as *data* rather than typed into the figure
#: script, because nothing under figures/ may name a virus (tests/test_figure_rules.py).
SUBSET_SEROTYPE = "H3N2"
SUBSET_HOST = "Human"
SUBSET_LABEL = f"{SUBSET_HOST.lower()} {SUBSET_SEROTYPE}"

#: The subtype the human subset sits inside: every corpus sequence of that haemagglutinin
#: subtype whatever its neuraminidase and whatever its host, so H3N8 equine and H3N2 swine
#: are in it and human H3N2 is a part of it. Reported beside the human subset because the
#: two answer different questions -- whether the model read the residue in *any* H3
#: background, and whether it was circulating in the population the lineage emerged from --
#: and conflating them is what made the earlier single "subset" column ambiguous.
H3_PREFIX = "H3"
H3_LABEL = "H3"


def _is_subset(states: pd.DataFrame) -> pd.Series:
    return (states["serotype"] == SUBSET_SEROTYPE) & (states["host"] == SUBSET_HOST)


def _is_h3(states: pd.DataFrame) -> pd.Series:
    # startswith, not equality: the corpus carries H3N1 … H3N9, bare "H3", "H3Nx" and
    # "H3,mixed". It cannot catch H13 by accident, those serotypes reading "H13N…".
    return states["serotype"].str.startswith(H3_PREFIX)


def per_replacement(states: pd.DataFrame) -> pd.DataFrame:
    """How often each replacement's derived residue occurs in the corpus."""
    subset = states[_is_subset(states)]
    h3 = states[_is_h3(states)]
    corpus_last_date = max(d for d in states["date"] if d[:4].isdigit())
    rows = []
    for label, site, ancestral, derived in REPLACEMENTS:
        row = {
            "replacement": label,
            "precursor_site": site,
            "ancestral_aa": ancestral,
            "derived_aa": derived,
            "subset": SUBSET_LABEL,
            "h3_label": H3_LABEL,
            "corpus_last_date": corpus_last_date,
        }
        for prefix, frame in (("corpus", states), ("h3", h3), ("subset", subset)):
            typed = frame[~frame[label].isin(NO_STATE)]
            carrying = typed[typed[label] == derived]
            dated = [d for d in carrying["date"] if d[:4].isdigit()]
            row[f"{prefix}_sequences"] = len(frame)
            row[f"{prefix}_typed"] = len(typed)
            row[f"{prefix}_derived"] = len(carrying)
            # Over the whole group, not over the sequences readable at the site. A share
            # headed "of the training data" has to have the training data underneath it,
            # or two rows four sequences apart print shares a percentage point apart
            # (K2N and N158D did exactly that). Where a site is unreadable in part of the
            # group the share is therefore a floor, and *_typed records by how much.
            row[f"{prefix}_pct"] = (
                100.0 * len(carrying) / len(frame) if len(frame) else float("nan")
            )
            row[f"{prefix}_first_year"] = min((d[:4] for d in dated), default="")
        rows.append(row)
    return pd.DataFrame(rows)


def per_pair(states: pd.DataFrame) -> pd.DataFrame:
    """How often two derived residues occur in the same corpus sequence."""
    derived = {label: aa for label, _s, _a, aa in REPLACEMENTS}
    flagged = {frozenset(pair) for pair in INTERACTING_PAIRS}
    in_subset = _is_subset(states)

    rows = []
    for first, second in itertools.combinations(derived, 2):
        typed = states[
            ~states[first].isin(NO_STATE) & ~states[second].isin(NO_STATE)
        ]
        both = typed[
            (typed[first] == derived[first]) & (typed[second] == derived[second])
        ]
        both_subset = both[in_subset.reindex(both.index, fill_value=False)]
        dated = [d for d in both["date"] if d[:4].isdigit()]
        rows.append(
            {
                "replacement_a": first,
                "replacement_b": second,
                "interacting": frozenset((first, second)) in flagged,
                "corpus_typed": len(typed),
                "corpus_both_derived": len(both),
                "corpus_pct": (
                    100.0 * len(both) / len(typed) if len(typed) else float("nan")
                ),
                "corpus_first_year": min((d[:4] for d in dated), default=""),
                "subset": SUBSET_LABEL,
                "subset_both_derived": len(both_subset),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["interacting", "corpus_both_derived"], ascending=[False, False]
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, required=True,
                        help="HA-80 fine-tuning FASTA (protein)")
    parser.add_argument("--meta", type=Path, required=True,
                        help="corpus metadata CSV: name,strain,date,host,serotype,country")
    parser.add_argument("--root", type=Path, required=True,
                        help="J.2 lineage root, nucleotide CDS")
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)

    protein = backbone_protein(args.root)
    check_backbone(protein)
    print(f"J.2 backbone: {len(protein)} aa, all nine ancestral residues confirmed")

    states = corpus_states(args.corpus, args.meta, protein)
    print(f"corpus: {len(states):,} sequences, "
          f"{int(_is_h3(states).sum()):,} of them {H3_LABEL}, "
          f"{int(_is_subset(states).sum()):,} of them {SUBSET_LABEL}, "
          f"latest collection date {max(d for d in states['date'] if d[:4].isdigit())}")

    args.results.mkdir(parents=True, exist_ok=True)

    replacements = per_replacement(states)
    pairs = per_pair(states)

    written = []
    for frame, name in ((replacements, "k_corpus_leakage.tsv.gz"),
                        (pairs, "k_corpus_leakage_pairs.tsv.gz")):
        path = args.results / name
        frame.to_csv(path, sep="\t", index=False, float_format="%.6g")
        written.append(path)

    absent = replacements[replacements["corpus_derived"] == 0]
    quiet = pairs[pairs["corpus_both_derived"] == 0]
    print(f"{len(replacements) - len(absent)} of {len(replacements)} derived residues "
          f"occur in the corpus; {len(pairs) - len(quiet)} of {len(pairs)} pairs co-occur")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())