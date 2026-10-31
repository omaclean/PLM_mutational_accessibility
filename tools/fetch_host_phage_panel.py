#!/usr/bin/env python3
"""Fetch a random human and bacteriophage protein panel from UniProt, for the S3 companion figure.

    python tools/fetch_host_phage_panel.py --out data/host_phage_panel

S3 asks how well two untuned protein language models fill in a hidden amino acid across viral
proteins. The obvious next question is what the same models do on sequence that is not viral at
all. This fetches that comparison: a few human proteins and a few bacteriophage proteins.

Why the selection is random, and what "random" means here
--------------------------------------------------------
Hand-picking proteins for a comparison panel is choosing the answer. The rule instead:

1. Take every **reviewed** (SwissProt) entry matching the group's query. Reviewed only, because
   an unreviewed entry may be a gene prediction that was never observed as a protein.
2. Keep those whose length lies inside the range the viral panel already spans. Scoring masks
   every residue in turn, so cost grows with the square of length; without this a single random
   human hit such as titin would cost more than the whole panel, and a 30-residue peptide would
   carry a mean over too few positions to mean anything.
3. Sort the surviving accessions and draw from them with a seeded generator. Sorting matters:
   UniProt's own result order is not stable between releases, so sampling it directly would not
   reproduce. The accession list is what the seed indexes into.

Nothing here is written from memory. Every sequence is pulled by accession and the accession,
full protein name, organism and length that came back are recorded beside it, so a wrong hit is
visible in the table rather than silent in the figure.

The caveat that has to travel with the output: these are not a control for the viral panel. They
differ from it in training-set abundance, in conservation, and in how much near-identical
sequence surrounds them -- the same confound S3 already carries. They show how the models behave
off viral sequence; they do not isolate why.
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REST = "https://rest.uniprot.org/uniprotkb"
AGENT = "mutacc-host-phage-panel/1.0 (academic use)"
#: The 20 the models score over. Anything else (X, U, B, Z, O) is dropped and counted.
AA = set("ACDEFGHIKLMNPQRSTVWY")
#: Length window, matching the span of the 32 viral proteins already in the panel.
MIN_LEN, MAX_LEN = 97, 1353
N_PER_GROUP = 3
SEED = 20261014

GROUPS = {
    "Human proteins": "(organism_id:9606) AND (reviewed:true)",
    # Host taxonomy 2 is Bacteria, so this is "viruses whose host is a bacterium" -- phages,
    # without relying on a morphological family name. Caudovirales was dissolved in 2022 and
    # querying the old family names now silently misses most of them.
    "Bacteriophage proteins": "(virus_host_id:2) AND (reviewed:true)",
}


def _get(url: str, *, retries: int = 5):
    """One GET, with a short backoff, returning (body, next-page url or None).

    UniProt rate-limits rather than refusing, and the /stream endpoint drops the connection
    mid-body on a result set this size, so everything here goes through paged /search and
    follows the Link: rel="next" header the API supplies.
    """
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                link = r.headers.get("Link", "") or ""
                nxt = None
                if 'rel="next"' in link:
                    nxt = link[link.index("<") + 1:link.index(">")]
                return body, nxt
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as exc:   # noqa: PERF203
            last = exc
            time.sleep(2 * (attempt + 1))
    raise SystemExit(f"UniProt did not answer after {retries} tries: {last}")


def accessions_for(query: str) -> list[str]:
    """Every accession matching the query, sorted. Sorted so the seed means the same thing twice."""
    full = f"{query} AND (length:[{MIN_LEN} TO {MAX_LEN}])"
    url = f"{REST}/search?" + urllib.parse.urlencode(
        {"query": full, "format": "list", "size": "500"})
    found, pages = [], 0
    while url:
        body, url = _get(url)
        found += [ln.strip() for ln in body.decode().splitlines() if ln.strip()]
        pages += 1
        if pages > 200:
            raise SystemExit("refusing to page past 100,000 accessions; narrow the query")
    accs = sorted(set(found))
    if len(accs) < N_PER_GROUP:
        raise SystemExit(f"only {len(accs)} entries matched {full!r}; cannot draw {N_PER_GROUP}")
    return accs


def fetch_entries(accs: list[str]) -> list[dict]:
    """Full records for the drawn accessions, one request, in the order asked for."""
    q = " OR ".join(f"accession:{a}" for a in accs)
    url = f"{REST}/search?" + urllib.parse.urlencode({
        "query": q, "format": "json", "size": str(len(accs)),
        "fields": "accession,protein_name,organism_name,length,sequence"})
    body, _ = _get(url)
    data = json.loads(body)["results"]
    by_acc = {d["primaryAccession"]: d for d in data}
    missing = [a for a in accs if a not in by_acc]
    if missing:
        raise SystemExit(f"UniProt returned no record for {missing}")
    return [by_acc[a] for a in accs]


def _protein_name(entry: dict) -> str:
    d = entry.get("proteinDescription", {})
    for block in (d.get("recommendedName"), *(d.get("submissionNames") or [])):
        if block and block.get("fullName", {}).get("value"):
            return block["fullName"]["value"]
    return entry["primaryAccession"]


def _short(name: str, organism: str, group: str) -> str:
    """A label short enough for a y tick. The accession stays in the table for the real identity."""
    head = name.split("(")[0].strip().rstrip(",")
    if len(head) > 42:
        head = head[:39].rstrip() + "..."
    return head


def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group, query in GROUPS.items():
        pool = accessions_for(query)
        drawn = [pool[i] for i in sorted(rng.choice(len(pool), size=N_PER_GROUP, replace=False))]
        print(f"{group}: drew {N_PER_GROUP} of {len(pool):,} reviewed entries "
              f"{MIN_LEN}-{MAX_LEN} aa -> {', '.join(drawn)}", file=sys.stderr)
        for entry in fetch_entries(drawn):
            seq_full = entry["sequence"]["value"]
            seq = "".join(c for c in seq_full if c in AA)
            name = _protein_name(entry)
            organism = entry["organism"]["scientificName"]
            label = _short(name, organism, group)
            rows.append({
                "label": f"{entry['primaryAccession']} {label}",
                "display_label": label,
                "group": group,
                "accession": entry["primaryAccession"],
                "protein_name": name,
                "organism": organism,
                "length": len(seq),
                "length_uniprot": int(entry["sequence"]["length"]),
                "unknown_residues_removed": len(seq_full) - len(seq),
                "sequence": seq,
            })
    out = pd.DataFrame(rows)
    if out["accession"].duplicated().any():
        raise SystemExit("the same accession was drawn twice")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="directory for the panel files")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    df = build(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "host_phage_panel_sequences.tsv", sep="\t", index=False)
    with open(args.out / "host_phage_panel.faa", "w") as fh:
        for r in df.itertuples():
            fh.write(f">{r.accession} {r.display_label} | {r.organism}\n{r.sequence}\n")
    print(df[["accession", "group", "display_label", "organism", "length",
              "unknown_residues_removed"]].to_string(index=False))
    print(f"\n{len(df)} proteins, {int(df['length'].sum()):,} residues -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
