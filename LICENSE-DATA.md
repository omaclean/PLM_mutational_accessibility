# Licence for the data and figures

The MIT licence in [`LICENSE`](LICENSE) covers the **code**: everything under
`mutacc/`, `figures/`, `tools/` and `tests/`.

The **derived tables and figures** — `results/`, `output/`, `data/` and `figures/` —
are released under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
matching the licence of the preprint they illustrate. Reuse them freely, with attribution
to the citation in [`CITATION.cff`](CITATION.cff).

## What is not ours to license

A few inputs came from elsewhere and keep their own terms. None of them is redistributed
here in a form that would breach those terms, but if you rebuild the upstream analysis you
will need them.

| Input | Source | Terms |
|---|---|---|
| H3N2 HA sequence alignments | GISAID EpiFlu | **Not redistributed.** GISAID's terms forbid it. The guides omit the `alignment` key, and `.gitignore` keeps FASTA alignments out of the repository. Derived, non-reconstructible summaries ship instead. |
| Per-sequence accessions and collection dates | GISAID EpiFlu | **Not redistributed.** Only the thirteen-row aggregate in `lineage_panel` ships. |
| SARS-CoV-2 reference genome `NC_045512.2` | GenBank | Public domain. |
| Deep mutational scanning measurements | Lee et al. (PNAS 2018) and the other assays named in [`DATA.md`](DATA.md) | As published by their authors; cite them directly. |
| Nucleotide mutation spectra | Pauly et al. (2017), De Maio et al. (2021) | As published; cite them directly. |
| ESM-2 and ESM-C model weights | Meta AI / EvolutionaryScale | Their own licences. **No weights are in this repository** — only the scores they produced. |

See [`DATA.md`](DATA.md) for what each table is and where it came from, and
[`DB_SOURCING.md`](DB_SOURCING.md) for the sequence database the upstream analysis drew on.
