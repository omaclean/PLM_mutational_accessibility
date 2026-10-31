# Reference genomes

Whole genomes that a stated number in the manuscript is computed from. They are here so the
number can be recomputed rather than taken on trust; nothing in the figure pipeline reads them.

| File | Accession | Length | Fetched | What it is for |
|---|---|---|---|---|
| `NC_045512.2.fasta` | `NC_045512.2` | 29,903 nt | 22 Oct 2026, NCBI Nuccore | The SARS-CoV-2 genome composition behind the `sars_cov_2.csv` scaling factor. |

MD5 of `NC_045512.2.fasta`: `863ee5dba1da0ca3f87783782284d489` (as fetched, 70-column FASTA).

## `NC_045512.2.fasta`

Severe acute respiratory syndrome coronavirus 2 isolate Wuhan-Hu-1, complete genome — the
RefSeq record, which is the reference the De Maio et al. (2021) spectrum is reported against.
Fetched with:

```
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_045512.2&rettype=fasta&retmode=text"
```

Base counts: **A 8,954, C 5,492, G 5,863, T 9,594** — 29.94%, 18.37%, 19.61%, 32.08%, which is
the "29.9% A, 18.4% C, 19.6% G, 32.1% U" of the Methods to the precision printed there. The count
includes the 33-nt 3′ poly-A tail of the record; trimming it moves each percentage by less than
0.08 points and the scaling factor by 0.08%, so nothing here turns on that choice.

**Why the whole genome and not the spike CDS.** Before this file the only SARS-CoV-2 nucleotide
sequences in the repo were the five spike roots under `data/roots/sars_cov_2_spike/`, and spike
is not representative of the genome: it runs about 1.2 points lower in G and 1.2 higher in T,
which moves the scaling factor by 1.6–2.3% — twenty times the 0.08% the whole genome leaves.

| Composition from | A | C | G | T | Factor | vs 8.186e-7 |
|---|---|---|---|---|---|---|
| `NC_045512.2` whole genome | 29.94% | 18.37% | 19.61% | 32.08% | 8.1798e-7 | −0.08% |
| spike CDS, Wuhan-Hu-1 root | 29.43% | 18.92% | 18.39% | 33.25% | 8.3210e-7 | +1.65% |
| spike CDS, JN.1 root | 29.67% | 18.64% | 18.35% | 33.33% | 8.3691e-7 | +2.24% |
| spike CDS, LP.8.1 root | 29.67% | 18.63% | 18.37% | 33.33% | 8.3682e-7 | +2.23% |
| spike CDS, XEC root | 29.70% | 18.62% | 18.35% | 33.33% | 8.3730e-7 | +2.28% |
| spike CDS, NB.1.8.1 root | 29.73% | 18.62% | 18.33% | 33.33% | 8.3777e-7 | +2.34% |

The Methods' 19.6% G decides it on its own: that is the genome's value. No spike CDS in the repo
exceeds 18.92%, and the Wuhan-Hu-1 spike — the one that would otherwise be the obvious candidate —
is the furthest of them from the stated figure.

`tests/test_sars_cov_2_scaling.py` pins the accession, the length, the four base counts and the
scaling arithmetic they feed. See `data/spectra/README.md` for the arithmetic itself.
