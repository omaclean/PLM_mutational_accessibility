# S3 and S11: run log and progress

> ## ⚠️ `N145S` was a phantom mutation in Figure 1B/C — FIXED 20 Oct 2026
>
> **The J.2 → K transition is NINE replacements, not ten:**
> `K2N`, `T135K`, `S144N`, `N158D`, `I160K`, `Q173R`, `K189R`, `T328A`, `HA2:S49N`.
> Residue 145 is Serine in J.2, J.2.4 and K alike, and in the A/Aichi/2/1968 H3-numbering
> reference. There is no change at 145 on that path, in nucleotide or amino acid.
>
> **Root cause.** The panel diffed a *stored* results table's `ref_aa` against a *live*
> reference FASTA. The `J.2_int` root was corrected at precursor codon 161 (`aat` → `agt`)
> on 15 Jun 2026 and the May tables were never rebuilt, so the 2 Jul 2026 render compared a
> stale J.2 against a current K and emitted both `S144N` and `N145S`. Nothing warned.
>
> **Fix.** `guides/composites/fig01.yaml` places the fine-tuned model (`ESM2_650M_HA80`)
> render (`data/frozen_panels/fig1bc_ranked_mutations_finetuned_esm2_ha80.pdf`), matching the
> Results text (HA2:S49N highest at rank 7, T328A lowest at rank 578); `output/fig01_composite.pdf`
> and `fig01_composite.png` carry exactly the canonical nine. The 2 Jul file is kept as
> `data/frozen_panels/fig1bc_ranked_mutations.pdf` **for provenance only — do not place it.**
>
> **Standing risk this exposed:** re-plotting an old table against newer roots can silently
> invent or erase replacements. Full account in `LINEAGE_DEFINITIONS.md` section 6.


Kept up to date as work lands. Times are server local time. Newest entries at the bottom of each table.

## What each figure is now

* **S3**: how well ESM-2 650M and ESM-C 600M (untuned) recover the true amino acid at a masked
  position, across a panel of viral proteins. **Every** position is masked in turn (not every 7th).
  Headline per protein = mean probability the model gave the true residue. Top-1 hit rate is kept
  as a second column. No fine-tuning anywhere (decision 12 Oct, replaces D5).
* **S11**: nucleotide changes per codon per internal branch of the H3N2 HA UShER tree.
  Tables: one row per branch, one row per codon column.

## Inputs, and where they came from

> ⚠️ **This table is stale (checked 15 Oct 2026) in two ways, and both matter.**
>
> **The tree is wrong.** The `h3_seg4_tree.pb` / 67,879-tip row below describes the
> 12 Oct run, which was superseded. `results/s11_h3n2_ha_tree/MANIFEST.tsv` and the
> shipped `summary.txt` both say **`h3n2_1968.pb`, 224,731 tips**, pruned from the
> treepatch repair `usher.pb` of **24 Sep 2026** — and `summary.txt`'s counts
> (224,731 target tips, 294,578 branches walked, 50,094 scored) are the numbers S11
> actually reports. The manifest is authoritative; this table is not.
>
> **The database is renamed.** `iav-db.rebuilt.db` no longer exists. It is now
> `iav-db.db` (33 GB, 15 Oct 13:45) in the same `db_build/` directory, so nothing
> described here runs as written.
>
> Both are tracked as [`docs/notes/PLAN.md`](PLAN.md) risks 23 and 25. The fix — a test that
> compares a run log against the manifest beside the tables it describes, so this
> cannot drift silently again — is [`DB_SOURCING.md`](../../DB_SOURCING.md) §7.7.

| Input | Source | Date of source |
|---|---|---|
| 32 viral protein sequences (21,003 residues) | UniProt reviewed entries via `fetch_viral_panel.py`; accession, protein name, organism recorded beside each sequence | fetched earlier; copy in `data/viral_panel/` |
| UShER tree used for S11 | `h3_seg4_tree.pb` (67,879 tips), **22 Sep 2026** | 22 Sep 2026 |
| UShER reference record | `DQ864721` aligned row, `sequence_alignment` table of `iav-db.db` (renamed 15 Oct; see the note above) | DB rebuilt 8 Oct 2026 |
| Newer tree in the DB | `trees` table, `usher_refset_4_aln_merged_MSA_dedup`, newick only, created **2 Oct 2026**. No `.pb`, so no per-branch mutations | 2 Oct 2026 |

## Run log

| Started | Finished | What | Where | Status |
|---|---|---|---|---|
| 12 Oct 17:25 | 17:25 | Scorer self-test, one short protein, both models; batched vs single pass agree (3e-7, 0) | gpu1 | ok |
| 12 Oct 17:25:41 | 18:55:50 | S3 all-site scoring, ESM-2 then ESM-C (`tmp_explore/IAV_vs_SC2_criss_cross/run_allsites.sh`, log `allsites_run.log`). ESM-2 17:25 to 18:05 (2,410 s), ESM-C 18:06 to 18:55 (2,990 s). 21,003 positions each; both exit 0 | gpu1 | ok |
| 12 Oct 18:57 | 18:58 | `tools/export_viral_panel.py`: sequences to `data/viral_panel/`, scores to `results/viral_panel/` | clean_repo | ok |
| 12 Oct 18:58 | 18:59 | S3 figure drawn (`figures/make_sC_viral_panel.py`); label collisions fixed on the second draw | `output/sC_viral_panel_recovery.*` | ok |
| 12 Oct 17:28 | 17:29 | `matUtils extract` on the 22 Sep tree: Newick, mutations per branch, reference VCF | `tmp_explore/s11_real/matutils_export/` | ok |
| 12 Oct 17:29 | 17:29 | S11 tables, 22 Sep tree, all tips, branches < 10 nt, terminal branches excluded | `tmp_explore/s11_real/tables_pb22sep/` | ok; first real S11 output, but see next rows |
| 12 Oct 17:34 | 17:34 | H3 mature numbering for the 553 reference codons, from 300 sampled GenBank tips (seed 1); landmarks Y98 W153 H183 Y195 all pass | `tmp_explore/s11_real/h3_numbering.tsv` (`tools/h3_numbering_from_db.py`) | ok |
| 12 Oct 17:35 | 17:35 | S11 tables with `--skip-unnumbered`: 18 codons with no H3 residue left out | `tmp_explore/s11_real/tables_pb22sep_h3only/`, shipped to `results/s11_h3n2_ha_tree/` | ok; **these are the S11 numbers** |
| 12 Oct 17:44 | 17:44 | Long comparison added: oldest scored node (`node_59`) against each of the 12,665 newest internal nodes (all children tips); 7 s. Tables re-shipped | `results/s11_h3n2_ha_tree/long_range_*.tsv.gz` | ok |

## S11 findings (22 Sep tree, H3 columns only)

* 20,589 internal branches with fewer than 10 nucleotide changes. Changes per branch: 1 → 15,898,
  2 → 3,016, 3 → 1,005, 4 → 427, 5 to 9 → 243. (A further 34 branches carry 10 or more and are left out.)
* Amino-acid-changing codons: 8,788 changed 1 position, 40 changed 2, 1 changed 3. Synonymous: 18,036, 1, 0.
* Of the 41 multi-position codons, 7 could have been reached with 1 change from the parent codon.
* About 13.5 same-codon pairs are expected if changes were spread evenly over the coding region; 42 are
  seen. Real clustering is expected (hot sites, shared selection), so this is a floor, not a null.
  Sites that recur most: 156 (12 times), 131 (5).
* **The artefact that was removed:** the tree's reference is not an H3 sequence. 18 of its 553 codons
  have no H3 residue. Left in, they gave 146 two-position codons and 261 stop-codon drops; left out,
  40 and 1. The old top "hot spots" were all in these columns.
* **Table row 2 (oldest H3 node to the newest internal nodes):** a typical newest internal node differs from
  the oldest node by 82 amino acids. 66 (80%) need 1 nucleotide change from the oldest node's codon, 14 (17%)
  need 2, 2 (3%) need 3. Row 1 (one branch at a time): 8,795 (99.6%) need 1, 33 (0.4%) need 2, 1 needs 3.
  So changes needing 2 or 3 nucleotides are almost absent on one branch but make up about a fifth of the
  long-range differences: they are built in steps. 1,043 distinct (site, new amino acid) changes are seen in
  all (812 need 1, 211 need 2, 20 need 3). Caveats: differences are counted per node, so shared history is
  counted once per node; the oldest node's codons are inferred; no branch-length cap is applied here.
* Not checkable here: whether the tree was built with `faToVcf -includeNoAltN`.

## S3 notes

* Mean probability of the true amino acid is the headline; top-1 hit rate is kept in the tables.
* ESM-2 at every position agrees with the old every-7th-residue top-1 rates to within a few points
  (haemagglutinin 63.8% against about 66%; neuraminidase 34.5% against about 37%).
* **Results.** Median across proteins of the mean probability on the true amino acid: ESM-2 14.6%,
  ESM-C 8.7%; guessing from the protein's own amino-acid mix scores about 6%. ESM-2 is more than 3 points
  above that floor on 21 of 32 proteins, ESM-C on 11. Above 30%: ESM-2 on PB1, PB2, HA and SARS-CoV-2
  nucleoprotein; ESM-C only on HIV-1 gp160 (82%, against 12% for ESM-2). The two models rarely agree on
  which proteins are easy.
* **Cross-check against the old every-7th-residue run:** top-1 rates correlate at 0.973 over the 64
  (model, protein) pairs; mean difference -1 point. The largest differences (up to 12 points) are short
  proteins, where every-7th had only 14 to 36 positions.
* **One sequence is a fragment with unknown residues:** influenza C HEF (P17005) has two X residues
  (positions 193-194 in UniProt). The fetch script dropped them, so it was scored with 639 of 641 residues.
  Recorded in `viral_panel_sequences.tsv` (`unknown_residues_removed`).
* Long proteins dominate run time (a 1,200-residue spike takes about 3.5 minutes per model).

## Re-running on a newer tree

If a `.pb` for the 2 Oct (or later) tree appears, repeat the two S11 rows above with that file and
compare against the 22 Sep tables. The reference record and the tip list come from the DB, so only the
`.pb` changes.
