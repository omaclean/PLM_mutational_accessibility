"""Synthetic replacement tables with a known answer, for the codon-learning and S6 tests.

Nothing here is typed from memory: codons, the genetic code and step counts come from
``mutacc.accessibility`` (Biopython). A table is built from sense codons drawn at random, with a
planted preference: the model's probability of a replacement is multiplied by ``exp(delta)``
when the reference codon reaches the new residue in one nucleotide change. ``delta = 0`` is the
null; larger ``delta`` raises the true within-replacement AUROC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mutacc import accessibility as acc

SENSE = [c for c in acc.CODONS if acc.GENETIC_CODE[c] != "*"]
MODEL = "m"


def write_root(path: Path, codons: list[str]) -> Path:
    path.write_text(">root\n" + "".join(codons) + "\n")
    return path


def make_scores(tmp_path: Path, *, n_pos: int = 2500, n_lin: int = 2, delta: float = 0.0, seed: int = 1,
                models=(MODEL,), q: np.ndarray | None = None, lineage_noise: float = 0.0):
    """Return (scores frame, roots dict). Every lineage carries the same codons (near-duplicates)."""
    rng = np.random.default_rng(seed)
    codons = [SENSE[i] for i in rng.integers(0, len(SENSE), n_pos)]
    lins = [f"L{i}" for i in range(n_lin)]
    roots = {l: write_root(tmp_path / f"{l}.fa", codons) for l in lins}
    steps = np.array([[acc.nucleotide_steps(c, a) for a in acc.AA20] for c in SENSE])
    sidx = {c: i for i, c in enumerate(SENSE)}
    rows = []
    for lin in lins:
        for p, c in enumerate(codons, start=1):
            ref = acc.GENETIC_CODE[c]
            for a_i, a in enumerate(acc.AA20):
                if a == ref:
                    continue
                rows.append((lin, p, ref, a, c, int(steps[sidx[c], a_i])))
    f = pd.DataFrame(rows, columns=["lineage", "position", "ref_aa", "aa", "codon", "nt_steps"])
    out = []
    for k, m in enumerate(models):
        g = np.random.default_rng(seed + 100 + k)
        noise = g.normal(size=len(f)) + lineage_noise * g.normal(size=len(f))
        logp = noise + delta * (f["nt_steps"].to_numpy() == 1) * (1 if k == len(models) - 1 else 0)
        sub = f.copy()
        sub["model"] = m
        sub["plm_prob"] = np.exp(logp - logp.max()) * 1e-3
        out.append(sub)
    frame = pd.concat(out, ignore_index=True)
    if q is not None:
        frame["mut_prob"] = acc.accessibility(frame["codon"].tolist(), frame["aa"].tolist(), q)
    return frame.drop(columns=["codon"]) if q is None else frame.drop(columns=["codon"]), roots
