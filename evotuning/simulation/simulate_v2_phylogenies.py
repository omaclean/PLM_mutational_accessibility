#!/usr/bin/env python3
"""Simulate constrained coding sequences on v2 star and branching-ladder phylogenies.

v2 changes vs simulate_star_phylogenies.py
==========================================
*  Star-90%: same star topology, target identity 0.90 instead of 0.70.
*  Branching ladder: the trunk is capped at --ladder-max-trunk-steps (default
   100) steps of 1 AA substitution each.  At every --ladder-branch-interval
   trunk steps a burst of --ladder-branch-size independent dead-end tips is
   emitted (each evolved from the same trunk node, treated as independent
   leaves).  Together, trunk nodes + branch tips = exactly n-sequences.

   With the defaults (trunk=100, interval=10, branch_size=990):
     - 100 trunk nodes (steps 0..99)
     - 10 branching points (steps 10, 20, …, 100)  →  10 × 990 = 9900 branch tips
     - Total: 10 000
     - Adjacent trunk identity: 99.86 %  (1/700 AA change)
     - Max founder divergence: 100/700 ≈ 14.3 %  (identity ≥ 85.7 %)

The script is invoked once per topology; the --mode flag selects which
simulation to run.

Usage
-----
  # Star 90%
  python simulate_v2_phylogenies.py --mode star \
      --matrix <matrix.csv> --output-dir <dir> \
      --star-identity 0.90 --n-sequences 10000 --length-aa 700 \
      --seed 20260924 --processes 8

  # Branching ladder
  python simulate_v2_phylogenies.py --mode ladder_branching \
      --matrix <matrix.csv> --output-dir <dir> \
      --n-sequences 10000 --length-aa 700 \
      --ladder-step-aa-changes 1 \
      --ladder-branch-interval 10 \
      --ladder-max-trunk-steps 100 \
      --seed 20260924 --processes 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


BASES = "ACGT"
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_GROUPS = (
    set("AVLIM"), set("FYW"), set("STNQ"), set("KRH"), set("DE"),
    set("GP"), set("C"),
)

GENETIC_CODE = {
    "TTT":"F", "TTC":"F", "TTA":"L", "TTG":"L",
    "TCT":"S", "TCC":"S", "TCA":"S", "TCG":"S",
    "TAT":"Y", "TAC":"Y", "TAA":"*", "TAG":"*",
    "TGT":"C", "TGC":"C", "TGA":"*", "TGG":"W",
    "CTT":"L", "CTC":"L", "CTA":"L", "CTG":"L",
    "CCT":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAT":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGT":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "ATT":"I", "ATC":"I", "ATA":"I", "ATG":"M",
    "ACT":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAT":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGT":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GTT":"V", "GTC":"V", "GTA":"V", "GTG":"V",
    "GCT":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAT":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGT":"G", "GGC":"G", "GGA":"G", "GGG":"G",
}
CODONS_BY_AA = {aa: [] for aa in AA_ORDER}
for _codon, _aa in GENETIC_CODE.items():
    if _aa != "*":
        CODONS_BY_AA[_aa].append(_codon)

AA_FREQ = {
    "A":.083, "C":.014, "D":.054, "E":.067, "F":.038,
    "G":.071, "H":.023, "I":.056, "K":.058, "L":.096,
    "M":.024, "N":.041, "P":.047, "Q":.039, "R":.055,
    "S":.066, "T":.053, "V":.068, "W":.011, "Y":.029,
}


@dataclass(frozen=True)
class MutationModel:
    rates: dict[str, tuple[float, ...]]
    row_sums: dict[str, float]
    max_row_sum: float


@dataclass(frozen=True)
class EvolutionResult:
    dna: str
    protein: str
    accepted: int
    aa_substitutions: int
    attempted: int
    identity: float


def weighted_choice(items: Sequence[str], weights: Sequence[float], rng: random.Random) -> str:
    total = sum(weights)
    x = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if x < cumulative:
            return item
    return items[-1]


def read_mutation_matrix(path: Path) -> MutationModel:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[str, tuple[float, ...]] = {}
        for record in reader:
            source = record["from_base"].strip().upper()
            values = tuple(float(record[b]) for b in BASES)
            rows[source] = values
    sums = {b: sum(rows[b]) for b in BASES}
    return MutationModel(rows, sums, max(sums.values()))


def make_founder(length_aa: int, rng: random.Random) -> tuple[str, str]:
    amino_acids = list(AA_FREQ)
    weights = [AA_FREQ[a] for a in amino_acids]
    protein = "".join(weighted_choice(amino_acids, weights, rng) for _ in range(length_aa))
    dna = "".join(rng.choice(CODONS_BY_AA[aa]) for aa in protein)
    return dna, protein


def aa_group(aa: str) -> set[str]:
    return next(group for group in AA_GROUPS if aa in group)


def is_preference_landscape(landscape: Sequence[dict[str, float]]) -> bool:
    """Check if landscape holds preferences (all > 0, summing ~ 1 per site) vs penalties."""
    if not landscape:
        return False
    first = landscape[0]
    return all(v > 0 for v in first.values()) and abs(sum(first.values()) - 1.0) < 0.05


def make_preference_landscape(
    calibration_path: Path,
    beta: float,
    length_aa: int,
    seed: int,
) -> tuple[str, list[dict[str, float]]]:
    """Sample per-site preference vectors from DMS calibration targets.

    Per site:
      1. Draw conc ~ Gamma(shape_k, scale) mapped via 0.03 * (draw ** 3.2)
      2. Baseline tolerance vector m from residue residual means: softmax(res_means / 0.55)
      3. Draw pi ~ Dirichlet(max(conc * m, 1e-4)), floor 1e-4, renormalise
      4. Sample founder residue founder_aa ~ Categorical(pi)
    """
    with calibration_path.open() as f:
        calib = json.load(f)

    beta_key = str(int(beta)) if str(int(beta)) in calib["calibrations"] else str(beta)
    if beta_key not in calib["calibrations"]:
        beta_key = list(calib["calibrations"].keys())[0]

    cal_data = calib["calibrations"][beta_key]
    gamma_fit = cal_data["gamma_fit"]
    shape_k = float(gamma_fit["shape_k"])
    scale = float(gamma_fit["scale"])
    res_means = cal_data["residue_residual_means"]

    logits = np.array([res_means[a] for a in AA_ORDER], dtype=float) / 0.55
    m = np.exp(logits) / np.sum(np.exp(logits))

    np_rng = np.random.default_rng(seed)
    raw_gamma = np_rng.gamma(shape_k, scale, size=length_aa)
    concs = 0.03 * (raw_gamma ** 3.2)

    landscape = []
    founder_aas = []

    for i in range(length_aa):
        c = concs[i]
        alpha = np.maximum(c * m, 1e-4)
        pi = np_rng.dirichlet(alpha)
        pi = np.maximum(pi, 1e-4)
        pi = pi / pi.sum()

        pi_dict = {a: float(pi[j]) for j, a in enumerate(AA_ORDER)}
        landscape.append(pi_dict)

        f_idx = np_rng.choice(len(AA_ORDER), p=pi)
        founder_aas.append(AA_ORDER[f_idx])

    founder_protein = "".join(founder_aas)
    return founder_protein, landscape


def make_fitness_landscape(
    founder_protein: str, rng: random.Random, conserved_fraction: float
) -> list[dict[str, float]]:
    """(Deprecated) v2 static penalty landscape."""
    landscape = []
    for founder_aa in founder_protein:
        conserved = rng.random() < conserved_fraction
        strength = rng.uniform(3.5, 6.0) if conserved else rng.uniform(0.35, 1.6)
        similar = aa_group(founder_aa)
        scores = {}
        for aa in AA_ORDER:
            if aa == founder_aa:
                scores[aa] = 0.0
            else:
                multiplier = rng.uniform(0.65, 1.05) if aa in similar else rng.uniform(1.1, 1.8)
                scores[aa] = -strength * multiplier
        landscape.append(scores)
    return landscape


def translate(dna: Sequence[str] | str) -> str:
    return "".join(GENETIC_CODE["".join(dna[i:i+3])] for i in range(0, len(dna), 3))


def propose_alt(base: str, model: MutationModel, rng: random.Random) -> str:
    return weighted_choice(BASES, model.rates[base], rng)


def accept_substitution(
    pi_site: dict[str, float] | Sequence[float],
    old_aa: str,
    new_aa: str,
    beta: float,
    rng: random.Random,
    fix_max: float = 20.0,
) -> bool:
    """Acceptance step using Halpern & Bruno (1998) fixation probability.

    S = beta * (ln pi[new_aa] - ln pi[old_aa])
    fix = 1.0 if abs(S) < 1e-8 else S / (1.0 - math.exp(-S))
    accept with probability min(1.0, fix / fix_max)
    """
    if old_aa == new_aa:
        return True
    p_old = pi_site[old_aa] if isinstance(pi_site, dict) else pi_site[AA_ORDER.index(old_aa)]
    p_new = pi_site[new_aa] if isinstance(pi_site, dict) else pi_site[AA_ORDER.index(new_aa)]

    S = beta * (math.log(p_new) - math.log(p_old))
    if abs(S) < 1e-8:
        fix = 1.0
    elif S > 50:
        fix = S
    elif S < -50:
        fix = -S * math.exp(S)
    else:
        fix = S / (1.0 - math.exp(-S))

    prob = fix / fix_max
    if prob > 1.0:
        prob = 1.0
    assert 0.0 <= prob <= 1.0, f"acceptance prob {prob} outside [0, 1]"
    return rng.random() < prob


def evolve_branch_length(
    start_dna: str,
    founder_protein: str,
    landscape: list[dict[str, float]],
    model: MutationModel,
    branch_length: float,
    beta: float,
    rng: random.Random,
    fix_max: float = 20.0,
) -> EvolutionResult:
    """Evolve for a fixed expected number of mutation proposals per site.

    branch_length is the expected proposals per protein site (e.g. 55 for star-90%,
    220 for star-70%). Values <= 1.0 are scaled by 550 to interpret as substitutions per site.
    Realised identity and substitutions are reported from the resulting sequence.
    """
    dna = list(start_dna)
    protein = list(translate(start_dna))
    n_sites = len(founder_protein)
    proposals_per_site = branch_length * 550.0 if branch_length <= 1.0 else branch_length
    num_proposals = int(round(proposals_per_site * n_sites))
    attempted = accepted = aa_substitutions = 0

    for _ in range(num_proposals):
        attempted += 1
        pos = rng.randrange(len(dna))
        old_base = dna[pos]
        if rng.random() > model.row_sums[old_base] / model.max_row_sum:
            continue
        new_base = propose_alt(old_base, model, rng)
        codon_site = pos // 3
        codon_start = 3 * codon_site
        old_aa = protein[codon_site]
        old_codon = "".join(dna[codon_start:codon_start + 3])
        offset = pos - codon_start
        new_codon = old_codon[:offset] + new_base + old_codon[offset + 1:]
        new_aa = GENETIC_CODE[new_codon]
        if new_aa == "*":
            continue
        if not accept_substitution(landscape[codon_site], old_aa, new_aa, beta, rng, fix_max=fix_max):
            continue
        dna[pos] = new_base
        protein[codon_site] = new_aa
        accepted += 1
        aa_substitutions += int(new_aa != old_aa)

    matches = sum(a == b for a, b in zip(protein, founder_protein))
    return EvolutionResult(
        "".join(dna), "".join(protein), accepted, aa_substitutions, attempted,
        matches / n_sites,
    )


def evolve_to_identity(
    start_dna: str,
    founder_protein: str,
    landscape: list[dict[str, float]],
    model: MutationModel,
    target_identity: float,
    temperature: float,
    rng: random.Random,
    max_attempts_per_site: int,
    beta: float = 2.0,
    fix_max: float = 20.0,
) -> EvolutionResult:
    """(Deprecated in v3) Evolve until sequence reaches target identity to founder."""
    is_pref = is_preference_landscape(landscape)
    dna = list(start_dna)
    protein = list(translate(start_dna))
    matches = sum(a == b for a, b in zip(protein, founder_protein))
    max_matches = math.floor(target_identity * len(founder_protein) + 1e-12)
    attempted = accepted = aa_substitutions = 0
    attempt_limit = max_attempts_per_site * len(founder_protein)

    while matches > max_matches and attempted < attempt_limit:
        attempted += 1
        pos = rng.randrange(len(dna))
        old_base = dna[pos]
        if rng.random() > model.row_sums[old_base] / model.max_row_sum:
            continue
        new_base = propose_alt(old_base, model, rng)
        codon_site = pos // 3
        codon_start = 3 * codon_site
        old_aa = protein[codon_site]
        old_codon = "".join(dna[codon_start:codon_start + 3])
        offset = pos - codon_start
        new_codon = old_codon[:offset] + new_base + old_codon[offset + 1:]
        new_aa = GENETIC_CODE[new_codon]
        if new_aa == "*":
            continue
        if is_pref:
            if not accept_substitution(landscape[codon_site], old_aa, new_aa, beta, rng, fix_max=fix_max):
                continue
        else:
            delta = landscape[codon_site][new_aa] - landscape[codon_site][old_aa]
            if delta < 0 and rng.random() >= math.exp(delta / temperature):
                continue
        founder_aa = founder_protein[codon_site]
        was_match = old_aa == founder_aa
        is_match = new_aa == founder_aa
        matches += int(is_match) - int(was_match)
        dna[pos] = new_base
        protein[codon_site] = new_aa
        accepted += 1
        aa_substitutions += int(new_aa != old_aa)

    if matches > max_matches:
        reached = matches / len(founder_protein)
        raise RuntimeError(
            f"failed to reach identity {target_identity:.4f}; stopped at {reached:.4f} "
            f"after {attempted} proposals."
        )
    return EvolutionResult(
        "".join(dna), "".join(protein), accepted, aa_substitutions, attempted,
        matches / len(founder_protein),
    )


def evolve_aa_steps(
    start_dna: str,
    founder_protein: str,
    landscape: list[dict[str, float]],
    model: MutationModel,
    aa_steps: int,
    temperature: float,
    rng: random.Random,
    max_attempts_per_site: int,
    beta: float = 2.0,
    fix_max: float = 20.0,
) -> EvolutionResult:
    """Evolve until exactly aa_steps accepted nonsynonymous events occur."""
    is_pref = is_preference_landscape(landscape)
    dna = list(start_dna)
    protein = list(translate(start_dna))
    attempted = accepted = nonsyn = 0
    attempt_limit = max_attempts_per_site * len(founder_protein)
    while nonsyn < aa_steps and attempted < attempt_limit:
        attempted += 1
        pos = rng.randrange(len(dna))
        old_base = dna[pos]
        if rng.random() > model.row_sums[old_base] / model.max_row_sum:
            continue
        new_base = propose_alt(old_base, model, rng)
        site = pos // 3
        start = 3 * site
        old_aa = protein[site]
        old_codon = "".join(dna[start:start + 3])
        offset = pos - start
        new_codon = old_codon[:offset] + new_base + old_codon[offset + 1:]
        new_aa = GENETIC_CODE[new_codon]
        if new_aa == "*":
            continue
        if is_pref:
            if not accept_substitution(landscape[site], old_aa, new_aa, beta, rng, fix_max=fix_max):
                continue
        else:
            delta = landscape[site][new_aa] - landscape[site][old_aa]
            if delta < 0 and rng.random() >= math.exp(delta / temperature):
                continue
        dna[pos] = new_base
        protein[site] = new_aa
        accepted += 1
        nonsyn += int(new_aa != old_aa)
    if nonsyn < aa_steps:
        raise RuntimeError(
            f"failed to accept {aa_steps} amino-acid changes in {attempted} proposals."
        )
    matches = sum(a == b for a, b in zip(protein, founder_protein))
    return EvolutionResult(
        "".join(dna), "".join(protein), accepted, nonsyn, attempted,
        matches / len(founder_protein),
    )


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = ":".join(map(str, (master_seed,) + parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def wrap_fasta(name: str, sequence: str, width: int = 80) -> str:
    lines = [f">{name}"]
    lines.extend(sequence[i:i+width] for i in range(0, len(sequence), width))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Worker state (for multiprocessing star tips)
# ---------------------------------------------------------------------------
_WORKER_STATE = None


def _init_worker(
    founder_protein: str,
    landscape: list[dict[str, float]],
    model: MutationModel,
    temperature: float,
    max_attempts: int,
    prefix: str,
    beta: float,
    fix_max: float,
    branch_length: float | None,
):
    global _WORKER_STATE
    _WORKER_STATE = (
        founder_protein, landscape, model, temperature, max_attempts, prefix,
        beta, fix_max, branch_length,
    )


def _evolve_star_tip(task):
    tip_index, founder_dna, target, seed = task
    (founder_protein, landscape, model, temperature, max_attempts, prefix,
     beta, fix_max, branch_length) = _WORKER_STATE
    rng = random.Random(seed)
    if branch_length is not None:
        result = evolve_branch_length(
            founder_dna, founder_protein, landscape, model, branch_length,
            beta, rng, fix_max=fix_max,
        )
    else:
        result = evolve_to_identity(
            founder_dna, founder_protein, landscape, model, target,
            temperature, rng, max_attempts, beta=beta, fix_max=fix_max,
        )
    name = f"{prefix}_s{tip_index:05d}"
    bl = result.aa_substitutions / len(result.protein)
    return name, result, bl


# ---------------------------------------------------------------------------
# Simulate star
# ---------------------------------------------------------------------------
def simulate_star(
    n_sequences: int,
    target_identity: float | None,
    founder_dna: str,
    founder_protein: str,
    landscape: list[dict[str, float]],
    model: MutationModel,
    temperature: float,
    max_attempts: int,
    seed: int,
    processes: int,
    output_dir: Path,
    dataset_name: str = "star",
    beta: float = 2.0,
    fix_max: float = 20.0,
    branch_length: float | None = None,
) -> dict:
    tasks = [
        (i, founder_dna, target_identity, stable_seed(seed, dataset_name, "tip", i))
        for i in range(1, n_sequences + 1)
    ]
    initargs = (
        founder_protein, landscape, model, temperature, max_attempts, dataset_name,
        beta, fix_max, branch_length,
    )
    if processes == 1:
        _init_worker(*initargs)
        results_iter = map(_evolve_star_tip, tasks)
        pool = None
    else:
        pool = mp.Pool(processes=processes, initializer=_init_worker, initargs=initargs)
        results_iter = pool.imap(_evolve_star_tip, tasks, chunksize=max(1, n_sequences // (processes * 20)))

    fasta_path = output_dir / f"{dataset_name}.faa"
    cds_path = output_dir / f"{dataset_name}.fna"
    metadata_path = output_dir / f"{dataset_name}.metadata.tsv"
    newick_tips = []
    identities = []
    accepted_counts = []
    repr_protein = None
    repr_dna = None
    repr_name = None
    try:
        with fasta_path.open("w") as faa, cds_path.open("w") as fna, metadata_path.open("w") as meta:
            meta.write("sequence_id\taa_identity_to_founder\taccepted_nt_substitutions\taccepted_aa_substitutions\tproposals\n")
            for i, (name, result, branch_length_tip) in enumerate(results_iter, 1):
                faa.write(wrap_fasta(name, result.protein))
                fna.write(wrap_fasta(name, result.dna))
                meta.write(
                    f"{name}\t{result.identity:.6f}\t{result.accepted}\t"
                    f"{result.aa_substitutions}\t{result.attempted}\n"
                )
                newick_tips.append(f"{name}:{branch_length_tip:.8f}")
                identities.append(result.identity)
                accepted_counts.append(result.accepted)
                # Use middle sequence as representative
                if i == n_sequences // 2:
                    repr_protein = result.protein
                    repr_dna = result.dna
                    repr_name = name
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    tree_path = output_dir / f"{dataset_name}.nwk"
    tree_path.write_text(f"({','.join(newick_tips)})founder;\n")

    # Write representative
    repr_fna_path = output_dir / f"{dataset_name}_representative.fna"
    if repr_name and repr_dna:
        repr_fna_path.write_text(wrap_fasta(repr_name, repr_dna))

    return {
        "dataset": dataset_name,
        "mode": "star",
        "sequences": n_sequences,
        "star_branch_length": branch_length,
        "target_max_aa_identity_to_founder": target_identity,
        "observed_identity_min": min(identities),
        "observed_identity_mean": sum(identities) / len(identities),
        "observed_identity_max": max(identities),
        "mean_accepted_substitutions_root_to_tip": sum(accepted_counts) / len(accepted_counts),
        "representative_sequence": repr_name,
        "files": [str(fasta_path), str(cds_path), str(metadata_path), str(tree_path), str(repr_fna_path)],
    }


simulate_star90 = simulate_star  # backward-compatible alias


# ---------------------------------------------------------------------------
# Simulate branching ladder
# ---------------------------------------------------------------------------
def simulate_ladder_branching(
    n_sequences: int,
    aa_steps: int,
    max_trunk_steps: int,
    branch_interval: int,
    founder_dna: str,
    founder_protein: str,
    landscape: list[dict[str, float]],
    model: MutationModel,
    temperature: float,
    max_attempts: int,
    seed: int,
    output_dir: Path,
    dataset_name: str = "ladder_branching",
    beta: float = 2.0,
    fix_max: float = 20.0,
) -> dict:
    """Branching ladder with dead-end side branches.

    Topology
    --------
    trunk_step_0 -> trunk_step_1 -> ... -> trunk_step_{max_trunk_steps}
         |                 |
     branch_0_1..N     branch_1_1..N
         (dead ends)       (dead ends)

    Every `branch_interval` trunk steps, `branch_size` independent dead-end
    tips are emitted from the current trunk node (each evolved from the same
    parent but with independent seeds => independent tip sequences).

    Parameters
    ----------
    n_sequences : total sequences to emit (trunk nodes + branch tips)
    max_trunk_steps : number of trunk nodes to emit (≥ 1)
    branch_interval : emit a branch burst every this many trunk steps
    branch_size : derived automatically so trunk + total_branches = n_sequences

    branch_size = (n_sequences - max_trunk_steps) // n_branch_points
    where n_branch_points = max_trunk_steps // branch_interval
    """
    n_branch_points = max_trunk_steps // branch_interval
    if n_branch_points == 0:
        raise ValueError("branch_interval larger than max_trunk_steps — no branching would occur")
    branch_size = (n_sequences - max_trunk_steps) // n_branch_points
    if branch_size < 1:
        raise ValueError(
            f"Cannot fit {n_sequences} sequences with {max_trunk_steps} trunk steps "
            f"and {n_branch_points} branch points (branch_size would be {branch_size})"
        )
    # Adjust to hit exactly n_sequences (distribute remainder to first branches)
    remainder = n_sequences - max_trunk_steps - n_branch_points * branch_size

    print(
        f"  Branching ladder: {max_trunk_steps} trunk steps, branch every {branch_interval} steps "
        f"({n_branch_points} branch points), ~{branch_size} tips/branch, total={n_sequences}",
        file=sys.stderr, flush=True,
    )

    fasta_path = output_dir / f"{dataset_name}.faa"
    cds_path = output_dir / f"{dataset_name}.fna"
    metadata_path = output_dir / f"{dataset_name}.metadata.tsv"

    all_names = []
    newick_trunk = []   # list of (name, parent_name_or_founder, branch_length)
    identities = []
    branch_lengths_by_trunk_step = []  # one per trunk step
    repr_name = repr_dna = None

    current_dna = founder_dna
    total_aa_from_founder = 0
    seq_counter = 0  # total sequences emitted

    with fasta_path.open("w") as faa, cds_path.open("w") as fna, metadata_path.open("w") as meta:
        meta.write(
            "sequence_id\ttype\ttrunk_step\taa_identity_to_founder\t"
            "branch_aa_substitutions\tcumulative_trunk_aa_substitutions\n"
        )

        for trunk_step in range(1, max_trunk_steps + 1):
            # Advance trunk by one aa step
            trunk_result = evolve_aa_steps(
                current_dna, founder_protein, landscape, model, aa_steps,
                temperature, random.Random(stable_seed(seed, "trunk", trunk_step)),
                max_attempts, beta=beta, fix_max=fix_max,
            )
            trunk_name = f"{dataset_name}_trunk_{trunk_step:04d}"
            total_aa_from_founder += trunk_result.aa_substitutions
            bl = trunk_result.aa_substitutions / len(founder_protein)

            faa.write(wrap_fasta(trunk_name, trunk_result.protein))
            fna.write(wrap_fasta(trunk_name, trunk_result.dna))
            meta.write(
                f"{trunk_name}\ttrunk\t{trunk_step}\t{trunk_result.identity:.6f}\t"
                f"{trunk_result.aa_substitutions}\t{total_aa_from_founder}\n"
            )
            all_names.append(trunk_name)
            identities.append(trunk_result.identity)
            branch_lengths_by_trunk_step.append(bl)
            seq_counter += 1

            current_dna = trunk_result.dna
            if repr_name is None and trunk_step == max_trunk_steps // 2:
                repr_name = trunk_name
                repr_dna = trunk_result.dna

            # Emit dead-end branch tips at every branch_interval
            if trunk_step % branch_interval == 0:
                branch_point_index = trunk_step // branch_interval  # 1-indexed
                # Add remainder to the first branch point to hit exactly n_sequences
                this_branch_size = branch_size + (1 if branch_point_index <= remainder else 0)
                for b in range(1, this_branch_size + 1):
                    branch_result = evolve_aa_steps(
                        current_dna, founder_protein, landscape, model, aa_steps,
                        temperature,
                        random.Random(stable_seed(seed, "branch", trunk_step, b)),
                        max_attempts, beta=beta, fix_max=fix_max,
                    )
                    branch_name = f"{dataset_name}_branch_t{trunk_step:04d}_b{b:04d}"
                    branch_bl = branch_result.aa_substitutions / len(founder_protein)
                    faa.write(wrap_fasta(branch_name, branch_result.protein))
                    fna.write(wrap_fasta(branch_name, branch_result.dna))
                    meta.write(
                        f"{branch_name}\tbranch\t{trunk_step}\t{branch_result.identity:.6f}\t"
                        f"{branch_result.aa_substitutions}\t{total_aa_from_founder}\n"
                    )
                    all_names.append(branch_name)
                    identities.append(branch_result.identity)
                    seq_counter += 1

    # Write Newick tree (comb topology: trunk is the spine, branches are leaves)
    tree_path = output_dir / f"{dataset_name}.nwk"
    # Build from innermost (founder) outward
    # Each trunk step is an internal node; branch tips hang off it
    # Format: (branch1:bl, branch2:bl, ..., nextTrunkNode:bl)trunk_name:bl_to_prev
    with tree_path.open("w") as tree:
        # Collect branch tips per trunk step
        # Re-read metadata for convenience (small enough)
        import csv as _csv
        tips_per_trunk = {t: [] for t in range(1, max_trunk_steps + 1)}
        trunk_bl_map = {}
        with metadata_path.open() as _mf:
            reader = _csv.DictReader(_mf, delimiter="\t")
            for row in reader:
                ts = int(row["trunk_step"])
                bl_val = float(row["branch_aa_substitutions"]) / len(founder_protein)
                if row["type"] == "branch":
                    tips_per_trunk[ts].append((row["sequence_id"], bl_val))
                else:
                    trunk_bl_map[ts] = bl_val

        # Build Newick bottom-up (outermost trunk step first in the string,
        # founder is root)
        def subtree(step: int) -> str:
            branch_strs = [f"{name}:{bl:.8f}" for name, bl in tips_per_trunk.get(step, [])]
            if step == max_trunk_steps:
                # Tip of trunk — no deeper trunk node
                if branch_strs:
                    inner = ",".join(branch_strs)
                    return f"({inner}){dataset_name}_trunk_{step:04d}:{trunk_bl_map[step]:.8f}"
                else:
                    return f"{dataset_name}_trunk_{step:04d}:{trunk_bl_map[step]:.8f}"
            else:
                children = branch_strs + [subtree(step + 1)]
                inner = ",".join(children)
                return f"({inner}){dataset_name}_trunk_{step:04d}:{trunk_bl_map[step]:.8f}"

        # Newick recursion may hit Python stack limits for 100 levels — fine
        import sys as _sys
        old_limit = _sys.getrecursionlimit()
        _sys.setrecursionlimit(max(old_limit, max_trunk_steps + 200))
        tree.write(f"({subtree(1)})founder;\n")
        _sys.setrecursionlimit(old_limit)

    # Write representative sequence
    repr_fna_path = output_dir / f"{dataset_name}_representative.fna"
    if repr_name and repr_dna:
        repr_fna_path.write_text(wrap_fasta(repr_name, repr_dna))
    else:
        # Fallback: use mid-trunk node
        repr_name = f"{dataset_name}_trunk_{max_trunk_steps // 2:04d}"
        repr_fna_path.write_text("# representative not saved (use metadata to identify)\n")

    total_emitted = seq_counter
    trunk_identities = [identities[i] for i, n in enumerate(all_names) if "trunk" in n]
    branch_identities = [identities[i] for i, n in enumerate(all_names) if "branch" in n]

    return {
        "dataset": dataset_name,
        "mode": "ladder_branching",
        "sequences": total_emitted,
        "max_trunk_steps": max_trunk_steps,
        "branch_interval": branch_interval,
        "branch_size_per_point": branch_size,
        "n_branch_points": n_branch_points,
        "aa_steps_per_trunk_node": aa_steps,
        "adjacent_trunk_aa_identity": 1.0 - aa_steps / len(founder_protein),
        "trunk_identity_to_founder_min": min(trunk_identities) if trunk_identities else None,
        "trunk_identity_to_founder_max": max(trunk_identities) if trunk_identities else None,
        "overall_identity_min": min(identities),
        "overall_identity_mean": sum(identities) / len(identities),
        "overall_identity_max": max(identities),
        "representative_sequence": repr_name,
        "files": [str(fasta_path), str(cds_path), str(metadata_path), str(tree_path), str(repr_fna_path)],
    }


# ---------------------------------------------------------------------------
# Simulate linear serial ladder
# ---------------------------------------------------------------------------
def simulate_ladder(
    n_sequences: int,
    aa_steps: int,
    founder_dna: str,
    founder_protein: str,
    landscape: list[dict[str, float]],
    model: MutationModel,
    temperature: float,
    max_attempts: int,
    seed: int,
    output_dir: Path,
    dataset_name: str = "ladder",
    beta: float = 2.0,
    fix_max: float = 20.0,
) -> dict:
    """Linear serial ladder: each sequence evolves aa_steps from the previous one."""
    fasta_path = output_dir / f"{dataset_name}.faa"
    cds_path = output_dir / f"{dataset_name}.fna"
    metadata_path = output_dir / f"{dataset_name}.metadata.tsv"
    names, branch_lengths, identities = [], [], []
    current_dna = founder_dna
    total_nt = total_aa = 0
    repr_name = repr_dna = None
    with fasta_path.open("w") as faa, cds_path.open("w") as fna, metadata_path.open("w") as meta:
        meta.write(
            "sequence_id\tstep\taa_identity_to_previous\taa_identity_to_founder\t"
            "branch_nt_substitutions\tbranch_aa_substitutions\tcumulative_nt_substitutions\t"
            "cumulative_aa_substitutions\tproposals\n"
        )
        previous_protein = founder_protein
        for i in range(1, n_sequences + 1):
            if i == 1:
                result = EvolutionResult(
                    founder_dna, founder_protein, 0, 0, 0, 1.0
                )
            else:
                result = evolve_aa_steps(
                    current_dna, founder_protein, landscape, model, aa_steps,
                    temperature, random.Random(stable_seed(seed, dataset_name, i)),
                    max_attempts, beta=beta, fix_max=fix_max,
                )
            name = f"{dataset_name}_s{i:05d}"
            adjacent_identity = sum(
                a == b for a, b in zip(previous_protein, result.protein)
            ) / len(founder_protein)
            total_nt += result.accepted
            total_aa += result.aa_substitutions
            faa.write(wrap_fasta(name, result.protein))
            fna.write(wrap_fasta(name, result.dna))
            meta.write(
                f"{name}\t{i}\t{adjacent_identity:.6f}\t{result.identity:.6f}\t"
                f"{result.accepted}\t{result.aa_substitutions}\t{total_nt}\t{total_aa}\t"
                f"{result.attempted}\n"
            )
            names.append(name)
            branch_lengths.append(result.aa_substitutions / len(founder_protein))
            identities.append(result.identity)
            if i == 1:
                repr_name = name
                repr_dna = result.dna
            current_dna = result.dna
            previous_protein = result.protein

    tree_path = output_dir / f"{dataset_name}.nwk"
    with tree_path.open("w") as tree:
        if n_sequences == 1:
            tree.write(f"({names[0]}:0)founder;\n")
        else:
            tree.write(f"({names[0]}:0,")
            for index in range(1, n_sequences - 1):
                tree.write(f"({names[index]}:0,")
            tree.write(f"{names[-1]}:{branch_lengths[-1]:.8f}")
            for index in range(n_sequences - 2, 0, -1):
                tree.write(
                    f"){dataset_name}_trunk_{index + 1:05d}:{branch_lengths[index]:.8f}"
                )
            tree.write(")founder;\n")

    repr_fna_path = output_dir / f"{dataset_name}_representative.fna"
    if repr_name and repr_dna:
        repr_fna_path.write_text(wrap_fasta(repr_name, repr_dna))

    return {
        "dataset": dataset_name,
        "mode": "ladder",
        "sequences": n_sequences,
        "accepted_aa_substitutions_per_step": aa_steps,
        "adjacent_aa_identity": 1.0 - aa_steps / len(founder_protein),
        "observed_identity_to_founder_min": min(identities),
        "observed_identity_to_founder_mean": sum(identities) / len(identities),
        "observed_identity_to_founder_max": max(identities),
        "representative_sequence": repr_name,
        "files": [str(fasta_path), str(cds_path), str(metadata_path), str(tree_path), str(repr_fna_path)],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["star", "ladder_branching", "ladder"], required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Prefix/name for output files (default: derived from output-dir or mode)")
    parser.add_argument("--n-sequences", type=int, default=10_000)
    parser.add_argument("--length-aa", type=int, default=700)
    parser.add_argument("--star-identity", type=float, default=0.90)
    parser.add_argument("--star-branch-length", type=float, default=None,
                        help="Expected proposals per site for star mode (e.g. 55 for star90, 220 for star70)")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Path to DMS calibration targets JSON (enables v3 preference landscape)")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="Selection stringency exponent beta (default: 2.0)")
    parser.add_argument("--fix-max", type=float, default=20.0,
                        help="Maximum fixation probability bound for rejection sampling (default: 20.0)")
    parser.add_argument("--ladder-step-aa-changes", type=int, default=1)
    parser.add_argument("--ladder-branch-interval", type=int, default=10)
    parser.add_argument("--ladder-max-trunk-steps", type=int, default=100)
    parser.add_argument("--conserved-fraction", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-attempts-per-site", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or args.output_dir.name

    model = read_mutation_matrix(args.matrix)
    calib_md5 = None
    if args.calibration is not None and args.calibration.exists():
        calib_md5 = hashlib.md5(args.calibration.read_bytes()).hexdigest()
        founder_protein, landscape = make_preference_landscape(
            args.calibration, args.beta, args.length_aa, args.seed
        )
        founder_rng = random.Random(stable_seed(args.seed, "founder_dna"))
        founder_dna = "".join(founder_rng.choice(CODONS_BY_AA[aa]) for aa in founder_protein)
        with (args.output_dir / "fitness_landscape_v2.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["site", "founder_aa"] + list(AA_ORDER))
            for site, (aa, prefs) in enumerate(zip(founder_protein, landscape), 1):
                writer.writerow([site, aa] + [f"{prefs[x]:.6f}" for x in AA_ORDER])
    else:
        founder_rng = random.Random(stable_seed(args.seed, "founder"))
        founder_dna, founder_protein = make_founder(args.length_aa, founder_rng)
        landscape = make_fitness_landscape(
            founder_protein, random.Random(stable_seed(args.seed, "fitness")),
            args.conserved_fraction,
        )
        with (args.output_dir / "fitness_landscape.tsv").open("w") as handle:
            handle.write("site\tfounder_aa\t" + "\t".join(AA_ORDER) + "\n")
            for site, (aa, scores) in enumerate(zip(founder_protein, landscape), 1):
                handle.write(
                    f"{site}\t{aa}\t" + "\t".join(f"{scores[x]:.6f}" for x in AA_ORDER) + "\n"
                )

    (args.output_dir / "founder.faa").write_text(wrap_fasta("founder", founder_protein))
    (args.output_dir / "founder.fna").write_text(wrap_fasta("founder", founder_dna))

    if args.mode == "star":
        target_str = (
            f"branch length {args.star_branch_length}"
            if args.star_branch_length is not None
            else f"target identity >= {args.star_identity:.3f}"
        )
        print(
            f"Simulating star: {args.n_sequences} independent tips, {target_str} [{dataset_name}]",
            file=sys.stderr, flush=True,
        )
        summary = simulate_star(
            args.n_sequences, args.star_identity, founder_dna, founder_protein,
            landscape, model, args.temperature, args.max_attempts_per_site,
            args.seed, args.processes, args.output_dir, dataset_name=dataset_name,
            beta=args.beta, fix_max=args.fix_max, branch_length=args.star_branch_length,
        )
    elif args.mode == "ladder_branching":
        print(
            f"Simulating ladder_branching: {args.n_sequences} sequences, "
            f"{args.ladder_step_aa_changes} AA/step, trunk={args.ladder_max_trunk_steps} "
            f"steps, branch every {args.ladder_branch_interval} steps [{dataset_name}]",
            file=sys.stderr, flush=True,
        )
        summary = simulate_ladder_branching(
            args.n_sequences, args.ladder_step_aa_changes,
            args.ladder_max_trunk_steps, args.ladder_branch_interval,
            founder_dna, founder_protein, landscape, model,
            args.temperature, args.max_attempts_per_site,
            args.seed, args.output_dir, dataset_name=dataset_name,
            beta=args.beta, fix_max=args.fix_max,
        )
    else:
        print(
            f"Simulating linear ladder: {args.n_sequences} sequential steps, "
            f"{args.ladder_step_aa_changes} AA/step [{dataset_name}]",
            file=sys.stderr, flush=True,
        )
        summary = simulate_ladder(
            args.n_sequences, args.ladder_step_aa_changes,
            founder_dna, founder_protein, landscape, model,
            args.temperature, args.max_attempts_per_site,
            args.seed, args.output_dir, dataset_name=dataset_name,
            beta=args.beta, fix_max=args.fix_max,
        )

    run_info = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "calibration_md5": calib_md5,
        "mutation_matrix": {
            source: {target: model.rates[source][i] for i, target in enumerate(BASES)}
            for source in BASES
        },
        "dataset": summary,
        "branch_length_definition": "accepted amino-acid substitutions / protein sites",
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
