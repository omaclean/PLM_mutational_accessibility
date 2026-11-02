#!/bin/bash
# =============================================================================
# Simulation arms for the symmetric-vs-directional tests (reviewer 2's simulation control).
#
#   Fine-tune ESM-C 600M on sequences evolved from ONE founder under ONE fitness landscape, changing only
#   the nucleotide spectrum:   H3N2 (Pauly et al. 2017)   and   JC69 (flat, no direction).
#   Existing arms (human germline, SARS-CoV-2) were fine-tuned earlier with identical settings.
#
#   Then score the FOUNDER itself with all five models (base, human, SC2, H3N2, JC69), so every model is
#   conditioned on the same sequence. The earlier scoring used a tip "representative" that differs from the
#   founder at 9-11% of sites while accessibility was defined from the founder.
#
# Run on gpu1:  ssh gpu1 'nohup bash <this file> > <log> 2>&1 &'
# Idempotent: finished stages are skipped, so a restart resumes. Threads are kept low (shared server).
# =============================================================================
set -uo pipefail

R=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
D=${SIM_DIR:-$R/evotuning/simulation}
ROOT=${OUT_ROOT:-$R/results/sim_v3_star90}
OUT=$ROOT/evotune_out
PYBIN=${PYBIN:-python}
mkdir -p "$OUT/run_logs" "$ROOT/score" "$ROOT/logs"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

say() { echo "[$(date -Is)] $*"; }

finetune() {            # name fasta port
  local name=$1 fasta=$2 port=$3 run="${1}_esmc600m"
  local ckpt="$OUT/$run/model/final_checkpoint/esmc_weights.pt"
  if [ -f "$ckpt" ]; then say "fine-tune $run: checkpoint exists, skipping"; return 0; fi
  say "fine-tune $run: start"
  cd "$R/evotuning"
  nice -n 10 ionice -c3 "$PYBIN/torchrun" --nnodes 1 --nproc-per-node 1 \
      --rdzv-backend c10d --rdzv-endpoint 127.0.0.1:"$port" evotune_gradient_MAGMA_ESMC.py \
      --input-fasta "$fasta" --output-dir "$OUT" --run-name "$run" --model-type esmc_600m \
      --num-epochs 10 --train-batch-size 8 --gradient-accumulation-steps 4 \
      --dataloader-num-workers 2 --tokenize-num-proc 4 --bf16 true --gradient-checkpointing true \
      --export-model-each-epoch false > "$OUT/run_logs/$run.log" 2>&1
  say "fine-tune $run: exit $?"
}

score() {               # tag guide [checkpoint-dir | RAW]
  local tag=$1 guide=$2 ckpt=$3 out="$ROOT/score/founder_$1"
  if ls "$out"/plm_cache/*profile.csv > /dev/null 2>&1; then say "score $tag: profile exists, skipping"; return 0; fi
  say "score $tag: start"
  cd "$R"
  local mode=(--checkpoint-dir "$ckpt")
  [ "$ckpt" = RAW ] && mode=(--raw-only)
  nice -n 10 bash slurm_mutation_acesiblity.sbatch.sh "${mode[@]}" --base-model esm-c600m \
      --guide-path "$guide" --output-dir "$out" --model-tag "founder_$tag" \
      --mutation-model HUMAN --no-biochem > "$ROOT/logs/score_founder_$tag.log" 2>&1
  say "score $tag: exit $?"
}

finetune v3_h3n2_star90 "$D/v3_h3n2_star90/v3_h3n2_star90.faa" 29531
finetune v3_jc69_star90 "$ROOT/v3_jc69_star90/v3_jc69_star90.faa" 29532

score base  "$ROOT/guides/guide_founder_base.csv"  RAW
score human "$ROOT/guides/guide_founder_human.csv" "$D/evotune_v3_out/v3_star90_esmc600m/model/final_checkpoint"
score sc2   "$ROOT/guides/guide_founder_sc2.csv"   "$D/evotune_sc2_out/v3_sc2_star90_esmc600m/model/final_checkpoint"
score h3n2  "$ROOT/guides/guide_founder_h3n2.csv"  "$OUT/v3_h3n2_star90_esmc600m/model/final_checkpoint"
score jc69  "$ROOT/guides/guide_founder_jc69.csv"  "$OUT/v3_jc69_star90_esmc600m/model/final_checkpoint"
say "queue finished"
