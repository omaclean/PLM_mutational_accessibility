#!/bin/bash
#SBATCH --job-name=evotune-esmc-full_magma
#SBATCH --output=SC2_out/slurm-esmc-full-%j.out
#SBATCH --error=SC2_out/slurm-esmc-full-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=workq
#SBATCH --nodes=1               # ← change this AND NNODES below together
#SBATCH --gres=gpu:4            # GPUs per node
#SBATCH --cpus-per-task=288     # 4 Grace CPUs per node (72 cores × 4 GH200 chips)
#SBATCH --mem=400G              # Request a significant portion of the 480 GB LPDDR5X
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail

# ── Topology ────────────────────────────────────────────────────────────────
NNODES=${SLURM_NNODES:-1}
NPROC_PER_NODE=4   # GPUs per node; must match --gres=gpu:N above
RDZV_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
RDZV_TIMEOUT=${RDZV_TIMEOUT:-1800}

MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)
export MASTER_ADDR MASTER_PORT=${RDZV_PORT}

# ── Args (see --help) ────────────────────────────────────────────────────────
MODEL_TYPE=esmc_600m
SPECIES=flu
DATASET_MODE=clustered           # clustered | alldata
EXPORT_MODEL_EACH_EPOCH=false
ALLOW_CLUSTERING=true
GRAD_ACCUM_OVERRIDE=
NUM_EPOCHS_OVERRIDE=
RUN_NAME_OVERRIDE=
SPLIT_STRATEGY_OVERRIDE=

usage() {
    cat <<'EOF'
Usage: sbatch [--nodes N --time HH:MM:SS] evotune_esmc_full_magma.sbatch [--flags]
  --model-type STR       esmc_300m | esmc_600m | esmc_6b       (default esmc_600m)
  --species NAME         flu | sc2 | og | allvirus | allRNAvirus | allDNAvirus | alltest |
                         bigvirus90_10k | alltest100k | bigvirus90_100k |
                         v812all | v812RNA | v812DNA   (v20260812 clean set)
  --dataset-mode MODE    clustered | alldata                   (default clustered)
                         alldata = train on every sequence (full-finetune; no
                         clustering, no reference-divergent split)
  --export-each-epoch B  true | false                          (default false)
  --allow-clustering B   true | false  (clustered mode only)   (default true)
  --grad-accum N         override the dataset-size default
  --num-epochs N         override the model-size default
  --run-name NAME        adopt an existing run directory (resume a prior job's
                         checkpoints and preprocessing cache, e.g. full_job5839815)
  --split-strategy S     random | reference-divergent          (default random)
  -h | --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-type)        MODEL_TYPE=$2;              shift 2;;
        --species)           SPECIES=$2;                 shift 2;;
        --dataset-mode)      DATASET_MODE=$2;            shift 2;;
        --export-each-epoch) EXPORT_MODEL_EACH_EPOCH=$2; shift 2;;
        --allow-clustering)  ALLOW_CLUSTERING=$2;        shift 2;;
        --grad-accum)        GRAD_ACCUM_OVERRIDE=$2;     shift 2;;
        --num-epochs)        NUM_EPOCHS_OVERRIDE=$2;     shift 2;;
        --run-name)          RUN_NAME_OVERRIDE=$2;       shift 2;;
        --split-strategy)    SPLIT_STRATEGY_OVERRIDE=$2; shift 2;;
        -h|--help)           usage; exit 0;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1;;
    esac
done

case "${DATASET_MODE}" in
    clustered|alldata) ;;
    *) echo "Unsupported --dataset-mode: ${DATASET_MODE}. Use 'clustered' or 'alldata'." >&2; exit 1;;
esac

# Validate and normalize model type, and scale hyperparameters
if [[ "${MODEL_TYPE}" == *300m* ]]; then
    MODEL_TYPE="esmc_300m"
    NUM_EPOCHS=10
    GRADIENT_ACCUMULATION_STEPS=64
    TRAIN_BATCH_SIZE=8
    GRADIENT_CHECKPOINTING=true
elif [[ "${MODEL_TYPE}" == *600m* ]]; then
    MODEL_TYPE="esmc_600m"
    NUM_EPOCHS=15
    GRADIENT_ACCUMULATION_STEPS=64
    TRAIN_BATCH_SIZE=8
    GRADIENT_CHECKPOINTING=true
elif [[ "${MODEL_TYPE}" == *6b* || "${MODEL_TYPE}" == *6B* ]]; then
    MODEL_TYPE="esmc_6b"
    NUM_EPOCHS=15
    GRADIENT_ACCUMULATION_STEPS=16
    TRAIN_BATCH_SIZE=2
    GRADIENT_CHECKPOINTING=true
else
    echo "Error: Invalid ESM-C model type '${MODEL_TYPE}'." >&2
    echo "Supported types contain '300m', '600m', or '6b'." >&2
    exit 1
fi

# ── Paths ────────────────────────────────────────────────────────────────────
WORKDIR=.

# The trainer's own default is 150; the v812 block below lowers it to 50 to
# match the floor its FASTA was built with. Every other species keeps 150.
MIN_SEQ_LEN=150

case "${SPECIES}" in
    flu)
        INPUT_FASTA=dev/ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.fas
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magma_esmc_${MODEL_TYPE}_1node_FLU
        CLUSTER_ASSIGNMENTS_CSV=flu_out/cluster_outputs/clusters.csv
        CLUSTER_IDENTITY_FOR_TRAINING=0.99
        CLUSTER_IDENTITY_FOR_SPLIT=0.95
        CLUSTER_THRESHOLDS=0.95,0.99
        ;;
    sc2|og)
        INPUT_FASTA=comb_old_new_filtered.fa
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magma_esmc_${MODEL_TYPE}_1node
        CLUSTER_ASSIGNMENTS_CSV=/projects/u6dr/OM/PLM_Out/cluster_outputs/clusters.csv
        CLUSTER_IDENTITY_FOR_TRAINING=0.99
        CLUSTER_IDENTITY_FOR_SPLIT=0.95
        CLUSTER_THRESHOLDS=0.95,0.99
        ;;
    alltest|bigvirus90_10k)
        INPUT_FASTA=/projects/u6dr/OM/full_virus_test_set/true_head10k_clean.fa
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magma_esmc_${MODEL_TYPE}_1node_bigvirus90_10k
        CLUSTER_ASSIGNMENTS_CSV=/projects/u6dr/OM/full_virus_test_set/true_head10k_clean_cluster_assignments.csv
        CLUSTER_IDENTITY_FOR_TRAINING=0.90
        CLUSTER_IDENTITY_FOR_SPLIT=0.90
        CLUSTER_THRESHOLDS=0.9
        ;;
    alltest100k|bigvirus90_100k)
        INPUT_FASTA=/projects/u6dr/OM/full_virus_test_set/true_head100k_clean.fa
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magma_esmc_${MODEL_TYPE}_1node_bigvirus90_100k
        CLUSTER_ASSIGNMENTS_CSV=/projects/u6dr/OM/full_virus_test_set/true_head100k_clean_cluster_assignments.csv
        CLUSTER_IDENTITY_FOR_TRAINING=0.90
        CLUSTER_IDENTITY_FOR_SPLIT=0.90
        CLUSTER_THRESHOLDS=0.9
        ;;
    allvirus|allRNAvirus|allDNAvirus)
        # Re-clustered 90% mmseqs representative set: duplicate IDs removed,
        # >=150 aa, truncated to 2000 for ESM-C. Built by
        # slurm/prepare_allvirus_fastas.sh and clustered by slurm/cluster_stuff.sbatch.
        # The previous table (meta_data_all_flipped_*.csv) assigned every record
        # to its own cluster because the join keyed on suffixed IDs, so training
        # and validation were effectively the same redundant pool.
        PREPPED_ROOT=/projects/u6dr/OM/allvirus_prepped
        case "${SPECIES}" in
            allRNAvirus) VIRUS_CLASS=RNA ;;
            allDNAvirus) VIRUS_CLASS=DNA ;;
            *)           VIRUS_CLASS= ;;
        esac
        if [[ -n "${VIRUS_CLASS}" ]]; then
            INPUT_FASTA=${PREPPED_ROOT}/${VIRUS_CLASS}/allvirus_rep90_${VIRUS_CLASS}_min150_max2000.fasta
            CLUSTER_ASSIGNMENTS_CSV=${PREPPED_ROOT}/${VIRUS_CLASS}/cluster_outputs/clusters_${VIRUS_CLASS}_esmc_max2000.csv
            DATASET_TAG=rep90${VIRUS_CLASS}_min150_max2000
        else
            INPUT_FASTA=${PREPPED_ROOT}/allvirus_rep90_min150_max2000.fasta
            CLUSTER_ASSIGNMENTS_CSV=${PREPPED_ROOT}/cluster_outputs/clusters_esmc_max2000.csv
            DATASET_TAG=rep90_min150_max2000
        fi
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magma_esmc_${MODEL_TYPE}_${DATASET_TAG}
        # Select training records at 90% (catches the redundancy truncation
        # introduces); split validation off at 70% so held-out sequences are
        # genuinely distant from anything trained on.
        CLUSTER_IDENTITY_FOR_TRAINING=0.90
        CLUSTER_IDENTITY_FOR_SPLIT=0.70
        # Only the columns actually used: 0.90 selects training records, 0.70
        # draws the split. This doubles as the validator's required-column
        # list, so listing a threshold the table lacks (e.g. 0.5) would make
        # the job discard a good CSV and re-cluster 1.3M sequences inline.
        CLUSTER_THRESHOLDS=0.7,0.9
        ;;

    v812all|v812RNA|v812DNA)
        # v20260812: mature-peptide-aware corpus, 90% representatives, >=50 aa,
        # truncated to 2000 for ESM-C. Same paths and thresholds as the FSDP2
        # launcher, so a run here is directly comparable to the FSDP2 run on the
        # same species -- which is the reason this species exists in both.
        #
        # The 50 aa floor is load-bearing: these sets extract mature peptides
        # rather than whole polyproteins, and this script's default floor of 150
        # would silently discard the short records the corpus exists to provide.
        V812_ROOT=/projects/u6dr/VirusFineTune_OM/v20260812/AA50min
        case "${SPECIES}" in
            v812RNA) VIRUS_CLASS=RNA ;;
            v812DNA) VIRUS_CLASS=DNA ;;
            *)       VIRUS_CLASS= ;;
        esac
        if [[ -n "${VIRUS_CLASS}" ]]; then
            INPUT_FASTA=${V812_ROOT}/${VIRUS_CLASS}/nr_id90_min50_${VIRUS_CLASS}_max2000.fasta
            CLUSTER_ASSIGNMENTS_CSV=${V812_ROOT}/${VIRUS_CLASS}/cluster_outputs/clusters_nr_id90_min50_${VIRUS_CLASS}_esmc_max2000.csv
            DATASET_TAG=v812${VIRUS_CLASS}_min50_max2000
        else
            INPUT_FASTA=${V812_ROOT}/nr_id90_min50_max2000.fasta
            CLUSTER_ASSIGNMENTS_CSV=${V812_ROOT}/cluster_outputs/clusters_nr_id90_min50_esmc_max2000.csv
            DATASET_TAG=v812all_min50_max2000
        fi
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magma_esmc_${MODEL_TYPE}_${DATASET_TAG}
        CLUSTER_IDENTITY_FOR_TRAINING=0.90
        CLUSTER_IDENTITY_FOR_SPLIT=0.70
        CLUSTER_THRESHOLDS=0.7,0.9
        MIN_SEQ_LEN=50
        ;;

    *)

        echo "Unsupported SPECIES: ${SPECIES}. Use 'flu', 'sc2', 'og', 'allvirus', 'allRNAvirus', 'allDNAvirus', 'alltest', 'bigvirus90_10k', 'alltest100k', 'bigvirus90_100k', 'v812all', 'v812RNA', or 'v812DNA'." >&2
        exit 1
        ;;
esac

# Keep all-data runs in their own tree so they never overwrite clustered runs.
if [[ "${DATASET_MODE}" == alldata ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_alldata"
fi

# ── Gradient accumulation scaled to dataset size ─────────────────────────────
# Effective batch = TRAIN_BATCH_SIZE x NPROC_PER_NODE x GRADIENT_ACCUMULATION_STEPS.
# grad_accum is a DATASET property, not a model one: it sets the effective batch,
# hence how many optimizer steps you get per epoch. The small clustered sets
# collapse to a few thousand training records after 0.99 clustering, where the
# model-default accum=64 (effective 2048) gives only ~2 steps/epoch and badly
# undertrains (see job 5771697: 4506 records -> 30 total steps). Scale it to the
# dataset so every run gets enough updates. Override for a one-off with
# GRAD_ACCUM_OVERRIDE=<n>.
case "${SPECIES}" in
    flu|sc2|og)
        # ~few-thousand records after 0.99 clustering -> keep the batch small.
        GRADIENT_ACCUMULATION_STEPS=4
        ;;
    alltest|bigvirus90_10k)
        GRADIENT_ACCUMULATION_STEPS=8
        ;;
    alltest100k|bigvirus90_100k)
        GRADIENT_ACCUMULATION_STEPS=16
        ;;
    allvirus|allRNAvirus|allDNAvirus)
        # Millions of records; a large effective batch is fine and desirable.
        GRADIENT_ACCUMULATION_STEPS=64
        ;;
    v812all|v812RNA|v812DNA)
        # 8 x 4 GPU x 4 = 128, the effective batch the FSDP2 v812 runs used.
        # Same optimizer step count per epoch, so the loss curves line up.
        GRADIENT_ACCUMULATION_STEPS=4
        ;;
    # Any other (future) species keeps the model-size default set above.
esac
GRADIENT_ACCUMULATION_STEPS=${GRAD_ACCUM_OVERRIDE:-${GRADIENT_ACCUMULATION_STEPS}}
NUM_EPOCHS=${NUM_EPOCHS_OVERRIDE:-${NUM_EPOCHS}}
EFFECTIVE_BATCH_SIZE=$(( TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS ))

CLUSTER_SUMMARY_CSV=${CLUSTER_ASSIGNMENTS_CSV%.csv}_summary.csv

# Local conda env prefix for ESM-C
ENV_PREFIX=plm_isambardconda_ESMC
JOB_ID=${SLURM_JOB_ID:-manual}
# RUN_NAME defaults to the job id, but --run-name lets a new job adopt an earlier
# run's directory. That is what makes a run resumable across a *resubmission*
# (not just a slurm requeue): the checkpoint search, the preprocessing cache and
# the prepared FASTAs all live under ${OUTPUT_DIR}/${RUN_NAME}, so without this a
# resubmitted job silently starts from scratch and rebuilds ~14.5h of prep.
RUN_NAME=${RUN_NAME_OVERRIDE:-full_job${JOB_ID}}
LOG_DIR=${OUTPUT_DIR}/slurm_logs/${JOB_ID}
STDOUT_LOG=${LOG_DIR}/console.stdout.log
STDERR_LOG=${LOG_DIR}/console.stderr.log
RUN_LOG=${LOG_DIR}/evotune_run.log
RUN_DIR=${OUTPUT_DIR}/${RUN_NAME}
MODEL_DIR=${RUN_DIR}/model
CLUSTER_VALIDATION_LOG=${LOG_DIR}/cluster_validation.stderr.log
CLUSTER_VALIDATION_STATUS=${LOG_DIR}/cluster_validation.status
FILTERED_INPUT_FASTA=${LOG_DIR}/input.filtered_to_cluster_assignments.fasta
MISSING_IDS_LOG=${LOG_DIR}/cluster_missing_fasta_ids.txt
EFFECTIVE_INPUT_FASTA=${INPUT_FASTA}

# NUM_EPOCHS is scaled dynamically in the model validation block above

LATEST_CHECKPOINT=""
if [[ -d "${MODEL_DIR}" ]]; then
    LATEST_CHECKPOINT=$(find "${MODEL_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
fi
if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    EVOTUNE_ARGS+=(--resume-from-checkpoint "${LATEST_CHECKPOINT}")
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

exec > >(tee -a "${STDOUT_LOG}")
exec 2> >(tee -a "${STDERR_LOG}" >&2)

cd "${WORKDIR}"

echo "[$(date -Is)] ESM-C full ${SPECIES} job started"
echo "job_id=${JOB_ID}"
echo "run_name=${RUN_NAME}"
echo "nnodes=${NNODES}  nproc_per_node=${NPROC_PER_NODE}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "input_fasta=${INPUT_FASTA}"
echo "output_dir=${OUTPUT_DIR}"
echo "species=${SPECIES}  dataset_mode=${DATASET_MODE}"
echo "train_batch_size=${TRAIN_BATCH_SIZE}  grad_accum=${GRADIENT_ACCUMULATION_STEPS}  num_epochs=${NUM_EPOCHS}"
echo "effective_batch=${TRAIN_BATCH_SIZE} x ${NPROC_PER_NODE} GPU x ${GRADIENT_ACCUMULATION_STEPS} accum = ${EFFECTIVE_BATCH_SIZE} seq/step"
echo "export_model_each_epoch=${EXPORT_MODEL_EACH_EPOCH}"
echo "allow_clustering=${ALLOW_CLUSTERING}"
echo "cluster_identity_for_training=${CLUSTER_IDENTITY_FOR_TRAINING}"
echo "cluster_identity_for_split=${CLUSTER_IDENTITY_FOR_SPLIT}"
echo "cluster_thresholds=${CLUSTER_THRESHOLDS}"
echo "latest_checkpoint=${LATEST_CHECKPOINT:-none}"
echo "log_dir=${LOG_DIR}"
echo "hostname=$(hostname)  nodelist=${SLURM_NODELIST}"

env | sort > "${LOG_DIR}/environment.txt"

if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "${SLURM_JOB_ID}" > "${LOG_DIR}/slurm_job.txt" 2>&1 || true
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "conda command not found in PATH" >&2
    exit 1
fi

conda activate "${ENV_PREFIX}"

if [[ "${DATASET_MODE}" == alldata ]]; then
    echo "[$(date -Is)] dataset_mode=alldata: skipping clustering; training on every sequence in ${INPUT_FASTA}"
else
mkdir -p "$(dirname "${CLUSTER_ASSIGNMENTS_CSV}")"

CLUSTER_CSV_VALID=false
if [[ -f "${CLUSTER_ASSIGNMENTS_CSV}" ]]; then
    : > "${CLUSTER_VALIDATION_LOG}"
    : > "${CLUSTER_VALIDATION_STATUS}"
    if INPUT_FASTA="${INPUT_FASTA}" CLUSTER_ASSIGNMENTS_CSV="${CLUSTER_ASSIGNMENTS_CSV}" REQUIRED_THRESHOLDS="${CLUSTER_THRESHOLDS}" \
        FILTERED_INPUT_FASTA="${FILTERED_INPUT_FASTA}" MISSING_IDS_LOG="${MISSING_IDS_LOG}" CLUSTER_VALIDATION_STATUS="${CLUSTER_VALIDATION_STATUS}" \
        python - <<'PY' 2>"${CLUSTER_VALIDATION_LOG}"
import csv
import os
import sys
from pathlib import Path

from mmseqs_cluster_table import threshold_column_name

input_fasta = Path(os.environ["INPUT_FASTA"])
cluster_csv = Path(os.environ["CLUSTER_ASSIGNMENTS_CSV"])
required_thresholds = [float(value) for value in os.environ["REQUIRED_THRESHOLDS"].split(",") if value.strip()]
required_columns = {threshold_column_name(value) for value in required_thresholds}
filtered_input_fasta = Path(os.environ["FILTERED_INPUT_FASTA"])
missing_ids_log = Path(os.environ["MISSING_IDS_LOG"])
status_path = Path(os.environ["CLUSTER_VALIDATION_STATUS"])


def write_status(**entries: object) -> None:
    status_path.write_text("".join(f"{key}={value}\n" for key, value in entries.items()))

with cluster_csv.open(newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    missing_columns = sorted(required_columns - set(fieldnames))
    if missing_columns:
        write_status(status="invalid")
        print(f"missing required cluster columns: {', '.join(missing_columns)}", file=sys.stderr)
        sys.exit(1)

    csv_ids = {(row.get("fasta_id") or "").strip() for row in reader if (row.get("fasta_id") or "").strip()}

total_records = 0
retained_records = 0
missing_ids: list[str] = []
current_lines: list[str] = []
current_id: str | None = None


def flush_record(output_handle) -> None:
    global retained_records
    if current_id is None:
        return
    if current_id in csv_ids:
        output_handle.writelines(current_lines)
        retained_records += 1
    else:
        missing_ids.append(current_id)


with input_fasta.open() as source, filtered_input_fasta.open("w") as filtered:
    for raw_line in source:
        if raw_line.startswith(">"):
            flush_record(filtered)
            total_records += 1
            current_id = raw_line[1:].strip().split()[0]
            current_lines = [raw_line]
        elif current_id is not None:
            current_lines.append(raw_line)
    flush_record(filtered)

missing_count = len(missing_ids)
if total_records == 0:
    write_status(status="invalid")
    print("input FASTA contains no records", file=sys.stderr)
    sys.exit(1)

if missing_count:
    missing_ids_log.write_text("\n".join(missing_ids) + "\n")
    preview = ", ".join(missing_ids[:10])
    write_status(
        status="filtered",
        total_records=total_records,
        retained_records=retained_records,
        missing_count=missing_count,
        missing_ids_log=missing_ids_log,
        filtered_input_fasta=filtered_input_fasta,
        preview=preview,
    )
    print(
        f"cluster CSV is missing {missing_count} FASTA IDs; those sequences will be ignored for this run. "
        f"Retaining {retained_records}/{total_records} records. Missing ID preview: {preview}",
        file=sys.stderr,
    )
else:
    write_status(status="valid", total_records=total_records, retained_records=retained_records)
PY
    then
        VALIDATION_MODE=$(awk -F= '$1 == "status" {print $2}' "${CLUSTER_VALIDATION_STATUS}")
        case "${VALIDATION_MODE}" in
            valid)
                CLUSTER_CSV_VALID=true
                ;;
            filtered)
                CLUSTER_CSV_VALID=true
                EFFECTIVE_INPUT_FASTA="${FILTERED_INPUT_FASTA}"
                FILTERED_TOTAL_RECORDS=$(awk -F= '$1 == "total_records" {print $2}' "${CLUSTER_VALIDATION_STATUS}")
                FILTERED_RETAINED_RECORDS=$(awk -F= '$1 == "retained_records" {print $2}' "${CLUSTER_VALIDATION_STATUS}")
                FILTERED_MISSING_COUNT=$(awk -F= '$1 == "missing_count" {print $2}' "${CLUSTER_VALIDATION_STATUS}")
                echo "[$(date -Is)] WARNING: Cluster assignments are incomplete for ${INPUT_FASTA}" >&2
                echo "[$(date -Is)] WARNING: Ignoring ${FILTERED_MISSING_COUNT} sequences without cluster assignments and continuing with ${FILTERED_RETAINED_RECORDS}/${FILTERED_TOTAL_RECORDS} FASTA records." >&2
                echo "[$(date -Is)] WARNING: Filtered FASTA for this run: ${FILTERED_INPUT_FASTA}" >&2
                echo "[$(date -Is)] WARNING: Full missing-ID list: ${MISSING_IDS_LOG}" >&2
                ;;
            *)
                echo "[$(date -Is)] Cluster validation returned an unexpected status: ${VALIDATION_MODE:-unset}" >&2
                ;;
        esac
    else
        if [[ -s "${CLUSTER_VALIDATION_LOG}" ]]; then
            echo "[$(date -Is)] Cluster validation failed for ${CLUSTER_ASSIGNMENTS_CSV}:"
            cat "${CLUSTER_VALIDATION_LOG}"
        fi
        echo "[$(date -Is)] Existing cluster assignments are stale or incomplete; regenerating ${CLUSTER_ASSIGNMENTS_CSV}"
    fi
fi

if [[ "${CLUSTER_CSV_VALID}" != true ]]; then
    case "${ALLOW_CLUSTERING,,}" in
        false|0|no)
            echo "[$(date -Is)] Clustering is disabled (ALLOW_CLUSTERING=${ALLOW_CLUSTERING}); refusing to rebuild ${CLUSTER_ASSIGNMENTS_CSV}" >&2
            if [[ ! -f "${CLUSTER_ASSIGNMENTS_CSV}" ]]; then
                echo "[$(date -Is)] Cluster assignments file is missing." >&2
            fi
            exit 1
            ;;
    esac
    echo "[$(date -Is)] Building cluster assignments at ${CLUSTER_ASSIGNMENTS_CSV}"
    python mmseqs_cluster_table.py \
        --input-fasta "${INPUT_FASTA}" \
        --output-csv "${CLUSTER_ASSIGNMENTS_CSV}" \
        --summary-csv "${CLUSTER_SUMMARY_CSV}" \
        --cluster-thresholds "${CLUSTER_THRESHOLDS}" \
        --min-length 0
else
    echo "[$(date -Is)] Reusing validated cluster assignments at ${CLUSTER_ASSIGNMENTS_CSV}"
fi
fi   # end: clustering skipped when dataset_mode=alldata

echo "effective_input_fasta=${EFFECTIVE_INPUT_FASTA}"

# Dataset selection differs by mode: alldata trains on every sequence via
# full-finetune (streaming FULL split, no clustering); clustered splits against
# the cluster assignments.
#
# SPLIT_STRATEGY defaults to random. It used to be reference-divergent, which
# ranks every record by distance to the longest sequence before splitting — on
# the 484,782-record all-virus set that dominated a ~14.5h preprocessing pass,
# i.e. most of a 24h allocation. It also no longer buys anything: the held-out
# guarantee now comes from --cluster-identity-for-split (0.70), which puts whole
# clusters on one side, so validation is already distant from training by
# construction. Override with --split-strategy for a one-off.
SPLIT_STRATEGY=${SPLIT_STRATEGY_OVERRIDE:-random}
if [[ "${DATASET_MODE}" == alldata ]]; then
    DATASET_ARGS=(
        --run-mode full-finetune
    )
else
    DATASET_ARGS=(
        --split-strategy "${SPLIT_STRATEGY}"
        --cluster-assignments-csv "${CLUSTER_ASSIGNMENTS_CSV}"
        --cluster-identity-for-training "${CLUSTER_IDENTITY_FOR_TRAINING}"
        --cluster-identity-for-split "${CLUSTER_IDENTITY_FOR_SPLIT}"
    )
fi

EVOTUNE_ARGS=(
    --input-fasta "${EFFECTIVE_INPUT_FASTA}"
    --output-dir "${OUTPUT_DIR}"
    --run-name "${RUN_NAME}"
    --model-type "${MODEL_TYPE}"
    --num-epochs "${NUM_EPOCHS}"
    --min-sequence-length "${MIN_SEQ_LEN}"
    --run-mmseqs-cluster-precheck false
    --tokenize-num-proc 8
    --dataloader-num-workers 10
    --bf16 true
    --gradient-checkpointing "${GRADIENT_CHECKPOINTING}"
    --eval-batch-size 2
    # change these together for bigger/smaller models (first up and second down for bigger models)
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --export-model-each-epoch "${EXPORT_MODEL_EACH_EPOCH}"
    --max-val-records 10000
    "${DATASET_ARGS[@]}"
)

if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    EVOTUNE_ARGS+=(--resume-from-checkpoint "${LATEST_CHECKPOINT}")
fi

export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NPROC_PER_NODE ))
if [[ "${OMP_NUM_THREADS}" -lt 1 ]]; then
    export OMP_NUM_THREADS=1
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export HF_HOME=/projects/u6dr/OM/hf_cache

TORCHRUN_ARGS=(
    --nnodes "${NNODES}"
    --nproc-per-node "${NPROC_PER_NODE}"
    --rdzv-backend c10d
    --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}"
    --rdzv-id "${SLURM_JOB_ID:-manual}"
    --rdzv-conf "timeout=${RDZV_TIMEOUT}"
)

SLURM_WALL_SECONDS=""
SIGNAL_BEFORE_SECONDS=""
SLEEP_DURATION=""
TIMER_PID=""
REQUEUE_REQUESTED=0
SRUN_PID=""

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v squeue >/dev/null 2>&1; then
    SLURM_WALL_SECONDS=$(squeue -j "${SLURM_JOB_ID}" --noheader -o "%l" \
        | awk -F'[:-]' '{
            if (NF==4) print ($1*86400 + $2*3600 + $3*60 + $4);
            else if (NF==3) print ($1*3600 + $2*60 + $3);
            else if (NF==2) print ($1*60 + $2);
        }')
fi

if [[ -n "${SLURM_WALL_SECONDS}" && "${SLURM_WALL_SECONDS}" -gt 0 ]]; then
    SIGNAL_BEFORE_SECONDS=$(( SLURM_WALL_SECONDS / 8 ))
    SIGNAL_BEFORE_SECONDS=$(( SIGNAL_BEFORE_SECONDS < 60 ? 60 : SIGNAL_BEFORE_SECONDS ))
    SIGNAL_BEFORE_SECONDS=$(( SIGNAL_BEFORE_SECONDS > 1800 ? 1800 : SIGNAL_BEFORE_SECONDS ))
    SLEEP_DURATION=$(( SLURM_WALL_SECONDS - SIGNAL_BEFORE_SECONDS ))

    if [[ "${SLEEP_DURATION}" -gt 0 ]]; then
        echo "[$(date -Is)] Self-signal timer started: USR1 in ${SLEEP_DURATION}s (wall=${SLURM_WALL_SECONDS}s, buffer=${SIGNAL_BEFORE_SECONDS}s)"
        (
            set +e
            sleep "${SLEEP_DURATION}"
            kill -USR1 $$
        ) &
        TIMER_PID=$!
    else
        echo "[$(date -Is)] Wall time too short for self-signal buffer; timer disabled"
    fi
else
    echo "[$(date -Is)] Could not determine wall time; self-signal timer disabled"
fi

handle_usr1() {
    echo "[$(date -Is)] USR1 received; requesting a graceful checkpoint save and requeue"
    REQUEUE_REQUESTED=1
    # Send SIGUSR1 directly to the python training processes on all nodes.
    # Exclude torchrun (which dies on untrapped USR1) and the bash -c helper
    # (whose cmdline also contains the script name via the pgrep pattern).
    srun --overlap -N "${NNODES}" bash -c 'pgrep -f "evotune_gradient_MAGMA_ESMC.py" | while read -r pid; do
        cmdline=$(ps -p "$pid" -o args= 2>/dev/null) || continue
        case "$cmdline" in *torchrun*|*bash*) continue ;; esac
        kill -USR1 "$pid" 2>/dev/null || true
    done' 2>/dev/null || \
    (pgrep -f "evotune_gradient_MAGMA_ESMC.py" | while read -r pid; do
        cmdline=$(ps -p "$pid" -o args= 2>/dev/null) || continue
        case "$cmdline" in *torchrun*|*bash*) continue ;; esac
        kill -USR1 "$pid" 2>/dev/null || true
    done) || true
}
trap handle_usr1 USR1

# ── FSDP via Accelerate env vars ─────────────────────────────────────────────
# ESM-C uses UnifiedTransformerBlock as the repeated transformer layer.
# Use FSDP_TRANSFORMER_CLS_TO_WRAP=UnifiedTransformerBlock for sharding.
export ACCELERATE_USE_FSDP=true
# 3 = NO_SHARD. Muon's Newton-Schulz iteration orthogonalises whole 2D weight
# matrices, which sharded parameters cannot provide, so the training script sets
# fsdp='no_shard auto_wrap'. Keep this in agreement with it.
export FSDP_SHARDING_STRATEGY=3                    # 3 = NO_SHARD
export FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP
export FSDP_TRANSFORMER_CLS_TO_WRAP=UnifiedTransformerBlock
export FSDP_BACKWARD_PREFETCH=BACKWARD_PRE
export FSDP_STATE_DICT_TYPE=FULL_STATE_DICT
export FSDP_OFFLOAD_PARAMS=false

# ── Log the exact command ─────────────────────────────────────────────────────
{
    printf 'torchrun'
    printf ' %q' "${TORCHRUN_ARGS[@]}"
    printf ' evotune_gradient_MAGMA_ESMC.py'
    printf ' %q' "${EVOTUNE_ARGS[@]}"
    printf '\n'
} > "${LOG_DIR}/command.txt"

nvidia-smi > "${LOG_DIR}/nvidia-smi.txt" 2>&1 || true
cat "${LOG_DIR}/nvidia-smi.txt"
python --version | tee "${LOG_DIR}/python_version.txt"
which python | tee "${LOG_DIR}/python_path.txt"
conda info --envs > "${LOG_DIR}/conda_envs.txt" 2>&1 || true
python - <<'PY' | tee "${LOG_DIR}/torch_preflight.txt"
import platform
import torch
print('platform.machine=', platform.machine())
print('torch=', torch.__version__)
print('torch.version.cuda=', torch.version.cuda)
print('torch.backends.cuda.is_built=', torch.backends.cuda.is_built())
print('torch.cuda.is_available=', torch.cuda.is_available())
print('torch.cuda.device_count=', torch.cuda.device_count())
try:
    from esm.models.esmc import ESMC
    print('esm.models.esmc.ESMC: OK')
except Exception as exc:
    print('esm import error:', exc)
PY

set +e
srun --nodes "${NNODES}" --ntasks "${NNODES}" --ntasks-per-node=1 torchrun \
    "${TORCHRUN_ARGS[@]}" \
    evotune_gradient_MAGMA_ESMC.py "${EVOTUNE_ARGS[@]}" \
    > >(tee -a "${RUN_LOG}") 2>&1 &
SRUN_PID=$!

while kill -0 "${SRUN_PID}" 2>/dev/null; do
    sleep 5
done

wait "${SRUN_PID}"
EXIT_CODE=$?
set -e

if [[ -n "${TIMER_PID}" ]]; then
    kill "${TIMER_PID}" 2>/dev/null || true
fi

if [[ "${EXIT_CODE}" -eq 0 && "${REQUEUE_REQUESTED}" -eq 1 && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[$(date -Is)] Requeuing job ${SLURM_JOB_ID} from latest checkpoint"
    scontrol requeue "${SLURM_JOB_ID}"
fi

if [[ "${EXIT_CODE}" -ne 0 ]]; then
    exit "${EXIT_CODE}"
fi

echo "[$(date -Is)] ESM-C full ${SPECIES} job finished"
