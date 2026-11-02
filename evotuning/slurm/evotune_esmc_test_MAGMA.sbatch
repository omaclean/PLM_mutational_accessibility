#!/bin/bash
#SBATCH --job-name=evotune-esmc-test_magma
#SBATCH --output=SC2_out/slurm-esmc-%j.out
#SBATCH --error=SC2_out/slurm-esmc-%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:25:00

# ── Topology ────────────────────────────────────────────────────────────────
NNODES=${SLURM_NNODES:-1}
NPROC_PER_NODE=2
RDZV_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
RDZV_TIMEOUT=${RDZV_TIMEOUT:-900}

MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)
export MASTER_ADDR MASTER_PORT=${RDZV_PORT}

set -euo pipefail

MODEL_TYPE=${1:-'esmc_300m'}
SPECIES=${2:-'sc2'}
EXPORT_MODEL_EACH_EPOCH=${3:-${EXPORT_MODEL_EACH_EPOCH:-false}}
ALLOW_CLUSTERING=${ALLOW_CLUSTERING:-true}

# Validate and normalize model type, and scale hyperparameters
if [[ "${MODEL_TYPE}" == *300m* ]]; then
    MODEL_TYPE="esmc_300m"
    GRAD_ACCUM_STEPS=4
    TRAIN_BATCH_SIZE=2
    GRADIENT_CHECKPOINTING=false
elif [[ "${MODEL_TYPE}" == *600m* ]]; then
    MODEL_TYPE="esmc_600m"
    GRAD_ACCUM_STEPS=4
    TRAIN_BATCH_SIZE=2
    GRADIENT_CHECKPOINTING=false
elif [[ "${MODEL_TYPE}" == *6b* || "${MODEL_TYPE}" == *6B* ]]; then
    MODEL_TYPE="esmc_6b"
    GRAD_ACCUM_STEPS=16
    TRAIN_BATCH_SIZE=1
    GRADIENT_CHECKPOINTING=true
else
    echo "Error: Invalid ESM-C model type '${MODEL_TYPE}'." >&2
    echo "Supported types contain '300m', '600m', or '6b'." >&2
    exit 1
fi

WORKDIR=.

case "${SPECIES}" in
    flu)
        INPUT_FASTA=dev/ncbiflu_HA_all_110424_noX_clu99_filt_HA-80.fas
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magmatest_${MODEL_TYPE}_FLU
        CLUSTER_ASSIGNMENTS_CSV=flu_out/cluster_outputs/clusters.csv
        CLUSTER_IDENTITY_FOR_TRAINING=0.99
        CLUSTER_IDENTITY_FOR_SPLIT=0.95
        CLUSTER_THRESHOLDS=0.95,0.99
        ;;
    sc2|og)
        INPUT_FASTA=comb_old_new_filtered.fa
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magmatest_${MODEL_TYPE}
        CLUSTER_ASSIGNMENTS_CSV=/projects/u6dr/OM/PLM_Out/cluster_outputs/clusters.csv
        CLUSTER_IDENTITY_FOR_TRAINING=0.99
        CLUSTER_IDENTITY_FOR_SPLIT=0.95
        CLUSTER_THRESHOLDS=0.95,0.99
        ;;
    alltest|bigvirus90_10k)
        INPUT_FASTA=/projects/u6dr/OM/full_virus_test_set/true_head10k_clean.fa
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magmatest_${MODEL_TYPE}_bigvirus90_10k
        CLUSTER_ASSIGNMENTS_CSV=/projects/u6dr/OM/full_virus_test_set/true_head10k_clean_cluster_assignments.csv
        CLUSTER_IDENTITY_FOR_TRAINING=0.90
        CLUSTER_IDENTITY_FOR_SPLIT=0.90
        CLUSTER_THRESHOLDS=0.9
        ;;
    alltest100k|bigvirus90_100k)
        INPUT_FASTA=/projects/u6dr/OM/full_virus_test_set/true_head100k_clean.fa
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magmatest_${MODEL_TYPE}_bigvirus90_100k
        CLUSTER_ASSIGNMENTS_CSV=/projects/u6dr/OM/full_virus_test_set/true_head100k_clean_cluster_assignments.csv
        CLUSTER_IDENTITY_FOR_TRAINING=0.90
        CLUSTER_IDENTITY_FOR_SPLIT=0.90
        CLUSTER_THRESHOLDS=0.9
        ;;
    allvirus|allRNAvirus|allDNAvirus)
        # Re-clustered 90% mmseqs representative set, >=150 aa, truncated to 2000
        # for ESM-C. Built by slurm/prepare_allvirus_fastas.sh, clustered by
        # slurm/cluster_stuff.sbatch.
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
        OUTPUT_DIR=/projects/u6dr/OM/PLM_Out/magmatest_${MODEL_TYPE}_${DATASET_TAG}
        CLUSTER_IDENTITY_FOR_TRAINING=0.90
        CLUSTER_IDENTITY_FOR_SPLIT=0.70
        # Only the columns actually used: 0.90 selects training records, 0.70
        # draws the split. This doubles as the validator's required-column
        # list, so listing a threshold the table lacks (e.g. 0.5) would make
        # the job discard a good CSV and re-cluster 1.3M sequences inline.
        CLUSTER_THRESHOLDS=0.7,0.9
        ;;
    *)
        echo "Unsupported SPECIES: ${SPECIES}. Use 'flu', 'sc2', 'og', 'allvirus', 'allRNAvirus', 'allDNAvirus', 'alltest', 'bigvirus90_10k', 'alltest100k', or 'bigvirus90_100k'." >&2
        exit 1
        ;;
esac

CLUSTER_SUMMARY_CSV=${CLUSTER_ASSIGNMENTS_CSV%.csv}_summary.csv
# GRAD_ACCUM_STEPS is set dynamically above
TEST_FASTA=${OUTPUT_DIR}/${SPECIES}.test64.fa
# Local conda env prefix for ESM-C
ENV_PREFIX=plm_isambardconda_ESMC
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_NAME=test64_ga${GRAD_ACCUM_STEPS}_job${JOB_ID}
LOG_DIR=${OUTPUT_DIR}/slurm_logs/${JOB_ID}
STDOUT_LOG=${LOG_DIR}/console.stdout.log
STDERR_LOG=${LOG_DIR}/console.stderr.log
RUN_LOG=${LOG_DIR}/evotune_run.log
CLUSTER_VALIDATION_LOG=${LOG_DIR}/cluster_validation.stderr.log
CLUSTER_VALIDATION_STATUS=${LOG_DIR}/cluster_validation.status
FILTERED_INPUT_FASTA=${LOG_DIR}/input.filtered_to_cluster_assignments.fasta
MISSING_IDS_LOG=${LOG_DIR}/cluster_missing_fasta_ids.txt
EFFECTIVE_INPUT_FASTA=${INPUT_FASTA}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
export OUTPUT_DIR
export GRAD_ACCUM_STEPS
export EFFECTIVE_INPUT_FASTA
export TEST_FASTA

exec > >(tee -a "${STDOUT_LOG}")
exec 2> >(tee -a "${STDERR_LOG}" >&2)

cd "${WORKDIR}"

echo "[$(date -Is)] ESM-C test job started"
echo "job_id=${JOB_ID}"
echo "workdir=${WORKDIR}"
echo "run_name=${RUN_NAME}"
echo "model_type=${MODEL_TYPE}"
echo "nnodes=${NNODES}  nproc_per_node=${NPROC_PER_NODE}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "input_fasta=${INPUT_FASTA}"
echo "output_dir=${OUTPUT_DIR}"
echo "species=${SPECIES}"
echo "export_model_each_epoch=${EXPORT_MODEL_EACH_EPOCH}"
echo "allow_clustering=${ALLOW_CLUSTERING}"
echo "cluster_identity_for_training=${CLUSTER_IDENTITY_FOR_TRAINING}"
echo "cluster_identity_for_split=${CLUSTER_IDENTITY_FOR_SPLIT}"
echo "cluster_thresholds=${CLUSTER_THRESHOLDS}"
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

echo "effective_input_fasta=${EFFECTIVE_INPUT_FASTA}"

EVOTUNE_ARGS=(
    --input-fasta "${TEST_FASTA}"
    --output-dir "${OUTPUT_DIR}"
    --run-name "${RUN_NAME}"
    --model-type "${MODEL_TYPE}"
    --num-epochs 1
    --max-records 64
    --tokenize-num-proc 1
    --dataloader-num-workers 10
    --bf16 true
    --gradient-checkpointing "${GRADIENT_CHECKPOINTING}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --eval-batch-size 4
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
    --split-strategy reference-divergent
    --cluster-assignments-csv "${CLUSTER_ASSIGNMENTS_CSV}"
    --cluster-identity-for-training "${CLUSTER_IDENTITY_FOR_TRAINING}"
    --cluster-identity-for-split "${CLUSTER_IDENTITY_FOR_SPLIT}"
    --test-mode true
    --test-checkpoint-save-and-cleanup true
    --export-model-each-epoch "${EXPORT_MODEL_EACH_EPOCH}"
    --test
)

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
    --rdzv-endpoint "${MASTER_ADDR}:${RDZV_PORT}"
    --rdzv-id "${SLURM_JOB_ID:-manual}"
    --rdzv-conf "timeout=${RDZV_TIMEOUT}"
)

# ── FSDP via Accelerate env vars ─────────────────────────────────────────────
# ESM-C uses UnifiedTransformerBlock as the repeated transformer layer.
export ACCELERATE_USE_FSDP=true
export FSDP_SHARDING_STRATEGY=1
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

# ── Extract 64-record test FASTA ──────────────────────────────────────────────
python - <<'PY'
import os
from pathlib import Path

input_fasta = Path(os.environ['EFFECTIVE_INPUT_FASTA'])
output_fasta = Path(os.environ['TEST_FASTA'])
max_records = 64
written = 0

with input_fasta.open() as src, output_fasta.open('w') as dst:
    current = []
    for line in src:
        if line.startswith('>'):
            if current:
                dst.writelines(current)
                written += 1
                if written >= max_records:
                    break
            current = [line]
        else:
            current.append(line)
    else:
        if current and written < max_records:
            dst.writelines(current)
            written += 1

print(f'wrote {written} records to {output_fasta}')
PY

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

srun --nodes "${NNODES}" --ntasks "${NNODES}" --ntasks-per-node=1 torchrun \
    "${TORCHRUN_ARGS[@]}" \
    evotune_gradient_MAGMA_ESMC.py "${EVOTUNE_ARGS[@]}" \
    2>&1 | tee -a "${RUN_LOG}"

echo "[$(date -Is)] ESM-C test job finished"
