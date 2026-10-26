#!/usr/bin/env bash
# Testbed parity gate: run one condition of the migrated Track-B testbed (python -m vmem_bench.testbed.run)
# on the first N production shots of a story with the paper backend and seed 2026, for comparison with the
# frozen paper artifacts (compare.py).
#
# Usage: run_parity.sh <variant> <gen_gpu> <encoder_gpu> <out_root> [segments] [story]
#   gen_gpu      physical GPU for the FLUX and Wan servers (co-resident, as in the paper runs)
#   encoder_gpu  physical GPU for the retrieval encoders (Qwen3-Embedding-4B + SigLIP2 do not fit next to FLUX+Wan)
# Needs: PUBLIC_MODELS_ROOT, VMEM_WAN22_TURBO_REPO, VMEM_FLUX_PYTHON (diffusers with Flux2KleinKVPipeline),
#        TESTBED_PYTHON (torch + transformers + flash-attn; runs the runner and the Wan server).
set -u
VARIANT="${1:?variant}"; GEN="${2:?gen gpu}"; ENC="${3:?encoder gpu}"; OUT="${4:?out root}"
SEGMENTS="${5:-5}"; STORY="${6:-0001_lighthouse_keeper}"
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="${TESTBED_PYTHON:?TESTBED_PYTHON}"
: "${PUBLIC_MODELS_ROOT:?}" "${VMEM_WAN22_TURBO_REPO:?}" "${VMEM_FLUX_PYTHON:?}"
mkdir -p "${OUT}/logs"
LOG="${OUT}/logs/parity_${VARIANT}_$(hostname -s).log"
exec >>"${LOG}" 2>&1
export PYTHONPATH="${BENCH}/src" VMEM_WAN_PYTHON="${PY}" VMEM_FLUX_GPU="${GEN}" VMEM_WAN_GPU="${GEN}"
export CUDA_VISIBLE_DEVICES="${ENC}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
echo "=== ${VARIANT} start $(date -Iseconds) host=$(hostname -s) gen_gpu=${GEN} encoder_gpu=${ENC} commit=$(git -C "${BENCH}" rev-parse --short HEAD)"
"${PY}" -u -m vmem_bench.testbed.run --variant "${VARIANT}" --budget 3 --backend flux2_wan22 --seed 2026 \
  --run-tag parity_testbed --out-root "${OUT}" --stories "${STORY}" --segments "${SEGMENTS}" --overwrite \
  --backend-opt wan_min_free_mib=40000
RC=$?
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n "$((GEN + 1))p;$((ENC + 1))p"
echo "=== ${VARIANT} EXIT:${RC} $(date -Iseconds)"
