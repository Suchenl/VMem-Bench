#!/usr/bin/env bash
# Resident MiniMax-H3 Ref2V Turbo server on one GPU of a training node (never on the dev host).
#
#   H3_PYTHON=<python with torch+fastapi+uvicorn+imageio> H3_DIFFSYNTH=<DiffSynth-Studio checkout with
#   MiniMax-H3> PUBLIC_MODELS_ROOT=<dir holding MiniMax-H3/ and MiniMax-H3-Turbo/> \
#     bash scripts/testbed/minimax_h3/serve.sh <gpu> <port> <endpoint_dir>
#
# VMEM_H3_ROOT / VMEM_H3_TURBO_LORA override the two weight paths. H3_SERVER_ARGS passes server flags:
#   --geometry 480x832|704x1280   default render preset (480x832)
#   --offload none|text_encoder|all, --vram-limit-gib N
# Fully resident (--offload none) needs ~120 GB VRAM (B200/H200); 80 GB A800 and 72 GB RTX PRO 5000
# need host offload (see src/vmem_bench/testbed/README.md for the measured settings).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU="${1:?usage: serve.sh <gpu> <port> <endpoint_dir>}"
PORT="${2:?usage: serve.sh <gpu> <port> <endpoint_dir>}"
END="${3:?usage: serve.sh <gpu> <port> <endpoint_dir>}"
: "${H3_PYTHON:?set H3_PYTHON}"
: "${H3_DIFFSYNTH:?set H3_DIFFSYNTH (DiffSynth-Studio with MiniMax-H3)}"

export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export PYTHONPATH="$H3_DIFFSYNTH:$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
mkdir -p "$END"
# shellcheck disable=SC2086
exec "$H3_PYTHON" -u -m vmem_bench.testbed.backends.minimax_h3.server \
  --host 0.0.0.0 --port "$PORT" --endpoint-dir "$END" ${H3_SERVER_ARGS:-}
