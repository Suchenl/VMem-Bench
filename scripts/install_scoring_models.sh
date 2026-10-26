#!/usr/bin/env bash
# Install every model the Track-B/Track-A scorers use into the layout they expect, verifying SHA256
# against src/vmem_bench/scoring/configs/scoring_models_manifest.json. Roots: MEMSTRATA_WEIGHTS_ROOT
# (ArcFace, MegaLoc) and PUBLIC_MODELS_ROOT (DINOv3, GroundingDINO, Qwen3-VL-32B, SigLIP2).
# Usage: scripts/install_scoring_models.sh [--dry-run] [--only name,...] [--with-optional]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}" exec "${PYTHON:-python3}" -m vmem_bench.install_models install "$@"
