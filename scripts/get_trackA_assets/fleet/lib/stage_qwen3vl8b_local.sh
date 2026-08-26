#!/usr/bin/env bash
# Copy Qwen3-VL-8B once onto this BDY node's local /tmp (NOT /dev/shm).
# Idempotent: skips if all 4 safetensor shards + .stage_ok exist.
set -euo pipefail

SRC="${SRC_MODEL:-${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-8B-Instruct}"
LOCAL_PUBLIC="${LOCAL_PUBLIC_MODELS_ROOT:-/tmp/memstrata_public_models}"
DST="$LOCAL_PUBLIC/Qwen/Qwen3-VL-8B-Instruct"
MARKER="$DST/.stage_ok"

echo "=== $(date -Is) stage 8B host=$(hostname -s) src=$SRC dst=$DST ==="
df -h /tmp | head -2

need_copy=1
if [[ -f "$MARKER" ]] \
  && [[ -f "$DST/model-00001-of-00004.safetensors" ]] \
  && [[ -f "$DST/model-00002-of-00004.safetensors" ]] \
  && [[ -f "$DST/model-00003-of-00004.safetensors" ]] \
  && [[ -f "$DST/model-00004-of-00004.safetensors" ]] \
  && [[ -f "$DST/config.json" ]]; then
  echo "already staged ($(cat "$MARKER")); skip copy"
  need_copy=0
fi

if (( need_copy == 1 )); then
  mkdir -p "$LOCAL_PUBLIC/Qwen"
  rm -rf "$DST"
  mkdir -p "$DST"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 --exclude '.cache' "$SRC/" "$DST/"
  else
    for f in README.md chat_template.json config.json generation_config.json \
             merges.txt model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
             model-00003-of-00004.safetensors model-00004-of-00004.safetensors \
             model.safetensors.index.json preprocessor_config.json tokenizer.json \
             tokenizer_config.json video_preprocessor_config.json vocab.json; do
      [[ -f "$SRC/$f" ]] || continue
      echo "  cp $f"
      cp -f "$SRC/$f" "$DST/$f"
    done
  fi
  date -Is > "$MARKER"
  echo "staged ok $(du -sh "$DST" | awk '{print $1}')"
fi

ls -lah "$DST" | head -20
export PUBLIC_MODELS_ROOT="$LOCAL_PUBLIC"
echo "PUBLIC_MODELS_ROOT=$PUBLIC_MODELS_ROOT"
echo "=== $(date -Is) stage done ==="
