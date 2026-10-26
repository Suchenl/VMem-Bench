#!/usr/bin/env bash
# Download the official Big Buck Bunny 720p file into the layout runner.py expects.
# Source: Blender Foundation / Peach (CC BY). We do not vendor the pixels in git.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${VMEM_DATASETS_ROOT:-$HERE/data}"
DEST="$ROOT/BlenderOpenMovies/Videos/big_buck_bunny"
URL="${BBB_URL:-https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov}"
mkdir -p "$DEST"
OUT="$DEST/big_buck_bunny_720p_h264.mov"
if [ -f "$OUT" ]; then
  echo "[prepare_blender] already have $OUT"
  exit 0
fi
echo "[prepare_blender] downloading BBB 720p -> $OUT"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "$OUT.part" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT.part" "$URL"
else
  echo "need curl or wget" >&2
  exit 2
fi
mv "$OUT.part" "$OUT"
echo "[prepare_blender] VMEM_DATASETS_ROOT=$ROOT"
echo "[prepare_blender] gold is already in assets/trackA/BlenderOpenMovies/big_buck_bunny/"
