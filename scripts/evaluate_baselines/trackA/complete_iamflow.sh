#!/bin/bash
# Complete the 3 iamflow combos left MISSING after the overnight hang.
# Each stage1 is wrapped in `timeout` so an in-process-VLM hang auto-kills and
# the script moves on instead of blocking for hours.
set -u
BENCH=.
ADIR=$BENCH/scripts/evaluate_baselines/trackA/baseline_adapters/causal
AGG=$BENCH/scripts/evaluate_baselines/trackA/aggregate_two_movie_run.py
VACE=python3
LOGDIR=$BENCH/_overnight_run/latest
MASTER=$LOGDIR/iamflow_complete.log
TIMEOUT=${TIMEOUT:-3900}   # 65 min per stage1

export VMEM_REQUIRE_A800_KEEPALIVE=${VMEM_REQUIRE_A800_KEEPALIVE:-1}
export VMEM_TASK_NICE=${VMEM_TASK_NICE:-10}
export VMEM_FFMPEG_THREADS=${VMEM_FFMPEG_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MAX_JOBS=${MAX_JOBS:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

BBB_DIR=$BENCH/assets/trackA/BlenderOpenMovies/big_buck_bunny
BBB_VID=${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos/big_buck_bunny/big_buck_bunny_720p_h264.mp4
RD_DIR=$BENCH/assets/trackA/LSMDC/0022_Reservoir_Dogs
RD_VID=${VMEM_DATASETS_ROOT}/LSMDC/LSMDC_Videos_Stitched/0022_Reservoir_Dogs.mp4

do_combo(){
  local mdir=$1 mvid=$2 mode=$3
  local mv; mv=$(basename "$mdir")
  local runname=iamflow; [ "$mode" = "description_provided" ] && runname=iamflow__descprov
  log "STAGE1 START iamflow | $mv | $mode (timeout=${TIMEOUT}s)"
  ( cd "$ADIR" && \
    CUDA_VISIBLE_DEVICES=0 NO_PROXY=localhost,127.0.0.1 PYTORCH_ALLOC_CONF=expandable_segments:True \
    timeout -k 60 "$TIMEOUT" "$VACE" runner.py --adapter iamflow --movie-dir "$mdir" --input-mode "$mode" ) \
    > "$LOGDIR/stage1_iamflow_${mv}_${mode}.log" 2>&1
  local rc=$?
  log "STAGE1 END   iamflow | $mv | $mode rc=$rc$([ $rc -eq 124 ] && echo ' (TIMEOUT)')"
  [ $rc -ne 0 ] && { log "  skip scoring"; return; }
  log "STAGE2 START iamflow | $mv | $mode runname=$runname"
  # PUBLIC_MODELS_ROOT lets the scorer load the LOCAL DINOv3 snapshot so redun_sim
  # is computed INLINE (no offline recompute). GPU0 is free here (this combo's
  # Stage-1 already finished) so DINOv3 runs on GPU; the VLM judge stays on HTTP.
  ( cd "$BENCH" && NO_PROXY=localhost,127.0.0.1 CUDA_VISIBLE_DEVICES=0 \
    PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT} PYTHONPATH=src \
    "$VACE" -m vmem_bench.scoring.visual_coverage --movie "$mdir" --system "$runname" --video "$mvid" ) \
    > "$LOGDIR/stage2_${runname}_${mv}.log" 2>&1
  log "STAGE2 END   iamflow | $mv | $mode rc=$?"
}

log "==== IAMFLOW COMPLETION START ===="
do_combo "$BBB_DIR" "$BBB_VID" description_provided
do_combo "$RD_DIR"  "$RD_VID"  name_anchored
do_combo "$RD_DIR"  "$RD_VID"  description_provided
log "==== IAMFLOW COMPLETION DONE — aggregating ===="
"$VACE" "$AGG" --out "$LOGDIR" >> "$MASTER" 2>&1
log "==== results: $LOGDIR/results.md ===="
