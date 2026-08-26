#!/bin/bash
# Overnight benchmark: 2 movies (BBB + Reservoir_Dogs) x name_anchored x 9 Track A systems.
# Stage1 (adapter trace) on GPU0; Stage2 (VLM visual-coverage scoring) via HTTP to the
# qwen3-vl-32b server on GPU1. Fault tolerant: one combo failing never kills the rest.
set -u

BENCH=.
ADIR=$BENCH/scripts/evaluate_baselines/trackA/baseline_adapters/causal
AGG=$BENCH/scripts/evaluate_baselines/trackA/aggregate_two_movie_run.py
VACE=python3
WAN=wan2_1/bin/python
HELIOS=helios/bin/python
MSM=MultiShotMaster/bin/python
STAGE1_GPU=${STAGE1_GPU:-0}

export MAVE_REQUIRE_A800_KEEPALIVE=${MAVE_REQUIRE_A800_KEEPALIVE:-1}
export MAVE_TASK_NICE=${MAVE_TASK_NICE:-10}
export MAVE_FFMPEG_THREADS=${MAVE_FFMPEG_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MAX_JOBS=${MAX_JOBS:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=$BENCH/_overnight_run/$STAMP
mkdir -p "$LOGDIR"
MASTER=$LOGDIR/master.log
ln -sfn "$LOGDIR" "$BENCH/_overnight_run/latest"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

# system -> conda python
env_of(){
  case "$1" in
    memstrata) echo "$HELIOS";;
    longlive_rag|memflow|memflow_sma) echo "$WAN";;
    retrieval_*) echo "$MSM";;
    *) echo "$VACE";;
  esac
}
adapter_of(){ case "$1" in retrieval_*) echo "retrieval_family";; *) echo "$1";; esac; }
retrieval_variant_of(){
  case "$1" in
    retrieval_frame_text_ablation) echo "frame_text";;
    retrieval_seg_uniform_ablation) echo "seg_uniform";;
    retrieval_seg_dinokey_ablation) echo "seg_dinokey";;
    retrieval_seg_framererank_ablation) echo "seg_framererank";;
    *) echo "";;
  esac
}
SYSTEMS=${SYSTEMS:-"memstrata longlive_rag memflow memflow_sma iamflow retrieval_frame_text_ablation retrieval_seg_uniform_ablation retrieval_seg_dinokey_ablation retrieval_seg_framererank_ablation"}
MODES=${MODES:-"name_anchored"}

BBB_DIR=$BENCH/assets/trackA/BlenderOpenMovies/big_buck_bunny
BBB_VID=${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos/big_buck_bunny/big_buck_bunny_720p_h264.mp4
RD_DIR=$BENCH/assets/trackA/LSMDC/0022_Reservoir_Dogs
RD_VID=${VMEM_DATASETS_ROOT}/LSMDC/LSMDC_Videos_Stitched/0022_Reservoir_Dogs.mp4

run_combo(){
  local sys=$1 mdir=$2 mvid=$3 mode=$4
  local penv; penv=$(env_of "$sys")
  local adapter; adapter=$(adapter_of "$sys")
  local retr_variant; retr_variant=$(retrieval_variant_of "$sys")
  local mv; mv=$(basename "$mdir")
  local runname=$sys; [ "$mode" = "description_provided" ] && runname="${sys}__descprov"
  local tag="${sys} | ${mv} | ${mode}"

  log "STAGE1 START  $tag  (env=$penv gpu=$STAGE1_GPU)"
  ( cd "$ADIR" && \
    [ -z "$retr_variant" ] || export RETR_VARIANT="$retr_variant" MEMSTRATA_RETRIEVAL_VARIANT="$retr_variant"; \
    CUDA_VISIBLE_DEVICES=$STAGE1_GPU NO_PROXY=localhost,127.0.0.1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$penv" runner.py --adapter "$adapter" --movie-dir "$mdir" --input-mode "$mode" ) \
    > "$LOGDIR/stage1_${sys}_${mv}_${mode}.log" 2>&1
  local rc=$?
  log "STAGE1 END    $tag  rc=$rc"
  if [ $rc -ne 0 ]; then log "  -> skip scoring (stage1 failed): tail below"; tail -n 5 "$LOGDIR/stage1_${sys}_${mv}_${mode}.log" | sed 's/^/     /' | tee -a "$MASTER"; return; fi

  log "STAGE2 START  $tag  runname=$runname"
  # PUBLIC_MODELS_ROOT -> local DINOv3 snapshot so redun_sim is computed INLINE.
  # GPU0 is free during Stage-2 (this combo's Stage-1 already finished): DINOv3 on
  # GPU, VLM judge on HTTP (GPU1).
  ( cd "$BENCH" && \
    NO_PROXY=localhost,127.0.0.1 CUDA_VISIBLE_DEVICES=$STAGE1_GPU \
    PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT} PYTHONPATH=src \
    "$VACE" -m vmem_bench.scoring.visual_coverage \
    --movie "$mdir" --system "$runname" --video "$mvid" ) \
    > "$LOGDIR/stage2_${runname}_${mv}.log" 2>&1
  local rc2=$?
  log "STAGE2 END    $tag  rc=$rc2"
  if [ $rc2 -ne 0 ]; then tail -n 5 "$LOGDIR/stage2_${runname}_${mv}.log" | sed 's/^/     /' | tee -a "$MASTER"; fi
}

log "==== OVERNIGHT RUN START (stamp=$STAMP) ===="
log "systems: $SYSTEMS | modes: $MODES | movies: big_buck_bunny, 0022_Reservoir_Dogs"
for sys in $SYSTEMS; do
  for mode in $MODES; do
    run_combo "$sys" "$BBB_DIR" "$BBB_VID" "$mode"
    run_combo "$sys" "$RD_DIR"  "$RD_VID"  "$mode"
  done
  log "---- system $sys done (both movies, configured modes) ----"
done

log "==== ALL COMBOS DONE — aggregating ===="
"$VACE" "$AGG" --out "$LOGDIR" 2>&1 | tee -a "$MASTER"
log "==== OVERNIGHT RUN COMPLETE — results: $LOGDIR/results.md ===="
