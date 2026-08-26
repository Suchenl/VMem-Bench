#!/bin/bash
# Fan out Stage-1 (adapter trace) across dedicated tgpu GPUs, one job per
# (system, movie, mode). The formal Track A matrix is 9 systems x name_anchored.
# Each job gets its OWN GPU (no sharing) -> maximal
# parallelism on the kml-a800/kml-h800 pool. iamflow additionally offloads its
# LLM+VLM to vLLM servers on node3 gpu1 so it
# does not deadlock in-process.
#
# Design: every job is a self-contained script on shared KFS (all nodes mount
# /data), launched under a remote tmux session, writing a per-job log with
# an "EXIT:<rc>" sentinel. This orchestrator (safe to nohup) then waits on those
# sentinels -- no nested shell-quoting, no ssh babysitting.
#
# Stage-2 (VLM visual-coverage scoring) is a SEPARATE step (needs the qwen3-vl-32b
# judge); run scripts/evaluate_baselines/trackA/tgpu_score_stage2.sh after this reports all done.
set -u

BENCH=.
ADIR=$BENCH/scripts/evaluate_baselines/trackA/baseline_adapters/causal
VACE=python3
WAN=wan2_1/bin/python
HELIOS=helios/bin/python
MSM=MultiShotMaster/bin/python

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=$BENCH/_tgpu_run/$STAMP
JOBDIR=$LOGDIR/jobs
mkdir -p "$JOBDIR"
ln -sfn "$LOGDIR" "$BENCH/_tgpu_run/latest"
MASTER=$LOGDIR/master.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

BBB_DIR=$BENCH/assets/trackA/BlenderOpenMovies/big_buck_bunny
RD_DIR=$BENCH/assets/trackA/LSMDC/0022_Reservoir_Dogs
declare -A MDIR=( [big_buck_bunny]=$BBB_DIR [0022_Reservoir_Dogs]=$RD_DIR )

MOVIES="big_buck_bunny 0022_Reservoir_Dogs"
MODES=${MODES:-"name_anchored"}
SYSTEMS=${SYSTEMS:-"memstrata longlive_rag memflow memflow_sma iamflow retrieval_frame_text_ablation retrieval_seg_uniform_ablation retrieval_seg_dinokey_ablation retrieval_seg_framererank_ablation"}

# --- GPU slot pools (cluster:node:gpu). One slot == one dedicated GPU. --------
# IMPORTANT: only schedule on nodes ALLOCATED TO ME (a800 node2/node3, h800 node1).
# a800 node1 is NOT my allocation (someone else's 69GB job on its gpu0); the cluster
# reaps my "foreign" processes there ~60-90s after model load (observed twice, both
# tmux and setsid). My-node marker = a persistent `memstrata_a800_8b_n*` tmux session.
# Layout (16 general = all non-IAMFlow jobs for 9 systems x 2 movies x 1 mode):
#   memstrata/longlive/memflow/memflow_sma + retrieval family -> a800 node2 + h800 node1
GENERAL_SLOTS=(
  kml-a800:2:0 kml-a800:2:1 kml-a800:2:2 kml-a800:2:3 kml-a800:2:4 kml-a800:2:5 kml-a800:2:6 kml-a800:2:7
  kml-h800:1:0 kml-h800:1:1 kml-h800:1:2 kml-h800:1:3 kml-h800:1:4 kml-h800:1:5 kml-h800:1:6 kml-h800:1:7
)
# iamflow pool: node3 only -- its jobs hit the vLLM servers via 127.0.0.1, so they
# MUST run on the same node as the servers (node3 gpu1). 2 formal jobs -> node3 gpu2-3.
IAMFLOW_SLOTS=( kml-a800:3:2 kml-a800:3:3 kml-a800:3:4 kml-a800:3:5 )
IAMFLOW_LLM=http://127.0.0.1:8100/v1
IAMFLOW_VLM=http://127.0.0.1:8101/v1

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

gi=0; ii=0
JOB_NAMES=(); JOB_LOGS=()

launch(){
  local sys=$1 movie=$2 mode=$3
  local penv; penv=$(env_of "$sys")
  local adapter; adapter=$(adapter_of "$sys")
  local retr_variant; retr_variant=$(retrieval_variant_of "$sys")
  local slot
  if [ "$sys" = "iamflow" ]; then slot=${IAMFLOW_SLOTS[$ii]:-}; ii=$((ii+1)); else slot=${GENERAL_SLOTS[$gi]:-}; gi=$((gi+1)); fi
  if [ -z "${slot:-}" ]; then
    log "ERROR no GPU slot left for $sys/$movie/$mode; reduce SYSTEMS/MODES or extend slot pools"
    exit 91
  fi
  local cl=${slot%%:*} rest=${slot#*:}; local nd=${rest%%:*} gpu=${rest#*:}
  local name="${sys}__${movie}__${mode}"   # already unique (system+movie+mode)
  local joblog=$LOGDIR/$name.log
  local jobsh=$JOBDIR/$name.sh

  {
    echo "#!/bin/bash"
    echo "cd $ADIR || exit 97"
    echo "export CUDA_VISIBLE_DEVICES=$gpu NO_PROXY=localhost,127.0.0.1 PYTORCH_ALLOC_CONF=expandable_segments:True"
    echo "export MAVE_REQUIRE_A800_KEEPALIVE=1 MAVE_TASK_NICE=10 MAVE_FFMPEG_THREADS=1"
    echo "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
    echo "export MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=false"
    if [ "$sys" = "iamflow" ]; then
      echo "export IAMFLOW_LLM_ENDPOINT=$IAMFLOW_LLM IAMFLOW_VLM_ENDPOINT=$IAMFLOW_VLM"
    fi
    if [ -n "$retr_variant" ]; then
      echo "export RETR_VARIANT=$retr_variant MEMSTRATA_RETRIEVAL_VARIANT=$retr_variant"
    fi
    echo "$penv -u runner.py --adapter $adapter --movie-dir ${MDIR[$movie]} --input-mode $mode > $joblog 2>&1"
    echo "echo EXIT:\$? >> $joblog"
  } > "$jobsh"
  chmod +x "$jobsh"
  : > "$joblog"

  log "LAUNCH $name -> $cl node $nd gpu $gpu (env=$(basename $(dirname $(dirname $penv))))"
  # setsid-detached: fully independent of tmux server / SSH session lifetime, so a
  # flaky tmux server (observed on node1) can't orphan/kill the job. Unique joblog
  # + EXIT sentinel on shared KFS is the only handle we need.
  tgpu -c "$cl" -node "$nd" bash -lc "setsid bash $jobsh </dev/null >/dev/null 2>&1 &" >/dev/null 2>&1
  JOB_NAMES+=("$name"); JOB_LOGS+=("$joblog")
}

log "==== TGPU PARALLEL STAGE-1 START (stamp=$STAMP) ===="
log "systems: $SYSTEMS | modes: $MODES | movies: $MOVIES"
for sys in $SYSTEMS; do
  for movie in $MOVIES; do
    for mode in $MODES; do
      launch "$sys" "$movie" "$mode"
    done
  done
done
log "launched ${#JOB_NAMES[@]} jobs; waiting for EXIT sentinels..."

# --- wait for all sentinels (background-safe polling on shared files) ---------
DEADLINE=$(( $(date +%s) + 4*3600 ))   # 4h cap
declare -A DONE=()
while :; do
  ndone=0
  for i in "${!JOB_LOGS[@]}"; do
    n=${JOB_NAMES[$i]}; l=${JOB_LOGS[$i]}
    if [ -n "${DONE[$n]:-}" ]; then ndone=$((ndone+1)); continue; fi
    if grep -q "^EXIT:" "$l" 2>/dev/null; then
      rc=$(grep "^EXIT:" "$l" | tail -1 | cut -d: -f2)
      DONE[$n]=$rc; ndone=$((ndone+1))
      log "DONE  $n  rc=$rc"
      [ "$rc" != "0" ] && tail -n 4 "$l" | sed 's/^/     /' | tee -a "$MASTER"
    fi
  done
  [ "$ndone" -ge "${#JOB_NAMES[@]}" ] && break
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "TIMEOUT after 4h; $ndone/${#JOB_NAMES[@]} done"; break; }
  sleep 30
done

log "==== STAGE-1 FAN-OUT COMPLETE: $ndone/${#JOB_NAMES[@]} finished ===="
log "results dir: $LOGDIR ; next: bash scripts/evaluate_baselines/trackA/tgpu_score_stage2.sh $STAMP"
