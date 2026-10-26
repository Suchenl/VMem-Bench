#!/usr/bin/env python3
"""Track B Stage-1 progress in a compact per-system / per-story table.

Analogue of the TrackA ``progress_detail.py``. It reads the multi-node
orchestrator run directory (``outputs/evaluation/trackB/_stage1_multinode/latest``)
plus the durable per-story ``trackb_manifest.json`` contracts and prints:

  * one summary row per system: DONE / RUN / QUEUED / FAIL / UNSCHED story counts
    and segment progress (segDONE/segTOTAL). segDONE counts every segment a system
    has produced, INCLUDING those of stories still running, so the four rows are
    directly comparable and a system whose first wave has not finished is not
    reported as having produced nothing,
  * an optional ``--detail`` per-story matrix with per-job state + RUN heartbeat,
  * an ETA extrapolated from segment throughput since the run started.

``memstrata`` is driven by its own fleet rather than by this orchestrator, so it
has no joblog and no EXIT sentinel to read. Its row is filled from the per-story
``progress.json`` each producer writes under
``outputs/evaluation/trackB/memstrata/<story>/name_anchored/<tag>/``, which also
makes partial progress visible while a story is still running.

State per (system, story):
  DONE  - trackb_manifest.json status==done, exit_code==0 (durable contract);
          for memstrata: every shot closed AND no interior gap
  FAIL  - manifest exit_code!=0, or joblog wrote EXIT:<non-zero>
  RUN   - a joblog exists (job launched) with no EXIT sentinel yet;
          for memstrata: progress.json touched within the last 25 min
  QUEUED- scheduled in this run but not launched; for memstrata: waiting for a
          card, or complete but still missing a shot (needs the fill pass)
  UNSCHED - not part of this run's story set

A memstrata story missing an interior shot is NOT counted as done: that shot has
no video, so the story no longer aligns shot-for-shot with the prompt stream it
is scored against. Such gaps are printed as ``gaps@[...]`` in ``--detail``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_orchestrator():
    spec = importlib.util.spec_from_file_location(
        "trackb_run_stage1_multinode", HERE / "run_stage1_multinode.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


ORCH = _load_orchestrator()
OUT_ROOT: Path = ORCH.OUT_ROOT
STEPS: dict = ORCH.STEPS

# memstrata first so its reserved monitoring row is always visible up top.
SYSTEMS = ["memstrata", "memflow", "longlive_rag", "iamflow"]
MANAGED = ["memflow", "longlive_rag", "iamflow"]  # driven by this orchestrator

STATE_KEYS = ["DONE", "RUN", "QUEUED", "FAIL", "UNSCHED"]

_EXIT_RE = re.compile(r"^EXIT:\s*(-?\d+)", re.MULTILINE)


def latest_run_dir(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    latest = OUT_ROOT / "_stage1_multinode" / "latest"
    if latest.exists():
        return latest.resolve()
    root = OUT_ROOT / "_stage1_multinode"
    stamps = [p for p in root.glob("2*_*") if p.is_dir()] if root.is_dir() else []
    return max(stamps, key=lambda p: p.name) if stamps else None


def run_started_at(run_dir: Path) -> float | None:
    name = run_dir.name
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        master = run_dir / "master.log"
        return master.stat().st_mtime if master.is_file() else None


def manifest_state(system: str, story_id: str, started: float | None = None) -> tuple[str, int]:
    """Return (state, exit_code) from the durable per-story manifest, else ('', 0).

    Per-story output dirs are shared across runs (not stamp-scoped), so a manifest
    written by an EARLIER run is stale for the current one. If ``started`` is given,
    ignore any manifest older than the current run start so a dead run's FAIL/DONE
    markers don't mask this run's live jobs.
    """
    out_dir = ORCH.out_dir_for(system, story_id)
    manifest = out_dir / "trackb_manifest.json"
    if not manifest.is_file():
        return "", 0
    if started is not None:
        try:
            if manifest.stat().st_mtime < started - 1:
                return "", 0  # stale manifest from a previous run
        except OSError:
            pass
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", 0
    rc = int(data.get("exit_code", 0) or 0)
    if data.get("status") == "done" and rc == 0:
        return "DONE", 0
    return "FAIL", rc


def run_log_path(system: str, story_id: str) -> Path:
    return ORCH.out_dir_for(system, story_id) / "logs" / f"{system}_run.log"


def load_seg_progress(run_dir: Path) -> dict[tuple[str, str], int]:
    """Read the unified per-segment heartbeat file (run_dir/seg_progress.jsonl).

    Returns {(system, story): max seg_done}. Every baseline appends one JSONL line
    per completed segment (see _trackb_seg_emit in each pipeline), so this gives
    real-time in-flight progress for ALL systems, not just iamflow. Empty/missing
    file (e.g. a run launched before this instrumentation) -> empty map.
    """
    out: dict[tuple[str, str], int] = {}
    f = run_dir / "seg_progress.jsonl"
    if not f.is_file():
        return out
    try:
        txt = f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            key = (r.get("system", ""), r.get("story", ""))
            sd = int(r.get("seg_done", 0))
        except (ValueError, TypeError):
            continue
        if sd > out.get(key, -1):
            out[key] = sd
    return out


def load_seg_progress_global() -> dict[tuple[str, str], int]:
    """Merge seg_progress heartbeats across ALL orchestrator run dirs.

    The 3 baselines can be driven by separate orchestrators (e.g. a longlive sweep
    plus memflow/iamflow backfills), each with its own run_dir + seg_progress.jsonl.
    A monitor pointed at a single run_dir would otherwise miss the other systems'
    live heartbeats. Keys are (system, story).

    IMPORTANT: merge by LATEST timestamp, not max seg_done. A story that FAILED in
    an old sweep (e.g. reached seg 118 then OOM'd at decode) and is now re-running
    under a backfill (currently seg 82) must report the CURRENT live count (82), not
    the stale peak (118). Max-seg_done merge froze segDONE at dead runs' peaks; the
    live run always owns the most recent heartbeat, so latest-ts wins.
    """
    best: dict[tuple[str, str], tuple[float, int]] = {}
    root = OUT_ROOT / "_stage1_multinode"
    if root.is_dir():
        for rd in root.glob("2*_*"):
            f = rd / "seg_progress.jsonl"
            if not f.is_file():
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = (r.get("system", ""), r.get("story", ""))
                    ts = float(r.get("ts", 0))
                    sd = int(r.get("seg_done", 0))
                except (ValueError, TypeError):
                    continue
                if ts >= best.get(key, (-1.0, 0))[0]:
                    best[key] = (ts, sd)
    return {k: v[1] for k, v in best.items()}


def load_seg_last_ts_global() -> dict[tuple[str, str], float]:
    """{(system, story): last heartbeat ts} merged across ALL run dirs."""
    out: dict[tuple[str, str], float] = {}
    root = OUT_ROOT / "_stage1_multinode"
    if not root.is_dir():
        return out
    for rd in root.glob("2*_*"):
        f = rd / "seg_progress.jsonl"
        if not f.is_file():
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r.get("system", ""), r.get("story", ""))
                ts = float(r.get("ts", 0))
            except (ValueError, TypeError):
                continue
            if ts > out.get(key, -1.0):
                out[key] = ts


        # (loop over run dirs continues)
    return out


def run_log_fresh(system: str, story_id: str, max_age_s: float = 900.0) -> bool:
    """True if the canonical per-story run log was written recently.

    Used only as a fallback for the LOADING phase (job launched, model still
    loading, no heartbeat yet). During generation, silent runners stop touching
    this file, so heartbeat ts is the primary liveness signal (see is_live).
    """
    rl = run_log_path(system, story_id)
    try:
        return rl.is_file() and (time.time() - rl.stat().st_mtime) < max_age_s
    except OSError:
        return False


def is_live(system: str, story_id: str, nseg: int,
            segprog: dict, seg_ts: dict, hb_fresh_s: float = 900.0) -> bool:
    """Whether a story NOT launched in the watched run_dir is actually running.

    Robust across orchestrator run_dirs (split backfills) and across the silent
    generation + whole-story-decode phases of longlive/memflow:
      * fully generated (seg_done>=nseg) but not done  -> final VAE decode -> live
      * fresh heartbeat (< hb_fresh_s)                  -> generating       -> live
      * no heartbeat yet but run log fresh              -> still loading     -> live
    """
    key = (system, story_id)
    sd = segprog.get(key)
    if sd is not None:
        if nseg and sd >= nseg:
            return True
        if (time.time() - seg_ts.get(key, 0.0)) < hb_fresh_s:
            return True
        return run_log_fresh(system, story_id)  # heartbeat gone stale mid-decode
    return run_log_fresh(system, story_id)


# iamflow logs one line per prompt switch; memflow/longlive are "silent runners"
# (a single in-memory tqdm/rollout, no per-segment log lines during generation),
# so their only reliable progress signal is story completion + GPU utilization.
SILENT_RUNNERS = {"memflow", "longlive_rag"}


def live_inflight(system: str, story_id: str, nseg: int,
                  segprog: dict[tuple[str, str], int] | None = None) -> tuple[int, int] | None:
    """Best-effort in-flight segment progress for a RUN job, else None.

    Primary source: the unified seg_progress.jsonl heartbeat (all systems). Falls
    back to iamflow's own "Transition complete" run-log signal for runs launched
    before the heartbeat instrumentation existed.
    """
    if segprog is not None:
        sd = segprog.get((system, story_id))
        if sd is not None:
            return min(sd, nseg), nseg
    # Fallback (pre-instrumentation runs): iamflow logs per-segment; others silent.
    if system in SILENT_RUNNERS:
        return None
    log = run_log_path(system, story_id)
    if not log.is_file():
        return None
    try:
        txt = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if system == "iamflow":
        return min(txt.count("Transition complete"), nseg), nseg
    return None


def gpu_snapshot(cluster: str, nodes: list[int]) -> dict[int, str]:
    """On-demand per-node VRAM usage % (used/total). One tgpu call/node; --gpu only."""
    import subprocess
    out: dict[int, str] = {}
    q = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
    for nd in nodes:
        try:
            res = subprocess.run(["tgpu", "-c", cluster, "-node", str(nd), "bash", "-lc", q],
                                 capture_output=True, text=True, timeout=60)
            cells = []
            for l in res.stdout.splitlines():
                parts = [p.strip() for p in l.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    used, total = int(parts[0]), int(parts[1])
                    pct = 100.0 * used / total if total else 0.0
                    cells.append(f"{pct:.0f}%")
            out[nd] = " | ".join(cells) if cells else "unreachable"
        except Exception:
            out[nd] = "unreachable"
    return out


def collect(run_dir: Path, limit_stories: int):
    stories = ORCH.enumerate_stories(limit_stories)  # [(story_id, prompt_path, nseg)]
    logdir = run_dir / "logs"
    started = run_started_at(run_dir)
    rows: dict[str, list[dict]] = {s: [] for s in SYSTEMS}
    segprog_g = load_seg_progress_global()
    seg_ts_g = load_seg_last_ts_global()

    for system in MANAGED:
        for story_id, _prompt, nseg in stories:
            # This-run truth first: a job counts only if THIS run launched it, i.e. a
            # joblog exists in this run_dir. Otherwise ignore any per-story manifest --
            # it may be a straggler from a previous run writing a fresh-mtime FAIL/DONE
            # into the shared (non-stamp-scoped) output dir, which must not mask or
            # pollute this run's state.
            joblog = logdir / f"{system}__{story_id}.log"
            if not joblog.is_file():
                # Not launched this run. It may still be a genuinely completed story
                # that --resume intentionally skipped: credit a durable status=done
                # manifest (authoritative -- the video exists) as DONE regardless of
                # mtime (done is done across runs). Anything else -> QUEUED; never
                # surface a stale FAIL from a prior run for an unlaunched story.
                mstate, _mrc = manifest_state(system, story_id)
                if mstate == "DONE":
                    state = "DONE"
                elif is_live(system, story_id, nseg, segprog_g, seg_ts_g):
                    # Live under a DIFFERENT orchestrator run_dir (split backfill):
                    # heartbeat / final-decode / loading -> RUN, not QUEUED.
                    state = "RUN"
                else:
                    state = "QUEUED"
                rows[system].append(
                    {"story_id": story_id, "nseg": nseg, "state": state, "rc": 0}
                )
                continue
            txt = joblog.read_text(encoding="utf-8", errors="replace")
            m = _EXIT_RE.search(txt)
            if m:
                rc = int(m.group(1))
                # The jobsh EXIT sentinel can be a FALSE 0: a child killed by a
                # signal returns rc=-9, and a buggy max(0,-9) aggregation used to
                # emit EXIT:0 even though nothing was produced. The durable manifest
                # records status from the TRUE child rc *and* whether the artifact
                # exists, so it is authoritative when present -- cross-check it.
                mstate, mrc = manifest_state(system, story_id, started)
                if mstate == "DONE":
                    state, rc = "DONE", 0
                elif mstate == "FAIL":
                    state, rc = "FAIL", (mrc or (rc if rc != 0 else 1))
                else:
                    state = "DONE" if rc == 0 else "FAIL"
            else:
                # Launched, no EXIT sentinel yet -> running. Corroborate a genuine
                # completion only from a manifest written by THIS run (mtime >= start).
                mstate, mrc = manifest_state(system, story_id, started)
                state, rc = ("DONE", 0) if mstate == "DONE" else ("RUN", 0)
            rows[system].append(
                {"story_id": story_id, "nseg": nseg, "state": state, "rc": rc}
            )

    rows["memstrata"] = memstrata_rows(stories)

    # Credit a RUN job's completed segments to the segment column. Without this the column mixes two
    # meanings across systems: memstrata reads its own per-shot progress file and so reports partial
    # work, while a baseline's row would count only whole finished stories — and the two rows sitting
    # side by side invite exactly the comparison they cannot support.
    segprog = load_seg_progress_global()
    for system in MANAGED:
        for row in rows[system]:
            if row["state"] == "RUN":
                live = live_inflight(system, row["story_id"], row["nseg"], segprog)
                row["seg_done"] = live[0] if live else 0
            elif row["state"] == "DONE":
                row["seg_done"] = row["nseg"]
            else:
                row["seg_done"] = 0
    return stories, rows


# A memstrata story is driven by its own fleet rather than this orchestrator, so it has no joblog and
# no EXIT sentinel. Its per-shot progress file is the equivalent contract, and reading it (instead of
# only the end-of-run manifest) is what makes partial progress visible while a story is still running.
MEMSTRATA_TAG_GLOBS = ("prod_*", "en_steps*")
MEMSTRATA_FRESH_S = 25 * 60


def memstrata_progress(run_dir: Path) -> tuple[int, int, list[int]] | None:
    """``(closed_shots, total_shots, interior_gaps)`` for one memstrata run directory."""

    progress = run_dir / "progress.json"
    try:
        data = json.loads(progress.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    total = int(data.get("total") or 0)
    ids = set()
    for record in data.get("segments") or []:
        if isinstance(record, dict):
            try:
                ids.add(int(record["segment_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not ids:
        return (0, total, [])
    # Interior only: shots above the highest produced one are simply not reached yet. A gap BELOW it is
    # a shot the generator could not produce, which leaves the story misaligned against its prompt
    # stream, so it is reported separately from plain progress.
    gaps = [i for i in range(max(ids)) if i not in ids]
    return (len(ids), total, gaps)


def memstrata_rows(stories: list[tuple[str, Path, int]]) -> list[dict]:
    """One row per SCHEDULED story, so the segment denominator matches the other systems.

    Keyed on the scheduled set rather than on the directories that happen to exist: a story that has
    not been handed a card yet has no progress file, and dropping it would shrink the denominator and
    report a higher completion percentage than the run has actually reached.
    """

    mem_root = OUT_ROOT / "memstrata"
    out: list[dict] = []
    for story_id, _prompts, nseg in stories:
        anchored = mem_root / story_id / "name_anchored"
        runs = [d for pattern in MEMSTRATA_TAG_GLOBS for d in anchored.glob(pattern)
                if (d / "progress.json").is_file()] if anchored.is_dir() else []
        if not runs:
            out.append({"story_id": story_id, "nseg": nseg, "state": "QUEUED", "rc": 0,
                        "seg_done": 0, "gaps": []})
            continue
        run_dir = max(runs, key=lambda d: (d / "progress.json").stat().st_mtime)
        prog = memstrata_progress(run_dir)
        if prog is None:
            out.append({"story_id": story_id, "nseg": nseg, "state": "QUEUED", "rc": 0,
                        "seg_done": 0, "gaps": []})
            continue
        closed, total, gaps = prog
        fresh = (time.time() - (run_dir / "progress.json").stat().st_mtime) < MEMSTRATA_FRESH_S
        total = total or nseg
        complete = bool(total) and closed >= total and not gaps
        state = "DONE" if complete else ("RUN" if fresh else "QUEUED")
        out.append({"story_id": story_id, "nseg": nseg or total, "state": state, "rc": 0,
                    "seg_done": closed, "gaps": gaps, "run_tag": run_dir.name})
    return out


def summarize(rows: list[dict]) -> dict:
    counts = {k: 0 for k in STATE_KEYS}
    seg_done = seg_total = 0
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
        seg_total += r["nseg"]
        # A row that reports its own closed-shot count is trusted over "all or nothing", so a system
        # whose progress is readable mid-story shows real progress instead of zero until it finishes.
        if "seg_done" in r:
            seg_done += int(r["seg_done"])
        elif r["state"] == "DONE":
            seg_done += r["nseg"]
    return {"counts": counts, "seg_done": seg_done, "seg_total": seg_total}


SCORE_SUBDIR = "_trackB_score"


def _memstrata_run_dir(story_id: str) -> Path | None:
    """Newest memstrata prod_/en_steps run dir with a progress.json (matches scoring)."""
    anch = OUT_ROOT / "memstrata" / story_id / "name_anchored"
    if not anch.is_dir():
        return None
    runs = [d for pat in MEMSTRATA_TAG_GLOBS for d in anch.glob(pat)
            if (d / "progress.json").is_file()]
    return max(runs, key=lambda d: (d / "progress.json").stat().st_mtime) if runs else None


def scoring_progress(stories: list, rows: dict) -> dict:
    """Per-system scored-story / scored-segment counts from _trackB_score/score.json.

    Targeted reads only (one score.json stat/read per DONE story). ``done`` is the
    scoreable denominator (generation-complete stories); ``scored`` counts those whose
    score.json exists; ``segs`` sums summary.n_segments_scored. This mirrors what the
    scoring driver actually processes so the monitor tracks it live.
    """
    out: dict[str, dict] = {}
    for system in SYSTEMS:
        done_runs: list[Path] = []
        if system == "memstrata":
            for r in memstrata_rows(stories):
                if r["state"] == "DONE" and r.get("run_tag"):
                    done_runs.append(OUT_ROOT / "memstrata" / r["story_id"]
                                     / "name_anchored" / r["run_tag"])
        else:
            for r in rows.get(system, []):
                if r["state"] == "DONE":
                    d = ORCH.out_dir_for(system, r["story_id"])
                    if d.is_dir():
                        done_runs.append(d)
        scored = 0
        segs = 0
        for run in done_runs:
            sc = run / SCORE_SUBDIR / "score.json"
            if not (sc.is_file() and sc.stat().st_size > 0):
                continue
            try:
                summ = json.loads(sc.read_text(encoding="utf-8")).get("summary") or {}
            except (OSError, ValueError):
                continue
            scored += 1
            segs += int(summ.get("n_segments_scored") or 0)
        out[system] = {"done": len(done_runs), "scored": scored, "segs": segs}
    return out


def load_snapshot(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fmt_delta(cur: int, prev: int | None) -> str:
    if prev is None:
        return f"{cur} (+0)"
    return f"{cur} ({cur - prev:+d})"


def fmt_delta_compact(cur: int, prev: int | None) -> str:
    """Tight '<v>(+<d>)' form for in-line table cells (delta always shown)."""
    d = 0 if prev is None else cur - prev
    return f"{cur}({d:+d})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="orchestrator run dir (default: _stage1_multinode/latest)")
    ap.add_argument("--limit-stories", type=int, default=30,
                    help="story cutoff K used by the run (defines the scheduled set)")
    ap.add_argument("--detail", action="store_true", help="print per-story matrix")
    ap.add_argument("--gpu", action="store_true",
                    help="probe per-node GPU utilization (confirms silent runners are alive)")
    ap.add_argument("--cluster", default="gpu-a800")
    ap.add_argument("--nodes", default="0,1,2,3,4", help="tgpu node ids for --gpu probe")
    ap.add_argument("--no-update-snapshot", action="store_true")
    args = ap.parse_args()

    run_dir = latest_run_dir(args.run_dir)
    if run_dir is None:
        print("no Track B multi-node run dir found under", OUT_ROOT / "_stage1_multinode")
        return 1

    stories, rows = collect(run_dir, args.limit_stories)
    segprog = load_seg_progress_global()
    started = run_started_at(run_dir)
    now = time.time()
    elapsed = (now - started) if started else None

    snap_path = run_dir / "progress_detail_snapshot.json"
    prev = load_snapshot(snap_path)
    prev_seg = (prev or {}).get("seg_done_by_system", {})
    prev_done_ct = (prev or {}).get("done_count_by_system", {})

    print(f"Track B Stage-1 monitor | run={run_dir.name}")
    print(f"generated_at={datetime.now(timezone.utc).isoformat(timespec='seconds')}"
          + (f" | elapsed={elapsed/3600:.2f}h" if elapsed else ""))
    print("system         DONE(+d)/RUN/QUEUED/FAIL/UNSCHED   segDONE(+d)/segTOTAL   done%")
    print("-" * 82)

    total_done = total_total = 0
    snap_seg_done, snap_done_ct = {}, {}
    for system in SYSTEMS:
        s = summarize(rows[system])
        c = s["counts"]
        reserved = system == "memstrata" and s["seg_total"] == 0 and c["DONE"] == 0
        done_cell = fmt_delta_compact(c["DONE"], prev_done_ct.get(system))
        state_str = "/".join([done_cell] + [str(c[k]) for k in STATE_KEYS[1:]])
        pct = (100.0 * s["seg_done"] / s["seg_total"]) if s["seg_total"] else 0.0
        seg_cell = f"{fmt_delta_compact(s['seg_done'], prev_seg.get(system))}/{s['seg_total']}"
        tag = "  (reserved)" if reserved else ""
        print(f"{system:<14} {state_str:<32}  {seg_cell:<21}  {pct:5.1f}%{tag}")
        snap_seg_done[system] = s["seg_done"]
        snap_done_ct[system] = c["DONE"]
        if system in MANAGED:
            total_done += s["seg_done"]
            total_total += s["seg_total"]

    print("-" * 82)
    gpct = (100.0 * total_done / total_total) if total_total else 0.0
    print(f"{'TOTAL(managed)':<14} {'':<32}  {f'{total_done}/{total_total}':<21}  {gpct:5.1f}%")

    # ---- Stage-2 scoring progress (live) ----
    # scored stories / generation-complete stories, and scored segments, per system.
    # Deltas are vs the previous snapshot so `watch` shows movement each refresh.
    score = scoring_progress(stories, rows)
    prev_score = (prev or {}).get("scored_by_system", {})
    prev_sseg = (prev or {}).get("scored_segs_by_system", {})
    print("\nStage-2 scoring: scored/done stories   scoredSeg(+d)")
    snap_scored, snap_sseg = {}, {}
    tot_sc = tot_dn = tot_sg = 0
    for system in SYSTEMS:
        s = score.get(system, {"done": 0, "scored": 0, "segs": 0})
        seg_cell = fmt_delta_compact(s["segs"], prev_sseg.get(system))
        print(f"  {system:<13} {s['scored']:>3}/{s['done']:<3} stories        {seg_cell}")
        snap_scored[system] = s["scored"]
        snap_sseg[system] = s["segs"]
        tot_sc += s["scored"]; tot_dn += s["done"]; tot_sg += s["segs"]
    print(f"  {'TOTAL':<13} {tot_sc:>3}/{tot_dn:<3} stories        {tot_sg} seg scored")

    # Throughput-based ETA. Counting only fully-completed stories wildly
    # overestimates ETA during the first wave (35 jobs start together, so for the
    # first ~1h almost everything is in-flight and little is "done"). Include
    # in-flight segments (heartbeat / iamflow log fallback) so the rate reflects
    # real throughput; it converges as stories complete.
    # ``total_done`` already carries in-flight segments (see collect), so adding them again here would
    # double count them and halve the ETA.
    effective = total_done
    if elapsed and effective > 0 and total_done < total_total:
        rate = effective / elapsed  # segments/sec (produced incl in-flight)
        eta = (total_total - effective) / rate if rate > 0 else None
        if eta:
            print(f"ETA ~{eta/3600:.1f}h  (throughput {rate*3600:.0f} seg/h incl in-flight; "
                  f"early-run estimate, converges as the first wave completes)")

    # Live in-flight segments (from the unified heartbeat file) for ALL systems.
    # This moves in real time as each segment of each RUN job completes, so you no
    # longer have to wait for a whole story to finish to see progress.
    live_now_total = live_cap_total = 0
    live_lines = []
    for system in SYSTEMS:
        if system == "memstrata":
            pairs = [(int(r["seg_done"]), int(r["nseg"]))
                     for r in rows.get(system, []) if r["state"] == "RUN"]
        else:
            pairs = [
                live_inflight(system, r["story_id"], r["nseg"], segprog)
                for r in rows.get(system, []) if r["state"] == "RUN"
            ]
        pairs = [x for x in pairs if x]
        if not pairs:
            continue
        seg_now = sum(a for a, _ in pairs)
        seg_cap = sum(b for _, b in pairs)
        live_now_total += seg_now
        live_cap_total += seg_cap
        live_lines.append(f"  {system:<13} {len(pairs)} RUN jobs  {seg_now}/{seg_cap} seg in-flight")
    if live_lines:
        seg_live = load_snapshot(snap_path) or {}
        prev_live = seg_live.get("live_now_total")
        d = "" if prev_live is None else f" ({live_now_total - prev_live:+d})"
        print(f"live in-flight segments (heartbeat): {live_now_total}/{live_cap_total}{d}")
        for ln in live_lines:
            print(ln)
    elif not segprog:
        print("live in-flight: (this run predates per-segment heartbeat; restart to "
              "enable real-time seg progress. iamflow still shows via its own log.)")

    if args.gpu:
        nds = [int(x) for x in args.nodes.split(",") if x.strip()]
        print("\nper-node VRAM used% (idx order):")
        for nd, util in gpu_snapshot(args.cluster, nds).items():
            print(f"  node {nd}: {util}")

    if args.detail:
        mark = {"DONE": "D", "RUN": "R", "QUEUED": "q", "FAIL": "F", "UNSCHED": "-"}
        print("\nper-story (ms=memstrata, mem=memflow, ll=longlive_rag, iam=iamflow):")
        print(f"{'story_id':<32} {'nseg':>5}  ms  mem ll  iam   RUN heartbeat")
        by_story = {sid: {"nseg": n} for sid, _p, n in stories}
        for system in SYSTEMS:
            for r in rows[system]:
                by_story.setdefault(r["story_id"], {"nseg": r["nseg"]})[system] = r
        for sid in [s for s, _p, _n in stories]:
            info = by_story.get(sid, {})
            cells = " ".join(
                f"{mark.get((info.get(sys_) or {}).get('state', 'UNSCHED'), '?'):>3}"
                for sys_ in SYSTEMS
            )
            hb_parts = []
            ms = info.get("memstrata") or {}
            if ms and ms.get("state") != "DONE":
                gaps = ms.get("gaps") or []
                hb_parts.append(f"memstrata:{ms.get('seg_done', 0)}/{ms.get('nseg', 0)}"
                                + (f" gaps@{gaps}" if gaps else ""))
            elif ms.get("gaps"):
                hb_parts.append(f"memstrata:gaps@{ms['gaps']}")
            for sys_ in MANAGED:
                r = info.get(sys_) or {}
                if r.get("state") != "RUN":
                    continue
                live = live_inflight(sys_, sid, int(r.get("nseg", 0)), segprog)
                if live is not None:
                    hb_parts.append(f"{sys_}:{live[0]}/{live[1]}")
                else:
                    hb_parts.append(f"{sys_}:silent")
            print(f"{sid:<32} {info.get('nseg', 0):>5}  {cells}   {'  '.join(hb_parts)}")

    if not args.no_update_snapshot:
        snap = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run": run_dir.name,
            "seg_done_by_system": snap_seg_done,
            "done_count_by_system": snap_done_ct,
            "live_now_total": live_now_total,
            "scored_by_system": snap_scored,
            "scored_segs_by_system": snap_sseg,
        }
        tmp = snap_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(snap_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
