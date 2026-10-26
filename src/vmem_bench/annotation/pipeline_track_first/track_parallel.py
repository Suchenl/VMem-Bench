"""Multi-GPU per-shot tracklet computation for the track-first pipeline.

Per-shot ``track_shot`` (GroundingDINO detect + DINOv3 crop embed + optional ArcFace) is the
dominant GPU cost and is INDEPENDENT across shots, so we shard shots across GPUs. On one GPU the
detector serializes on its own lock, so thread-level "parallelism" over one card is a no-op; real
speedup needs one model instance PER card. Each worker process therefore pins itself to one card
(CUDA_VISIBLE_DEVICES), loads its own in-process perception stack once, and drains a shared shot
queue -- true N-way parallelism across N cards. Perception models are load-once-per-worker singletons
(the persistent-model-serving rule classes gdino/dino/face as in-process singletons), reused across
all of that worker's shots.

Cross-shot ordering (re-ID, entity ids) stays deterministic: tracklets are pure per-shot data, and
the pipeline renumbers track ids in shot order after all shots are computed, so the parallel result
is bit-identical to the sequential one. Each computed shot is checkpointed (resume.save_shot), so
this module also IS the tracking-phase resume: a re-run skips shots whose checkpoint exists.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Callable
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first import resume
from vmem_bench.annotation.pipeline_track_first.perception.base import Frame, RosterEntry
from vmem_bench.common.media import extract_frame, sample_frame_indices

logger = logging.getLogger(__name__)


def make_frame_path(video: Path, frames_dir: Path, fps: float, n_frames: int,
                    extract_fn: Callable | None = None) -> Callable[[int], Path]:
    """Frame extractor with the same tail-guard as the pipeline (clamp to range, step back a few
    frames if ffmpeg cannot decode the very tail). Shared by the pipeline and the workers so both
    decode/cache frames identically into ``frames_dir`` (idempotent; safe under concurrent workers,
    which only ever touch disjoint shots -> disjoint frame indices)."""
    _extract = extract_fn or extract_frame

    def _frame_path(i: int) -> Path:
        if n_frames:
            i = max(0, min(i, n_frames - 1))
        p = frames_dir / f"f{i:07d}.jpg"
        if p.is_file():
            return p
        last_err: Exception | None = None
        for j in range(i, max(-1, i - 4), -1):
            try:
                return _extract(video, p, frame_index=j, fps=fps)
            except Exception as exc:  # noqa: BLE001 (tail-frame decode; step back and retry)
                last_err = exc
        raise last_err  # type: ignore[misc]

    return _frame_path


def compute_shot(shot_idx: int, first: int, last: int, *, config, roster_entries, backend,
                 embedder, face_encoder, frame_path: Callable[[int], Path], fps: float) -> dict:
    """All GPU work for one shot -> a JSON-able checkpoint payload (tracklets + per-tracklet face
    signature + scene vector). Face + scene embedding are done here (not in the sequential re-ID
    pass) so every GPU op of a shot runs inside its worker; re-ID then stays pure CPU."""
    n = max(1, int(round((last - first + 1) / fps * config.track_fps)))
    idxs = sample_frame_indices(first, last + 1, max_samples=max(n, config.track_min_len))
    frames = [Frame(frame_index=i, path=frame_path(i)) for i in idxs]
    tracklets = backend.track_shot(frames, roster_entries, next_track_id=0)
    by_phrase = {e.grounding_phrase: e for e in roster_entries}

    face_sigs: list[list[float] | None] = []
    for tk in tracklets:
        entry = by_phrase.get(tk.phrase)
        kind = entry.kind if entry else "prop"
        best = tk.best_detection()
        fs = None
        if (config.use_face and kind == "character" and face_encoder is not None
                and best.crop_path):
            vec = face_encoder.encode(Path(best.crop_path))
            fs = list(vec) if vec is not None else None
        face_sigs.append(fs)

    scene_frame = idxs[len(idxs) // 2]
    scene_vec = list(embedder.embed_image(frame_path(scene_frame)))
    return resume.shot_payload(shot_idx, first, last, tracklets, face_sigs, scene_frame, scene_vec)


def _roster_entries_from_dicts(data: list[dict]) -> list[RosterEntry]:
    return [RosterEntry(name=d["name"], kind=d["kind"], grounding_phrase=d["grounding_phrase"],
                        static_attributes=dict(d.get("static_attributes") or {}),
                        exemplar_crop=str(d.get("exemplar_crop") or ""),
                        canonical_entity_id=str(d.get("canonical_entity_id") or ""),
                        identity_scope=str(d.get("identity_scope") or "individual"),
                        aliases=tuple(d.get("aliases") or ()),
                        exemplar_crops=tuple(d.get("exemplar_crops") or ()),
                        allowed_state_events=tuple(d.get("allowed_state_events") or ()))
            for d in data]


def _worker(device: int, task_q, done_q, config, roster_data: list[dict], out_str: str,
            video_str: str, frames_dir_str: str, crop_dir_str: str, fps: float,
            n_frames: int) -> None:
    """One GPU worker: pin to ``device``, load perception once, drain shots, checkpoint each."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    from vmem_bench.annotation.pipeline_track_first.perception.factory import build_inprocess_perception
    roster_entries = _roster_entries_from_dicts(roster_data)
    backend, embedder, face_encoder = build_inprocess_perception(config, Path(crop_dir_str))
    frame_path = make_frame_path(Path(video_str), Path(frames_dir_str), fps, n_frames)
    out = Path(out_str)
    while True:
        item = task_q.get()
        if item is None:
            break
        si, first, last = item
        try:
            payload = compute_shot(si, first, last, config=config, roster_entries=roster_entries,
                                   backend=backend, embedder=embedder, face_encoder=face_encoder,
                                   frame_path=frame_path, fps=fps)
            resume.save_shot(out, payload)
            done_q.put((si, True, "", len(payload.get("tracklets", []))))
        except Exception as exc:  # noqa: BLE001 (one bad shot must not kill the worker)
            logger.exception("shot %d failed on gpu %s", si, device)
            done_q.put((si, False, str(exc), 0))


def compute_tracklets(shots, todo: list[int], *, config, roster_entries, roster_data: list[dict],
                      out: Path, video: Path, frames_dir: Path, crop_dir: Path, fps: float,
                      n_frames: int, devices: list[int], backend=None, embedder=None,
                      face_encoder=None, frame_path=None,
                      progress: Callable[[int], None] | None = None) -> list[tuple[int, str]]:
    """Compute + checkpoint tracklets for the shot indices in ``todo`` (already-checkpointed shots
    are the caller's job to exclude). ``len(devices) <= 1`` runs in-process with the injected models
    (single-GPU / test path); otherwise one spawn worker per device runs in parallel. ``progress``
    is called with the cumulative done count plus, when known, the finished shot's index and its
    tracklet count (kwargs ``shot_idx``/``n_tracklets``) so live progress can show real yield.
    Returns [(shot_idx, error), ...] for failed shots."""
    failures: list[tuple[int, str]] = []
    if not todo:
        return failures

    def _notify(done: int, si: int, n_tracklets: int) -> None:
        if not progress:
            return
        try:
            progress(done, shot_idx=si, n_tracklets=n_tracklets)
        except TypeError:  # older positional-only callback
            progress(done)

    if len(devices) <= 1:
        fp = frame_path or make_frame_path(video, frames_dir, fps, n_frames)
        for k, si in enumerate(todo):
            first, last = shots[si]
            n_tracklets = 0
            try:
                payload = compute_shot(si, first, last, config=config,
                                       roster_entries=roster_entries, backend=backend,
                                       embedder=embedder, face_encoder=face_encoder,
                                       frame_path=fp, fps=fps)
                resume.save_shot(out, payload)
                n_tracklets = len(payload.get("tracklets", []))
            except Exception as exc:  # noqa: BLE001
                logger.exception("shot %d failed (in-process)", si)
                failures.append((si, str(exc)))
            _notify(k + 1, si, n_tracklets)
        return failures

    # Multi-GPU: spawn (never fork: fork + CUDA in the parent = corrupt contexts) one worker/card.
    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue()
    done_q: mp.Queue = ctx.Queue()
    for si in todo:
        task_q.put((si, *shots[si]))
    for _ in devices:
        task_q.put(None)  # one stop sentinel per worker
    procs = [ctx.Process(target=_worker,
                         args=(dev, task_q, done_q, config, roster_data, str(out), str(video),
                               str(frames_dir), str(crop_dir), fps, n_frames))
             for dev in devices]
    for p in procs:
        p.start()
    done = 0
    while done < len(todo):
        result = done_q.get()
        si, ok, err = result[0], result[1], result[2]
        n_tracklets = int(result[3]) if len(result) > 3 else 0
        done += 1
        if not ok:
            failures.append((si, err))
        _notify(done, si, n_tracklets)
    for p in procs:
        p.join()
    return failures
