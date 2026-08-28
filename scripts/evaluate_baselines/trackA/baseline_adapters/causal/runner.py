"""Stage-1 driver: run one causal baseline over a movie under the new protocol.

Per segment, in time order:
  1. cut the real segment clip from the source video (``seconds_span``);
  2. ``compose`` -- hand the SUT the prompt; it retrieves from CURRENT memory
     (built only from earlier segments) and returns temporal items;
  3. ``observe_segment`` -- hand the SUT the real segment; it updates its memory.

Step 2 strictly precedes step 3 for the same segment, so the SUT cannot peek at
segment t's video while composing segment t's context. Retrieved memory is then
rendered into real frames + a scorer manifest by :mod:`frame_materializer`.

The bench does no perception, produces no crops, and passes no gold to the SUT.

Input modes (fairness axis, reported side by side -- see
``docs/experiments/fairness_experiment_plan.md``):

* ``name_anchored`` (default, main table): the prompt is the S4 screenplay prose
  verbatim; recurring entities are referred to by their natural names. Systems
  that index memory by name (e.g. MemStrata's name-anchoring, and to a degree the
  text-conditioned baselines) get a strong textual handle here.
* ``description_provided``: same name-anchored prompt PLUS a deterministic
  appearance-description suffix appended for every entity **whose name already
  appears in that prompt**. This ADDS visual-appearance text (it does not strip
  names), so systems that match a described appearance to their stored visual
  memory get a fair textual handle too. Leak-safe: we describe only entities the
  prompt already names, so no ``present``/roster answer is revealed; the same
  deterministic rule is applied identically to every system.
* ``description_only``: deterministic neutral references plus appearance text
  for entities whose names already occur in the prompt. Empty descriptions are
  skipped/fallback-neutralized; no ``present``/roster field is exposed.

The runner also owns retrieval budget governance: every adapter's returned items
are score-sorted and clipped to ``--budget`` (B in {1,2,4,8,16}); adapters may
return ranked lists in ``extras["rrf_rankings"]`` for RRF fusion (k=60) before
the same clipping.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from urllib import error as urllib_error
from urllib import request as urllib_request
from pathlib import Path

from contract import ComposeRequest, MovieContext, RetrievedItem, SegmentObservation
from frame_materializer import materialize_record_checkpoint, materialize_system
from _local_roots import expand_dataset_root

_DEFAULT_FFMPEG = "ffmpeg"
_HOSTNAME = socket.gethostname()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[stage1][warn] invalid {name}={raw!r}; using {default}",
              file=sys.stderr, flush=True)
        return default
_BENCH_ROOT = Path(__file__).resolve().parents[5]  # public VMem-Bench checkout
_INPUT_MODES = ("name_anchored", "description_provided", "description_only")
_BUDGET_CHOICES = (1, 2, 4, 8, 16)
_RRF_K = 60
_KEEPALIVE_STATUS_DIR = Path(os.environ["VMEM_KEEPALIVE_STATUS_DIR"]) if os.environ.get("VMEM_KEEPALIVE_STATUS_DIR") else None


def _apply_runtime_safety_defaults() -> None:
    """Cap BLAS/tokenizers threads so a worker does not pin the whole node."""
    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MAX_JOBS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    nice_delta = int(os.environ.get("VMEM_TASK_NICE", "10"))
    if nice_delta > 0:
        try:
            os.nice(nice_delta)
        except OSError:
            pass

    try:
        subprocess.run(
            ["ionice", "-c", "2", "-n", "7", "-p", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        pass


def _require_a800_keepalive_if_requested() -> None:
    mode = os.environ.get("VMEM_REQUIRE_A800_KEEPALIVE", "auto").lower()
    if mode in {"0", "false", "no"}:
        return
    if _KEEPALIVE_STATUS_DIR is None:
        if mode == "1":
            raise SystemExit("VMEM_REQUIRE_A800_KEEPALIVE=1 but VMEM_KEEPALIVE_STATUS_DIR is unset")
        return
    host = subprocess.check_output(["hostname"], text=True).strip()
    status_path = _KEEPALIVE_STATUS_DIR / f"{host}.status"
    if mode != "1":
        if not status_path.is_file():
            return
    if not status_path.is_file():
        raise SystemExit(
            "VMEM_REQUIRE_A800_KEEPALIVE=1 but keepalive status is missing: "
            f"{status_path}"
        )
    status = status_path.read_text(encoding="utf-8", errors="replace")
    if "alive_gpu_processes=8/8" not in status:
        raise SystemExit(
            "VMEM_REQUIRE_A800_KEEPALIVE=1 but keepalive is not healthy: "
            f"{status_path}"
        )


def _require_iamflow_http_services(adapter_name: str) -> None:
    """Fail fast if an IAMFlow runner is placed on a node without local services."""
    if adapter_name != "iamflow":
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    service_gpus = {
        gpu
        for gpu in (
            os.environ.get("IAMFLOW_LLM_SERVICE_GPU"),
            os.environ.get("IAMFLOW_VLM_SERVICE_GPU"),
        )
        if gpu
    }
    if visible and visible in service_gpus:
        print(
            json.dumps(
                {
                    "error": "iamflow_runner_on_service_gpu",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "service_gpus": sorted(service_gpus),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(35)

    endpoints = {
        "llm": os.environ.get("IAMFLOW_LLM_ENDPOINT", "http://127.0.0.1:8100/v1"),
        "vlm": os.environ.get("IAMFLOW_VLM_ENDPOINT", "http://127.0.0.1:8101/v1"),
    }
    errors: dict[str, str] = {}
    for role, base_url in endpoints.items():
        url = base_url.rstrip("/") + "/models"
        try:
            req = urllib_request.Request(url, headers={"Accept": "application/json"})
            with urllib_request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
                if resp.status >= 500:
                    errors[role] = f"{url} returned HTTP {resp.status}"
        except (OSError, urllib_error.URLError, TimeoutError) as exc:
            errors[role] = f"{url}: {exc}"

    if errors:
        print(
            json.dumps(
                {
                    "error": "iamflow_http_service_not_ready",
                    "message": (
                        "IAMFlow runners require local LLM/VLM HTTP services on "
                        "the same node before loading the VAE or starting Stage-1."
                    ),
                    "endpoints": endpoints,
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(34)


def _resolve_source_video(movie_dir: Path) -> Path:
    """Resolve the real source video via data/dataset_dirs.txt (see running_eval §7)."""
    dataset_dirs = _BENCH_ROOT / "assets" / "trackA" / "dataset_dirs.txt"
    roots: dict[str, str] = {}
    if dataset_dirs.is_file():
        for line in dataset_dirs.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            roots[key.strip()] = val.strip()
    dataset = movie_dir.parent.name
    movie_id = movie_dir.name
    raw_root = roots.get(dataset)
    if not raw_root:
        raise SystemExit(f"dataset root not found for {dataset!r} in {dataset_dirs}")
    root = expand_dataset_root(raw_root)
    exts = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
    cand_dir = Path(root) / movie_id
    # Layout A (e.g. BlenderOpenMovies): data root/<movie_id>/<video>.
    if cand_dir.is_dir():
        vids = sorted(p for p in cand_dir.iterdir() if p.suffix.lower() in exts)
        if not vids:
            raise SystemExit(f"no video file under {cand_dir}")
        return vids[0]
    # Layout B (e.g. LSMDC stitched): flat file root/<movie_id>.<ext>.
    flat = sorted(
        p for p in Path(root).glob(f"{movie_id}.*") if p.suffix.lower() in exts
    )
    if flat:
        return flat[0]
    raise SystemExit(
        f"source video not found for {movie_id!r} under {root} "
        f"(tried dir {cand_dir} and flat {Path(root) / (movie_id + '.<ext>')})"
    )


def _load_layout(movie_dir: Path) -> tuple[dict[int, tuple[float, float]], dict[int, str]]:
    ca = json.loads((movie_dir / "gold/chunk_annotations.json").read_text(encoding="utf-8"))
    spans: dict[int, tuple[float, float]] = {}
    prompts: dict[int, str] = {}
    for c in ca["chunks"]:
        cid = int(c["chunk_id"])
        span = c.get("seconds_span")
        if span:
            spans[cid] = (float(span[0]), float(span[1]))
        prompts[cid] = c.get("prompt", "") or ""
    return spans, prompts


def _selection_complete(path: Path, expected_chunks: int) -> bool:
    """Whether a prior Stage-1 output is complete enough to skip safely."""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    chunks = data.get("chunks") if isinstance(data, dict) else None
    return isinstance(chunks, list) and len(chunks) >= int(expected_chunks)


def _summary_selection_complete(summary: dict) -> bool:
    """Whether a run_movie summary points to a durable complete Stage-1 output."""
    system = summary.get("system")
    dataset = summary.get("dataset")
    movie = summary.get("movie")
    if not system or not dataset or not movie:
        return False
    chunks = (
        _BENCH_ROOT
        / "assets"
        / "trackA"
        / str(dataset)
        / str(movie)
        / "gold"
        / "chunk_annotations.json"
    )
    if not chunks.is_file():
        return False
    try:
        expected = len(json.loads(chunks.read_text(encoding="utf-8"))["chunks"])
    except Exception:
        return False
    selection = (
        _BENCH_ROOT
        / "outputs"
        / "evaluation"
        / "trackA"
        / str(system)
        / str(dataset)
        / str(movie)
        / "visual_selections"
        / f"{system}.json"
    )
    return _selection_complete(selection, expected)


def _lock_owner_is_alive(path: Path) -> bool:
    """Whether the process that holds ``path`` still looks alive.

    Two failure modes have to be told apart. A runner killed by the cgroup OOM
    killer never runs its cleanup, so it leaves a lock that would block the movie
    forever. A runner that is merely slow (a single memflow_sma segment can take
    165 s) must keep its lock, or a "clean up stale locks" pass starts a second
    runner on the same movie and both burn a GPU on identical work.

    So: trust a same-host pid probe when we can, and otherwise fall back to the
    heartbeat that ``_touch_job_lock`` refreshes after every segment.
    """
    stale_after = _env_float("VMEM_STAGE1_LOCK_STALE_MINUTES", 45.0) * 60.0
    try:
        meta = dict(
            token.split("=", 1)
            for token in path.read_text(encoding="utf-8").split()
            if "=" in token
        )
    except OSError:
        return False
    if "host" not in meta:
        # Written by a runner that predates the heartbeat, so its mtime is just
        # the creation time and says nothing about liveness. Assume alive: the
        # fleet has memflow_sma runs that legitimately hold a lock for 8+ hours,
        # and stealing one would put two runners on the same movie.
        return True
    if meta["host"] == _HOSTNAME and meta.get("pid", "").isdigit():
        try:
            os.kill(int(meta["pid"]), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age <= stale_after


def _acquire_job_lock(path: Path) -> int | None:
    """Best-effort per-run lock for concurrent accelerators.

    Returns an open fd when acquired; returns None if another live process owns
    the run. The fd must be closed and the file unlinked by the caller.
    """
    for attempt in (0, 1):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt or _lock_owner_is_alive(path):
                return None
            # Provably dead owner (killed mid-run): reclaim instead of leaving the
            # movie permanently unrunnable and needing a manual stale-lock wave.
            print(
                f"[stage1][lock] reclaiming stale lock {path} "
                f"(owner gone or heartbeat older than "
                f"{_env_float('VMEM_STAGE1_LOCK_STALE_MINUTES', 45.0):.0f} min)",
                file=sys.stderr,
                flush=True,
            )
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        os.write(
            fd,
            f"pid={os.getpid()} host={_HOSTNAME} start={time.time():.3f}\n".encode("utf-8"),
        )
        return fd
    return None


def _touch_job_lock(path: Path) -> None:
    """Heartbeat the lock so other hosts can tell slow from dead."""
    try:
        os.utime(path, None)
    except OSError:
        pass


# description_provided describes EVERY registered entity kind (character / prop /
# location): props (a specific car, a briefcase) and locations (a specific room)
# are first-class recurring visual identities too, not noise. Leak-safety comes
# from only describing entities whose NAME already appears in that chunk's prompt
# (bounded to the few named per segment), NOT from filtering by kind. NOTE: a
# text-saliency baseline (MemFlow) can still score descprov <= name_anchored -- that
# is STRUCTURAL (its text is a write-time bank-compression cue, not a query-time
# description matcher), an honest benchmark finding, not something to "fix" by
# dropping entities.
_DESCRIBED_KINDS: set[str] | None = None  # None => all kinds


def _load_entities(movie_dir: Path) -> list[dict]:
    """Registered entities, longest-name-first.

    Bench-side metadata used ONLY to append appearance descriptions for names that
    already occur in a prompt (description_provided mode). Never handed to the SUT
    as a present/roster list. All kinds are eligible (see ``_DESCRIBED_KINDS``).
    """
    reg_p = movie_dir / "gold/entity_registry.json"
    if not reg_p.is_file():
        return []
    reg = json.loads(reg_p.read_text(encoding="utf-8"))
    ents = reg.get("entities", []) if isinstance(reg, dict) else (reg or [])
    out: list[dict] = []
    for e in ents:
        name = (e.get("name") or "").strip()
        desc = (e.get("description") or "").strip()
        kind = (e.get("kind") or "").strip().lower()
        aliases = e.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        if name and (_DESCRIBED_KINDS is None or kind in _DESCRIBED_KINDS):
            out.append({
                "name": name,
                "description": desc,
                "kind": kind,
                "aliases": [str(alias) for alias in aliases if str(alias).strip()],
            })
    # Longest name first so substring checks prefer the most specific name.
    out.sort(key=lambda x: -len(x["name"]))
    return out


def _term_in_prompt(prompt: str, term: str) -> int:
    term = str(term or "").strip()
    if not term:
        return -1
    if any("\u4e00" <= char <= "\u9fff" for char in term):
        return prompt.casefold().find(term.casefold())
    pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
    m = re.search(pattern, prompt, re.IGNORECASE)
    return -1 if m is None else int(m.start())


def _neutral_label(kind: str, *, chinese: bool) -> str:
    key = str(kind or "").lower()
    if chinese:
        if key == "character":
            return "那个人"
        if key == "location":
            return "那个地方"
        return "那个物体"
    if key == "character":
        return "the person"
    if key == "location":
        return "the place"
    return "the object"


def _apply_description_provided(prompt: str, entities: list[dict]) -> str:
    """Append a deterministic appearance suffix for entities NAMED in this prompt.

    Only entities whose name is a substring of ``prompt`` are described, so the
    added text never reveals presence beyond the names the prompt already carries.
    Deterministic: matched entities are ordered by first occurrence in the prompt.
    """
    if not prompt or not entities:
        return prompt
    matched: list[tuple[int, str, str]] = []
    for ent in entities:
        name = ent["name"]
        desc = ent.get("description") or ""
        if not desc:
            continue
        idx = _term_in_prompt(prompt, name)
        if idx >= 0:
            matched.append((idx, name, desc))
    if not matched:
        return prompt
    matched.sort(key=lambda x: x[0])
    suffix = "；".join(f"{name}：{desc.rstrip('。')}" for _, name, desc in matched)
    return f"{prompt}\n\n[实体外观参考] {suffix}。"


def _replace_name_terms(prompt: str, entities: list[dict]) -> str:
    out = prompt
    chinese = any("\u4e00" <= ch <= "\u9fff" for ch in prompt)
    for ent in entities:
        terms = [ent.get("name") or "", *(ent.get("aliases") or [])]
        label = _neutral_label(ent.get("kind") or "", chinese=chinese)
        for term in sorted({str(t).strip() for t in terms if str(t).strip()}, key=len, reverse=True):
            if any("\u4e00" <= char <= "\u9fff" for char in term):
                out = out.replace(term, label)
            else:
                pat = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
                out = re.sub(pat, label, out, flags=re.IGNORECASE)
    return out


def _apply_description_only(prompt: str, entities: list[dict]) -> str:
    """Leak-safe description-only prompt transform.

    The transform only uses entity names that already occur in the prompt. When a
    matched entity has no description yet, it is skipped; if nothing descriptive
    remains, the deterministic neutralized prompt is used as a fallback.
    """
    if not prompt or not entities:
        return prompt
    chinese = any("\u4e00" <= ch <= "\u9fff" for ch in prompt)
    matched: list[tuple[int, str, str]] = []
    for ent in entities:
        desc = (ent.get("description") or "").strip()
        if not desc:
            continue
        idx = _term_in_prompt(prompt, ent.get("name") or "")
        if idx >= 0:
            matched.append((idx, _neutral_label(ent.get("kind") or "", chinese=chinese), desc))
    if not matched:
        return _replace_name_terms(prompt, entities)
    matched.sort(key=lambda x: x[0])
    seen: set[tuple[str, str]] = set()
    parts: list[str] = []
    for _, label, desc in matched:
        key = (label, desc)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{label}：{desc.rstrip('。')}" if chinese else f"{label}: {desc.rstrip('.')}")
    prefix = "[外观描述]" if chinese else "[appearance description]"
    sep = "；" if chinese else "; "
    tail = "。" if chinese else "."
    return f"{prefix} {sep.join(parts)}{tail}"


def _cut_segment(ffmpeg: str, src: Path, out: Path, s0: float, s1: float) -> Path:
    if out.is_file() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_suffix(out.suffix + ".lock")
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        deadline = time.time() + 900.0
        while time.time() < deadline:
            if out.is_file() and out.stat().st_size > 0:
                return out
            time.sleep(1.0)
        raise RuntimeError(f"timed out waiting for segment cache lock: {lock}")
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp.mp4")
    dur = max(0.2, float(s1) - float(s0))
    try:
        os.write(fd, f"pid={os.getpid()} start={time.time():.3f}\n".encode("utf-8"))
        os.close(fd)
        # Downscale to the Wan backbone's native 832x480 (see _video_io.WAN_W/WAN_H).
        # read_segment_pixels re-scales to the same size anyway, so the decoded tensor
        # is identical; cutting at 480p just makes segment decode (and disk) cheaper.
        subprocess.run(
            [ffmpeg, "-y", "-ss", f"{float(s0):.3f}", "-i", str(src), "-t", f"{dur:.3f}",
             "-an", "-threads", os.environ.get("VMEM_FFMPEG_THREADS", "1"),
             "-vf", "scale=832:480", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "veryfast", str(tmp)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.replace(tmp, out)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    return out


def _item_identity(item: RetrievedItem) -> tuple:
    sec = round(float(item.source_seconds), 3) if item.source_seconds is not None else None
    if sec is not None:
        return (item.evidence_kind, sec)
    if item.source_chunk_id is not None:
        return (item.evidence_kind, "chunk", item.source_chunk_id)
    if item.latent_index is not None:
        return (item.evidence_kind, "latent", item.latent_index)
    return (item.evidence_kind, "raw", item.raw_ref)


def _rrf_fuse_rankings(rankings: list[list[RetrievedItem]], *, k: int = _RRF_K) -> list[RetrievedItem]:
    scores: dict[tuple, float] = {}
    best: dict[tuple, RetrievedItem] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            key = _item_identity(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best.setdefault(key, item)
    fused: list[RetrievedItem] = []
    for key, score in scores.items():
        item = best[key]
        fused.append(RetrievedItem(
            evidence_kind=item.evidence_kind,
            source_seconds=item.source_seconds,
            source_chunk_id=item.source_chunk_id,
            latent_index=item.latent_index,
            score=float(score),
            raw_ref=item.raw_ref,
        ))
    fused.sort(key=lambda item: (item.score is not None, item.score or float("-inf")), reverse=True)
    return fused


def _apply_runner_fusion(rec) -> None:
    rankings = rec.extras.pop("rrf_rankings", None)
    if not rankings:
        return
    rec.items = _rrf_fuse_rankings(rankings, k=int(rec.extras.get("rrf_k", _RRF_K)))
    rec.extras["fusion"] = "rrf"


def _apply_budget(rec, budget: int | None) -> None:
    if budget is None:
        return
    indexed = list(enumerate(rec.items))
    indexed.sort(
        key=lambda pair: (
            pair[1].score is not None,
            pair[1].score if pair[1].score is not None else float("-inf"),
            -pair[0],
        ),
        reverse=True,
    )
    rec.items = [item for _, item in indexed[: int(budget)]]


def _run_name(adapter_name: str, input_mode: str, budget: int | None) -> str:
    """Output/system name. Distinct per input mode so runs never overwrite."""
    if input_mode == "name_anchored":
        run_name = adapter_name
    elif input_mode == "description_provided":
        run_name = f"{adapter_name}__descprov"
    else:
        run_name = f"{adapter_name}__desconly"
    if budget is not None:
        run_name = f"{run_name}__B{int(budget)}"
    return run_name


def run_movie(adapter, movie_dir: Path, *, ffmpeg: str, fps: float, limit: int | None,
              input_mode: str = "name_anchored", budget: int | None = None) -> dict:
    if input_mode not in _INPUT_MODES:
        raise SystemExit(f"--input-mode must be one of {_INPUT_MODES}, got {input_mode!r}")
    if budget is not None and int(budget) not in _BUDGET_CHOICES:
        raise SystemExit(f"--budget must be one of {_BUDGET_CHOICES}, got {budget!r}")
    if budget is not None and hasattr(adapter, "set_budget"):
        adapter.set_budget(int(budget))
    spans, prompts = _load_layout(movie_dir)
    entities: list[dict] = []
    if input_mode == "description_provided":
        entities = _load_entities(movie_dir)
        prompts = {cid: _apply_description_provided(p, entities) for cid, p in prompts.items()}
    elif input_mode == "description_only":
        entities = _load_entities(movie_dir)
        prompts = {cid: _apply_description_only(p, entities) for cid, p in prompts.items()}
    run_name = _run_name(adapter.name, input_mode, budget)
    src = _resolve_source_video(movie_dir)
    dataset = movie_dir.parent.name
    run_dir = _BENCH_ROOT / "outputs" / "evaluation" / "trackA" / run_name / dataset / movie_dir.name
    cids = sorted(spans)
    if limit:
        cids = cids[:limit]
    selection_path = run_dir / "visual_selections" / f"{run_name}.json"
    if _selection_complete(selection_path, len(cids)):
        return {
            "system": run_name,
            "dataset": dataset,
            "movie": movie_dir.name,
            "input_mode": input_mode,
            "budget": budget,
            "skipped": True,
            "reason": "complete_visual_selection_exists",
            "visual_selection": str(selection_path),
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".stage1.lock"
    lock_fd = _acquire_job_lock(lock_path)
    if lock_fd is None:
        return {
            "system": run_name,
            "dataset": dataset,
            "movie": movie_dir.name,
            "input_mode": input_mode,
            "budget": budget,
            "skipped": True,
            "reason": "stage1_lock_exists",
            "lock": str(lock_path),
        }
    work_dir = run_dir / "_adapter_work" / run_name
    seg_dir = _BENCH_ROOT / "outputs" / "evaluation" / "trackA" / "_shared_segments" / dataset / movie_dir.name
    frames_dir = run_dir / "_ref_frames" / run_name
    try:
        work_dir.mkdir(parents=True, exist_ok=True)

        movie = MovieContext(
            movie_id=movie_dir.name,
            source_video=str(src),
            fps=fps,
            seconds_span_by_chunk=spans,
            work_dir=str(work_dir),
        )
        # Optional adapter admission check. An adapter that knows it cannot finish
        # this movie under the pod's resource budget returns a reason string here,
        # so we skip it cheaply instead of burning hours and dying mid-run.
        preflight = getattr(adapter, "preflight", None)
        if callable(preflight):
            refusal = preflight(movie)
            if refusal:
                return {
                    "system": run_name,
                    "dataset": dataset,
                    "movie": movie_dir.name,
                    "input_mode": input_mode,
                    "budget": budget,
                    "skipped": True,
                    "reason": str(refusal),
                }
        adapter.reset(movie)

        records = []
        total_chunks = len(cids)
        movie_t0 = time.perf_counter()
        for idx, cid in enumerate(cids, start=1):
            s0, s1 = spans[cid]
            print(
                f"[stage1] {run_name}/{dataset}/{movie_dir.name} segment={idx}/{total_chunks} id={cid} start",
                file=sys.stderr,
                flush=True,
            )
            seg = _cut_segment(ffmpeg, src, seg_dir / f"chunk_{cid:05d}.mp4", s0, s1)
            # 1) compose from current memory (built only from earlier segments).
            # Time the retrieval itself: this is the per-segment RETRIEVAL latency, a
            # time-efficiency metric that grows with movie length (see running_eval
            # §"time efficiency"). Excludes segment cutting (bench I/O, not the SUT).
            _t = time.perf_counter()
            rec = adapter.compose(ComposeRequest(chunk_id=cid, prompt_text=prompts.get(cid, ""),
                                                 seconds_span=(s0, s1)))
            compose_ms = round((time.perf_counter() - _t) * 1000.0, 2)
            _apply_runner_fusion(rec)
            _apply_budget(rec, budget)
            # 2) then observe this segment's real video to update memory (memory WRITE).
            _t = time.perf_counter()
            adapter.observe_segment(SegmentObservation(chunk_id=cid, segment_video=str(seg),
                                                       seconds_span=(s0, s1), fps=fps,
                                                       prompt_text=prompts.get(cid, "")))
            observe_ms = round((time.perf_counter() - _t) * 1000.0, 2)
            rec.extras["compose_ms"] = compose_ms
            rec.extras["observe_ms"] = observe_ms
            rec.extras["n_retrieved"] = len(rec.items)
            records.append(rec)
            if os.environ.get("VMEM_STAGE1_INCREMENTAL_SELECTIONS", "1").lower() not in {"0", "false", "no"}:
                materialize_record_checkpoint(
                    system=run_name,
                    movie=movie,
                    rec=rec,
                    out_dir=run_dir / "visual_selections",
                    frames_dir=frames_dir,
                    ffmpeg=ffmpeg,
                    prompts=prompts,
                    expected_chunks=total_chunks,
                )
            # Heartbeat the lock: proves to other hosts that a slow run is alive,
            # so nobody "cleans up" this lock and starts a duplicate runner.
            _touch_job_lock(lock_path)
            # ETA from this movie's own mean segment cost. Per-segment cost swings
            # 11-165 s with node co-tenancy, so an operator cannot eyeball whether
            # a long run is worth keeping; print the answer instead.
            eta_min = (time.perf_counter() - movie_t0) / idx * (total_chunks - idx) / 60.0
            print(
                f"[stage1] {run_name}/{dataset}/{movie_dir.name} segment={idx}/{total_chunks} "
                f"id={cid} done compose_ms={compose_ms} observe_ms={observe_ms} "
                f"n_retrieved={len(rec.items)} eta_min={eta_min:.1f}",
                file=sys.stderr,
                flush=True,
            )

        summary = materialize_system(
            system=run_name, movie=movie, records=records,
            out_dir=run_dir / "visual_selections", frames_dir=frames_dir,
            ffmpeg=ffmpeg, prompts=prompts,
        )
        final = adapter.finalize() or {}
        final["input_mode"] = input_mode
        final["budget"] = budget
        final["budget_policy"] = "runner_score_sorted_topB"
        (work_dir / "finalize.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["input_mode"] = input_mode
        summary["budget"] = budget
        summary["finalize"] = final
        # materialize_system() returns {"system", ...} but not the dataset/movie
        # identity keys that _summary_selection_complete() needs to locate the
        # durable selection file. Without them every single-movie success is
        # (wrongly) flagged incomplete and the process exits 31. Supply them.
        summary.setdefault("dataset", movie_dir.parent.name)
        summary.setdefault("movie", movie_dir.name)
        return summary
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _load_adapter(name: str):
    """Instantiate a registered causal adapter by name."""
    import importlib

    mod = importlib.import_module(name)
    factory = getattr(mod, "build_adapter", None)
    if factory is None:
        raise SystemExit(f"adapter module {name!r} must expose build_adapter()")
    return factory()


def main(argv: list[str] | None = None) -> int:
    _apply_runtime_safety_defaults()
    _require_a800_keepalive_if_requested()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="adapter module name, e.g. slotmem")
    ap.add_argument("--movie-dir", type=Path, default=None,
                    help="single movie directory to run")
    ap.add_argument("--movie-list", type=Path, default=None,
                    help="newline-delimited movie directories; reuses one loaded adapter")
    ap.add_argument("--ffmpeg", default=_DEFAULT_FFMPEG)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--limit", type=int, default=None, help="only first N segments (smoke)")
    ap.add_argument("--input-mode", choices=list(_INPUT_MODES), default="name_anchored",
                    help="name_anchored (main) | description_provided (append appearance "
                         "descriptions for prompt-named entities; fairness axis) | "
                         "description_only (replace prompt-named entities with neutral "
                         "appearance descriptions)")
    ap.add_argument("--budget", type=int, choices=list(_BUDGET_CHOICES),
                    default=int(os.environ.get("RETR_BUDGET", "16")),
                    help="runner-level top-B cap over returned items")
    ap.add_argument("--budget-sweep", action="store_true",
                    help="run B in {1,2,4,8,16}; outputs are suffixed with __B<budget>")
    args = ap.parse_args(argv)

    if (args.movie_dir is None) == (args.movie_list is None):
        raise SystemExit("provide exactly one of --movie-dir or --movie-list")

    _require_iamflow_http_services(args.adapter)

    slotmem_disable = (
        _BENCH_ROOT / "scripts" / "evaluate_baselines" / "trackA" / ".disable_slotmem_mainline"
    )
    if args.adapter == "slotmem" and slotmem_disable.is_file():
        print(json.dumps({
            "system": "slotmem",
            "skipped": True,
            "reason": (
                "SlotMem requires externally supplied or scripted role names for stable "
                "Track A operation, so it is disabled for the no-oracle mainline run."
            ),
            "disable_sentinel": str(slotmem_disable),
        }, ensure_ascii=False, indent=2))
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.movie_list is not None:
        movie_dirs = [
            Path(line.strip())
            for line in args.movie_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not movie_dirs:
            raise SystemExit(f"--movie-list is empty: {args.movie_list}")
    else:
        movie_dirs = [args.movie_dir]

    budgets = list(_BUDGET_CHOICES) if args.budget_sweep else [args.budget]
    summaries = []
    for budget in budgets:
        adapter = _load_adapter(args.adapter)
        for movie_dir in movie_dirs:
            # Isolate per-movie failures: a movie-list is a batch of independent
            # jobs, so one movie that OOMs or raises must not discard the movies
            # queued behind it (that is how whole IAMFlow lists were lost).
            try:
                summary = run_movie(adapter, movie_dir, ffmpeg=args.ffmpeg, fps=args.fps,
                                    limit=args.limit, input_mode=args.input_mode, budget=budget)
            # SystemExit is included on purpose: run_movie raises it for per-movie
            # data problems such as a missing source video, which must not take the
            # rest of the list down either.
            except (Exception, SystemExit) as exc:  # noqa: BLE001 - recorded, then exit 31
                traceback.print_exc()
                summary = {
                    "system": _run_name(adapter.name, args.input_mode, budget),
                    "dataset": movie_dir.parent.name,
                    "movie": movie_dir.name,
                    "input_mode": args.input_mode,
                    "budget": budget,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:2000],
                }
                print(
                    f"[stage1][error] {adapter.name}/{movie_dir.parent.name}/{movie_dir.name} "
                    f"{type(exc).__name__}: {str(exc)[:500]}",
                    file=sys.stderr,
                    flush=True,
                )
            summaries.append(summary)
    print(json.dumps(summaries[0] if len(summaries) == 1 else summaries,
                     ensure_ascii=False, indent=2))
    incomplete = [
        summary
        for summary in summaries
        if not _summary_selection_complete(summary)
    ]
    if incomplete:
        print(
            json.dumps(
                {
                    "error": "incomplete_stage1_output",
                    "count": len(incomplete),
                    "items": incomplete,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 31
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
