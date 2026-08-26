"""Shared helpers for VMem-Bench Track B baseline runners.

Track B is generator-in-the-loop: a runner consumes the frozen SUT-facing
prompt stream and writes the generated long-video artifact under
``outputs/evaluation/trackB/<system>/<story>/<register>/<run_tag>/``.
Scoring is intentionally out of scope for these runners.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCH_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = Path(__file__).resolve().parents[6]
ASSETS_ROOT = BENCH_ROOT / "assets" / "trackB"
DEFAULT_OUTPUT_ROOT = BENCH_ROOT / "outputs" / "evaluation" / "trackB"
DEFAULT_PYTHON = "python3"


@dataclass(frozen=True)
class PromptSegment:
    segment_id: str
    prompt: str
    duration_sec: float
    transition: str


@dataclass(frozen=True)
class PromptStream:
    path: Path
    story_id: str
    title: str
    register: str
    segments: list[PromptSegment]
    raw: dict[str, Any]

    @property
    def prompts(self) -> list[str]:
        return [seg.prompt for seg in self.segments]


def default_prompt_paths() -> list[Path]:
    paths: list[Path] = []
    for path in sorted((ASSETS_ROOT / "sut_prompts").glob("*_name_anchored.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if len(data.get("segments") or []) > 0:
            paths.append(path)
    return paths


def load_prompt_stream(path: str | Path, *, limit: int = 0) -> PromptStream:
    p = Path(path).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    segments: list[PromptSegment] = []
    for idx, row in enumerate(raw.get("segments") or []):
        if limit and idx >= limit:
            break
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"{p}: empty prompt at segment {idx}")
        segments.append(
            PromptSegment(
                segment_id=str(row.get("segment_id") or f"seg_{idx + 1:03d}"),
                prompt=prompt,
                duration_sec=float(row.get("duration_sec", 5.0)),
                transition=str(row.get("transition") or "cut").lower(),
            )
        )
    if not segments:
        raise ValueError(f"{p}: no prompt segments")
    return PromptStream(
        path=p,
        story_id=str(raw.get("story_id") or p.stem),
        title=str(raw.get("title") or raw.get("story_id") or p.stem),
        register=str(raw.get("register") or "name_anchored"),
        segments=segments,
        raw=raw,
    )


def output_dir_for(system: str, stream: PromptStream, output_root: Path, run_tag: str) -> Path:
    return output_root / system / stream.story_id / stream.register / run_tag


def prepare_output_dir(path: Path, *, overwrite: bool = False, resume: bool = False) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f"output directory exists; use --resume or --overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "input").mkdir(exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    (path / "review").mkdir(exist_ok=True)


def copy_prompt_stream(stream: PromptStream, out_dir: Path) -> Path:
    dst = out_dir / "input" / stream.path.name
    dst.write_text(json.dumps(stream.raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dst


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_prompt_list(path: Path, prompts: list[str]) -> Path:
    return write_json(path, prompts)


def write_prompt_jsonl(path: Path, prompts: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prompts": prompts}, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def prompt_stream_to_minimal_screenplay(stream: PromptStream) -> dict[str, Any]:
    """Prompt-only wrapper for MemStrata production.

    This deliberately leaves ``main_entities`` empty. The SUT receives only the
    public prompt stream; any memory must be discovered from generated pixels and
    its own previous outputs.
    """
    shots = []
    for idx, seg in enumerate(stream.segments):
        shots.append(
            {
                "scene_id": f"scene_{idx + 1:03d}",
                "shot_id": seg.segment_id,
                "duration_sec": seg.duration_sec,
                "transition": seg.transition,
                "visual_track": {"actions": [seg.prompt]},
                "planned_assets": [],
                "active_characters": [],
            }
        )
    return {
        "story_id": stream.story_id,
        "title": stream.title,
        "source": "VMem-Bench TrackB sut_prompts prompt-only wrapper",
        "main_entities": [],
        "production_screenplay": {"shots": shots},
    }


def patch_simple_yaml(template: Path, updates: dict[str, str]) -> str:
    """Patch simple top-level YAML scalar keys without adding a YAML dependency."""
    def render_update(key: str, value: str) -> str:
        if key == "switch_frame_indices":
            return json.dumps(str(value))
        return value

    lines = template.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        key = stripped.split(":", 1)[0] if ":" in stripped else ""
        if indent == "" and key in updates:
            out.append(f"{key}: {render_update(key, updates[key])}")
            seen.add(key)
        elif key == "SMA" and "model_kwargs.SMA" in updates:
            out.append(f"{indent}SMA: {updates['model_kwargs.SMA']}")
            seen.add("model_kwargs.SMA")
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen and "." not in key:
            out.append(f"{key}: {render_update(key, value)}")
    return "\n".join(out) + "\n"


def command_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def merge_rc(current: int, rc: int) -> int:
    """Combine per-stream return codes preserving ANY failure.

    ``max(current, rc)`` is WRONG here: a subprocess killed by a signal reports a
    NEGATIVE returncode (e.g. -9 for SIGKILL / OOM-killer), and ``max(0, -9) == 0``
    silently turns a kill into a "success", making the launcher's EXIT sentinel and
    the monitor report a false DONE. Keep the first non-zero (incl. negative) code.
    """
    if rc == 0:
        return current
    return current if current != 0 else rc


def shell_rc(rc: int | None) -> int:
    """Normalize a (possibly negative) subprocess returncode to a valid 0..255 shell
    exit code so the launching shell's ``EXIT:$?`` sentinel is truthful. A signal
    kill (-N) maps to the conventional 128+N (e.g. -9 -> 137)."""
    if rc is None:
        return 1
    if rc < 0:
        return 128 + (-rc)
    return rc & 0xFF


def acquire_node_slot(name: str, *, slots: int = 1, wait_sec: float = 0.0,
                      poll_sec: float = 1.0):
    """Acquire one of ``slots`` NODE-LOCAL concurrency slots named ``name``.

    Backed by ``flock`` on files under ``/dev/shm`` (a per-node tmpfs), so this
    serializes processes on the SAME physical node without any cross-node
    coordination. Used to cap how many RAM-heavy jobs (e.g. LongLive-RAG, whose
    DynamicSwap text encoder pins ~69GB and spikes another ~69GB when it moves to
    GPU) run concurrently on one node, which otherwise trips the kernel OOM-killer
    and SIGKILLs several jobs at once.

    Returns the held file handle (keep it alive for the whole critical section;
    process exit or ``.close()`` releases it) or ``None`` if no slot became free
    within ``wait_sec``.
    """
    base = Path("/dev/shm")
    if not (base.is_dir() and os.access(base, os.W_OK)):
        base = Path("/tmp")
    slots = max(1, int(slots))
    deadline = time.time() + max(0.0, wait_sec)
    handles = []
    try:
        for i in range(slots):
            handles.append(open(base / f"{name}.slot{i}", "w"))
    except OSError:
        for h in handles:
            h.close()
        return None
    while True:
        for h in handles:
            try:
                fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Keep the winning handle; release the rest.
                for other in handles:
                    if other is not h:
                        other.close()
                return h
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    for other in handles:
                        other.close()
                    raise
        if time.time() >= deadline:
            for h in handles:
                h.close()
            return None
        time.sleep(poll_sec)


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (log_path.parent / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + json.dumps(str(cwd)) + "\n"
        + " ".join(json.dumps(x) for x in cmd)
        + "\n",
        encoding="utf-8",
    )
    if dry_run:
        log_path.write_text("[dry-run] " + " ".join(cmd) + "\n", encoding="utf-8")
        return 0
    with log_path.open("ab") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def write_manifest(
    out_dir: Path,
    *,
    system: str,
    stream: PromptStream,
    command: list[str],
    status: str,
    exit_code: int | None,
    artifacts: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "track": "B",
        "system": system,
        "story_id": stream.story_id,
        "register": stream.register,
        "prompt_stream": str(stream.path),
        "n_segments": len(stream.segments),
        "output_dir": str(out_dir),
        "status": status,
        "exit_code": exit_code,
        "command": command,
        "artifacts": artifacts or {},
        "notes": notes or [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return write_json(out_dir / "trackb_manifest.json", payload)


def add_common_args(parser: argparse.ArgumentParser, *, default_system: str) -> None:
    parser.add_argument("--prompts", type=Path, default=None, help="TrackB sut_prompts JSON")
    parser.add_argument("--run-tag", default="bench", help="Output run tag under story/register")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="Limit prompt segments for smoke runs")
    parser.add_argument("--system", default=default_system)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dry-run", action="store_true", help="Write inputs/manifests but do not launch GPU code")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def iter_requested_streams(args: argparse.Namespace) -> list[PromptStream]:
    paths = [args.prompts] if args.prompts else default_prompt_paths()
    if not paths:
        raise ValueError(f"no non-empty TrackB prompt streams found under {ASSETS_ROOT / 'sut_prompts'}")
    return [load_prompt_stream(path, limit=int(args.limit or 0)) for path in paths]
