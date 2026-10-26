"""File job queue between a realization backend (client) and its resident model server.

Requests land in ``<server_dir>/inbox/`` by atomic rename, results come back in ``outbox/``; the server
writes ``ready`` (its pid) once weights are resident and exits on ``stop`` or after an idle timeout.
One sequential producer per server; dependency-free so the server processes can import it from any env.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def server_dirs(server_dir: Path) -> tuple[Path, Path]:
    inbox, outbox = Path(server_dir) / "inbox", Path(server_dir) / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    return inbox, outbox


def read_pid(server_dir: Path) -> int:
    try:
        return int((Path(server_dir) / "ready").read_text().strip())
    except (OSError, ValueError):
        return -1


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def server_alive(server_dir: Path) -> bool:
    return (Path(server_dir) / "ready").exists() and pid_alive(read_pid(server_dir))


def request_stop(server_dir: Path) -> None:
    (Path(server_dir) / "stop").touch()


def submit_job(server_dir: Path, request: dict, *, timeout: float, poll: float = 1.0) -> dict:
    """Submit one job and block until the server writes its result; raise on timeout or server death."""
    inbox, outbox = server_dirs(server_dir)
    job_id = str(request.get("job_id") or uuid.uuid4().hex)
    result_path = outbox / f"{job_id}.json"
    atomic_write_json(inbox / f"{job_id}.json", {**request, "job_id": job_id})
    deadline = time.time() + timeout
    while time.time() < deadline:
        if result_path.exists():
            result = read_json(result_path)
            result_path.unlink(missing_ok=True)
            return result
        if not server_alive(server_dir):
            raise RuntimeError(f"model server at {server_dir} died while job {job_id} was pending")
        time.sleep(poll)
    raise TimeoutError(f"job {job_id} at {server_dir} timed out after {timeout}s")


def next_job(inbox: Path) -> Path | None:
    jobs = sorted(Path(inbox).glob("*.json"))
    return jobs[0] if jobs else None


def serve_forever(server_dir: Path, handle, *, idle_timeout: float, poll: float = 0.5, log=print) -> None:
    """Server loop: ``handle(request) -> result dict``; exceptions become ``{"status": "error"}``."""
    inbox, outbox = server_dirs(server_dir)
    (Path(server_dir) / "ready").write_text(str(os.getpid()))
    last = time.time()
    try:
        while True:
            job = next_job(inbox)
            if job is None:
                if (Path(server_dir) / "stop").exists() or time.time() - last > idle_timeout:
                    break
                time.sleep(poll)
                continue
            try:
                request = read_json(job)
            except (OSError, ValueError):
                job.unlink(missing_ok=True)
                continue
            job.unlink(missing_ok=True)
            job_id = str(request["job_id"])
            t0 = time.time()
            try:
                result = {"status": "ok", **handle(request)}
            except Exception as exc:  # noqa: BLE001 - the client raises on status=error
                result = {"status": "error", "error": repr(exc)}
            result["server_s"] = round(time.time() - t0, 3)
            atomic_write_json(outbox / f"{job_id}.json", result)
            log(f"job {job_id}: {result.get('status')} in {result['server_s']}s")
            last = time.time()
    finally:
        (Path(server_dir) / "ready").unlink(missing_ok=True)
