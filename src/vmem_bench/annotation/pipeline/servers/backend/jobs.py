"""Small subprocess job registry for annotation pipeline batches."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vmem_bench.annotation.pipeline.servers.backend.catalog_state import (
    find_sample,
    list_samples,
    manifest_from_sample,
)
from vmem_bench.annotation.pipeline.servers.backend import remote_dispatch
from vmem_bench.annotation.pipeline.servers.backend.review_service import (
    accept_all_s4,
)
from vmem_bench.annotation.pipeline.stages.s7_freeze_publish.freeze import (
    continue_after_s6,
)
from vmem_bench.annotation.pipeline.servers.fleet.registry import (
    default_fleet_root,
    list_fleet,
    resolve_dispatch_urls,
)
from vmem_bench.annotation.pipeline.servers.fleet.timeutil import now_beijing
from vmem_bench.annotation.pipeline.stages.s3_segment_auto_review_revise.reviewer_pool import (
    parse_endpoint_urls,
)

_BLOCK_REASONS = {
    "missing_source_video": "缺少源视频路径（未配置 LSMDC/Blender 索引，或索引中无此片）",
    "source_video_missing_on_disk": "源视频路径在磁盘上不存在",
    "missing_vlm_output": "缺少非空 VLM 输出（vlm_output.json / vlm_outputs.json）",
}


def _now_id() -> str:
    stamp = now_beijing().replace("-", "").replace(" ", "-").replace(":", "")
    return f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def _now_stamp() -> str:
    return now_beijing()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass(slots=True)
class JobStore:
    jobs_root: Path
    data_root: Path
    python: str = sys.executable
    blender_index: Path | None = None
    lsmdc_index: Path | None = None
    fleet_root: Path | None = None

    def __post_init__(self) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        if self.fleet_root is None:
            self.fleet_root = default_fleet_root()
        else:
            self.fleet_root = Path(self.fleet_root)

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _job_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _active_job_for_samples(self, samples: list[dict[str, Any]]) -> str | None:
        selected = {
            (str(item.get("dataset") or ""), str(item.get("movie_id") or ""))
            for item in samples
        }
        if not selected:
            return None
        for path in self.jobs_root.glob("*/job.json"):
            payload = self.refresh_job(path.parent.name)
            if not isinstance(payload, dict) or str(payload.get("status") or "") not in {
                "queued",
                "running",
                "stopping",
            }:
                continue
            job_samples = {
                (str(item.get("dataset") or ""), str(item.get("movie_id") or ""))
                for item in payload.get("samples") or []
                if isinstance(item, dict)
            }
            if selected & job_samples:
                return str(payload.get("job_id") or path.parent.name)
        return None

    def _partition_busy_samples(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Split samples into idle vs already covered by an active job.

        Busy samples are skipped so a large multi-select resume does not fail
        the whole batch when one movie is already running.
        """
        activity = self.sample_activity()
        idle: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for item in samples:
            if not isinstance(item, dict):
                continue
            dataset = str(item.get("dataset") or "")
            movie_id = str(item.get("movie_id") or "")
            key = f"{dataset}::{movie_id}"
            busy = activity.get(key)
            if busy:
                skipped.append(
                    {
                        "dataset": dataset,
                        "movie_id": movie_id,
                        "job_id": str(busy.get("job_id") or ""),
                        "reason": "already_running",
                    }
                )
                continue
            idle.append(item)
        return idle, skipped

    @staticmethod
    def _has_reviewed_checkpoint(movie_dir: Path) -> bool:
        pipeline = movie_dir / "tmp" / "pipeline"
        for relative in (
            "s4_segment_sampling_human_review/review_audit.json",
            "s6_entities_visual_crop_human_review/review_audit.json",
        ):
            audit = _read_json(pipeline / relative, {})
            if isinstance(audit, dict) and audit.get("human_reviewed"):
                return True
        return False

    @staticmethod
    def _repo_root() -> Path:
        # jobs.py -> backend -> servers -> pipeline -> annotation ->
        # vmem_bench -> src -> MemStrata -> benchmarks -> Montage
        return Path(__file__).resolve().parents[8]

    def _submit_bdy_job(
        self,
        *,
        body: dict[str, Any],
        job_id: str,
        job_dir: Path,
        catalog_path: Path,
        out_path: Path,
        status_path: Path,
    ) -> dict[str, Any]:
        """Submit a job manifest to a BDY shared-FS worker."""
        node = str(body.get("bdy_node") or "0")
        if node not in {"0", "1", "2", "3"}:
            raise ValueError("bdy_node must be one of 0, 1, 2, 3")
        repo = self._repo_root()
        queue_script = repo / "scripts" / "tgpu_fs.py"
        runner = repo / "benchmarks" / "MemStrata" / "scripts" / "vmem_bench" / "ops" / "bdy-a800" / "s3" / "run_console_job.py"
        if not queue_script.is_file() or not runner.is_file():
            raise FileNotFoundError("BDY shared executor scripts are unavailable")
        queue_root = Path(
            os.environ.get(
                "MEMSTRATA_BDY_FS_ROOT",
                str(repo.parent.parent / "ssh_tunnel" / "fs_queue"),
            )
        )
        bdy_python = os.environ.get("MEMSTRATA_BDY_EXECUTOR_PYTHON", "").strip()
        if not bdy_python:
            bdy_python = str(Path(self.python).resolve().parents[2] / "vllm" / "bin" / "python")
        spec_path = job_dir / "bdy_execution.json"
        spec = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "catalog_path": str(catalog_path),
            "out_path": str(out_path),
            "status_path": str(status_path),
            "repo": str(repo),
            "python": bdy_python,
            "options": _public_options(body),
        }
        _write_json(spec_path, spec)
        queue_job_id = f"bdy-console-{job_id}"
        command = f"{bdy_python} {runner} --spec {spec_path}"
        submitted = subprocess.run(
            [
                sys.executable,
                str(queue_script),
                "--root",
                str(queue_root),
                "run",
                "--cluster",
                "bdy-a800",
                "--node",
                node,
                "--cwd",
                str(repo),
                "--timeout",
                "3600",
                "--wait",
                "0",
                "--job-id",
                queue_job_id,
                "--cmd",
                command,
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        if submitted.returncode != 0:
            raise RuntimeError(f"BDY queue submission failed: {submitted.stderr or submitted.stdout}")
        catalog_rows = [
            json.loads(line)
            for line in catalog_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        queue_state = _bdy_queue_state(
            {
                "bdy_queue_job_id": queue_job_id,
                "options": {"bdy_node": node, "execution_target": "bdy"},
                "dispatch": {"bdy_node": node, "execution_target": "bdy"},
            }
        )
        payload = {
            "job_id": job_id,
            "status": "queued" if queue_state == "queued" else "running",
            "created_at": _now_stamp(),
            "samples": [{"dataset": row["dataset"], "movie_id": row["movie_id"]} for row in catalog_rows],
            "options": _public_options(body),
            "dispatch": {
                "execution_target": "bdy",
                "bdy_node": node,
                "fleet_root": str(self.fleet_root),
                "n_reviewer_endpoints": 8,
            },
            "catalog_path": str(catalog_path),
            "out_path": str(out_path),
            "log_path": str(job_dir / "bdy_runner.log"),
            "bdy_spec_path": str(spec_path),
            "bdy_queue_job_id": queue_job_id,
            "queue_state": queue_state or "submitted",
            "command": command,
        }
        _write_json(self._job_path(job_id), payload)
        return payload

    def _remote_env_exports(self) -> dict[str, str]:
        """Env vars the KML batch needs that a login shell won't already have.

        Paths are all under the shared ``/data`` tree, so they resolve
        identically on the training nodes.  PYTHONPATH is rebuilt from scratch
        (vendor SAM3 deps + repo src) rather than inheriting the dev-box value.
        """
        src_root = str(Path(__file__).resolve().parents[5])
        montage_root = Path(__file__).resolve().parents[8]
        sam3_deps = (
            os.environ.get("MEMSTRATA_SAM3_DEPS", "").strip()
            or str(montage_root / "models" / "vendor" / "sam3_transformers59")
        )
        path_parts = [src_root]
        exports: dict[str, str] = {}
        if Path(sam3_deps).is_dir():
            path_parts.insert(0, sam3_deps)
            exports["MEMSTRATA_SAM3_DEPS"] = sam3_deps
        exports["PYTHONPATH"] = os.pathsep.join(path_parts)
        exports["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
        exports["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
        exports["PUBLIC_MODELS_ROOT"] = os.environ.get(
            "PUBLIC_MODELS_ROOT",
            "${PUBLIC_MODELS_ROOT}",
        )
        return exports

    def _active_kml_node_counts(self) -> dict[str, int]:
        """How many live jobs we've placed on each KML node (for load spread)."""
        counts: dict[str, int] = {}
        for job in self.active_jobs():
            dispatch = job.get("dispatch") if isinstance(job.get("dispatch"), dict) else {}
            target = str(job.get("execution_target") or dispatch.get("execution_target") or "")
            if target != "kml":
                continue
            cluster = str(dispatch.get("kml_cluster") or "")
            node = str(dispatch.get("kml_node") or "")
            if not cluster or not node:
                continue
            key = f"{cluster}#{node}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _submit_kml_job(
        self,
        *,
        body: dict[str, Any],
        job_id: str,
        job_dir: Path,
        catalog_path: Path,
        out_path: Path,
        status_path: Path,
        log_path: Path,
        manifests: list[Any],
        catalog_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Launch a batch detached on the least-loaded node via MEMSTRATA_TGPU.

        The remote batch writes ``progress.json`` / ``return_code.txt`` into the
        shared ``job_dir``, so the backend never needs SSH to read status.
        """
        nodes = remote_dispatch.load_kml_nodes()
        if not nodes:
            raise ValueError(
                "nodes.tsv 里没有 kml-* 节点；请用 execution_target=local 在本机跑"
            )
        placement = remote_dispatch.select_node(
            nodes, active_counts=self._active_kml_node_counts()
        )
        if placement is None:
            raise ValueError(
                "所有训练节点当前不可达（集群 launcher 探测失败）；请确认节点在线，"
                "或改用 execution_target=local 在本机跑"
            )
        inner = self._command(body, catalog_path, out_path, status_path)
        remote_dispatch.launch(
            placement.node,
            inner_script=inner,
            cwd=str(self.data_root),
            log_path=str(log_path),
            env_exports=self._remote_env_exports(),
        )
        samples = (
            [{"dataset": m.dataset, "movie_id": m.movie_id} for m in manifests]
            if manifests and hasattr(manifests[0], "dataset")
            else [{"dataset": r["dataset"], "movie_id": r["movie_id"]} for r in catalog_rows]
        )
        payload = {
            "job_id": job_id,
            "status": "running",
            "created_at": _now_stamp(),
            "samples": samples,
            "options": _public_options(body),
            "execution_target": "kml",
            "dispatch": {
                "execution_target": "kml",
                "kml_cluster": placement.node.cluster,
                "kml_node": placement.node.node,
                "kml_ip": placement.node.ip,
                "placement_reason": placement.reason,
                "reviewer_base_url": str(body.get("reviewer_base_url") or ""),
                "grounder_base_url": str(body.get("grounder_base_url") or ""),
                "fleet_root": str(self.fleet_root),
                "n_reviewer_endpoints": len(
                    [u for u in str(body.get("reviewer_base_url") or "").split(",") if u.strip()]
                ),
            },
            "catalog_path": str(catalog_path),
            "out_path": str(out_path),
            "log_path": str(log_path),
            "command": inner,
            # pkill -f target for stop_job: unique job_dir path in the cmdline.
            "remote_match": str(job_dir),
        }
        _write_json(self._job_path(job_id), payload)
        return payload

    def _stop_kml_job(self, payload: dict[str, Any]) -> None:
        """SIGTERM the remote batch for a KML job, best-effort."""
        dispatch = payload.get("dispatch") if isinstance(payload.get("dispatch"), dict) else {}
        cluster = str(dispatch.get("kml_cluster") or "")
        node_id = str(dispatch.get("kml_node") or "")
        match = str(payload.get("remote_match") or self._job_dir(str(payload.get("job_id"))))
        if cluster and node_id:
            node = remote_dispatch.KmlNode(
                cluster=cluster, node=node_id, ip=str(dispatch.get("kml_ip") or "")
            )
            try:
                remote_dispatch.stop(node, match=match)
            except Exception:  # noqa: BLE001 — stop must never raise into the API
                pass

    def _create_hybrid_job(self, body: dict[str, Any]) -> dict[str, Any]:
        """Split independent movie samples between local and BDY executors.

        A movie's pipeline directory is mutable, so hybrid mode shards only at
        the sample boundary. It intentionally never runs two executors against
        the same movie.
        """
        selected = list(body.get("samples") or [])
        if len(selected) < 2:
            raise ValueError(
                "hybrid execution requires at least two samples; one movie must use either local or BDY execution"
            )
        local_samples = selected[::2]
        bdy_samples = selected[1::2]
        parent_id = _now_id()
        parent_dir = self._job_dir(parent_id)
        parent_dir.mkdir(parents=True, exist_ok=False)
        local_body = {**body, "samples": local_samples, "execution_target": "local"}
        bdy_body = {**body, "samples": bdy_samples, "execution_target": "bdy"}
        local_job = self.create_job(local_body)
        bdy_job = self.create_job(bdy_body)
        payload = {
            "job_id": parent_id,
            "status": "running",
            "created_at": _now_stamp(),
            "samples": selected,
            "options": _public_options(body),
            "execution_target": "hybrid",
            "children": [
                {"target": "local", "job_id": local_job["job_id"]},
                {"target": "bdy", "job_id": bdy_job["job_id"]},
            ],
        }
        _write_json(self._job_path(parent_id), payload)
        return payload

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for path in sorted(self.jobs_root.glob("*/job.json")):
            payload = _read_json(path)
            if not isinstance(payload, dict):
                continue
            job_id = str(payload.get("job_id") or path.parent.name)
            jobs.append(self.refresh_job(job_id))
        return jobs

    def active_jobs(self, jobs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Return jobs that are still live: queued, running, or stopping."""
        source = jobs if jobs is not None else self.list_jobs()
        return [
            job
            for job in source
            if str(job.get("status") or "") in {"queued", "running", "stopping"}
        ]

    def sample_activity(self, jobs: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        """Map ``dataset::movie_id`` → active job membership for the console."""
        activity: dict[str, dict[str, Any]] = {}
        for job in self.active_jobs(jobs):
            job_id = str(job.get("job_id") or "")
            job_status = str(job.get("status") or "")
            samples = [item for item in (job.get("samples") or []) if isinstance(item, dict)]
            states = _sample_states_for_job(
                self._job_dir(job_id),
                job_status,
                samples,
                data_root=self.data_root,
            )
            for item in samples:
                dataset = str(item.get("dataset") or "")
                movie_id = str(item.get("movie_id") or "")
                if not dataset or not movie_id:
                    continue
                key = f"{dataset}::{movie_id}"
                sample_state = states.get(key, "queued" if job_status != "stopping" else "stopping")
                existing = activity.get(key)
                rank = {"running": 0, "queued": 1, "done": 2, "stopping": 3}
                if existing and rank.get(str(existing.get("sample_state")), 9) < rank.get(sample_state, 9):
                    continue
                activity[key] = {
                    "job_id": job_id,
                    "job_status": job_status,
                    "sample_state": sample_state,
                    "n_samples": len(samples),
                }
        return activity

    def jobs_snapshot(self) -> dict[str, Any]:
        """One refreshed pass used by the console jobs endpoints."""
        jobs = self.list_jobs()
        activity = self.sample_activity(jobs)
        running_samples = sum(1 for row in activity.values() if row.get("sample_state") == "running")
        queued_samples = sum(1 for row in activity.values() if row.get("sample_state") == "queued")
        return {
            "jobs": jobs,
            "active": self.active_jobs(jobs),
            "sample_activity": activity,
            "sample_counts": {
                "running": running_samples,
                "queued": queued_samples,
                "covered": len(activity),
            },
        }

    def refresh_job(self, job_id: str) -> dict[str, Any]:
        payload = _read_json(self._job_path(job_id))
        if not isinstance(payload, dict):
            return {"job_id": job_id, "status": "missing"}
        status = str(payload.get("status") or "")
        if payload.get("execution_target") == "hybrid" and status in {"running", "queued"}:
            children = [self.refresh_job(str(child["job_id"])) for child in payload.get("children") or []]
            child_states = {str(child.get("status") or "") for child in children}
            if children and child_states.isdisjoint({"queued", "running", "stopping"}):
                payload["status"] = "failed" if "failed" in child_states else "succeeded"
                payload["ended_at"] = _now_stamp()
                payload["child_statuses"] = [
                    {"job_id": child.get("job_id"), "status": child.get("status"), "error_summary": child.get("error_summary")}
                    for child in children
                ]
                _write_json(self._job_path(job_id), payload)
            elif children and "running" in child_states:
                if status != "running":
                    payload["status"] = "running"
                    _write_json(self._job_path(job_id), payload)
            elif children and child_states & {"queued", "stopping"} and "running" not in child_states:
                next_status = "stopping" if "stopping" in child_states else "queued"
                if status != next_status:
                    payload["status"] = next_status
                    _write_json(self._job_path(job_id), payload)
            return payload
        job_dir = self._job_dir(job_id)
        status_file = job_dir / "return_code.txt"
        if status in {"queued", "running", "stopping"}:
            bdy_state = _bdy_queue_state(payload)
            if bdy_state == "queued" and status in {"queued", "running"}:
                if status != "queued":
                    payload["status"] = "queued"
                    payload["queue_state"] = "pending"
                    _write_json(self._job_path(job_id), payload)
                else:
                    payload["queue_state"] = "pending"
                return payload
            if bdy_state == "running" and status == "queued":
                payload["status"] = "running"
                payload["queue_state"] = "running"
                _write_json(self._job_path(job_id), payload)
                status = "running"
            return_code = _return_code_from_status_file(job_dir)
            result_path = job_dir / "batch_result.json"
            pid = int(payload.get("pid") or 0)
            if status_file.is_file() or return_code is not None:
                failed = _batch_has_failures(job_dir)
                payload["status"] = (
                    "stopped"
                    if status == "stopping"
                    else ("failed" if (return_code not in (0, None) or failed) else "succeeded")
                )
                if return_code is not None:
                    payload["return_code"] = return_code
                payload["ended_at"] = _now_stamp()
                if result_path.is_file():
                    payload["result_path"] = str(result_path)
                summary = _batch_error_summary(job_dir)
                if summary:
                    payload["error_summary"] = summary
                _write_json(self._job_path(job_id), payload)
            elif pid and not _pid_alive(pid):
                payload["status"] = "stopped" if status == "stopping" else "failed"
                payload["ended_at"] = _now_stamp()
                summary = _batch_error_summary(job_dir)
                if summary:
                    payload["error_summary"] = summary
                _write_json(self._job_path(job_id), payload)
            elif bdy_state == "missing" and status in {"queued", "running"} and not pid:
                # BDY job left the shared queue without writing a terminal status yet.
                pass
        elif str(payload.get("status") or "") == "failed" and "error_summary" not in payload:
            summary = _batch_error_summary(job_dir)
            if summary:
                payload["error_summary"] = summary
        return payload

    def create_job(self, body: dict[str, Any]) -> dict[str, Any]:
        selected = body.get("samples")
        if not isinstance(selected, list) or not selected:
            raise ValueError("samples must be a non-empty list")
        body = dict(body)
        # Default to remote KML dispatch: CPU-heavy batch work runs on the shared
        # training nodes so the dev machine only serves the console + Cursor.
        # 'local' stays as a manual fallback for on-box debugging.
        execution_target = str(body.get("execution_target") or "kml").strip().lower()
        # bdy-a800 was retired 2026-07-23 (SSH-less shared-FS cluster, unusable).
        if execution_target in {"bdy", "hybrid"}:
            raise ValueError(
                "execution_target 'bdy'/'hybrid' 已废弃（bdy-a800 集群已下线）；"
                "请使用 kml（远程训练机）或 local（本机兜底）"
            )
        if execution_target not in {"local", "kml"}:
            raise ValueError("execution_target must be 'kml' (remote) or 'local'")
        idle_samples, skipped_busy = self._partition_busy_samples(list(selected))
        if not idle_samples:
            ids = ", ".join(
                str(item.get("job_id") or "?") for item in skipped_busy[:3]
            )
            raise ValueError(
                "选中样本都已在跑或排队"
                + (f"（{ids}）" if ids else "")
                + "；请等待完成或先暂停后再启动"
            )
        body["samples"] = idle_samples
        if execution_target == "hybrid":
            payload = self._create_hybrid_job(body)
            if skipped_busy:
                payload["skipped_busy"] = skipped_busy
                _write_json(self._job_path(str(payload["job_id"])), payload)
            return payload
        reviewer = str(body.get("reviewer") or "passthrough").strip()
        reviewer_base_url = str(body.get("reviewer_base_url") or "").strip()
        # kml reuses the same shared VLM fleet URLs as local: the fleet endpoints
        # (10.82.x / 10.83.x) are routable from the training nodes, so the remote
        # batch talks to the exact same reviewers/grounders (now load-balanced).
        if execution_target in {"local", "kml"} and reviewer == "qwen" and not reviewer_base_url:
            urls = resolve_dispatch_urls(
                fleet_root=self.fleet_root,
                probe=False,
                role="reviewer",
            )
            if not urls:
                fleet = list_fleet(fleet_root=self.fleet_root, probe=False)
                raise ValueError(
                    "no online VLM fleet endpoints for qwen reviewer; "
                    f"online={fleet.get('online_count', 0)}/{fleet.get('total_count', 0)} "
                    f"under {fleet.get('fleet_root')}. "
                    "Start reviewers with start_reviewer_pool.sh "
                    "(writes runtime/services/vlm_fleet status)."
                )
            body["reviewer_base_url"] = ",".join(urls)
            if not str(body.get("reviewer_model") or "").strip():
                body["reviewer_model"] = list_fleet(fleet_root=self.fleet_root, probe=False).get(
                    "default_model"
                ) or "qwen3-vl-32b"
        grounder = str(body.get("grounder") or "full-frame").strip()
        if execution_target in {"local", "kml"} and grounder == "qwen" and not str(body.get("grounder_base_url") or "").strip():
            # Prefer grounder-role; fall back to any online reviewer endpoint.
            g_urls = resolve_dispatch_urls(
                fleet_root=self.fleet_root,
                probe=False,
                role="grounder",
            )
            if not g_urls:
                g_urls = resolve_dispatch_urls(
                    fleet_root=self.fleet_root,
                    probe=False,
                    role="reviewer",
                )
            if not g_urls:
                raise ValueError(
                    "grounder=qwen needs online fleet endpoints under runtime/services/vlm_fleet"
                )
            body["grounder_base_url"] = ",".join(g_urls)
            if not str(body.get("grounder_model") or "").strip():
                body["grounder_model"] = list_fleet(
                    fleet_root=self.fleet_root, probe=False
                ).get("default_model") or str(body.get("reviewer_model") or "qwen3-vl-32b")
        force_restart = bool(body.get("force_restart"))
        resume = bool(body.get("resume")) and not force_restart
        body["resume"] = resume
        all_samples = list_samples(
            data_root=self.data_root,
            blender_index=self.blender_index,
            lsmdc_index=self.lsmdc_index,
        )
        manifests = []
        missing: list[dict[str, str]] = []
        blocked: list[dict[str, str]] = []
        for item in idle_samples:
            dataset = str(item.get("dataset") or "")
            movie_id = str(item.get("movie_id") or "")
            sample = find_sample(all_samples, dataset, movie_id)
            if sample is None:
                missing.append({"dataset": dataset, "movie_id": movie_id})
                continue
            if not sample.get("source_video"):
                blocked.append({"dataset": dataset, "movie_id": movie_id, "reason": "missing_source_video"})
                continue
            if not sample.get("source_video_exists"):
                blocked.append(
                    {"dataset": dataset, "movie_id": movie_id, "reason": "source_video_missing_on_disk"}
                )
                continue
            if not sample.get("has_vlm_output"):
                blocked.append({"dataset": dataset, "movie_id": movie_id, "reason": "missing_vlm_output"})
                continue
            if (
                not force_restart
                and not resume
                and not str(body.get("continue_from") or "").strip()
                and self._has_reviewed_checkpoint(Path(sample["movie_dir"]))
            ):
                raise ValueError(
                    f"{dataset}/{movie_id}: S4/S6 has an approved checkpoint; "
                    "use 续跑选中样本 / the row's resume action, or choose 重跑 to discard pipeline state"
                )
            # force_restart clearing happens inside the batch subprocess so
            # POST /api/jobs returns immediately (avoid nginx/proxy 504).
            manifests.append(manifest_from_sample(sample))

        skipped_unready = [
            {
                "dataset": str(item.get("dataset") or ""),
                "movie_id": str(item.get("movie_id") or ""),
                "reason": "missing_sample",
            }
            for item in missing
        ] + [
            {
                "dataset": str(item.get("dataset") or ""),
                "movie_id": str(item.get("movie_id") or ""),
                "reason": str(item.get("reason") or "blocked"),
            }
            for item in blocked
        ]
        if not manifests:
            raise ValueError(
                "没有可启动的样本（其余已跳过）:\n"
                + _format_selection_error(missing, blocked, self.blender_index, self.lsmdc_index)
            )

        # Multi-movie batches must share the VLM fleet across films. Defaulting
        # to 1 leaves later movies idle while one film sits in S5 with spare cards.
        if body.get("max_parallel_movies") in (None, ""):
            try:
                n_endpoints = len(parse_endpoint_urls(str(body.get("reviewer_base_url") or "")))
            except ValueError:
                n_endpoints = 1
            body["max_parallel_movies"] = _default_max_parallel_movies(
                n_endpoints=n_endpoints,
                n_movies=len(manifests),
            )

        job_id = _now_id()
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        catalog_path = job_dir / "catalog.jsonl"
        catalog_rows = []
        for manifest in manifests:
            row = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
            row["status"] = "source_ready"
            catalog_rows.append(row)
        catalog_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in catalog_rows),
            encoding="utf-8",
        )
        progress_path = job_dir / "progress.json"
        _write_json(
            progress_path,
            {
                "updated_at": _now_stamp(),
                "pending": [
                    {"dataset": str(row.get("dataset") or ""), "movie_id": str(row.get("movie_id") or "")}
                    for row in catalog_rows
                ],
                "running": [],
                "done": [],
            },
        )
        out_path = job_dir / "batch_result.json"
        log_path = job_dir / "run.log"
        status_path = job_dir / "return_code.txt"

        def _attach_skips(payload: dict[str, Any]) -> dict[str, Any]:
            if skipped_busy:
                payload["skipped_busy"] = skipped_busy
            if skipped_unready:
                payload["skipped_unready"] = skipped_unready
            return payload

        if execution_target == "kml":
            payload = self._submit_kml_job(
                body=body,
                job_id=job_id,
                job_dir=job_dir,
                catalog_path=catalog_path,
                out_path=out_path,
                status_path=status_path,
                log_path=log_path,
                manifests=manifests,
                catalog_rows=catalog_rows,
            )
            payload = _attach_skips(payload)
            _write_json(self._job_path(job_id), payload)
            return payload
        cmd = self._command(body, catalog_path, out_path, status_path)
        env = dict(os.environ)
        # .../servers/backend/jobs.py -> MemStrata/src is parents[5]; Montage root is parents[8].
        src_root = str(Path(__file__).resolve().parents[5])
        montage_root = Path(__file__).resolve().parents[8]
        sam3_deps = (
            os.environ.get("MEMSTRATA_SAM3_DEPS", "").strip()
            or str(montage_root / "models" / "vendor" / "sam3_transformers59")
        )
        path_parts = [src_root]
        if Path(sam3_deps).is_dir():
            # Prepend vendor transformers>=5.9 + mistral_common before site-packages.
            path_parts.insert(0, sam3_deps)
            env.setdefault("MEMSTRATA_SAM3_DEPS", sam3_deps)
        if env.get("PYTHONPATH"):
            path_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(path_parts)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Keep PUBLIC_MODELS_ROOT pointing at the shared local tree (SAM3 etc.).
        env.setdefault(
            "PUBLIC_MODELS_ROOT",
            "${PUBLIC_MODELS_ROOT}",
        )
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(  # noqa: S603
                ["bash", "-lc", cmd],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(self.data_root),
                start_new_session=True,
            )
        payload = {
            "job_id": job_id,
            "status": "running",
            "pid": proc.pid,
            "created_at": _now_stamp(),
            "samples": [{"dataset": m.dataset, "movie_id": m.movie_id} for m in manifests]
            if manifests and hasattr(manifests[0], "dataset")
            else [{"dataset": r["dataset"], "movie_id": r["movie_id"]} for r in catalog_rows],
            "options": _public_options(body),
            "dispatch": {
                "reviewer_base_url": str(body.get("reviewer_base_url") or ""),
                "grounder_base_url": str(body.get("grounder_base_url") or ""),
                "fleet_root": str(self.fleet_root),
                "n_reviewer_endpoints": len(
                    [u for u in str(body.get("reviewer_base_url") or "").split(",") if u.strip()]
                ),
            },
            "catalog_path": str(catalog_path),
            "out_path": str(out_path),
            "log_path": str(log_path),
            "command": cmd,
        }
        payload = _attach_skips(payload)
        _write_json(self._job_path(job_id), payload)
        return payload

    def stop_job(self, job_id: str) -> dict[str, Any]:
        payload = self.refresh_job(job_id)
        status = str(payload.get("status") or "")
        if payload.get("execution_target") == "hybrid":
            for child in payload.get("children") or []:
                self.stop_job(str(child.get("job_id") or ""))
            payload["status"] = "stopping"
            payload["stopped_at"] = _now_stamp()
            _write_json(self._job_path(job_id), payload)
            return payload
        if status not in {"queued", "running", "stopping"}:
            return payload
        dispatch = payload.get("dispatch") if isinstance(payload.get("dispatch"), dict) else {}
        if str(payload.get("execution_target") or dispatch.get("execution_target") or "") == "kml":
            # Remote job: SIGTERM it over SSH, then mark terminal. No local pid to
            # poll, and refresh is file-based, so settle status immediately.
            self._stop_kml_job(payload)
            payload["status"] = "stopped"
            payload["ended_at"] = _now_stamp()
            payload["stopped_at"] = _now_stamp()
            _write_json(self._job_path(job_id), payload)
            return payload
        cancelled_pending = _cancel_bdy_pending(payload)
        pid = int(payload.get("pid") or 0)
        if pid:
            _terminate_process_tree(pid)
        # No local process and either cancelled pending ticket or never started:
        # mark terminal immediately so the console does not stick on "stopping".
        if (cancelled_pending or status == "queued" or not pid) and not (pid and _pid_alive(pid)):
            payload["status"] = "stopped"
            payload["ended_at"] = _now_stamp()
            payload["stopped_at"] = _now_stamp()
            if cancelled_pending:
                payload["queue_state"] = "cancelled"
        else:
            payload["status"] = "stopping"
            payload["stopped_at"] = _now_stamp()
        _write_json(self._job_path(job_id), payload)
        return payload

    def stop_all_jobs(self) -> dict[str, Any]:
        """Stop every queued/running/stopping annotation job."""
        stopped: list[dict[str, Any]] = []
        for job in self.active_jobs():
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            stopped.append(self.stop_job(job_id))
        return {
            "ok": True,
            "n_stopped": len(stopped),
            "jobs": [
                {
                    "job_id": item.get("job_id"),
                    "status": item.get("status"),
                    "samples": item.get("samples") or [],
                }
                for item in stopped
            ],
        }

    def stop_sample_job(self, *, dataset: str, movie_id: str) -> dict[str, Any]:
        """Stop the active job that owns ``dataset/movie_id``."""
        key = f"{dataset}::{movie_id}"
        activity = self.sample_activity().get(key)
        if not activity:
            raise ValueError(f"no active job for {dataset}/{movie_id}")
        payload = self.stop_job(str(activity["job_id"]))
        return {
            **payload,
            "ok": True,
            "sample": {"dataset": dataset, "movie_id": movie_id},
            "sample_state": activity.get("sample_state"),
        }

    def create_continue_job(self, body: dict[str, Any]) -> dict[str, Any]:
        """Resume one movie after a human-review gate (S4/S6).

        S7 is a deterministic local freeze, so ``after_s6`` runs synchronously
        and deliberately never creates a background annotation job.  S4/S5
        continuations remain subprocess jobs because they can invoke models.

        When ``rerun_s5`` is true, discard S5/S6 artifacts first and force
        ``continue_from=after_s4`` so crop acquisition runs again from a clean slate.
        """
        dataset = str(body.get("dataset") or "")
        movie_id = str(body.get("movie_id") or "")
        if not dataset or not movie_id:
            raise ValueError("dataset and movie_id are required")
        continue_from = str(body.get("continue_from") or "")
        if continue_from == "after_s6":
            samples = list_samples(
                data_root=self.data_root,
                blender_index=self.blender_index,
                lsmdc_index=self.lsmdc_index,
            )
            sample = find_sample(samples, dataset, movie_id)
            if sample is None:
                raise ValueError(f"sample not found: {dataset}/{movie_id}")
            result = continue_after_s6(
                movie_dir=Path(sample["movie_dir"]),
                automation_smoke=False,
            )
            return {
                **result,
                "ok": True,
                "synchronous": True,
                "continue_from": continue_from,
            }
        job_body = dict(body)
        job_body["samples"] = [{"dataset": dataset, "movie_id": movie_id}]
        job_body.setdefault("skip_human", False)
        rerun_s5 = bool(body.get("rerun_s5"))
        if rerun_s5:
            job_body["continue_from"] = "after_s4"
            job_body.setdefault("grounder", "qwen")
            job_body.setdefault("s4_mode", "blocking")
            job_body.setdefault("crop_route", "propose_and_pick")
            job_body.setdefault("proposer", "fusion")
            job_body.setdefault("task_mode", "coverage")
            samples = list_samples(
                data_root=self.data_root,
                blender_index=self.blender_index,
                lsmdc_index=self.lsmdc_index,
            )
            sample = find_sample(samples, dataset, movie_id)
            if sample is None:
                raise ValueError(f"sample not found: {dataset}/{movie_id}")
            _clear_s5_s6(Path(sample["movie_dir"]))
        payload = self.create_job(job_body)
        payload["continue_from"] = str(job_body.get("continue_from") or "")
        if rerun_s5:
            payload["rerun_s5"] = True
        _write_json(self._job_path(str(payload["job_id"])), payload)
        return payload

    def accept_all_s4(self, body: dict[str, Any]) -> dict[str, Any]:
        """Accept pending S4 queues for selected idle samples."""
        selected = body.get("samples")
        if not isinstance(selected, list) or not selected:
            raise ValueError("samples is required")
        active_job = self._active_job_for_samples(selected)
        if active_job:
            raise ValueError(f"选中样本仍在任务 {active_job} 中运行，无法批量接受 S4")
        all_samples = list_samples(
            data_root=self.data_root,
            blender_index=self.blender_index,
            lsmdc_index=self.lsmdc_index,
        )
        samples: list[dict[str, Any]] = []
        for item in selected:
            dataset = str(item.get("dataset") or "")
            movie_id = str(item.get("movie_id") or "")
            sample = find_sample(all_samples, dataset, movie_id)
            if sample is None:
                raise ValueError(f"sample not found: {dataset}/{movie_id}")
            samples.append(sample)
        results = [
            {
                "dataset": str(sample["dataset"]),
                "movie_id": str(sample["movie_id"]),
                **accept_all_s4(sample),
            }
            for sample in samples
        ]
        continue_samples = [
            {"dataset": item["dataset"], "movie_id": item["movie_id"]}
            for item in results
            if item["status"] == "accepted"
        ]
        return {
            "ok": True,
            "results": results,
            "continue_samples": continue_samples,
        }

    def _command(
        self,
        body: dict[str, Any],
        catalog_path: Path,
        out_path: Path,
        status_path: Path,
    ) -> str:
        inner = [
            self.python,
            "-m",
            "vmem_bench.annotation.pipeline.batch",
            "--catalog",
            str(catalog_path),
            "--out",
            str(out_path),
            "--reviewer",
            str(body.get("reviewer") or "passthrough"),
            "--reviewer-base-url",
            str(body.get("reviewer_base_url") or ""),
            "--reviewer-model",
            str(body.get("reviewer_model") or "qwen3-vl-32b"),
            "--grounder",
            str(body.get("grounder") or "full-frame"),
            "--grounder-base-url",
            str(body.get("grounder_base_url") or ""),
            "--grounder-model",
            str(body.get("grounder_model") or "qwen3-vl-32b"),
            "--s4-mode",
            str(body.get("s4_mode") or "auto"),
            "--crop-route",
            str(body.get("crop_route") or "propose_and_pick"),
            "--proposer",
            str(body.get("proposer") or "fusion"),
            "--task-mode",
            str(body.get("task_mode") or "coverage"),
            "--progress",
            str(catalog_path.parent / "progress.json"),
        ]
        if body.get("continue_from"):
            inner.extend(["--continue-from", str(body["continue_from"])])
        if body.get("resume") and not body.get("continue_from"):
            inner.append("--resume")
        if body.get("auto_accept_s4"):
            inner.append("--auto-accept-s4")
        if body.get("force_restart"):
            inner.append("--force-restart")
        if body.get("skip_human"):
            inner.append("--skip-human")
        if body.get("max_tasks") not in (None, ""):
            inner.extend(["--max-tasks", str(int(body["max_tasks"]))])
        if body.get("max_review_rounds") not in (None, ""):
            inner.extend(["--max-review-rounds", str(int(body["max_review_rounds"]))])
        if body.get("max_clip_workers") not in (None, ""):
            inner.extend(["--max-clip-workers", str(int(body["max_clip_workers"]))])
        if body.get("max_parallel_movies") not in (None, ""):
            inner.extend(["--max-parallel-movies", str(int(body["max_parallel_movies"]))])
        if body.get("clip_queue_root"):
            inner.extend(["--clip-queue-root", str(body["clip_queue_root"])])
        if body.get("limit") not in (None, ""):
            inner.extend(["--limit", str(int(body["limit"]))])
        script = " ".join(_shell_quote(part) for part in inner)
        return f"{script}; rc=$?; echo $rc > {_shell_quote(str(status_path))}; exit $rc"


def _public_options(body: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "reviewer",
        "reviewer_base_url",
        "reviewer_model",
        "grounder",
        "grounder_base_url",
        "grounder_model",
        "skip_human",
        "s4_mode",
        "continue_from",
        "crop_route",
        "proposer",
        "task_mode",
        "max_tasks",
        "max_review_rounds",
        "max_clip_workers",
        "max_parallel_movies",
        "clip_queue_root",
        "limit",
        "force_restart",
        "resume",
        "auto_accept_s4",
        "rerun_s5",
        "execution_target",
        "bdy_node",
        "bdy_mode",
    )
    options = {key: body.get(key) for key in keys if key in body}
    requested = str(body.get("s4_mode") or "auto")
    options["s4_mode"] = requested
    options["s4_mode_effective"] = (
        ("nonblocking" if body.get("skip_human") else "blocking")
        if requested == "auto"
        else requested
    )
    return options


def _format_selection_error(
    missing: list[dict[str, str]],
    blocked: list[dict[str, str]],
    blender_index: Path | None,
    lsmdc_index: Path | None,
) -> str:
    lines: list[str] = []
    for item in missing:
        lines.append(f"{item.get('dataset')}/{item.get('movie_id')}: 样本不存在")
    for item in blocked:
        reason = str(item.get("reason") or "")
        lines.append(
            f"{item.get('dataset')}/{item.get('movie_id')}: {_BLOCK_REASONS.get(reason, reason)}"
        )
    lines.append(f"blender_index={blender_index or '(none)'}")
    lines.append(f"lsmdc_index={lsmdc_index or '(none)'}")
    return "\n".join(lines)


def _movie_key(dataset: str, movie_id: str) -> str:
    return f"{dataset}::{movie_id}"


def _default_max_parallel_movies(*, n_endpoints: int, n_movies: int) -> int:
    """Pick a shared-pool concurrency for multi-movie console batches.

    One film in S5 rarely saturates the fleet; leaving max_parallel=1 parks every
    other movie until that film finishes the full pipeline.
    """
    if n_movies <= 1 or n_endpoints <= 1:
        return 1
    # Cap bounds concurrent SAM3/GDINO residency while still feeding S3.
    return max(1, min(int(n_movies), int(n_endpoints), 12))


def _sample_states_for_job(
    job_dir: Path,
    job_status: str,
    samples: list[dict[str, Any]],
    data_root: Path | None = None,
) -> dict[str, str]:
    """Resolve per-sample running/queued/done inside one annotation job."""
    if job_status == "queued":
        return {
            _movie_key(str(item.get("dataset") or ""), str(item.get("movie_id") or "")): "queued"
            for item in samples
            if item.get("dataset") and item.get("movie_id")
        }
    if job_status == "stopping":
        return {
            _movie_key(str(item.get("dataset") or ""), str(item.get("movie_id") or "")): "stopping"
            for item in samples
            if item.get("dataset") and item.get("movie_id")
        }

    progress = _read_json(job_dir / "progress.json", {})
    states: dict[str, str] = {}
    has_progress = isinstance(progress, dict) and (
        progress.get("running") is not None
        or progress.get("pending") is not None
        or progress.get("done") is not None
    )
    if has_progress:
        for item in progress.get("pending") or []:
            if isinstance(item, dict) and item.get("dataset") and item.get("movie_id"):
                states[_movie_key(str(item["dataset"]), str(item["movie_id"]))] = "queued"
        for item in progress.get("done") or []:
            if isinstance(item, dict) and item.get("dataset") and item.get("movie_id"):
                states[_movie_key(str(item["dataset"]), str(item["movie_id"]))] = "done"
        for item in progress.get("running") or []:
            if isinstance(item, dict) and item.get("dataset") and item.get("movie_id"):
                states[_movie_key(str(item["dataset"]), str(item["movie_id"]))] = "running"
        for item in samples:
            dataset = str(item.get("dataset") or "")
            movie_id = str(item.get("movie_id") or "")
            if dataset and movie_id:
                states.setdefault(_movie_key(dataset, movie_id), "queued")

        # Cheap live refresh: verify progress.running first (1–few reads).
        # Full scan only when those look stale (legacy workers without --progress).
        sample_by_key = {
            _movie_key(str(item.get("dataset") or ""), str(item.get("movie_id") or "")): item
            for item in samples
            if item.get("dataset") and item.get("movie_id")
        }
        preferred = [
            _movie_key(str(item["dataset"]), str(item["movie_id"]))
            for item in (progress.get("running") or [])
            if isinstance(item, dict) and item.get("dataset") and item.get("movie_id")
        ]
        still_live = [
            key
            for key in preferred
            if key in sample_by_key
            and _sample_has_live_stage(sample_by_key[key], data_root=data_root)
        ]
        if still_live:
            for key in preferred:
                if key not in still_live and states.get(key) == "running":
                    states[key] = "queued"
            for key in still_live:
                if states.get(key) != "done":
                    states[key] = "running"
            return states

        for item in samples:
            dataset = str(item.get("dataset") or "")
            movie_id = str(item.get("movie_id") or "")
            if not dataset or not movie_id:
                continue
            key = _movie_key(dataset, movie_id)
            if states.get(key) == "done":
                continue
            if _sample_has_live_stage(item, data_root=data_root):
                for other, value in list(states.items()):
                    if value == "running":
                        states[other] = "queued"
                states[key] = "running"
                return states
        return states

    # Legacy in-flight jobs without progress.json: only live stage work is "running".
    if len(samples) <= 1:
        for item in samples:
            dataset = str(item.get("dataset") or "")
            movie_id = str(item.get("movie_id") or "")
            if dataset and movie_id:
                states[_movie_key(dataset, movie_id)] = "running"
        return states

    live_key: str | None = None
    for item in samples:
        dataset = str(item.get("dataset") or "")
        movie_id = str(item.get("movie_id") or "")
        if not dataset or not movie_id:
            continue
        key = _movie_key(dataset, movie_id)
        states[key] = "queued"
        if live_key is None and _sample_has_live_stage(item, data_root=data_root):
            live_key = key
    if live_key is not None:
        states[live_key] = "running"
    elif samples:
        # Sequential batch just started: treat the first catalog row as running.
        first = samples[0]
        key = _movie_key(str(first.get("dataset") or ""), str(first.get("movie_id") or ""))
        if key in states:
            states[key] = "running"
    return states


def _sample_has_live_stage(
    item: dict[str, Any],
    data_root: Path | None = None,
) -> bool:
    """Best-effort: inspect on-disk pipeline state for an active stage."""
    raw_dir = str(item.get("movie_dir") or "").strip()
    movie_dir = Path(raw_dir) if raw_dir else None
    if movie_dir is None or not movie_dir.is_dir():
        dataset = str(item.get("dataset") or "")
        movie_id = str(item.get("movie_id") or "")
        if data_root is not None and dataset and movie_id:
            movie_dir = Path(data_root) / dataset / movie_id
        else:
            return False
    if not movie_dir.is_dir():
        return False
    state = _read_json(movie_dir / "tmp" / "pipeline" / "state.json", {})
    if not isinstance(state, dict):
        return False
    for entry in (state.get("stages") or {}).values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status") or "") in {"running", "in_progress"}:
            return True
    return False


def _clear_movie_runtime(movie_dir: Path) -> list[str]:
    """Clear resumable pipeline state so a movie can be restarted from S2."""
    cleared: list[str] = []
    pipeline = movie_dir / "tmp" / "pipeline"
    if pipeline.is_dir():
        shutil.rmtree(pipeline)
        cleared.append(str(pipeline))
    return cleared


def _clear_s5_s6(movie_dir: Path) -> list[str]:
    """Drop S5/S6 stage dirs so crop acquisition can restart after S4."""
    cleared: list[str] = []
    pipeline = movie_dir / "tmp" / "pipeline"
    for relative in (
        "s5_entities_visual_crop_acquisition",
        "s6_entities_visual_crop_human_review",
    ):
        path = pipeline / relative
        if path.is_dir():
            shutil.rmtree(path)
            cleared.append(str(path))
        elif path.exists():
            path.unlink()
            cleared.append(str(path))
    state_path = pipeline / "state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {"stages": {}}
        stages = dict(state.get("stages") or {})
        removed = False
        for key in (
            "s5_entities_visual_crop_acquisition",
            "s6_entities_visual_crop_human_review",
            "s7_freeze_publish",
        ):
            if key in stages:
                stages.pop(key, None)
                removed = True
        if removed:
            state["stages"] = stages
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cleared.append(str(state_path))
    return cleared


def _return_code_from_status_file(job_dir: Path) -> int | None:
    status_path = job_dir / "return_code.txt"
    if not status_path.is_file():
        return None
    try:
        return int(status_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


# Per-movie statuses that mean the movie did NOT make successful progress. A
# batch process that only hits these still exits 0, so without this set a job
# whose sole movie is blocked at S2 was mislabeled "succeeded" (the console then
# said 已完成 while the sample stayed at S2 invalid_structure). "awaiting_human"
# / "already_complete" / "ok" are legitimate outcomes and are NOT blockers.
_BLOCKED_MOVIE_STATUSES = frozenset({
    "failed",
    "invalid_structure",
    "invalid_json",
    "input_unreadable",
    "skipped_empty_input",
})


def _batch_has_failures(job_dir: Path) -> bool:
    payload = _read_json(job_dir / "batch_result.json")
    if not isinstance(payload, list):
        return False
    return any(
        isinstance(item, dict) and item.get("status") in _BLOCKED_MOVIE_STATUSES
        for item in payload
    )


def _batch_error_summary(job_dir: Path, *, limit: int = 3) -> str:
    payload = _read_json(job_dir / "batch_result.json")
    if not isinstance(payload, list):
        return ""
    lines: list[str] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("status") not in _BLOCKED_MOVIE_STATUSES:
            continue
        movie = item.get("movie_id") or "?"
        status = str(item.get("status") or "failed")
        stage = str(item.get("stage") or "")
        # invalid_structure has no free-text error; surface the status + stage so
        # the operator sees "blocked at S2", not a vague failure.
        detail = item.get("error") or item.get("error_type") or status
        label = f"{movie}: {detail}"
        if stage and stage not in str(detail):
            label += f" ({stage})"
        lines.append(label)
        if len(lines) >= limit:
            break
    return " | ".join(lines)


def _pid_alive(pid: int) -> bool:
    try:
        stat_path = Path(f"/proc/{pid}/stat")
        if stat_path.is_file():
            tail = stat_path.read_text(encoding="utf-8").rsplit(")", maxsplit=1)[-1].strip()
            if tail.startswith("Z "):
                return False
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_tree(pid: int) -> None:
    """SIGTERM the process group first, then the leader pid."""
    if not pid:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _bdy_queue_root() -> Path:
    repo = Path(__file__).resolve().parents[8]
    return Path(
        os.environ.get(
            "MEMSTRATA_BDY_FS_ROOT",
            str(repo.parent.parent / "ssh_tunnel" / "fs_queue"),
        )
    )


def _bdy_node_from_payload(payload: dict[str, Any]) -> str | None:
    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    dispatch = payload.get("dispatch") if isinstance(payload.get("dispatch"), dict) else {}
    node = str(
        options.get("bdy_node")
        or dispatch.get("bdy_node")
        or payload.get("bdy_node")
        or ""
    ).strip()
    return node or None


def _bdy_queue_paths(payload: dict[str, Any]) -> tuple[Path, Path] | None:
    queue_job_id = str(payload.get("bdy_queue_job_id") or "").strip()
    node = _bdy_node_from_payload(payload)
    if not queue_job_id or not node:
        return None
    base = _bdy_queue_root() / "bdy-a800" / str(node)
    return base / "pending" / f"{queue_job_id}.json", base / "running" / f"{queue_job_id}.json"


def _bdy_queue_state(payload: dict[str, Any]) -> str | None:
    """Return pending/running/missing for a BDY shared-FS job, else None."""
    paths = _bdy_queue_paths(payload)
    if paths is None:
        return None
    pending, running = paths
    if pending.is_file():
        return "queued"
    if running.is_file():
        return "running"
    return "missing"


def _cancel_bdy_pending(payload: dict[str, Any]) -> bool:
    """Remove a still-pending BDY queue ticket so the worker never starts it."""
    paths = _bdy_queue_paths(payload)
    if paths is None:
        return False
    pending, _running = paths
    if not pending.is_file():
        return False
    try:
        pending.unlink()
        return True
    except OSError:
        return False


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
