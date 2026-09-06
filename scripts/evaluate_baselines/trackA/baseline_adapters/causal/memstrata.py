"""Thin Track-A bridge to MemStrata's canonical production read/write entry points."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from contract import ComposeRequest, MovieContext, RetrievedItem, RetrievedMemory, SegmentObservation
from _local_roots import find_memstrata_src


def _ensure_method_package() -> None:
    """Resolve the method package despite this adapter's top-level ``memstrata`` name."""
    source = str(find_memstrata_src())
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    shadow = sys.modules.get("memstrata")
    if shadow is not None and getattr(shadow, "__file__", "").endswith("/causal/memstrata.py"):
        sys.modules.pop("memstrata", None)


class MemStrataAdapter:
    """JSON/process glue only; method semantics live under ``methods/MemStrata``."""

    name = "memstrata"

    def __init__(
        self,
        *,
        public_models_root: str | None = None,
        ffmpeg: str = "ffmpeg",
        device: str = "",
        enable_perception: bool = True,
        identity_threshold: float = 0.25,
        frame_pos: float = 0.5,
        name_source: str = "perception",
        max_reps_per_asset: int = 5,
        decompose_frames: int = 3,
        decompose_fps: float = 2.0,
    ) -> None:
        del ffmpeg, decompose_fps  # production crop acquisition owns media decoding.
        models = public_models_root or os.environ.get("PUBLIC_MODELS_ROOT", "").strip()
        if not models:
            raise ValueError("PUBLIC_MODELS_ROOT or public_models_root is required")
        os.environ.setdefault("PUBLIC_MODELS_ROOT", str(Path(models).expanduser().resolve()))
        self.device = str(device)
        self.enable_perception = bool(enable_perception)
        self.identity_threshold = float(identity_threshold)
        self.frame_pos = min(max(float(frame_pos), 0.0), 1.0)
        self.name_source = str(name_source)
        self.max_reps_per_asset = int(max_reps_per_asset)
        self.decompose_frames = max(1, int(decompose_frames))
        self.read_slow_fallback = os.environ.get(
            "MEMSTRATA_TRACKA_READ_SLOW_FALLBACK",
            "1" if self.name_source == "mllm" else "0",
        ).lower() in {"1", "true", "on", "yes"}
        self.read_max_reps_per_asset = max(
            1, int(os.environ.get("MEMSTRATA_TRACKA_READ_MAX_REPS", "1") or 1)
        )
        raw_budget = os.environ.get("MEMSTRATA_TRACKA_READ_CONTEXT_BUDGET", "").strip()
        self.read_context_budget = int(raw_budget) if raw_budget else None
        self.snapshot_each_segment = os.environ.get(
            "MEMSTRATA_TRACKA_SNAPSHOT_EACH_SEGMENT", "1"
        ).lower() in {"1", "true", "on", "yes"}
        self._movie: MovieContext | None = None
        self._mem: Any = None
        self._work_dir: Path | None = None
        self._retrieval_sources: dict[str, int] = {}

    def reset(self, movie: MovieContext) -> None:
        _ensure_method_package()
        try:
            from memstrata.production.realized import build_realized_segment_pipeline
        except ModuleNotFoundError as exc:
            if exc.name not in {"memstrata.production", "memstrata.production.realized"}:
                raise
            raise RuntimeError(
                "This Track-A adapter requires MemStrata's production entry "
                "memstrata.production.realized.build_realized_segment_pipeline"
            ) from exc

        self._movie = movie
        self._work_dir = Path(movie.work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        provider = os.environ.get("MEMSTRATA_GENERAL_EMBEDDER_PROVIDER", "dinov3")
        self._mem = build_realized_segment_pipeline(
            run_dir=self._work_dir,
            persist_path=self._work_dir / "bank.json",
            movie_id=movie.movie_id,
            write_naming=self.name_source,
            discovery=self.enable_perception and self.name_source != "mllm",
            crop_acq_device=self.device,
            embedder_provider=provider,
            identity_threshold=self.identity_threshold,
            frame_pos=self.frame_pos,
            namer_frames=self.decompose_frames,
            read_slow_fallback=self.read_slow_fallback,
            read_max_reps_per_asset=self.read_max_reps_per_asset,
            read_context_rep_budget=self.read_context_budget,
            max_reps_per_asset=self.max_reps_per_asset,
        )
        self._mem.fps = float(movie.fps)
        self._mem.long_video_path = str(movie.source_video)
        self._retrieval_sources = {}

    def compose(self, req: ComposeRequest) -> RetrievedMemory:
        record = RetrievedMemory(chunk_id=req.chunk_id)
        if self._mem is None:
            return record
        request, context, _calls = self._mem.step1_compose(
            req.prompt_text, segment_id=int(req.chunk_id)
        )
        source = str(getattr(request, "intent_resolution_source", "recency"))
        record.extras["intent_resolution_source"] = source
        record.extras["intent_asset_ids"] = [ref.asset_id for ref in request.references]
        rep_seconds = self._mem.representation_seconds()
        for asset_id in context.asset_ids:
            for rep_id in context.representation_ids.get(asset_id, []):
                seconds = rep_seconds.get(rep_id)
                if seconds is None or seconds >= float(req.seconds_span[0]):
                    continue
                found = self._mem.bank.find_representation(rep_id)
                if found is None:
                    continue
                image_path = str(Path(found[1].object_uri).expanduser().resolve())
                record.items.append(
                    RetrievedItem(
                        evidence_kind="reference_image",
                        source_seconds=seconds,
                        raw_ref=f"memstrata:{asset_id}:{rep_id}",
                        image_path=image_path,
                    )
                )
        self._retrieval_sources[source] = self._retrieval_sources.get(source, 0) + 1
        return record

    def observe_segment(self, obs: SegmentObservation) -> None:
        if self._mem is None:
            raise RuntimeError("reset() must be called before observe_segment()")
        start, end = map(float, obs.seconds_span)
        self._mem.observe_realized_segment(
            segment_id=int(obs.chunk_id),
            segment_video=obs.segment_video,
            prompt=obs.prompt_text,
            source_start_sec=start,
            source_duration_sec=max(0.0, end - start),
            fps=float(obs.fps),
        )
        if self.snapshot_each_segment:
            self._mem.write_memory_snapshot()

    def finalize(self) -> dict[str, Any]:
        if self._mem is None:
            return {"system": self.name, "assets": 0, "representations": 0}
        self._mem.finalize()
        bank = self._mem.bank
        return {
            "system": self.name,
            "implementation": "memstrata.production.realized",
            "policy": self._mem.policy.name,
            "name_source": self.name_source,
            "read_slow_fallback": self.read_slow_fallback,
            "read_max_reps_per_asset": self.read_max_reps_per_asset,
            "read_context_budget": self.read_context_budget,
            "assets": len(bank.assets),
            "representations": sum(len(asset.representations) for asset in bank.assets.values()),
            "retrieval_sources": dict(sorted(self._retrieval_sources.items())),
            "stratification": self._mem.stratification(),
        }


def build_adapter() -> MemStrataAdapter:
    name_source = os.environ.get("MEMSTRATA_TRACKA_NAME_SOURCE", "perception").strip().lower()
    if name_source not in {"perception", "mllm"}:
        raise SystemExit(
            "MEMSTRATA_TRACKA_NAME_SOURCE must be 'perception' or 'mllm', "
            f"got {name_source!r}"
        )
    return MemStrataAdapter(name_source=name_source)
