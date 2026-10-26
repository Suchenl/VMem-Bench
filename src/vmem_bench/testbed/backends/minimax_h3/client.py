"""``RealizationBackend`` client for resident MiniMax-H3 Ref2V Turbo servers.

The references go straight into the video model (no composed keyframe). The client turns the
segment instruction into an H3 prompt (:func:`build_h3_prompt`), posts it with the reference paths
to one of the discovered servers and returns the clip the server wrote under ``request.out_dir``.
Several runner threads may share one client; each call takes a free endpoint from the pool.
"""

from __future__ import annotations

import json
import os
import queue
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from vmem_bench.testbed.backends.base import RealizationRequest, RealizationResult
from vmem_bench.testbed.backends.minimax_h3 import geometry as geo
from vmem_bench.testbed.backends.minimax_h3.geometry import MODEL_ID
from vmem_bench.testbed.backends.minimax_h3.prompt import (
    MAX_REFERENCES,
    PROMPT_FORMATS,
    build_h3_prompt,
    build_h3_prompt_subjects,
    build_h3_prompt_v2,
    normalize_instruction,
)

ENDPOINTS_ENV = "VMEM_H3_ENDPOINTS"
BOARD_CELL_HEIGHT = 480
BOARD_GUTTER = 16


def write_reference_board(references: Sequence[str | Path], out: Path) -> Path:
    """Tile the stills into one image, request order, row-major, 2 columns, black gutters.

    Every still is Lanczos-resized to ``BOARD_CELL_HEIGHT`` keeping its aspect ratio; cells are as wide as
    the widest still. The board is a pure function of the input images.
    """
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in references]
    ims = [im.resize((max(1, round(im.width * BOARD_CELL_HEIGHT / im.height)), BOARD_CELL_HEIGHT),
                     Image.Resampling.LANCZOS) for im in ims]
    cols = min(2, len(ims))
    rows = (len(ims) + cols - 1) // cols
    cw = max(im.width for im in ims)
    board = Image.new("RGB", (cols * cw + (cols - 1) * BOARD_GUTTER,
                              rows * BOARD_CELL_HEIGHT + (rows - 1) * BOARD_GUTTER))
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        board.paste(im, (c * (cw + BOARD_GUTTER) + (cw - im.width) // 2, r * (BOARD_CELL_HEIGHT + BOARD_GUTTER)))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.png")
    board.save(tmp, format="PNG")
    os.replace(tmp, out)
    return out

# Resident servers sit on cluster-internal IPs; never route them through http(s)_proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with _OPENER.open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def healthy(url: str, *, timeout: float = 5.0, model_id: str = MODEL_ID) -> bool:
    try:
        info = _get_json(url.rstrip("/") + "/health", timeout)
    except (OSError, ValueError):
        return False
    return bool(info.get("ok")) and info.get("model") == model_id


def candidate_urls(spec: str | Path | Sequence[str], model_id: str = MODEL_ID) -> list[str]:
    """Expand an endpoint spec without contacting anything.

    ``spec`` is a directory of server endpoint JSONs (``<model>_*.json`` with a ``url``), a text file
    with one URL per line (``#`` comments allowed), a comma-separated URL string, or a list of URLs.
    """
    if isinstance(spec, (list, tuple)):
        urls = [str(u).strip() for u in spec]
    else:
        path = Path(str(spec)).expanduser()
        if path.is_dir():
            urls = []
            for f in sorted(path.glob(f"{model_id}_*.json")):
                try:
                    urls.append(str(json.loads(f.read_text(encoding="utf-8"))["url"]))
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    continue
        elif path.is_file():
            urls = [ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        else:
            urls = [u.strip() for u in str(spec).split(",")]
    return sorted({u.rstrip("/") for u in urls if u})


def discover_endpoints(spec: str | Path | Sequence[str], *, timeout: float = 5.0,
                       model_id: str = MODEL_ID) -> list[str]:
    """Healthy MiniMax-H3 endpoints for ``spec`` (see :func:`candidate_urls`)."""
    return [u for u in candidate_urls(spec, model_id) if healthy(u, timeout=timeout, model_id=model_id)]


def build_request_body(request: RealizationRequest, *, out: Path, prompt: str, num_frames: int, fps: int,
                       geometry: str, steps: int, ref_image_short_edge: int,
                       references: Sequence[str | Path] | None = None) -> dict[str, Any]:
    g = geo.geometry(geometry)
    refs = request.references if references is None else references
    return {
        "prompt": prompt,
        "references": [str(Path(p).resolve()) for p in refs],
        "seed": int(request.seed),
        "out": str(out.resolve()),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "geometry": g.name,
        "height": g.height,
        "width": g.width,
        "out_height": g.out_height,
        "out_width": g.out_width,
        "steps": int(steps),
        "ref_image_short_edge": int(ref_image_short_edge),
    }


class MiniMaxH3TurboBackend:
    """Realization backend: MiniMax-H3 Ref2V + 4-step Turbo LoRA, references into the video model."""

    name = "minimax_h3_turbo"

    def __init__(
        self,
        endpoints: str | Path | Sequence[str] | None = None,
        *,
        num_frames: int = geo.NUM_FRAMES,
        fps: int = geo.FPS,
        geometry: str = geo.DEFAULT_GEOMETRY,
        steps: int = geo.STEPS,
        ref_image_short_edge: int = geo.REF_IMAGE_SHORT_EDGE,
        duration_sec: float = geo.SEGMENT_SECONDS,
        max_references: int = 3,
        timeout_sec: float = 1800.0,
        prompt_format: int = 1,
        reference_board: bool = False,
        subject_crops: bool = False,
        subject_detector: str | None = None,
    ):
        self.endpoints_spec = endpoints if endpoints is not None else os.environ.get(ENDPOINTS_ENV)
        self.num_frames, self.fps = num_frames, fps
        self.geometry = geo.geometry(geometry).name
        self.steps, self.ref_image_short_edge = steps, ref_image_short_edge
        self.duration_sec, self.max_references = duration_sec, max_references
        self.timeout_sec = timeout_sec
        if int(prompt_format) not in PROMPT_FORMATS:
            raise ValueError(f"prompt_format must be one of {sorted(PROMPT_FORMATS)}")
        if reference_board and int(prompt_format) != 2:
            raise ValueError("reference_board needs prompt_format=2")
        if subject_crops and (int(prompt_format) != 2 or reference_board):
            raise ValueError("subject_crops needs prompt_format=2 and no reference_board")
        self.prompt_format, self.reference_board = int(prompt_format), bool(reference_board)
        self.subject_crops, self.subject_detector = bool(subject_crops), subject_detector
        self._cropper = None
        self.endpoints: list[str] = []
        self._pool: queue.Queue[str] | None = None

    def start(self) -> None:
        if self._pool is not None:
            return
        if not self.endpoints_spec:
            raise RuntimeError(f"no MiniMax-H3 endpoints: pass endpoints= or set ${ENDPOINTS_ENV}")
        urls = discover_endpoints(self.endpoints_spec)
        if not urls:
            raise RuntimeError(f"no healthy MiniMax-H3 endpoint in {self.endpoints_spec!r}")
        self.endpoints = urls
        self._pool = queue.Queue()
        for u in urls:
            self._pool.put(u)

    def stop(self) -> None:
        """Servers are shared and started out of band (serve.sh); only drop the pool."""
        self._pool = None

    def prompt_for(self, request: RealizationRequest, n_sent: int | None = None) -> str:
        n = len(request.references) if n_sent is None else n_sent
        if self.prompt_format == 1:
            return build_h3_prompt(request.prompt, n, duration_sec=self.duration_sec)
        board = len(request.references) if self.reference_board and n == 1 and len(request.references) > 1 else 0
        return build_h3_prompt_v2(request.prompt, n, duration_sec=self.duration_sec, board_stills=board)

    def sent_references(self, request: RealizationRequest) -> list[str | Path]:
        """What the server receives: the stills, or one board tiling them (``reference_board``)."""
        refs = list(request.references)
        if self.reference_board and len(refs) > 1:
            board = Path(request.out_dir) / f"seg_{int(request.segment_id):03d}_refboard.png"
            return [write_reference_board(refs, board)]
        return refs

    def subject_body(self, request: RealizationRequest, out: Path) -> dict[str, Any]:
        """``subject_crops``: the stills become labelled crops of what this instruction names."""
        if self._cropper is None:
            from vmem_bench.testbed.backends.minimax_h3.subjects import SubjectCropper

            self._cropper = SubjectCropper(self.subject_detector)
        stem = f"seg_{int(request.segment_id):03d}"
        crops = self._cropper.crops(list(request.references), normalize_instruction(request.prompt),
                                    Path(request.out_dir) / f"{stem}_subjects", stem)[:MAX_REFERENCES]
        (Path(request.out_dir) / f"{stem}_subjects.json").write_text(json.dumps(
            [{"crop": str(c.path), "label": c.label, "source": str(c.source), "box": c.box} for c in crops],
            indent=1), encoding="utf-8")
        return build_request_body(
            request, out=out, prompt=build_h3_prompt_subjects(request.prompt, [c.label for c in crops],
                                                              duration_sec=self.duration_sec),
            num_frames=self.num_frames, fps=self.fps, geometry=self.geometry, steps=self.steps,
            ref_image_short_edge=self.ref_image_short_edge, references=[c.path for c in crops],
        )

    def body_for(self, request: RealizationRequest) -> dict[str, Any]:
        if len(request.references) > self.max_references:
            raise ValueError(f"{len(request.references)} references > max_references={self.max_references}")
        out = Path(request.out_dir) / f"seg_{int(request.segment_id):03d}.mp4"
        if self.subject_crops and request.references:
            Path(request.out_dir).mkdir(parents=True, exist_ok=True)
            return self.subject_body(request, out)
        refs = self.sent_references(request)
        return build_request_body(
            request, out=out, prompt=self.prompt_for(request, len(refs)), num_frames=self.num_frames,
            fps=self.fps, geometry=self.geometry, steps=self.steps, ref_image_short_edge=self.ref_image_short_edge,
            references=refs,
        )

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(url + "/v1/realize", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with _OPENER.open(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MiniMax-H3 {url} HTTP {exc.code}: {detail}") from exc
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"MiniMax-H3 {url} request failed: {type(exc).__name__}: {exc}") from exc

    def realize(self, request: RealizationRequest) -> RealizationResult:
        if self._pool is None:
            self.start()
        assert self._pool is not None
        body = self.body_for(request)
        Path(request.out_dir).mkdir(parents=True, exist_ok=True)
        url = self._pool.get()
        try:
            meta = self._post(url, body)
        finally:
            self._pool.put(url)
        clip = Path(meta.get("clip") or body["out"])
        if not clip.is_file() or clip.stat().st_size == 0:
            raise RuntimeError(f"MiniMax-H3 {url} reported success but {clip} is missing or empty")
        meta = {"geometry": {"name": body["geometry"], "render": [body["height"], body["width"]],
                             "delivered": [body["out_height"], body["out_width"]]},
                **meta, "endpoint": url, "backend": self.name, "prompt_format": PROMPT_FORMATS[self.prompt_format],
                "reference_board": self.reference_board, "subject_crops": self.subject_crops,
                "n_references": len(request.references), "n_sent": len(body["references"])}
        return RealizationResult(clip=clip, keyframe=None, meta=meta)
