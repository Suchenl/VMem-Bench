#!/usr/bin/env python3
"""LocateAnything-3B grounding service for the MiniMax-H3 ``subject_crops`` option (separate process/env).

The model's remote code needs transformers 4.57.x (not the H3 server/runner env), and its weights are under
the NVIDIA non-commercial research license, so it runs here and the testbed talks to it over stdlib HTTP
(``vmem_bench.testbed.backends.minimax_h3.subjects``).

    POST /detect {image_path, phrases: [..]}
      -> {answer, width, height, boxes: [{label, box: [x0, y0, x1, y1] px}]}
    GET  /health -> {ok: true, model: "locateanything-3b"}

Deterministic within one stack: greedy "slow" (autoregressive) decoding (``temperature=0``; the remote code
ignores ``do_sample`` and samples whenever temperature > 0), SDPA attention (the release config names MagiAttention,
which is not installed; its LM implements only magi|sdpa), one request at a time.

    CUDA_VISIBLE_DEVICES=<gpu> <python with transformers 4.57.x> serve.py --model <LocateAnything-3B dir> --port 18480
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "locateanything-3b"
_STATE: dict = {}
_LOCK = threading.Lock()


def load(path: str) -> None:
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    for sub in ("vision_config", "text_config", "llm_config"):
        if getattr(cfg, sub, None) is not None:
            getattr(cfg, sub)._attn_implementation = "sdpa"
    _STATE["tok"] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    _STATE["proc"] = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    _STATE["model"] = AutoModel.from_pretrained(path, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                attn_implementation="sdpa").cuda().eval()


def parse(answer: str, width: int, height: int) -> list[dict]:
    """``<ref>label</ref><box><x0><y0><x1><y1></box>...`` on a 0-1000 grid -> pixel boxes with their label."""
    out, label = [], ""
    for m in re.finditer(r"<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        if m.group(1) is not None:
            label = m.group(1).strip()
        else:
            x0, y0, x1, y1 = (int(m.group(i)) for i in range(2, 6))
            out.append({"label": label, "box": [x0 / 1000 * width, y0 / 1000 * height,
                                                x1 / 1000 * width, y1 / 1000 * height]})
    return out


def detect(image_path: str, phrases: list[str]) -> dict:
    import torch
    from PIL import Image

    im = Image.open(image_path).convert("RGB")
    q = "Locate all the instances that matches the following description: " + "</c>".join(phrases) + "."
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": q}]}]
    proc, tok, model = _STATE["proc"], _STATE["tok"], _STATE["model"]
    text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = proc.process_vision_info(msgs)
    x = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
    with torch.no_grad():
        r = model.generate(pixel_values=x["pixel_values"].to(torch.bfloat16), input_ids=x["input_ids"],
                           attention_mask=x["attention_mask"], image_grid_hws=x.get("image_grid_hws"), tokenizer=tok,
                           max_new_tokens=1024, use_cache=True, generation_mode="slow", do_sample=False,
                           temperature=0.0, repetition_penalty=1.0, verbose=False)  # remote code samples iff temperature > 0
    answer = r[0] if isinstance(r, tuple) else r
    answer = answer if isinstance(answer, str) else str(answer)
    return {"answer": answer, "width": im.width, "height": im.height, "boxes": parse(answer, im.width, im.height)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(200, {"ok": True, "model": MODEL_ID}) if self.path == "/health" else self._send(404, {})

    def do_POST(self) -> None:
        if self.path != "/detect":
            return self._send(404, {})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            phrases = [str(p).strip() for p in req.get("phrases") or [] if str(p).strip()]
            if not req.get("image_path") or not phrases:
                return self._send(400, {"error": "image_path and phrases required"})
            with _LOCK:
                self._send(200, detect(req["image_path"], phrases))
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18480)
    a = ap.parse_args()
    load(a.model)
    print(f"[locateanything] ready at http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
