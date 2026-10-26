"""MiniMax-H3 Ref2V + 4-step Turbo LoRA realization backend (references go straight into the video model).

``client`` and ``prompt`` are dependency-free; ``server`` needs FastAPI/uvicorn and ``pipeline`` needs
DiffSynth-Studio with MiniMax-H3 support, both only inside the resident server process.
"""

from vmem_bench.testbed.backends.minimax_h3.client import MiniMaxH3TurboBackend, discover_endpoints
from vmem_bench.testbed.backends.minimax_h3.prompt import build_h3_prompt, normalize_instruction

__all__ = ["MiniMaxH3TurboBackend", "build_h3_prompt", "discover_endpoints", "normalize_instruction"]
