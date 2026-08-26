# MemStrata Benchmark Package
import os as _os

# GPU nodes have no outbound network. Without offline mode, transformers.from_pretrained fires HEAD
# requests to huggingface.co on every load, hanging ~40s/file across 5 retries per config file; the
# detached perception servers (services/launch.py) never inherited the caller's HF_HUB_OFFLINE, so
# the first /detect after cast_roster stalls for minutes and the client times out (BrokenPipe) ->
# the run appears "stuck". huggingface_hub reads these flags at import, so they MUST be set before
# any transformers import: this package __init__ runs first for every `python -m vmem_bench.*`.
# Weights are always local (model-weights rule); still overridable (set to "0") to enable a download.
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
