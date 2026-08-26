# Track B Baseline Runners

These runners launch generator-in-the-loop Track B rollouts from the frozen
SUT-facing prompt streams in `assets/trackB/sut_prompts/`.

Output layout:

```text
outputs/evaluation/trackB/<system>/<story_id>/<register>/<run_tag>/
  input/                 # copied prompt stream + converted runner inputs
  logs/                  # stdout/stderr and command.sh
  review/                # generated long video(s); stage-2 scorer reads here
  trackb_manifest.json   # machine-readable run summary
```

The runners do not call the stage-2 scorer. They only generate and register
long-video artifacts.

## Stage-2 Scoring Service

Track B uses the same shared 32B judge service layer as Track A. The scorer's
task shape is different (generated segment videos instead of selected reference
images), but endpoint discovery and pooling are identical:

```bash
PYTHONPATH=benchmarks/VMem-Bench/src \
python -m vmem_bench.scoring.end2end_coverage \
  --gt benchmarks/VMem-Bench/assets/trackB/gt/<story>.json \
  --run benchmarks/VMem-Bench/outputs/evaluation/trackB/<system>/<story>/<register>/<run_tag> \
  --prompts benchmarks/VMem-Bench/assets/trackB/sut_prompts/<story>_<register>.json \
  --fleet --workers 0
```

Alternatively pass explicit endpoints with `--api-list
http://host1:8110/v1,http://host2:8110/v1`. The underlying reviewer services
are launched by `src/vmem_bench/annotation/pipeline/servers/backend/start_reviewer_pool.sh`:
H800 uses one GPU per service (`6:8110`), while A800 uses two GPUs per service
(`0+1:8110`).

## Implemented Systems

- `memstrata/run.py`: prompt-only wrapper around `memstrata.production.run`.
  It reads `sut_prompts` only, writes a minimal screenplay with empty
  `main_entities`, and runs MemStrata in bench-mode.
- `memflow/run.py`: writes a MemFlow JSONL prompt sample and temporary
  `interactive_inference.yaml`, then calls the native interactive runner.
- `memflow_sma/run.py`: same as MemFlow with `model_kwargs.SMA=True`.
- `longlive_rag/run.py`: calls LongLive-RAG `interactive_inference.py` with
  `--prompts_file` and `--output_path`.
- `iamflow/run.py`: calls `python -m iamflow.run_iamflow` with
  `--data_path`, `--output_folder`, and `--mapping_path` overrides.

## Environment Mapping

Use the same library stacks that already passed Track A minismoke. Override the
interpreter with `--python` / `WAN_PYTHON` if needed (default `python3`).

| system | stack | notes |
|---|---|---|
| `memstrata` | CPython 3.11 + torch; SAM3 vendored bundle on `PYTHONPATH` | Sibling `../MemStrata/src` or `MEMSTRATA_SRC` |
| `memflow` | torch 2.6 + flash-attn 2.6 | Same as Track A MemFlow |
| `memflow_sma` | same | Native MemFlow with `model_kwargs.SMA=True` |
| `longlive_rag` | torch (VAE + AE) | Same as Track A LongLive-RAG |
| `iamflow` | torch 2.5 + flash-attn; optional vLLM for Qwen | Full Track B VLM mode needs a working vLLM service; `no_vlm_hf` smokes must not be reported as full IAMFlow |

For causal Wan-style runners, keep `--frames-per-segment` block-aligned
(`39` in the 2026-07-26 smoke) because the native pipelines generate in
`num_frame_per_block=3` blocks. The Track B smoke also uses an `imageio` H.264
fallback when torchvision/PyAV video writing is missing or broken in the target
environment.

## IAMFlow Services

Full IAMFlow Track B runs should offload Qwen3-4B and Qwen3-VL-2B to long-lived
OpenAI-compatible vLLM services on the same node. Do not reload these models per
story.

```bash
# On each kml-a800 node that will run IAMFlow workers:
cd .
IAMFLOW_SERVICE_GPU=6 \
IAMFLOW_VLLM_PY=python3 \
IAMFLOW_LLM_PORT=8100 \
IAMFLOW_VLM_PORT=8101 \
bash benchmarks/VMem-Bench/scripts/evaluate_baselines/trackB/baseline_runners/iamflow/launch_vllm_services.sh

source benchmarks/VMem-Bench/outputs/evaluation/trackB/_services/iamflow_vllm/latest/iamflow_service.env
```

Then launch `iamflow/run.py` in the same Python that has IAMFlow + Wan. The native
IAMFlow LLM/VLM agents read `IAMFLOW_LLM_ENDPOINT` and `IAMFLOW_VLM_ENDPOINT`;
when unset, they fall back to in-process loading, which is suitable only for tiny
smoke runs and will be too slow for full Track B.

## Examples

Dry-run one story:

```bash
python benchmarks/VMem-Bench/scripts/evaluate_baselines/trackB/baseline_runners/memstrata/run.py \
  --prompts benchmarks/VMem-Bench/assets/trackB/sut_prompts/0001_lighthouse_keeper_name_anchored.json \
  --limit 2 --run-tag smoke --dry-run
```

Real MemFlow one story:

```bash
python benchmarks/VMem-Bench/scripts/evaluate_baselines/trackB/baseline_runners/memflow/run.py \
  --prompts benchmarks/VMem-Bench/assets/trackB/sut_prompts/0001_lighthouse_keeper_name_anchored.json \
  --run-tag bench --cuda-visible-devices 0
```
