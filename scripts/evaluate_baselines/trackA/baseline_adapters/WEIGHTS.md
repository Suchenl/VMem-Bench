# baseline_adapters — baseline 权重与环境登记

> 本文件记录各因果 baseline 的**权重绝对路径**、共享 backbone、以及推荐的 Python 依赖栈。
> 采集日期 2026-07-24。改动权重位置时更新此处，不要在 adapter 里硬编码散落路径。

## 共享 backbone / 辅助模型

| 用途 | 路径 |
|---|---|
| Wan2.1-T2V-1.3B（DiT+VAE+T5，MemFlow/IAMFlow/LongLive 共用 backbone） | `${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.1-T2V-1.3B` |
| Wan2.1-T2V-1.3B（LongLive vendored 已下好的等价副本） | `baselines/Causal/LongLive-RAG/wan_models/Wan2.1-T2V-1.3B` |
| Qwen3-4B-Instruct-2507（IAMFlow LLM） | `${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-4B-Instruct-2507` |
| Qwen3-VL-2B-Instruct（IAMFlow 帧选 VLM） | `${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-2B-Instruct` |
| Qwen3-VL-32B（打分 judge，见 §打分） | 见下 §打分 |

## 各 baseline 权重

| baseline | 权重根 | 关键文件 |
|---|---|---|
| LongLive-RAG | `.../Causal_Video_Generation/LongLive-RAG/checkpoints` | `longlive_base.pt` / `longlive_lora.pt` / `ae_latent_mem.pt`（已就位于 vendored repo） |
| MemFlow | `${PUBLIC_MODELS_ROOT}/KlingTeam/MemFlow` | `base.pt` / `lora.pt` |
| IAMFlow | `${PUBLIC_MODELS_ROOT}/Causal_Video_Generation/IAMFlow` | `iamflow_fp8.safetensors` / `tinyvae.pth` |
| SlotMem | LoRA/encoder：`${PUBLIC_MODELS_ROOT}/Causal_Video_Generation/SlotMem/ckpt/{stage1,stage2}`；base：`${PUBLIC_MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B` | TrackA only uses native Wan2.2-I2V-A14B + `ckpt/{stage1,stage2}/{stage*_low.pt,stage*_high.pt}`. Do **not** use distilled/lightx2v Wan2.2 for SlotMem formal runs; the smoke can load LoRA but visual quality is unusable. |
| MemStrata（本系统，感知权重） | `${PUBLIC_MODELS_ROOT}` | 默认 grounder：**WeDetect-Ref**（describe→bbox 服务，`MEMSTRATA_WEDETECT_URL`，需另行部署）；DINOv3：`facebook/dinov3-vitb16-pretrain-lvd1689m`；SAM3（仅 WeDetect 不可用时回退）：`facebook/sam3`。`name_source=mllm` 另需 `Qwen/Qwen3.5-9B-Instruct`。下载与布局见 MemStrata 仓库 `MODELS.md`。 |
| （备）DecMem | `${PUBLIC_MODELS_ROOT}/KlingTeam/DecMem` | 需 H100/WorldMem，暂不接 |

## Python / library mapping

| baseline | stack | 备注 |
|---|---|---|
| LongLive-RAG | torch 2.5 + einops | 仅用 VAE+AE，绕开 DiT/flash_attn |
| MemFlow | torch 2.6 + flash-attn 2.6 | 已验证 forward |
| IAMFlow | torch 2.5 + flash-attn；可选独立 vLLM 进程跑 Qwen | `iamflow_fp8.safetensors` 在 adapter 内反量化成 bf16，可在 SM80 上跑 |
| SlotMem | torch 2.5 + flash-attn 2.8（vendored diffsynth，非 lightx2v） | TrackA 正式实验禁止 distilled/lightx2v Wan2.2 |
| MemStrata | CPython 3.11 + torch；SAM3 vendored `models/vendor/sam3_transformers59` | GPU 验证：BBB name_anchored limit=4 |

## backbone/ckpt 接线方式（保持 vendored repo 零改动）

各 vendored repo 期望权重在其**仓库根下的相对路径**（如 MemFlow 期望 `wan_models/Wan2.1-T2V-1.3B/` 与
`checkpoints/base.pt`）。统一用 **symlink** 从上表绝对路径接进去，不拷贝、不改源码：

- MemFlow: `baselines/Causal/MemFlow/wan_models/Wan2.1-T2V-1.3B` → Wan backbone；`.../MemFlow/checkpoints/{base.pt,lora.pt}` → KlingTeam/MemFlow
- IAMFlow: `baselines/Causal/IAMFlow/pretrained/{Wan2.1-T2V-1.3B, iamflow_models/iamflow_fp8.safetensors, Qwen3-4B-Instruct-2507, Qwen3-VL-2B-Instruct}` → 上表路径
- SlotMem: `causal/slotmem.py` 直接接绝对路径 native Wan2.2-I2V-A14B + SlotMem `ckpt/{stage1,stage2}`；vendored repo 零改动。正式 TrackA 不使用 distilled/lightx2v Wan2.2。
- MemStrata: 无需 vendored symlink；感知权重按 `PUBLIC_MODELS_ROOT` 解析。跑法（本地 gpu0）：

```bash
SAM3=./models/vendor/sam3_transformers59
SRC=./src
cd scripts/evaluate_baselines/trackA/baseline_adapters/causal   # from the VMem-Bench repo root
PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT} \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$SAM3:$SRC:$(pwd)" \
python3 -u runner.py \
  --adapter memstrata --movie-dir ../../../data/BlenderOpenMovies/big_buck_bunny \
  --limit 4 --input-mode name_anchored
# 产物：<movie>/benchmark_run/visual_selections/memstrata.json（+ _ref_frames/memstrata/*.png）
```

> PYTHONPATH 里 SAM3 bundle 前置以拿到 transformers 5.9；adapter `reset` 会把 `src` 提到最前并解除
> `causal/memstrata.py` 对 `memstrata` 包的同名遮蔽（否则 `import memstrata.bank` 会解析到 adapter 文件）。

## 打分 judge（Qwen3-VL-32B）—— 已接通并验证

权重：`${PUBLIC_MODELS_ROOT}/Qwen/Qwen3-VL-32B-Instruct`
栈：torch + vLLM。`scoring.visual_coverage` 默认 `127.0.0.1:8110`、model `qwen3-vl-32b`。

**起服务**（共享节点，加载 64GB 权重时若被外部任务抢卡会 OOM，重试即可；用 `nvidia-smi` 选空闲卡）：

```bash
NVJIT=$(python3 -c "import pathlib,sys;p=pathlib.Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'/'nvidia'/'nvjitlink'/'lib';print(p if p.is_dir() else '')")
CUDA_VISIBLE_DEVICES=<free_gpu> LD_LIBRARY_PATH=$NVJIT NO_PROXY=localhost,127.0.0.1,0.0.0.0 PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 -m vllm.entrypoints.openai.api_server --model $Qwen3-VL-32B-Instruct \
  --served-model-name qwen3-vl-32b --host 0.0.0.0 --port 8110 --tensor-parallel-size 1 \
  --max-model-len 28672 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":96,"video":1}' --allowed-local-media-path /data --trust-remote-code
```

- `--limit-mm-per-prompt image=96`：大足迹系统（MemFlow 每 segment 可达 40+ 参考图）需要更高的单请求图上限（默认 24 不够）。
- `--max-model-len 28672`：0.92 显存利用率下 KV cache 可容纳的稳妥上限（32768 会 KV 不足）。

**跑打分**（`NO_PROXY` 直连本地 judge）：

```bash
cd VMem-Bench
NO_PROXY=localhost,127.0.0.1 CUDA_VISIBLE_DEVICES=<free_gpu> PYTHONPATH=src \
python3 -m vmem_bench.scoring.visual_coverage \
  --movie data/BlenderOpenMovies/big_buck_bunny --system <memflow|longlive_rag|...> \
  --video <source_video.mp4> [--limit N]
```

本轮对 scorer 做的三处稳健性修复（`src/vmem_bench/scoring/visual_coverage.py`）：
1. `run()` 把 `--movie/--video` 解析为绝对路径 —— 否则 clip 的 `file://` 相对路径服务端解析不到（500）。
2. `_call` 把服务端错误体带进异常信息 —— 便于定位（原来只报裸 `HTTP 500`）。
3. `_img` 客户端下采样到 `JUDGE_IMG_MAX_SIDE=384` 并 base64 内嵌 —— 统一、确定性地压 token，让 MemFlow 这类
   大足迹系统的 40+ 参考图能进上下文（服务端 `mm_processor_kwargs.max_pixels` 对 Qwen3-VL 无效）。

**已验证结果**（big_buck_bunny，judge=qwen3-vl-32b，视频/参考图统一 480p 见 running_eval §3.1）：
- LongLive-RAG（8 segment）：prec 0.74 / rec 0.72 / f1 0.69 / redun_vlm 0.58 / redun_sim 0.92 / eff 0.24 / budget 6.0
- MemFlow (w/o SMA)（5 segment）：prec 0.44 / rec 0.67 / f1 0.62 / redun_vlm 0.62 / redun_sim 0.79 / eff 0.04 / budget 31.5
- MemFlow (with SMA)（5 segment）：prec 0.56 / rec 1.00 / f1 0.84 / redun_vlm 0.88 / redun_sim 0.83 / eff 0.06 / budget 29.5
  - **两变体写入完全相同**（`compress_kv_bank` 文本显著性 top-k），**只差读取**：w/o SMA 用整块 bank；with SMA 用 φ 紧凑描述子路由 top-3 chunk（`dynamic_topk_routing_attention`，`memflow.py` 的 `sma` 开关）。φ 路由聚焦更相关的历史块 → 精度/召回/F1 更高、帧数略少。adapter 名：`memflow` / `memflow_sma`。
- IAMFlow（15 segment，6 个有召回）：prec 1.00 / rec 0.22 / f1 0.65 / redun_vlm 0.04 / redun_sim 0.72 / eff 0.96 / budget 3.0（有召回段均值）/ 1.2（全段均值）
  - rec 偏低是因为前 9 个 chunk 无角色复现、召回=0 被计入 15 块均值；仅看有召回块 rec 明显更高。IAMFlow 特点：**少而准、几乎零冗余**（与 MemFlow 的大足迹高冗余正相反）。

**两种冗余口径（redun_vlm vs redun_sim）**：
- `redundancy_vlm`：VLM 口径。对每个「同实体」参考图组问判官「有几个视觉上互不相同的 view」，`冗余数 = 组大小 − distinct_views`，再对全 segment 归一。产出**整数计数**，`efficiency` 用它。
- `redundancy_sim`：**非 VLM 口径**。同一分组内两两算 DINOv3 CLS 余弦、取 off-diagonal 均值（无阈值，unit-norm 下 cos=dot），再按对数加权平均。1.0=选帧几乎相同（全冗余）；越低=视角越多样。
- 二者分组一致（都用判官判定的 present + entity_id），只是「冗余程度」的度量不同：一个靠 VLM 语义判重、一个靠 DINOv3 特征相似度。redun_sim 由 `scripts/baselines/recompute_redundancy_sim.py` 离线补算（复用 details.json 的分组，仅需 DINOv3，不重跑判官）；需 `PUBLIC_MODELS_ROOT=${PUBLIC_MODELS_ROOT}`。
- ⚠️ 跨 baseline 直接比 headline 前需注意 segment 数不一（8/5/15）；正式发布应统一 segment 范围重跑。
