# External Baseline Audit for MemStrata-Bench

> **⚠️ SUPERSEDED (2026-07-22):** baseline 选择 / 公平性的**当前权威**是
> [`fairness_decisions.md`](fairness_decisions.md)。本文件保留为 **2026-06 的历史核验记录**
> （各系统能输出什么、是否 generator-decoupled、如何映射到 `ComposedContextRecord` 仍有参考价值）。
> **凡与 `fairness_decisions.md` 冲突处以后者为准**，尤其：
> - 本文下面把 **ViMax / MovieAgent / CoAgent** 定为 Table 1 主表候选的结论**已作废**。按 **D5**，
>   脚本化 / agentic 系统（ViMax / MovieAgent / VideoMemory / StoryMem / Memento / MM-StoryAgent）
>   为**非因果**（先出整段计划再渲染），**移出定量主表、仅进附录定性说明**。
> - 定量主表现在放**因果**系统：`helios / longlive_rag / memflow / iamflow / decmem`（其中 decmem
>   在本节点被 SM90a kernel 阻塞，见 [`track_a.md`](track_a.md)）。MemFlow / IAMFlow 已有**真实**
>   Track A retrieval trace，不再是"secondary/coupled，不进主表"。

Date: 2026-06-30 UTC.
Freshness window: 2025-06 to 2026-06, with emphasis on generator-decoupled context organizers, retrievable-memory / long-context video generation methods, and agentic video systems.

Terminology note (2026-07-16): local checkout categories were renamed from `NVG` / `SVG` to **`Scripted`** (fixed global plan / known horizon) and **`Causal`** (future prompts and horizon unknown). Older prose in this audit may still say narrative/streaming; treat those as the same two regimes.

Purpose: record which external baselines are viable for the MemStrata-Bench main comparison, what each method can output, whether it is generator-decoupled, and how its artifacts can be converted into the benchmark's `ComposedContextRecord`.

Core comparison boundary:

```text
Main claim: passive retrieval over flat history vs hierarchical memory management for causal long video generation.
Not main targets: sliding window, full history, history compression, selection oracle.
```

Baseline selection rule: an external method enters the main table only if it has an accessible, non-empty, plausibly runnable implementation and can be fairly adapted to the same data/protocol. Otherwise it is recorded as a candidate, fallback, or exclusion.

Track A Table 1 decision, finalized on 2026-06-30 and clarified on 2026-07-02: the main context-quality comparison table should include only generator-decoupled external context organizers for video generation. Current Track A candidates are ViMax, MovieAgent, and CoAgent if runnable code becomes available. Track B is separate: generator-in-the-loop rollout compares external long-video generation frameworks such as StoryMem, MemFlow, Helios, LongLive-RAG, LongLive, Memento, IAMFlow, and DecMem against Montage with its plugged-in generation backends.

## 0. Fit Criterion: External Context Organizer vs Generator-Coupled Memory

The user's comparison target is now stricter than "anything with memory/retrieval":

```text
Preferred baseline = external context organizer
```

Required properties:

- Generator-decoupled or at least generator-agnostic in principle.
- Maintains explicit history/context/asset/reference state outside the generator backbone.
- Chooses or constructs the next chunk's external context before generation.
- Exposes intermediate artifacts such as selected references, character/scene bank entries, shot-level context, retrieved frames, or reference prompts.
- Can be run under gold-replay / generator-free evaluation, or at least can export decisions without relying on the final generated video quality.

Methods that retrieve internal latents, KV cache, sparse attention chunks, or generator-specific tokens are useful retrieval-memory baselines, but they are **not ideal primary baselines** for MemStrata's main claim because they are coupled to a specific generator architecture or latent space.

## 1. Candidate Status Summary

| Candidate | Role | Generator-decoupled? | Code status | Current decision | Reproduction risk | Main reason |
|---|---|---:|---|---|---|
| ViMax | External agentic context/reference organizer | yes/mostly | official GitHub | run-check priority A | medium | Exposes per-shot reference-selection artifacts and can be adapted generator-free |
| MovieAgent | External agentic planner/context organizer | yes/partly | official GitHub | run-check priority A/B | medium-high | Training-free agentic pipeline; need artifact inspection for selected character/reference context |
| CoAgent | External global context manager candidate | likely yes conceptually | no usable GitHub found | watchlist | blocked | Paper explicitly describes a Global Context Manager, but code availability is not verified |
| MemFlow | Generator-coupled retrieval-memory method | no | official GitHub + checkpoints | secondary retrieval baseline, not main organizer | medium-high | Retrieves historical frames and activates memory tokens inside streaming generator/KV-cache pathway |
| LongLive-RAG | Generator-coupled latent retrieval method | no/weak | official GitHub + checkpoints | secondary retrieval baseline, not main organizer | medium-high | Retrieves latent entries inside an AR generator context; plug-and-play across some AR backbones but still latent/backbone-coupled |
| Context-as-Memory | Retrieval-memory candidate | maybe partly | project page says GitHub coming soon | watchlist / excluded for main table now | blocked | Relevant idea, but no usable code as of this audit |
| MM-StoryAgent | Lightweight story/agent baseline | yes/partly | official GitHub | optional / run-check priority C | medium | Runnable repo but storybook modality differs from long-video production memory |
| EgoLCD | Long-context video generation | unclear | GitHub shell only | excluded | blocked | Repository appears to contain only README as of audit |
| VideoDirectorGPT | External planning/layout baseline | yes, but no history memory | code appears available via project/GitHub ecosystem, needs re-check | weak-match watchlist | medium/unknown | External LLM planner, but it plans scenes/layouts rather than organizing generated history context |
| StoryDiffusion / StoryMaker | Consistency or personalization generation systems | no/weak | GitHub exists | excluded from main table | medium | Useful related work, but not external long-context organizers |
| StoryMem | Complete multi-shot long video storytelling framework | no, full generator framework | official GitHub + HF weights | Track B rollout baseline | medium-high | Provides ST-Bench and memory-to-video generation; useful as generator-in-the-loop baseline, not Track A context organizer |
| Helios | Complete real-time long video generation framework | no, full generator framework | official GitHub + HF weights | Track B rollout baseline | medium | Strong efficient long-video generator baseline |
| LongLive | AR interactive long video generation framework | no, full generator framework | official GitHub + HF weights | Track B rollout baseline / dependency backbone | medium-high | Important backbone for LongLive-RAG and MemFlow-style baselines |
| Memento | Multi-shot narrative video with reconstruct-to-remember memory | no, full generator framework | official GitHub + HF weights | Track B rollout baseline | medium | Direct memory-based narrative video baseline |
| IAMFlow | Training-free identity-aware memory framework | no, full generator framework | official GitHub | Track B rollout baseline | medium | Identity consistency baseline for evolving narrative prompts |
| DecMem | Decoupled memory for minute-long world generation | no, full generator framework | official GitHub + HF weights | Track B rollout baseline | medium | Sparse global memory / anchored local memory baseline |
| Mixture of Contexts / OmniMem / Context Forcing / MemCam / PermaVid | Internal or specialized memory candidates | mostly no/unclear | no selected usable external-organizer repo found in quick audit | watchlist | blocked/unknown | Relevant memory papers, but not ideal generator-decoupled context organizers or not yet code-verified |

## 2. Evidence Table

| Method | Evidence | GitHub reproducibility fields | Selection note |
|---|---|---|---|
| MemFlow | Project page links Paper/GitHub/HF. GitHub repo `KlingAIResearch/MemFlow` has `model/`, `pipeline/`, `trainer/`, `inference.py`, `interactive_inference.py`, `train.py`, scripts, requirements, Apache-2.0 license, and HF checkpoints. README states it retrieves relevant historical frames with the coming chunk prompt, but also activates relevant tokens in the memory bank inside attention layers and is compatible with streaming video generation models with KV cache. | repo_exists=yes; repo_nonempty=yes; method_code_present=yes; train_script_present=yes; eval/inference_script_present=yes; requirements_present=yes; checkpoint_present=yes; paper_to_code_mapping_clear=yes; score=4 | Strong retrieval-memory baseline, but not the ideal main comparison because it is generator/KV-cache coupled. Use as secondary retrieval baseline or separate "generator-coupled memory" table. |
| LongLive-RAG | GitHub repo `qixinhu11/LongLive-RAG` has `ae/`, `configs/`, `pipeline/`, `wan/`, `inference.py`, `generate_latent.py`, scripts, requirements, Apache-2.0 license, and HF assets. README says it inserts retrieved historical latent entries `M_t` into an AR generator context between sink and local windows; the base generator stays frozen but the method operates in the generator's latent space. | repo_exists=yes; repo_nonempty=yes; method_code_present=yes; train_script_present=yes; eval/inference_script_present=yes; requirements_present=yes; checkpoint_present=yes; paper_to_code_mapping_clear=yes; score=4 | Strong latent retrieval baseline, but not an external context organizer. Use as secondary evidence against retrieval if adapter is feasible; do not make it the central Table 1 baseline. |
| Context-as-Memory | Project page states GitHub "Coming Soon" and describes FOV-overlap memory retrieval over historical context frames. Dataset link exists. | repo_exists=no usable code; score=0 | Exclude from main table until code is released. Keep as related work / watchlist. |
| ViMax | GitHub repo `HKUDS/ViMax` has agent runtime, agents, configs, pipelines, tools, tests, pyproject, uv.lock, MIT license, and runnable entrypoints. README documents reference-image selection, asset indexing, character/environment tracking, and working directory artifacts. | repo_exists=yes; repo_nonempty=yes; method_code_present=yes; run instructions=yes; requirements_present=yes via pyproject/uv; score=4 | Strong external agentic system. Adapter already planned around `shots/*/*_frame_selector_output.json`. Not a pure retrieval baseline; use as external system comparison. |
| MovieAgent | GitHub repo `showlab/MovieAgent` has `movie_agent/`, dataset, requirements, and installation/inference guidance. README says inference code is released and training-free. | repo_exists=yes; repo_nonempty=yes; method_code_present=yes; requirements_present=yes; run instructions=partial; score=3 | Candidate external agentic system. Need run-check and artifact inspection: scene plans, camera settings, character selections, generated prompts. |
| MM-StoryAgent | GitHub repo `X-PLUG/MM_StoryAgent` has `mm_story_agent/`, `story_eval/`, configs, requirements, `run.py`, setup, Apache-2.0 license. | repo_exists=yes; repo_nonempty=yes; method_code_present=yes; requirements_present=yes; eval/run_script_present=yes; score=3 | Optional external baseline. It targets narrated storybook video, so task match is weaker. |
| CoAgent | ArXiv paper describes a Storyboard Planner, Global Context Manager, Visual Consistency Controller, and Verifier Agent for coherent video generation. Quick search did not find a usable official GitHub implementation. | repo_exists=not verified; score=0 | Conceptually close external context organizer; watchlist only until code appears. |
| VideoDirectorGPT | Paper proposes an LLM video planner producing scene descriptions, entities, layouts, backgrounds, and consistency groups for Layout2Vid. | repo status needs re-check for current availability; score pending | Good external planning reference, but weak main-baseline fit because it does not manage a growing history/asset bank from generated chunks. |
| StoryDiffusion / StoryMaker | Public code exists for consistency/image-video story generation and character consistency, respectively. | repo_exists=yes for known projects; score not assigned here | Related consistency baselines, but not external history context organizers. Use in related work or qualitative comparisons, not Table 1 main context construction. |
| EgoLCD | GitHub repo `AIGeeksGroup/EgoLCD` currently shows only README and minimal commits. | repo_exists=yes; repo_nonempty=no; score=1 | Exclude for now; no method implementation visible. |
| Mixture of Contexts / OmniMem / Context Forcing | Search surfaced papers, but no verified runnable repo selected in this audit. | github_unknown or no usable code; score=0-1 | Watchlist only; do not use in Table 1 until code and exportable outputs are verified. |

## 3. What Each Viable External Method Can Output

| Method | Native output likely available | What we need to extract | Mapping to MemStrata-Bench |
|---|---|---|---|
| MemFlow | Generated video chunks; memory bank; text-prompt-driven retrieved historical frames; attention/token memory activation | For each chunk: prompt, candidate memory frame ids, selected memory frame ids, optionally token/block ids, generated chunk path | Secondary table only: `selected_assets` = selected frame/block references; `instruction` = next prompt; role/lifecycle/forbidden fields absent unless method exposes them |
| LongLive-RAG | Generated latent blocks; retrieved historical latent entries `M_t`; configs for native vs latentmem | For each generated block: retrieved latent ids, corresponding source chunk/frame ids if recoverable, retrieval score | Secondary table only: `selected_assets` = retrieved latent/frame blocks; `instruction` = prompt; role/lifecycle/forbidden absent |
| ViMax | Working directory with characters, portraits, shot descriptions, frame selector outputs, prompts, references | For each shot/chunk: `reference_image_path_and_text_pairs`, text prompt, character ids, camera/scene references | `selected_assets` from reference paths; partial role from path type; instruction from per-image text and prompt; lifecycle/forbidden absent unless explicit |
| MovieAgent | Multi-agent planning outputs, scene/camera/character plans, possibly selected character photos/prompts | Need run-check: locate per-shot character/reference selection and prompt artifacts | Convert selected characters/references to `selected_assets`; use shot prompt as instruction; roles partial |
| MM-StoryAgent | Story plan, generated images/audio/video, tool outputs, story evaluation artifacts | Need run-check: locate per-scene image/reference selections and story prompts | Likely only weak mapping; use as optional qualitative/external baseline if artifacts align |

Important adapter rule: do not hand-fill missing MemStrata fields for external methods. If a method does not expose lifecycle, forbidden assets, or typed use functions, leave them absent and let the metric report the gap. This avoids artificially penalizing or helping baselines.

## 4. Recommended Baseline Sets

Table 1 finalized scope:

1. ViMax.
2. MovieAgent.
3. CoAgent if runnable code is found; otherwise mark as code-unverified and do not report numbers.
4. MemStrata.

Fallback policy: do not put MemFlow / LongLive-RAG in Track A Table 1 as external context organizers. Use them as Track B generator-in-the-loop baselines and, if retrieval ids can be exported, as secondary generator-coupled retrieval-memory evidence.

### Minimal Set

Use if time is tight:

1. ViMax (external context/reference organizer).
2. MovieAgent if artifact inspection passes.
3. MemStrata.

### Standard Set

Use for the paper main table:

1. ViMax.
2. MovieAgent.
3. CoAgent if code becomes available.
4. MemStrata.

### Defensive Set

Use if reviewer-risk tolerance is low:

1. ViMax.
2. MovieAgent.
3. CoAgent if official code becomes available.
4. MM-StoryAgent or another runnable generator-decoupled context organizer if its output can be normalized.
5. Track B generator-in-the-loop rollout: StoryMem / MemFlow / Helios / LongLive-RAG / LongLive / Memento / IAMFlow / DecMem vs Montage+{VACE, LongCat-Video, LTX-2.3, MultiShotMaster}.
6. Appendix-only diagnostics: sliding-window, full-history, selection-oracle.

## 5. Reproduction Plan

### Priority A: ViMax

1. Use existing ViMax run kit.
2. Run generator-free with oracle frame injection and local VLM.
3. Use existing `vimax_to_records(...)` adapter.
4. Verify id_map quality manually on 2-3 samples.

### Priority A/B: MovieAgent

1. Install repo and run one official example.
2. Locate per-shot plans / selected character references / generated prompt files.
3. If artifacts are structured enough, add `movieagent_to_records(...)`; otherwise use normalized JSON export manually once per run.

### Secondary: MemFlow

1. Vendor or clone `KlingAIResearch/MemFlow` under `benchmarks/MemStrata-Bench/baselines/MemFlow/` with license.
2. Install in its own environment; expected hardware is 80GB-class GPU according to README.
3. Run official `interactive_inference.sh` on a tiny prompt sequence.
4. Patch or wrap retrieval module to log per-chunk selected historical frame ids and retrieval scores.
5. Write `memflow_to_records(...)` that maps selected frame ids to benchmark asset ids.

### Secondary: LongLive-RAG

1. Vendor or clone `qixinhu11/LongLive-RAG` with license.
2. Download official HF checkpoints/toy latent set.
3. Run same-machine native vs latentmem inference to confirm repo works.
4. Instrument latent retrieval stage to log selected memory block ids.
5. Write `longlive_rag_to_records(...)`; treat latent blocks as visual memory assets.

## 6. Open Questions

1. Can we find another runnable external context organizer closer than ViMax/MovieAgent?
2. Should MemFlow and LongLive-RAG be in a separate "generator-coupled retrieval memory" table instead of Table 1?
3. Can latent/frame retrieval baselines be fairly scored against asset-level GT, or should they have a separate retrieval-oriented metric panel?
4. For methods without typed roles/lifecycle, should missing fields be zero-scored or "not applicable" with a separate relevance-only score? The main claim suggests zero/absent is acceptable, but this must be stated carefully.

## 6.1 Current Interpretation

As of this audit, the best match for MemStrata's main comparison is not "any method with memory", but methods that organize external context before generation. Under that stricter definition:

- `ViMax` and `MovieAgent` are currently more appropriate main-table candidates than MemFlow / LongLive-RAG because they expose external planning/reference-selection artifacts.
- `MemFlow` and `LongLive-RAG` are still valuable, but they should be framed as generator-coupled retrieval-memory baselines. They can answer "does internal retrieval memory solve the problem?", but not "does an external context organizer match MemStrata?"
- `CoAgent` is conceptually close because it explicitly names a Global Context Manager, but it cannot enter the selected set without usable code.
- `VideoDirectorGPT` is an external planner but not a history-context manager; it may be useful in related work or as a weak planning baseline if code/runs are easy.
- `StoryDiffusion` / `StoryMaker` are consistency-generation or personalization methods, not context organizers.

## 7. Source Links

- MemFlow paper: https://arxiv.org/abs/2512.14699
- MemFlow project: https://sihuiji.github.io/MemFlow.github.io/
- MemFlow GitHub: https://github.com/KlingAIResearch/MemFlow
- LongLive-RAG GitHub: https://github.com/qixinhu11/LongLive-RAG
- StoryMem GitHub: https://github.com/Kevin-thu/StoryMem
- Helios GitHub: https://github.com/PKU-YuanGroup/Helios
- LongLive GitHub: https://github.com/NVlabs/LongLive
- Memento GitHub: https://github.com/ernie-research/Memento
- IAMFlow GitHub: https://github.com/Eddie0521/IAMFlow
- DecMem GitHub: https://github.com/KlingAIResearch/DecMem
- Context-as-Memory project: https://context-as-memory.github.io/
- ViMax GitHub: https://github.com/HKUDS/ViMax
- MovieAgent GitHub: https://github.com/showlab/MovieAgent
- MM-StoryAgent GitHub: https://github.com/X-PLUG/MM_StoryAgent
- CoAgent paper: https://arxiv.org/abs/2512.22536
- VideoDirectorGPT paper: https://arxiv.org/abs/2309.15091
- StoryDiffusion paper: https://arxiv.org/abs/2405.01434
- StoryMaker paper: https://arxiv.org/abs/2409.12576
- MemCam paper: https://arxiv.org/abs/2603.26193
- Mixture of Contexts paper: https://arxiv.org/abs/2508.21058
- PermaVid paper: https://arxiv.org/abs/2606.16449
- EgoLCD GitHub: https://github.com/AIGeeksGroup/EgoLCD
