# MemStrata-Bench Academic Design Principles (学术设计原则)

This document outlines the core, system-independent design principles that must be followed when building and executing academic benchmarks for long-video memory and context management. These principles ensure that the evaluation is **rigorous, reproducible, cheat-proof, and reflective of real-world performance**.

> **方法论根基（通用纲领）**：下面 12 条 MemStrata-specific 原则都可回溯到三条超越具体任务领域（长视频、NLP、多模态等）的通用学术基准法则——**关注点分离与解耦**、**确定性指标计算**、**自包含与环境解耦**。它们分别在本文的 #2/#6/#7 中被具体化，完整表述见文末《附录：通用基准方法论》。术语以 [`glossary.md`](https://github.com/Suchenl/MemStrata/blob/main/docs/glossary.md) 为准。

---

## 1. Causal Streaming Evaluation (流式因果评估原则)
- **Definition**: Long-video generation is inherently a chronological, causal process. The evaluation must proceed chunk-by-chunk ($t = 0, 1, 2, \dots$).
- **Constraint**: At any step $t$, the System Under Test (SUT) must only have access to the historical video chunks ($< t$) and the current generation prompt ($p_t$). It must never have access to future video chunks ($> t$) or future prompts.
- **Academic Value**: This prevents "cheating" via global planning or future-looking heuristics, forcing the system to rely purely on its active memory retrieval capabilities.

---

## 2. Generator-Oracle Passthrough (生成器神谕原则)
- **Definition**: To evaluate the quality of the *composed context* independently of the *video generator's rendering capabilities*, the actual video generation step must be replaced by a perfect passthrough of the ground-truth (GT) video chunk.
- **Constraint**: The SUT's composed context is evaluated directly against the GT video chunk $t$. After evaluation, the GT video chunk $t$ is fed back to the SUT as its "perfect generation result" for memory ingestion.
- **Academic Value**: This completely decouples context composition quality from the video generator's quality, ensuring that all scoring pressure lands purely on the memory management mechanism.

---

## 3. Cheat-Proof Implicit Prompting (防作弊隐式提示原则)
- **Definition**: Prompts fed to the SUT must be purely descriptive (semantic/visual) and must not contain explicit retrieval hints (such as *"retrieve asset from chunk 0"*).
- **Constraint**: The benchmark must only describe *what* needs to be generated in the current chunk (e.g., *"Alice walks in the forest"*). The SUT must independently understand the prompt, map the semantic entities to its internal memory bank, and retrieve the correct assets.
- **Academic Value**: This prevents baselines from bypassing active memory management via simple "flat-history lookup table" strategies, forcing them to perform active semantic matching and deduplication.

---

## 4. Entity Reference & Naming Alignment (实体引用与命名一致性原则)
- **Definition**: To prevent reference drift (e.g., the benchmark calling a character "Alice" while the SUT calls her "woman_in_red"), the benchmark must act as the **Naming Authority (命名神谕)**.
- **Constraint**: 
  1. The benchmark discovers and assigns unique names to entities in the GT video chunks.
  2. The benchmark propagates these exact names to the SUT during ingestion feedback (via `EntityObservation`).
  3. The SUT must use these benchmark-provided names to index its memory, ensuring a shared, consistent semantic namespace.
- **Academic Value**: This eliminates semantic misalignment and ensures that prompt-based retrieval is robust and deterministic across all evaluated baselines.

---

## 5. State Persistence & Oracle Tracking (上帝视角实体追踪原则)
- **Definition**: The benchmark must maintain an internal, read-only, and perfect "Reference Asset Space" (上帝视角资产库) that is updated as the video progresses.
- **Constraint**: This reference space is *only* used by the benchmark for generating consistent prompts/instructions and evaluating metrics (e.g., checking if the baseline correctly retrieved a historical asset). It must never be shared with the SUT.
- **Academic Value**: Without this state persistence, the benchmark would suffer from naming drift and would be unable to verify visual consistency and retrieval fidelity across chunks.

---

## 6. Objective & Rule-Based Metrics (客观与规则化指标原则)
- **Definition**: Evaluation metrics must be mathematically defined and calculated using objective visual/semantic features, rather than relying on subjective, open-ended VLM scoring.
- **Constraint**: Use VLM purely as a **semantic extractor** (e.g., binary same-entity classification, or instruction execution verification). The final metrics (Sufficiency, Parsimony, Compactness, Avoidance) must be computed using rigid mathematical formulas over these extractions.
- **Academic Value**: This ensures that the evaluation is highly reproducible, mathematically sound, and free from the non-deterministic scoring biases of LLMs/VLMs.

---

## 7. Resource & Environment Decoupling (资源与环境解耦原则)
- **Definition**: The benchmark package must be **code-self-contained**: it stands alone as a folder that can be open-sourced and run in isolation. Self-containment is about *source-code coupling that ties the benchmark's evolution to code we would have to change*, NOT about forbidding every external import and NOT about forbidding external pretrained models.
- **What "coupling" actually means (the line we don't cross)**: the forbidden thing is importing code **that we would need to modify** as part of benchmark work (mutable/in-flux code), because then editing the benchmark means editing the SUT/repo internals and vice-versa. Importing **stable, unchanged** code that we treat as a frozen dependency is fine — we depend on its public behavior, we never touch it.
- **Constraint (code)**:
  1. `src/memstrata` (the SUT) and `src/vmem_bench` (the benchmark) **must not import each other**. This boundary is absolute (it is the SUT↔benchmark contract), regardless of stability — they interoperate only through the JSON contracts in `schemas_and_contracts.md`.
  2. **Code we would modify** (anything in-flux we might edit while building/iterating the benchmark) must be **vendored/implemented inside `benchmarks/VMem-Bench/`**, not imported — so the benchmark never carries a hidden edit-time dependency on repo internals. This is why SBD, tracking, re-ID, consolidation, and metrics live inside the benchmark.
  3. **Stable, unchanged code is importable.** Standard third-party libraries (`torch`, `transformers`, `numpy`, `scipy`, `PIL`, `safetensors`, optional `supervision`/`faiss`, `insightface`, …) and any equally-frozen upstream module we treat as a fixed dependency are allowed — they are not "the codebase we own and change". For open-source release, such a dependency should be pip-installable or vendored so the folder still runs in isolation.
- **Explicitly allowed — external pretrained MODEL WEIGHTS**: loading pretrained weights is **not** a code coupling. The pipeline may load any model (GroundingDINO, DINOv3, SigLIP, SAM3/3.1, a face encoder, a served VLM) from the shared model roots (`models/model_weights` / `${PUBLIC_MODELS_ROOT}`) via `common/model_weights.py`. Weights are data, resolved by path/config; swapping them does not touch the benchmark's code boundary.
- **Academic Value**: This makes the benchmark easy to open-source, run in isolated environments, and integrate with third-party baselines, while still leveraging the best available perception/VLM models.

---

## 8. Extreme Parallelization (极致并行化原则)
- **Definition**: Any independent operations within the evaluation loop must be designed with task-level parallelism to maximize compute efficiency.
- **Constraint**: Batch VLM calls, multi-image feature extractions, and multi-card baseline evaluations must run concurrently (e.g., via thread pools or multi-process distribution) rather than in serial loops.
- **Academic Value**: This drastically reduces evaluation latency and wall-clock time, making large-scale benchmarking over massive datasets (like LSMDC) practical and cost-effective.

---

## 9. Prompt-Complete Generation Source (提示词即完整生成源原则)
- **Definition**: Under the Generator-Oracle assumption (Principle 2), the prompt $p_t$ plus the composed context is the **sole semantic source** of chunk $t$. Everything visible in the GT chunk — entities, their first-appearance looks, and **state changes** (e.g., "the apple is eaten") — must be narrated in the prompt of the chunk where it occurs.
- **Constraint**:
  1. The annotation pipeline must enforce **prompt completeness deterministically**: every canonical gold-present entity name and every accepted finite-ontology state event of chunk $t$ is materialized in $p_t$; embedding/VLM checks may add diagnostics but cannot replace this gate.
  2. The SUT learns about world changes through the prompt narration and the post-chunk observation feedback — never through the benchmark's materialized forbidden/deprecation tables, which remain scoring-only.
- **Academic Value**: Without this, the oracle passthrough is self-contradictory (content appears that no input could have produced), and lifecycle metrics (Avoidance) would test access to hidden answers rather than the SUT's own lifecycle reasoning.

---

## 10. Embedding Sidecar Storage (Embedding 独立落盘原则)
- **Definition**: Any embedding vector produced during benchmark construction or evaluation must be stored in `.safetensors` sidecar files, keyed by `representation_id`.
- **Constraint**: JSON artifacts store only string references (`embedding_key`), never inline float arrays. This applies to gold data, intermediate pipeline artifacts, and SUT-facing contracts alike.
- **Academic Value**: Keeps human-reviewable JSON small and diff-able, keeps numeric payloads binary-exact (no float-to-text round-trip loss), and cleanly separates the reviewable annotation from the machine-only feature store.

---

## 11. Seeded, Self-Verifying Annotation Loop (人工定本体 + 标注自检闭环原则)
- **Definition**: VLM-produced annotations are drafts, not truth. Production identity comes from a human-confirmed canonical roster/exemplar seed; models localize and fill evidence but cannot mint or rename gold entities. Deterministic gates run before optional independent VLM verification.
- **Constraint**:
  1. Tracklets are assigned only to same-kind seed candidates using multi-view exemplars; low-score/low-margin observations become `unknown/reject`, never forced labels.
  2. State events belong to a finite irreversible ontology and must be allowed by that seed entity; camera/visibility/focus/position changes are rejected.
  3. Prompt entity/event coverage, stable IDs, alias splits, missing evidence, and unknown tracks are deterministic blocking checks. A production draft with any blocking finding cannot freeze.
  4. Human review is entity-centric (one canonical entity crop grid; one state timeline per entity), plus unknowns and audit samples. Human review remains the final freeze gate (`human_reviewed: true`).
- **Academic Value**: Moves the small amount of human judgment to the high-leverage ontology boundary, preventing a large model-generated pairwise cleanup queue while retaining reproducible machine evidence extraction.

---

## 12. The Annotation-System Triad: Fast, Accurate, Low-Touch (标注系统三元原则)
- **Definition**: An excellent automated annotation system is judged on exactly three axes, simultaneously: **速度快 (wall-clock speed)**, **效果好 (annotation quality)**, and **人工工作量低 (human review workload)**. Every pipeline design decision must state which axis it improves and must not silently regress the other two.
- **Constraint**:
  1. **Speed**: every model call carries an explicit output budget (`max_tokens`) and a bounded, parameter-varying retry policy; a single degenerate call must never consume minutes. Stage wall-times are logged per run (events.jsonl timestamps) so speed regressions are measurable, not anecdotal.
  2. **Quality**: errors must be pushed to the *cheapest* layer that can catch them — deterministic filters (species guards, reversible-event patterns, kind lexicons, credits exclusion) before embedding thresholds, embedding thresholds before VLM adjudication, VLM adjudication before humans. A "fast" run that loses entities (e.g. crashed shots silently skipped) is a quality failure, not a speed win.
  3. **Low-touch**: the human review queue is a *decision* surface, not a *log*: no card without an actionable question, one card per decision (grouped events/components, never per-datum cards), and every card ships machine evidence plus a recommended action. Queue size per movie is a tracked metric alongside time and quality.
- **Academic Value**: The triad is the benchmark's production-cost model. Reporting all three per release (time / quality metrics / review-card count) makes annotation cost reproducible and comparable, and prevents "quality" improvements that quietly explode human cost or runtime.

---

## 附录：通用基准方法论 (Academic Benchmark Design, cross-domain)

上面 12 条是 MemStrata-Bench 在长视频记忆/上下文管理任务上的具体化。它们背后是三条**超越具体任务领域**（长视频生成、NLP、多模态理解等）的系统级学术基准法则——任何要构建"具备高学术信度、防作弊能力和范式级对比价值"的评测基准都应遵守。这里只作纲领性表述，避免与正文重复；每条给出它在正文中的落点。

- **I. Separation of Concerns & Decoupling (关注点分离与解耦律)** — 下游系统的噪声、渲染偏差或硬件非确定性会掩盖被测系统（SUT）的核心能力。评测必须把被测的核心能力（检索、规划、记忆管理）与下游的渲染/生成/执行模块**完全解耦**：下游被抽象为确定性黑盒或完美神谕（Oracle），所有评测压力落在 SUT 输出的结构化契约上，而非下游引入的随机噪声或美学偏好。→ 在本文由 **#2 Generator-Oracle Passthrough** 具体化（另见 #1 因果流式约束）。
- **II. Deterministic Metric Formulation (确定性指标计算律)** — 直接让 LLM/VLM 输出主观数值评分（"给一致性打 1–10 分"）会带来幻觉、非确定性和提示词敏感性。大模型只能扮演**离散语义提取器**（回答二分类/高度结构化问题），最终指标由刚性、基于规则的数学公式在这些离散提取结果上计算，保证数学可复现性。→ 在本文由 **#6 Objective & Rule-Based Metrics** 具体化。
- **III. Self-Containment & Environment Decoupling (自包含与环境解耦律)** — 合格的学术基准必须能被第三方独立复现，不绑定特定系统实现或复杂本地环境。基准包（`vmem_bench`）不得导入 SUT（`memstrata`）的内部业务逻辑，两者仅通过中立标准的 JSON/YAML 数据契约交互；基准应提供自包含运行沙盒与轻量自检测试以保证跨平台可移植。→ 在本文由 **#7 Resource & Environment Decoupling** 具体化（其中进一步区分了"可编辑代码耦合"与"冻结依赖/模型权重"的边界）。
