# VMem-Bench

**视觉记忆一定让当前故事更正确吗？** VMem-Bench 分开评估“检索到了什么”与“生成器最终画出了什么”。

论文：*More Memory, Better Videos? A Controlled Study of Visual Memory in Long-Video Generation*（2026）。arXiv 链接将在可用后补充。

作者：Yuzhuo Chen、Huafeng Shi、Xinyu Wang、Yulong Xu、Yucheng Wang、Haoqin Hong、Guoxin Zhang、Zehua Ma、Weiming Zhang。

## 两条评测轨道

- **Track A**：生成前选择参考；选择完成后才揭示当前真实视频片段。论文覆盖 92 部电影、16,548 个片段。
- **Track B**：基于按时间顺序提供的提示和自身生成历史，评估召回、精度、禁止实体回避、状态正确性及条件身份指标。主要比较覆盖 30 个故事、每条件 2,824 个片段。

检索可以提高召回及历史外观相似性，同时降低精度和禁止实体回避率。长间隔成功也可能来自当前提示中的外观描述。这两条轨道的实例不匹配，不能把 Track A 的检索质量直接解释为 Track B 的生成效果。

主 Track-B 比较只有 seed 2026 的单次 rollout；置信区间不涵盖换种子的变异。Last-seen 使用 gold ID，是特权诊断控制，不是可部署方法或预算匹配上限。50 故事创作池与实际评测的 30 故事不同。

## 快速开始

```bash
git clone https://github.com/Suchenl/VMem-Bench.git
cd VMem-Bench
python3 -m pip install -e .
python3 scripts/evaluate_baselines/your_method/run_tracka_example.py \
  --movie-dir assets/trackA/BlenderOpenMovies/charge --limit 5
```

这是无 GPU、无原视频的接口演示，使用占位参考，不产生可报告的评测分数，也不需要 MemStrata。

- 接入方法：[接口模板](scripts/evaluate_baselines/your_method/README.md)。
- 下载及放置视频：[DATA.md](docs/DATA.md)。Track B 不需要原电影。
- 复现入口：[REPRODUCE.md](REPRODUCE.md)。区分重画结果、重算统计和重新生成视频。
- 公开包边界：[PUBLIC_RELEASE.md](docs/PUBLIC_RELEASE.md)。初版不包含原始 rollout 视频或逐片段评分包。
- 完整使用说明及引用：[英文首页](README.md)。

代码采用 Apache-2.0；自主编写的 gold/prompts 采用 CC BY 4.0。原电影、LSMDC 官方 CSV/AD 和模型权重不随仓库分发。
