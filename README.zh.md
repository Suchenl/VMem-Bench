# VMem-Bench

面向因果长视频生成的视觉记忆评测。Track A 评估生成前选到的参考，Track B 评估生成视频中的实体、状态和排除要求。

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-007ec6?style=flat)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2610.35001-b31b1b?style=flat)](https://arxiv.org/abs/2610.35001) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-FFD21E?style=flat)](https://huggingface.co/datasets/Suchenl/VMem-Bench) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E?style=flat)](https://huggingface.co/papers/2610.35001) [![Project Page](https://img.shields.io/badge/Project-Page-E6550D?style=flat&logo=googlechrome&logoColor=white)](https://suchenl.github.io/VMem-Bench/)

[English](README.md)

## 快速开始

```bash
git clone https://github.com/Suchenl/VMem-Bench.git
cd VMem-Bench
python3 -m pip install -e .
python3 scripts/evaluate_baselines/your_method/run_tracka_example.py \
  --movie-dir assets/trackA/BlenderOpenMovies/charge --limit 5
```

这是无 GPU 的接口示例，使用占位参考，不产生可报告的评测分数。

## 数据与评测

仓库只保留示例与测试所需输入：两部 Track-A 电影的文本标注，以及 30 个英文 Track-B 故事的提示和 gold。完整数据从 Hugging Face 下载：

```bash
python3 -m pip install huggingface_hub
hf download Suchenl/VMem-Bench --repo-type dataset --local-dir assets \
  --include 'trackA/**' 'trackB/**'
```

- 原始视频获取与目录：[DATA.md](docs/DATA.md)。Track B 不需要原电影。
- 方法接入：[接口模板](scripts/evaluate_baselines/your_method/README.md)。
- Track A：[协议](docs/trackA.md)、[运行说明](docs/benchmark/running_eval.md)。
- Track B：[共享生成器](src/vmem_bench/testbed/README.md)、[评分说明](docs/benchmark/trackB_scoring.md)。

原电影、模型权重、完整生成视频和逐片段评分包不随代码仓库分发。论文主 Track-B 对比只测量 seed 2026 的单次 rollout；新评测应记录各自的代码版本、输入、模型和种子。

## 引用

如果在研究中使用 VMem-Bench，请引用对应的 arXiv 预印本。GitHub 仓库侧边栏也可通过 [Cite this repository](CITATION.cff) 获取引用信息。

```bibtex
@misc{chen2026vmembench,
  title={More Memory, Better Videos? A Controlled Study of Visual Memory in Long-Video Generation},
  author={Chen, Yuzhuo and Shi, Huafeng and Wang, Xinyu and Xu, Yulong and Wang, Yucheng and Hong, Haoqin and Zhang, Guoxin and Ma, Zehua and Zhang, Weiming},
  year={2026},
  eprint={2610.35001},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.35001}
}
```

代码采用 Apache-2.0；自主编写的 gold/prompts 采用 CC BY 4.0。详见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
