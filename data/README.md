# `data/` —— 跨实验共享数据

**目前空**（只有 `.gitkeep`）。

## 什么放这里 vs 放 `experiments/XXX/data/`

- **这里**：多个实验复用的数据。比如 SRBench 数据集 dump、Feynman FSRD 原始文件。
- **实验目录**：某个实验专属的数据 / 中间产出。比如 `experiments/003_nls_bench/data/nguyen/1.parquet`。

两者都用 dataset registry 模式管（`registry.yaml` + parquet + sha256），只是物理位置不同。

## gitignore 策略

看 `.gitignore`：
```
data/*
!data/.gitkeep
```

顶层 `data/` 里的内容**默认 gitignored**——大数据集不进 git，通过下载脚本还原。如果是小的 ground-truth 表或 schema 文件（<1MB），加一个显式 `!data/some_small_file.yaml` 破例。

## 现在暂缓的原因

在 T-20 之前我们用实验内部的 `experiments/003_nls_bench/data/` 就够了。等真的有多个实验要共享同一份数据（比如 003 和未来 004 都用 Nguyen）再往这里搬 + 改 registry 路径。
