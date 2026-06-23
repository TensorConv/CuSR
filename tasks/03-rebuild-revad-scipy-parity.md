# 重新编译反向求导 + 和 scipy 对数值

- **编号**：03
- **状态**：完成（数值已验证正确；附一个要记下的发现，见末尾）
- **来自**：`docs/PAPER.md` 第一节第 3 条
- **需要**：GPU

## 目标

重新编译反向求导的二进制，并用 scipy 的双精度结果对一遍数值，确认对了再去烧 GPU 时间。同时产出扫描要用的 `batch_lm_revad_prof`。

## 背景

烧 GPU 前先确认数值对。已有现成的 scipy 对拍脚本 `experiments/revad_v5/compare_scipy.py`（容差 rtol 2% / atol 1e-4），上次结果：both-finite 300/300 全合，分歧的 68 处里反向求导 68/68 合 scipy。把这个一次性脚本做成可重复的回归测试。

## 验收标准（做完要能逐条勾上）

- [x] `compare_scipy.py` 把主比较逻辑抽成可导入的 `classify(sample_path) -> dict`
- [x] `cusr/kernel/tests/test_revad_scipy_parity.py`：没有 nvcc/GPU 时自动 skip（GPU 上跑过，绿）
- [x] `make` 出 `batch_lm_revad` / `batch_lm_revad_prof`（`_dump_jac_sample` 直接 nvcc 编）
- [x] 反向求导数值正确：函数良态处（两种 AD 都有限）144/144 全合 scipy；反向不会比前向多出 NaN

## TDD 步骤

1. **红**：抽出 `classify()` 后，写 `cusr/kernel/tests/test_revad_scipy_parity.py`：没 GPU → `pytest.skip`；否则 `make` → 用一个已提交的小 fixture 群体重新 dump 样本到临时文件 → `classify()` → 断言计数达标。建二进制之前跑 → 失败/skip。
2. **绿**：`cd cusr/kernel && make batch_lm_revad batch_lm_revad_prof _dump_jac_sample`，跑测试 → 通过。
3. **重构**：（暂无）

## 怎么跑

```bash
cd cusr/kernel && make batch_lm_revad batch_lm_revad_prof _dump_jac_sample
uv run python -m pytest cusr/kernel/tests/test_revad_scipy_parity.py -q
# 或直接：
python experiments/revad_v5/compare_scipy.py <新 dump 的样本.jsonl>
```

## 关联文件

- `experiments/revad_v5/compare_scipy.py` — 轻改：抽出 `classify()`
- `cusr/kernel/tests/test_revad_scipy_parity.py` — 新建
- `cusr/kernel/Makefile` — 已有目标 `batch_lm_revad{,_prof}`、`_dump_jac_sample`
- `cusr/kernel/batch_lm_revad.cu` — 反向求导内核源码

## 结果与发现（2026-06-23，A100，data/fixtures/pop.bin 1000 棵，scipy fp64 对拍）

跑了官方质量闸门 `test_parity_gate.py`（任何新 kernel 要替换在用版本必须过的那道）三个变体对比：

| 变体 | loss 下降率 | 落在 scipy 1.05× 内 | 10× 内 | 闸门 |
|---|---|---|---|---|
| fusedfd（有限差分） | 94.0% | **92.1%** ✓ | 99.7% | **3/3 过** |
| ad（前向求导） | 93.3% | **86.6%** ✗ | 100% | 2/3 |
| revad（反向求导） | 93.3% | **86.6%** ✗ | 100% | 2/3 |

**两个要点：**

1. **反向求导和前向求导数值完全一样**（86.6 / 93.3 / 100，`both_conv` 都是 321，逐位相同）。说明这次重编的反向求导是对的、没退化——它忠实复现前向求导。这是本任务要确认的核心，已确认。
2. **1.05× 没过是"AD 路线"的事，不是反向求导特有的 bug**（前向、反向 AD 一模一样，重编没问题）。但这个差距的性质要老实讲清楚，别美化成"度量假象"：
   - 它是**单向的**——AD 比 scipy *差*，不是"各有高低"。
   - 最可能的原因：AD 在接近奇点的树上做的 NaN 防护（0 湮灭乘法）把本该非零的梯度也归零了 → 这些树没被充分优化 → loss 偏高 → 落到 scipy 1.05× 之外。前面 jacobian 对拍里 `m=513` 就是活例子：函数值有限，AD 给梯度 0，scipy 有限差分给 0.277。fusedfd 没有这个归零，所以更贴 scipy。

   **还没坐实**：要实锤这个机制，得把"AD 落在 scipy 1.05× 外、fusedfd 落在内"的那批树，和"AD 归零梯度的奇异树"求交集——重叠就是归零的锅。这要改 `verify.py` 吐 per-tree loss + 改 `_dump_jac_sample` 列出全部奇异树，属正式质量评测（规模扫描那步）的活，留到那时做。

**对论文/后续**：这是 AD 路线一个真实、可量化的质量代价，要如实写进质量评测。规模扫描里 fusedfd / ad / revad 都在，正好量这个差。注：90% 阈值大概率按有限差分基线（92.4%）定的（fusedfd 的 92.1% 与之吻合），但没核实是哪个变体定的——写"吻合"，不写"就是它定的"。
