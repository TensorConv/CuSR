# 锁住显卡频率

- **编号**：01
- **状态**：工具 + 单测完成；真正上机锁频留到正式跑扫描时做
- **来自**：`docs/PAPER.md` 第一节第 1 条
- **需要**：sudo + GPU（辅助逻辑离线可测）

## 目标

有一套能"读 / 锁 / 解锁 / 验证"显卡频率的小工具；跑正式实验前把一张卡的频率锁死、记下锁定值，让结果不再是"频率没锁"的草稿。

## 背景

现在所有实验结果都标着 "clocks unlocked, DRAFT"，速度数字进不了论文。仓库里只有"读频率"的写法（`results/tier0_a100__20260616/sweep.py` 的 `sm_clock()`），没有"锁频率"的工具。真正的"锁"是 `sudo nvidia-smi -lgc` 的副作用、没法纯单元测；但**解析 nvidia-smi 输出、选频、验证是否锁住**这几段逻辑可以离线测。

## 验收标准（做完要能逐条勾上）

- [ ] `cusr/benchmark/gpu_clocks.py` 提供 `query_sm_clock / supported_graphics_clocks / lock_sm_clock / unlock_sm_clock / verify_locked`
- [ ] `tests/bench/test_gpu_clocks.py` 全绿（mock `subprocess.run`，不需要真 GPU）
- [ ] 在 A100 上能锁定一张卡、`verify_locked` 通过、记录下锁定的 MHz（人工 + sudo）

## TDD 步骤

1. **红**：写 `tests/bench/test_gpu_clocks.py`，mock nvidia-smi 的假输出，断言频率解析正确、`verify_locked` 在接近/偏离时分别返回 True/False。跑 → 失败（模块还没有）。
2. **绿**：实现 `cusr/benchmark/gpu_clocks.py`，跑 → 通过。
3. **重构**：把查询命令的拼装收成一个小函数，和 tier0 的写法对齐。

## 怎么跑

```bash
# 离线单元测试
uv run python -m pytest tests/bench/test_gpu_clocks.py -q
# 上机锁频（sudo）
python -m cusr.benchmark.gpu_clocks --lock --gpu 0
python -m cusr.benchmark.gpu_clocks --unlock --gpu 0
```

## 关联文件

- `cusr/benchmark/gpu_clocks.py` — 新建，工具本体
- `tests/bench/test_gpu_clocks.py` — 新建，离线单元测试
- `results/tier0_a100__20260616/sweep.py` — 参考：现成的读频率写法
