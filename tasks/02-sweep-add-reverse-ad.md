# 让规模扫描脚本用上反向求导

- **编号**：02
- **状态**：完成
- **来自**：`docs/PAPER.md` 第一节第 2 条
- **需要**：离线

## 目标

让规模扫描脚本除了 fusedfd、前向求导，也能跑**反向求导（reverse-AD）**；保留前向作对比（清单第 6 条要这张对比图）。

## 背景

`sweep_e6.py` 现在只认两个变体：`fusedfd` + `ad`（前向求导），**漏了我们做好的反向求导**。直接锁频重跑 = 测错方法、白跑（和之前 e5 测错变体同一个坑）。反向求导的二进制 `batch_lm_revad_prof` 是 drop-in（同样的命令行、同样的 PROFILE_JSON 输出），只要把它注册进去并过掉反作弊白名单即可。`revad` 这个名字不含 `_fd`/`co_fd`/`libcusr_co`，不会被防"假数据"的子串陷阱误杀；它也照样发 `fd_jacobian` 这个 profile 类别，所以结构性 guard 不用改。

## 验收标准（做完要能逐条勾上）

- [x] `test_sweep_e6.py` 新增的测试：改前失败、改后通过
- [x] `"revad"` 出现在 `VARIANT_BINARY`、`GPU_VARIANTS`、`ALLOWED_BINARIES`
- [x] `assert_gpu_binary` 不误杀 `batch_lm_revad_prof`
- [x] 原有 20 个测试仍然全绿（现共 21 个）

## TDD 步骤

1. **红**：在 `experiments/e6_kernel_sweep/test_sweep_e6.py` 加：
   - 把 `batch_lm_revad_prof` 加进 `test_assert_gpu_binary_accepts_allowlisted_prof_binaries`（行 100）的元组；
   - 新测试 `test_revad_variant_registered`：`"revad"` 在 `VARIANT_BINARY`（basename 是 `batch_lm_revad_prof`）和 `GPU_VARIANTS`；
   - 新测试 `test_revad_not_trapped_by_fd_substring`：临时建名为 `batch_lm_revad_prof` 的文件，`assert_gpu_binary` 不报错。
   跑 → 失败。
2. **绿**：改 `experiments/e6_kernel_sweep/sweep_e6.py`：行 44 `ALLOWED_BINARIES` 加 `"batch_lm_revad_prof"`；行 46–49 `VARIANT_BINARY` 加 `"revad"`；行 562 `GPU_VARIANTS = ["fusedfd", "ad", "revad"]`。跑 → 通过。
3. **重构**：（暂无）`EXPECTED_TPUT` / `_POINTS_PER_S` 的 revad 项等校准出真值再补，不影响正确性。

## 怎么跑

```bash
uv run python -m pytest experiments/e6_kernel_sweep/test_sweep_e6.py -q
```

## 关联文件

- `experiments/e6_kernel_sweep/sweep_e6.py` — 改：行 44、46–49、562
- `experiments/e6_kernel_sweep/test_sweep_e6.py` — 加测试
