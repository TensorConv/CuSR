# walkthrough — 一级一级搭出 batch_lm.cu

目标:不啃 13 课教程,直接把 `../batch_lm.cu` 读懂。办法是**自己从最小的
一块重搭一遍**:每级一个自包含 `.cu`,只加一个新概念,本机 GPU 上跑通、和
CPU 参考对拍打印 PASS,你说"懂了"再写下一级。enum / fixture 全程和生产代码
`../batch_lm.cu` 严格对齐,所以搭到最后,生产代码就是"这些零件 + 工程外壳"。

## 怎么跑

```bash
cd docs/kernel/walkthrough
./build.sh s1        # 编译并运行 s1_*.cu,看 PASS
```

`build.sh` 自动 `source scripts/env.sh` 拿 nvcc。产物不进 git(见 .gitignore)。

## 阶梯(地图,不是合同 —— 短的级可能合并)

| 级 | 文件 | 新概念 | 对应生产代码 |
|----|------|--------|--------------|
| s1 | `s1_tree_host.cu` | 树用 3 数组怎么存 + 逆波兰栈机求值(**纯 host,不碰 GPU**) | `eval_tree_d` |
| s2 | `s2_tree_gpu_1thread.cu` | 同一个解释器搬上 GPU,1 个线程 | `eval_tree_d` + 启动 |
| s3 | `s3_tree_gpu_Npoints.cu` | 一棵树 N 个数据点,1 线程 1 点 | eval over N |
| s4 | `s4_warp_per_tree.cu` | **warp-per-problem**:M 棵树,一个 warp 一棵树 | `eval_kernel_batched` |
| s5 | `s5_residual_loss.cu` | residual + loss,`__shfl_xor_sync` 归约 | `residual_kernel` / `loss_kernel` |
| s6 | `s6_jacobian_fd.cu` | finite-difference Jacobian | FD 段 |
| s7 | `s7_jtj_jtr.cu` | 装配 JᵀJ / JᵀR | `build_jtj_jtr_kernel` |
| s8 | `s8_cholesky_solve.cu` | K×K Cholesky 解 | `solve_kernel` |
| s9 | `s9_lm_loop.cu` | host LM 主循环 A–G 串起来 | `main` 的循环 |
| s10| `s10_production_readthrough.md` | 读生产 `batch_lm.cu`(文件 loader + status code + 失败处理) | 整文件 |

## 进度

| 级 | 状态 |
|----|------|
| s1 | ✅ 你确认掌握 |
| s2 | 🚧 待你跑 + 反馈 |
| s3–s9 | 🚧 已写 + 编译跑 PASS + 审查/连贯修过,待你跑 + 反馈 |
| s10 | 🚧 已写(生产通读),待你读 + 反馈 |

状态:⬜ 未写 / 🚧 待你跑+反馈 / ✅ 你确认掌握

## 约定(沿用 learn/cuda 那套)

- 每个 `.cu` 结尾跟 CPU 参考**对拍**,打印 `PASS/FAIL` —— "看起来跑了" 不等于 "算对了"。
- 每个 `.cu` 末尾附 `## 自检 + 参考答案`,先自己想再看。
- 正确性锚:每级逻辑和 `../../../learn/cuda/project_hetero_lm/m3_*` 的对应
  kernel 一致(那批是已验证 PASS 的参考实现)。
