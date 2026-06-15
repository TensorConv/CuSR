# `tests/bench/` —— bench 核心库单测

目前 25 个 test 分布在 5 个文件里。覆盖 `bench/` 的核心正确性不变式。

## 文件清单

| 文件 | 测啥 |
|------|------|
| `__init__.py` | 空包标识 |
| `test_skeleton.py` | Skeleton 构造/校验、evaluate 形状、residual NaN 守卫、Jacobian 对比解析导数、Jacobian NaN 守卫（review CRITICAL 回归）、n_outputs 属性、hash 稳定性 |
| `test_canonicalize.py` | hash_l1 结构敏感、hash_l2 对 commutative reorder 不变、对常数重编不变、确定性、共享 vs 独立常数区分得开 |
| `test_dataset.py` | parquet 读写往返、sha256 校验、transform 可重现性、base+transform 组合链、未知 transform 抛错 |
| `test_backends.py` | ScipyLSBackend LM 收敛到线性问题的真值、bad fit 标记未收敛、ScipyMinimizeBackend BFGS 收敛 |
| `test_sources_evogp.py` | 手搓 EvoGP Tree（CPU tensor）→ forest_member_to_skeleton 往返正确（const order + no loose ops） |

## 关键回归测试

几个测试是"曾经踩过坑、写测试防复发"的：

1. **`test_jacobian_nan_guard`**（`test_skeleton.py`）：
   起因：review 发现 `Skeleton.jacobian` 没 NaN 守卫，当骨架含 `sqrt(c0 - x0²)` 且初值让参数落到 NaN 域时，scipy LM 收到 NaN Jacobian 会静默宣布 "tol 收敛"，bogus final_loss=1e20。测试固定这个 case，断言 jacobian 返回全 finite 且为零（零梯度让 LM 不在该方向瞎走）。

2. **`test_hash_l2_distinguishes_shared_vs_independent_consts`**（`test_canonicalize.py`）：
   起因：review 要求验证 `c0 + c0*x0`（一个常数两处用）和 `c0 + c1*x0`（两个独立常数）的哈希要不一样——这俩结构上就是不同骨架，不该合并。

3. **`test_linear_tree_to_skeleton`**（`test_sources_evogp.py`）：
   起因：T-02 agent 发现 T-01 note 里常数编号顺序有歧义（forward-prefix vs reverse-prefix）。测试固定 `2.0*x0 + 1.0` 这棵树，断言 `init_constants == [2.0, 1.0]`（按自然阅读顺序编号）。

## 跑法

```bash
uv run pytest tests/bench/ -q
# → 25 passed, 2 warnings in ~2s
# warnings 是 test_residual_nan_guard + test_jacobian_nan_guard 主动触发 NaN 的预期行为
```
