# `bench/` —— SR bench 平台核心库

这是一个**跨实验复用**的库，不绑定任何具体实验。里面放的是 NLS 常数拟合 bench 的通用组件：数据结构、优化器适配、数据集代理、流水线。

## 设计思路（和 `discussion/nls_bench_design.md` 对齐）

三轴正交：
- **Source**（骨架来源）× **Backend**（优化器）× **Runner**（粘合剂）。
- 各轴独立演化，加新东西只要实现对应 Protocol，不改其他两轴。

核心数据类在 `skeleton.py`：
- `Skeleton`：骨架 = sympy 表达式 + 变量符号 + 常数符号。不可变。
- `FitRequest`：一次拟合的输入（骨架 + 初值 + 数据代理 + 溯源信息）。
- `FitRecord`：一次拟合的输出，一行 JSONL。

## 文件清单

| 文件 | 作用 |
|------|------|
| `__init__.py` | 空包标识 |
| `skeleton.py` | `Skeleton` / `FitRequest` / `FitRecord` 三件套。含 `residual` 的 NaN 守卫、`jacobian` 的解析微分（走 `sp.diff` + `lambdify`）和同样的 NaN 守卫 |
| `canonicalize.py` | 骨架结构哈希。`hash_l1`（裸 srepr）和 `hash_l2`（expand + 常数重编，默认），用于 dedup 和 GPU bucketing |
| `dataset.py` | `Dataset` 懒加载代理 + `from_registry()`。支持 transform-layered 引用、sha256 校验、路径逃逸防护 |
| `transform.py` | 数据变换（目前只有 `gaussian_noise`） |
| `runner.py` | `Runner` 主循环：source × backend × seed × restart，流式写 JSONL |
| `restart.py` | Restart 策略。目前只实现 `NoRestart`；`RandomPerturb` / `WarmStartOnly` / `BestOfN` 在 T-22 填 |
| `eval.py` | 指标 + 画图（T-23 stub，目前是空的） |
| `sources/` | 骨架提供方。详见 `sources/README.md` |
| `backends/` | NLS 优化器适配。详见 `backends/README.md` |

## 怎么用

```python
from bench.skeleton import Skeleton, FitRequest
from bench.dataset import Dataset
from bench.sources.synthetic import SyntheticSource
from bench.backends.scipy_common import ScipyLSBackend
from bench.runner import Runner

runner = Runner(
    source=SyntheticSource(problems=["nguyen/1"], n_perturbed=3),
    backends=[ScipyLSBackend(method="lm")],
    seeds=[0, 1, 2],
    output_dir=Path("runs/my_experiment"),
)
run_dir = runner.run()   # 产出 config.yaml / requests.jsonl / records.jsonl
```

## 扩展

- **加新 source**：见 `sources/README.md`。
- **加新 backend**：见 `backends/README.md`。
- **加新 canonicalize 方式**：在 `canonicalize.py` 里写一个 `hash_foo(expr) -> str`，构造 Skeleton 时传 `canonicalize_fn=hash_foo`。

## 不变式（review 过）

- `Skeleton.jacobian` 对 NaN 守卫到 0（scipy LM 看到零梯度就不会在该方向瞎走）。
- `Skeleton.residual` 对 NaN 守卫到 1e10 哨兵（scipy LM 看到大残差自动回退步长）。
- Backend 的 `converged=True` 必须满足 **既 scipy 说成功**，**又实际 loss 下降**——防止 NaN 守卫下算法"原地宣布收敛"。
- `loss` 统一定义为 `mean(r²)`，跟数据集大小无关。

## 相关文档

- 完整设计：`discussion/nls_bench_design.md`
- EvoGP 接入细节：`notes/evogp_constants.md`
- ROADMAP：`ROADMAP.md`（T-25/T-21/T-22/T-23/T-33 相关）
