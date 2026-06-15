# `bench/sources/` —— 骨架来源

**Source = 骨架提供方**。从某个地方拿到"骨架 + 初值 + 数据 + 溯源信息"，yield 出 `FitRequest` 流给 Runner。

## Protocol

```python
class SkeletonSource(Protocol):
    name: str
    def iter_requests(self) -> Iterable[FitRequest]: ...
```

`iter_requests` 是 generator——**流式**产出，不一次性物化所有 request（EvoGP 跑 1000 代也不会撑爆内存）。

## 文件清单

| 文件 | 作用 |
|------|------|
| `__init__.py` | **故意不 eager 导入 `EvoGPSource`**——它会拉 `evogp` 包 + CUDA 扩展，CPU-only 用户会炸。需要时显式 `from bench.sources.evogp import EvoGPSource` |
| `base.py` | `SkeletonSource` Protocol |
| `synthetic.py` | 手写 Nguyen 1-3 + 自动物化 parquet 到 `experiments/003_nls_bench/data/nguyen/`。每题可生成 N 个扰动初值的 request |
| `evogp.py` | `EvoGPSource` + 三个辅助件。详见下 |
| `jsonl.py` | 从之前 dump 的 `requests.jsonl` 回放。用 `sp.parse_expr` + 白名单（防 malicious 输入做 RCE） |

## `sources/evogp.py` 三件套

1. **`DEGRADE_MAP`**：EvoGP 有 `LOOSE_DIV` / `LOOSE_LOG` / `LOOSE_INV` 三个"GP 容错版"算子。sympy 侧这三个是 `sp.Function` 子类，没实现 `.fdiff()`，我们拿不到 Jacobian，lambdify 也不认。降级成标准 `/` / `log` / `1/x`。`LOOSE_SQRT` 和 `LOOSE_POW` 在 EvoGP 里已经是原生 sympy lambda（`sqrt(Abs(x))` / `Pow(Abs(x), y)`），不用降级。
2. **`forest_member_to_skeleton(tree, n_vars)`**：把一棵 EvoGP `Tree` 转成 `(Skeleton, init_constants, degraded_ops)`。算法是两遍 walk——先正向 prefix 扫一遍给每个 CONST 位置分配 `c_i`，再反向 prefix 走栈机构造 sympy Expr。
3. **`_TopKPipeline(StandardPipeline)`**：关键的 **时序 subclass**。原版 `StandardPipeline.step()` 会在 return 前调 `algorithm.step()`，**后者内部把 `self.forest` 换成下一代**——所以 naive 的"step 返回后再做 `torch.topk(fitness)[0] → forest[idx]`"会错位拿到下一代的成员。我们的 subclass 在 `super().step()` **之前** snapshot top-K（还 clone 三个 tensor 切断 view 关联），保证抓到当代。

## 怎么写一个新 Source

```python
from bench.sources.base import SkeletonSource
from bench.skeleton import FitRequest, Skeleton
from bench.dataset import Dataset
import sympy as sp

class MySource:
    name = "my_source"
    def __init__(self, ...): ...
    def iter_requests(self):
        for ...:
            x0 = sp.Symbol("x0", real=True)
            c0 = sp.Symbol("c0", real=True)
            skel = Skeleton(expr=c0*sp.sin(x0), variables=(x0,), constants=(c0,))
            yield FitRequest(
                skeleton=skel,
                init_constants=np.array([1.0]),
                dataset=my_dataset,
                source={"origin": "my_source", "extra_info": "..."},
            )
```

`source` dict 是"溯源口袋"——随便塞，runner 不看，原样透传到 `FitRecord.source`。

## 相关设计

- 设计 §3（Frontend/Backend 接口）：`discussion/nls_bench_design.md`
- EvoGP 接入的关键 caveat（时序、loose ops）：`notes/evogp_constants.md`
