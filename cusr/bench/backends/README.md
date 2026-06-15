# `bench/backends/` —— NLS 优化器适配

**Backend = 优化算法的包装**。吃一个 `FitRequest`，用某个数值优化算法拟合常数，产出一个 `FitRecord`。

## Protocol

```python
class NLSBackend(Protocol):
    name: str
    def fit(self, request: FitRequest, seed: int, **kwargs) -> FitRecord: ...
```

`fit` 必须：
- 读 `request.skeleton.residual` / `.jacobian` / `.X` / `.y`
- 跑优化
- 返回完整填写的 `FitRecord`（含 `final_loss`、`converged`、`stop_reason` 等所有字段）

## 文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `__init__.py` | 空包标识 | — |
| `base.py` | `NLSBackend` Protocol | ✓ |
| `scipy_common.py` | 两个类：`ScipyLSBackend`（method=lm/trf/dogbox）+ `ScipyMinimizeBackend`（method=BFGS/L-BFGS-B/Newton-CG/NelderMead/Powell） | ✓ |
| `julia_optim.py` | `raise NotImplementedError("Coming in T-22")`。T-22 填 `juliacall` → `Optim.jl` + `LeastSquaresOptim.jl` | stub |
| `jaxopt_lm.py` | `raise NotImplementedError("Coming in T-33")`。T-33 填 `jaxopt.LevenbergMarquardt` + `vmap` | stub |

## `scipy_common.py` 关键点

### `ScipyLSBackend`

包装 `scipy.optimize.least_squares`。核心调用：

```python
res = least_squares(
    skeleton.residual,  # 接 (c, X, y) 返回残差向量 (n_samples,)
    init,
    jac=skeleton.jacobian,  # 接 (c, X) 返回 Jacobian (n_samples, n_constants)
    method=self.method,  # 'lm' / 'trf' / 'dogbox'
)
```

支持传 `bounds`、`max_nfev`、`xtol` 等 kwargs 透传给 scipy。

### `ScipyMinimizeBackend`

包装 `scipy.optimize.minimize`。自己构造 loss 和梯度：

```python
def _loss(c):
    r = skeleton.residual(c, X, y)
    return float(np.mean(r * r))   # mean(r²)

def _grad(c):
    r = skeleton.residual(c, X, y)
    J = skeleton.jacobian(c, X)
    return (2.0 / n_samples) * (J.T @ r)   # ∂/∂c 的 mean(r²)
```

不同 method 是否用 gradient：`BFGS / L-BFGS-B / Newton-CG / CG / TNC / SLSQP / trust-ncg / trust-krylov / trust-exact` 用；`Nelder-Mead / Powell` 无导数。

## 关键不变式（review 过）

1. **`final_loss` 统一 `mean(r²)`**——跟数据集大小无关，不同题可以横着比。
2. **`converged = True` 必须两件事都满足**：
   - scipy 报告 `success=True` 或 status > 0
   - `final_loss < initial_loss`（实际进步了）
   
   否则 `stop_reason='no_progress'`。这是防止"Jacobian 里有 NaN → scipy LM 在 step 0 宣布 tol 收敛 → 假装拟合了"的陷阱。
3. **异常兜底**：优化器抛异常 → 记 `stop_reason='error'` + `error_msg`，不 crash 整条 pipeline。
4. **种子**：因为 scipy LM/BFGS 都是确定性方法（给定初值即给定轨迹），我们**不**调 `np.random.seed(seed)`——避免污染全局 RNG 又拿不到任何复现性收益。需要随机性的 backend（比如 NelderMead 如果有 restart）自己管理。

## 怎么写一个新 Backend

```python
from dataclasses import dataclass, field
from bench.backends.base import NLSBackend
from bench.skeleton import FitRecord, FitRequest

@dataclass
class MyBackend:
    # 任何 ctor 配置
    some_hyperparam: float = 1.0
    
    @property
    def name(self) -> str:
        return f"my_backend_{self.some_hyperparam}"
    
    def fit(self, request: FitRequest, seed: int, **kwargs) -> FitRecord:
        # 1. 拿到 skeleton.residual / jacobian / request.X / request.y / init
        # 2. 跑你的优化算法
        # 3. 算 initial_loss / final_loss / n_fn_evals / ...
        # 4. 判定 converged：要求 loss 真下降
        # 5. 返回 FitRecord（所有字段都填）
        ...
```

## 相关文档

- 设计 §3：`discussion/nls_bench_design.md`
- 算法选型 / 对比（T-10..T-14 会补）：`notes/nls_algorithms.md`（待写）
