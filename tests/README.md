# `tests/` —— 项目单测

顶层 pytest 测试目录。Phase 0 只有 `tests/bench/`（核心库的单测）。未来其他实验自己的单测也放这里（`tests/experiments/XXX/`）或者放各自实验目录内，两种都行，看规模。

## 跑法

```bash
# 全部
uv run pytest tests/

# 只跑 bench 单测
uv run pytest tests/bench/

# 带详细输出 + 遇错即停
uv run pytest tests/bench/ -v -x

# 单个测试
uv run pytest tests/bench/test_skeleton.py::test_jacobian_nan_guard -v
```

## 配置

`pyproject.toml` 里：
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

`pythonpath=["."]` 让 `from bench.xxx import ...` 直接 work，不用 `pip install -e`。

## 约定

- 每个测试文件配套一个 source 文件（`test_skeleton.py` ↔ `bench/skeleton.py`）。
- 不测 stub 文件（`julia_optim.py` / `jaxopt_lm.py` / `eval.py`）。
- 需要 GPU 的测试用 `pytest.importorskip("evogp")` 开头——这样 CPU-only 机器 gracefully skip。
- CUDA 外的逻辑（比如手搓 Tree 做 `forest_member_to_skeleton`）用 CPU tensor，避免绑死 CUDA。
