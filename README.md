# CuSR

GPU constant optimization for symbolic regression.

The core is a batched **Levenberg–Marquardt** solver written as a CUDA kernel.
It fits the numeric constants of many candidate expressions at once, where each
expression has a *different* number of constants (heterogeneous K) — and it does
so without padding every expression up to a common K. This is the part of a
symbolic-regression search that a discrete operator/constant pool cannot reach
on its own: structure search proposes the shape `sin(w*x + b)`, and the solver
recovers the `w`, `b` that actually fit the data.

This repository is the research code for our HPEC 2026 paper (see [Paper](#paper) below).
It also bundles an operator-level benchmark that compares the kernel against other constant-
optimization backends (scipy, a Torch LM, Operon's optimizer, PySR's), and a
demonstrator that plugs the kernel into a GPU GP search as the inner solver.

## Layout

```
cusr/                 installable package (`import cusr.*`)
  kernel/             the CUDA kernel: .cu/.h sources, Makefile, loader,
                      tree interpreter, dump/verify tools, parity tests
  bench/              SR/NLS bench library (skeleton × source × backend × runner)
  benchmark/          operator-level CO benchmark: backends, runner, pop I/O,
                      numpy oracle interpreter, synthetic workload generator
  demonstrator/       kernel-as-inner-solver: CO backend, bridge, judge, pipeline
experiments/
  e1_operator_bench/  scripts that produce the benchmark numbers
  e2_demonstrator/    scripts that run the GP-search demonstrator
data/
  feynman/            Feynman formula set + sampled parquet
  fixtures/           pop.bin fixtures used by the kernel tests
  workload/           benchmark presets (large snapshots are not committed)
docs/                 protocol, results, kernel internals, design notes
scripts/              environment setup, upstream bootstrap, job templates
tests/                pytest suite for cusr/bench
```

The CUDA kernel input is a generic postfix-bytecode format (`pop.bin`): a flat
encoding of a population of expression trees. Any front end that can emit that
format can use the solver — it is not tied to one GP system.

## Install

Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # resolve and install dependencies from uv.lock
uv pip install -e .     # register the cusr package (editable)
```

After the editable install, `import cusr.benchmark.runner` (etc.) resolves from
any working directory.

For a full fresh-machine walkthrough — building the kernel, the benchmark data,
optional `evogp`, and the performance variants — see [docs/SETUP.md](docs/SETUP.md).

### Building the kernel

The kernel needs a CUDA toolchain (`nvcc`). Set the build/runtime environment,
then build:

```bash
source scripts/env.sh
make -C cusr/kernel        # produces cusr/kernel/batch_lm
```

`scripts/env.sh` exports `CUDA_HOME`, library paths, and the target GPU
architecture. Built binaries are not committed; build on the target machine.
Set the architecture flag for your card before building.

## Running

```bash
# kernel parity gate — fits a fixture population, checks against a fp64 reference
uv run python cusr/kernel/tests/test_parity_gate.py

# bench library tests
uv run pytest tests/bench

# operator-level benchmark self-tests
uv run python experiments/e1_operator_bench/test_harness.py

# demonstrator tests
uv run pytest experiments/e2_demonstrator
```

## Data

Small fixtures and the Feynman set live under `data/`. The benchmark's large
workload snapshots are generated, not committed; `data/workload/presets.json`
pins them by checksum so a run can confirm it is using the intended workload.

## Upstream dependencies

Some paths (the GP front end used by the demonstrator, baseline optimizers)
depend on third-party repositories. They are cloned at pinned revisions:

```bash
bash scripts/setup_upstream.sh
```

## Paper

Hao Mao, Xu Tony Liu, Shuai Lu, Peng Zhao, Wenzheng Jiang, and Yuntian Chen.
**Efficient Constant Optimization for Symbolic Regression with GPU-Accelerated Tree-Based Genetic Programming.**
IEEE High Performance Extreme Computing Conference (HPEC 2026).
[arXiv:2609.03352](https://arxiv.org/abs/2609.03352)

If you use this code, please cite:

```bibtex
@inproceedings{mao2026efficient,
  title         = {Efficient Constant Optimization for Symbolic Regression with GPU-Accelerated Tree-Based Genetic Programming},
  author        = {Mao, Hao and Liu, Xu Tony and Lu, Shuai and Zhao, Peng and Jiang, Wenzheng and Chen, Yuntian},
  booktitle     = {2026 IEEE High Performance Extreme Computing Conference (HPEC)},
  year          = {2026},
  eprint        = {2609.03352},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE}
}
```
