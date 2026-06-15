# Setup

How to bring CuSR up on a fresh machine (e.g. a GPU server). Steps are layered:
do the minimal block to get the kernel running; add the rest only if you need it.

## What you need

- [uv](https://docs.astral.sh/uv/) and Python 3.12.
- `nvcc` (a CUDA toolkit) to build the kernel. If the machine already has one
  (`nvcc --version` works), use it. Otherwise `scripts/env.sh` points at a conda
  toolkit, or load a system module (`module load cuda`).
- An NVIDIA GPU. The kernel builds for `sm_80` by default (matches A100); change
  the arch in `cusr/kernel/Makefile` for a different card.

## Minimal — kernel + parity gate

No external data and no `evogp` needed; the parity gate runs on the `pop.bin`
fixture that ships with the repo.

```bash
git clone https://github.com/TensorConv/CuSR.git && cd CuSR

uv sync                 # install dependencies from uv.lock
uv pip install -e .     # register the cusr package

make -C cusr/kernel     # build cusr/kernel/batch_lm (needs nvcc)

uv run python cusr/kernel/tests/test_parity_gate.py   # expect 3/3 PASS
uv run pytest tests/bench                              # expect ~99 passed
```

The parity gate fits the fixture population with the kernel and checks it against
a scipy fp64 reference. Expect recovery rates around loss-down 94%, within-1.05x
92%, within-10x 99.8% (the thresholds are 93 / 90 / 99; a point or two of drift
across machines is normal).

> `uv sync` also pulls `pyoperon` and `pysr` (two baseline optimizers used by the
> benchmark). The kernel does not need them — if either fails to install on your
> machine, skip it and continue.

## Operator benchmark (E1)

The repo ships only the *recipe* for the benchmark workloads, not the bulk data:
`data/feynman/`, one example `data/fixtures/pop.bin`, and
`data/workload/presets.json`. Two workload tiers are added at run time.

```bash
# self-test the harness
uv run python experiments/e1_operator_bench/test_harness.py    # expect 16 pass / 0 fail

# regenerate the synthetic workloads (numpy only — no GPU, no evogp, deterministic).
# the exact per-preset command is in data/workload/presets.json (gen_cmd field), e.g.:
uv run python cusr/benchmark/workload/gen_synth.py --preset inner-const-heavy --M 4000 --N 1000 --seed 0
```

The benchmark backends `scipy`, `torch`, and `kernel` need nothing beyond the
minimal install. See `docs/benchmark_README.md` for the runner; the runner lives
in `cusr/benchmark/`.

## Data: what ships vs. what you generate

| Data | In the repo? | How to get it on a server |
|---|---|---|
| `data/feynman/`, `data/fixtures/pop.bin`, `presets.json` | yes (~11M) | comes with the clone |
| `data/workload/synth/` (synthetic, known constants) | no (~50M) | regenerate with `gen_synth.py` (numpy, deterministic) |
| `data/workload/snapshots/` (real GP populations) | no (~675M) | `scp` from another machine, **or** regenerate (needs evogp — see below) |

Regeneration is seeded and reproducible. GP populations are not guaranteed to be
byte-identical across different GPUs, so if you regenerate the real snapshots
rather than copying them, the checksums pinned in `presets.json` may need
re-pinning (the runner only warns on a mismatch).

## Optional — evogp (demonstrator, or regenerating real snapshots)

`evogp` is not on PyPI and not in `uv.lock`. Three modules import it at module
level (`cusr/kernel/dump_evogp.py`, `cusr/bench/sources/evogp.py`,
`cusr/demonstrator/pipeline.py`), so anything that touches the GP front end needs
it installed from source:

```bash
source scripts/env.sh              # CUDA env for building the extension
bash scripts/setup_upstream.sh     # clones upstream repos (incl. evogp) at pinned SHAs
uv pip install -e upstream/evogp   # builds EvoGP's CUDA extension (needs nvcc + torch)

uv run python -c "import evogp; print(evogp.__file__)"   # verify
```

Then regenerate real snapshots with `experiments/e1_operator_bench/harvest.py`,
or run the demonstrator under `experiments/e2_demonstrator/`.

## Optional — performance variants

Three kernel sources ship: `batch_lm.cu` (baseline), `batch_lm_devjac.cu`, and
`batch_lm_fusedfd.cu`. The Makefile only builds the baseline; build the others by
hand and select one with the `BATCH_LM` env var:

```bash
cd cusr/kernel
nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo --use_fast_math -o batch_lm_devjac  batch_lm_devjac.cu  loader.c
nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo --use_fast_math -o batch_lm_fusedfd batch_lm_fusedfd.cu loader.c
cd ../..

BATCH_LM=cusr/kernel/batch_lm_devjac uv run python cusr/kernel/tests/test_fixture_scale.py
```

For the throughput numbers that go in the paper and the full rerun checklist, see
`docs/kernel/RERUN_A100.md`.
