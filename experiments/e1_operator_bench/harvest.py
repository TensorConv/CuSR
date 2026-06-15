"""harvest.py — 跑 6 个真实 EvoGP run, 收割逐代种群快照到 snapshots/, 写 manifest.json.

3 题 × pop {1000, 4000} × 50 代, checkpoint 每 5 代 (含 gen 0 初始种群).
调 008 的 dump_evogp.py (需 --checkpoint-every 支持). 每个 run 一个子目录:
    snapshots/<tag>_pop<P>_seed<S>/pop_gen00XX.bin

manifest.json: 每份快照一条 {file, dataset, gen, pop, seed, N, M_kept, ...}.

用法:
    source scripts/env.sh && uv run python experiments/012_op_bench/workload/harvest.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import cusr.kernel
SNAP = HERE.parents[1] / "data" / "workload" / "snapshots"
DUMP = Path(cusr.kernel.__file__).parent / "dump_evogp.py"

from cusr.benchmark.popio import load_pop_bin

GENS = 50
CKPT = 5
SEED = 0
N = 1000
RUNS = [
    # (dataset, pop)  — 3 题: 默认题(SIN, 内部非线性常数) / 浅树题 / 高K难题(易 bloat)
    ("feynman/I.18.12", 1000), ("feynman/I.18.12", 4000),
    ("feynman/I.12.1", 1000), ("feynman/I.12.1", 4000),
    ("feynman/I.6.2", 1000), ("feynman/I.6.2", 4000),
]


def run_one(dataset: str, pop: int) -> Path:
    tag = dataset.split("/")[-1]
    rundir = SNAP / f"{tag}_pop{pop}_seed{SEED}"
    rundir.mkdir(parents=True, exist_ok=True)
    # -o 指到末代 checkpoint 同名文件 → 最终 dump 与 gen-50 checkpoint 重合, 不产生重复文件
    out = rundir / f"pop_gen{GENS:04d}.bin"
    cmd = [sys.executable, str(DUMP),
           f"--dataset={dataset}", f"--gen={GENS}", f"--pop={pop}",
           f"--N={N}", f"--seed={SEED}", f"--checkpoint-every={CKPT}",
           "-o", str(out)]
    print(f"[harvest] {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, check=True)
    return rundir


def build_manifest() -> list[dict]:
    entries = []
    pat = re.compile(r"pop_gen(\d{4})\.bin$")
    for rundir in sorted(SNAP.iterdir()):
        if not rundir.is_dir():
            continue
        m = re.match(r"(.+)_pop(\d+)_seed(\d+)$", rundir.name)
        if not m:
            continue
        tag, pop, seed = m.group(1), int(m.group(2)), int(m.group(3))
        for f in sorted(rundir.glob("pop_gen*.bin")):
            g = int(pat.search(f.name).group(1))
            p = load_pop_bin(f)
            entries.append(dict(
                file=str(f.relative_to(SNAP)), dataset=f"feynman/{tag}",
                gen=g, pop=pop, seed=seed, N=p["N"], n_vars=p["n_vars"],
                M_kept=p["M"], total_nodes=p["total_nodes"], total_c=p["total_c"],
                K_max=p["K_max"], max_stack=p["max_stack"],
            ))
    return entries


def main():
    for dataset, pop in RUNS:
        run_one(dataset, pop)
    entries = build_manifest()
    (SNAP / "manifest.json").write_text(json.dumps(entries, indent=1))
    print(f"[harvest] manifest.json: {len(entries)} snapshots", flush=True)


if __name__ == "__main__":
    main()
