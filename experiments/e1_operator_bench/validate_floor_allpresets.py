"""validate_floor_allpresets.py — 保守 pivot-floor (F9) 全 preset 验证, 产品化前的 gate.

判据 = 选择保真度不退化 (§3c, 非 bit-parity): F9 的 Spearman/top10% 在每个 preset 上
都 ≥ A (容噪声). tier-B 顺带报. 通过 → 把 floor 改相对容差设默认; 不过 → 查哪个 preset 退化.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import cusr.kernel
from cusr.benchmark import interp, popio, runner

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling run-scripts
from probe_fp64solve import CACHE, run_binary  # noqa: E402

_KERNEL = Path(cusr.kernel.__file__).parent
BIN_A = (_KERNEL / "batch_lm_fusedfd_A").resolve()
BIN_F9 = (_KERNEL / "batch_lm_fusedfd_F9").resolve()
PRESETS = ["early-gen", "late-gen-bloated", "inner-const-heavy",
           "synth-early-gen", "synth-late-gen-bloated", "synth-inner-const-heavy"]
SPEARMAN_EPS = 0.005   # 噪声容差
TOP10_EPS = 0.010


def measure(binp, pop, Lstar, K):
    _, c, _ = run_binary(binp, pop, reps=1)
    Lb = interp.loss_pop(pop, c)
    _, tB, elig = runner.classify_tiers(Lb, Lstar, K)
    sf = runner.selection_fidelity(Lb, Lstar, elig)
    return int(tB.sum()), sf


def main():
    print(f"{'preset':26s} {'tierB A→F9':>13s} {'Spearman A→F9':>18s} {'top10 A→F9':>18s} {'gate':>5s}")
    all_pass = True
    for p in PRESETS:
        path = runner.resolve_pop(f"preset:{p}")
        pop = popio.load_pop_bin(path)
        K = pop["metas"][:, 3]
        Lstar = runner.get_oracle(path, pop, None, CACHE)["loss_star"]
        tBa, sfa = measure(BIN_A, pop, Lstar, K)
        tBf, sff = measure(BIN_F9, pop, Lstar, K)
        ok = (sff["spearman"] >= sfa["spearman"] - SPEARMAN_EPS
              and sff["top10"] >= sfa["top10"] - TOP10_EPS)
        all_pass &= ok
        print(f"{p:26s} {tBa:5d}→{tBf:<5d}    {sfa['spearman']:.3f}→{sff['spearman']:.3f}    "
              f"  {sfa['top10']:.3f}→{sff['top10']:.3f}   {'PASS' if ok else 'FAIL':>5s}")
    print(f"\n{'='*60}\n全 preset 选择保真度 no-harm: "
          f"{'PASS — 可产品化 (改相对容差设默认)' if all_pass else 'FAIL — 有 preset 退化, 先查'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
