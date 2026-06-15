"""probe_floor_basin.py — 噪声地板的"够不到"是基底吸引域, 不是求解器精度/预算.

噪声地板 = loss(c_true) (合成 preset 注入 1% 噪声的能量, 见 gen_synth.py).
runner 报 "oracle 够到地板的比例" 偏低 (高 K 尤甚). 本探针证伪两个朴素解释:

* 不是 max_nfev=200 预算: 够不到的树绝大多数 status=converged (非 iter-limit);
* 不是 c_true 不是真极小: scipy 从 c_true 重启 → 100% 守住地板 (median L/floor≈0.998,
  即有限样本 MLE 比地板低 ~K/N, 健康).

真因: 从演化式 ±30% 初值 (c_init = c_true·U(0.7,1.3)+N(0,0.02)) 出发, LM 收敛到
**另一个差得多的局部点** —— 基底吸引域问题 (不区分真·多峰 vs FD-Jacobian 坏步; kernel
同样 FD, 两种解释下结论都成立). 这是 problem hardness, 对 CPU fp64 reference 同样成立 ——
kernel 够到地板的比例 ≈ oracle (见 runner 输出), 不是 kernel 短板. 注: reach% 随初值
散布而变 (±30% 是生成器假设), 故主推**不变量** oracle≈kernel, 不是绝对档位.

跑: uv run python probe_floor_basin.py   (本目录, 读 _cache 的 oracle + workload/synth)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cusr.benchmark import backends, interp, popio

HERE = Path(__file__).resolve().parent
WORKLOAD_DIR = HERE.parents[1] / "data" / "workload"
PRESETS = {  # (bin, oracle-cache sha) — sha 见 data/workload/presets.json
    "synth-early-gen": "1706766eaddeffdc",
    "synth-late-gen-bloated": "068dfe4e40a34b58",
    "synth-inner-const-heavy": "a2dfe3b2ede185d9",
}
SAMPLE = 150  # 每 preset 抽多少 converged-miss 树做 c_true 重启验证


def _cslice(c_flat, metas, idxs):
    return np.concatenate([c_flat[metas[m][2]:metas[m][2] + metas[m][3]] for m in idxs])


def _subpop(pop, idxs, c_source):
    trees = []
    for m in idxs:
        no, n, _, _ = pop["metas"][m].tolist()
        co, k = pop["metas"][m][2], pop["metas"][m][3]
        trees.append((pop["nt"][no:no + n], pop["nv"][no:no + n],
                      pop["ci"][no:no + n], c_source[co:co + k]))
    return popio.build_pop(trees, pop["xs"], pop["ym"][idxs])


def run_preset(name: str, sha: str) -> dict:
    pop = popio.load_pop_bin(WORKLOAD_DIR / "synth"
                             / f"synth_{name[len('synth-'):]}_M4000_N1000_seed0.bin")
    ct = np.load(WORKLOAD_DIR / "synth"
                 / f"synth_{name[len('synth-'):]}_M4000_N1000_seed0.ctrue.npy")[:pop["total_c"]]
    z = np.load(HERE / "_cache" / f"oracle_{sha}_Lall_serial.npz")
    Lstar, status = z["loss_star"], z["status"]
    Ln = interp.loss_pop(pop, ct)
    K = pop["metas"][:, 3]
    elig = (K > 0) & np.isfinite(Lstar) & np.isfinite(Ln)
    reach = elig & (Lstar <= 1.005 * Ln)
    conv_miss = np.where(elig & (status == 0) & ~reach)[0]
    cap_miss = np.where(elig & (status == 1) & ~reach)[0]

    smp = conv_miss[:SAMPLE]
    sp_true = _subpop(pop, smp, ct)
    sp_init = _subpop(pop, smp, pop["c_init"])
    Ln_smp = interp.loss_pop(sp_true, _cslice(ct, pop["metas"], smp))
    r_true = backends.ScipyPop(nproc=1).fit_pop(sp_true)
    r_init = backends.ScipyPop(nproc=1).fit_pop(sp_init)
    L_true = interp.loss_pop(sp_true, r_true.c_final)
    L_init = interp.loss_pop(sp_init, r_init.c_final)
    return dict(
        name=name, elig=int(elig.sum()), reach=int(reach.sum()),
        reach_pct=100 * reach.sum() / elig.sum(),
        conv_miss=int(conv_miss.size), cap_miss=int(cap_miss.size),
        from_true_reach=100 * float((L_true <= 1.005 * Ln_smp).mean()),
        from_true_med=float(np.median(L_true / Ln_smp)),
        from_init_reach=100 * float((L_init <= 1.005 * Ln_smp).mean()),
        from_init_med=float(np.median(L_init / Ln_smp)))


def main():
    rows = [run_preset(n, s) for n, s in PRESETS.items()]
    print(f"{'preset':26s} {'oracle够地板':>10s} {'conv-miss':>9s} {'cap-miss':>8s} "
          f"{'重启c_true':>10s} {'重启c_init':>10s}")
    for r in rows:
        print(f"{r['name']:26s} {r['reach_pct']:9.1f}% {r['conv_miss']:9d} {r['cap_miss']:8d} "
              f"{r['from_true_reach']:8.1f}% {r['from_init_reach']:8.1f}%  "
              f"(med L/floor true {r['from_true_med']:.3f} / init {r['from_init_med']:.0f})")
    print("\n够不到地板的树绝大多数 status=converged (conv-miss >> cap-miss) → 非预算上限;")
    print("从 c_true 重启 ~100% 守住地板 (med≈0.998 = MLE 比地板低 ~K/N) → c_true 是真极小;")
    print("从 c_init 重启复现 miss → 演化初值的基底吸引域问题, 对 fp64 oracle 同样成立.")


if __name__ == "__main__":
    main()
