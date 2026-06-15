"""characterize.py — 刻画 snapshots/ 里全部 pop.bin 的 workload 结构.

吃 harvest.py 产出的 snapshots/manifest.json + pop_gen*.bin, 对每份快照算:
  - K 分布: mean / p50 / p90 / max, K=0 占比, K>=2 占比
  - n_nodes 分布 + 树深 (从 prefix 序自己算, 不依赖 EvoGP)
  - 算子配比: opcode 频次, 分组 四则/三角/exp-log/pow/比较类/其他
  - 跨代漂移: 同 run 内 nodes / K / depth 随代数变化 (= bloat 曲线, 即每 run 表的行)
再汇总全局 opcode 频次表 (012 后端覆盖面用), 并按规则选 3 个 preset.

输出: characterization.md (结构化表格).

用法:
    uv run python experiments/012_op_bench/workload/characterize.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKLOAD_DIR = HERE.parents[1] / "data" / "workload"
from cusr.benchmark.popio import load_pop_bin, NTYPE_VAR, NTYPE_CONST, NTYPE_UFUNC, NTYPE_BFUNC

# opcode 枚举, 跟 008 tree_interpreter.py / EvoGP tree/utils.py 对齐
OP_NAMES = {
    1: "ADD", 2: "SUB", 3: "MUL", 4: "DIV", 5: "LOOSE_DIV", 6: "POW", 7: "LOOSE_POW",
    8: "MAX", 9: "MIN", 10: "LT", 11: "GT", 12: "LE", 13: "GE",
    14: "SIN", 15: "COS", 16: "TAN", 17: "SINH", 18: "COSH", 19: "TANH",
    20: "LOG", 21: "LOOSE_LOG", 22: "EXP", 23: "INV", 24: "LOOSE_INV",
    25: "NEG", 26: "ABS", 27: "SQRT", 28: "LOOSE_SQRT",
}
OP_GROUPS = {
    "四则": {1, 2, 3, 4, 5},
    "三角": {14, 15, 16, 17, 18, 19},
    "exp-log": {20, 21, 22, 27, 28},
    "pow": {6, 7},
    "比较类": {10, 11, 12, 13},
    # 其他: MAX MIN INV NEG ABS + 未列出的
}
ARITY = {NTYPE_VAR: 0, NTYPE_CONST: 0, NTYPE_UFUNC: 1, NTYPE_BFUNC: 2}


def tree_depth(nt: np.ndarray) -> int:
    """prefix 序算树深 (根=1). 栈存每层剩余孩子数."""
    stack: list[int] = []
    maxd = 0
    for t in nt:
        d = len(stack) + 1
        if d > maxd:
            maxd = d
        a = ARITY[int(t)]
        if a:
            stack.append(a)
        else:
            while stack:
                stack[-1] -= 1
                if stack[-1] == 0:
                    stack.pop()
                else:
                    break
    return maxd


def snapshot_stats(path: Path) -> dict:
    p = load_pop_bin(path)
    metas = p["metas"]
    nt, nv = p["nt"], p["nv"]
    n_nodes = metas[:, 1].astype(np.float64)
    K = metas[:, 3].astype(np.float64)
    depths = np.array([tree_depth(nt[o:o + n]) for o, n, _, _ in metas], dtype=np.float64)

    op_mask = (nt == NTYPE_UFUNC) | (nt == NTYPE_BFUNC)
    opcodes = Counter(nv[op_mask].astype(np.int64).tolist())
    n_ops = sum(opcodes.values())

    def grp(name):
        ids = OP_GROUPS[name]
        return sum(c for o, c in opcodes.items() if o in ids) / max(n_ops, 1)

    other = 1.0 - sum(grp(g) for g in OP_GROUPS)
    q = lambda a, p_: float(np.percentile(a, p_)) if len(a) else 0.0
    return dict(
        M=p["M"],
        K_mean=float(K.mean()), K_p50=q(K, 50), K_p90=q(K, 90), K_max=int(K.max()),
        K0_frac=float((K == 0).mean()), K2_frac=float((K >= 2).mean()),
        nodes_mean=float(n_nodes.mean()), nodes_p50=q(n_nodes, 50),
        nodes_p90=q(n_nodes, 90), nodes_max=int(n_nodes.max()),
        depth_mean=float(depths.mean()), depth_p90=q(depths, 90), depth_max=int(depths.max()),
        const_frac=float((nt == NTYPE_CONST).mean()), var_frac=float((nt == NTYPE_VAR).mean()),
        grp={g: grp(g) for g in OP_GROUPS} | {"其他": other},
        opcodes=opcodes, n_ops=n_ops,
        total_nodes=p["total_nodes"], max_stack=p["max_stack"],
    )


def fmt_row(e: dict, s: dict) -> str:
    g = s["grp"]
    return (f"| {e['gen']} | {s['M']} "
            f"| {s['nodes_mean']:.1f} / {s['nodes_p50']:.0f} / {s['nodes_p90']:.0f} / {s['nodes_max']} "
            f"| {s['depth_mean']:.1f} / {s['depth_p90']:.0f} / {s['depth_max']} "
            f"| {s['K_mean']:.2f} / {s['K_p50']:.0f} / {s['K_p90']:.0f} / {s['K_max']} "
            f"| {s['K0_frac']*100:.0f}% | {s['K2_frac']*100:.0f}% "
            f"| {g['四则']*100:.0f}% | {g['三角']*100:.0f}% | {(g['exp-log']+g['pow']+g['比较类']+g['其他'])*100:.0f}% |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, default=WORKLOAD_DIR / "snapshots")
    ap.add_argument("-o", "--out", type=Path, default=HERE / "characterization.md")
    args = ap.parse_args()

    manifest = json.loads((args.snapshots / "manifest.json").read_text())
    stats = {e["file"]: snapshot_stats(args.snapshots / e["file"]) for e in manifest}

    runs: dict[tuple, list[dict]] = {}
    for e in manifest:
        runs.setdefault((e["dataset"], e["pop"], e["seed"]), []).append(e)
    for v in runs.values():
        v.sort(key=lambda e: e["gen"])

    L = []
    L.append("# Workload characterization — 真实 EvoGP 逐代种群快照")
    L.append("")
    L.append(f"生成: characterize.py, {date.today().isoformat()}. "
             f"快照: harvest.py (3 题 × pop {{1000,4000}} × 50 代, checkpoint 每 5 代, seed 0, N=1000).")
    L.append(f"EvoGP 配置 (008 dump_evogp.py `_build_evogp`): max_tree_len=32, max_layer_cnt=4, "
             f"USING_FUNCS={{+,-,*,/,sin,cos,tan}}, const_samples 6 个, survival_rate=0.3, mutation_rate=0.2.")
    L.append("")
    L.append("每 run 一张表; 行 = 代数 → 同表纵向读即跨代漂移 (bloat 曲线). "
             "列缩写: nodes/depth/K 为 mean/p50/p90/max (depth 为 mean/p90/max). "
             "M = 留存树数 (TFUNC/K>32 跳过后). 算子组「其余」= exp-log + pow + 比较类 + 其他.")
    L.append("")

    header = ("| gen | M | nodes m/p50/p90/max | depth m/p90/max | K m/p50/p90/max "
              "| K=0 | K≥2 | 四则 | 三角 | 其余 |")
    sep = "|---" * 10 + "|"
    for (ds, pop, seed), entries in sorted(runs.items()):
        L.append(f"## {ds}  pop={pop}  seed={seed}")
        L.append("")
        L.append(header); L.append(sep)
        for e in entries:
            L.append(fmt_row(e, stats[e["file"]]))
        L.append("")

    # ---- 全局 opcode 频次表 ----
    total_ops = Counter()
    per_run_ops: dict[tuple, Counter] = {}
    for (key, entries) in runs.items():
        c = Counter()
        for e in entries:
            c.update(stats[e["file"]]["opcodes"])
        per_run_ops[key] = c
        total_ops.update(c)
    n_total = sum(total_ops.values())

    L.append("## 全局 opcode 频次 (全部快照合计)")
    L.append("")
    L.append("012 后端覆盖面用: Operon / PySR 映射不了的 op (MAX/MIN/LT/GT/LE/GE) 在真实 workload 里占比见下.")
    L.append("")
    L.append("| opcode | name | count | % of op nodes |")
    L.append("|---|---|---|---|")
    for op, cnt in sorted(total_ops.items()):
        L.append(f"| {op} | {OP_NAMES.get(op, '?')} | {cnt} | {cnt/n_total*100:.2f}% |")
    L.append("")
    absent = sorted(set(OP_NAMES) - set(total_ops))
    L.append(f"未出现 opcode: {', '.join(OP_NAMES[o] for o in absent)} — "
             f"dump_evogp.py 的 USING_FUNCS 只含 +,-,*,/,sin,cos,tan, 进化不会产生其它 op "
             f"(LOOSE_DIV 在 dump 端已退化成 DIV). MAX/MIN/比较类占比 = 0%.")
    L.append("")

    # ---- preset 推荐 ----
    flat = [(e, stats[e["file"]]) for e in manifest]

    def pick(cands, key):
        return max(cands, key=key)

    early = pick([(e, s) for e, s in flat if e["gen"] == 5 and e["pop"] == 4000 and "I.18.12" in e["dataset"]],
                 key=lambda t: t[1]["M"])
    constheavy = pick([(e, s) for e, s in flat if e["gen"] >= 30 and e["pop"] == 4000],
                      key=lambda t: (t[1]["K_mean"], t[1]["K2_frac"]))
    ch_run = (constheavy[0]["dataset"], constheavy[0]["pop"])
    # bloated 限定与 const-heavy 不同 run: 3 preset 覆盖 3 个 regime (小树 / 大树低K / 大树高K)
    bloat = pick([(e, s) for e, s in flat
                  if e["gen"] == 50 and e["pop"] == 4000 and (e["dataset"], e["pop"]) != ch_run],
                 key=lambda t: t[1]["nodes_mean"])

    L.append("## Preset 推荐 (012 算子层 benchmark 输入)")
    L.append("")
    L.append("| preset | 快照文件 (snapshots/ 下) | 题 / gen / pop | 关键统计 | 选择依据 |")
    L.append("|---|---|---|---|---|")

    def prow(name, e, s, why):
        ks = f"nodes m={s['nodes_mean']:.1f} max={s['nodes_max']}, depth max={s['depth_max']}, K m={s['K_mean']:.2f}, K=0 {s['K0_frac']*100:.0f}%, K≥2 {s['K2_frac']*100:.0f}%"
        L.append(f"| {name} | `{e['file']}` | {e['dataset']} / gen {e['gen']} / pop {e['pop']} | {ks} | {why} |")

    prow("early-gen", *early, "默认题 gen 5: 选择压前期, 树小, 接近初始分布")
    prow("late-gen-bloated", *bloat, "gen 50 nodes_mean 最大且与 const-heavy 不同 run — 大树低 K regime, 压力在树 eval 不在 LM")
    prow("inner-const-heavy", *constheavy, "gen≥30 中 K_mean 最大 (K≥2 占比 tie-break) — 大树高 K, 压力在 batched LM")
    L.append("")

    args.out.write_text("\n".join(L) + "\n")
    print(f"[characterize] {len(manifest)} snapshots → {args.out}")
    # 终端摘要
    for name, (e, s) in [("early-gen", early), ("late-gen-bloated", bloat), ("inner-const-heavy", constheavy)]:
        print(f"  preset {name}: {e['file']}  nodes_mean={s['nodes_mean']:.1f} K_mean={s['K_mean']:.2f}")


if __name__ == "__main__":
    main()
