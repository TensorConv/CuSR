"""test_harness.py — W1 harness TDD 套件.

跑法: uv run python test_harness.py    (kernel 后端需要先 source scripts/env.sh)

覆盖:
- popio: build/save/load 往返
- interp: 向量化 numpy 解释器 vs 008 tree_interpreter 交叉验证 (含 LOOSE op)
- torch 解释器 vs numpy 解释器
- tier 判定数学 (含 L*=0 / inf / K=0 边界)
- ScipyPop / TorchPop / CudaKernelPop 在已知真值 fixture 上恢复常数
- runner 端到端 (oracle 缓存 + 结果文件)

kernel 二进制缺失或跑不起来 (无 CUDA) 时 SKIP, 不算失败 — 其余测试必须全过.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import cusr.kernel
from cusr.benchmark import popio, interp, backends, runner

# ---- op enums (同 008 pop_fixture_gen.py) ----
N_VAR, N_CONST, N_UFUNC, N_BFUNC = 0, 1, 2, 3
F_ADD, F_SUB, F_MUL, F_DIV, F_POW = 1, 2, 3, 4, 6
F_LOOSE_DIV, F_LOOSE_POW = 5, 7
F_SIN, F_COS, F_LOG, F_LOOSE_LOG, F_EXP, F_NEG, F_SQRT = 14, 15, 20, 21, 22, 25, 27

# (name, nt, nv, ci, c_true, y_func) — 3 个 NLS archetype + 1 棵 K=0 树
TREES = [
    ("powerlaw_K2",
     [N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST],
     [F_MUL, 0.0, F_POW, 0.0, 0.0],
     [-1, 0, -1, -1, 1],
     [2.5, 1.3],
     lambda x, c: c[0] * x ** c[1]),
    ("quadratic_K3",
     [N_BFUNC, N_CONST, N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_VAR],
     [F_ADD, 0.0, F_ADD, F_MUL, 0.0, 0.0, F_MUL, 0.0, F_MUL, 0.0, 0.0],
     [-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, -1],
     [0.5, -1.2, 0.3],
     lambda x, c: c[0] + c[1] * x + c[2] * x * x),
    ("expsat_K3",
     [N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_UFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST],
     [F_ADD, F_MUL, 0.0, F_EXP, F_NEG, F_MUL, 0.0, 0.0, 0.0],
     [-1, -1, 0, -1, -1, -1, 1, -1, 2],
     [3.0, 0.4, 0.7],
     lambda x, c: c[0] * np.exp(-c[1] * x) + c[2]),
    ("sin_K0",
     [N_UFUNC, N_VAR],
     [F_SIN, 0.0],
     [-1, -1],
     [],
     lambda x, c: np.sin(x)),
]

C_REL_TOL_F64 = 1e-5   # scipy / torch (fp64)
C_REL_TOL_F32 = 5e-3   # kernel (fp32, 同 008 scale-test 容差)


def make_test_pop():
    N = 64
    xs = (0.1 + 4.9 * np.arange(N) / (N - 1)).astype(np.float32).reshape(N, 1)
    trees, ym = [], []
    for name, nt, nv, ci, c_true, y_func in TREES:
        c_init = np.asarray([0.9 * v for v in c_true], dtype=np.float32)
        trees.append((np.asarray(nt, np.int32), np.asarray(nv, np.float32),
                      np.asarray(ci, np.int32), c_init))
        ym.append(y_func(xs[:, 0].astype(np.float64),
                         np.asarray(c_true, np.float64)).astype(np.float32))
    pop = popio.build_pop(trees, xs, np.stack(ym))
    return pop


def check_recovery(name, res, pop, tol):
    """后端结果 vs c_true: status==0 且逐常数相对误差 < tol (K=0 树只查 status==3)."""
    ok = True
    for m, (tname, *_rest) in enumerate(TREES):
        c_true = np.asarray(TREES[m][4], np.float64)
        _, _, c_off, K = pop["metas"][m].tolist()
        if K == 0:
            good = res.status[m] == 3
            print(f"  [{'PASS' if good else 'FAIL'}] {name} {tname}: status={res.status[m]} (want 3)")
            ok &= bool(good)
            continue
        c_got = res.c_final[c_off:c_off + K]
        c_rel = float(np.max(np.abs(c_got - c_true) / np.maximum(np.abs(c_true), 1e-30)))
        good = (res.status[m] == 0) and (c_rel < tol)
        print(f"  [{'PASS' if good else 'FAIL'}] {name} {tname}: status={res.status[m]} c_rel={c_rel:.2e}")
        ok &= bool(good)
    return ok


# ============================================================ tests

def test_popio_roundtrip():
    pop = make_test_pop()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tiny.bin"
        popio.save_pop_bin(pop, p)
        pop2 = popio.load_pop_bin(p)
    for k in ("M", "total_nodes", "total_c", "N", "n_vars", "K_max", "max_stack"):
        assert pop[k] == pop2[k], f"{k}: {pop[k]} != {pop2[k]}"
    for k in ("nt", "nv", "ci", "metas", "c_init", "xs", "ym"):
        assert np.array_equal(pop[k], pop2[k]), f"array {k} mismatch"
    assert pop["max_stack"] >= 3 and pop["K_max"] == 3
    print("  [PASS] roundtrip: header + 7 arrays bit-equal")
    return True


def test_slice_pop():
    pop = make_test_pop()
    sub = popio.slice_pop(pop, 2)
    assert sub["M"] == 2 and sub["total_c"] == 5
    y = interp.eval_pop_tree(sub, 1, sub["c_init"][2:5].astype(np.float64))
    y_full = interp.eval_pop_tree(pop, 1, pop["c_init"][2:5].astype(np.float64))
    assert np.array_equal(y, y_full)
    print("  [PASS] slice_pop: offsets rebuilt, eval identical")
    return True


def test_interp_np_vs_scalar():
    """向量化解释器 vs 008 tree_interpreter (标量 oracle), 含 LOOSE op 树."""
    from cusr.kernel.tree_interpreter import eval_batch
    pop = make_test_pop()
    rng = np.random.default_rng(7)
    ok = True
    for m in range(pop["M"]):
        node_off, n_nodes, c_off, K = pop["metas"][m].tolist()
        nt = pop["nt"][node_off:node_off + n_nodes]
        nv = pop["nv"][node_off:node_off + n_nodes]
        c = rng.uniform(0.3, 2.0, K)
        got = interp.eval_pop_tree(pop, m, c)
        ss = np.array([n_nodes], np.int32)
        want = eval_batch(nt, nv, ss, pop["xs"], c)
        good = np.allclose(got, want, rtol=1e-12, atol=1e-12)
        print(f"  [{'PASS' if good else 'FAIL'}] tree {m}: max|Δ|={np.max(np.abs(got - want)):.2e}")
        ok &= bool(good)
    # LOOSE 树: log|c0/x| — 单独构造
    nt = np.asarray([N_UFUNC, N_BFUNC, N_CONST, N_VAR], np.int32)
    nv = np.asarray([F_LOOSE_LOG, F_LOOSE_DIV, 0.0, 0.0], np.float32)
    ci = np.asarray([-1, -1, 0, -1], np.int32)
    xs = np.linspace(-2, 2, 32).reshape(-1, 1).astype(np.float32)
    xs = xs[np.abs(xs[:, 0]) > 0.1]  # 避开 0 除
    lp = popio.build_pop([(nt, nv, ci, np.asarray([1.7], np.float32))], xs,
                         np.zeros((1, len(xs)), np.float32))
    got = interp.eval_pop_tree(lp, 0, np.asarray([1.7]))
    want = eval_batch(nt, nv, np.array([4], np.int32), xs, [1.7])
    good = np.allclose(got, want, rtol=1e-12)
    print(f"  [{'PASS' if good else 'FAIL'}] LOOSE tree: max|Δ|={np.max(np.abs(got - want)):.2e}")
    return ok and bool(good)


def test_interp_torch_vs_np():
    import torch  # noqa: F401
    pop = make_test_pop()
    rng = np.random.default_rng(11)
    ok = True
    for m in range(pop["M"]):
        K = int(pop["metas"][m][3])
        c = rng.uniform(0.3, 2.0, K)
        want = interp.eval_pop_tree(pop, m, c)
        got = backends.eval_pop_tree_torch(pop, m, c).numpy()
        good = np.allclose(got, want, rtol=1e-10, atol=1e-10)
        print(f"  [{'PASS' if good else 'FAIL'}] tree {m}: max|Δ|={np.max(np.abs(got - want)):.2e}")
        ok &= bool(good)
    return ok


def test_operon_mapping():
    """prefix bytecode → Operon postfix 节点表: eval 交叉验证 + rank 回映约定.

    覆盖方向敏感 op (Sub/Div/Pow)、NEG/INV 的冻结常数表示、LOOSE op 原生映射.
    Operon 是 f32, 容差放到 1e-4.
    """
    import pyoperon as op_
    F_INV = 23
    cases = []
    pop = make_test_pop()
    for m in range(pop["M"]):
        node_off, n_nodes, c_off, K = pop["metas"][m].tolist()
        sl = slice(node_off, node_off + n_nodes)
        cases.append((TREES[m][0], pop["nt"][sl], pop["nv"][sl], pop["ci"][sl],
                      K, None))   # K 占位: int → 随机 c; ndarray → 固定 c
    # c0 / (x0 - c1): Div+Sub 双重方向敏感, 常数发射序反转 → rank_order 应为 [1, 0].
    # c 固定 (c1=6 → x0-c1 远离 0): 随机 c1 会让分母擦过零点, f32 在极点附近
    # 放大到超出容差 — 那是数值现象不是映射错误
    cases.append(("div_sub_K2",
                  np.asarray([N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST], np.int32),
                  np.asarray([F_DIV, 0.0, F_SUB, 0.0, 0.0], np.float32),
                  np.asarray([-1, 0, -1, -1, 1], np.int32),
                  np.asarray([1.5, 6.0]), [1, 0]))
    # c0 * inv(x0) + neg(c1): INV/NEG 冻结常数表示
    cases.append(("inv_neg_K2",
                  np.asarray([N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_VAR,
                              N_UFUNC, N_CONST], np.int32),
                  np.asarray([F_ADD, F_MUL, 0.0, F_INV, 0.0, F_NEG, 0.0], np.float32),
                  np.asarray([-1, -1, 0, -1, -1, -1, 1], np.int32), 2, None))
    # logabs(loose_div(c0, x0)): LOOSE 原生映射
    cases.append(("loose_K1",
                  np.asarray([N_UFUNC, N_BFUNC, N_CONST, N_VAR], np.int32),
                  np.asarray([F_LOOSE_LOG, F_LOOSE_DIV, 0.0, 0.0], np.float32),
                  np.asarray([-1, -1, 0, -1], np.int32), 1, None))

    xs = pop["xs"]
    N = xs.shape[0]
    data = np.asfortranarray(np.column_stack([xs.astype(np.float64), np.zeros(N)]))
    ds = op_.Dataset(data)
    dsvars = sorted(ds.Variables, key=lambda v: v.Index)
    var_hashes = [v.Hash for v in dsvars[:-1]]
    rng = np.random.default_rng(3)
    ok = True
    for name, nt, nv, ci, K, want_ranks in cases:
        c = K if isinstance(K, np.ndarray) else rng.uniform(0.4, 2.0, K)
        K = len(c)
        nodes, ranks = backends.build_operon_nodes(op_, nt, nv, ci, c, var_hashes)
        tree = op_.Tree(nodes)
        tree.UpdateNodes()
        got = np.asarray(op_.Evaluate(tree, ds, op_.Range(0, N)), np.float64)
        want = interp.eval_tree_arrays(nt, nv, ci, xs, c)
        good = np.allclose(got, want, rtol=1e-4, atol=1e-5)
        if want_ranks is not None:
            good &= ranks == want_ranks
        n_coef = len(tree.GetCoefficients())
        good &= n_coef == K
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: max|Δ|="
              f"{np.max(np.abs(got - want)):.2e} ranks={ranks} coeffs={n_coef}")
        ok &= bool(good)
    return ok


def test_operon_backend():
    pop = make_test_pop()
    res = backends.OperonLM().fit_pop(pop)
    return check_recovery("operon", res, pop, C_REL_TOL_F32)


def test_tier_classification():
    loss_star = np.asarray([1.0, 0.0, 2.0, 5.0])
    loss_b = np.asarray([1.04, 5e-11, np.inf, 7.0])
    K = np.asarray([2, 1, 3, 0])
    tA, tB, elig = runner.classify_tiers(loss_b, loss_star, K)
    assert elig.tolist() == [True, True, True, False]
    assert tA.tolist() == [False, True, False, False], f"tierA {tA}"
    assert tB.tolist() == [True, True, False, False], f"tierB {tB}"
    assert runner.tier_rate(tA, elig) == 1 / 3 and runner.tier_rate(tB, elig) == 2 / 3
    print("  [PASS] tier math: 1.05 边界 / L*=0 垫底 / inf / K=0 排除")
    return True


def test_scipy_backend():
    pop = make_test_pop()
    res = backends.ScipyPop().fit_pop(pop)
    return check_recovery("scipy", res, pop, C_REL_TOL_F64)


def test_torch_backend():
    pop = make_test_pop()
    res = backends.TorchPop().fit_pop(pop)
    return check_recovery("torch", res, pop, C_REL_TOL_F64)


def test_pysr_expr_builder():
    """bytecode → Julia 表达式字符串: 映射 + NEG(CONST) 字面量折叠的符号回映."""
    F_INV = 23
    # c0 * inv(x0) + neg(c1)
    nt = np.asarray([N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_VAR, N_UFUNC, N_CONST], np.int32)
    nv = np.asarray([F_ADD, F_MUL, 0.0, F_INV, 0.0, F_NEG, 0.0], np.float32)
    ci = np.asarray([-1, -1, 0, -1, -1, -1, 1], np.int32)
    s, signs = backends.build_pysr_expr(nt, nv, ci, np.asarray([2.0, 0.7], np.float32))
    ok = (s == "((2.0 * inv(x1)) + -0.699999988079071)") and signs.tolist() == [1.0, -1.0]
    print(f"  [{'PASS' if ok else 'FAIL'}] inv_neg: {s} signs={signs}")
    # powerlaw: POW → safe_pow
    pop = make_test_pop()
    no, nn, co, K = pop["metas"][0].tolist()
    s2, signs2 = backends.build_pysr_expr(pop["nt"][no:no+nn], pop["nv"][no:no+nn],
                                          pop["ci"][no:no+nn], pop["c_init"][co:co+K])
    ok2 = s2.startswith("(") and "safe_pow(x1," in s2 and signs2.tolist() == [1.0, 1.0]
    print(f"  [{'PASS' if ok2 else 'FAIL'}] powerlaw: {s2}")
    return ok and ok2


def test_pysr_backend():
    """放宽预算 (200 iter, nrestarts=0 确定性) 下的恢复 — 测的是管线不是预算."""
    pop = make_test_pop()
    be = backends.PySRBFGS(iterations=200, nrestarts=0, name="pysr_test")
    res = be.fit_pop(pop)
    ok = check_recovery("pysr", res, pop, 1e-3)   # BFGS+FD 梯度, 容差松于 LM
    # NEG(CONST) 折叠路径端到端: y = c0/x - c1, init 0.9×
    F_INV = 23
    nt = np.asarray([N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_VAR, N_UFUNC, N_CONST], np.int32)
    nv = np.asarray([F_ADD, F_MUL, 0.0, F_INV, 0.0, F_NEG, 0.0], np.float32)
    ci = np.asarray([-1, -1, 0, -1, -1, -1, 1], np.int32)
    c_true = np.asarray([2.0, 0.7])
    xs = (0.1 + 4.9 * np.arange(64) / 63).astype(np.float32).reshape(-1, 1)
    ym = (c_true[0] / xs[:, 0].astype(np.float64) - c_true[1]).astype(np.float32)
    mini = popio.build_pop([(nt, nv, ci, (0.9 * c_true).astype(np.float32))],
                           xs, ym.reshape(1, -1))
    r2 = be.fit_pop(mini)
    c_rel = float(np.max(np.abs(r2.c_final - c_true) / np.abs(c_true)))
    good = (r2.status[0] == 0) and (c_rel < 1e-3)
    print(f"  [{'PASS' if good else 'FAIL'}] pysr inv_neg 端到端: status={r2.status[0]} "
          f"c_rel={c_rel:.2e} c={r2.c_final}")
    return ok and good


def test_kernel_backend():
    binary = Path(cusr.kernel.__file__).parent / "batch_lm"
    if not binary.exists():
        print("  [SKIP] kernel binary not built")
        return None
    pop = make_test_pop()
    try:
        res = backends.CudaKernelPop(binary).fit_pop(pop)
    except backends.KernelRunError as e:
        print(f"  [SKIP] kernel run failed (no CUDA? source scripts/env.sh?): {e}")
        return None
    return check_recovery("kernel", res, pop, C_REL_TOL_F32)


def test_runner_end_to_end():
    pop = make_test_pop()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tiny.bin"
        popio.save_pop_bin(pop, p)
        out = runner.run_bench(p, backend_names=["scipy"], repeats=1,
                               smoke=True, out_dir=Path(td), cache_dir=Path(td))
        r = out["backends"]["scipy"]
        # oracle 与 scipy 后端同设置 → eligible 树全 tier-A
        assert r["tier_a_rate"] == 1.0, f"tier_a_rate={r['tier_a_rate']}"
        assert r["tier_b_rate"] == 1.0
        assert r["wall_e2e"] > 0
        files = list(Path(td).glob("*.json")) + list(Path(td).glob("*.md"))
        assert len(files) >= 2, f"missing outputs: {files}"
        # oracle 缓存命中: 第二次跑 oracle 不重算 (走文件)
        out2 = runner.run_bench(p, backend_names=["scipy"], repeats=1,
                                smoke=True, out_dir=Path(td), cache_dir=Path(td))
        assert out2["oracle"]["cached"] is True
    print("  [PASS] runner: tier-A 100% on self-reference, json+md written, oracle cache hit")
    return True


def test_selection_fidelity():
    Lstar = np.arange(1.0, 11.0)
    elig = np.ones(10, bool)
    sf = runner.selection_fidelity(Lstar.copy(), Lstar, elig)            # 同序
    ok = abs(sf["spearman"] - 1.0) < 1e-9 and sf["top50"] == 1.0 and sf["top10"] == 1.0
    print(f"  [{'PASS' if ok else 'FAIL'}] 同序: spearman={sf['spearman']:.3f} "
          f"top10={sf['top10']} top50={sf['top50']}")
    sf2 = runner.selection_fidelity(Lstar[::-1].copy(), Lstar, elig)     # 反序
    ok2 = abs(sf2["spearman"] + 1.0) < 1e-9 and sf2["top50"] == 0.0
    print(f"  [{'PASS' if ok2 else 'FAIL'}] 反序: spearman={sf2['spearman']:.3f} top50={sf2['top50']}")
    Lb3 = Lstar.copy(); Lb3[0] = np.inf                                  # 非有限/非elig 排除
    elig3 = elig.copy(); elig3[1] = False
    sf3 = runner.selection_fidelity(Lb3, Lstar, elig3)
    ok3 = sf3["n"] == 8
    print(f"  [{'PASS' if ok3 else 'FAIL'}] 排除非有限+非eligible: n={sf3['n']} (want 8)")
    return ok and ok2 and ok3


def test_synth_gen():
    from cusr.benchmark.workload import gen_synth
    ok = True
    # 确定性: 同 (preset, M, N, seed) 两次生成全数组相等
    a = gen_synth.gen_pop("early-gen", 80, 32, seed=7)
    b = gen_synth.gen_pop("early-gen", 80, 32, seed=7)
    keys = ("nt", "nv", "ci", "metas", "c_init", "xs", "ym")
    same = all(np.array_equal(a[k], b[k]) for k in keys)
    print(f"  [{'PASS' if same else 'FAIL'}] 确定性: 同 seed 两次生成逐数组相等")
    ok &= same
    fin = bool(np.all(np.isfinite(a["ym"])))
    print(f"  [{'PASS' if fin else 'FAIL'}] 拒绝采样后全部 ym 有限")
    ok &= fin
    # 边际进 ballpark (松容差防 flaky; 精确校准见 gen_synth.py PRESETS 注释)
    big = gen_synth.gen_pop("inner-const-heavy", 600, 16, seed=1)
    marg = gen_synth.pop_marginals(big)
    tg = gen_synth.PRESETS["inner-const-heavy"]["target"]
    k_ok = abs(marg["K_mean"] - tg["K_mean"]) / tg["K_mean"] < 0.15
    t_ok = abs(marg["trig"] - tg["trig"]) < 0.06
    print(f"  [{'PASS' if k_ok and t_ok else 'FAIL'}] 边际: K_mean={marg['K_mean']:.2f} "
          f"(目标 {tg['K_mean']}), trig={marg['trig']:.2f} (目标 {tg['trig']})")
    ok &= k_ok and t_ok
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.bin"
        popio.save_pop_bin(a, p)
        back = popio.load_pop_bin(p)
        rt = all(np.array_equal(back[k], a[k]) for k in keys)
    print(f"  [{'PASS' if rt else 'FAIL'}] save/load roundtrip")
    ok &= rt
    # preset: 解析 (钉档文件在本机存在; sha 不匹配只警告不炸)
    rp = runner.resolve_pop("preset:synth-early-gen")
    pr_ok = rp.exists()
    print(f"  [{'PASS' if pr_ok else 'FAIL'}] runner.resolve_pop(preset:) → {rp.name}")
    ok &= pr_ok
    return bool(ok)


def test_noise_floor():
    """合成 pop 的 c_true sidecar + 噪声地板不变量 (审计本身由重跑验证, 非此处).

    查: c_true 长度==total_c / K>0 树地板有限 / 真常数 loss << 扰动初值 loss /
    确定性 / npy roundtrip. runner 也认这个 sidecar (noise_floor_path 约定)."""
    from cusr.benchmark.workload import gen_synth
    ok = True
    pop = gen_synth.gen_pop("early-gen", 120, 64, seed=3)
    ct = pop["c_true"]
    len_ok = ct.shape[0] == pop["total_c"]
    print(f"  [{'PASS' if len_ok else 'FAIL'}] c_true 长度 {ct.shape[0]} == total_c {pop['total_c']}")
    ok &= len_ok
    L_noise = interp.loss_pop(pop, ct)
    L_init = interp.loss_pop(pop, pop["c_init"])
    elig = pop["metas"][:, 3] > 0
    fin = bool(np.all(np.isfinite(L_noise[elig])))
    print(f"  [{'PASS' if fin else 'FAIL'}] 地板 loss 在 K>0 树全有限")
    ok &= fin
    med_n, med_i = float(np.median(L_noise[elig])), float(np.median(L_init[elig]))
    better = med_n < 0.5 * med_i  # 真常数应远优于 ±30% 扰动初值
    print(f"  [{'PASS' if better else 'FAIL'}] 地板中位 {med_n:.3e} < 0.5×初值中位 {med_i:.3e}")
    ok &= better
    det = np.array_equal(ct, gen_synth.gen_pop("early-gen", 120, 64, seed=3)["c_true"])
    print(f"  [{'PASS' if det else 'FAIL'}] c_true 确定性 (同 seed 逐元素相等)")
    ok &= det
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ct.npy"
        np.save(p, ct)
        rt = np.array_equal(np.load(p), ct)
    print(f"  [{'PASS' if rt else 'FAIL'}] sidecar npy roundtrip")
    ok &= rt
    return bool(ok)


TESTS = [
    test_popio_roundtrip, test_slice_pop, test_interp_np_vs_scalar,
    test_interp_torch_vs_np, test_operon_mapping, test_tier_classification,
    test_scipy_backend, test_torch_backend, test_operon_backend,
    test_pysr_expr_builder, test_pysr_backend,
    test_kernel_backend, test_runner_end_to_end, test_synth_gen,
    test_selection_fidelity, test_noise_floor,
]


def main():
    n_pass = n_fail = n_skip = 0
    for t in TESTS:
        print(f"\n== {t.__name__} ==")
        try:
            r = t()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = False
        if r is None:
            n_skip += 1
        elif r:
            n_pass += 1
        else:
            n_fail += 1
    print(f"\nresult: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
