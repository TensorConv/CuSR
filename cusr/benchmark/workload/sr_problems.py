"""sr_problems.py — SR problem library + (X, y) sampler, PURE numpy + sympy.

This module is the single source of truth for the benchmark problem set shared by
BOTH the evogp dump path (``cusr/kernel/dump_evogp.py``) and the Operon harvest
path. It is deliberately free of any evogp / torch import so it loads under the
operon-venv python.

The ``PROBLEMS`` dict and ``sample_xy`` are copied **verbatim** (same math) from
``dump_evogp.py`` (``PROBLEMS``, ``USING_FUNCS``, ``CONST_SAMPLES``, ``_sample_xy``)
so that ``(X, y)`` matches the evogp corpus bit-for-bit for the same
``(problem_id, N, seed, noise)``: same ``np.random.default_rng(seed)``, same column
sampling order + ranges, same sympy skeleton with ``ground_truth_constants``
substituted, same RMS-relative Gaussian noise model.

A regression test (``tests/test_sr_problems.py``) parses ``dump_evogp.py``'s source
with ``ast`` (without importing it) and asserts ``PROBLEMS`` here is identical.
"""
from __future__ import annotations

import numpy as np

# ---- funcset / constant-init pool (kept identical to dump_evogp for parity) ----
USING_FUNCS = {"+": 1.0, "-": 1.0, "*": 1.0, "/": 1.0, "sin": 0.5, "cos": 0.5, "tan": 0.5}
CONST_SAMPLES = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5]


# 题目库 — Feynman Symbolic Regression Database 的子集 (Udrescu & Tegmark 2020).
# 想加题目把 (id, skeleton_expr, variables, constants, ground_truth_constants,
# sampling_ranges) 加进来即可. 仓库自带的 pop.bin 是用 I.18.12 跑出来的;
# 其它几道附在这里是为了方便友人试不同 K + 不同算子组合.
PROBLEMS = {
    "feynman/I.12.1": {
        # F = μ·N  (最简单的 K=1, 纯乘法)
        "skeleton_expr": "c0*x0*x1",
        "variables": ["x0", "x1"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0)],
        "_note": "I.12.1  mu*Nn  — K=1 baseline",
    },
    "feynman/I.18.12": {
        # τ = r·F·sin(θ)  (默认题, 仓库 pop.bin 来源)
        "skeleton_expr": "c0*x0*x1*sin(x2)",
        "variables": ["x0", "x1", "x2"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0), (0.0, 5.0)],
        "_note": "I.18.12  r*F*sin(theta)  — torque magnitude, K=1 with SIN",
    },
    "feynman/I.27.6": {
        # 薄透镜公式  1 / (1/d1 + n/d2)  (嵌套 DIV)
        "skeleton_expr": "c0/(x2/x1 + 1/x0)",
        "variables": ["x0", "x1", "x2"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        "_note": "I.27.6  1/(1/d1 + n/d2)  — thin-lens, K=1 with nested DIV",
    },
    "feynman/I.6.2": {
        # 标准正态  exp(-(θ/σ)²/2) / (√(2π)·σ)  (高 K, SQRT + EXP + 平方)
        "skeleton_expr": "c0*sqrt(c1)*exp(c3*x1**2/x0**2)/(sqrt(c2)*x0)",
        "variables": ["x0", "x1"],
        "constants": ["c0", "c1", "c2", "c3"],
        "ground_truth_constants": [0.5, 2.0, 3.141592653589793, -0.5],
        "sampling_ranges": [(1.0, 3.0), (1.0, 3.0)],
        "_note": "I.6.2  Gaussian exp(-(theta/sigma)^2/2)/(sqrt(2pi)*sigma)  — K=4, SQRT+EXP+POW",
    },

    # --- more multivariate Feynman (K=1, div/inv-heavy; different n_vars) ---
    "feynman/I.12.2": {
        # Coulomb F = q1*q2/(4π ε0 r²)
        "skeleton_expr": "c0*x0*x1/x2**2",
        "variables": ["x0", "x1", "x2"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        "_note": "I.12.2  Coulomb q1*q2/(4pi eps r^2)  — 3 var, K=1, DIV+sq",
    },
    "feynman/I.13.12": {
        # gravitational PE  U = G*m1*m2*(1/r2 - 1/r1)
        "skeleton_expr": "c0*x0*x1*(1/x3 - 1/x2)",
        "variables": ["x0", "x1", "x2", "x3"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        "_note": "I.13.12  G*m1*m2*(1/r2-1/r1)  — 4 var, K=1, INV",
    },
    "feynman/II.3.24": {
        # radiated flux  φ = P/(4π r²)
        "skeleton_expr": "c0*x0/x1**2",
        "variables": ["x0", "x1"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0)],
        "_note": "II.3.24  flux P/(4pi r^2)  — 2 var, K=1, DIV+sq",
    },

    # --- Nguyen GP-SR benchmark (Uy et al. 2011). Targets are constant-free
    # (K=0 in ground truth); the GP discovers constants, so harvested pops still
    # have K>0 to optimize. GP funcset = {+,-,*,/,sin,cos,tan}; log/sqrt targets
    # (N7/N8) are only approximable — kept for workload variety, same situation as
    # the SQRT/EXP Feynman entries (skeleton only generates y, not a GP constraint). ---
    "nguyen/1": {"skeleton_expr": "x0**3 + x0**2 + x0", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-1  x^3+x^2+x"},
    "nguyen/2": {"skeleton_expr": "x0**4 + x0**3 + x0**2 + x0", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-2  x^4+..+x"},
    "nguyen/3": {"skeleton_expr": "x0**5 + x0**4 + x0**3 + x0**2 + x0", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-3  x^5+..+x"},
    "nguyen/4": {"skeleton_expr": "x0**6 + x0**5 + x0**4 + x0**3 + x0**2 + x0", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-4  x^6+..+x"},
    "nguyen/5": {"skeleton_expr": "sin(x0**2)*cos(x0) - 1", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-5  sin(x^2)cos(x)-1"},
    "nguyen/6": {"skeleton_expr": "sin(x0) + sin(x0 + x0**2)", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-6  sin(x)+sin(x+x^2)"},
    "nguyen/7": {"skeleton_expr": "log(x0 + 1) + log(x0**2 + 1)", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(0.0, 2.0)], "_note": "Nguyen-7  ln(x+1)+ln(x^2+1) (approx)"},
    "nguyen/8": {"skeleton_expr": "sqrt(x0)", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(0.0, 4.0)], "_note": "Nguyen-8  sqrt(x) (approx)"},
    "nguyen/9": {"skeleton_expr": "sin(x0) + sin(x1**2)", "variables": ["x0", "x1"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(0.0, 1.0), (0.0, 1.0)], "_note": "Nguyen-9  sin(x)+sin(y^2)"},
    "nguyen/10": {"skeleton_expr": "2*sin(x0)*cos(x1)", "variables": ["x0", "x1"],
                  "constants": [], "ground_truth_constants": [],
                  "sampling_ranges": [(0.0, 1.0), (0.0, 1.0)], "_note": "Nguyen-10  2sin(x)cos(y)"},

    # --- Korns inner-constant magnifier (Korns 2011), FREQUENCY-class subset. ---
    # Unlike every entry above (whose true constants are OUTER scales or absent),
    # these bury constants INSIDE cos/sin as frequencies — positions linear
    # least-squares cannot fit, so constant optimization (CO) is decisive here.
    # Expressible with the GP funcset {+,-,*,/,sin,cos,tan}: the x0**3 in korns/11
    # is only a y-generator (GP approximates it with x*x*x), same status as the
    # POW/SQRT/LOG Feynman & Nguyen skeletons. Ranges restricted to the CO-benchmark
    # [-5,5] convention (Korns' original U[-50,50] aliases the frequencies into
    # noise). Inner-constant taxonomy: cusr/demonstrator/{problems,taxonomy}.py.
    "korns/11": {
        # 6.87 + 11*cos(7.23*x^3) — one inner frequency (7.23) with a cube ->
        # severe aliasing, a known-ceiling target even with perfect CO.
        "skeleton_expr": "c0 + c1*cos(c2*x0**3)",
        "variables": ["x0"],
        "constants": ["c0", "c1", "c2"],
        "ground_truth_constants": [6.87, 11.0, 7.23],
        "sampling_ranges": [(-5.0, 5.0)],
        "_note": "Korns-11  6.87+11*cos(7.23*x^3)  — inner freq (cube), known ceiling",
    },
    "korns/12": {
        # 2 - 2.1*cos(9.8*x0)*sin(1.3*x1) — two inner frequencies (9.8 & 1.3).
        "skeleton_expr": "c0 + c1*cos(c2*x0)*sin(c3*x1)",
        "variables": ["x0", "x1"],
        "constants": ["c0", "c1", "c2", "c3"],
        "ground_truth_constants": [2.0, -2.1, 9.8, 1.3],
        "sampling_ranges": [(-5.0, 5.0), (-5.0, 5.0)],
        "_note": "Korns-12  2-2.1*cos(9.8*x0)*sin(1.3*x1)  — two inner freqs",
    },
}


def get_problem(problem_id: str) -> dict:
    """Look up a problem by id; raise a helpful KeyError if missing.

    Mirrors ``dump_evogp._load_feynman_problem`` semantics.
    """
    if problem_id not in PROBLEMS:
        raise KeyError(
            f"{problem_id} not in PROBLEMS. Known: {sorted(PROBLEMS.keys())}. "
            f"加新题目: 在 sr_problems.py 的 PROBLEMS dict 里加一条 (并同步 dump_evogp.py)."
        )
    return PROBLEMS[problem_id]


def sample_xy(problem_id: str, N: int, seed: int, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-sample N points; ``y = skeleton(c_true, X)``.

    Returns ``(X float32 (N, n_vars), y float32 (N,))``. ``noise > 0`` adds
    RMS-relative Gaussian noise. The math is identical to
    ``dump_evogp._sample_xy`` (same rng, ranges, sympy skeleton + GT constants,
    noise model) so the produced ``(X, y)`` matches the evogp corpus exactly.
    """
    prob = get_problem(problem_id)
    return _sample_xy(prob, N, seed, noise=noise)


def _sample_xy(prob: dict, N: int, seed: int, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """uniform sample N 个数据点, y = skeleton(c_true, X). noise>0: 加 RMS 相对高斯噪声.

    Copied verbatim from dump_evogp._sample_xy (math must match bit-for-bit).
    """
    import sympy as sp
    rng = np.random.default_rng(seed)
    n_vars = len(prob["variables"])
    X = np.empty((N, n_vars), dtype=np.float64)
    for i, (lo, hi) in enumerate(prob["sampling_ranges"]):
        X[:, i] = rng.uniform(lo, hi, size=N)
    var_syms = tuple(sp.Symbol(v) for v in prob["variables"])
    const_syms = tuple(sp.Symbol(c) for c in prob["constants"])
    locals_ = {str(s): s for s in (*var_syms, *const_syms)}
    expr = sp.sympify(prob["skeleton_expr"], locals=locals_)
    subs = {c: float(v) for c, v in zip(const_syms, prob["ground_truth_constants"])}
    expr = expr.subs(subs)
    f = sp.lambdify(var_syms, expr, modules="numpy")
    y = np.asarray(f(*X.T), dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError("sampled y has non-finite — check ranges")
    if noise > 0.0:
        sigma = noise * float(np.sqrt(np.mean(y ** 2)))  # RMS-relative Gaussian noise
        y = y + rng.normal(0.0, sigma, size=y.shape)
    return X.astype(np.float32), y.astype(np.float32)
