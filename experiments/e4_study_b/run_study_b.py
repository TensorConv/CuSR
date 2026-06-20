"""Study B — the 4-arm end-to-end EvoGP+CO study (fixed-generation protocol).

Answers Claim 2 empirically: does embedding CO in the per-generation evolutionary
loop improve end-to-end solution accuracy / symbolic recovery over stock EvoGP,
and where does the gain concentrate?

  arms (arms.py):  no_co < sparse_gpu < cpu_every < gpu_every  (only the CO knob
                   differs; the seed fixes an IDENTICAL initial population across
                   all four arms, which licenses the per-problem PAIRED stats).
  corpus (corpus.py): the e3 21-problem constructed inner-CO corpus (ADMIT) +
                   the REJECT controls (validity: CO must NOT unlock a control).
  metrics:        held-out R² (EvoGP tree forward on a fresh test set — works for
                  bloated trees, no sympy) + best-tree symbolic recovery
                  (end-to-end, NOT the per-tree CO recovery rate) + honest compute
                  accounting (co_fits / evals / wall / kernel-vs-fallback).

PROTOCOL = fixed-generation only (this build). It makes NO hardware claim:
no_co-vs-CO answers "does CO help" (device-independent); cpu_every-vs-gpu_every
is a QUALITY-PARITY check (fp64 scipy vs fp32-guard kernel reach the same
recovery?). The "GPU wins" claim is wallclock-only — a separate phase-2 protocol.

RUN (env MUST be sourced so the inproc .so + CUDA libs load):
  source scripts/env.sh
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.e4_study_b.run_study_b --smoke
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.e4_study_b.run_study_b --seeds 5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as _mp
import os as _os
import pathlib
import signal as _signal
import sys
import time
import warnings as _warnings

import numpy as np
import torch

from cusr.demonstrator import seed_bench as sb
from cusr.demonstrator.pipeline import MemeticPipeline

from . import arms as ARMS
from . import corpus as CORPUS
from ._judge import recovery_hard as _judge_recovery_hard

HERE = pathlib.Path(__file__).resolve().parent

# scipy LM probes pathological c-values on bloated trees -> floods of benign
# overflow/invalid RuntimeWarnings. Suppress at module level so the per-cell
# SPAWN subprocess (which runs _cell_worker, not main) is quiet too.
_warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

# Op-set: the audited union set from run_bench (covers every true_expr operator).
FUNCS = {"+": 1.0, "-": 1.0, "*": 1.0, "/": 1.0,
         "sin": 0.5, "cos": 0.5, "exp": 0.3, "log": 0.3, "sqrt": 0.3, "tanh": 0.2}
# Defaults mirror run_bench; smoke shrinks them. CO_MAX_ITER aligns scipy
# max_nfev and the kernel LM-iteration budget at the SAME number (50) so the
# cpu_every/gpu_every parity is at matched budget (currencies still differ — a
# named factor, reported, not silently equated).
CFG = dict(pop=600, n_gen=60, top_k=16, n_train=200, n_test=1000,
           co_max_iter=50, early_r2=0.9999, sample_cnt=10000)
SMOKE = dict(pop=200, n_gen=15, top_k=16, n_train=200, n_test=1000,
             co_max_iter=50, early_r2=0.9999, sample_cnt=10000)


def run_arm(prob: sb.Problem, arm: ARMS.Arm, seed: int, cfg: dict) -> dict:
    from evogp.algorithm import (DefaultCrossover, DefaultMutation,
                                 GeneticProgramming, TournamentSelection)
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor

    X_np, y_np = sb.generate(prob, seed=seed, n_samples=cfg["n_train"])
    # Held-out test set: SAME ranges, a disjoint seed (interpolation hold-out).
    X_te, y_te = sb.generate(prob, seed=seed + 100_000, n_samples=cfg["n_test"])

    # Seed BEFORE forest generation -> identical initial population across arms.
    torch.manual_seed(seed)
    np.random.seed(seed)
    desc = GenerateDescriptor(
        max_tree_len=64, input_len=prob.n_vars, output_len=1,
        using_funcs=FUNCS, max_layer_cnt=6, layer_leaf_prob=0.3,
        const_range=[-5.0, 5.0], sample_cnt=cfg["sample_cnt"],
    )
    forest = Forest.random_generate(pop_size=cfg["pop"], descriptor=desc)
    problem = SymbolicRegression(
        datapoints=torch.from_numpy(X_np.astype(np.float32)).cuda(),
        labels=torch.from_numpy(y_np.astype(np.float32).reshape(-1, 1)).cuda())
    algo = GeneticProgramming(
        initial_forest=forest, crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.1, descriptor=desc.update(max_layer_cnt=4)),
        selection=TournamentSelection(tournament_size=20, survivor_rate=0.5, elite_rate=0.1))

    backend = arm.make_backend()
    pipe = MemeticPipeline(
        algo, problem, backend, problem_n_vars=prob.n_vars, X_np=X_np, y_np=y_np,
        top_k=cfg["top_k"], co_every=arm.co_every, co_probability=arm.co_probability,
        co_rng_seed=seed, co_max_iter=cfg["co_max_iter"], generation_limit=cfg["n_gen"])

    early = -(1.0 - cfg["early_r2"]) * float(np.var(y_np))
    t0 = time.time()
    gens = 0
    for _ in range(cfg["n_gen"]):
        pipe.step()
        gens += 1
        if pipe.best_fitness >= early:
            break
    wall = time.time() - t0
    print(f"  .. {prob.id} {arm.name} s{seed}: run done ({wall:.1f}s) -> r2",
          file=sys.stderr, flush=True)

    # --- held-out R² via the EvoGP tree forward (bloated-safe, no sympy) ---
    r2_test = float("nan")
    if pipe.best_tree is not None:
        try:
            with torch.no_grad():
                pred = pipe.best_tree.forward(
                    torch.from_numpy(X_te.astype(np.float32)).cuda())
            pred = np.asarray(pred.detach().cpu(), dtype=float).reshape(-1)
            if pred.shape[0] == y_te.shape[0] and np.all(np.isfinite(pred)):
                ss_res = float(np.sum((y_te - pred) ** 2))
                ss_tot = float(np.sum((y_te - float(np.mean(y_te))) ** 2))
                r2_test = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        except Exception:  # noqa: BLE001 — held-out R² is best-effort
            pass

    # --- best-tree symbolic recovery (end-to-end, not per-tree CO recovery) ---
    # The ENTIRE symbolic path (to_sympy_expr + judge) is gated on r2_test >= 0.99:
    # a tree that does not fit cannot be a symbolic recovery, so this is correct,
    # AND it keeps pathological non-recovery trees out of to_sympy_expr — an
    # unbounded PARENT call on an unpicklable tree that NO timeout (SIGALRM or
    # subprocess SIGKILL) can interrupt. This is the real hang guard.
    print(f"  .. {prob.id} {arm.name} s{seed}: r2={r2_test:.3f} -> recovery",
          file=sys.stderr, flush=True)
    cols = {"strict": False, "lenient": False, "srbench": False}
    best_expr = None
    if pipe.best_tree is not None and np.isfinite(r2_test) and r2_test >= 0.99:
        try:
            n_nodes = int(pipe.best_tree.subtree_size[0].item())
        except Exception:  # noqa: BLE001
            n_nodes = 0
        if n_nodes > 50:
            best_expr = f"<{n_nodes} nodes (bloated) -> non-recovery>"
        else:
            try:
                best_expr = str(pipe.best_tree.to_sympy_expr())
            except Exception as e:  # noqa: BLE001
                best_expr = f"<to_sympy err: {e}>"
            if not best_expr.startswith("<"):
                # Decouple the judge from the already-computed r2_test: a judge
                # failure (or its subprocess dying) must degrade to no-recovery,
                # NOT discard this cell's valid r2_test by raising out of run_arm.
                try:
                    cols = _judge_recovery_hard(best_expr, prob.true_expr, X_np, timeout=15)
                except Exception:  # noqa: BLE001
                    cols = {"strict": False, "lenient": False, "srbench": False}

    # --- kernel-actually-ran accounting (advisor #3) ---
    ks = dict(getattr(backend, "cum_stats", {}) or {})
    n_kernel = int(ks.get("n_kernel", 0))
    n_fb = int(ks.get("n_fallback", 0))
    n_capped = int(ks.get("cum_capped", 0))
    kernel_frac = (n_kernel / (n_kernel + n_fb)) if (n_kernel + n_fb) else 0.0

    return {
        "id": prob.id, "arm": arm.name, "seed": seed,
        "is_control": prob.is_control, "n_inner": prob.n_inner_consts,
        "difficulty": prob.difficulty,
        "r2_test": r2_test,
        "solved": bool(np.isfinite(r2_test) and r2_test > 0.999),  # SRBench accuracy-solution
        "best_fit": float(pipe.best_fitness),
        "strict": cols["strict"], "lenient": cols["lenient"], "srbench": cols["srbench"],
        "gens": gens, "co_fits": pipe.n_co_fits, "rollbacks": pipe.n_rollbacks,
        "evals": pipe.n_evals, "wall_s": round(wall, 2),
        "n_kernel": n_kernel, "n_fallback": n_fb, "n_capped": n_capped,
        "kernel_frac": round(kernel_frac, 3), "uses_kernel": arm.uses_kernel,
        "best_expr": best_expr, "true_expr": prob.true_expr,
    }


# --- per-cell hard isolation (SIGKILL timeout) ------------------------------
# scipy/MINPACK can hang inside a SINGLE least_squares call on a pathological
# tree, in C code that ignores SIGALRM and that max_nfev does not bound. The only
# reliable interrupt is to run each cell in a spawn subprocess and SIGKILL it on
# timeout. Cost: each cell re-inits CUDA (~3-5 s) — acceptable for the
# fixed-GENERATION protocol (we measure recovery/quality, not wallclock; the
# kernel's "context paid once" amortization is a wallclock-phase concern).

def _timeout_row(prob, arm, seed, reason: str, wall: float) -> dict:
    return {
        "id": prob.id, "arm": arm.name, "seed": seed,
        "is_control": prob.is_control, "n_inner": prob.n_inner_consts,
        "difficulty": prob.difficulty,
        "r2_test": float("nan"), "solved": False, "best_fit": float("nan"),
        "strict": False, "lenient": False, "srbench": False,
        "gens": 0, "co_fits": 0, "rollbacks": 0, "evals": 0,
        "wall_s": round(wall, 1),   # a killed cell consumed >= timeout, NOT 0
        "n_kernel": 0, "n_fallback": 0, "n_capped": 0, "kernel_frac": 0.0,
        "uses_kernel": arm.uses_kernel, "best_expr": None, "true_expr": prob.true_expr,
        "_failed": reason,
    }


def _cell_worker(q, prob, arm_name, seed, cfg, device, kmi):
    # Own session/process-group so a timeout can killpg the WHOLE subtree (this
    # worker + any judge grandchild it spawned), not just this pid.
    try:
        _os.setsid()
    except Exception:  # noqa: BLE001
        pass
    try:
        arms = ARMS.make_arms(device_id=device, kernel_max_iter=kmi, scipy_max_nfev=kmi)
        arm = next(a for a in arms if a.name == arm_name)
        q.put(run_arm(prob, arm, seed, cfg))
    except Exception as e:  # noqa: BLE001
        q.put({"_err": repr(e)})


def run_cell_guarded(prob, arm, seed, cfg, *, device, kmi, timeout) -> dict:
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    # daemon=False is REQUIRED: the cell may spawn the judge subprocess, and
    # CPython forbids a daemonic process from having children. We always join (or
    # killpg) below, so there is no orphan risk from non-daemon.
    p = ctx.Process(target=_cell_worker, args=(q, prob, arm.name, seed, cfg, device, kmi),
                    daemon=False)
    t0 = time.time()
    p.start()
    p.join(timeout)
    if p.is_alive():
        try:  # kill the whole process group (worker + judge grandchild)
            _os.killpg(_os.getpgid(p.pid), _signal.SIGKILL)
        except Exception:  # noqa: BLE001
            try:
                _os.kill(p.pid, _signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        p.join(5)
        print(f"  !! TIMEOUT {prob.id} {arm.name} s{seed} (>{timeout}s) — SIGKILLed",
              file=sys.stderr, flush=True)
        return _timeout_row(prob, arm, seed, f"killed>{timeout}s", time.time() - t0)
    try:
        r = q.get(timeout=10)
    except Exception:  # noqa: BLE001
        r = {"_err": "no result from worker (died)"}
    if "_err" in r:
        print(f"  !! ERROR {prob.id} {arm.name} s{seed}: {r['_err']}", file=sys.stderr, flush=True)
        return _timeout_row(prob, arm, seed, r["_err"], time.time() - t0)
    return r


def _median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def _clamp0(x):
    """Floor R² at 0 for aggregation. A model worse than predicting the mean has
    'no skill' (0), not arbitrarily-large-negative — keeps blow-ups (r2=-296) from
    dominating medians/paired deltas. CRITICALLY, a non-finite r2 (a FAIL/timeout
    cell, or an inf-blowup prediction) ALSO maps to 0.0: 'no usable model in
    budget' == 'no skill' == the same floor. This makes FAIL cells enter the
    clamped median + paired vectors the SAME way the rate denominator already
    counts them (the reviewer-found NaN asymmetry that flipped the winner). The
    RAW median_r2_test (nan-dropped, completed runs only) stays separately honest."""
    if x is None or not np.isfinite(x):
        return 0.0
    return max(float(x), 0.0)


def summarize(rows: list[dict], arm_names: list[str]) -> dict:
    """Per-arm recovery/R² + per-problem PAIRED stats on the ADMITTED set, plus
    the multi-inner split and the control-leak check."""
    from scipy.stats import wilcoxon

    adm = [r for r in rows if not r["is_control"]]
    ctl = [r for r in rows if r["is_control"]]

    def arm_block(subset):
        out = {}
        for a in arm_names:
            ar = [r for r in subset if r["arm"] == a]
            n = len(ar)
            out[a] = {
                "n": n,
                "lenient_rec": sum(int(r["lenient"]) for r in ar),
                "lenient_rate": (sum(int(r["lenient"]) for r in ar) / n) if n else 0.0,
                "solved_rec": sum(int(r["solved"]) for r in ar),
                "solved_rate": (sum(int(r["solved"]) for r in ar) / n) if n else 0.0,
                "median_r2_test": _median([r["r2_test"] for r in ar]),           # raw (honest)
                "median_r2_clamped": _median([_clamp0(r["r2_test"]) for r in ar]),  # floored at 0
                "n_failed": sum(1 for r in ar if r.get("_failed")),
                "mean_wall_s": (sum(r["wall_s"] for r in ar) / n) if n else 0.0,
                "median_kernel_frac": _median([r["kernel_frac"] for r in ar if r["uses_kernel"]]),
            }
        return out

    # per-problem median r2_test per arm (paired object across problems)
    prob_ids = sorted({r["id"] for r in adm})
    by = {(r["id"], r["arm"]): [] for r in adm}
    for r in adm:
        by[(r["id"], r["arm"])].append(r["r2_test"])
    # paired object = per-problem median of CLAMPED held-out R² (blow-up-robust)
    pp = {a: [_median([_clamp0(v) for v in by.get((pid, a), [])]) for pid in prob_ids]
          for a in arm_names}

    def paired(a, b):
        xa = np.array(pp[a]); xb = np.array(pp[b])
        m = np.isfinite(xa) & np.isfinite(xb)
        xa, xb = xa[m], xb[m]
        d = xb - xa
        res = {"n_pairs": int(m.sum()),
               "median_delta_r2": float(np.median(d)) if d.size else float("nan"),
               "wins_b": int(np.sum(d > 0)), "wins_a": int(np.sum(d < 0))}
        if d.size >= 1 and np.any(d != 0):
            try:
                res["wilcoxon_p"] = float(wilcoxon(xa, xb).pvalue)
            except Exception:  # noqa: BLE001
                res["wilcoxon_p"] = float("nan")
        else:
            res["wilcoxon_p"] = float("nan")
        return res

    pairs = {}
    if {"no_co", "gpu_every"} <= set(arm_names):
        pairs["gpu_every_vs_no_co"] = paired("no_co", "gpu_every")
    if {"cpu_every", "gpu_every"} <= set(arm_names):
        pairs["gpu_every_vs_cpu_every"] = paired("cpu_every", "gpu_every")
    if {"no_co", "cpu_every"} <= set(arm_names):
        pairs["cpu_every_vs_no_co"] = paired("no_co", "cpu_every")

    # multi-inner split (Claim 2's "largest gains on multiple inner constants")
    split = {}
    for bucket in ("single_inner", "multi_inner"):
        sub = [r for r in adm if r["difficulty"] == bucket]
        split[bucket] = arm_block(sub) if sub else {}

    # multi-inner per-problem clamped R² per arm (DESCRIPTIVE, n is tiny and the
    # problems are one near-clone family -> NO inferential p-value reported here;
    # the inferential Claim-2 multi-inner test is deferred to the pre-registered
    # extension set, per the review).
    multi_ids = sorted({r["id"] for r in adm if r["difficulty"] == "multi_inner"})
    id_index = {pid: i for i, pid in enumerate(prob_ids)}
    multi_inner_deltas = [
        {"id": pid, "by_arm": {a: pp[a][id_index[pid]] for a in arm_names}}
        for pid in multi_ids if pid in id_index
    ]

    return {
        "admitted": arm_block(adm),
        "controls": arm_block(ctl),
        "paired_on_r2_test": pairs,
        "by_inner_count": split,
        "multi_inner_deltas": multi_inner_deltas,
        "n_admitted_problems": len(prob_ids),
        "n_multi_inner_problems": len(multi_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 problems (1 single + 1 multi inner) x 2 seeds x 4 arms, small config + cost projection")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--problems", nargs="*", default=None, help="restrict to these ids")
    ap.add_argument("--controls", action="store_true", help="also run the REJECT controls")
    ap.add_argument("--device", type=int, default=0, help="device_id for the kernel arms")
    ap.add_argument("--timeout", type=int, default=None,
                    help="per-cell SIGKILL timeout in s (default: 60 smoke / 150 full)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = dict(SMOKE if args.smoke else CFG)
    admitted, controls = CORPUS.load(include_controls=args.controls)
    probs = admitted + (controls if args.controls else [])
    if args.problems:
        keep = set(args.problems)
        probs = [p for p in probs if p.id in keep]
    if args.smoke:
        # one single-inner + one multi-inner, to exercise both buckets + the kernel
        single = next((p for p in admitted if p.difficulty == "single_inner"), None)
        multi = next((p for p in admitted if p.difficulty == "multi_inner"), None)
        probs = [p for p in (single, multi) if p is not None]
        n_seeds = 2
    else:
        n_seeds = args.seeds

    timeout = args.timeout or (60 if args.smoke else 150)
    arm_list = ARMS.make_arms(device_id=args.device, kernel_max_iter=cfg["co_max_iter"],
                              scipy_max_nfev=cfg["co_max_iter"])
    arm_names = [a.name for a in arm_list]

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    fn = out / ("smoke.jsonl" if args.smoke else "study_b_results.jsonl")
    sink = fn.open("w")

    print(f"Study B  problems={[p.id for p in probs]}  arms={arm_names}  seeds={n_seeds}")
    print(f"  cfg={cfg}")
    print(f"\n{'id':30}{'arm':12}{'seed':>5}{'r2_test':>9}{'len':>5}{'gens':>5}"
          f"{'co_fits':>8}{'kf':>6}{'wall':>8}")
    rows = []
    kernel_checked = False
    for prob in probs:
        for arm in arm_list:
            for seed in range(n_seeds):
                r = run_cell_guarded(prob, arm, seed, cfg, device=args.device,
                                     kmi=cfg["co_max_iter"], timeout=timeout)
                rows.append(r)
                sink.write(json.dumps(r) + "\n"); sink.flush()
                flag = " FAIL:" + r["_failed"] if r.get("_failed") else ""
                print(f"{r['id'][:29]:30}{r['arm']:12}{seed:>5}{r['r2_test']:>9.4f}"
                      f"{str(r['lenient'])[:1]:>5}{r['gens']:>5}{r['co_fits']:>8}"
                      f"{r['kernel_frac']:>6.2f}{r['wall_s']:>7.1f}s{flag}")
                # advisor #3: the first time a GPU arm actually does CO (some trees
                # reached the backend, i.e. not all capped), it MUST route through the
                # kernel — else the .so/env is broken and it silently became a CPU arm.
                if arm.uses_kernel and not kernel_checked and (r["n_kernel"] + r["n_fallback"]) > 0:
                    kernel_checked = True
                    if r["n_kernel"] == 0:
                        sink.close()
                        raise RuntimeError(
                            f"GPU arm {arm.name!r} did CO on {r['n_fallback']} trees but routed "
                            f"0 through the kernel. The inproc .so/env is broken — did you "
                            f"`source scripts/env.sh`? Aborting before burning the sweep."
                        )
    sink.close()

    summary = summarize(rows, arm_names)
    report = {"config": cfg, "arms": {a.name: a.label for a in arm_list},
              "n_seeds": n_seeds, "summary": summary}
    rep_fn = out / ("smoke_report.json" if args.smoke else "report.json")
    rep_fn.write_text(json.dumps(report, indent=2))

    # --- console summary ---
    print("\n=== per-arm (ADMITTED) ===")
    print(f"{'arm':12}{'solved':>9}{'lenient':>9}{'medR2raw':>10}{'medR2>=0':>10}{'fail':>5}{'mwall':>8}")
    for a in arm_names:
        b = summary["admitted"][a]
        print(f"{a:12}{b['solved_rec']:>3}/{b['n']:<5}{b['lenient_rec']:>3}/{b['n']:<5}"
              f"{b['median_r2_test']:>10.3f}{b['median_r2_clamped']:>10.3f}"
              f"{b['n_failed']:>5}{b['mean_wall_s']:>7.1f}s")
    print("\n=== paired on held-out R² (per-problem, ADMITTED) ===")
    for k, v in summary["paired_on_r2_test"].items():
        print(f"  {k:24} median Δr2={v['median_delta_r2']:+.4f}  "
              f"wins {v['wins_b']}/{v['n_pairs']}  wilcoxon_p={v['wilcoxon_p']:.4g}")
    print(f"\n=== by inner-count (CLAMPED med R2; Claim 2; n_multi={summary['n_multi_inner_problems']}) ===")
    for bucket in ("single_inner", "multi_inner"):
        blk = summary["by_inner_count"].get(bucket) or {}
        if blk:
            line = "  ".join(f"{a}:{blk[a]['median_r2_clamped']:.3f}(f{blk[a]['n_failed']})"
                             for a in arm_names if a in blk)
            print(f"  {bucket:13} {line}")
    md = summary.get("multi_inner_deltas") or []
    if md:
        print(f"  multi-inner per-problem clamped R2 (n={len(md)}, DESCRIPTIVE — no p-value, one near-clone family):")
        for d in md:
            print(f"    {d['id'][:30]:32} "
                  + " ".join(f"{a}:{d['by_arm'][a]:.3f}" for a in arm_names))
    if args.controls:
        print("\n=== controls (CO must NOT unlock — recovery should be ~equal/low) ===")
        for a in arm_names:
            b = summary["controls"][a]
            print(f"  {a:12} lenient {b['lenient_rec']}/{b['n']}  med_r2_test {b['median_r2_test']:.4f}")

    if args.smoke:
        cpu = [r for r in rows if r["arm"] == "cpu_every"]
        gpu = [r for r in rows if r["arm"] == "gpu_every"]
        cpu_med = _median([r["wall_s"] for r in cpu])
        gpu_med = _median([r["wall_s"] for r in gpu])
        n_full = len(CORPUS.load()[0])
        print(f"\n=== COST PROJECTION (smoke wall, SMALL config — full config is bigger) ===")
        print(f"  cpu_every median {cpu_med:.1f}s/run  |  gpu_every median {gpu_med:.1f}s/run")
        print(f"  full = {n_full} admitted x {len(arm_names)} arms x N seeds "
              f"(cpu_every is the bottleneck; scale by full pop/gen too)")

    print(f"\nwrote {fn}\nwrote {rep_fn}")


if __name__ == "__main__":
    main()
