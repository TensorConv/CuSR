"""cross_kernel.py — run the SAME batch_lm_fusedfd build on matched cells of BOTH
the Operon and evogp corpora, to compare LM conditioning (fail_cholesky =
rank-deficient JtJ) on the two engines' pre-CO populations.

Advisor Guard 2: this measures the SYMPTOM (population fail_cholesky% vs generation
/ node count). The MECHANISM distinction (Operon = structural per-leaf-weight
over-parameterization, proven singular at zero bloat; evogp = bloat-driven redundant
constants) is proven separately in mechanism_proof.py. We report the rates honestly
— including that k0_skip differs (evogp has many K=0 trees; Operon has none) — and do
NOT claim "similar rates => same cause".

Matched grid: all 17 problems x gens {0,4,16,64,100} x cap32 x seed0, both corpora,
same freshly-built kernel (default --max-iter 50, --quiet). Sharded across free GPUs
(one process per GPU; fail_cholesky is deterministic so co-tenancy would not bias it,
but we still leave 2 GPUs free per the standing rule).
"""
import sys; sys.path.insert(0, "/home/weish/hao/CuSR")
import json, queue, re, subprocess, threading, time
from pathlib import Path

ROOT = Path("/home/weish/hao/CuSR")
KERNEL = ROOT / "cusr/kernel/batch_lm_fusedfd"
SNAP = ROOT / "data/workload/snapshots"
OUT = ROOT / "results/operon_corpus__20260617/data/cross_kernel.json"

PROBLEMS = ["feynman/I.12.1","feynman/I.18.12","feynman/I.27.6","feynman/I.6.2",
    "feynman/I.12.2","feynman/I.13.12","feynman/II.3.24",
    "nguyen/1","nguyen/2","nguyen/3","nguyen/4","nguyen/5","nguyen/6","nguyen/7",
    "nguyen/8","nguyen/9","nguyen/10"]
GENS = [0, 4, 16, 64, 100]
GPUS = [0, 1, 2, 3, 4, 5]          # leave 6,7 free (standing rule)
DONE_RE = re.compile(
    r"converged=(\d+) \(([\d.]+)%\).*?maxiter=(\d+).*?fail_nan=(\d+).*?"
    r"fail_cholesky=(\d+).*?k0_skip=(\d+)")

def cell_dir(corpus, prob):
    safe = prob.replace("/", "_")
    base = f"{safe}_pop4000_noise0_len32_seed0"
    return SNAP / (f"operon_{base}" if corpus == "operon" else base)

def manifest_stat(corpus, prob, gen):
    mf = cell_dir(corpus, prob) / "manifest.json"
    if not mf.exists(): return {}
    m = json.loads(mf.read_text())
    for s in m.get("snapshots", []):
        if s.get("gen") == gen:
            return {k: s.get(k) for k in ("mean_nodes","mean_K","K_max","total_c","M")}
    return {}

def run_one(corpus, prob, gen, gpu, outdir):
    f = cell_dir(corpus, prob) / f"pop_gen{gen:04d}.bin"
    if not f.exists():
        return dict(corpus=corpus, dataset=prob, gen=gen, error="missing_bin", file=str(f))
    import os
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # NOTE: do NOT pass --quiet — it suppresses the "完成: converged=..." summary
    # line that DONE_RE parses (the iteration prints it leaves are ignored by the regex).
    p = subprocess.run([str(KERNEL), str(f), str(outdir), "--max-iter", "50"],
                       capture_output=True, text=True, env=env, timeout=600)
    mt = DONE_RE.search(p.stdout)
    rec = dict(corpus=corpus, dataset=prob, gen=gen)
    rec.update(manifest_stat(corpus, prob, gen))
    if mt:
        conv, pct, maxit, fnan, fchol, k0 = mt.groups()
        M = int(re.search(r"M_prob=(\d+)", p.stdout).group(1)) if "M_prob=" in p.stdout else rec.get("M")
        rec.update(M=M, conv_pct=float(pct), maxiter=int(maxit), fail_nan=int(fnan),
                   fail_chol=int(fchol), k0_skip=int(k0),
                   fail_chol_pct=round(100*int(fchol)/M, 2) if M else None,
                   active=M-int(k0) if M else None)
    else:
        rec.update(error="no_match", stderr=p.stderr[-300:])
    return rec

def worker(gpu, q, results, lock):
    outdir = Path(f"/tmp/ks_{gpu}"); outdir.mkdir(parents=True, exist_ok=True)
    while True:
        try: corpus, prob, gen = q.get_nowait()
        except queue.Empty: return
        try: rec = run_one(corpus, prob, gen, gpu, outdir)
        except Exception as e: rec = dict(corpus=corpus, dataset=prob, gen=gen, error=repr(e))
        with lock:
            results.append(rec)
            tag = f"{rec.get('fail_chol_pct','?')}%fc" if "fail_chol_pct" in rec else rec.get("error","?")
            print(f"  gpu{gpu} {corpus:6} {prob:15} g{gen:<3} -> {tag}", flush=True)
        q.task_done()

def main():
    cells = [(c, p, g) for c in ("operon","evogp") for p in PROBLEMS for g in GENS]
    print(f"matched sweep: {len(PROBLEMS)} problems x {len(GENS)} gens x 2 corpora = {len(cells)} kernel runs on GPUs {GPUS}")
    q = queue.Queue()
    for c in cells: q.put(c)
    results, lock = [], threading.Lock()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(g, q, results, lock), daemon=True) for g in GPUS]
    for t in threads: t.start()
    for t in threads: t.join()
    OUT.write_text(json.dumps(results, indent=1))
    errs = [r for r in results if "error" in r]
    print(f"\ndone in {(time.time()-t0)/60:.1f} min: {len(results)} runs, {len(errs)} errors -> {OUT}")
    if errs: print("errors:", [(e['corpus'],e['dataset'],e['gen'],e['error']) for e in errs[:10]])

if __name__ == "__main__":
    main()
