"""Tier0 throughput M-sweep driver — one variant on one pinned GPU.
Paper-grade: warmup discarded, N timed repeats, SM clock sampled per run,
device wall-time parsed from the binary's 总耗时 line. fp32.
"""
import argparse, json, os, re, statistics, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--gpu", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mlist", default="1000,4000,16000,64000")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--reps", type=int, default=7)
args = ap.parse_args()

SYNTH = Path("data/workload/synth")
REAL = Path("data/fixtures/pop.bin")
env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = args.gpu
TIME_RE = re.compile(r"总耗时\s+([0-9.]+)")

def sm_clock():
    try:
        r = subprocess.run(["nvidia-smi", "-i", args.gpu, "--query-gpu=clocks.sm",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1

def run_once(popbin):
    odir = f"/tmp/sweep_{args.label}_{args.gpu}"
    os.makedirs(odir, exist_ok=True)
    r = subprocess.run([args.bin, str(popbin), odir],
                       capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        return None, r.stderr[-200:]
    m = TIME_RE.search(r.stdout)
    return (float(m.group(1)) if m else None), ""

points = [("real1000", REAL, 1000)]
for M in [int(x) for x in args.mlist.split(",")]:
    f = SYNTH / f"synth_early-gen_M{M}_N1000_seed0.bin"
    if f.exists():
        points.append((f"synth{M}", f, M))

rows = []
for name, popbin, M in points:
    for _ in range(args.warmup):
        run_once(popbin)
    times, clocks, err = [], [], ""
    for _ in range(args.reps):
        clocks.append(sm_clock())
        t, e = run_once(popbin)
        if t is not None:
            times.append(t)
        elif e:
            err = e
    if not times:
        rows.append(dict(label=args.label, point=name, M=M, ok=False, err=err))
        print(f"{args.label} {name} M={M}: FAIL {err}", flush=True)
        continue
    tps = sorted(M / t for t in times)
    row = dict(label=args.label, point=name, M=M, ok=True, n=len(times),
               t_med=statistics.median(times), t_min=min(times), t_max=max(times),
               tps_med=statistics.median(tps), tps_p10=tps[len(tps)//10], tps_p90=tps[-1],
               sm_clock_med=statistics.median(clocks))
    rows.append(row)
    print(f"{args.label} {name} M={M}: {row['tps_med']:.0f} t/s "
          f"(t_med={row['t_med']:.3f}s clk~{row['sm_clock_med']}MHz n={row['n']})", flush=True)

Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"DONE {args.label} -> {args.out}", flush=True)
