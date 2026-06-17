import subprocess, statistics, os, re
K = '/home/weish/hao/CuSR/cusr/kernel'
SYN = '/home/weish/hao/CuSR/data/workload/synth'
Ms = [1000, 4000, 16000, 64000, 256000]
bins = {'AD': f'{K}/batch_lm_ad', 'FD': f'{K}/batch_lm_fusedfd'}
os.makedirs('/tmp/sw_out', exist_ok=True)


def run(b, pop):
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = '0'
    r = subprocess.run([b, pop, '/tmp/sw_out'], capture_output=True, text=True, env=env)
    m = re.search(r'总耗时 ([\d.]+) 秒 \((\d+) trees/s', r.stdout)
    if not m:
        print("PARSE FAIL:", r.stdout[-300:], r.stderr[-300:]); return None, None
    return float(m.group(1)), int(m.group(2))


print(f"{'M':>8} | {'AD trees/s':>11} {'FD trees/s':>11} {'AD/FD':>7} | {'AD wall':>8} {'FD wall':>8}")
print("-" * 64)
for M in Ms:
    pop = f'{SYN}/synth_early-gen_M{M}_N1000_seed0.bin'
    out = {}
    for tag, b in bins.items():
        ts, secs = [], []
        for _ in range(3):
            s, t = run(b, pop)
            if t:
                ts.append(t); secs.append(s)
        out[tag] = (statistics.median(ts) if ts else 0, statistics.median(secs) if secs else 0)
    (adt, ads), (fdt, fds) = out['AD'], out['FD']
    ratio = adt / fdt if fdt else 0
    print(f"{M:>8} | {adt:>11.0f} {fdt:>11.0f} {ratio:>6.2f}x | {ads:>7.3f}s {fds:>7.3f}s", flush=True)
