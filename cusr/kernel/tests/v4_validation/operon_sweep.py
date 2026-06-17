import subprocess, os, re, glob, statistics, sys
sys.path.insert(0, '/home/weish/hao/CuSR')
from cusr.kernel.verify import load_pop_bin
SNAP = '/home/weish/hao/CuSR/data/workload/snapshots'
BINS = {'AD': '/home/weish/hao/CuSR/cusr/kernel/batch_lm_ad',
        'FD': '/home/weish/hao/CuSR/cusr/kernel/batch_lm_fusedfd'}
os.makedirs('/tmp/op_out', exist_ok=True)

# distinct Operon problems (first variant per problem)
cells = {}
for d in sorted(glob.glob(f'{SNAP}/operon_*')):
    if not os.path.isdir(d):
        continue
    m = re.match(r'operon_(.+?)_pop', os.path.basename(d))
    key = m.group(1) if m else os.path.basename(d)
    cells.setdefault(key, d)
sel = list(cells.items())[:5]


def run(b, pop):
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = '0'
    r = subprocess.run([b, pop, '/tmp/op_out', '--max-iter', '50'],
                       capture_output=True, text=True, env=env)
    m = re.search(r'总耗时 ([\d.]+) 秒 \((\d+) trees/s', r.stdout)
    return int(m.group(2)) if m else None


print(f"{'dataset':22} {'gen':>4} {'Kmean':>6} | {'AD t/s':>9} {'FD t/s':>9} {'AD/FD':>6}")
print('-' * 64)
rows = []
for key, d in sel:
    for g in [0, 16, 100]:
        pop = f'{d}/pop_gen{g:04d}.bin'
        if not os.path.exists(pop):
            continue
        km = float(load_pop_bin(pop)['metas'][:, 3].mean())
        ad = [run(BINS['AD'], pop) for _ in range(2)]
        fd = [run(BINS['FD'], pop) for _ in range(2)]
        if None in ad or None in fd:
            print(f"{key:22} {g:>4} {km:>6.1f} | PARSE/RUN FAIL"); continue
        a, f = min(ad), min(fd)   # best-of-2 (least jitter)
        rows.append((key, g, km, a, f))
        print(f"{key:22} {g:>4} {km:>6.1f} | {a:>9.0f} {f:>9.0f} {a/f:>5.2f}x", flush=True)

if rows:
    import statistics as st
    print('-' * 64)
    print(f"AD/FD ratio: min={min(r[3]/r[4] for r in rows):.2f}x "
          f"median={st.median([r[3]/r[4] for r in rows]):.2f}x "
          f"max={max(r[3]/r[4] for r in rows):.2f}x  (over {len(rows)} cells)")
