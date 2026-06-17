"""build_fixture.py — extract the kernel's fp32-honesty offenders into small committed
pop.bin fixtures, so any kernel build can be regression-tested without the full corpus.

An "offender" (SERIOUS subset) is a pre-CO tree on which the established kernel (FD or AD)
reports status==0 (converged) yet the neutral fp64 0.5·SSE at its returned c_final is
> 2× the loss at the starting c_init — i.e. the kernel CLAIMS success but made the tree
worse in exact arithmetic (the fast-math/fp32 honesty gap, see
results/operon_baseline__20260617/report.html finding F5).

Per problem (all gens share one (X,y) at cap32/seed0/noise0) we collect those trees and
write ONE fixture pop.bin (the trees with their ORIGINAL c_init, so a kernel re-runs CO
from the same start) plus manifest.json recording, per tree, where it came from and how
FD/AD behaved on the baseline (the numbers a fix must beat).

SOURCE (one-time, needs the harvested corpus + the baseline run on disk):
  * tree structure + (X,y)         : data/workload/snapshots/operon_<prob>_.../pop_gen{g}.bin
  * which trees offend + baseline  : results/operon_baseline__20260617/data/<prob>__{fd,ad}.json
The COMMITTED outputs (fixtures + manifest) are the durable artifact; rerun this only to
regenerate them.

    python cusr/kernel/tests/fp32_honesty_regression/build_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/weish/hao/CuSR")
sys.path.insert(0, str(ROOT))
from cusr.benchmark.workload.pop_io import read_pop_bin, write_pop_bin   # noqa: E402

HERE = ROOT / "cusr/kernel/tests/fp32_honesty_regression"
BASE = ROOT / "results/operon_baseline__20260617/data"
SNAP = ROOT / "data/workload/snapshots"
GENS = [0, 4, 16, 64, 100]
SERIOUS = 2.0    # >2x worse in fp64 while status==0 = a serious offender (not fp32 noise)

PROBLEMS = ["feynman/I.12.1", "feynman/I.18.12", "feynman/I.27.6", "feynman/I.6.2",
            "feynman/I.12.2", "feynman/I.13.12", "feynman/II.3.24",
            "nguyen/1", "nguyen/2", "nguyen/3", "nguyen/4", "nguyen/5", "nguyen/6",
            "nguyen/7", "nguyen/8", "nguyen/9", "nguyen/10"]


def cell_bin(prob, g):
    safe = prob.replace("/", "_")
    return SNAP / f"operon_{safe}_pop4000_noise0_len32_seed0" / f"pop_gen{g:04d}.bin"


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    manifest = {"description": "fp32-honesty regression fixtures: pre-CO trees where the "
                "established kernel (FD/AD) reports status==0 yet is >2x worse in fp64 at "
                "c_final. A fixed kernel must NOT report status==0+fp64-worse on these.",
                "serious_mult": SERIOUS, "gens": GENS, "fixtures": {}}
    total = 0
    for prob in PROBLEMS:
        safe = prob.replace("/", "_")
        fd = json.loads((BASE / f"{safe}__fd.json").read_text())
        ad = json.loads((BASE / f"{safe}__ad.json").read_text())
        trees, recs, X, y = [], [], None, None
        for g in GENS:
            fg, ag = fd["gens"].get(str(g)), ad["gens"].get(str(g))
            if not fg or not ag or "error" in fg:
                continue
            pop = read_pop_bin(cell_bin(prob, g))
            if X is None:
                X, y = pop["xs"], pop["ym"][0]
            ls = np.array(fg["neutral_loss_start"])
            fl = np.array(fg["neutral_loss_final"]); al = np.array(ad["gens"][str(g)]["neutral_loss_final"])
            fs = np.array(fg["status"]); as_ = np.array(ag["status"]); K = np.array(fg["K"])
            opt = K > 0
            fd_ser = opt & (fs == 0) & ((~np.isfinite(fl)) | (fl > ls * SERIOUS))
            ad_ser = opt & (as_ == 0) & ((~np.isfinite(al)) | (al > ls * SERIOUS))
            for m in np.where(fd_ser | ad_ser)[0]:
                meta = pop["metas"][m]
                n_off, n_nodes, c_off, Km = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])
                sl = slice(n_off, n_off + n_nodes)
                trees.append((pop["nt"][sl].copy(), pop["nv"][sl].copy(),
                              pop["ci"][sl].copy(), pop["c_init"][c_off:c_off + Km].copy()))
                recs.append(dict(
                    problem=prob, gen=g, orig_index=int(m), K=Km,
                    loss_start_fp64=float(ls[m]),
                    fd=dict(status=int(fs[m]), final_fp64=float(fl[m]),
                            fp32=float(fg["kernel_fp32_loss"][m]), serious=bool(fd_ser[m])),
                    ad=dict(status=int(as_[m]), final_fp64=float(al[m]),
                            fp32=float(ag["kernel_fp32_loss"][m]), serious=bool(ad_ser[m]))))
        if not trees:
            continue
        out = HERE / f"fixture__{safe}.bin"
        stats = write_pop_bin(out, trees, X, y)
        for i, rc in enumerate(recs):
            rc["fixture_index"] = i
        manifest["fixtures"][prob] = dict(file=out.name, M=stats["M_prob"], trees=recs)
        total += stats["M_prob"]
        print(f"  {prob:16}: {stats['M_prob']:3} trees -> {out.name} ({out.stat().st_size//1024} KB)")
    manifest["total_trees"] = total
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"total {total} offender trees across {len(manifest['fixtures'])} problems; manifest.json written")


if __name__ == "__main__":
    main()
