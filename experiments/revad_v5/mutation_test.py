#!/usr/bin/env python3
"""Mutation test for reverse-AD v5 (the verify-workflow mutation agent died on stream-idle).

Proves tests/test_revad_host.cu actually has teeth: inject N behaviour-changing bugs into a
COPY of eval_tree_vjp_d, recompile the host test against each mutant, and confirm each one
turns the test RED. A mutant that still passes = a hole in the test. Canonical source untouched.
"""
import subprocess, shutil, os, sys

SRC = "/home/weish/hao/CuSR/cusr/kernel"
MUT = "/tmp/revad_mut"
ENV = "/home/weish/hao/CuSR/scripts/env.sh"

# (description, exact_old_substring, new_substring) — each MUST change behaviour on the host trees.
MUTATIONS = [
    ("flip sign of F_SIN derivative",
     "case F_SIN:  r = sinf(a);  d = cosf(a);",
     "case F_SIN:  r = sinf(a);  d = -cosf(a);"),
    ("swap F_DIV dl/dr partials",
     "case F_DIV: o = l / rv; dl = 1.0f / rv;  dr = -o / rv;",
     "case F_DIV: o = l / rv; dl = -o / rv;  dr = 1.0f / rv;"),
    ("zero F_MUL left partial (dl)",
     "case F_MUL: o = l * rv; dl = rv;         dr = l;",
     "case F_MUL: o = l * rv; dl = 0.0f;       dr = l;"),
    ("adjoint stack init 1.0 -> 2.0",
     "sa[asp++] = 1.0f;",
     "sa[asp++] = 2.0f;"),
    ("flip sign of F_EXP derivative",
     "case F_EXP:  r = expf(a);  d = r;",
     "case F_EXP:  r = expf(a);  d = -r;"),
    ("swap bfunc rv-adj / l-adj push order",
     "            sa[asp++] = adj * d2[i];        // rv-adjoint (深)\n            sa[asp++] = adj * d1[i];        // l-adjoint  (顶); 镜像 forward 弹序 (l 在顶)",
     "            sa[asp++] = adj * d1[i];        // rv-adjoint (深)\n            sa[asp++] = adj * d2[i];        // l-adjoint  (顶); 镜像 forward 弹序 (l 在顶)"),
    ("flip F_TANH derivative (1-r^2 -> 1+r^2)",
     "case F_TANH: r = tanhf(a); d = 1.0f - r * r;",
     "case F_TANH: r = tanhf(a); d = 1.0f + r * r;"),
]

def setup():
    if os.path.exists(MUT):
        shutil.rmtree(MUT)
    os.makedirs(MUT + "/tests")
    for f in ("revad_interp.cuh", "ad_interp.cuh", "pop_format.h"):
        shutil.copy(f"{SRC}/{f}", f"{MUT}/{f}")
    shutil.copy(f"{SRC}/tests/test_revad_host.cu", f"{MUT}/tests/test_revad_host.cu")

def build_and_run(tag):
    cmd = (f"source {ENV} 2>/dev/null && cd {MUT} && "
           f"nvcc -O2 -std=c++17 -o tests/t_{tag} tests/test_revad_host.cu 2>/tmp/revad_mut/build_{tag}.err "
           f"&& ./tests/t_{tag}")
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    out = p.stdout
    if "RESULT: PASS" in out and p.returncode == 0:
        return "PASS"
    if "RESULT: FAIL" in out and p.returncode != 0:
        return "FAIL"
    # compile error or crash -> also counts as "not PASS" (mutation caught), but flag it
    return f"NONPASS(rc={p.returncode})"

def main():
    setup()
    base_src = open(f"{SRC}/revad_interp.cuh").read()

    print("=== baseline (unmutated) ===")
    base = build_and_run("base")
    print(f"  baseline: {base}")
    if base != "PASS":
        print("BASELINE NOT PASS — harness broken, aborting"); sys.exit(2)

    print("\n=== mutants (each must turn the test RED) ===")
    caught = 0
    rows = []
    for i, (desc, old, new) in enumerate(MUTATIONS):
        if old not in base_src:
            rows.append((desc, "NO-MATCH", False))
            print(f"  [{i}] {desc:42s} -> OLD STRING NOT FOUND (mutation N/A)")
            continue
        mutated = base_src.replace(old, new, 1)
        assert mutated != base_src
        open(f"{MUT}/revad_interp.cuh", "w").write(mutated)
        res = build_and_run(f"m{i}")
        is_caught = (res != "PASS")
        caught += int(is_caught)
        rows.append((desc, res, is_caught))
        print(f"  [{i}] {desc:42s} -> {res:14s} {'CAUGHT' if is_caught else 'HOLE!!!'}")

    applied = sum(1 for _, r, _ in rows if r != "NO-MATCH")
    print(f"\n=== SUMMARY: {caught}/{applied} mutations caught ===")
    if caught == applied and applied >= 6:
        print("RESULT: PASS (test has teeth; no holes)")
        sys.exit(0)
    print("RESULT: FAIL (a mutation slipped through OR too few applied)")
    sys.exit(1)

if __name__ == "__main__":
    main()
