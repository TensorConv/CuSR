import numpy as np, collections
ad = np.fromfile('/tmp/cmp/ad_out/status.bin', dtype=np.int32)
fd = np.fromfile('/tmp/cmp/fd_out/status.bin', dtype=np.int32)
nm = {0: 'converged', 1: 'maxiter', 2: 'fail_nan', 3: 'k0_skip', 4: 'fail_cholesky'}
fd_only = [int(a) for a, f in zip(ad, fd) if f == 0 and a != 0]   # FD solves, AD doesn't
ad_only = [int(f) for a, f in zip(ad, fd) if a == 0 and f != 0]   # AD solves, FD doesn't
print(f"FD-only-converged (FD solves, AD does NOT): {len(fd_only)}")
for k, v in sorted(collections.Counter(fd_only).items()):
    print(f"    AD status={k} {nm.get(k,k):14s}: {v}")
print(f"AD-only-converged (AD solves, FD does NOT): {len(ad_only)}")
for k, v in sorted(collections.Counter(ad_only).items()):
    print(f"    FD status={k} {nm.get(k,k):14s}: {v}")
print(f"\noverall fail_nan: AD={int((ad==2).sum())}  FD={int((fd==2).sum())}")
print(f"overall cholesky: AD={int((ad==4).sum())}  FD={int((fd==4).sum())}")
