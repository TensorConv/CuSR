#!/usr/bin/env python3
"""Convergence-honesty regression test for the batched CUDA LM solver.

Invariant under test (the "fake convergence" bug):
    A tree the solver reports as CONVERGED (status == 0) must NOT have ended at a
    loss WORSE than where it started. The Levenberg-Marquardt accept rule only
    ever moves to a state with non-increasing loss; therefore for every converged
    tree we must have

        loss_final <= loss_init * (1 + REL_TOL) + ABS_TOL

    and loss_final must be finite.

The buggy convergence branch commits the trial step (and overwrites the stored
loss with the trial loss) whenever the *step size* is tiny, WITHOUT checking that
the trial loss did not increase. On ill-conditioned / high-K trees the very first
regularized step can be both tiny and uphill, so the solver "converges" to a
worse objective than its starting point — exactly what this invariant catches.

This check is scipy-free (numpy only). It reads three sidecar blobs written by
the kernel into <out_dir>:
    status.bin     int32   , M entries  (0 == CONVERGED)
    loss_init.bin  float32 , M entries  (per-tree loss BEFORE the LM loop)
    loss_final.bin float32 , M entries  (per-tree loss AFTER the LM loop)

Usage:
    python test_convergence_honesty.py <out_dir>      # exit 1 if violations > 0
    from test_convergence_honesty import check
    n = check(out_dir)                                 # -> violation count
"""
import os
import sys
import numpy as np

# status code for a tree the solver claims it converged (see batch_lm_*.cu:
#   #define STATUS_CONVERGED 0)
STATUS_CONVERGED = 0

# loss_final may legitimately differ from loss_init by fp32 round-off even when
# the optimizer genuinely held or lowered the objective. These tolerances absorb
# that noise without masking a real uphill commit (a fake-convergence violation
# overshoots loss_init by far more than fp32 epsilon).
REL_TOL = 1e-4
ABS_TOL = 1e-12


def _load_i32(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.int32)


def _load_f32(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float32)


def check(out_dir, verbose=True, max_samples=8):
    """Return the number of converged trees that violate the honesty invariant.

    A violation is a tree with status == CONVERGED whose final loss is either
    non-finite, or exceeds loss_init * (1 + REL_TOL) + ABS_TOL.
    """
    status = _load_i32(os.path.join(out_dir, "status.bin"))
    loss_init = _load_f32(os.path.join(out_dir, "loss_init.bin")).astype(np.float64)
    loss_final = _load_f32(os.path.join(out_dir, "loss_final.bin")).astype(np.float64)

    n = status.shape[0]
    if loss_init.shape[0] != n or loss_final.shape[0] != n:
        raise ValueError(
            "length mismatch: status=%d loss_init=%d loss_final=%d in %s"
            % (n, loss_init.shape[0], loss_final.shape[0], out_dir)
        )

    conv = status == STATUS_CONVERGED
    n_conv = int(conv.sum())

    bound = loss_init * (1.0 + REL_TOL) + ABS_TOL
    worse = conv & (loss_final > bound)
    nonfinite = conv & ~np.isfinite(loss_final)
    viol = worse | nonfinite
    viol_idx = np.nonzero(viol)[0]
    n_viol = int(viol_idx.shape[0])

    if verbose:
        print(
            "[honesty] %s: M=%d converged=%d violations=%d"
            % (out_dir, n, n_conv, n_viol)
        )
        if n_viol:
            print(
                "[honesty]   (a CONVERGED tree ended worse than it started — "
                "fake convergence)"
            )
            for i in viol_idx[:max_samples]:
                tag = "nonfinite" if not np.isfinite(loss_final[i]) else "uphill"
                print(
                    "[honesty]   idx=%d loss_init=%.6g loss_final=%.6g "
                    "ratio=%.4g [%s]"
                    % (
                        int(i),
                        loss_init[i],
                        loss_final[i],
                        (loss_final[i] / loss_init[i]) if loss_init[i] > 0 else float("inf"),
                        tag,
                    )
                )
            if n_viol > max_samples:
                print("[honesty]   ... and %d more" % (n_viol - max_samples))

    return n_viol


def main(argv):
    if len(argv) != 2:
        print("usage: %s <out_dir>" % argv[0], file=sys.stderr)
        return 2
    out_dir = argv[1]
    n_viol = check(out_dir)
    if n_viol > 0:
        print("FAIL: %d converged tree(s) violate the honesty invariant" % n_viol)
        return 1
    print("PASS: all converged trees ended at <= their initial loss")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
