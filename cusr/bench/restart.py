"""Restart strategies for the bench Runner.

A strategy is a callable ``(request, seed) -> Iterable[np.ndarray]`` that
yields N init-constant vectors for a given FitRequest. The Runner then
invokes the backend once per yielded init, writing one FitRecord per trial
with a distinct ``restart_idx``.

Three strategies cover the common experiment arms:

- **NoRestart** — 1 trial from ``request.init_constants`` as-is. Default.
  Use for warm-start arms where the incoming init is meaningful (e.g.
  EvoGP-fit constants handed to LM).
- **RandomPerturb(n, scale)** — n trials around ``request.init_constants``.
  Trial 0 is the base init verbatim; trials 1..n-1 are ``base + N(0,
  scale)`` per constant. Use for warm-start exploration.
- **BestOfN(n, scale, distribution)** — n fresh random inits, **ignoring**
  ``request.init_constants``. Use for cold-start arms where the init is
  uninformative.

Post-hoc, reports aggregate trials by ``skeleton_id`` and pick the best
``final_loss`` across ``restart_idx`` for best-of-N style metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cusr.bench.skeleton import FitRequest


@dataclass
class NoRestart:
    @property
    def name(self) -> str:
        return "no_restart"

    def __call__(self, request: FitRequest, seed: int) -> Iterable[np.ndarray]:
        yield np.asarray(request.init_constants, dtype=float)


@dataclass
class RandomPerturb:
    """Warm-start exploration: n inits around ``request.init_constants``.

    Trial 0 is always the base init verbatim — so downstream reports can
    diff "original warm-start vs perturbation" cleanly without running an
    extra NoRestart pass.
    """
    n: int
    scale: float
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"RandomPerturb.n must be ≥ 1, got {self.n}")
        if self.scale < 0:
            raise ValueError(f"RandomPerturb.scale must be ≥ 0, got {self.scale}")

    @property
    def name(self) -> str:
        return f"random_perturb_n{self.n}_s{self.scale:g}"

    def __call__(self, request: FitRequest, seed: int) -> Iterable[np.ndarray]:
        rng = np.random.default_rng(np.asarray([seed, self.seed_offset, 0x9E3779B9], dtype=np.uint64))
        base = np.asarray(request.init_constants, dtype=float)
        for i in range(self.n):
            if i == 0:
                yield base.copy()
            else:
                yield base + rng.normal(0.0, self.scale, size=base.shape)


@dataclass
class BestOfN:
    """Cold-start: n fresh random inits, ignoring ``request.init_constants``.

    Distribution is ``'gaussian'`` (N(0, scale)) or ``'uniform'`` (U(-scale,
    +scale)). Trials are independent draws — the "best of" is a post-hoc
    aggregation in the reporting layer, not a filter inside this strategy.

    All N trials are emitted so FitRecords retain full trial history (for
    failure mode analysis). Downstream reports pick min-final_loss per
    (skeleton_id, seed) group for the best-of-N metric.
    """
    n: int
    scale: float = 1.0
    distribution: str = "gaussian"
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"BestOfN.n must be ≥ 1, got {self.n}")
        if self.scale <= 0:
            raise ValueError(f"BestOfN.scale must be > 0, got {self.scale}")
        if self.distribution not in ("gaussian", "uniform"):
            raise ValueError(f"BestOfN.distribution must be gaussian|uniform, got {self.distribution!r}")

    @property
    def name(self) -> str:
        return f"best_of_{self.n}_{self.distribution}_s{self.scale:g}"

    def __call__(self, request: FitRequest, seed: int) -> Iterable[np.ndarray]:
        rng = np.random.default_rng(np.asarray([seed, self.seed_offset, 0xBF58476D], dtype=np.uint64))
        k = request.skeleton.n_constants
        for _ in range(self.n):
            if self.distribution == "gaussian":
                yield rng.normal(0.0, self.scale, size=k)
            else:  # uniform
                yield rng.uniform(-self.scale, self.scale, size=k)
