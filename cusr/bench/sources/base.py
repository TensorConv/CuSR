from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from cusr.bench.skeleton import FitRequest


@runtime_checkable
class SkeletonSource(Protocol):
    name: str

    def iter_requests(self) -> Iterable[FitRequest]: ...
