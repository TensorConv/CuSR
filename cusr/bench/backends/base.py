from __future__ import annotations

from typing import Protocol, runtime_checkable

from cusr.bench.skeleton import FitRecord, FitRequest


@runtime_checkable
class NLSBackend(Protocol):
    name: str

    def fit(self, request: FitRequest, seed: int, **kwargs) -> FitRecord: ...
