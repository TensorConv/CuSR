from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import yaml

from cusr.bench.backends.base import NLSBackend
from cusr.bench.restart import NoRestart
from cusr.bench.skeleton import FitRequest
from cusr.bench.sources.base import SkeletonSource
from cusr.bench.sources.jsonl import request_to_json


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class Runner:
    source: SkeletonSource
    backends: Sequence[NLSBackend]
    seeds: Sequence[int]
    output_dir: Path
    restart_strategy: object = field(default_factory=NoRestart)
    name: str = "run"
    registry_path: Optional[Path] = None
    config: dict = field(default_factory=dict)

    def run(self) -> Path:
        ts = _utc_ts()
        run_dir = Path(self.output_dir) / self.name / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            "source": getattr(self.source, "name", type(self.source).__name__),
            "backends": [b.name for b in self.backends],
            "seeds": list(self.seeds),
            "restart_strategy": getattr(self.restart_strategy, "name", type(self.restart_strategy).__name__),
            "registry_path": str(self.registry_path) if self.registry_path else None,
            **self.config,
        }
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)

        req_path = run_dir / "requests.jsonl"
        rec_path = run_dir / "records.jsonl"

        strategy_name = getattr(
            self.restart_strategy, "name", type(self.restart_strategy).__name__
        )
        with open(req_path, "a") as req_f, open(rec_path, "a") as rec_f:
            for request in self.source.iter_requests():
                # Dataset._cache handles per-instance caching; same Dataset
                # across requests hits memory. No extra Runner-level map needed.
                req_f.write(request_to_json(request, self.registry_path or Path("")) + "\n")
                req_f.flush()
                os.fsync(req_f.fileno())
                for backend in self.backends:
                    for seed in self.seeds:
                        for restart_idx, init in enumerate(
                            self.restart_strategy(request, seed)
                        ):
                            sub_req = replace(request, init_constants=init)
                            record = backend.fit(sub_req, seed)
                            record.restart_idx = restart_idx
                            record.restart_strategy = strategy_name
                            rec_f.write(json.dumps(record.to_dict()) + "\n")
                            rec_f.flush()
                            os.fsync(rec_f.fileno())
        return run_dir
