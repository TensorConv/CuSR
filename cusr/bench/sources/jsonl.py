from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from cusr.bench.dataset import Dataset
from cusr.bench.skeleton import FitRequest, Skeleton


# Allowlisted Skeleton vocabulary. Anything else → parse_expr raises.
# Prevents RCE via malicious `expr` strings (sp.sympify would exec them).
_ALLOWED_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
    "Pow": sp.Pow, "Add": sp.Add, "Mul": sp.Mul, "Rational": sp.Rational,
    "Integer": sp.Integer, "Float": sp.Float, "Symbol": sp.Symbol,
}
_NAME_RE = re.compile(r"^[xc]\d+$")


def _safe_parse_expr(s: str, variables: tuple, constants: tuple) -> sp.Expr:
    locals_ = {sym.name: sym for sym in (*variables, *constants)}
    for name in locals_:
        if not _NAME_RE.match(name):
            raise ValueError(f"disallowed symbol name {name!r} in JSONL record")
    return parse_expr(s, local_dict={**_ALLOWED_FUNCS, **locals_}, global_dict={}, evaluate=True)


@dataclass
class JSONLSource:
    path: Path
    registry_path: Optional[Path] = None
    name: str = "jsonl"

    def iter_requests(self) -> Iterable[FitRequest]:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                variables = tuple(sp.Symbol(n, real=True) for n in obj["variables"])
                constants = tuple(sp.Symbol(n, real=True) for n in obj["constants"])
                expr = _safe_parse_expr(obj["expr"], variables, constants)
                skel = Skeleton(expr=expr, variables=variables, constants=constants)
                init = np.asarray(obj["init_constants"], dtype=float)
                ds_meta = obj["dataset"]
                if "registry_path" in ds_meta:
                    reg = Path(ds_meta["registry_path"])
                elif self.registry_path is not None:
                    reg = Path(self.registry_path)
                else:
                    raise ValueError("JSONLSource needs registry_path (ctor or per-line)")
                dataset = Dataset.from_registry(ds_meta["id"], reg)
                yield FitRequest(
                    skeleton=skel,
                    init_constants=init,
                    dataset=dataset,
                    source=obj.get("source", {}),
                )


def request_to_json(request: FitRequest, registry_path: Path) -> str:
    obj = {
        "expr": str(request.skeleton.expr),
        "variables": [s.name for s in request.skeleton.variables],
        "constants": [s.name for s in request.skeleton.constants],
        "init_constants": [float(v) for v in np.asarray(request.init_constants)],
        "dataset": {
            "id": request.dataset.id,
            "registry_path": str(registry_path),
        },
        "source": request.source,
    }
    return json.dumps(obj)
