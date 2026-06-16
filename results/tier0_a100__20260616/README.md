# tier0_a100__20260616

First entry in the `results/` archive. A100 kernel-baseline re-measurement (Tier 0).

- **Read `report.html`** (self-contained) for the full structured writeup — question, params, findings, tables, plot, caveats, reproduce.
- `report.json` — the structured source (edit this, re-run `make_report.py` to regenerate the HTML).
- `data/` — `tp2_*.jsonl` (raw per-(variant,M) sweep rows) + `tier0_throughput.csv` (aggregated).
- `plots/` — `tier0_throughput.png`.
- `logs/` — per-gen, parity, scale run logs + the session manifest.

Headline: fp32 throughput scales to **fusedfd 50k trees/s @256k (still climbing)**; the
baseline host-FD variant saturates ~1.3-1.5k (H<->D serial bottleneck), so the **variant gap
widens at scale on A100** — opposite of the laptop-PCIe hypothesis. Parity 3/3 (94.0/92.4/99.8),
scale tier1 11/11 x3. CO stays the GP-loop bottleneck (per-gen ~5ms vs CO hundreds of ms).

Regenerate the report after editing report.json:
```
uv run python scripts/make_report.py results/tier0_a100__20260616/report.json
```
