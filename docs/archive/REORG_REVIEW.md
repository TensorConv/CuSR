# Reorg review (Phase 0 + Phase 1)

reviewed 2026-06-15 — 13 read-only agents, one angle each. record of what's
verified, what got fixed, what's left, and the push blocker.

## verdict

Reorg is functionally correct. All four test gates green, numbers identical to
pre-reorg. Safe to keep building on. The remaining work before a public push is
(a) a few polish items below and (b) one decision about git history (see end).

## verified green

* imports — all 40 `cusr.*` modules import from a non-package cwd via the
  editable install. 0 reorg-broken. (3 submodules hard-import upstream `evogp`;
  works because it's installed locally — see "evogp" below.)
* tests — `tests/bench` 99 pass / 1 skip; kernel parity gate 3/3 PASS
  (loss_down 94.0%, within_1.05 92.4%, within_10 99.8% — same as pre-reorg);
  e1 harness 16/0; e2 89 pass.
* paths — every `parents[N]` lands on the right dir for the file's new depth;
  formulas.yaml / pop.bin / presets.json / snapshots all resolve.
* binary — every `batch_lm` reference resolves via `cusr.kernel.__file__`, none
  point at the old experiments/008 path.
* git history — 0 gitlinks, no .gitmodules, parent tracks 0 files in the 008
  backup. Phase 0 = 211 pure renames. Moved code keeps ancestry.
* data — pop.bin loads (1000 trees, matches meta); formulas.yaml 98 entries =
  98 parquet; all 6 preset sha1 pins MATCH their snapshots.
* 008 backup vs cusr/kernel — all .cu/.h/.c byte-identical; the only diffs are
  the expected import-path fixes (dual try/except) and the pop.bin path
  repoint. No source dropped. **The 008 backup is a faithful copy, safe to
  delete.**
* no orphans, no dangling cusr.* imports, no duplicate modules, __init__ clean.

## fixed this pass

* untracked `experiments/e1_operator_bench/_out/` (16 files, all carried the
  local hostname in name + body) and added `experiments/*/_out/`,
  `experiments/*/_cache/`, `legacy/*/_out/` to .gitignore.
* pyproject `name` sr → cusr, version 0.0.0 → 0.1.0, real description.
  reinstalls clean as `cusr==0.1.0`.
* doc broken paths: benchmark_README (pop.bin), walkthrough/README (cd path),
  demonstrator_README (formulas.yaml + bench → cusr/bench).
* top-level README rewritten for the new layout (was the old sandbox doc).

## left for later (not blockers)

* deps trim for a clean OSS install: `pytest` and `pypdf` shouldn't be core
  (pypdf also declared twice); `graphviz`/`matplotlib` are legacy-only → extras;
  `pyarrow` may be implicit-only. NOT touched — they brush against collaborator
  churn, decide before trimming. NOTE: `pysr` IS used by
  cusr/benchmark/backends.py (Julia LM backend) — keep it, the old
  "pysr = collaborator churn" note was about exp004, not cusr.
* evogp — `cusr/kernel/dump_evogp.py`, `cusr/bench/sources/evogp.py`,
  `cusr/demonstrator/pipeline.py` hard-import upstream `evogp` at module level.
  It's not on PyPI (cloned via setup_upstream). So a clean `pip install cusr`
  can't import those three until evogp is bootstrapped. Either declare it as a
  VCS dep or make the imports lazy. Fine for A100 (setup bootstraps it).
* Makefile builds only `batch_lm` (+ inspect). The variants the e1 probes use
  (batch_lm_devjac/fusedfd, and the _A/_B/_F3/_F9/_nofloor ablations — some have
  no .cu source, only compiled blobs) won't regenerate on a clean A100 `make`.
  Reconstruct sources or add targets before relying on them there.
* docs/benchmark_README "跑法" block still names runner.py/workload as if in the
  e1 dir; they live in cusr/benchmark now. Path fixed, commands not rewritten.
* CLAUDE.md names the local host + cluster — fine internally, flag if pushing.

## push blocker — git history exposure

Decision A (push sr's existing repo to the CuSR remote) carries the FULL commit
history, not just the clean tip. `git rm --cached legacy` only cleans the tip.
History audit found:

* legacy/ — REACHABLE at old SHAs under the old `experiments/00X_` paths
  (Phase 0 was a pure rename). `git show <old-sha>:experiments/007_.../...`
  returns the file.
* collaborator exp004 — committed once (`75d45d2`, 16 files: their notebooks,
  talk slides, figs). Untracked at the tip but in history → would publish.
  This conflicts with the don't-publish-collaborator-work constraint.
* secrets — none found. large blobs — none > 1 MB. So it's a content/authorship
  problem, not security/bloat.

Tip deletion is not enough. Options: (1) push a single squashed/orphan commit =
clean tip only, no history — simplest, drops legacy + 004 cleanly; (2) git
filter-repo to strip 004 (and maybe legacy) from history — keeps cusr history,
more work; (3) accept full history — publishes 004, not acceptable.
**Pending user decision.**
