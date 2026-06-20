#!/bin/bash
# Parallel Study B full run, sharded by (SEED x problem-GROUP) across all 8 GPUs.
#
# Why (seed x group) and not (problem only): the wall-time bottleneck is the
# cpu_every (scipy, CPU-bound) cells that hit the 150s SIGKILL timeout. Sharding
# by problem alone puts all 5 seeds of a timeout-problem on ONE shard => 5x150s
# serial. Sharding by seed too spreads them => at most 1x150s per shard. With 256
# CPU cores and 8 A100s the 20 concurrent shards are nowhere near a resource wall.
# Each shard writes its own jsonl; aggregate.py merges study_b_sh*.jsonl.
set -u
cd /home/weish/hao/CuSR
source scripts/env.sh 2>/dev/null
OUTD=experiments/e4_study_b/out
NGPU=${NGPU:-8}          # GPUs 0..NGPU-1
NGROUP=${NGROUP:-4}      # problem groups; shards = SEEDS * NGROUP
SEEDS=${SEEDS:-5}

echo "=== GPU idle check (0..$((NGPU-1))) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n "1,${NGPU}p"

# clean prior shard outputs (the pre-rerun GOOD state is in _pre_rerun_20260620/)
rm -f ${OUTD}/study_b_sh*.jsonl ${OUTD}/study_b_gpu*.jsonl ${OUTD}/sh*_log.txt

# split the 53 ids (admits + controls) into NGROUP round-robin groups
.venv/bin/python - "$NGROUP" > /tmp/e4_groups.txt <<'PY'
import sys
from experiments.e4_study_b import corpus
n = int(sys.argv[1])
adm, ctl = corpus.load()
ids = [p.id for p in adm] + [p.id for p in ctl]
groups = [[] for _ in range(n)]
for i, pid in enumerate(ids):
    groups[i % n].append(pid)
for g in groups:
    print(" ".join(g))
PY
# NB: array MUST NOT be named GROUPS — that is a bash-special readonly-ish var
# (the caller's group IDs); += silently no-ops on it. Use PG[].
PG=()
while IFS= read -r line; do [ -n "$line" ] && PG+=("$line"); done < /tmp/e4_groups.txt
if [ "${#PG[@]}" -ne "$NGROUP" ]; then
  echo "FATAL: expected $NGROUP groups, got ${#PG[@]} — aborting (not launching broken shards)"; exit 1
fi

echo "=== launching $((SEEDS * NGROUP)) shards (SEEDS=${SEEDS} x NGROUP=${NGROUP}) over ${NGPU} GPUs ==="
pids=(); idx=0
for seed in $(seq 0 $((SEEDS-1))); do
  for g in $(seq 0 $((NGROUP-1))); do
    gpu=$((idx % NGPU))
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m experiments.e4_study_b.run_study_b \
      --controls --problems ${PG[$g]} --seed-list $seed --device 0 \
      --out study_b_sh${idx} > ${OUTD}/sh${idx}_log.txt 2>&1 &
    pids+=($!)
    idx=$((idx + 1))
  done
done
echo "  launched ${#pids[@]} shards: pids ${pids[*]}"

echo "=== waiting for ${#pids[@]} shards ==="
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "=== shards done (non-zero exits: $fail) ==="
wc -l ${OUTD}/study_b_sh*.jsonl 2>/dev/null | tail -1

echo "=== aggregating ==="
.venv/bin/python -m experiments.e4_study_b.aggregate 'study_b_sh*.jsonl'
echo "=== ALL DONE ==="
