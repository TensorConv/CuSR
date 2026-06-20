#!/bin/bash
# Parallel Study B full run: 33 admits + 20 controls x 4 arms x 5 seeds, sharded
# round-robin across 6 idle GPUs (1..6; GPU0 has food_condor). Each shard writes
# its own jsonl; aggregate.py merges them into out/report.json at the end.
set -u
cd /home/weish/hao/CuSR
source scripts/env.sh 2>/dev/null
OUTD=experiments/e4_study_b/out
NGPU=6
FIRST=1
SEEDS=5

echo "=== GPU idle check (1..6) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n "$((FIRST+1)),$((FIRST+NGPU))p"

rm -f ${OUTD}/study_b_gpu*.jsonl ${OUTD}/study_b_gpu*_report.json ${OUTD}/gpu*_log.txt

# round-robin id groups across NGPU
.venv/bin/python - "$NGPU" > /tmp/e4_groups.txt <<'PY'
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

echo "=== launching ${NGPU} shards (seeds=${SEEDS}) ==="
pids=()
i=0
while IFS= read -r group; do
  gpu=$((FIRST + i))
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m experiments.e4_study_b.run_study_b \
    --controls --problems $group --seeds $SEEDS --device 0 --out study_b_gpu${gpu} \
    > ${OUTD}/gpu${gpu}_log.txt 2>&1 &
  pids+=($!)
  echo "  GPU${gpu}: pid $! ($(echo $group | wc -w) problems)"
  i=$((i + 1))
done < /tmp/e4_groups.txt

echo "=== waiting for ${#pids[@]} shards ==="
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "=== shards done (non-zero exits: $fail) ==="
wc -l ${OUTD}/study_b_gpu*.jsonl 2>/dev/null

echo "=== aggregating ==="
.venv/bin/python -m experiments.e4_study_b.aggregate 'study_b_gpu*.jsonl'
echo "=== ALL DONE ==="
