# GPU exclusivity workflow for paper-grade timing on this SHARED box.
# We do NOT use SLURM (the box has it but nobody routes through it, so it is
# blind to direct runs and cannot guarantee exclusivity). Instead: manual GPU
# pinning + a co-tenancy guard. Source this, then use the helpers below.
#
#   source scripts/gpu_guard.sh
#   gpu_precheck                  # print occupancy table (FREE vs whose)
#   pick_free_gpus 6              # echo up to N free GPU indices (space-sep)
#   assert_exclusive 2            # exit 1 + message if GPU 2 has any foreign compute proc
#   watch_exclusive "2 3 4 5"     # foreground watchdog: prints a line if an intruder appears
#
# "Foreign" = any compute process on the GPU NOT owned by the current $USER.
# (We pin one job per GPU, so between/around our runs the only PIDs should be
#  ours; a different user's PID = contamination -> that timing run is suspect.)

_me="$(id -un)"

# uuid<->index map (index order matches nvidia-smi -L)
_gpu_uuid() { nvidia-smi -i "$1" --query-gpu=gpu_uuid --format=csv,noheader 2>/dev/null; }

# echo the foreign compute PIDs on GPU $1 (empty = exclusive/idle for us)
foreign_pids() {
  local idx="$1" uuid; uuid="$(_gpu_uuid "$idx")"
  [ -z "$uuid" ] && return 0
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -F',' -v u="$uuid" '$1 ~ u {gsub(/ /,"",$2); print $2}' \
    | while read -r pid; do
        [ -z "$pid" ] && continue
        local owner; owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
        [ "$owner" != "$_me" ] && echo "$pid:$owner"
      done
}

gpu_precheck() {
  echo "=== GPU pre-check ($(date -Iseconds)) ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
    | while IFS=',' read -r idx util mem tot; do
        local f; f="$(foreign_pids "$(echo "$idx"|tr -d ' ')")"
        if [ -n "$f" ]; then echo "  GPU$idx util=$util mem=$mem  <-- IN USE by: $f"
        else echo "  GPU$idx util=$util mem=$mem  FREE-for-us"; fi
      done
}

# echo up to $1 GPU indices that have no foreign compute proc (and ~idle mem)
pick_free_gpus() {
  local want="${1:-6}" n=0 out=""
  for idx in $(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '); do
    [ -n "$(foreign_pids "$idx")" ] && continue
    out="$out $idx"; n=$((n+1)); [ "$n" -ge "$want" ] && break
  done
  echo "${out# }"
}

# hard gate before a timed run: nonzero exit if GPU $1 has a foreign proc
assert_exclusive() {
  local idx="$1" f; f="$(foreign_pids "$idx")"
  if [ -n "$f" ]; then
    echo "[guard] GPU$idx NOT exclusive — foreign procs: $f" >&2
    return 1
  fi
  return 0
}

# foreground watchdog: sample the given GPUs; print a line ONLY when an
# intruder appears (use with Monitor, or run in background and tail).
# usage: watch_exclusive "2 3 4 5" [interval_s]
watch_exclusive() {
  local gpus="$1" iv="${2:-5}"
  echo "[guard] watching GPUs [$gpus] every ${iv}s for foreign procs (Ctrl-C to stop)"
  while true; do
    for idx in $gpus; do
      local f; f="$(foreign_pids "$idx")"
      [ -n "$f" ] && echo "[guard] $(date +%H:%M:%S) INTRUDER on GPU$idx: $f"
    done
    sleep "$iv"
  done
}
