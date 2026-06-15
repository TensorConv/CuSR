#!/usr/bin/env bash
# Example SLURM submission script for this repo.
#
# Why SLURM on a single machine?
#   - Queue jobs so they don't fight over GPUs / OOM each other
#   - Per-job CUDA_VISIBLE_DEVICES isolation via --gres=gpu:N
#   - Built-in logging (one file per job, named by %x-%j)
#   - Painless scale-out later if more nodes get added
#
# Usage:
#   sbatch scripts/slurm/example.sbatch <python-script> [args...]
#
# Quick queue inspection:
#   squeue -u $USER
#   scancel <jobid>
#   sinfo

#SBATCH --job-name=sr-job
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                        # request N GPUs; max 8 on this node
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Load CUDA toolkit + project env vars (no-op if already sourced)
source scripts/env.sh

echo "--- job ${SLURM_JOB_ID} ---"
echo "host:      $(hostname)"
echo "gpus:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "cwd:       $(pwd)"
echo "start:     $(date -Iseconds)"
echo "script:    $*"
echo "----------------------------"

# uv run picks up the project's .venv automatically.
uv run python "$@"

echo "end:       $(date -Iseconds)"
