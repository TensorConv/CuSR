# SLURM helpers

This machine has slurm-wlm 21.08.5 preconfigured with one partition:

```
PARTITION  NODES  GRES
gpu*       1      gpu:a100:8   (NVIDIA A100-SXM4-80GB)
```

## When to use SLURM vs. just running directly

| Scenario                          | Use                    |
|-----------------------------------|------------------------|
| Quick debug, REPL, single GPU     | `uv run python ...`    |
| Long runs, background             | `sbatch ...`           |
| Multiple runs sweeping seeds/hp   | `sbatch` (queue them)  |
| Need isolated `CUDA_VISIBLE_DEVICES` | `sbatch` w/ `--gres=gpu:N` |

## Submit

```bash
sbatch scripts/slurm/example.sbatch experiments/001_evogp_repro/run.py
squeue -u $USER
tail -f logs/sr-job-<jobid>.out
```

Adjust `--gres=gpu:N`, `--time`, `--cpus-per-task` per job. Copy the
template to a job-specific file when a run's parameters stabilize.
