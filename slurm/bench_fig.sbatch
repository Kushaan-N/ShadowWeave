#!/bin/bash
#SBATCH --job-name=sw-benchfig
#SBATCH --output=slurm/logs/benchfig_%j.out
#SBATCH --error=slurm/logs/benchfig_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
# End-to-end latency (incl. monocular depth) + architecture figure, on a GPU node.
# Depth-Anything-V2 loads from the local HF cache (offline), so no internet is needed.
#
#   sbatch slurm/bench_fig.sbatch

source slurm/env.sh
mkdir -p slurm/logs

# Point transformers at the pre-downloaded cache and forbid network fetches.
export HF_HOME="${SW_HF_HOME:-/scratch4/workspace/knaskar_umass_edu-shadowweave/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "== per-stage latency incl. monocular depth (real-time budget) =="
srun python scripts/benchmark.py --stages --with-depth --cam-hw 480 640

echo
echo "== architecture figure =="
srun python scripts/plot_architecture.py --out figures/

echo "benchfig done -> figures/architecture.{png,svg}"
