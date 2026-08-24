#!/bin/bash
#SBATCH --job-name=sw-figures
#SBATCH --output=slurm/logs/figures_%j.out
#SBATCH --error=slurm/logs/figures_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#
# Regenerate all analysis/paper figures (pure matplotlib; on a GPU node per the
# no-login-node-compute policy). Reads the decomposition JSONs from scratch.
#
#   sbatch slurm/figures.sbatch

source slurm/env.sh
mkdir -p slurm/logs
WS="${SW_WS:-/scratch4/workspace/knaskar_umass_edu-shadowweave}"

srun python scripts/plot_architecture.py --out figures/
srun python scripts/plot_decomp.py --dir "$WS/val_decomp/full" \
     --randgeom "$WS/val_decomp/randgeom.json" --out figures/

echo "figures done -> figures/{architecture,decomp_curves,reliability,shadow_gain_ci}.{png,svg?}"
