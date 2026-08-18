#!/bin/bash
# Phase 1 — generate rollouts to scratch. Run from the login node.
# Untracked helper (not committed); safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts
export SW_CKPT_DIR=$WS/checkpoints/world_model
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu            # the sbatch scripts have no --partition line
export SBATCH_ACCOUNT=pi_sniekum_umass_edu   # this is a Sniekum-lab project; don't bill pi_andrewlan

echo "writing rollouts to: $SW_DATA_ROOT"
TRAIN=$(sbatch --parsable slurm/gen_data.sbatch)
VAL=$(SW_SPLIT=val SW_EPISODES=80 sbatch --parsable --array=0-3 slurm/gen_data.sbatch)
echo "submitted:  train array=$TRAIN   val array=$VAL"
echo
echo "watch:  squeue -u $USER"
echo "logs :  tail -f slurm/logs/gendata_*.out"
echo
echo "When BOTH arrays finish, run the data check (bash run_check_data.sh)"
echo "BEFORE launching training — degenerate data is the #1 failure here."
