#!/bin/bash
# Phase 2 — train the world model, then PPO, then eval, as a SLURM dependency chain.
# Run ONLY after run_check_data.sh prints "DATA OK". Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts
export SW_CKPT_DIR=$WS/checkpoints/world_model
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu   # this is a Sniekum-lab project; don't bill pi_andrewlan

# Any GPU by default (cu126 runs on all of them). For the full 100-epoch run you may
# want to pin a faster card, e.g.  SW_TRAIN_GPU=gpu:l40s:1 bash run_train_eval.sh
GPU="${SW_TRAIN_GPU:-gpu:1}"

# Cap at 8h (this cluster's max). 100 epochs on an a100 is ~5-6h; if it needs more, the
# script's SIGUSR1 handler requeues and resumes from last.pt.
WM=$(sbatch --parsable --gres="$GPU" --time="${SW_TIME:-08:00:00}" slurm/train_worldmodel.sbatch)
RL=$(sbatch --parsable --gres="$GPU" --dependency=afterok:"$WM" slurm/train_rl.sbatch)
EV=$(SW_CKPT="$SW_CKPT_DIR/best.pt" SW_EVAL_EPISODES=90 \
     sbatch --parsable --dependency=afterok:"$RL" slurm/eval.sbatch)

echo "submitted chain:  worldmodel=$WM  ->  rl=$RL  ->  eval=$EV"
echo "watch:  squeue -u $USER"
echo "logs :  tail -f slurm/logs/worldmodel_*.out"
echo "results land in ./results/eval_summary.json when eval ($EV) finishes."
