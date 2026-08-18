#!/bin/bash
# Ablation — retrain + eval the world model WITHOUT the visibility (shadow) input channel,
# to show the shadow signal is what drives the forecasting-into-unobserved-space gain.
# Everything (checkpoint AND results) stays on scratch — nothing written to /work.
# Compare results_noshadow shadow_iou/gain against the full run in ./results/.
# Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts
export SW_CKPT_DIR=$WS/checkpoints/world_model_noshadow   # ablation ckpt -> scratch
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu

GPU="${SW_TRAIN_GPU:-gpu:1}"
ABLATE="bev.input_channels=[occupancy,flow]"   # drop the visibility/shadow channel

# Train the ablated world model (same 100-epoch budget + early stopping as the full run).
WM=$(SW_OVERRIDES="$ABLATE" \
     sbatch --parsable --gres="$GPU" --time="${SW_TIME:-08:00:00}" slurm/train_worldmodel.sbatch)

# Eval it with the SAME channel override so the whole pipeline stays 2-channel-consistent.
# Results to scratch. Navigation numbers here are NOT comparable (policy was trained on the
# full 3-channel model) — only the world-model metrics (shadow_iou, gain) matter.
EV=$(SW_CKPT="$SW_CKPT_DIR/best.pt" SW_RESULTS_DIR=$WS/results_noshadow SW_EVAL_EPISODES=90 \
     SW_OVERRIDES="$ABLATE" \
     sbatch --parsable --gres="$GPU" --dependency=afterok:"$WM" slurm/eval.sbatch)

echo "ablation chain:  worldmodel(no-visibility)=$WM  ->  eval=$EV"
echo "results land in $WS/results_noshadow/eval_summary.json"
echo "compare:  full-model shadow_gain@5s = +0.320  (this run should be LOWER if shadow signal matters)"
echo "watch:  squeue -u $USER   |   tail -f slurm/logs/worldmodel_${WM}.out"
