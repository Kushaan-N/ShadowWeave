#!/bin/bash
# Ablation — retrain + eval the world model as a conditional DDPM (diffusion) instead of the
# BCE U-Net, to answer the standard world-model reviewer attack: "the U-Net predictions look
# like mean-seeking grey mush — would a proper generative model forecast the multimodal
# occupancy in shadow better?" Architecture is fully config-selectable (build_world_model
# dispatches on world_model.architecture; train.py uses the architecture-agnostic
# model.loss() = eps-MSE for diffusion; eval uses model.predict()). Everything on scratch.
#
# IMPORTANT — how to READ this result (do not run a naive IOU horse-race):
#   * The diffusion model is NOT capacity-matched to the U-Net (CondUNet adds timestep
#     embeddings + extra conditioning channels). Report param counts; don't claim parity.
#   * eval scores a thresholded MEAN over several DDIM samples, which re-introduces the very
#     mean-seeking diffusion is meant to beat — so shadow_iou may only TIE the U-Net.
#     Frame the diffusion win on shadow_diversity_ratio (>1 => uncertainty concentrates in
#     unobserved space) and calibration_error, with IOU shown as roughly on-par.
#   * Per-epoch DDIM sampling for checkpoint selection + multi-step sampling at eval make
#     BOTH training and eval slower than the U-Net — budget more wall time (this is the
#     expensive ablation; run it first as the long pole, or defer if the queue is tight).
# Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts
export SW_CKPT_DIR=$WS/checkpoints/world_model_diffusion   # ablation ckpt -> scratch
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu

GPU="${SW_TRAIN_GPU:-gpu:1}"
ARCH="world_model.architecture=diffusion"

WM=$(SW_OVERRIDES="$ARCH" \
     sbatch --parsable --gres="$GPU" --time="${SW_TIME:-08:00:00}" slurm/train_worldmodel.sbatch)

# Architecture is recovered from the checkpoint cfg at eval, but pass the override anyway so
# the whole pipeline is unambiguously in diffusion mode.
EV=$(SW_CKPT="$SW_CKPT_DIR/best.pt" SW_RESULTS_DIR=$WS/results_diffusion SW_EVAL_EPISODES=90 \
     SW_OVERRIDES="$ARCH" \
     sbatch --parsable --gres="$GPU" --dependency=afterok:"$WM" slurm/eval.sbatch)

echo "diffusion ablation chain:  worldmodel=$WM  ->  eval=$EV"
echo "results land in $WS/results_diffusion/eval_summary.json"
echo "compare vs U-Net: shadow_diversity_ratio (want >1) and calibration_error, NOT raw shadow_iou."
echo "watch:  squeue -u $USER   |   tail -f slurm/logs/worldmodel_${WM}.out"
