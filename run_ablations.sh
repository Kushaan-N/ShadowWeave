#!/bin/bash
# Both remaining world-model ablations (no-flow, then diffusion) under ONE launch, chained
# to run ONE GPU AT A TIME. Two full trainings cannot share a single 8h allocation (each is
# ~4-5h and diffusion is slower), so instead the diffusion chain is made to start only after
# the no-flow chain finishes: only the first job waits in the queue, the rest are pre-queued
# with dependencies. Reuses the tested train_worldmodel/eval sbatch (with their SIGUSR1
# requeue+resume, so a >8h diffusion run survives the wall). Everything on scratch.
#
#   bash run_ablations.sh            # no-flow -> (frees GPU) -> diffusion, sequential
#   SW_PARALLEL=1 bash run_ablations.sh   # drop the cross-dependency: run on 2 GPUs at once
#
# Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu
GPU="${SW_TRAIN_GPU:-gpu:1}"

# ---- no-flow ablation: drop the optical-flow channels ----
NF_OVR="bev.input_channels=[occupancy,visibility]"
NF_DIR=$WS/checkpoints/world_model_noflow
NF_WM=$(SW_CKPT_DIR=$NF_DIR SW_OVERRIDES="$NF_OVR" \
        sbatch --parsable --gres="$GPU" --time=08:00:00 slurm/train_worldmodel.sbatch)
NF_EV=$(SW_CKPT=$NF_DIR/best.pt SW_RESULTS_DIR=$WS/results_noflow SW_EVAL_EPISODES=90 SW_OVERRIDES="$NF_OVR" \
        sbatch --parsable --gres="$GPU" --dependency=afterok:"$NF_WM" slurm/eval.sbatch)

# ---- diffusion ablation: conditional DDPM instead of the BCE U-Net ----
DF_OVR="world_model.architecture=diffusion"
DF_DIR=$WS/checkpoints/world_model_diffusion
# By default gate the diffusion training on the no-flow chain finishing, so only one GPU is
# used at a time. Set SW_PARALLEL=1 to remove the gate and run both concurrently on 2 GPUs.
DEP=""
[[ "${SW_PARALLEL:-0}" != "1" ]] && DEP="--dependency=afterany:$NF_EV"
DF_WM=$(SW_CKPT_DIR=$DF_DIR SW_OVERRIDES="$DF_OVR" \
        sbatch --parsable --gres="$GPU" --time=08:00:00 $DEP slurm/train_worldmodel.sbatch)
DF_EV=$(SW_CKPT=$DF_DIR/best.pt SW_RESULTS_DIR=$WS/results_diffusion SW_EVAL_EPISODES=90 SW_OVERRIDES="$DF_OVR" \
        sbatch --parsable --gres="$GPU" --dependency=afterok:"$DF_WM" slurm/eval.sbatch)

echo "no-flow:    worldmodel=$NF_WM -> eval=$NF_EV   (results_noflow)"
echo "diffusion:  worldmodel=$DF_WM -> eval=$DF_EV   (results_diffusion)"
[[ "${SW_PARALLEL:-0}" != "1" ]] && echo "diffusion waits for the no-flow chain (one GPU at a time)."
echo "watch:  squeue -u $USER"
