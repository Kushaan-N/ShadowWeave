#!/bin/bash
# Confound-free per-difficulty-tier shadow gain, on the FIXED validation set — as a SINGLE
# batch job (one queue wait, one fair-share hit) via slurm/val_pertier.sbatch. That job
# generates a clean single-tier val set per tier (reusing any already present) and scores
# one world model on each with the no-policy eval. ~10-15 min, all on GPU nodes, all on
# scratch.
#
#   bash run_val_pertier.sh                                  # full model -> val_pertier/full/
#   SW_CKPT=<ckpt> SW_TAG=novis bash run_val_pertier.sh      # ablation -> val_pertier/novis/
#
# Reruns REUSE the same rollouts_val_<tier> sets, so different checkpoints are compared on
# identical data. Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu

CKPT="${SW_CKPT:-$WS/checkpoints/world_model/best.pt}"
TAG="${SW_TAG:-full}"                       # output subdir, per checkpoint, avoids clobber
OUTDIR=$WS/val_pertier/$TAG

JOB=$(SW_CKPT="$CKPT" SW_VAL_PERTIER_DIR="$OUTDIR" \
      sbatch --parsable --gres="${SW_TRAIN_GPU:-gpu:1}" slurm/val_pertier.sbatch)

echo "per-tier (single job) for [$TAG]: $JOB"
echo "  checkpoint: $CKPT"
echo "  results   : $OUTDIR/{static,moving,debris}.json"
echo "watch:  squeue -j $JOB   |   tail -f slurm/logs/valpertier_${JOB}.out"
