#!/bin/bash
#SBATCH --job-name=sw-worldmodel
#SBATCH --output=slurm/logs/worldmodel_%j.out
#SBATCH --error=slurm/logs/worldmodel_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@120
#
# World model training. Scales to multiple GPUs by raising --gres; the launcher
# switches to torchrun automatically.
#
#   sbatch slurm/train_worldmodel.sbatch
#   sbatch --gres=gpu:4 slurm/train_worldmodel.sbatch
#   SW_OVERRIDES="world_model.base_channels=96 world_model.lr=2e-4" sbatch slurm/train_worldmodel.sbatch
#
# On SIGUSR1 (sent 120s before the time limit) the job resubmits itself and resumes
# from checkpoints/world_model/last.pt, so a long run survives the queue limit.

source slurm/env.sh
mkdir -p slurm/logs

CKPT_DIR="${SW_CKPT_DIR:-checkpoints/world_model}"
DATA_ROOT="${SW_DATA_ROOT:-data/rollouts}"
OVERRIDES="${SW_OVERRIDES:-}"

# Resume automatically if a previous run of this job left a checkpoint.
RESUME_ARG=""
if [[ -f "${CKPT_DIR}/last.pt" && "${SW_FRESH:-0}" != "1" ]]; then
  RESUME_ARG="--resume ${CKPT_DIR}/last.pt"
  echo "resuming from ${CKPT_DIR}/last.pt"
fi

requeue_handler() {
  echo "time limit approaching — requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_handler USR1

NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
NGPU="${NGPU:-1}"
echo "training on ${NGPU} GPU(s)"

if [[ "${NGPU}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NGPU}" \
    -m shadowweave.world_model.train \
    --data "${DATA_ROOT}" ${RESUME_ARG} \
    --overrides world_model.checkpoint_dir="${CKPT_DIR}" ${OVERRIDES} &
else
  srun python -m shadowweave.world_model.train \
    --data "${DATA_ROOT}" ${RESUME_ARG} \
    --overrides world_model.checkpoint_dir="${CKPT_DIR}" ${OVERRIDES} &
fi

# Wait in a way that lets the USR1 trap fire (a bare `wait` blocks signals).
wait $!
echo "world model training finished"
