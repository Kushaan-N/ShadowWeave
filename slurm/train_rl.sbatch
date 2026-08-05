#!/bin/bash
#SBATCH --job-name=sw-rl
#SBATCH --output=slurm/logs/rl_%j.out
#SBATCH --error=slurm/logs/rl_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#
# PPO training for the local agent. Needs a world model checkpoint to exist first
# (checkpoints/world_model/best.pt) or it warns and runs on random weights.
#
#   sbatch slurm/train_rl.sbatch
#   SW_RL_STEPS=2000000 sbatch slurm/train_rl.sbatch

source slurm/env.sh
mkdir -p slurm/logs

STEPS="${SW_RL_STEPS:-1000000}"
# One SubprocVecEnv worker per core, minus headroom for the learner and renderer.
NENVS="${SW_RL_ENVS:-$(( ${SLURM_CPUS_PER_TASK:-8} / 2 ))}"

echo "PPO: ${STEPS} steps across ${NENVS} envs"

# Honor SW_CKPT_DIR like train_worldmodel.sbatch does — otherwise a world model
# trained into a custom dir is silently ignored and PPO runs on random weights.
CKPT_OVERRIDE=""
if [[ -n "${SW_CKPT_DIR:-}" ]]; then
  CKPT_OVERRIDE="world_model.checkpoint_dir=${SW_CKPT_DIR}"
fi

srun python train_rl.py \
  --steps "${STEPS}" \
  --overrides agents.local.ppo_n_envs="${NENVS}" ${CKPT_OVERRIDE} ${SW_OVERRIDES:-}

echo "RL training finished"
