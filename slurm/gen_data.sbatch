#!/bin/bash
#SBATCH --job-name=sw-gendata
#SBATCH --output=slurm/logs/gendata_%A_%a.out
#SBATCH --error=slurm/logs/gendata_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#
# Rollout generation, sharded across an array. Each task writes its own episode
# range, so the shards never collide and a failed shard can be resubmitted alone.
#
#   sbatch slurm/gen_data.sbatch                       # 8 shards, train split
#   SW_SPLIT=val SW_EPISODES=80 sbatch slurm/gen_data.sbatch
#
# MuJoCo rendering needs a GPU for EGL; drop --gres and set SW_MUJOCO_GL=osmesa for
# a CPU-only partition.

source slurm/env.sh

SPLIT="${SW_SPLIT:-train}"
TOTAL="${SW_EPISODES:-400}"
STEPS="${SW_STEPS:-300}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

PER_SHARD=$(( (TOTAL + NSHARDS - 1) / NSHARDS ))
SEED0=$(( SHARD * PER_SHARD ))
# The last shard trims any overshoot from the ceiling division above.
REMAINING=$(( TOTAL - SEED0 ))
(( REMAINING < PER_SHARD )) && PER_SHARD=$REMAINING
(( PER_SHARD <= 0 )) && { echo "shard ${SHARD}: nothing to do"; exit 0; }

# Offset val seeds so they cannot coincide with train seeds and leak scenes.
[[ "${SPLIT}" == "val" ]] && SEED0=$(( SEED0 + 900000 ))

echo "shard ${SHARD}/${NSHARDS}: ${PER_SHARD} episodes, split=${SPLIT}, seed0=${SEED0}"

srun python -m shadowweave.sim.synthetic_data \
  --episodes "${PER_SHARD}" \
  --split "${SPLIT}" \
  --steps "${STEPS}" \
  --seed0 "${SEED0}" \
  --out "${SW_DATA_ROOT:-data/rollouts}"

echo "shard ${SHARD} done"
