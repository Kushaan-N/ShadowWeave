#!/bin/bash
# RL fix validation — a SHORT PPO run (100k steps, not the full 1M/8h) + a quick eval,
# to prove the reactive-agent fix works before spending a GPU on the full run.
# Reuses the already-trained world model. Writes to *_rl_smoke dirs so it never touches
# the full run's checkpoints/results. Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts
export SW_CKPT_DIR=$WS/checkpoints/world_model      # the trained world model to load
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu

GPU="${SW_TRAIN_GPU:-gpu:1}"                          # any GPU
LOCAL_SMOKE=$WS/checkpoints/local_agent_smoke         # keep off the full run's ckpt

# Short PPO: 100k steps is ~10-15 min and enough to see the learning signals turn.
RL=$(SW_RL_STEPS=100000 \
     SW_OVERRIDES="agents.local.checkpoint_dir=$LOCAL_SMOKE" \
     sbatch --parsable --gres="$GPU" --time="${SW_TIME:-01:00:00}" slurm/train_rl.sbatch)

# Quick eval (30 episodes) using the smoke policy we just trained.
EV=$(SW_CKPT="$SW_CKPT_DIR/best.pt" SW_RESULTS_DIR=results_rl_smoke SW_EVAL_EPISODES=30 \
     SW_OVERRIDES="agents.local.checkpoint_dir=$LOCAL_SMOKE" \
     sbatch --parsable --gres="$GPU" --dependency=afterok:"$RL" slurm/eval.sbatch)

echo "rl-smoke chain:  ppo(100k)=$RL  ->  eval=$EV"
echo
echo "WATCH the PPO log for the fix taking hold (these were the broken values):"
echo "   ep_rew_mean / ep_len_mean  -> should now APPEAR (were absent) and rew trend UP"
echo "   explained_variance         -> should rise from ~0.1 toward >=0.4"
echo "   clip_fraction              -> should drop from ~0.46 toward ~0.1-0.2"
echo "   value_loss                 -> should drop well below ~3000"
echo "   tail -f slurm/logs/rl_${RL}.out"
echo
echo "PASS/FAIL at eval -> results_rl_smoke/eval_summary.json :"
echo "   collision_rate    should beat the UNtrained baseline 0.60 (lower is better)"
echo "   path_efficiency   should beat the UNtrained baseline 0.85 is the bar to clear... "
echo "   (untrained was 0.60 / 0.848; the broken trained run was 0.878 / 0.226)"
echo
echo "If EV stays ~0.1 and clip_fraction ~0.46, the fix did NOT take — do not launch the full run."
