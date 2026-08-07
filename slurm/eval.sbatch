#!/bin/bash
#SBATCH --job-name=sw-eval
#SBATCH --output=slurm/logs/eval_%j.out
#SBATCH --error=slurm/logs/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
#   sbatch slurm/eval.sbatch
#   SW_EVAL_EPISODES=300 sbatch slurm/eval.sbatch

source slurm/env.sh
mkdir -p slurm/logs

# The 10s horizon needs h*fps+1 = 301 steps/episode to come due; run_eval defaults to
# 200, which silently drops iou_10s — the flagship long-horizon metric. Bake in a
# sufficient default so the one-command pipeline scores every horizon. SW_OVERRIDES is
# appended LAST so a caller can raise the step count or redirect the results (e.g.
# SW_RESULTS_DIR=results_diffusion, or eval.results_dir=... in SW_OVERRIDES) without
# having to re-supply these defaults — later keys win in an OmegaConf dotlist.
srun python -m shadowweave.eval.run_eval \
  --ckpt "${SW_CKPT:-checkpoints/world_model/best.pt}" \
  --episodes "${SW_EVAL_EPISODES:-90}" \
  --overrides \
    eval.steps_per_episode="${SW_EVAL_STEPS:-301}" \
    eval.results_dir="${SW_RESULTS_DIR:-results}" \
    ${SW_OVERRIDES:-}

echo "eval finished — see results/eval_summary.json"
