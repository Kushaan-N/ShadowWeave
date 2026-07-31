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

srun python -m shadowweave.eval.run_eval \
  --ckpt "${SW_CKPT:-checkpoints/world_model/best.pt}" \
  --episodes "${SW_EVAL_EPISODES:-90}" \
  --overrides ${SW_OVERRIDES:-eval.results_dir=results}

echo "eval finished — see results/eval_summary.json"
