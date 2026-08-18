#!/bin/bash
# Ablation — retrain + eval the world model WITHOUT the optical-flow input channel, to
# isolate whether the model uses motion cues (vs copying static layout) when forecasting
# into shadow. Complements run_ablation.sh (no-visibility): together they form the input
# ablation table. Everything stays on scratch — nothing written to /work.
# Compare results_noflow shadow_gain@5s against the full run (+0.320) in ./results/.
# Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts
export SW_CKPT_DIR=$WS/checkpoints/world_model_noflow    # ablation ckpt -> scratch
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu

GPU="${SW_TRAIN_GPU:-gpu:1}"
ABLATE="bev.input_channels=[occupancy,visibility]"   # drop the flow (motion) channels

WM=$(SW_OVERRIDES="$ABLATE" \
     sbatch --parsable --gres="$GPU" --time="${SW_TIME:-08:00:00}" slurm/train_worldmodel.sbatch)

# Same channel override to eval so the whole pipeline stays consistent (the input stack is
# built from the LIVE cfg at eval, run_eval.py, while the model is built from the ckpt cfg).
EV=$(SW_CKPT="$SW_CKPT_DIR/best.pt" SW_RESULTS_DIR=$WS/results_noflow SW_EVAL_EPISODES=90 \
     SW_OVERRIDES="$ABLATE" \
     sbatch --parsable --gres="$GPU" --dependency=afterok:"$WM" slurm/eval.sbatch)

echo "no-flow ablation chain:  worldmodel=$WM  ->  eval=$EV"
echo "results land in $WS/results_noflow/eval_summary.json"
echo "compare:  full-model shadow_gain@5s = +0.320  (lower here => motion cue matters for shadow forecasting)"
echo "watch:  squeue -u $USER   |   tail -f slurm/logs/worldmodel_${WM}.out"
