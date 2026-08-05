#!/bin/bash
# Cluster-specific environment for every ShadowWeave SLURM job.
# Every value can be overridden by exporting it before calling sbatch, e.g.
#     SW_ENV=shadowweave-cuda SW_CUDA_MODULE=cuda/12.6 sbatch slurm/train_worldmodel.sbatch
#
# Edit the defaults here once to match your cluster.

set -euo pipefail

# ── Conda ───────────────────────────────────────────────────────────────
SW_ENV="${SW_ENV:-shadowweave}"
SW_CONDA_BASE="${SW_CONDA_BASE:-}"

if [[ -z "${SW_CONDA_BASE}" ]]; then
  if command -v conda &>/dev/null; then
    SW_CONDA_BASE="$(conda info --base)"
  else
    for guess in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /opt/homebrew/anaconda3; do
      [[ -d "$guess" ]] && SW_CONDA_BASE="$guess" && break
    done
  fi
fi

if [[ -z "${SW_CONDA_BASE}" || ! -f "${SW_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: could not locate conda. Set SW_CONDA_BASE=/path/to/conda" >&2
  exit 1
fi

# ── Modules (optional; harmless when there is no module system) ─────────
SW_CUDA_MODULE="${SW_CUDA_MODULE:-cuda/12.4}"
if command -v module &>/dev/null && [[ -n "${SW_CUDA_MODULE}" ]]; then
  module load "${SW_CUDA_MODULE}" 2>/dev/null || \
    echo "note: module '${SW_CUDA_MODULE}' unavailable, relying on the conda CUDA runtime"
fi

# shellcheck disable=SC1091
source "${SW_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${SW_ENV}"

# ── Headless rendering ──────────────────────────────────────────────────
# Compute nodes have no X display, so MuJoCo must render off-screen. EGL uses the
# GPU; set SW_MUJOCO_GL=osmesa to fall back to software rendering.
export MUJOCO_GL="${SW_MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${MUJOCO_GL}"

# ── Threads ─────────────────────────────────────────────────────────────
# Without this each DataLoader worker spawns a full OpenMP pool and they thrash.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Weights & Biases ────────────────────────────────────────────────────
# Many clusters block outbound traffic from compute nodes; offline still records
# everything for a later `wandb sync`.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/wandb}"

# ── Project root ────────────────────────────────────────────────────────
SW_ROOT="${SW_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${SW_ROOT}"
export PYTHONPATH="${SW_ROOT}:${PYTHONPATH:-}"

echo "─────────────────────────────────────────────"
echo " job        : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo " node       : $(hostname)"
echo " conda env  : ${SW_ENV}"
echo " python     : $(which python)"
echo " MUJOCO_GL  : ${MUJOCO_GL}"
echo " root       : ${SW_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
  || echo " gpu        : none visible"
echo "─────────────────────────────────────────────"
