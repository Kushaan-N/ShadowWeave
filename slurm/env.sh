#!/bin/bash
# Cluster-specific environment for every ShadowWeave SLURM job.
# Every value can be overridden by exporting it before calling sbatch, e.g.
#     SW_ENV=shadowweave-cuda SW_CUDA_MODULE=cuda/12.6 sbatch slurm/train_worldmodel.sbatch
#
# Edit the defaults here once to match your cluster.

set -euo pipefail

# ── Python environment: venv or conda ───────────────────────────────────
# venv wins if SW_VENV is set or a .venv exists in the repo. Both work; venv is
# often the better fit on HPC, where a Python module is already provided and pip
# wheels ship their own CUDA runtime. Conda is only genuinely needed if you want
# its mesalib for the osmesa software-rendering fallback — EGL on an NVIDIA node
# comes from the driver, not from the environment.
SW_ENV="${SW_ENV:-shadowweave}"
SW_VENV="${SW_VENV:-}"
SW_CONDA_BASE="${SW_CONDA_BASE:-}"
SW_PYTHON_MODULE="${SW_PYTHON_MODULE:-}"

# ── Modules (optional; harmless when there is no module system) ─────────
SW_CUDA_MODULE="${SW_CUDA_MODULE:-cuda/12.4}"
if command -v module &>/dev/null; then
  [[ -n "${SW_PYTHON_MODULE}" ]] && { module load "${SW_PYTHON_MODULE}" 2>/dev/null || \
    echo "note: module '${SW_PYTHON_MODULE}' unavailable"; }
  [[ -n "${SW_CUDA_MODULE}" ]] && { module load "${SW_CUDA_MODULE}" 2>/dev/null || \
    echo "note: module '${SW_CUDA_MODULE}' unavailable, relying on the wheel's CUDA runtime"; }
fi

_SW_REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -z "${SW_VENV}" && -f "${_SW_REPO}/.venv/bin/activate" ]] && SW_VENV="${_SW_REPO}/.venv"

if [[ -n "${SW_VENV}" ]]; then
  if [[ ! -f "${SW_VENV}/bin/activate" ]]; then
    echo "ERROR: no venv at ${SW_VENV} — create one with ./scripts/setup_venv.sh" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${SW_VENV}/bin/activate"
  SW_ENV_KIND="venv:${SW_VENV}"
else
  if [[ -z "${SW_CONDA_BASE}" ]]; then
    if command -v conda &>/dev/null; then
      SW_CONDA_BASE="$(conda info --base)"
    else
      for guess in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        [[ -d "$guess" ]] && SW_CONDA_BASE="$guess" && break
      done
    fi
  fi
  if [[ -z "${SW_CONDA_BASE}" || ! -f "${SW_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: no venv found and could not locate conda." >&2
    echo "  Either: ./scripts/setup_venv.sh   (then re-submit)" >&2
    echo "  Or:     export SW_CONDA_BASE=/path/to/conda" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${SW_CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${SW_ENV}"
  SW_ENV_KIND="conda:${SW_ENV}"
fi

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
echo " python env : ${SW_ENV_KIND}"
echo " python     : $(which python)"
echo " MUJOCO_GL  : ${MUJOCO_GL}"
echo " root       : ${SW_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
  || echo " gpu        : none visible"
echo "─────────────────────────────────────────────"

# Guard against the silent-CPU trap: a CPU-only wheel (or one newer than the driver
# supports) leaves torch.cuda.is_available() False, and every job then crawls on CPU
# with no error. If a GPU is visible but torch can't use it, say so loudly.
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
  if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "!! WARNING: a GPU is visible but torch.cuda.is_available() is False."
    echo "!!          This is almost always a CPU-only or driver-mismatched torch wheel;"
    echo "!!          every job will run on CPU. Rebuild the env ON A GPU NODE:"
    echo "!!          SW_CUDA=cuXXX ./scripts/setup_venv.sh   (see HANDOFF §3)"
  fi
fi
