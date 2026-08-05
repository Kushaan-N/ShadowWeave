#!/bin/bash
# Create a plain Python venv for ShadowWeave — the pip-only alternative to conda.
#
#   ./scripts/setup_venv.sh                     # autodetect CUDA from nvidia-smi
#   SW_CUDA=cu121 ./scripts/setup_venv.sh       # force a wheel index
#   SW_CUDA=cpu   ./scripts/setup_venv.sh       # CPU-only (login node, CI)
#   SW_VENV=/scratch/$USER/venv ./scripts/setup_venv.sh
#
# Why venv is fine here: PyTorch's pip wheels bundle their own CUDA runtime, so the
# only thing conda uniquely provided was mesalib for the osmesa software-rendering
# fallback. EGL on a GPU node comes from the NVIDIA driver, not from the Python
# environment, so headless MuJoCo works either way.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${SW_VENV:-$PWD/.venv}"
PYTHON_BIN="${SW_PYTHON:-python3}"

# ── Python version gate ────────────────────────────────────────────────
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} not found. On a cluster try: module load python/3.11" >&2
  exit 1
fi
PY_VER=$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "ERROR: Python >= 3.10 required, found ${PY_VER}." >&2
  echo "  Load a newer one (module avail python) or set SW_PYTHON=/path/to/python3.11" >&2
  exit 1
fi

# ── Which torch wheel index ────────────────────────────────────────────
# The wheel's bundled CUDA must be <= the driver's supported version. nvidia-smi
# reports the driver's max; picking a newer wheel is the usual cause of a silent
# "CUDA available: False" on an otherwise healthy node.
CUDA_TAG="${SW_CUDA:-}"
if [[ -z "${CUDA_TAG}" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || echo "")
    case "${DRIVER_CUDA}" in
      12.[6-9]|12.1[0-9]|13.*) CUDA_TAG=cu124 ;;
      12.[4-5])                CUDA_TAG=cu124 ;;
      12.[1-3])                CUDA_TAG=cu121 ;;
      11.*)                    CUDA_TAG=cu118 ;;
      *)                       CUDA_TAG=cu124 ;;
    esac
    echo "detected driver CUDA ${DRIVER_CUDA:-unknown} -> torch wheels ${CUDA_TAG}"
  else
    CUDA_TAG=cpu
    echo "no nvidia-smi (login node?) -> CPU wheels; re-run on a GPU node for CUDA"
  fi
fi

INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "─────────────────────────────────────────────"
echo " python : ${PYTHON_BIN} (${PY_VER})"
echo " venv   : ${VENV}"
echo " torch  : ${CUDA_TAG}"
echo "─────────────────────────────────────────────"

"${PYTHON_BIN}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel --quiet

# torch first and from its own index, so the CUDA build is not silently replaced by
# the default PyPI (CPU) wheel while resolving a later dependency.
python -m pip install torch torchvision --index-url "${INDEX}"
python -m pip install -e ".[sim,viz,audio,depth,dev,usd]"

echo
echo "─────────────────────────────────────────────"
python - <<'EOF'
import torch
print(f" torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f" built for CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s)")
    print(f" {torch.cuda.get_device_name(0)}")
else:
    print(" no CUDA visible — expected on a login node; verify on a GPU node")
EOF
echo "─────────────────────────────────────────────"
echo
echo "Next:"
echo "  source ${VENV}/bin/activate"
echo "  srun --gres=gpu:1 --pty python slurm/preflight.py"
echo
echo "SLURM picks this venv up automatically if it lives at ./.venv;"
echo "otherwise export SW_VENV=${VENV} before sbatch."
