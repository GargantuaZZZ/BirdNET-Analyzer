#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for RTX 5090 + TensorFlow nightly on Linux.
# What it does:
# 1) Clears XLA flags that may pin an old PTX version.
# 2) Optionally installs CUDA Toolkit 12.8 via apt.
# 3) Creates/uses a fresh conda env and installs tf-nightly[and-cuda].
# 4) Writes stable runtime env vars to ~/.bashrc.
# 5) Runs a TensorFlow GPU validation.

ENV_NAME="tf-nightly-5090"
PYTHON_VER="3.11"
CUDA_VER="12-8"
INSTALL_CUDA="false"
VISIBLE_GPUS=""

usage() {
  cat <<'EOF'
Usage:
  bash example/setup_tf5090_nightly.sh [options]

Options:
  --env-name NAME         Conda env name (default: tf-nightly-5090)
  --python VERSION        Python version (default: 3.11)
  --cuda-ver VER          CUDA apt suffix (default: 12-8)
  --install-cuda          Install cuda-toolkit-${VER} via apt
  --visible-gpus IDS      Set CUDA_VISIBLE_DEVICES, e.g. 3,4,5,6,7
  -h, --help              Show this help
EOF
}

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

die() {
  echo "[ERR ] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VER="$2"
      shift 2
      ;;
    --cuda-ver)
      CUDA_VER="$2"
      shift 2
      ;;
    --install-cuda)
      INSTALL_CUDA="true"
      shift
      ;;
    --visible-gpus)
      VISIBLE_GPUS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

log "Step 1/8: Diagnostics"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  warn "nvidia-smi not found. NVIDIA driver may be missing."
fi

if command -v ptxas >/dev/null 2>&1; then
  log "Current ptxas version:"
  ptxas --version || true
else
  warn "ptxas not found in PATH."
fi

log "Current XLA/TF/CUDA env vars:"
env | egrep "^(XLA_FLAGS|TF_XLA_FLAGS|TF_|CUDA)" || true

log "Step 2/8: Clear problematic XLA flags in current shell"
unset XLA_FLAGS || true
unset TF_XLA_FLAGS || true

log "Step 3/8: Remove pinned PTX-related lines from ~/.bashrc"
BASHRC="${HOME}/.bashrc"
[[ -f "${BASHRC}" ]] || touch "${BASHRC}"
cp "${BASHRC}" "${BASHRC}.bak.$(date +%Y%m%d_%H%M%S)"

tmp_file="$(mktemp)"
grep -vE "xla_gpu_ptx_version|XLA_FLAGS|TF_XLA_FLAGS" "${BASHRC}" > "${tmp_file}" || true
mv "${tmp_file}" "${BASHRC}"

log "Step 4/8: Optional CUDA toolkit installation"
if [[ "${INSTALL_CUDA}" == "true" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "cuda-toolkit-${CUDA_VER}"
  else
    warn "apt-get not found; skipping CUDA installation."
  fi
fi

log "Step 5/8: Resolve CUDA_HOME"
CUDA_HOME=""
if command -v ptxas >/dev/null 2>&1; then
  PTXAS_PATH="$(command -v ptxas)"
  CUDA_HOME="$(cd "$(dirname "${PTXAS_PATH}")/.." && pwd)"
fi

if [[ -z "${CUDA_HOME}" || ! -d "${CUDA_HOME}" ]]; then
  if [[ -d "/usr/local/cuda-12.8" ]]; then
    CUDA_HOME="/usr/local/cuda-12.8"
  elif [[ -d "/usr/local/cuda" ]]; then
    CUDA_HOME="/usr/local/cuda"
  fi
fi

if [[ -n "${CUDA_HOME}" ]]; then
  log "Using CUDA_HOME=${CUDA_HOME}"
else
  warn "CUDA_HOME not auto-detected; continuing anyway."
fi

log "Step 6/8: Persist runtime env vars to ~/.bashrc"
{
  echo ""
  echo "# Added by setup_tf5090_nightly.sh"
  if [[ -n "${CUDA_HOME}" ]]; then
    echo "export CUDA_HOME=${CUDA_HOME}"
    echo 'export PATH=${CUDA_HOME}/bin:${PATH}'
    echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}'
  fi
  echo "export TF_FORCE_GPU_ALLOW_GROWTH=true"
  echo "export TF_GPU_ALLOCATOR=cuda_malloc_async"
  echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID"
  if [[ -n "${VISIBLE_GPUS}" ]]; then
    echo "export CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS}"
  fi
} >> "${BASHRC}"

if [[ -n "${CUDA_HOME}" ]]; then
  export CUDA_HOME="${CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${VISIBLE_GPUS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
fi

log "Step 7/8: Create conda env and install tf-nightly"
command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda/Anaconda first."

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Conda env already exists: ${ENV_NAME}"
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y
fi

conda activate "${ENV_NAME}"
python -m pip install -U pip setuptools wheel
python -m pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tf-nightly tf-nightly-cpu keras keras-nightly || true
python -m pip install --pre --upgrade "tf-nightly[and-cuda]"

log "Step 8/8: Validate TensorFlow + GPU"
python - <<'PY'
import os
import tensorflow as tf
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
print("TF_XLA_FLAGS:", os.environ.get("TF_XLA_FLAGS"))
try:
    print("Build info:", tf.sysconfig.get_build_info())
except Exception as e:
    print("Build info unavailable:", e)
PY

log "Done. Open a new shell or run: source ~/.bashrc"
