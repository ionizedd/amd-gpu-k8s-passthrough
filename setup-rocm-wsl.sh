#!/usr/bin/env bash
# =============================================================================
# setup-rocm-wsl.sh
# One-shot setup: torch+ROCm venv + libhsa WSL fix + GPU test
# Tested: WSL2, ROCm 6.4, RX 9070 XT, Ubuntu 22.04
# Run as a normal user (sudo used only where needed).
# =============================================================================
set -euo pipefail

VENV=/opt/rocm_venv
WHEEL_CACHE=/tmp/torch_rocm_wheel
ROCM_PATH=/opt/rocm

# 0. sanity checks
echo "=== Checking prerequisites ==="
[ -e /dev/dxg ]    || { echo "ERROR: /dev/dxg not found — WSL GPU feature not enabled"; exit 1; }
[ -d /opt/rocm ]   || { echo "ERROR: /opt/rocm not found — install ROCm first"; exit 1; }
command -v python3 || { echo "ERROR: python3 not found"; exit 1; }
command -v kubectl || echo "WARNING: kubectl not found (needed for the pod step)"

# 1. create venv
echo "=== Creating venv at $VENV ==="
sudo mkdir -p "$VENV"
sudo chown "$USER":"$USER" "$VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel

# 2. install torch+ROCm (~4.4 GB wheel, cached in /tmp so re-runs are fast)
echo "=== Installing torch+ROCm ==="
mkdir -p "$WHEEL_CACHE"

ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d. -f1,2 || echo "6.4")
echo "Detected ROCm $ROCM_VER"

# Adjust URL if your ROCm version differs
TORCH_URL="https://download.pytorch.org/whl/rocm6.4/torch-2.9.1%2Brocm6.4-cp310-cp310-linux_x86_64.whl"
WHEEL_FILE="$WHEEL_CACHE/torch-rocm64.whl"

if [ ! -f "$WHEEL_FILE" ]; then
    echo "Downloading wheel (4.4 GB)..."
    wget -c -O "$WHEEL_FILE" "$TORCH_URL"
else
    echo "Wheel already cached, skipping download."
fi

pip install "$WHEEL_FILE" numpy

# 3. libhsa WSL fix
# The pip wheel ships libhsa-runtime64.so built for bare-metal /dev/kfd.
# On WSL, /dev/dxg is used instead — swap in /opt/rocm's WSL-aware build.
echo "=== Applying libhsa WSL fix ==="
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
for f in "$SITE_PKG"/torch/lib/libhsa-runtime64.so*; do
    [ -e "$f" ] || continue
    echo "  Swapping $f -> $ROCM_PATH/lib/libhsa-runtime64.so"
    ln -sf "$ROCM_PATH/lib/libhsa-runtime64.so" "$f"
done

# 4. copy gpu-test.py into the venv dir (pod will find it there)
echo "=== Installing gpu-test.py ==="
cp "$(dirname "$0")/gpu-test.py" "$VENV/gpu-test.py" 2>/dev/null || \
    echo "WARNING: gpu-test.py not found — copy it manually to $VENV/gpu-test.py"

# 5. smoke test
echo "=== Running GPU smoke test ==="
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
python gpu-test.py

echo ""
echo "=== Setup complete! ==="
echo "To run the k8s pod:"
echo "  kubectl apply -f gpu-pod.yaml"
echo "  kubectl logs -f gpu-train"
