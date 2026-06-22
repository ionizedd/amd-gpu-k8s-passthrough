# AMD GPU → Kubernetes Passthrough (WSL2 + ROCm)

These files let you run GPU-accelerated training inside a Kubernetes pod on Windows via WSL2 + AMD ROCm. Tested on an RX 9070 XT — works on any RDNA2/3/4 card with ROCm support.

## Prerequisites

- Windows 11 with WSL2 + GPU feature enabled (`/dev/dxg` exists)
- ROCm installed at `/opt/rocm` inside your WSL distro
  - Install: https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html
- k3s (or any k8s): `curl -sfL https://get.k3s.io | sh -`
- `python3`, `python3-venv`

Quick sanity check:
```bash
ls /dev/dxg
/opt/rocm/bin/rocminfo | grep -i "marketing name"
```

---

## Step 1 — Run setup (first time only)

```bash
chmod +x setup-rocm-wsl.sh
./setup-rocm-wsl.sh
```

This will:
1. Create a venv at `/opt/rocm_venv`
2. Download + install torch 2.9.1+rocm6.4 (~4.4 GB, cached in `/tmp`)
3. Apply the **libhsa WSL fix** (critical — see below)
4. Run `gpu-test.py` to confirm the GPU works

Expected output:
```
device: AMD Radeon RX 9070 XT
fp16 matmul: 9.9 TFLOP/s  (WMMA matrix cores)
train loss 2.327 -> 0.000  | peak VRAM 0.35 GB
GPU_TRAINING_OK   EXIT_CODE=0
```

---

## Step 2 — Launch the pod

```bash
kubectl apply -f gpu-pod.yaml
kubectl logs -f gpu-train
```

To re-run: `kubectl delete pod gpu-train && kubectl apply -f gpu-pod.yaml`

---

## The libhsa WSL fix — what and why

The pip `torch+rocm` wheel ships `libhsa-runtime64.so` built for bare-metal Linux (`/dev/kfd`). WSL exposes `/dev/dxg` instead. `/opt/rocm`'s libhsa already knows how to use it.

Fix = symlink the wheel's copy to ROCm's WSL-aware one:

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
for f in "$SITE"/torch/lib/libhsa-runtime64.so*; do
    ln -sf /opt/rocm/lib/libhsa-runtime64.so "$f"
done
```

Without this: `torch.cuda.is_available()` → `False`.
With it: GPU detected, WMMA cores active, training works.

---

## Plugging in your own training script

Replace the `python /opt/rocm_venv/gpu-test.py` line in `gpu-pod.yaml` with your script, or mount your code as an extra `hostPath` volume.

For larger jobs, convert the pod to a `batch/v1` Job with `restartPolicy: OnFailure`.

---

## Perf notes

| Path | fp16 matmul | Notes |
|------|------------|-------|
| Windows-native (ROCm 7.2) | much higher | Fastest; no WSL overhead |
| WSL in-pod (ROCm 6.4) | ~10 TFLOP/s | Works; ~15% overhead |

For max throughput, train on the Windows-native env. Use the pod for orchestration/reproducibility.

---

## Files

| File | Purpose |
|------|---------|
| `setup-rocm-wsl.sh` | One-shot host setup + libhsa fix |
| `gpu-pod.yaml` | Kubernetes pod manifest with GPU passthrough |
| `gpu-test.py` | fp16 matmul + training loop smoke test |
