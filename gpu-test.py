"""
gpu-test.py — verifies ROCm GPU access and runs a quick fp16 training loop.

Expected output (numbers vary by GPU):
    device: AMD Radeon RX 9070 XT
    fp16 matmul: 9.9 TFLOP/s  (WMMA matrix cores)
    train loss 2.327 -> 0.000  | peak VRAM 0.35 GB
    GPU_TRAINING_OK   EXIT_CODE=0
"""
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn

# 1. verify GPU is visible
if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False")
    print("  -> Did you run the libhsa WSL fix in setup-rocm-wsl.sh?")
    print("  -> Is /dev/dxg mounted into the pod?")
    sys.exit(1)

device = torch.device("cuda:0")
name   = torch.cuda.get_device_name(0)
print(f"device: {name}")

# 2. fp16 matmul throughput (exercises WMMA matrix cores)
N = 4096
a = torch.randn(N, N, dtype=torch.float16, device=device)
b = torch.randn(N, N, dtype=torch.float16, device=device)

for _ in range(3):  # warm-up
    _ = torch.matmul(a, b)
torch.cuda.synchronize()

RUNS = 20
t0 = time.perf_counter()
for _ in range(RUNS):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

flops  = 2 * N**3 * RUNS
tflops = flops / elapsed / 1e12
print(f"fp16 matmul: {tflops:.1f} TFLOP/s  (WMMA matrix cores)")

del a, b, c
torch.cuda.empty_cache()

# 3. tiny training loop
torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(128, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10),
).to(device, dtype=torch.float16)

opt     = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(256, 128, device=device, dtype=torch.float16)
y = torch.randint(0, 10, (256,), device=device)

first_loss = last_loss = None
for step in range(200):
    opt.zero_grad()
    logits = model(x)
    loss   = loss_fn(logits, y)
    loss.backward()
    opt.step()
    if first_loss is None:
        first_loss = loss.item()
    last_loss = loss.item()

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"train loss {first_loss:.3f} -> {last_loss:.3f}  | peak VRAM {peak_gb:.2f} GB")
print("GPU_TRAINING_OK   EXIT_CODE=0")
sys.exit(0)
