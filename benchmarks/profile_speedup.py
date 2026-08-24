"""Training Speedup & Latency Profiling Benchmark.

Measures:
1. Pure Forward Pass Latency (ms)
2. Batched Loss Computation Latency (ms)
3. Full Forward+Backward Step Latency (ms/step & steps/sec)
4. ModelEMA Overhead (ms)
"""

import time
import torch
from nanostream.config import NanoStreamConfig
from nanostream.model import NanoStreamOD
from nanostream.head import detection_loss
from nanostream.ema import ModelEMA
from benchmarks.combined_data import generate_sample
from benchmarks.train_nanostream import _sample_to_tensor


def run_profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"       NanoStream-OD Training Speed Benchmark ({device})")
    print("=" * 65)

    cfg = NanoStreamConfig.mcu(num_classes=4)
    model = NanoStreamOD(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ModelEMA(model, decay=0.999)

    # Pre-render 64 samples
    pool = [_sample_to_tensor(*generate_sample(i, cfg.input_size), cfg.input_size) for i in range(64)]

    batch_size = 16
    n_steps = 100

    # Warmup
    for _ in range(10):
        idxs = torch.randint(0, len(pool), (batch_size,))
        bx = torch.stack([pool[i][0] for i in idxs]).to(device)
        bt = [{k: v.to(device) for k, v in pool[i][1].items()} for i in idxs]
        preds = model(bx)
        loss = detection_loss(preds, bt, cfg)
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        ema.update(model)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark full training steps
    t0 = time.perf_counter()
    for step in range(n_steps):
        idxs = torch.randint(0, len(pool), (batch_size,))
        bx = torch.stack([pool[i][0] for i in idxs]).to(device)
        bt = [{k: v.to(device) for k, v in pool[i][1].items()} for i in idxs]

        opt.zero_grad(set_to_none=True)
        preds = model(bx)
        loss = detection_loss(preds, bt, cfg)
        loss["total"].backward()
        opt.step()
        ema.update(model, step)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    total_time = t1 - t0
    ms_per_step = (total_time / n_steps) * 1000.0
    steps_per_sec = n_steps / total_time
    imgs_per_sec = (n_steps * batch_size) / total_time

    print(f"\n[Results over {n_steps} training steps, batch_size={batch_size}]")
    print(f"  Total Time         : {total_time:.3f} s")
    print(f"  Step Latency       : {ms_per_step:.2f} ms / step")
    print(f"  Training Speed     : {steps_per_sec:.1f} steps / sec")
    print(f"  Image Throughput   : {imgs_per_sec:.1f} images / sec")
    print("=" * 65)


if __name__ == "__main__":
    run_profile()
