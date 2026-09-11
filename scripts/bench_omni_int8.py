#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按真实训练形状比较三条 XPU 路径：

  1. omni_xpu_kernel int8_linear（张量级 int8 内核，无 autograd）
  2. torchao int8 权重 → 反量化 + bf16 matmul（ai-toolkit 现在的路径）
  3. 纯 bf16 matmul（权重常驻显存，作为速度上限参考）

用法: .venv\\Scripts\\python.exe scripts\\bench_omni_int8.py
"""

from __future__ import annotations

import time

import torch

CASES = [
    # (M, K, N)  —— krea2 常見的 attention / FFN 形状，M 是 token 数
    (4096, 3072, 3072),
    (4096, 3072, 12288),
    (8192, 4096, 4096),
]
WARMUP = 3
ITERS = 10


def timeit(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) / ITERS * 1000.0


def main() -> int:
    from omni_xpu_kernel import int8

    print(f"torch {torch.__version__}  device {torch.xpu.get_device_name(0)}")
    print(f"{'shape':>22} | {'omni int8':>10} | {'torchao deq':>12} | {'pure bf16':>10} | int8 显存")
    print("-" * 82)

    for m, k, n in CASES:
        x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
        w_bf16 = torch.randn(n, k, device="xpu", dtype=torch.bfloat16)

        qmax = 127.0
        scale = w_bf16.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        w_int8 = (w_bf16 / scale).round().clamp(-qmax, qmax).to(torch.int8)

        def omni():
            return int8.int8_linear(x, w_int8, scale)

        def torchao_like():
            return torch.nn.functional.linear(x, (w_int8.to(torch.bfloat16) * scale))

        def pure():
            return torch.nn.functional.linear(x, w_bf16)

        t_omni = timeit(omni)
        t_deq = timeit(torchao_like)
        t_pure = timeit(pure)
        w_int8_mb = w_int8.numel() / 1024 / 1024
        w_bf16_mb = w_bf16.numel() * 2 / 1024 / 1024
        print(
            f"{f'{m}x{k}x{n}':>22} | {t_omni:>8.2f}ms | {t_deq:>10.2f}ms | {t_pure:>8.2f}ms |"
            f" {w_int8_mb:.0f} vs {w_bf16_mb:.0f} MiB"
        )
        del x, w_bf16, w_int8

    print("\n说明：omni int8 是无 autograd 的推理内核；训练还要自定义 autograd.Function。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
