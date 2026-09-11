#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""omni 里除 int8 之外还能拿什么：sdp / norm 实测 + fp8 可行性。

用法: .venv\\Scripts\\python.exe scripts\\bench_omni_more.py
（sdp 想测 DG2 ESIMD 内核需先设 OMNI_ATTN_BACKEND=esimd）
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

WARMUP = 3
ITERS = 10


def timeit(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def bench_sdp() -> None:
    from omni_xpu_kernel import sdp

    hr(f"注意力 SDP（OMNI_ATTN_BACKEND={os.environ.get('OMNI_ATTN_BACKEND')}）")
    print(f"{'shape B,H,L,D':>24} | {'omni sdp':>10} | {'torch SDPA':>11} | 比率")
    print("-" * 66)
    for b, h, l, d in ((1, 24, 4096, 128), (1, 24, 8192, 128), (1, 32, 4096, 64)):
        q = torch.randn(b, h, l, d, device="xpu", dtype=torch.bfloat16)
        k = torch.randn(b, h, l, d, device="xpu", dtype=torch.bfloat16)
        v = torch.randn(b, h, l, d, device="xpu", dtype=torch.bfloat16)
        try:
            t_omni = timeit(lambda: sdp.sdp(q, k, v))
        except Exception as exc:  # noqa: BLE001
            print(f"{f'{b},{h},{l},{d}':>24} | 失败: {type(exc).__name__}: {str(exc)[:40]}")
            continue
        t_torch = timeit(lambda: F.scaled_dot_product_attention(q, k, v))
        print(f"{f'{b},{h},{l},{d}':>24} | {t_omni:>8.2f}ms | {t_torch:>9.2f}ms | {t_torch / t_omni:.2f}x")


def bench_norm() -> None:
    from omni_xpu_kernel import norm

    hr("RMSNorm")
    print(f"{'tokens x hidden':>18} | {'omni rms':>9} | {'eager rms':>10} | 比率")
    print("-" * 60)
    for tokens, hidden in ((8192, 3072), (4096, 4096)):
        x = torch.randn(tokens, hidden, device="xpu", dtype=torch.bfloat16)
        w = torch.randn(hidden, device="xpu", dtype=torch.bfloat16)

        def eager():
            xf = x.float()
            return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(torch.bfloat16)

        t_omni = timeit(lambda: norm.rms_norm(w, x))
        t_eager = timeit(eager)
        print(f"{f'{tokens}x{hidden}':>18} | {t_omni:>7.3f}ms | {t_eager:>8.3f}ms | {t_eager / t_omni:.2f}x")


def try_fp8() -> None:
    from omni_xpu_kernel import linear

    hr("FP8 W8A16（权重 fp8_e4m3 + bf16 激活）")
    x = torch.randn(4096, 3072, device="xpu", dtype=torch.bfloat16)
    w = torch.randn(12288, 3072, device="xpu", dtype=torch.bfloat16)
    try:
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 448.0
        w_fp8 = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        out = linear.onednn_w8a16_fp8(x, w_fp8, scale)
        ref = F.linear(x, (w_fp8.float() * scale).to(torch.bfloat16))
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item() * 100
        t_fp8 = timeit(lambda: linear.onednn_w8a16_fp8(x, w_fp8, scale))
        t_bf16 = timeit(lambda: F.linear(x, w))
        print(f"  可用 ✅  相对误差 {rel:.2f}%  | fp8 {t_fp8:.2f}ms vs bf16 {t_bf16:.2f}ms  ({t_bf16 / t_fp8:.2f}x)")
        print(f"  权重显存：fp8 {w_fp8.numel() / 1024 / 1024:.0f} MiB vs bf16 {w.numel() * 2 / 1024 / 1024:.0f} MiB")
    except Exception as exc:  # noqa: BLE001
        print(f"  不可用/失败：{type(exc).__name__}: {str(exc)[:120]}")


def main() -> int:
    print(f"torch {torch.__version__}  {torch.xpu.get_device_name(0)}")
    bench_sdp()
    bench_norm()
    try_fp8()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
