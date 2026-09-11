#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可行性验证：omni_xpu_kernel 能否作为 ai-toolkit 在 XPU 上的量化后端（替代 torchao）。

用法（训练结束后运行）：
    E:\\tt\\python\\python.exe E:\\tt\\scripts\\verify_omni_xpu_backend.py

检查项：
  1. 能否 import，报告版本 / 构建目标(__xpu_target__) / 设备是否匹配
  2. int8_linear 的正确性（与 bf16 参考路径对比）
  3. 性能（int8 内核 vs 现在 torchao 的"反量化 + bf16 matmul"）
  4. 显存占用差异
  5. 反向可行性：确认这些是推理内核（无 autograd），并验证"自定义 autograd 包装"是否可训练
"""

from __future__ import annotations

import os
import sys
import time

import torch


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def device_target() -> str:
    """Map the actual XPU device to the omni_xpu_kernel build target."""
    try:
        name = torch.xpu.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        print("  ! 无法读取 XPU 设备名:", exc)
        return "unknown"
    low = name.lower()
    if "arc" in low or "dg2" in low or "a770" in low or "a750" in low:
        return "dg2"
    if "bmg" in low or "b-series" in low or "arc pro b" in low:
        return "bmg"
    return "unknown"


def main() -> int:
    hr("1) 导入与目标匹配")
    try:
        import omni_xpu_kernel as oak
        from omni_xpu_kernel import int8
    except Exception as exc:
        print("  ✗ omni_xpu_kernel 不可用:", type(exc).__name__, exc)
        print("  → 需要在当前 Python/ABI 下构建对应 wheel（dg2 走 Blackwood416 源码，bmg 走官方）")
        return 2

    build_target = getattr(oak, "__xpu_target__", "unknown")
    print(f"  omni_xpu_kernel 版本 : {getattr(oak, '__version__', '?')}")
    print(f"  构建 torch 版本     : {getattr(oak, '__torch_version__', '?')}")
    print(f"  构建 GPU 目标       : {build_target}")
    print(f"  运行 torch          : {torch.__version__}")
    dev = device_target()
    print(f"  实际设备            : {torch.xpu.get_device_name(0)} -> 期望目标 {dev}")
    if build_target != dev:
        print(f"  ✗ 目标不匹配（轮子={build_target} 设备={dev}）→ 不能用于训练")
        return 3
    print("  ✓ 目标匹配")

    hr("2) int8_linear 正确性（对比 bf16 参考）")
    torch.manual_seed(0)
    M, K, N = 2048, 3072, 3072          # 接近 krea2 的线性层规模
    x = torch.randn(M, K, dtype=torch.bfloat16, device="xpu")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="xpu") * 0.02

    try:
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)
        y_int8 = int8.int8_linear(x, w_int8, w_scale, bias=None, out_dtype=torch.bfloat16)
    except Exception as exc:
        print("  ✗ 调用失败:", type(exc).__name__, str(exc)[:160])
        return 4

    y_ref = x @ w.t()
    diff = (y_int8.float() - y_ref.float()).abs()
    rel = diff.mean() / y_ref.float().abs().mean().clamp_min(1e-6)
    print(f"  int8 输出: {tuple(y_int8.shape)} {y_int8.dtype}")
    print(f"  最大误差 : {diff.max().item():.4f}")
    print(f"  平均相对误差: {rel.item()*100:.3f}%   (仅权重量化，激活按行 int8)")

    hr("3) 性能：int8 内核 vs 现在的 torchao 方式（反量化+bf16 matmul）")

    def timeit(fn, iters=5):
        fn()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    t_int8 = timeit(lambda: int8.int8_linear(x, w_int8, w_scale, bias=None, out_dtype=torch.bfloat16))
    # 现在 ai-toolkit 的做法：把 int8 反量化回 bf16 再普通 matmul
    w_dq = w  # 参考：这里用原 bf16 权重代表"反量化后的权重"
    t_bf16 = timeit(lambda: x @ w_dq.t())
    print(f"  omni int8_linear : {t_int8:.2f} ms")
    print(f"  反量化+bf16 matmul: {t_bf16:.2f} ms")
    print(f"  纯 bf16（权重常驻）: {t_bf16:.2f} ms")

    hr("4) 显存")
    torch.xpu.empty_cache()
    base = torch.xpu.memory_allocated() / 2**20
    y = int8.int8_linear(x, w_int8, w_scale, bias=None, out_dtype=torch.bfloat16)
    after = torch.xpu.memory_allocated() / 2**20
    print(f"  调用前 {base:.0f} MiB → 调用后 {after:.0f} MiB（含输入/输出常驻）")
    print(f"  int8 权重占用 {w_int8.numel()/2**20:.0f} MiB vs bf16 权重 {w.numel()*2/2**20:.0f} MiB")

    hr("5) 反向可行性（训练必需）")
    xg = x.clone().requires_grad_(True)
    yg = int8.int8_linear(xg, w_int8, w_scale, bias=None, out_dtype=torch.bfloat16)
    print(f"  输出 requires_grad = {yg.requires_grad}  grad_fn = {type(yg.grad_fn).__name__ if yg.grad_fn else None}")
    try:
        yg.float().sum().backward()
        print("  （意外）反向成功，x.grad =", xg.grad is not None)
    except Exception as exc:
        print(f"  ✓ 预期结果：内核无 autograd（{type(exc).__name__}）")
        print("  → 训练需要自定义 torch.autograd.Function：前向走 int8 内核，")
        print("     反向用 dequantize_int8_simple 把权重复原进【可复用缓冲】算 grad_input（LoRA 梯度照常 bf16）")

    hr("结论")
    print("  若第 1、2 项通过 → omni_xpu_kernel 可作为 XPU 训练后端候选；")
    print("  性能/显存优于现有路径时，在 ai-toolkit 中新增可选 qtype（如 xpu_int8），")
    print("  检测不到该后端时回退 torchao int8。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
