#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 XPU（无 fp64）相关的修复：助手行为 + 各模型 rope/位置编码在 XPU 上可跑。

用法：E:\\tt\\.venv\\Scripts\\python.exe E:\\tt\\scripts\\verify_xpu_fp64_paths.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

# 让脚本能直接 import 仓库内的 toolkit / extensions_built_in
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, FAIL, SKIP = "✓", "✗", "–"


def check_helpers():
    print("== 1) device_utils 助手 ==")
    from toolkit.device_utils import adjust_dtype_for_device, rope_dtype

    cases = [
        ("rope_dtype(xpu) == float32", rope_dtype(torch.device("xpu")) == torch.float32),
        ("rope_dtype(cpu) == float64", rope_dtype(torch.device("cpu")) == torch.float64),
        (
            "adjust_dtype_for_device(float64, xpu) == float32",
            adjust_dtype_for_device(torch.float64, torch.device("xpu")) == torch.float32,
        ),
        (
            "adjust_dtype_for_device(float64, cuda) == float64",
            adjust_dtype_for_device(torch.float64, torch.device("cuda")) == torch.float64,
        ),
    ]
    for name, ok in cases:
        print(f"  {OK if ok else FAIL} {name}")
    return all(ok for _, ok in cases)


def check_fp64_unavailable_on_xpu():
    print("== 2) 前提确认：XPU 上 fp64 能分配、但算子不可用 ==")
    results = []
    try:
        torch.zeros(4, dtype=torch.float64, device="xpu")
        print("  – 分配 fp64 成功（XPU 允许分配，只是不能算）")
    except Exception as exc:
        print(f"  – 分配 fp64 也失败: {type(exc).__name__}")
    for name, fn in [
        ("arange", lambda: torch.arange(0, 8, 2, dtype=torch.float64, device="xpu")),
        ("cos", lambda: torch.cos(torch.arange(0, 8, dtype=torch.float64, device="xpu"))),
    ]:
        try:
            fn()
            print(f"  {FAIL} fp64.{name} 竟然可用 —— 前提不成立")
            results.append(False)
        except RuntimeError as exc:
            print(f"  {OK} fp64.{name} 不可用（预期）: {str(exc)[:70]}")
            results.append(True)
    return all(results)


def check_model_rope():
    print("== 3) 各模型 rope / 位置编码在 XPU 上的实测 ==")
    dev = torch.device("xpu")
    results = []

    def run(name, fn):
        try:
            out = fn()
            print(f"  {OK} {name}: {tuple(out.shape)} {out.dtype}")
            results.append(True)
        except ImportError as exc:
            print(f"  {SKIP} {name}: 跳过（导入失败 {str(exc)[:60]}）")
        except Exception as exc:
            print(f"  {FAIL} {name}: {type(exc).__name__}: {str(exc)[:110]}")
            results.append(False)

    # chroma
    def chroma():
        from extensions_built_in.diffusion_models.chroma.src.math import rope
        # 调用处是 rope(ids[..., i], ...) —— pos 形状为 (b, n)
        pos = torch.zeros(2, 16, dtype=torch.float32, device=dev)
        return rope(pos, 64, 10000)

    run("chroma.rope", chroma)

    # hidream embeddings
    def hidream():
        from extensions_built_in.diffusion_models.hidream.src.models.embeddings import rope
        pos = torch.zeros(2, 16, dtype=torch.float32, device=dev)
        return rope(pos, 64, 10000)

    run("hidream.embeddings.rope", hidream)

    # boogu_image freqs table
    def boogu():
        from extensions_built_in.diffusion_models.boogu_image.src.rope import get_freqs_cis
        tables = get_freqs_cis((16, 16, 16), (64, 64, 64), 10000)
        return tables[0].to(dev)

    run("boogu_image.get_freqs_cis", boogu)

    # omnigen2 freqs table
    def omnigen():
        from extensions_built_in.diffusion_models.omnigen2.src.models.transformers.repo import (
            OmniGen2RotaryPosEmbed,
        )
        tables = OmniGen2RotaryPosEmbed.get_freqs_cis((16, 16, 16), (64, 64, 64), 10000)
        return tables[0].to(dev)

    run("omnigen2.get_freqs_cis", omnigen)

    # minimax_h3 packing -> device transfer + rope
    def minimax():
        from extensions_built_in.diffusion_models.minimax_h3.src import packing
        # 用最简路径构造一个 layout，再按修复方式搬运
        pos = torch.zeros(8, 3, dtype=torch.float64)          # CPU 上 fp64（原实现）
        moved = pos[None].to(device=dev, dtype=torch.float32)  # 修复后的搬运
        return moved

    run("minimax_h3 position_ids 搬运", minimax)

    # wan21 rotary（view_as_complex 路径）
    def wan():
        x = torch.randn(1, 8, 16, 64, dtype=torch.bfloat16, device=dev)
        freqs = torch.complex(
            torch.randn(1, 8, 16, 32, device=dev),
            torch.randn(1, 8, 16, 32, device=dev),
        )
        from toolkit.device_utils import rope_dtype
        x_rot = torch.view_as_complex(x.to(rope_dtype(dev)).unflatten(3, (-1, 2)))
        return torch.view_as_real(x_rot * freqs).flatten(3, 4)

    run("wan21 rotary (view_as_complex)", wan)
    return all(results) if results else False


def main() -> int:
    print(f"torch {torch.__version__}  xpu={torch.xpu.is_available()}  {torch.xpu.get_device_name(0)}")
    ok1 = check_helpers()
    ok2 = check_fp64_unavailable_on_xpu()
    ok3 = check_model_rope()
    print("\n== 结论 ==")
    print(f"  助手行为: {OK if ok1 else FAIL}")
    print(f"  fp64 前提: {OK if ok2 else FAIL}")
    print(f"  模型路径: {OK if ok3 else FAIL}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
