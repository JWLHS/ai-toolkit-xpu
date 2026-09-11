#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注意力/归一化的正确性与边界检查，外加 fp8 W8A16 复测。"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> int:
    from omni_xpu_kernel import linear, sdp

    hr(f"1) omni sdp 正确性（OMNI_ATTN_BACKEND={os.environ.get('OMNI_ATTN_BACKEND')}）")
    b, h, l, d = 1, 8, 1024, 128
    q = torch.randn(b, h, l, d, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(b, h, l, d, device="xpu", dtype=torch.bfloat16)
    v = torch.randn(b, h, l, d, device="xpu", dtype=torch.bfloat16)
    ref = F.scaled_dot_product_attention(q, k, v)
    got = sdp.sdp(q, k, v)
    diff = (got.float() - ref.float()).abs().max().item()
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item() * 100
    print(f"   BHLD 布局  : 形状 {tuple(got.shape)}  最大误差 {diff:.4f}  相对误差 {rel:.2f}%")

    hr("2) GQA（H_q != H_kv）与 causal 支持")
    q2 = torch.randn(1, 16, 512, 128, device="xpu", dtype=torch.bfloat16)
    k2 = torch.randn(1, 4, 512, 128, device="xpu", dtype=torch.bfloat16)
    v2 = torch.randn(1, 4, 512, 128, device="xpu", dtype=torch.bfloat16)
    try:
        out2 = sdp.sdp(q2, k2, v2)
        ref2 = F.scaled_dot_product_attention(q2, k2, v2)
        rel2 = ((out2.float() - ref2.float()).norm() / ref2.float().norm()).item() * 100
        print(f"   GQA 16:4   : 可用 ✅ 相对误差 {rel2:.2f}%")
    except Exception as exc:  # noqa: BLE001
        print(f"   GQA 16:4   : 不支持 ❌ {type(exc).__name__}: {str(exc)[:80]}")

    hr("3) FP8 W8A16 复测（scale 用 1D）")
    x = torch.randn(4096, 3072, device="xpu", dtype=torch.bfloat16)
    w = torch.randn(12288, 3072, device="xpu", dtype=torch.bfloat16) * 0.02
    scale = (w.abs().amax(dim=1).clamp_min(1e-8) / 448.0)
    w_fp8 = (w / scale.unsqueeze(1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    try:
        out = linear.onednn_w8a16_fp8(x, w_fp8, scale)
        ref = F.linear(x, (w_fp8.float() * scale.unsqueeze(1)).to(torch.bfloat16))
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item() * 100
        print(f"   可用 ✅ 形状 {tuple(out.shape)} 与反量化参考的相对误差 {rel:.2f}%")
        print(f"   权重显存：fp8 {w_fp8.numel() / 1024 / 1024:.0f} MiB vs bf16 {w.numel() * 2 / 1024 / 1024:.0f} MiB")
    except Exception as exc:  # noqa: BLE001
        print(f"   失败 ❌ {type(exc).__name__}: {str(exc)[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
