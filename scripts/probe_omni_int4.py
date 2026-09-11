#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针：omni svdq 的 W4A4 int4 GEMM（torchao 打包格式）在 A770 上的可用性。

契约（见 omni_xpu_kernel/svdq/__init__.py）::

    onednn_int4_gemm_torchao(act[M,K] bf16, packed_u4[N,K/2] uint8,
                             zp_u8[num_groups,N] uint8, scales_f16[num_groups,N] f16)

用法: .venv\\Scripts\\python.exe scripts\\probe_omni_int4.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def timeit(fn, n: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> int:
    hr = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
    from omni_xpu_kernel import svdq

    hr("1) torchao 自带的 int4 量化（XPU 上大概率不可用，仅作对照）")
    try:
        from torchao.quantization.quant_api import Int4WeightOnlyConfig, quantize_

        lin = torch.nn.Linear(64, 64, bias=False, device="xpu", dtype=torch.bfloat16)
        quantize_(lin, Int4WeightOnlyConfig())
        print("  可用（意外）")
    except Exception as exc:  # noqa: BLE001
        print(f"  不可用：{type(exc).__name__}: {str(exc)[:80]}")
        print("  → 改为按它的格式自己打包（group=64，非对称 zero-point，[N, K/2] uint8）")

    def pack_int4_torchao(w: torch.Tensor, group: int = 64):
        """返回 (packed[N,K/2] u8, zp[ng,N] u8, scales[ng,N] f16, 反量化参考 w_ref)。"""
        n, k = w.shape
        assert k % group == 0
        ng = k // group
        wg = w.reshape(n, ng, group).float()
        mn = wg.amin(-1, keepdim=True)
        mx = wg.amax(-1, keepdim=True)
        scale = ((mx - mn) / 15.0).clamp_min(1e-8)
        zp = torch.round(-mn / scale).clamp(0, 15)
        q = torch.round(wg / scale + zp).clamp(0, 15).to(torch.uint8).reshape(n, k)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()      # 低半字节 = 偶数下标
        zp_u8 = zp.squeeze(-1).t().contiguous().to(torch.uint8)      # [ng, N]
        scales_f16 = scale.squeeze(-1).t().contiguous().to(torch.float16)
        w_ref = ((q.reshape(n, ng, group).float() - zp) * scale).reshape(n, k)
        return packed, zp_u8, scales_f16, w_ref

    lin = torch.nn.Linear(3072, 3072, bias=False, device="xpu", dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.normal_(0, 0.02)
    ref_w = lin.weight.detach().clone()
    qdata, zp_u8, sc_f16, w_ref = pack_int4_torchao(ref_w)
    print(f"  packed {tuple(qdata.shape)} {qdata.dtype} | zp {tuple(zp_u8.shape)} | scales {tuple(sc_f16.shape)}")

    hr("2) omni int4 GEMM 正确性 / 速度 / 显存")
    x = torch.randn(4096, 3072, device="xpu", dtype=torch.bfloat16)
    ref = F.linear(x, ref_w)
    try:
        out = svdq.onednn_int4_gemm_torchao(x, qdata.contiguous(), zp_u8.contiguous(), sc_f16)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item() * 100
        ref_self = F.linear(x, w_ref.to(torch.bfloat16))
        rel_self = ((out.float() - ref_self.float()).norm() / ref_self.float().norm()).item() * 100
        t_i4 = timeit(lambda: svdq.onednn_int4_gemm_torchao(x, qdata.contiguous(), zp_u8.contiguous(), sc_f16))
        t_bf = timeit(lambda: F.linear(x, ref_w))
        w_i4 = qdata.numel() / 1024 / 1024
        w_bf = ref_w.numel() * 2 / 1024 / 1024
        print(f"  与自身打包反量化参考的误差 : {rel_self:.3f}%  （≈0 说明 nibble 顺序/布局对得上）")
        print(f"  与原始 bf16 权重的误差     : {rel:.2f}%   （int8 1.23%，fp8 2.66%）")
        print(f"  速度       : int4 {t_i4:.2f}ms  vs  bf16 {t_bf:.2f}ms  ({t_bf / t_i4:.2f}x)")
        print(f"  权重显存   : int4 {w_i4:.0f} MiB  vs  bf16 {w_bf:.0f} MiB")
        deq = (out.float() - ref.float()).abs().max().item()
        print(f"  最大绝对差 : {deq:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  失败 ❌ {type(exc).__name__}: {str(exc)[:200]}")
        return 3
    return 0


def symmetric_w4a16(hr) -> int:
    """官方原生格式：W4A16 对称（激活保持 bf16，权重 signed int4 + per-group scale）。"""
    from omni_xpu_kernel import svdq

    hr("3) 官方原生 W4A16 对称格式")
    lin = torch.nn.Linear(3072, 3072, bias=False, device="xpu", dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.normal_(0, 0.02)
    w = lin.weight.detach().clone()
    n, k = w.shape
    group = 64
    ng = k // group
    wg = w.reshape(n, ng, group).float()
    scale = (wg.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-8)
    q = torch.round(wg / scale).clamp(-8, 7).to(torch.int16)
    u = (q & 0xF).to(torch.uint8).reshape(n, k)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()          # signed nibbles
    wscales = scale.squeeze(-1).t().contiguous().to(torch.bfloat16)  # [ng, N]
    w_ref = (q.reshape(n, ng, group).float() * scale).reshape(n, k)

    x = torch.randn(4096, k, device="xpu", dtype=torch.bfloat16)
    ref_bf16 = F.linear(x, w)
    try:
        out = svdq.onednn_int4_gemm(x, packed, wscales)
        pu4, sc16 = svdq.prepare_onednn_weights(packed, wscales)
        out_pre = svdq.onednn_int4_gemm_preconverted(x, pu4, sc16)
        ref_self = F.linear(x, w_ref.to(torch.bfloat16))
        rel_self = ((out.float() - ref_self.float()).norm() / ref_self.float().norm()).item() * 100
        rel_pre = ((out_pre.float() - ref_self.float()).norm() / ref_self.float().norm()).item() * 100
        rel_bf16 = ((out.float() - ref_bf16.float()).norm() / ref_bf16.float().norm()).item() * 100
        t_sym = timeit(lambda: svdq.onednn_int4_gemm(x, packed, wscales))
        t_pre = timeit(lambda: svdq.onednn_int4_gemm_preconverted(x, pu4, sc16))
        t_bf = timeit(lambda: F.linear(x, w))
        print(f"  自洽性（vs 自身反量化）: {rel_self:.3f}%（走 prepare 的: {rel_pre:.3f}%）")
        print(f"  与 bf16 权重的误差      : {rel_bf16:.2f}%   （非对称 W4A4 是 9.09%，int8 1.23%）")
        print(f"  速度                    : 对称 {t_sym:.2f}ms / 预转换 {t_pre:.2f}ms / bf16 {t_bf:.2f}ms")
        print(f"  权重显存                : int4 {packed.numel() / 1048576:.0f} MiB vs bf16 {w.numel() * 2 / 1048576:.0f} MiB")
    except Exception as exc:  # noqa: BLE001
        print(f"  失败 ❌ {type(exc).__name__}: {str(exc)[:200]}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
