#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XPU 训练算子自检：确认当前 torch/驱动组合能不能跑训练（不只是推理）。

背景：torch 2.14.0+xpu + 本机 Intel 驱动在 "conv 反向 -> attention 前向" 这类
训练型算子序列上会崩在 `ze_intel_gpu64.dll`（0xC0000005，偏移 0x404ed4）。
该崩溃与 ai-toolkit 代码无关，纯 torch 就能复现，因此这个脚本故意**只依赖 torch**。

用法::

    python scripts/xpu_train_ops_check.py          # 全流程
    python scripts/xpu_train_ops_check.py --quick  # 只跑最容易触发的那一组

判定：
  * 打印 `ALL OK` 并以 0 退出 -> 这套 torch + 驱动可以训练；
  * 进程直接消失（无 traceback、shell 退出码 -1073741819 / 0xC0000005）-> 驱动级崩溃，
    换 torch 版本（本项目用 2.13.0+xpu）或更新显卡驱动。
"""

from __future__ import annotations

import argparse
import sys

import torch

DEV = "xpu"


def mark(msg: str) -> None:
    print(f"[mark] {msg}", flush=True)
    torch.xpu.synchronize()


def matmul_bwd() -> None:
    w = torch.randn(1024, 4096, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    x = torch.randn(8, 1024, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    mark("alloc matmul")
    (x @ w).sum().backward()
    mark("matmul fwd+bwd")
    del w, x
    torch.xpu.empty_cache()


def conv_gn_bwd() -> None:
    conv = torch.nn.Conv2d(320, 320, 3, padding=1).to(DEV).to(torch.bfloat16)
    gn = torch.nn.GroupNorm(32, 320).to(DEV).to(torch.bfloat16)
    inp = torch.randn(1, 320, 96, 96, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    mark("alloc conv")
    out = gn(conv(inp))
    mark("conv+gn fwd")
    out.float().pow(2).mean().backward()
    mark("conv+gn bwd")
    del conv, gn, inp, out
    torch.xpu.empty_cache()


def sdpa_bwd() -> None:
    shape = (2, 8, 512, 64)
    q, k, v = (
        torch.randn(*shape, device=DEV, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    )
    mark("alloc qkv")
    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    mark("sdpa fwd")
    attn.float().pow(2).mean().backward()
    mark("sdpa bwd")
    del q, k, v, attn
    torch.xpu.empty_cache()


def adam_steps(count: int = 3) -> None:
    lin = torch.nn.Linear(1024, 1024, bias=False).to(DEV).to(torch.bfloat16)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-4)
    for i in range(count):
        loss = lin(torch.randn(4, 1024, device=DEV, dtype=torch.bfloat16)).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        mark(f"adam step {i} loss={loss.item():.5f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="只跑 conv 反向 + attention 这一组")
    args = parser.parse_args()

    print(f"python {sys.version.split()[0]} | torch {torch.__version__}", flush=True)
    if not torch.xpu.is_available():
        print("XPU 不可用，退出", file=sys.stderr)
        return 2
    print(f"device {torch.xpu.get_device_name(0)}", flush=True)
    torch.manual_seed(0)

    if args.quick:
        conv_gn_bwd()
        sdpa_bwd()
    else:
        matmul_bwd()
        conv_gn_bwd()
        sdpa_bwd()
        adam_steps()

    mark("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
