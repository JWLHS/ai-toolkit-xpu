#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 omni int8 作为训练后端是否可用（前向精度、反向梯度、Linear 兼容性）。

用法: .venv\\Scripts\\python.exe scripts\\verify_omni_training_path.py
"""

from __future__ import annotations

import torch

from toolkit.util import omni_int8


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    hr("0) 后端探测")
    ok, reason, info = omni_int8.probe(force=True)
    for k, v in info.items():
        print(f"  {k:14}: {v}")
    print(f"  可用          : {ok}  {reason}")
    if not ok:
        return 2

    hr("1) 结构与 Linear 兼容性")
    from torch import nn

    linear = nn.Linear(512, 384, bias=True, device="xpu", dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.normal_(0, 0.02)
        linear.bias.normal_(0, 0.02)
    ref_w = linear.weight.detach().clone()
    ref_b = linear.bias.detach().clone()

    holder = nn.Sequential(nn.Identity(), linear)
    assert omni_int8.convert_linear_to_omni(holder, "1", linear)
    converted = holder[1]
    print(f"  类型            : {type(converted).__name__}")
    print(f"  isinstance Linear: {isinstance(converted, nn.Linear)}")
    print(f"  weight 形状/类型 : {tuple(converted.weight.shape)} / {converted.weight.dtype}")
    print(f"  参数量          : {sum(p.numel() for p in converted.parameters())} (只有 bias)")
    print(f"  缓冲 qweight    : {tuple(converted.qweight.shape)} {converted.qweight.dtype}")

    hr("2) 前向精度 vs bf16 参考")
    x = torch.randn(64, 512, device="xpu", dtype=torch.bfloat16)
    with torch.no_grad():
        got = converted(x)
        ref = torch.nn.functional.linear(x, ref_w, ref_b)
    diff = (got.float() - ref.float())
    print(f"  最大绝对误差    : {diff.abs().max().item():.4f}")
    print(f"  相对误差(Frobenius): {(diff.norm() / ref.float().norm()).item() * 100:.2f}%")

    hr("3) 反向梯度（grad_input）")
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    out1 = converted(x1).float().square().sum()
    out1.backward()
    out2 = torch.nn.functional.linear(x2, ref_w, ref_b).float().square().sum()
    out2.backward()
    g1, g2 = x1.grad.float(), x2.grad.float()
    print(f"  梯度相对误差    : {((g1 - g2).norm() / g2.norm()).item() * 100:.2f}%")
    print(f"  梯度范数        : omni={g1.norm().item():.2f}  bf16={g2.norm().item():.2f}")
    print("  → 反向走的是反量化 bf16 权重，误差只来自 int8 量化本身，量级与误差 2 一致即正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
