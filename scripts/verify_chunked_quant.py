#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证分块量化：正确性 + RSS 增长（对比整层一次量化）。

用法: .venv\\Scripts\\python.exe scripts\\verify_chunked_quant.py
"""

from __future__ import annotations

import gc

import psutil
import torch
from omni_xpu_kernel import int8 as omni_int8

from toolkit.util import omni_int8 as O

LAYERS = 20
SHAPE = (12288, 3072)  # ≈75MB bf16，krea2 常见 FFN 尺寸


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def run(whole: bool) -> float:
    gc.collect()
    base = rss_gb()
    for i in range(LAYERS):
        w = torch.randn(*SHAPE, dtype=torch.bfloat16) * 0.02
        if whole:
            q, s = omni_int8.quantize_int8_rowwise(w.contiguous())
        else:
            q, s = O.quantize_weight(w)
        del w, q, s
    gc.collect()
    return rss_gb() - base


def main() -> int:
    print(f"层数={LAYERS} 形状={SHAPE}  (每层 {SHAPE[0] * SHAPE[1] * 2 / 1e6:.0f} MB bf16)")

    # 正确性：整层 vs 分块，逐行 scale 应完全一致
    w = torch.randn(*SHAPE, dtype=torch.bfloat16) * 0.02
    q1, s1 = omni_int8.quantize_int8_rowwise(w.contiguous())
    q2, s2 = O.quantize_weight(w)
    print(
        f"分块 vs 整层: qweight 相同={torch.equal(q1, q2)}  scale 最大差={(s1 - s2).abs().max().item():.3e}"
    )
    del w, q1, s1, q2, s2

    d_whole = run(whole=True)
    d_chunk = run(whole=False)
    print(f"整层一次量化 : RSS 增长 {d_whole:5.2f} GB")
    print(f"分块量化     : RSS 增长 {d_chunk:5.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
