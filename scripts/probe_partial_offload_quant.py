"""Throwaway probe: does the af89a27 bug (partial offload + torchao quant) hit XPU?

The upstream fix says: for an *unmanaged* quantized linear (partial offloading),
`param.data = param.data.to(device)` only updates the outer metadata and leaves the
inner storage on the old device. Check that on this machine:

    python scripts/probe_partial_offload_quant.py
"""
import sys

import torch

DEV = "xpu" if torch.xpu.is_available() else "cpu"


def storage_device(t: torch.Tensor):
    """Mirror of the upstream helper: walk __tensor_flatten__ down to a leaf."""
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        return t.device
    for name in names:
        inner = getattr(t, name, None)
        if isinstance(inner, torch.Tensor):
            return storage_device(inner)
    return t.device


def main() -> int:
    print(f"device={DEV} torch={torch.__version__}")
    try:
        import torchao
        from torchao.quantization import quantize_
        # 0.17 moved to config objects; this is what toolkit/util/quantize.py uses
        from torchao.quantization.quant_api import Int8WeightOnlyConfig

        config = Int8WeightOnlyConfig()
        print(f"torchao={torchao.__version__}")
    except Exception as exc:
        print("torchao unavailable:", exc)
        return 2

    linear = torch.nn.Linear(128, 128, bias=False).to(DEV)
    quantize_(linear, config)
    weight = linear.weight
    text = hasattr(weight, "__tensor_flatten__")
    print(f"weight type   : {type(weight).__name__}  flatten={text}")
    print(f"weight.device : {weight.device}   storage={storage_device(weight)}")
    if not text:
        print("=> not a tensor subclass on this backend: the upstream fix cannot apply here")
        return 0

    # what the unmanaged path in manager.py used to do
    linear.weight.data = linear.weight.data.to("cpu")
    linear.weight.data = linear.weight.data.to(DEV)
    after = storage_device(linear.weight)
    print(f"after data= bounce -> outer={linear.weight.device} storage={after}")
    if after.type != torch.device(DEV).type:
        print("=> BUG REPRODUCES on this backend (inner storage left on cpu)")
    else:
        print("=> no bug on this backend (`.data =` moved the real storage too)")

    # does a forward/backward still work?
    x = torch.randn(4, 128, device=DEV, dtype=linear.weight.dtype if linear.weight.dtype.is_floating_point else torch.float32)
    try:
        out = linear(x)
        print("forward ok:", tuple(out.shape), out.dtype)
    except Exception as exc:
        print("forward FAILED:", type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
