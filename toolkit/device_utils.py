import torch
import gc
import os

def get_device() -> torch.device:
    """
    Returns the best available device.
    Prioritizes XPU, then CUDA, then CPU.
    """
    if torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def is_xpu_available() -> bool:
    return torch.xpu.is_available()

def is_cuda_available() -> bool:
    return torch.cuda.is_available()

def empty_cache():
    """
    Empties the cache for the current device.
    """
    gc.collect()
    if is_xpu_available():
        if per_step_empty_cache_enabled():
            torch.xpu.empty_cache()
    elif is_cuda_available():
        torch.cuda.empty_cache()


def _torch_version_at_least(major: int, minor: int) -> bool:
    try:
        parts = torch.__version__.split("+")[0].split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


def per_step_empty_cache_enabled() -> bool:
    """XPU：训练循环里每步调用 torch.xpu.empty_cache() 是否可以开。

    torch 2.14 的 XPU 缓存分配器和 Intel 驱动在这里有缺陷：**释放过大张量之后**
    调用 torch.xpu.empty_cache() 会把 ze_intel_gpu64.dll 打崩（0xC0000005），
    最小复现（A770，两个驱动版本 32.0.101.8991 / 32.0.101.8860 都复现）::

        conv 反向 -> del 张量 + torch.xpu.empty_cache() -> attention   # 3/3 崩
        同样序列去掉 del + empty_cache                                # 0/3 通过

    所以 **torch >= 2.14 默认关掉**这个每步回收（2.13 保持原样：它靠这次回收把
    reserved pool 从 18GB 压回实际用量，关掉会更快撑爆显存）。
    需要时可用环境变量强制覆盖：AITK_XPU_EMPTY_CACHE=1 / 0。
    """
    override = os.environ.get("AITK_XPU_EMPTY_CACHE")
    if override is not None:
        return override.strip() != "0"
    return not _torch_version_at_least(2, 14)

def manual_seed(seed: int):
    """
    Sets the seed for the current device.
    """
    torch.manual_seed(seed)
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    elif is_cuda_available():
        torch.cuda.manual_seed(seed)

def get_device_name() -> str:
    if is_xpu_available():
        return "xpu"
    elif is_cuda_available():
        return "cuda"
    else:
        return "cpu"

def rope_dtype(device=None) -> torch.dtype:
    """dtype for RoPE / frequency tables.

    XPU (and Apple MPS) do not implement float64, so *device-side* fp64 math
    raises. Use fp32 there and keep fp64 elsewhere (CPU/CUDA) for the extra
    precision the reference implementations ask for.

    Pass the target device when it is known; without one the current
    accelerator decides.
    """
    if device is not None:
        dev_type = getattr(device, "type", None) or str(device)
        return torch.float32 if dev_type in ("xpu", "mps") else torch.float64
    if is_xpu_available() or torch.backends.mps.is_available():
        return torch.float32
    return torch.float64

def adjust_dtype_for_device(dtype: torch.dtype, device) -> torch.dtype:
    """Return a device-safe dtype: fp64 is unavailable on XPU / MPS -> fp32."""
    if dtype == torch.float64:
        dev_type = getattr(device, "type", None) or str(device)
        if dev_type in ("xpu", "mps"):
            return torch.float32
    return dtype

def autocast():
    if is_xpu_available():
        return torch.autocast(device_type="xpu")
    elif is_cuda_available():
        return torch.autocast(device_type="cuda")
    else:
        # Fallback to cpu or simple context manager
        return torch.autocast(device_type="cpu")
