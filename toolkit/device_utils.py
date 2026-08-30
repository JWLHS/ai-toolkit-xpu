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
        torch.xpu.empty_cache()
    elif is_cuda_available():
        torch.cuda.empty_cache()

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

def autocast():
    if is_xpu_available():
        return torch.autocast(device_type="xpu")
    elif is_cuda_available():
        return torch.autocast(device_type="cuda")
    else:
        # Fallback to cpu or simple context manager
        return torch.autocast(device_type="cpu")
