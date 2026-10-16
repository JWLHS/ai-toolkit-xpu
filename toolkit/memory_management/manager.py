import ctypes
import gc

import torch
from .manager_modules import (
    LinearLayerMemoryManager,
    ConvLayerMemoryManager,
    OstrisLinearLayerMemoryManager,
    _DEVICE_STATE,
)
import random

LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "QLinear",
    'OstrisLinear',
    # omni_xpu_kernel 的 XPU int8/fp8 层：量化数据同样放在 buffer 里，
    # 不登记就会被当作"非目标模块"跳过 —— 权重根本不参与卸载（实测多占 4.2GB 显存）。
    "OmniInt8Linear",
    "OmniFp8Linear",
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
    "QConv2d",
]

UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
    "Conv3d",
    # Conv1d 也必须登记：层级卸载开启时模型不会整体 .to(device)，
    # 只有登记过的模块会跟着 memory_managed_to() 一起走。漏登记的话
    # Conv1d 权重会留在 CPU（state_dict 落地处），而输入在设备上 ->
    # "XPUBFloat16Type does not equal CPUBFloat16Type (xpu_convolution)"。
    # 实测于 Qwen3.5 线性注意力里的 depthwise conv1d（moss_image 的文本编码器）。
    # 这类短卷积权重极小（每层几十 KB），常驻设备代价可忽略。
    "Conv1d",
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]

# 有些模块把参数直接挂在自己身上（不在任何子模块里），典型例子是
# Qwen3.5 线性注意力的 GatedDeltaNet：A_log / dt_bias 是裸 Parameter，
# 而它的 Linear 子层照常需要被搬运。层级卸载开启时模型不会整体
# .to(device)，这类裸参数没人搬 -> "found at least two devices, xpu:0 and cpu"。
# 这里把命中的模块的**直接**参数/缓冲登记为 unmanaged（跟着模型整体走），
# 它的子模块仍按常规流程处理。
UNMANAGED_PARAM_MODULES_INCLUDES = ["GatedDeltaNet"]


class MemoryManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: list[torch.nn.Module] = []

    def memory_managed_to(self, *args, **kwargs):
        # first move all the unmanaged modules
        for module in self.unmanaged_modules:
            if isinstance(module, torch.Tensor):
                # Parameters and bare tensor buffers cannot move this way
                module.data = module.data.to(*args, **kwargs)
            else:
                module.to(*args, **kwargs)
        # check for a dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @classmethod
    def attach(
        cls,
        module: torch.nn.Module,
        device: torch.device,
        offload_percent: float = 1.0,
        ignore_modules: list[torch.nn.Module] = []
    ):
        if hasattr(module, "_memory_manager"):
            # already attached
            return

        module._memory_manager = cls(module, device)

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)

        # count ignore modules as processed — including their whole subtree: an
        # ignored module stays resident, so none of its children may get a
        # layer manager (a managed child would pin its weight back to cpu
        # behind the parent's back)
        modules_processed = [x for x in ignore_modules]
        for im in ignore_modules:
            if isinstance(im, torch.nn.Module):
                modules_processed.extend(im.modules())
        # attach to all modules
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if (
                    child_module.__class__.__name__ in LINEAR_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # linear; OstrisLinear bounces its quantized buffers instead
                        # of a dequantized weight (module.weight is a property)
                        if getattr(child_module, "is_ostris_quantized", False) or getattr(
                            child_module, "is_omni_quantized", False
                        ):
                            # 两者的共同点：module.weight 是 property，真正的权重在 buffer 里，
                            # 所以要走"搬 buffer"的管理器，而不是搬 dequantized 权重。
                            OstrisLinearLayerMemoryManager.attach(
                                child_module, module._memory_manager
                            )
                        else:
                            LinearLayerMemoryManager.attach(
                                child_module, module._memory_manager
                            )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                )
                    modules_processed.append(child_module)
                elif (
                    child_module.__class__.__name__ in CONV_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # conv
                        ConvLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                )
                            modules_processed.append(ara)
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged
                    module._memory_manager.unmanaged_modules.append(child_module)
                elif any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_PARAM_MODULES_INCLUDES
                ):
                    # 只登记"挂在模块上的裸参数/缓冲"，子模块照常参与搬运
                    for param in child_module.parameters(recurse=False):
                        module._memory_manager.unmanaged_modules.append(param)
                    for buf in child_module.buffers(recurse=False):
                        module._memory_manager.unmanaged_modules.append(buf)
                else:
                    continue

    @classmethod
    def detach(cls, module: torch.nn.Module):
        """
        Reverse of attach(). Moves unmanaged modules back to CPU, restores the
        original .to() and forward methods on all child layers, unpins CPU weight
        tensors, and clears the global CUDA device state.

        Call this before unloading/replacing a module that had attach() applied.
        """
        if not hasattr(module, "_memory_manager"):
            return

        for unmanaged in module._memory_manager.unmanaged_modules:
            try:
                if isinstance(unmanaged, torch.Tensor):
                    unmanaged.data = unmanaged.data.to('cpu')
                else:
                    unmanaged.to('cpu')
            except Exception:
                pass

        if hasattr(module, "_mm_to"):
            module.to = module._mm_to
            del module._mm_to

        del module._memory_manager

        for child in module.modules():
            lmm = getattr(child, "_layer_memory_manager", None)
            if lmm is None:
                continue

            original_forward = getattr(lmm, "_original_forward", None)
            if original_forward is not None:
                if hasattr(child, "ara_lora_ref"):
                    ara = child.ara_lora_ref()
                    if ara is not None:
                        ara.org_forward = original_forward
                else:
                    child.forward = original_forward

            for param_name in ("weight", "bias"):
                # read _parameters directly: OstrisLinear.weight is a property that
                # materializes a full dequantized weight on access
                param = child._parameters.get(param_name, None)
                if param is None or not isinstance(param, torch.nn.Parameter):
                    continue
                try:
                    if param.data.is_pinned():
                        object.__setattr__(
                            child,
                            param_name,
                            torch.nn.Parameter(
                                param.data.clone(),
                                requires_grad=param.requires_grad,
                            ),
                        )
                except Exception:
                    pass

            if getattr(child, "is_ostris_quantized", False):
                # move quantized buffers home and unpin them (clone drops pinning)
                for buf_name, buf in list(child._buffers.items()):
                    if buf is None:
                        continue
                    try:
                        if buf.device.type != "cpu":
                            buf = buf.to("cpu")
                        if buf.is_pinned():
                            buf = buf.clone()
                        child._buffers[buf_name] = buf
                    except Exception:
                        pass

            del child._layer_memory_manager
            if hasattr(child, "_memory_management_device"):
                del child._memory_management_device
            if hasattr(child, "_is_memory_managed"):
                del child._is_memory_managed

        keys_to_delete = [
            dev for dev in _DEVICE_STATE
            if isinstance(dev, torch.device) and dev.type == "cuda"
        ]
        for key in keys_to_delete:
            del _DEVICE_STATE[key]

        torch.cuda.empty_cache()

    @classmethod
    def free(cls, module: torch.nn.Module):
        """
        Detach memory management (if attached) and destroy the module's weights
        by moving them to the meta device.

        Unlike detach(), nothing is staged back to CPU first: to('meta') frees
        each storage from wherever it currently lives, so no transient host
        allocation is made for data that is about to be discarded, and pinned
        tensors are freed without the clone that unpinning requires. Freed
        pinned storages land in torch's caching host allocator, not the OS;
        call release_cached_memory() afterward to get the RSS back.
        """
        if hasattr(module, "_memory_manager"):
            if hasattr(module, "_mm_to"):
                module.to = module._mm_to
                del module._mm_to

            del module._memory_manager

            for child in module.modules():
                lmm = getattr(child, "_layer_memory_manager", None)
                if lmm is None:
                    continue

                original_forward = getattr(lmm, "_original_forward", None)
                if original_forward is not None:
                    if hasattr(child, "ara_lora_ref"):
                        ara = child.ara_lora_ref()
                        if ara is not None:
                            ara.org_forward = original_forward
                    else:
                        child.forward = original_forward

                del child._layer_memory_manager
                if hasattr(child, "_memory_management_device"):
                    del child._memory_management_device
                if hasattr(child, "_is_memory_managed"):
                    del child._is_memory_managed

            keys_to_delete = [
                dev for dev in _DEVICE_STATE
                if isinstance(dev, torch.device) and dev.type == "cuda"
            ]
            for key in keys_to_delete:
                del _DEVICE_STATE[key]

        # bypass any overridden/nopped-out .to() so the storages are actually freed
        torch.nn.Module.to(module, "meta")
        torch.cuda.empty_cache()

    @classmethod
    def release_cached_memory(cls):
        """
        Return freed memory to the OS. Freed pinned-host storages sit in
        torch's caching host allocator and freed pageable memory sits in
        glibc's arenas; neither shows up as reclaimed RSS without an
        explicit flush. Call after free()ing a large module.
        """
        gc.collect()
        torch.cuda.empty_cache()

        # torch's pinned-host cache; private API, name varies by torch version
        for fn_name in ("_accelerator_emptyHostCache", "_host_emptyCache"):
            fn = getattr(torch._C, fn_name, None)
            if fn is not None:
                try:
                    fn()
                    break
                except Exception:
                    pass

        # glibc keeps freed arenas mapped; CDLL(None) resolves malloc_trim in
        # the running process where glibc is present and fails cleanly on
        # macOS/musl
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except Exception:
            pass
