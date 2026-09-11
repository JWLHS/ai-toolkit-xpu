"""XPU 专用的可选 INT8 后端：omni_xpu_kernel（Intel）+ 自动回退。

为什么需要它
------------
torchao 的 int8 权重在 XPU 上没有可用的 int8 matmul 内核，只能"反量化成 bf16 再算"；
`omni_xpu_kernel` 提供真正的 int8 内核（oneDNN s8 GEMM + ESIMD 融合），实测在 krea2
的真实形状上比反量化路径快 2～2.8 倍，权重显存也减半。

但它有两个限制，所以必须是"可选 + 可回退"：

1. 它是按 **架构** 编译的 wheel（A 系列 = dg2，B 系列 = bmg），装错架构必须拒绝；
2. 内核是**推理内核，没有 autograd**，训练要自己包一层 autograd.Function
   （前向走 int8 内核，反向用反量化后的 bf16 权重算 grad_input）。

识别规则（三者全过才启用）：

* 能 import `omni_xpu_kernel` 且 `int8.int8_linear` 存在；
* `__xpu_target__` 与当前设备的架构类别一致；
* wheel 构建时的 torch 主版本与运行时一致。

任何一条不过 → 由 `toolkit/util/quantize.py` 回退到 torchao int8，并打印一行原因。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import nn

from toolkit.print import print_acc

# 只在第一次做一次探测，避免每个 Linear 都 import/检查
_probe_cache: Optional[Tuple[bool, str, dict]] = None
_LOGGED: set = set()


def _device_class() -> str:
    """A 系列(dg2) / B 系列(bmg) —— 以驱动返回的设备名为准。"""
    try:
        name = torch.xpu.get_device_name(0)
    except Exception:
        return "unknown"
    low = name.lower()
    # A770/A750/A580/A380 等 driver 名形如 "Intel(R) Arc(TM) A770 Graphics"
    if "arc(tm) a" in low or "arc a" in low or "dg2" in low or "graphics [0x56" in low:
        return "dg2"
    if "arc(tm) b" in low or "arc b" in low or "arc pro b" in low or "bmg" in low or "battlemage" in low:
        return "bmg"
    return "unknown"


def probe(force: bool = False) -> Tuple[bool, str, dict]:
    """返回 (是否可用, 原因, 信息)。可反复调用，结果会缓存。"""
    global _probe_cache
    if _probe_cache is not None and not force:
        return _probe_cache

    info: dict = {}
    if not torch.xpu.is_available():
        _probe_cache = (False, "没有可用的 XPU 设备", info)
        return _probe_cache

    try:
        import omni_xpu_kernel as oak
        from omni_xpu_kernel import int8 as omni_int8
    except Exception as exc:  # noqa: BLE001 - 任何导入问题都算"不可用"
        _probe_cache = (False, f"未安装 omni_xpu_kernel（{type(exc).__name__}）", info)
        return _probe_cache

    target = str(getattr(oak, "__xpu_target__", "unknown"))
    build_torch = str(getattr(oak, "__torch_version__", "?"))
    version = str(getattr(oak, "__version__", "?"))
    run_torch = torch.__version__.split("+")[0]
    device_class = _device_class()
    info.update(
        version=version,
        target=target,
        build_torch=build_torch,
        run_torch=run_torch,
        device_class=device_class,
    )

    if target not in ("dg2", "bmg"):
        _probe_cache = (False, f"wheel 构建目标未知：{target}", info)
        return _probe_cache
    if device_class == "unknown":
        _probe_cache = (False, "无法判断当前显卡属于 A 系列还是 B 系列", info)
        return _probe_cache
    if target != device_class:
        _probe_cache = (
            False,
            f"wheel 是 {target} 版，当前设备是 {device_class} 系列（装错了架构）",
            info,
        )
        return _probe_cache
    if not build_torch.startswith(run_torch):
        _probe_cache = (
            False,
            f"wheel 按 torch {build_torch} 编译，当前 torch {run_torch}",
            info,
        )
        return _probe_cache
    for fn in ("int8_linear", "quantize_int8_rowwise", "dequantize_int8_simple_dtype"):
        if not hasattr(omni_int8, fn):
            _probe_cache = (False, f"omni int8 API 缺少 {fn}()", info)
            return _probe_cache

    _probe_cache = (True, "", info)
    return _probe_cache


def available() -> bool:
    return probe()[0]


def unavailable_reason() -> str:
    return probe()[1]


def log_once(key: str, message: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        print_acc(message)


class _OmniInt8LinearFn(torch.autograd.Function):
    """前向走 int8 内核；反向只需要 grad_input（底模权重冻结）。

    反向用反量化后的 bf16 权重算 grad_input —— 底模权重在 LoRA 训练里是冻结的，
    不需要 weight 梯度，所以这里只把临时 bf16 权重用于一次 matmul，用完即释放。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor], out_dtype: torch.dtype):
        from omni_xpu_kernel import int8 as omni_int8

        # 只把「源张量」（可能还在 CPU）存进 ctx：GPU 搬运副本不进 ctx，否则一次前向里
        # 每层的 GPU 副本都会挂到各自反向为止，卸载等于白做。
        # 反向只需要 x 的形状/精度 —— 不能存 x 本身，否则开了梯度检查点也白搭
        # （每层激活会被我们吊住，显存峰值直接被抬高）。
        ctx.save_for_backward(qweight, scale)
        ctx.x_shape = x.shape
        ctx.out_dtype = out_dtype
        q = _stage(qweight, x.device)
        s = _stage(scale, x.device)
        b = None if bias is None else _stage(bias, x.device)
        return omni_int8.int8_linear(x, q, s, bias=b, out_dtype=out_dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        from omni_xpu_kernel import int8 as omni_int8

        qweight, scale = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            q = _stage(qweight, grad_out.device)
            s = _stage(scale, grad_out.device)
            weight = omni_int8.dequantize_int8_simple_dtype(q, s, out_dtype=torch.bfloat16)
            flat = grad_out.reshape(-1, grad_out.shape[-1]).to(torch.bfloat16)
            grad_x = (flat @ weight).reshape(ctx.x_shape).to(ctx.out_dtype)
            del weight, flat
        return grad_x, None, None, None, None


def _stage(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """把张量放到目标设备（已在目标设备则原样返回；跨设备才拷贝）。"""
    if t.device == device:
        return t
    # 同步拷贝：异步 H2D 的目标张量会挂到复制完成，逐层累积会抬高峰值
    return t.to(device, non_blocking=False)


class OmniInt8Linear(nn.Linear):
    """等价于 nn.Linear，但权重以 int8 保存、前向走 omni 内核。

    刻意继承 nn.Linear：上游有一大堆代码用 `isinstance(m, nn.Linear)` 找底模层
    （LoRA 注入、量化、注意力替换…），不继承就会静默漏层。
    dense weight 被删除，量化数据放在 qweight/scale 两个 buffer 里，
    `.weight` 属性按需反量化（与 OstrisLinear 的做法一致）。
    """

    is_omni_quantized = True

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        out_dtype: torch.dtype,
    ):
        in_features = int(qweight.shape[1])
        out_features = int(qweight.shape[0])
        super().__init__(in_features, out_features, bias=False, device="meta", dtype=torch.bfloat16)
        del self._parameters["weight"]
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        self.out_dtype = out_dtype

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        from omni_xpu_kernel import int8 as omni_int8

        if self.qweight.device.type != "xpu":
            # CPU（层级卸载把 buffer 搬走了）时不能调内核：会报 "input must be on XPU"。
            # 合并/导出/存档都会走到这里，用纯 torch 算。
            return (self.qweight.to(torch.float32) * self.scale).to(self.out_dtype)
        return omni_int8.dequantize_int8_simple_dtype(self.qweight, self.scale, out_dtype=self.out_dtype)

    @property
    def weight(self):
        # 返回惰性权重：上游有代码只是 isinstance(getattr(m,'weight'),Parameter) 的检查，
        # 直接反量化会在 buffer 还在 CPU 时炸掉（"input must be on XPU"），也会白白算一遍。
        return _OmniLazyWeight(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        # buffer 可能还在 CPU（层级卸载）：搬运交给 Function 内部，且**不**把 GPU 副本
        # 留在 ctx 里，这样峰值只跟"同时在场的那一层"有关，而不是层数之和。
        return _OmniInt8LinearFn.apply(x, self.qweight, self.scale, self.bias, self.out_dtype)

    def requantize_(self, fp_weight: torch.Tensor) -> None:
        """把新的浮点权重量化回 int8（合并/重置流程用）。"""
        from omni_xpu_kernel import int8 as omni_int8

        weight = fp_weight.detach().to(torch.bfloat16).contiguous()
        qweight, scale = omni_int8.quantize_int8_rowwise(weight)
        self.qweight = qweight
        self.scale = scale

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # 与 OstrisLinear 一致：不把整块反量化权重塞进 state_dict（大模型会 OOM）
        destination[prefix + "weight"] = _OmniLazyWeight(self)
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()


class _OmniLazyWeight(torch.Tensor):
    """Shape/dtype/device 正确、但不持数据的权重替身，真正参与运算时才反量化。"""

    @staticmethod
    def __new__(cls, module: "OmniInt8Linear"):
        buf = next((b for b in module._buffers.values() if b is not None), None)
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            (module.out_features, module.in_features),
            dtype=module.out_dtype,
            device=buf.device if buf is not None else torch.device("cpu"),
            requires_grad=False,
        )
        r._omni_module = module
        # 供 toolkit.util.quantize.is_quantized_tensor / dequantize_if_quantized 识别
        r._is_omni_weight = True
        return r

    def dequantize(self) -> torch.Tensor:
        return self._omni_module.dequantize_weight()

    def __repr__(self):
        return f"OmniLazyWeight(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from torch.utils._pytree import tree_map

        def unwrap(t):
            return t._omni_module.dequantize_weight() if isinstance(t, cls) else t

        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {}))


@torch.no_grad()
def quantize_weight(
    weight: torch.Tensor, chunk_rows: int = 4096
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按行（每个输出通道一个 scale）量化权重。

    分块做：内核每条路径都会产生若干个与权重同尺寸的 bf16 临时张量，
    整层一次算的话单层峰值 ~1.3GB；Windows 的分配器不会把这段还给系统，
    几十层下来 RSS 就停在 40GB+（实测：活张量只剩 3 个，RSS 仍是 41.4GB）。
    分块后每块临时量降到 ~100MB 级。
    """
    from omni_xpu_kernel import int8 as omni_int8

    w = weight.detach()
    if w.dtype != torch.bfloat16:
        w = w.to(torch.bfloat16)
    rows = int(w.shape[0])
    if rows <= chunk_rows:
        return omni_int8.quantize_int8_rowwise(w.contiguous())

    qweight = torch.empty_like(w, dtype=torch.int8)
    scales = []
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        q, s = omni_int8.quantize_int8_rowwise(w[start:stop].contiguous())
        qweight[start:stop] = q
        scales.append(s)
        del q, s
    return qweight, torch.cat(scales, dim=0)


def convert_linear_to_omni(model: nn.Module, name: str, module: nn.Linear) -> bool:
    """把 model 里名为 name 的 Linear 换成 OmniInt8Linear。"""
    if not available():
        return False
    if module.weight is None or module.weight.ndim != 2:
        return False
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    attr = name.rsplit(".", 1)[-1]
    out_dtype = module.weight.dtype
    qweight, scale = quantize_weight(module.weight)
    bias = None if module.bias is None else module.bias.detach().clone()
    setattr(parent, attr, OmniInt8Linear(qweight, scale, bias, out_dtype))
    # 关键：把原模块的 bf16 权重丢掉。不丢的话它在内存里和 int8 副本同时存在，
    # 实测训练进程会多占 ~20GB RAM（torchao 路径量完就回落到 20GB，我们不会回落）。
    module._parameters.pop("weight", None)
    return True


def is_omni_linear(module: nn.Module) -> bool:
    return isinstance(module, OmniInt8Linear)


def enabled_by_env() -> bool:
    """AI_TOOLKIT_XPU_INT8=off 可以强制关掉这个后端（排查用）。"""
    return os.environ.get("AI_TOOLKIT_XPU_INT8", "on").lower() not in ("0", "off", "false")


def debug_memory_report(tag: str = "", min_mb: int = 100) -> None:
    """AI_TOOLKIT_MEM_DEBUG=1 时，打印大张量的分布和持有者（定位内存问题用）。"""
    import gc

    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1e9
    except Exception:  # noqa: BLE001
        rss = -1.0

    tensors = [o for o in gc.get_objects() if isinstance(o, torch.Tensor)]
    big = [t for t in tensors if t.numel() * t.element_size() >= min_mb * 1024 * 1024]
    total = sum(t.numel() * t.element_size() for t in big) / 1e9
    print_acc(f"[MEMDBG{(' ' + tag) if tag else ''}] RSS={rss:.1f} GB | 大张量 {len(big)} 个，合计 {total:.1f} GB")

    agg: dict = {}
    for t in big:
        key = (str(t.dtype), str(t.device))
        n, b = agg.get(key, (0, 0))
        agg[key] = (n + 1, b + t.numel() * t.element_size())
    for key, (n, b) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:6]:
        print_acc(f"[MEMDBG]   {key[0]:16} {key[1]:8} : {n:5} 个, {b / 1e9:5.1f} GB")

    if not big:
        return
    biggest = max(big, key=lambda t: t.numel() * t.element_size())
    holders = []
    for ref in gc.get_referrers(biggest)[:12]:
        if isinstance(ref, dict):
            for k, v in list(ref.items())[:8]:
                if v is biggest:
                    holders.append(f"dict[{k!r}]")
        elif isinstance(ref, (list, tuple)):
            holders.append(f"{type(ref).__name__}[len={len(ref)}]")
        else:
            holders.append(type(ref).__name__)
    print_acc(
        f"[MEMDBG] 最大张量 {tuple(biggest.shape)} {biggest.dtype} {biggest.device} "
        f"({biggest.numel() * biggest.element_size() / 1e9:.1f} GB) 持有者: {holders[:8]}"
    )


def release_process_memory(tag: str = "") -> None:
    """把量化期间产生的、被分配器"吃住不放"的内存还给系统（Windows）。

    实测：量化完 RSS 41.4GB，但活着的张量只有 3 个（1.7GB）——多出来的是
    torch/CRT 分配器的历史水位，Windows 不会主动归还。这里做两件事：
      1. gc + XPU 缓存回收；
      2. SetProcessWorkingSetSizeEx(-1,-1)：让系统回收本进程的工作集
         （页仍在提交空间里，下次访问会重新调入，但不再占用物理内存）。
    量化之后原始 bf16 权重已不需要，回收是安全的。
    """
    import gc

    gc.collect()
    try:
        torch.xpu.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # 必须声明 restype/argtypes：否则句柄会被 ctypes 当成 32 位截断，
        # 调用静默失败（这就是之前"回收了但 RSS 不降"的原因）。
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = kernel32.GetCurrentProcess()

        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.EmptyWorkingSet.restype = wintypes.BOOL
        psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
        ok = bool(psapi.EmptyWorkingSet(handle))

        kernel32.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL
        kernel32.SetProcessWorkingSetSizeEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_size_t,
            ctypes.c_size_t,
            wintypes.DWORD,
        ]
        kernel32.SetProcessWorkingSetSizeEx(
            handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1), 0
        )
        if os.environ.get("AI_TOOLKIT_MEM_DEBUG") == "1":
            import psutil

            rss = psutil.Process().memory_info().rss / 1e9
            print_acc(f"[MEMDBG] 回收工作集 {tag}: ok={ok}, 之后 RSS={rss:.1f} GB")
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("AI_TOOLKIT_MEM_DEBUG") == "1":
            print_acc(f"[MEMDBG] 回收工作集失败: {exc}")


# ---------------------------------------------------------------------------
# FP8 W8A16（可选）
#
# A 系列没有原生 FP8 算力，但 omni 的 oneDNN W8A16 路径是"fp8 权重存储 + bf16 计算"，
# 在 A770 上实测可用：约 1.2~1.3x bf16，误差 0.24%（int8 是 1.5%），权重显存减半。
# 代价：内核有形状约束（M<=8192、K>=2048、N<=4096），不合格的层/调用要回退。
# ---------------------------------------------------------------------------

FP8_MAX_M = 8192
FP8_MIN_K = 2048
FP8_MAX_N = 4096


def fp8_supported() -> bool:
    ok, _reason, _info = probe()
    if not ok:
        return False
    try:
        from omni_xpu_kernel import linear as omni_linear

        return hasattr(omni_linear, "onednn_w8a16_fp8")
    except Exception:  # noqa: BLE001
        return False


def fp8_shape_ok(in_features: int, out_features: int) -> bool:
    """转换期能判断的部分（M 在运行期才知道）。"""
    return in_features >= FP8_MIN_K and out_features <= FP8_MAX_N


class _OmniFp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_u8, scale_u8, bias, use_kernel: bool):
        from omni_xpu_kernel import linear as omni_linear

        # 同样只存源张量（uint8 缓冲，可能在 CPU），GPU 副本不进 ctx
        ctx.save_for_backward(x, w_u8, scale_u8)
        ctx.use_kernel = use_kernel
        w_fp8 = _stage(w_u8, x.device).view(torch.float8_e4m3fn)
        scale = _stage(scale_u8, x.device).view(torch.float32)
        bias = None if bias is None else _stage(bias, x.device)
        if use_kernel:
            # 内核只接受 2D [M, K]；训练里传进来的是 [B, L, K]
            shape = x.shape
            flat = x.reshape(-1, shape[-1]).contiguous()
            out = omni_linear.onednn_w8a16_fp8(flat, w_fp8, scale, bias)
            return out.reshape(*shape[:-1], out.shape[-1])
        weight = (w_fp8.float() * scale.unsqueeze(1)).to(torch.bfloat16)
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, w_u8, scale_u8 = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            w_fp8 = _stage(w_u8, grad_out.device).view(torch.float8_e4m3fn)
            scale = _stage(scale_u8, grad_out.device).view(torch.float32)
            weight = (w_fp8.float() * scale.unsqueeze(1)).to(torch.bfloat16)
            # weight 是 [N, K]，grad_x[.., K] = grad_out[.., N] @ weight
            grad_x = torch.matmul(grad_out.to(torch.bfloat16), weight).to(x.dtype)
            del weight
        return grad_x, None, None, None, None


class OmniFp8Linear(nn.Linear):
    """FP8(W8A16) 线性层：权重以 fp8_e4m3 保存，形状不合格时回退 bf16。"""

    is_omni_quantized = True

    def __init__(self, w_fp8, scale, bias, out_dtype):
        super().__init__(
            int(w_fp8.shape[1]), int(w_fp8.shape[0]), bias=False, device="meta", dtype=torch.bfloat16
        )
        del self._parameters["weight"]
        # 用 uint8 存 fp8 数据：模块级 .to(bfloat16) 之类的 dtype 转换会把 float8 buffer
        # 一起转掉，导致内核报 "weight must have dtype float8_e4m3fn"
        self.register_buffer("w_u8", w_fp8.view(torch.uint8))
        # scale 同样用 uint8 存（float32 也会被模块级 dtype 转换改掉）
        self.register_buffer("scale_u8", scale.view(torch.uint8))
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        self.out_dtype = out_dtype

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        scale = self.scale_u8.view(torch.float32)
        return (self.w_u8.view(torch.float8_e4m3fn).float() * scale.unsqueeze(1)).to(self.out_dtype)

    @property
    def weight(self):
        return _OmniLazyWeight(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        device = x.device
        rows = x.numel() // x.shape[-1] if x.ndim > 1 else 1
        use_kernel = rows <= FP8_MAX_M
        return _OmniFp8LinearFn.apply(x, self.w_u8, self.scale_u8, self.bias, use_kernel)

    @torch.no_grad()
    def requantize_(self, fp_weight: torch.Tensor) -> None:
        w_fp8, scale = quantize_weight_fp8(fp_weight)
        self.w_u8 = w_fp8.view(torch.uint8)
        self.scale_u8 = scale.view(torch.uint8)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "weight"] = _OmniLazyWeight(self)
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()


@torch.no_grad()
def quantize_weight_fp8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """按行量化到 fp8_e4m3，scale 必须是 float32 1D（内核要求）。"""
    w = weight.detach().to(torch.bfloat16)
    scale = (w.abs().amax(dim=1).clamp_min(1e-8) / 448.0).to(torch.float32)
    w_fp8 = (w / scale.unsqueeze(1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return w_fp8.view(torch.uint8), scale


def convert_linear_to_omni_fp8(model: nn.Module, name: str, module: nn.Linear) -> bool:
    """把 Linear 换成 OmniFp8Linear；形状不合格的层不转换（保持原精度）。"""
    if not fp8_supported() or module.weight is None or module.weight.ndim != 2:
        return False
    out_features, in_features = module.weight.shape
    if not fp8_shape_ok(in_features, out_features):
        return False
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    attr = name.rsplit(".", 1)[-1]
    w_fp8, scale = quantize_weight_fp8(module.weight)
    bias = None if module.bias is None else module.bias.detach().clone()
    setattr(parent, attr, OmniFp8Linear(w_fp8, scale, bias, module.weight.dtype))
    module._parameters.pop("weight", None)
    return True
