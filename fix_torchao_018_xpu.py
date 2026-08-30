"""torchao 0.17/0.18 在 XPU 上的量化兼容补丁（自注册缺失算子）。

背景
----
torchao 0.17 起逐步引入新的量张量子类（Float8Tensor，0.18 再加入 Int8Tensor），
改用 ``__torch_dispatch__`` 表驱动算子分发，但没有补齐这两类的全部算子注册，
导致 ai-toolkit 的量化流程（先量化容器、再逐个重量化子层，
内部会调用 choose_qparams_affine -> input.view(...) / torch.abs(...)）失败：

  * Int8Tensor  缺 ``aten.view``
  * Float8Tensor 缺 ``aten.abs``、``aten.div``、``aten.mul``、``aten.add``、
    ``aten.sub``、``aten.round``、``aten.clamp``、``aten.neg``

0.17 的 int8 仍走老的 AffineQuantizedTensor（完整且幂等），只有 float8 受影响；
0.18 的 int8 也换成了缺 view 的 Int8Tensor，int8/float8 都受影响。

本脚本在运行时把这些算子注册进各自的 dispatch 表。实现策略是
“反量化 -> 执行原操作 -> 返回普通张量”：量化权重反量化后近似还原浮点值，
对重量化/选参流程语义正确（与 torchao 0.17 的 AffineQuantizedTensor 行为一致）。

用法
----
在任何 torchao 量化调用之前导入即可：

    import fix_torchao_018_xpu   # 或 from fix_torchao_018_xpu import patch_torchao_018_xpu

脚本对版本不敏感：存在哪个类就补哪个类缺失的算子。
"""

import torch


def _dq(x):
    """反量化 torchao 量化张量，其它类型原样返回。"""
    from torchao.quantization import Int8Tensor, Float8Tensor

    if isinstance(x, (Int8Tensor, Float8Tensor)):
        return x.dequantize()
    return x


def _patch_int8():
    """Int8Tensor 缺 aten.view：反量化 -> view。"""
    from torchao.quantization import Int8Tensor

    if torch.ops.aten.view.default in Int8Tensor._ATEN_OP_TABLE.get(Int8Tensor, {}):
        return

    @Int8Tensor.implements(torch.ops.aten.view.default)
    def _(func, types, args, kwargs):
        t = args[0]
        kwargs = kwargs or {}
        size = args[1] if len(args) > 1 else kwargs.get("size")
        return t.dequantize().view(size)


def _patch_float8():
    """Float8Tensor 缺 abs 及一组算术算子：反量化 -> 执行 -> 返回普通张量。"""
    from torchao.quantization import Float8Tensor

    table = Float8Tensor._ATEN_OP_TABLE.setdefault(Float8Tensor, {})

    if torch.ops.aten.abs.default in table:
        return

    @Float8Tensor.implements(torch.ops.aten.abs.default)
    def _(func, types, args, kwargs):
        return torch.abs(_dq(args[0]))

    @Float8Tensor.implements(torch.ops.aten.div.Tensor)
    def _(func, types, args, kwargs):
        return _dq(args[0]) / _dq(args[1])

    @Float8Tensor.implements(torch.ops.aten.div.Tensor_mode)
    def _(func, types, args, kwargs):
        return torch.div(
            _dq(args[0]), _dq(args[1]),
            rounding_mode=args[2] if len(args) > 2 else None,
        )

    @Float8Tensor.implements(torch.ops.aten.mul.Tensor)
    def _(func, types, args, kwargs):
        return _dq(args[0]) * _dq(args[1])

    @Float8Tensor.implements(torch.ops.aten.add.Tensor)
    def _(func, types, args, kwargs):
        return _dq(args[0]) + _dq(args[1])

    @Float8Tensor.implements(torch.ops.aten.sub.Tensor)
    def _(func, types, args, kwargs):
        return _dq(args[0]) - _dq(args[1])

    @Float8Tensor.implements(torch.ops.aten.round.default)
    def _(func, types, args, kwargs):
        return torch.round(_dq(args[0]))

    @Float8Tensor.implements(torch.ops.aten.clamp.default)
    def _(func, types, args, kwargs):
        return torch.clamp(
            _dq(args[0]),
            args[1] if len(args) > 1 else kwargs.get("min"),
            args[2] if len(args) > 2 else kwargs.get("max"),
        )

    @Float8Tensor.implements(torch.ops.aten.neg.default)
    def _(func, types, args, kwargs):
        return -_dq(args[0])


def patch_torchao_018_xpu() -> bool:
    """注册缺失算子。返回是否执行了补丁。"""
    import torchao

    version = tuple(int(x) for x in torchao.__version__.split("+")[0].split("."))
    patched = []
    try:
        _patch_int8()
        patched.append("Int8Tensor.view")
    except Exception:
        pass
    try:
        _patch_float8()
        patched.append("Float8Tensor.abs+算术")
    except Exception:
        pass
    if patched:
        print(f"[fix_torchao_018_xpu] torchao {torchao.__version__}: "
              f"已注册缺失算子 ({', '.join(patched)})。")
    else:
        print(f"[fix_torchao_018_xpu] torchao {torchao.__version__}: 无需补丁。")
    return len(patched) > 0


if __name__ == "__main__":
    patch_torchao_018_xpu()

    # 自检：直接量化 + 前向
    from torchao.quantization.quant_api import quantize_ as torchao_quantize_
    from torchao.quantization.quant_api import Int8WeightOnlyConfig, Float8WeightOnlyConfig

    for name, cfg in [("int8", Int8WeightOnlyConfig()), ("float8", Float8WeightOnlyConfig())]:
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128), torch.nn.Linear(128, 16)
        ).to("xpu")
        torchao_quantize_(model, cfg)
        with torch.no_grad():
            out = model(torch.randn(4, 64, device="xpu"))
        print(f"self-test {name}: OK {tuple(out.shape)}")
