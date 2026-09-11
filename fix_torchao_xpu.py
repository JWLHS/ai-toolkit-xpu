"""
fix_torchao_xpu.py — 修复 torchao XPU 版的两个已知问题:
  1. 删除 MPS 模块（Windows 上不存在 .dylib，导致 RuntimeError）
  2. 修复 toolkit/util/quantize.py 中的 API 变更
  (UIntXWeightOnlyConfig → IntxWeightOnlyConfig, torch.uint → torch.int)

用法: E:\tt\python\python.exe fix_torchao_xpu.py
"""
import os
import shutil
import re
import sys

TOOLKIT_DIR = os.path.dirname(os.path.abspath(__file__))


def fix_mps():
    """删除 torchao 中的 MPS 模块（Windows 上不可用）"""
    try:
        import torchao
    except ImportError:
        print("[SKIP] torchao 未安装，跳过 MPS 修复")
        return

    mps_dir = os.path.join(os.path.dirname(torchao.__file__),
                           "experimental", "ops", "mps")
    if os.path.isdir(mps_dir):
        shutil.rmtree(mps_dir)
        print(f"[OK] 已删除 MPS 模块: {mps_dir}")
    else:
        print("[OK] MPS 模块已不存在，跳过")


def fix_quantize_py():
    """修复 quantize.py 中 torchao API 变更:
    UIntXWeightOnlyConfig → IntxWeightOnlyConfig
    torch.uintX → torch.intX"""
    quantize_path = os.path.join(TOOLKIT_DIR, "toolkit", "util", "quantize.py")

    if not os.path.exists(quantize_path):
        print(f"[SKIP] {quantize_path} 不存在，跳过")
        return

    with open(quantize_path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    content = content.replace("UIntXWeightOnlyConfig", "IntxWeightOnlyConfig")
    content = re.sub(r"torch\.uint(\d)", r"torch.int\1", content)

    if content != original:
        with open(quantize_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[OK] 已修复 {quantize_path}")
    else:
        print(f"[OK] {quantize_path} 无需修复")


if __name__ == "__main__":
    fix_mps()
    fix_quantize_py()
    print("完成。")
