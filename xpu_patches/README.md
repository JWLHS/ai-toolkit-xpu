# XPU 补丁包（ai-toolkit 0.12.27）

适用于从官方仓库**从头改造**：`git clone https://github.com/ostris/ai-toolkit` 后应用本目录补丁。

## 文件说明

| 文件 | 作用 |
|---|---|
| `001-manager_modules-xpu.patch` | 内存分块搬运（MemoryManager）支持 XPU |
| `002-optimizer-xpu-fallback.patch` | bitsandbytes 8bit 优化器在 XPU 上回退标准优化器 |
| `003-custom_adapter-xpu-bnb.patch` | LLM 适配器 BitsAndBytes 4bit 在 XPU 上禁用 |
| `004-stable_diffusion-te-quant-warn.patch` | 文本编码器 4/8bit 量化在 XPU 上忽略 |
| `005-add-device_utils.patch` | 新增 `toolkit/device_utils.py`（XPU 设备工具） |
| `requirements_base.txt.xpu` | 官方依赖 + XPU 覆盖（复制为 `requirements_base.txt`） |

## 应用方法

```powershell
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit

# 1) 打补丁
git apply xpu_patches/001-manager_modules-xpu.patch `
            xpu_patches/002-optimizer-xpu-fallback.patch `
            xpu_patches/003-custom_adapter-xpu-bnb.patch `
            xpu_patches/004-stable_diffusion-te-quant-warn.patch `
            xpu_patches/005-add-device_utils.patch

# 2) 覆盖依赖（官方 + XPU 版本）
Copy-Item xpu_patches/requirements_base.txt.xpu requirements_base.txt

# 3) 安装依赖（torch/torchao 等自动从 XPU 索引下载）
python -m pip install -r requirements.txt
```

## 验证

```powershell
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
python -m pip check
```

详细说明见仓库根目录 `XPU_ADAPTATION_GUIDE.md`。
