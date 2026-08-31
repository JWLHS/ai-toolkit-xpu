# ai-toolkit XPU 改造说明书（0.12.27）

> 把 [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) 跑在 Intel Arc GPU（XPU）上，保留中文界面与训练曲线图。
> 本说明基于 **2026-08-28** 实测：Windows + Python 3.12 + Intel Arc A770 16GB + oneAPI 2026，代码版本官方最新 **0.12.27**。

## 0. 结论先行

**从官方仓库从头改造完全可行，且比改造拼凑版更干净。** 官方 0.12.27 自带：

- ComfyUI 式**显存/内存分块搬运**（`MemoryManager` + `layer_offloading`）；
- 训练**曲线图**完整链路（`loss_log.db` + `/api/jobs/[id]/loss` + uPlot）；
- 新建任务自动写入 `logging.use_ui_logger: true`、完整 Prisma schema。

从头改造只需 **5 个补丁 + 1 份依赖覆盖**（本仓库 `xpu_patches/` 已备好，已验证能干净 `git apply` 到官方最新提交）。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| Windows | 10/11 | 示例路径 `E:\tt` |
| Python | 3.12.x | 建议便携版 |
| Intel 显卡驱动 + oneAPI 2026 | runtime | `torch.xpu.is_available()` 必须为 True |
| Arc 显卡 | A770 16GB 等 | 显存不足靠内存分块搬运（第 4 节） |

## 2. 从零开始（推荐）

```powershell
git clone https://github.com/ostris/ai-toolkit.git E:\tt
cd E:\tt

# 1) 打 XPU 补丁（把 xpu_patches/ 复制进仓库后）
git apply xpu_patches/001-manager_modules-xpu.patch `
          xpu_patches/002-optimizer-xpu-fallback.patch `
          xpu_patches/003-custom_adapter-xpu-bnb.patch `
          xpu_patches/004-stable_diffusion-te-quant-warn.patch `
          xpu_patches/005-add-device_utils.patch

# 2) 依赖（官方 + XPU 覆盖）
Copy-Item xpu_patches/requirements_base.txt.xpu requirements_base.txt
python -m pip install -r requirements.txt

# 3) torchcodec 用 0.15.0（0.9.1 与 torch 2.13+xpu 不兼容）
python -m pip install torchcodec==0.15.0
```

> **更省事**：克隆本仓库后直接运行根目录 `setup_xpu.bat`——
> 自动装 git/Python、建 `.venv-xpu` 虚拟环境、装 XPU 依赖、下载 FFmpeg、验证 `torch.xpu`；
> 日常启动 UI 用 `run_xpu.bat`。（本仓库 `requirements_base.txt` 已内置 XPU 依赖与 torchcodec 0.15.0）

### 依赖覆盖表

| 包 | 官方 | 本改造 | 原因 |
|---|---|---|---|
| torch | 不钉 | `2.13.0+xpu` | XPU 轮子，`--extra-index-url .../whl/xpu` |
| torchvision / torchaudio | 不钉 | `0.28.0+xpu` / `2.11.0+xpu` | 同上 |
| triton-xpu | 不钉 | `3.7.2` | XPU 编译后端 |
| torchao | `0.10.0` | `0.17.0+xpu` | XPU 版（默认）；0.18 需补丁（本机实测其速度/显存劣于 0.17，根因未定位），见 FAQ |
| diffusers | git `c943837` | 同官方 | pip 版缺 Anima/Krea2 等新 API |
| transformers | `5.5.3` | 同官方 | 新版代码按 5.x 编写 |

### FFmpeg（torchcodec 需要）

FFmpeg 8.1 **full-shared**（带 DLL）bin 目录放 PATH 最前：

```bat
set "PATH=E:\tt\ffmpeg-shared\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin;%PATH%"
```

`xpu_patches/launcher_ui.bat.example` / `launcher_run.bat.example` 已内置。C 盘静态版 ffmpeg.exe 无 DLL 不能用；Adobe 目录的 DLL 会与 torch 的 c10.dll 冲突。

### 可选：中文 UI

用本仓库的 `ui/`（已汉化 + 曲线图验证过）覆盖官方 `ui/`，然后：

```powershell
cd ui && npm install && npm run build && npm start
```

## 3. XPU 补丁（5 个文件）

| 文件 | 作用 |
|---|---|
| `toolkit/device_utils.py`（新增） | 设备优先级 XPU→CUDA→CPU |
| `memory_management/manager_modules.py` | 分块搬运加 XPU 分支（权重驻内存、逐层进出显存） |
| `optimizer.py` | bitsandbytes 8bit 优化器在 XPU 回退标准优化器 |
| `custom_adapter.py` | LLM 适配器 BnB 4bit 在 XPU 禁用 |
| `stable_diffusion_model.py` | 文本编码器 4/8bit 量化在 XPU 忽略 |

## 4. 显存不足：分块搬运实测（A770）

```yaml
model:
  layer_offloading: true
  layer_offloading_transformer_percent: 1.0
  layer_offloading_text_encoder_percent: 1.0   # 文本编码器也进搬运（压显存关键）
  quantize: true
  qtype: int8          # 或 float8；int8 更省
network:
  layer_offloading: true
```

官方自带基准（1.2B 参数 + LoRA）：

| 配置 | 峰值显存 | 峰值内存 | 每步耗时 |
|---|---:|---:|---:|
| bf16 不搬运 | 2.72 GB | 1.53 GB | 415 ms |
| **bf16 + 全量搬运** | **0.47 GB** | 4.16 GB | 914 ms |
| float8 不搬运 | 1.75 GB | 1.80 GB | 653 ms |
| **float8 + 全量搬运** | **0.57 GB** | 3.10 GB | 2088 ms |

经验：**文本编码器默认留在显存**（`layer_offloading_text_encoder_percent` 不设时），12B 模型会把显存顶到 14GB+；设成 1.0 后整体可压到 5-6GB。搬运是“参数常驻内存、按层进出”，不会把整块塞进共享显存拖慢速度。

## 5. Krea2 专属适配（本仓库实测）

### 5.1 底模选择

官方（Krea 官网 / ComfyUI 博客 / krea-community HF / krea-ai GitHub）一致建议：

> **训练用 `krea-2-raw`（非蒸馏底模），推理用 `krea-2-turbo`。** LoRA 在 RAW 上训练后在 Turbo 上表现同样强。

Turbo 是 8 步蒸馏模型，用于训练会出伪影。参数参考：AdamW8Bit（XPU 自动回退 AdamW）、lr 1e-4、batch 1、bf16、FlowMatch。

### 5.2 使用 ComfyUI 单文件底模

`name_or_path` 直接指向本地 safetensors 即可（`krea-2-raw-bf16.safetensors`）。ComfyUI 版底模多带 `last.down.weight / last.up.weight` 两个未使用键，已把加载改为 `strict=False` 忽略。

### 5.3 XPU 兼容修复（2 处）

1. `src/mmdit.py` 的 `rope()` 用 `float64`，XPU 不支持 → 改为 XPU 上 `float32`；
2. （如上）状态字典 strict 加载放宽。

### 5.4 文本编码器与 VAE

- 文本编码器：Qwen3-VL-4B（HF `Qwen/Qwen3-VL-4B-Instruct`，未 gated，约 10GB）。本地 ComfyUI 的 `qwen3vl_4b_fp8_scaled.safetensors` 是单文件 fp8 格式，transformers 不能直接用，需下载（一次缓存永久复用）。
- VAE：Qwen-Image VAE（HF `Qwen/Qwen-Image`，约 0.24GB）。本地 `vae\qwen\qwen_image_vae.safetensors` 是 ComfyUI 键名，与当前 diffusers 命名不一致，直接下载缓存最省事。
- 断网/网络不稳时：文件进缓存后设 `HF_HUB_OFFLINE=1` 可跳过网络。

## 6. 曲线图与汉化 UI

链路：训练器 → `UILogger` → `output/<任务>/loss_log.db` → `/api/jobs/[id]/loss` → uPlot。

```powershell
curl "http://localhost:8675/api/jobs/<jobID>/loss?key=loss/loss"
```

注意：任务名相同且 output 有 checkpoint 时会**自动续训**跳过步数；`log_every` 默认 100，测试时设 1。

## 7. 验证清单

```powershell
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
python run.py config/train_krea2_xpu.yaml   # krea2 训练
python toolkit/memory_management/test_memory_manager.py  # 分块搬运基准
cd ui && npm start                          # 中文 UI + 曲线图
```

## 8. FAQ

- **torchao 用 0.17 还是 0.18？** 默认 **0.17.0+xpu**（krea2 int8 实测每步 19-22s、显存 8.7GB，最稳）。注意：0.17 的 **float8** 同样缺 `Float8Tensor.abs`（二次量化会静默失败），0.18 则 int8/float8 都缺（int8 换成新 `Int8Tensor` 且丢了幂等保护）。[fix_torchao_018_xpu.py](fix_torchao_018_xpu.py) 已改为“缺啥补啥、不看版本”，`run.py` 自动加载，0.17 的 float8 也被兜住（实测通过）。补充：0.18 在本机 ai-toolkit 训练中实测速度/显存劣于 0.17，但根因未定位（可能与我们 dequantize 式补丁实现或使用方式有关），不作为 0.18 缺陷结论。已向官方提 issue 并补充修正说明：[pytorch/ao#4845](https://github.com/pytorch/ao/issues/4845#issuecomment-5454705829)。
- **transformers 用 4.57.3 还是 5.5.3？** 5.5.3（官方 0.12.27 按 5.x 编写）。
- **`fix_torchao_xpu.py` 还有用吗？** 已过时，0.12.27 自带 ostris 量化后端。
- **HF 报 “client has been closed”？** huggingface_hub 的 httpx 线程问题，文件缓存后设 `HF_HUB_OFFLINE=1` 重跑即可。

## 9. 分享清单

1. 本仓库（对齐 0.12.27 后）；
2. `xpu_patches/`（5 补丁 + 依赖模板 + 启动脚本示例）；
3. `E:\tt\ffmpeg-shared\`（FFmpeg 8.1 full-shared）；
4. 本说明文档。
