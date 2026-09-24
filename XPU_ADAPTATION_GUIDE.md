# ai-toolkit XPU 改造说明书（0.12.27）

> 把 [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) 跑在 Intel Arc GPU（XPU）上，保留中文界面与训练曲线图。
> 本说明基于实测：Windows + Python **3.12 / 3.13** + Intel Arc A770 16GB + oneAPI 2026，
> 代码基线官方 **0.12.27**（UI 已跟进到官方 0.13.6）。

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
| Python | **3.11 / 3.12 / 3.13 均可** | 3.12 为已验证版本（推荐）；便携版或官方安装版都行 |
| Intel 显卡驱动 + oneAPI 2026 | runtime | `torch.xpu.is_available()` 必须为 True |
| Arc 显卡 | A770 16GB 等 | 显存不足靠内存分块搬运（第 4 节） |

## 2. 快速开始

### 2.1 克隆本仓库（推荐：uv 一键补全环境）

本仓库用 **uv** 管理环境：`pyproject.toml` + `uv.lock` + `.python-version` 就是环境定义
（备份只需这几个 KB 级文件，不必备份整个虚拟环境）。

```powershell
git clone https://github.com/JWLHS/ai-toolkit-xpu
cd ai-toolkit-xpu
setup_xpu.bat        # = uv python install + uv sync + 下载 FFmpeg + 验证 torch.xpu
run_xpu.bat          # 之后日常启动 Web UI (http://localhost:8675)
```

**uv 常用命令**

| 目的 | 命令 |
|---|---|
| 安装/同步依赖（幂等，已一致时秒过） | `uv sync` |
| 换 Python 版本 | `uv python pin 3.13` 然后 `uv sync` |
| 不激活环境直接跑训练 | `uv run python run.py config\你的配置.yaml` |
| 备份/迁移环境 | 复制 `pyproject.toml` + `uv.lock` + `.python-version` |

索引说明：`pyproject.toml` 里 `[[tool.uv.index]]`（XPU 索引，`explicit = true`）+
`[tool.uv.sources]` **只把 torch / torchvision / torchaudio / torchao / triton-xpu 指向 XPU 索引**，
其余包走 PyPI。这样既拿到 XPU 轮子，又不会被 XPU 索引里同名的其他包（如 torchcodec）劫持版本。

> `setup_xpu.bat` 未装 uv 时会尝试自动安装；若失败请按提示安装
> <https://docs.astral.sh/uv/getting-started/installation/>。

### 2.2 从官方仓库从头改造（自己打补丁）

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

> 从官方仓库改造时，`requirements_base.txt` 需换成 `xpu_patches/requirements_base.txt.xpu`；
> 本仓库已内置（另有 uv 用的 `pyproject.toml`）。

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

- **XPU 支持 fp64 吗？** **能分配、不能算**：`torch.zeros(dtype=float64, device="xpu")` 可以成功，但任何
  算子（`arange` / `cos` / `sin` / `einsum` / `matmul`…）都会报
  `Required aspect fp64 is not supported on the device`。
  因此**规则是：凡是把 float64 放到设备上参与计算的地方，一律改成 float32**（CPU 侧保留 float64 无妨，
  参考实现里那点精度只有 CPU 才算得到）。
  本仓库已把这条规则**铺开到所有模型**（不再只修 krea2）：
  `toolkit/device_utils.py` 提供 `rope_dtype(device)` 与 `adjust_dtype_for_device(dtype, device)`
  （XPU/MPS → float32，其余 → float64），已改的地点包括
  chroma `rope`、hidream `rope`/`expand_timesteps`、boogu_image `get_freqs_cis`、omnigen2 `get_freqs_cis`、
  zeta_chroma 时间步、prx_pixel_t2i 的 `maybe_adjust_dtype_for_device`、wan21 rotary、krea2 `rope`；
  minimax_h3 的位置网格在 CPU 上以 float64 构建（保证网格精度），**搬运到设备时降为 float32**。
  回归验证脚本：`scripts/verify_xpu_fp64_paths.py`（助手行为 + 前提确认 + 6 条模型路径实测）。
- **环境用 uv 还是 pip？** 推荐 **uv**：`pyproject.toml` + `uv.lock` + `.python-version` 即环境定义，
  `uv sync` 幂等安装、`uv python pin 3.13` 换版本、备份只需几个 KB 文件。
  XPU 轮子通过 `[[tool.uv.index]]`（`explicit = true`）+ `[tool.uv.sources]` 精确指向，
  避免 uv 默认的 first-index 策略把 `torchcodec` 之类解析到 XPU 索引上的错误版本（实测踩过）。
  Python 版本不写死：`requires-python = ">=3.11,<3.14"`，3.12/3.13 均已实测可用。

- **torchao 用 0.17 还是 0.18？** 默认 **0.17.0+xpu**（krea2 int8 实测每步 19-22s、显存 8.7GB，最稳）。注意：0.17 的 **float8** 同样缺 `Float8Tensor.abs`（二次量化会静默失败），0.18 则 int8/float8 都缺（int8 换成新 `Int8Tensor` 且丢了幂等保护）。[fix_torchao_018_xpu.py](fix_torchao_018_xpu.py) 已改为“缺啥补啥、不看版本”，`run.py` 自动加载，0.17 的 float8 也被兜住（实测通过）。补充：0.18 在本机 ai-toolkit 训练中实测速度/显存劣于 0.17，但根因未定位（可能与我们 dequantize 式补丁实现或使用方式有关），不作为 0.18 缺陷结论。已向官方提 issue 并补充修正说明：[pytorch/ao#4845](https://github.com/pytorch/ao/issues/4845#issuecomment-5454705829)。
- **transformers 用 4.57.3 还是 5.5.3？** 5.5.3（官方 0.12.27 按 5.x 编写）。
- **torch 用 2.13 还是 2.14？** **必须用 2.13.0+xpu**（本仓库钉死）。2.14.0+xpu 在 A770 + 当前 Intel 驱动上
  **训练必崩**，不是配置问题、不是本仓库代码问题——实测证据（2026-09-24，同机同驱动）：

  | 测试 | 环境 | 结果 |
  | --- | --- | --- |
  | krea2 2 步 / 768 / omni int8 后端 | torch 2.14 + 上游 `+torch214.dg2` 轮子 | 崩：`ze_intel_gpu64.dll` +0x404ed4 `0xC0000005` |
  | SDXL 512 / int8 / 关层级卸载 | torch 2.14 | 崩（同偏移） |
  | SDXL 512 / 不量化 / 不卸载 | torch 2.14 | 崩（同偏移） |
  | **`scripts/xpu_train_ops_check.py`**（纯 torch，无本仓库代码） | torch 2.14 + **ComfyUI 自带 cp311 环境** | 崩（同偏移） |
  | 同一脚本 | torch 2.13.0+xpu | **全绿** |
  | torch 2.14 + Intel 运行时降到 2026.0.0 | — | 起不来：`c10_xpu.dll` 符号缺失（WinError 127） |

  结论：崩溃发生在 **conv 反向 → attention 前向** 这类训练型算子序列上，与 Python 版本（3.11/3.13 都复现）、
  与量化后端、与层级卸载、与本仓库代码均无关；ComfyUI 上"2.14 正常"只是因为它只跑推理。
  最小复现脚本已随仓库提供（见 `scripts/xpu_train_ops_check.py`），可用于换机器/换驱动时快速判定。
  其它理由：2.14 的 XPU 仍**没有编译 flash attention**（SDPA 只有 `MATH` 可用），升级无 attention 收益；
  且 2.14 会连带 torchao 只能用 0.18（int8 重复量化有已知问题，见 [pytorch/ao#4845](https://github.com/pytorch/ao/issues/4845)）。
  注：代码用的是 SDPA 优先级列表，将来任一版本 XPU 支持 flash attention 会自动启用，无需改代码。
- **`fix_torchao_xpu.py` 还有用吗？** 已过时（0.12.27 自带 ostris 量化后端），已从仓库移除；保留的本 fork 补丁只有 `fix_torchao_018_xpu.py`。
- **triton-xpu 能手动升到 3.8.0 吗？** 能装上、也能跑（2026-09-24 实测：`triton-xpu==3.8.0` 覆盖 +
  `CC="C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin\icx.exe"`，自写 add/silu kernel JIT 编译通过、
  数值正确，随后 krea2 训练冒烟 2/2 步正常），**但没必要也不建议**：
  1. `torch==2.13.0+xpu` 的元数据里**硬依赖 `triton-xpu==3.7.2`**（2.14 才依赖 3.8.0），
     写进 `pyproject.toml` 会让 `uv lock` 直接无解，所以仓库保持 3.7.2；手动覆盖的版本下次 `uv sync` 会被还原；
  2. 本 fork 的训练路径**不使用 triton**：`toolkit/util/convrot_quant.py` 的 triton 分支有
     `packed.is_cuda` 守卫，omnigen2 的 triton kernel 同理，XPU 上一律走 torch 回退。
  唯一要注意的是：真要跑 triton kernel（或 `torch.compile`）时必须有一个 C 编译器在 `CC` 里，
  本机可指向 oneAPI 的 `icx.exe`；否则会报 `Failed to find C compiler`。
- **HF 报 “client has been closed”？** huggingface_hub 的 httpx 线程问题，文件缓存后设 `HF_HUB_OFFLINE=1` 重跑即可。

## 9. 分享清单

1. 本仓库（对齐 0.12.27 后）；
2. `xpu_patches/`（5 补丁 + 依赖模板 + 启动脚本示例）；
3. `E:\tt\ffmpeg-shared\`（FFmpeg 8.1 full-shared）；
4. 本说明文档。
