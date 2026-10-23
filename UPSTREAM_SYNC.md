# 上游跟进状态（ai-toolkit → ai-toolkit-xpu）

本 fork 的基线：后端代码 `0.12.27`，UI 同步到上游 `0.13.6`（中文词条见 `ui_i18n/`）。
上游仓库：<https://github.com/ostris/ai-toolkit>（本文件记录最后一次核对：**2026-10-24**，上游 HEAD `7861f92b`）。

合并原则：**只手工挑后端能安全落地的提交**。上游近期的模型支持大量依赖新的
`toolkit/models/v2/` 模型栈（我们这条线只有 9 个文件的旧版、上游已有 40 个），
整块换会牵动所有现存模型的加载路径，因此拆成"安全合并"和"待跟进"两类。

## 已合并（本轮）

| 上游提交 | 内容 | XPU 侧说明 |
| --- | --- | --- |
| `543a0d7` | 文本编码器被卸载后保存 device state 报错 | 需要 `toolkit/unloader.py::FakeTextEncoder`，本仓库早有该桩类 |
| `3a39d8c` | `cache_text_encoder` device preset 缺失 | 纯 2 行 |
| `881143c` | EMA decay 方向反了（越训越"跟上"而不是"平滑"）| 只取 `toolkit/ema.py` 一行；`version.py` 的版本号是上游自增，本 fork 不跟 |
| `a35d833` | TE 卸载在 pipeline 不暴露 TE 的模型上被跳过（Qwen-Image-2.1 / Z-Image / LTX2）| 见 `toolkit/unloader.py` |
| `e65c4d0` | Krea2 在带 shift 训练时 patch size 不对 | 见 `krea2/src/mmdit.py` |
| `f073188` | Qwen-Image-Edit-Plus batch>1 且未缓存文本嵌入时崩溃 | 见 `qwen_image_edit_plus.py` |
| `07abdbe` / `086b663` 的通用部分 | `target_size`（按分桶尺寸缩放参考图）与 RGBA 全链路 | 见下 |

### "通用部分"具体指什么

Qwen-Image 2.1 顺带在上游核心做了两件对**所有模型**都无害的基础设施改动，本 fork 已合并：

1. **`target_size` 管道**：`SDTrainer` → `encode_prompt(..., target_size=(w, h))` →
   `get_prompt_embeds`。只有签名里带 `target_size` 的模型才会收到它（用
   `inspect.signature` 判断），其余模型行为不变；
   文本嵌入缓存键也会带上 `control_target_size`。
2. **RGBA 全链路**：`BaseModel.load_rgba` 属性（默认 `False`）、dataloader 的 RGBA 加载、
   控制图 alpha 保留、增广时 alpha 跟着几何变换、缩略图带 alpha 时存 PNG、
   `save_image` 遇到 RGBA 自动改存 png。**本 fork 现存模型全部返回 `False`**，
   即默认行为与合并前完全一致，只是为将来接入 RGBA VAE 的模型铺好路。

### Qwen-Image 2.1（`c2622ed` 等 5 个提交）— **已合并并实测**

为了让它能在 XPU 上加载并训练，本轮把上游的 **`toolkit/models/v2` 整套模型栈**搬了过来
（新增 38 个文件；`_mixin.py` 232 → 850 行），并在 XPU 侧做了三处对接：

1. `toolkit/util/quantize.py`：补上 v2 需要的 `dequantize_ostris_to_linear` /
   `quantize_module` / `attach_ara_and_quantize`（外加 `_wrap_qlinear_ndim`、
   `keep_on_quantize_device`），我们自己的 `omnitype`（xpu_int8/xpu_fp8）实现原样保留，
   所以 v2 模型的量化照样走 XPU 内核，缺轮子时仍然自动回退 torchao int8。
2. `toolkit/lora_special.py`：`LINEAR_MODULES` 补上 `OmniInt8Linear` / `OmniFp8Linear`。
   v2 模型是"先量化、后建 LoRA"，少了这两行会直接 `create LoRA for U-Net: 0 modules`。
3. `toolkit/models/base_model.py`：补 `component_load_kwargs` / `get_latent_space_version` /
   `get_text_embedding_space_version` / sample-step 钩子；`toolkit/sample_step_hook.py`。

实测（A770 16GB，本地 ComfyUI 权重 + `MODELS_PATH`，`qtype: xpu_int8`，层级卸载 100%）：

| 项目 | 结果 |
| --- | --- |
| 加载 | transformer 32 blocks、文本编码器 36 blocks 全部走 XPU int8；VAE 正常 |
| 训练 | 2 步 / 512 / LoRA r16：**5.73–6.56 s/it**，loss 0.604，checkpoint 正常落盘 |
| 显存 | 峰值 **10.3GB**（16GB 卡留有余量），训练中稳态约 2–3GB |
| 内存 | 峰值 47GB（含缓存阶段） |

UI 里已新增 `Qwen-Image 2.1（文生图 / 参考图）` 入口，默认值就是上面这套。

### FLUX.3 Image Klein（上游 PR #1078 + 3 个后续修复）— **已合并并实测**

上游 `91bdc6d`（PR #1078）新增 `extensions_built_in/diffusion_models/flux3/`（9 个文件、
约 2200 行），随后三个提交补了 comfy 格式文本编码器加载、文本嵌入缓存，以及
"本地 ComfyUI 单文件优先"（`12395a1`）。本 fork 已整套搬入并接上 XPU：

1. **新增目录 + 注册**：`flux3/`（flux3.py / transformer / image_vae / pipeline /
   positional / text_encoder / checkpoints）+ `__init__.py` 两行 + `registry.py` 的
   `flux3` 条目（模型列表现为 38 个）。
2. **同步它依赖的工具模块**（上游 9 月之后改过，我们此前停在 9 月版）：
   `toolkit/util/comfy_quant_import.py`（177→378 行：header 式量化元数据、nvfp4/fp8
   合并键拆分等）、`convrot_quant.py`、`ostris_quant.py`、`dequantize.py`，外加 4 个
   新文件 `float8_quant.py` / `mixed_precision.py` / `streamed_safetensors.py` /
   `unified_memory.py`。**我们自己的 `omni_int8.py` 与 XPU 补丁未动。**
3. **`toolkit/models/v2/_mixin.py`**：补上游的 `_markers_from_header()` —— comfy 的
   header 式量化布局把每层配置放在 safetensors metadata（`_quantization_metadata`）里，
   这里合成 `<layer>.comfy_quant` marker，让导入器只面对一种布局。
4. **XPU 适配（1 处）**：`flux3/src/transformer.py::rope()` 原本用 `torch.float64`
   建频率表 → XPU 报 `Required aspect fp64 is not supported on the device`；改为
   设备感知（XPU 用 fp32，CUDA 保持 fp64；该表两行后本来就会 `.float()`）。

实测（A770 16GB、`xpu_int8`、100% 层级卸载、512 档 → 分桶 384×672、LoRA r16、batch 1）：

| 项目 | 结果 |
| --- | --- |
| 训练 | **2/2 步**，**4.41–5.34 s/it** |
| 显存 | 峰值 **8.1GB**（16GB 卡余量充足） |
| 内存 | 峰值 13.6GB |
| 产物 | LoRA 80MB 正常存盘 |
| 权重来源 | **全部本地**：transformer `flux3_image_klein_bf16`、TE `text_encoders/qwen/qwen3vl_4b_bf16`、VAE `vae/flux/flux2-vae`（选择器递归查 MODELS_PATH，零下载） |

UI 已加 `FLUX.3 Image Klein（文生图）` 入口（默认 xpu_int8 + 层级卸载）；
示例配置：`config/examples/train_lora_flux3_xpu_a770.yaml`。

> 注意：`flux3_image_klein_sd_*`（step-distilled 变体）不是训练候选，加载也会被
> `ref_time_in` 键拒绝，别拿它当底模。

### MOSS-Image（`e05a3a4` / `c6a9b56` / `b7cd8a1` / `29f5ff4` / `80cb872`）— **已合并并实测**

上游这一串（含 Base / SFT / 编辑训练 / tokenizer 修复）已整套搬入：

1. **新增目录**：`extensions_built_in/diffusion_models/moss_image/`（5 个文件：
   moss_image.py + src/{checkpoints,pipeline,transformer}）+ 新文本编码器
   `toolkit/models/v2/text_encoders/qwen35.py`（Qwen3.5-9B，纯语言塔）；
2. **注册**：模型列表（39 个）+ `registry.py` 三条（`moss_image` / `moss_image:base` /
   `moss_image:base_edit`，编辑版带 `edit/kv_cache/match_target_res`）；
   UI 加了三张卡（默认 `xpu_int8`、`uncond_skip_layers: [22]`、LoRA 绕开第 22 层）；
3. **`toolkit/models/v2/_mixin.py`**：并入 `80cb872` 的 tokenizer 完整性校验
   （`_local_first(..., validate=...)` + `_tokenizer_complete`）——transformers 5.5.3
   在"部分缓存"的仓库上会返回**只有 1 个 token 的 tokenizer** 而不报错，没有这个校验就会静默用坏；
4. **权重来源**：全部走本地 ComfyUI 单文件（选择器递归查找 MODELS_PATH）：
   DiT `moss_image_base_bf16` / `_int8_convrot`、VAE `vae/**/flux2-vae.safetensors`、
   TE `text_encoders/**/qwen3.5_9b_int8_convrot.safetensors`；只有 Qwen/Qwen3.5-9B 的
   config/tokenizer 需要联网（国内可设 `HF_ENDPOINT=https://hf-mirror.com`）。

**XPU 适配两处（都在本 fork 侧）**：

- `moss_image/src/transformer.py::_time_modulation()` 用 `torch.float64` 建时间频率表 →
  XPU 报 `Required aspect fp64 is not supported on the device`；改为 `rope_dtype(device)`
  （与该表下方本来就会做的 `.float()` 一致）。
- **层级卸载登记表补两类**（`toolkit/memory_management/manager.py`）——这是 MOSS 暴露出来的
  一个**通用机制缺口**，上游**没有**处理（`git grep A_log/GatedDeltaNet` 只有我们移植时写的注释）：
  1. `Conv1d` 补进 `UNMANAGED_MODULES`：Qwen3.5 线性注意力里的 depthwise conv1d 之前没人登记，
     卸载时权重留在 CPU → `XPUBFloat16Type ≠ CPUBFloat16Type (xpu_convolution)`；
  2. 新增 `UNMANAGED_PARAM_MODULES_INCLUDES = ["GatedDeltaNet"]`：把这类模块**直接挂着的裸参数**
     （`A_log` / `dt_bias`）登记为 unmanaged，子层照常搬运 → 否则崩在
     `found at least two devices, xpu:0 and cpu`。

  机制说明：层级卸载开启时模型**不做整体 `.to(device)`**，只有"登记过的"模块/参数会跟着
  `memory_managed_to()` 走；**没登记的带权重模块或裸参数就永远留在 CPU**。CUDA 上同样成立，
  只是那边显存大、通常不给 TE 开卸载，所以没暴露。

实测（A770 16GB、`xpu_int8`、层级卸载 100%（**含文本编码器**）、512 档、LoRA r16、batch 1）：

| 项目 | 结果 |
| --- | --- |
| 训练 | **2/2 步**，5.85–7.78 s/it，checkpoint 正常落盘 |
| **显存峰值** | **8.8GB**（修复前 TE 不卸载时是 15.9GB；现在 TE 能安全卸载） |
| 内存峰值 | 38GB（含 TE 反量化→再量化 + latent/TE 缓存） |
| 权重来源 | 全部本地，零下载 |

示例配置：`config/examples/train_lora_moss_image_xpu_a770.yaml`。

> 待办：`ace_step`（音频）的 DiT 里也有 `nn.Conv1d`（patchify），开层级卸载时会遇到与
> Conv1d 相同的场景——它现在已因为上面的登记而受益，但值得单独跑一轮验证。

### FIBO 2.0 Dev 与 MOSS VAE 解码（2026-10-24）— **已合并并实测**

| 上游提交 | 内容 | 本 fork 状态 |
| --- | --- | --- |
| `7cc9fef4` | 新增 Bria FIBO 2.0 Dev，支持文生图与参考图编辑训练 | 已完整合并；13 个模型/注册/测试文件，UI 按本 fork 的旧版页面结构手工移植 |
| `7861f92b` | 移除 MOSS VAE decode 的强制 `no_grad`，供 DFE 使用 | 已合并；与 XPU 训练路径不冲突 |

FIBO 的 XPU 适配与上游代码分开保留，后续同步时不要用上游文件覆盖：

1. `fibo2/src/transformer.py::_rope()` 在设备上创建 fp64 频率表，XPU 不支持 fp64。
   改为 `rope_dtype(ids.device)`：XPU 用 fp32，CPU/CUDA 继续用 fp64。
2. FIBO 的注入层把 `cross_attn_gate` 直接挂在 block 上，Perceiver 把
   `latent_queries` 直接挂在自身；层级卸载时这些裸参数原本不会被搬运，会触发
   `found at least two devices, xpu:0 and cpu`。已把 `Fibo2Block` /
   `Fibo2Perceiver` 登记到 `UNMANAGED_PARAM_MODULES_INCLUDES`。
3. UI 新增中文 `FIBO 2.0 Dev（文生图 / 编辑训练）`，默认
   `xpu_int8` + 100% transformer/TE 层级卸载；有 control 图走编辑，无 control 图走文生图。
   同时把 Next.js 15 的动态路由 `params` 全部改成 `Promise` 类型，并移除
   `ignoreBuildErrors`；主 `tsc --noEmit` 和 Next 生产构建均通过。
4. 示例配置：`config/examples/train_lora_fibo2_xpu_a770.yaml`。

**A770 16GB 实测（torch 2.13.0+xpu、`omni_xpu_kernel 0.2.0b1+torch213.dg2.2`、
`xpu_int8`、Transformer/TE 层级卸载 100%）**：

| 项目 | 结果 |
| --- | --- |
| 权重来源 | 复用本地 ComfyUI 的 FIBO2 bf16、Qwen3-VL-4B bf16、FLUX.2 VAE，零下载 |
| 文生图 | 384×384、2 步通过，约 **8.95 s/it** |
| 编辑图 | 384×384、2 步通过，约 **8.71 s/it** |
| 显存 | 采样峰值分配 **7.69GB**、峰值保留 **8.80GB**；结束后保留池回收到 1.56GB |
| 编辑训练 | 1 步真实反向/优化器更新，**17.64 s/it**，loss **0.7477** |
| LoRA | 261 个目标模块、522 个张量、29.3MB；全部 finite，261 个 `lora_B` 均已更新 |
| UI | 新增中文入口；`localize_ui.py --check` 无待应用变更；`tsc --noEmit` 与 Next 生产构建通过 |

采样输出经过实际文件检查，并非只验证进程退出；训练后预览图也正常生成。

### Ming-Image（`77847d7` + `460c29b`）— 已合并，未实测

- 代码、v2 依赖、UI 入口都已就位（`Ming-Image 0.1 Design（MoE·实验）`），
  但本机没有 Ming-Image 权重，**未做真机验证**。
- MoE 部分走 triton，上游有 `x.is_cuda` 守卫 → XPU 会走 torch 回退（速度未知）。
- 有权重后可用同一条路径验证：
  `python testing/test_model_loading.py --arch ming_image --device xpu --allow-download`

### 2026-10-23 上游同步（排除 H3 / Qwen3.8）

本轮从上游 `3770c8f` 之后挑入以下提交，H3 系列和 Qwen3.8 27B 暂不进入本 fork：

| 上游提交 | 内容 | XPU 状态 |
| --- | --- | --- |
| `8cf43dd2` | 缓存文本嵌入后把文本编码器放回 CPU | 已合入；避免缓存阶段结束后 TE 占着 XPU |
| `9d408eb6` | FLUX.3 多参考图缩放修复 | 已合入 |
| `9a1828f2` | MOSS q/k/v LoRA 保存为 ComfyUI 兼容的未融合格式 | 已合入；保存与重新加载互相转换 |
| `c22f21fa` | JPEG/PNG/WebP EXIF 方向与 WebP 尺寸识别 | 已合入；XPU 无关，纯图片元数据路径 |
| `7c6b168c` / `85d01c6a` | caption 标签可读取的文件扩展名白名单、路径穿越防护 | 已合入 |
| `5305fef` | x0 预测逻辑下沉到模型类 | 已合入；后续 x-prediction 模型复用 |
| `0cbac2c1` / `dc3e230` | 批次并行解码、跨数据集合并分桶 | 已合入；XPU 下 `batch_load_threads` 默认降为 2，避免和层级卸载抢 CPU |
| `2f5dc0d` / `2702e7b` / `65b2ca8` | MOSS SFT Edit、参考隔离 SDPA、批量文本编码 | 已合入；XPU int8 编辑采样 2 步通过 |
| `e1973f7` | ConvRot Triton `libdevice` 兼容 | 已合入 |
| `dee332e` | Iris-3B 像素空间文生图 | 已合入；A770 实测通过，但本机测试权重已按要求移除 |
| `f69d4f4` | Qwen-Image 2.1 Turbo + 训练适配器 | 已合入；A770 XPU 实测通过 |

**Qwen-Image 2.1 Turbo 实测（A770 16GB）**

- 官方 Turbo 权重 14.2GB，训练适配器约 159MB。
- `xpu_int8`、100% 层级卸载、训练适配器挂载后，8 步首轮采样约 28 秒。
- 完全卸载后的 2 步采样约 14 秒。
- XPU 峰值分配约 3.43GB，reserved 约 15.04GB。
- 训练适配器只用于训练，采样时关闭；它不会写入最终用户 LoRA。

**MOSS-Image SFT Edit 实测**

- 本地 `moss_image_sft_bf16.safetensors`、本地 Qwen3.5 TE、FLUX.2 VAE。
- `xpu_int8` + 100% 层级卸载，2 步编辑采样约 15 秒。
- XPU 峰值分配约 2.93GB，reserved 约 7.16GB。

示例配置：

- `config/examples/train_lora_qwen_image_21_turbo_xpu_a770.yaml`
- `config/examples/train_lora_moss_image_edit_xpu_a770.yaml`
- `config/examples/train_lora_iris3b_xpu_a770.yaml`

**暂不合并**

- H3 / Fast H3 系列：本机显存和 XPU attention 支持不合适，按用户要求跳过。
- Qwen3.8 27B：用户要求暂缓；移出本轮代码与 UI，待后续单独验证推理栈后再接入。
- Docker 专用提交：本 fork 不维护 XPU 容器。

### 模型卸载稳定性 + D-OPSD bleed loss（`683fe8a`）— 未合并

- 动 `toolkit/memory_management/manager.py`、`toolkit/models/v2/_mixin.py`、
  `toolkit/util/quantize.py` —— 都是本 fork 为了 XPU 显存/搬运改过的文件，
  且 D-OPSD 部分依赖 v2 栈。**建议随 v2 同步一起做**，单独合容易把已验证的搬运逻辑改坏。

### 性能/杂项合集（`7195abc`）— 未合并

- 同样大量落在 `manager.py` / `manager_modules.py` / `v2/_mixin.py`。
- 里面每个模型的"性能修复"大多是几行，等 v2 同步时一起对齐。

### 其它上游提交（与本 fork 无关或不适用）

- YuE2 / MOSS / Qwen2.5-Omni 等音频、字幕、captioner 相关：本 fork 未裁剪这些功能，
  但它们不涉及 XPU 关键路径，按需再合。
- `8fa15e3`（模型卡片改为插件目录 `ui.tsx`）：本 fork 的 UI 停在 0.13.6，
  模型卡片仍在 `ui/src/app/jobs/new/options.tsx`，要跟需要连 UI 一起升级。

## 下次同步的做法

## torch 2.14 崩溃：根因已定位（2026-09-24）

之前记录的"2.14 训练必崩（`ze_intel_gpu64.dll` 0xC0000005）"已经找到确切触发条件：

> **`del` 掉大张量之后调用 `torch.xpu.empty_cache()`**，随后任何算子都可能触发驱动空指针。
> conv 反向 → `del` + `empty_cache` → attention 这条序列 **3/3 稳定崩**；
> 同样的序列**不调** empty_cache 则 **0/3 通过**（两个驱动版本、cp311/cp313、三种量化后端都验证过）。

训练侧对上的就是我们 `SDTrainer` 每步的那次 `torch.xpu.empty_cache()`（本 fork 当初为了
压 reserved pool 加的）。修复方式：`toolkit/device_utils.per_step_empty_cache_enabled()`
—— **torch ≥ 2.14 默认跳过，2.13 维持原样**，环境变量 `AITK_XPU_EMPTY_CACHE=1/0` 可覆盖；
`scripts/xpu_train_ops_check.py` 用同一套策略做自检。

修复后实测（A770 + 驱动 32.0.101.8860）：探针默认 0/3 崩；krea2 768 **10/10 步**
（12.5–13.6 s/it）、Qwen-Image 2.1 768 **2/2 步**，checkpoint 均正常保存。
仓库默认依赖仍钉 2.13，2.14 属于"已适配、可选升级"状态。

1. `git clone https://github.com/ostris/ai-toolkit` 到临时目录并 `git fetch`；
2. 拉出 `origin/main` 的工作树（`git worktree add --detach <dir> <sha>`）；
3. 用"提交级 diff + `git apply`"逐个试；**新增目录直接 copy**，改动核心文件的按 hunk 手工过；
   新增模型如果带 `extensions_built_in/diffusion_models/ui.tsx`，不要整文件覆盖，
   要移植到本 fork 的 `ui/src/app/jobs/new/options.tsx`；
4. 冲突集中在这些文件（本 fork 的 XPU 补丁所在地）：
   `toolkit/memory_management/manager.py`、`manager_modules.py`、`toolkit/optimizer.py`、
   `toolkit/custom_adapter.py`、`toolkit/stable_diffusion_model.py`、`toolkit/models/base_model.py`、
   `toolkit/util/quantize.py`、`toolkit/unloader.py`、`extensions_built_in/sd_trainer/*`；
5. 每次合并后至少跑：`python -m py_compile`（改动文件）→
   `python -c "import extensions_built_in.diffusion_models"` →
   `python run.py config/<小配置>`（2 步冒烟）→ 需要时再跑 UI `npm run build`。

## 用本地 ComfyUI 权重验证新模型（不用重新下模型）

`toolkit/models/v2` 的加载器会在 `MODELS_PATH` 指向的目录里找 comfy 单文件，
命中本地文件就不下载权重，只从 HF 取配置/processor：

```powershell
$env:MODELS_PATH = "E:\Comfyui\ComfyUI\models"   # 你自己的 ComfyUI models 目录
$env:HF_HUB_OFFLINE = "0"                        # 配置/processor 还是要联网
.\.venv\Scripts\python.exe testing\test_model_loading.py --arch qwen_image_2 --device xpu --allow-download
```

文件优先级按请求的 qtype 排：`qtype: xpu_int8`（或其它"非 convrot/float8/nvfp4"后端）
会优先选 **bf16** 单文件，然后由 XPU 内核现场量化；想要"直接用预量化文件"就把
qtype 设成 `convrot8`（那个走的是 Ostris 后端，XPU 上没有专门内核，速度不划算）。
