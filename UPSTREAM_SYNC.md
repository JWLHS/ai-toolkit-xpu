# 上游跟进状态（ai-toolkit → ai-toolkit-xpu）

本 fork 的基线：后端代码 `0.12.27`，UI 同步到上游 `0.13.6`（中文词条见 `ui_i18n/`）。
上游仓库：<https://github.com/ostris/ai-toolkit>（本文件记录最后一次核对：**2026-09-24**，上游 HEAD `460c29b`）。

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

### Ming-Image（`77847d7` + `460c29b`）— 已合并，未实测

- 代码、v2 依赖、UI 入口都已就位（`Ming-Image 0.1 Design（MoE·实验）`），
  但本机没有 Ming-Image 权重，**未做真机验证**。
- MoE 部分走 triton，上游有 `x.is_cuda` 守卫 → XPU 会走 torch 回退（速度未知）。
- 有权重后可用同一条路径验证：
  `python testing/test_model_loading.py --arch ming_image --device xpu --allow-download`

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

1. `git clone https://github.com/ostris/ai-toolkit` 到临时目录并 `git fetch`；
2. 拉出 `origin/main` 的工作树（`git worktree add --detach <dir> <sha>`）；
3. 用"提交级 diff + `git apply`"逐个试；**新增目录直接 copy**，改动核心文件的按 hunk 手工过；
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
