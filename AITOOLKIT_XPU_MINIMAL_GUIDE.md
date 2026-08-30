# ai-toolkit 在 Intel Arc (XPU) 上的最小改造与素材准备指南

> 通用版，适用于 [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) 支持的全部模型
> （FLUX、Qwen-Image、SD/SDXL、Krea 2、Wan 等）。ai-toolkit 是通用 LoRA/微调训练器，
> 本文不针对某个模型：**XPU 最小改造只有一套，所有模型通用；素材规则也是通用的。**
> 详细改造步骤见 `XPU_ADAPTATION_GUIDE.md`，本文是"最小可用"子集。

## 1. 最小 XPU 改造（核心，唯一必做的代码改动）

从官方仓库出发，只需要 **5 个补丁 + 1 份依赖覆盖**（`xpu_patches/` 已备好，验证过能干净 `git apply`）：

```powershell
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit

# 1) 打补丁（把 xpu_patches/ 复制进仓库后）
git apply xpu_patches/001-manager_modules-xpu.patch `
          xpu_patches/002-optimizer-xpu-fallback.patch `
          xpu_patches/003-custom_adapter-xpu-bnb.patch `
          xpu_patches/004-stable_diffusion-te-quant-warn.patch `
          xpu_patches/005-add-device_utils.patch

# 2) 依赖覆盖（官方 + XPU 版本）
Copy-Item xpu_patches/requirements_base.txt.xpu requirements_base.txt
python -m pip install -r requirements.txt
```

这 5 个补丁是**最小集合**——不打 UI 汉化、不修曲线图、不装 FFmpeg，训练照样能跑：

| 补丁 | 作用 | 为什么必须 |
|---|---|---|
| `001-manager_modules-xpu.patch` | 显存/内存分块搬运（MemoryManager）支持 XPU | 没有它低显存会直接溢出 |
| `002-optimizer-xpu-fallback.patch` | 8bit 优化器在 XPU 上回退标准优化器 | bitsandbytes 没有 XPU 后端 |
| `003-custom_adapter-xpu-bnb.patch` | LLM 适配器 BitsAndBytes 4bit 在 XPU 上禁用 | 同上 |
| `004-stable_diffusion-te-quant-warn.patch` | 文本编码器 4/8bit 量化在 XPU 上跳过 | XPU 不支持该量化路径 |
| `005-add-device_utils.patch` | 新增 `toolkit/device_utils.py`（XPU 设备工具） | 各模块依赖的设备判断 |

验证：

```powershell
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
python -m pip check
```

## 2. 依赖锁定（XPU 核心）

| 包 | 锁定版本 | 说明 |
|---|---|---|
| torch | `2.13.0+xpu` | XPU 轮子，需 `--extra-index-url https://download.pytorch.org/whl/xpu` |
| torchvision / torchaudio | `0.28.0+xpu` / `2.11.0+xpu` | 与 torch 配套 |
| triton-xpu | `3.7.2` | XPU 编译后端 |
| torchao | `0.17.0+xpu` | **不要升 0.18**，见 FAQ |
| diffusers | git `c943837`（官方锁定） | pip 版缺新模型 API |
| transformers | `5.5.3`（官方锁定） | 新版代码按 5.x 编写 |

`requirements_base.txt.xpu` 就是上面这份；其余依赖跟随官方不动。

## 3. 素材准备（与具体模型无关的通用规则）

### 3.1 数量

| 训练目标 | 最少 | 推荐 |
|---|---|---|
| 角色/物体（DreamBooth 式） | 3–5 张 | 20–40 张 |
| 风格/画风 | 10–30 张 | 50–100+ |
| 概念/物件（单物体） | 5–10 张 | 20–50 张 |

具体模型的官方文档可能给出不同区间，例如 Krea 官方下限 3 张、RunComfy 建议 15–40 张；
上面是通用经验值。小数据集用 `num_repeats` 补训练量。

### 3.2 质量

- 干净：无水印、无文字、无 UI/边框、无遮挡，主体清晰。
- 原始素材分辨率尽量不低于训练分辨率（训练只缩不放大）。
- 素材间清晰度/色调统一，尽量别混入大量 AI 生成图。
- 多样性 > 数量：姿势、角度、服装、场景、光线铺开。
- 一个 LoRA 一个独立文件夹，不混视频、截图、缓存图。

### 3.3 分辨率与 bucketing（ai-toolkit 通用机制）

- `resolution` 支持列表，如 `[512, 768, 1024]`，自动拆成多个数据集副本。
- 每张图按"总面积不超过该分辨率上限"等比缩放、保持宽高比、对齐 8 的倍数。
- 具体用哪几档看目标模型的示例配置：Qwen-Image / Krea2 官方推荐三档；
  显存紧张先去最高档。

### 3.4 打标与触发词（ai-toolkit 通用机制）

- 每张图配同名 `.txt`（`0001.png` ↔ `0001.txt`），UTF-8 纯文本；
  逗号 tag 和自然语言句子都支持。
- 触发词两种写法，效果等价：
  1. 配置里写 `trigger_word: "xxx"`——加载时若 caption 里没有该词会自动加到最前面；
  2. 直接写进每个 txt。
- `[trigger]` 占位符：txt 里写 `[trigger] ...`，会被替换成配置的触发词，位置随占位符走。
- **采样 prompt 必须自己写触发词**：训练侧自动补，采样侧不补，漏写会误判训练失败。
- `caption_dropout_rate: 0.05` 可选（触发词仍保留）。
- 文件夹放一个 `default.txt` 可给没打标的图兜底。
- 官方示例注释提醒：开启 `cache_text_embeddings` 时触发词可能不生效（视版本），
  稳妥做法是写进 txt 或用 `[trigger]`。

### 3.5 目录结构

```
datasets/
└── 我的角色/
    ├── 0001.png
    ├── 0001.txt
    ├── 0002.png
    ├── 0002.txt
    └── ...
```

## 4. 通用最小 YAML 模板

以官方 `config/examples/train_lora_qwen_image_24gb.yaml` 为骨架，改 `arch` /
`name_or_path` / `resolution` / `sampler` 即可换模型；XPU 只需改 `device: xpu`：

```yaml
---
job: extension
config:
  name: "my_first_lora_v1"
  process:
    - type: sd_trainer
      training_folder: "output"
      device: xpu                # XPU 核心改动；NVIDIA 卡才是 cuda:0
      trigger_word: "myconcept"  # 可选；也可写进每个 txt
      network:
        type: lora
        linear: 32               # 小数据集 16 也够
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 100
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "E:\\datasets\\我的角色"
          caption_ext: txt
          caption_dropout_rate: 0.05
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]   # 显存小先去掉 1024
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1    # 官方示例写作 gradient_accumulation，二选一
        train_unet: true
        train_text_encoder: false        # 大多数新模型不支持
        gradient_checkpointing: true
        noise_scheduler: flowmatch       # 按模型：flux/qwen/krea2 用 flowmatch
        optimizer: adamw8bit             # XPU 上自动回退普通 adamw
        lr: 1e-4
        dtype: bf16
      model:
        name_or_path: "Qwen/Qwen-Image"  # 换成你要训练的模型
        arch: qwen_image                 # 换成对应 arch
        quantize: true                   # 16GB 显存建议开
        qtype: int8
        quantize_te: true
        qtype_te: int8
        low_vram: true
      sample:
        sampler: flowmatch               # 与 noise_scheduler 一致
        sample_every: 100
        width: 768
        height: 768
        prompts:
          - "myconcept, a photo of ..."  # 触发词必须自己写
        neg: ""
        seed: 42
        guidance_scale: 4.0
        sample_steps: 30                 # 蒸馏模型（如 turbo）用 8
meta:
  name: "[name]"
  version: "1.0"
```

换模型时只动这几处：`arch`、`name_or_path`、`noise_scheduler`（及 `sampler`）、
`resolution`、采样步数/guidance。其它字段是通吃的。

## 5. 显存策略（XPU 实测）

16GB Intel Arc A770 上验证过的组合，按优先级：

1. `quantize: true` + `qtype: int8`（更激进可用 `uint3` + ARA 适配器）。
2. `layer_offloading: true` + `layer_offloading_transformer_percent: 1.0` +
   `layer_offloading_text_encoder_percent: 1.0`（模型层按百分比搬运到内存）。
3. `gradient_checkpointing: true`。
4. `resolution` 去掉最高档。
5. `cache_text_embeddings: true` + `cache_latents_to_disk: true`。

实测：768 档训练显存约 **8.7GB、19–22 秒/步**，未溢出。
不要靠系统"溢出到共享显存"硬扛——一旦溢出速度慢数倍。

## 6. 可选增强（不是最小路线必需的）

| 项目 | 说明 |
|---|---|
| 中文 UI | 覆盖 `ui/` 并 `npm install && npm run build`；不影响训练 |
| 训练曲线图 | 随新版 UI 自带（`loss_log.db` + API + uPlot）；同上，不影响训练 |
| FFmpeg 8.1 full-shared | 只有 torchcodec / 视频相关功能需要；纯文生图 LoRA 可跳过 |
| torchao 0.18 修复脚本 | 本机实验过；结论是直接钉 0.17，脚本只是备选 |

## 7. FAQ

**为什么 torchao 钉 0.17 不升 0.18？**
0.18 把量化权重改成新张量子类（Int8Tensor/Float8Tensor），XPU 后端缺算子注册
（int8 缺 `aten.view`、float8 缺 `aten.abs`），量化会静默失效、白占显存还变慢。
0.17 是旧量化路径，无此问题。详见 XPU 说明书 FAQ。

**为什么配置里写 adamw8bit 也能跑？**
002 补丁让 bitsandbytes 8bit 优化器在 XPU 上自动回退标准 AdamW，配置不用改。

**这套改造支持哪些模型？**
补丁在设备/内存/优化器层面，与具体模型无关——官方列出的 arch 都能训。
个别模型在 XPU 上可能有零散小问题（例如 Krea2 的 rope 需要 float32、
ComfyUI 单文件权重加载需 `strict=False`），属于模型级适配，不属于通用最小集。

**这套最小集和本机完整版差在哪？**
本机还做了中文 UI、曲线图验证、FFmpeg 配置等增强；最小集只含第 1、2 节，
练出 LoRA 完全够用。

## 8. 参考资料

- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) 官方 README 与示例配置
- Krea 官方 Training 文档 / RunComfy Krea 2 训练指南（素材数量与打标参考）
- HuggingFace diffusers DreamBooth 示例（风格 LoRA 打标规则）
- krea2edit-trainer README（edit 类 LoRA 的配对与显存实测）
