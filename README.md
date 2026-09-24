# ai-toolkit XPU（Intel Arc）适配版

> 让 ai-toolkit 在 **Intel Arc / oneAPI (XPU)** 上直接训练和出图。
> 实测环境：Windows 11 + **Intel Arc A770 16GB** + Python 3.13；
> Python 3.11 / 3.12 / 3.13 均可（`pyproject.toml` 声明 `>=3.11,<3.14`）。

**新用户只要两条命令**：装好 Git 和 Intel 显卡驱动，然后

```bat
git clone https://github.com/JWLHS/ai-toolkit-xpu
cd ai-toolkit-xpu
setup_xpu.bat      :: 第一次：准备 Python(uv) + XPU 依赖 + FFmpeg + UI 依赖（10~30 分钟）
run_xpu.bat        :: 以后每次：启动中文 Web UI，自动打开 http://localhost:8675
```

完整步骤、耗时和排错见下一节。

## 新用户上手：从克隆到开始训练

### 0. 前置条件（只有这几样）

| 需要 | 说明 |
| --- | --- |
| Windows 10/11 x64 | 本 fork 的一键脚本面向 Windows；Linux 请照 [XPU_ADAPTATION_GUIDE.md](XPU_ADAPTATION_GUIDE.md) 手动做 |
| Intel Arc 独显 + 最新显卡驱动 | XPU 运行时随驱动安装，**不需要**单独装 oneAPI 编译器 |
| Git | `setup_xpu.bat` 会自动补 uv / Node / FFmpeg，但不会替你装 Git |
| 磁盘 ≥ 15GB | Python 环境约 5.5GB + UI 依赖约 1GB + 模型权重（另算） |

### 1. 首次初始化（只跑一次）

```bat
setup_xpu.bat
```

它按顺序做四件事，可重复执行（幂等，中断了重跑即可）：

1. 检查 Git / uv / Node，缺的自动装（uv 用官方安装脚本，Node 用 winget）
2. `uv python install` + `uv sync`：按 `pyproject.toml` + `uv.lock` 建 `.venv`（Python 3.13），装 XPU 版 torch / torchao / triton-xpu（约 5.5GB，71 秒实测是千兆网速下的本机结果）
3. 下载 FFmpeg 8.1 full-shared 到 `ffmpeg-shared/`（torchcodec 依赖，约 70MB）
4. 装 UI 依赖 `ui/node_modules` 并生成 Prisma 数据库客户端

最后打印出 `xpu available: True` 和你的显卡名，就算成了。

耗时：网络好 10 分钟上下；国内网络建议先把脚本开头的 `USE_CN_MIRROR` 改成 `1`
（PyPI 走阿里云镜像、模型下载走 hf-mirror，官方源兜底；torch/torchao 的 XPU
轮子没有国内镜像，仍走官方 PyTorch 索引，实测国内可达）。

### 1.5 资源监控（不需要额外装东西）

UI 里 GPU/CPU 卡片的数据是这套顺序取到的，**全自动**：

| 顺序 | 数据源 | 需要装什么 | 能读到 |
| --- | --- | --- | --- |
| 1（默认） | **Level Zero Sysman**（`ze_loader.dll`，随 Intel 显卡驱动安装） | 无 | 型号、显存、负载、温度、功耗、频率 |
| 2（可选） | Intel XPU Manager（`xpu-smi`） | 手动装；装了就作为第二顺位 | 同上，另有多媒体引擎/带宽等（UI 未展示） |
| 3（兜底） | `torch.xpu` | 环境里本来就有 | 型号、显存（无温度/功耗/频率） |

想强制指定数据源，设环境变量 `AI_TOOLKIT_XPU_MONITOR=zes|xpu-smi|torch`（不加就是自动）。
风扇转速不显示：Intel 的 Sysman 在 Arc 上返回“无转速传感器”，`xpu-smi` 的指标表里也没有这一项。

### 2. 启动 Web UI（每次）

```bat
run_xpu.bat
```

脚本会自动挑 `.venv`（没有则回退 `.venv-xpu`）、把 FFmpeg 加进 PATH、
缺 UI 依赖就自动 `npm install` + `prisma generate/db push` + `npm run build`，
然后起在 <http://localhost:8675> 并打开浏览器。UI 是**中文界面**
（词条在 [ui_i18n/](ui_i18n/)，用 `scripts/localize_ui.py` 应用），loss 曲线图正常显示。

训练流程就是 UI 里点「新建训练任务」：选模型、填素材目录、设步数和分辨率。
16GB 显存跑大模型时，在任务高级设置里打开「层级卸载」（把权重放到内存）和
「低显存」相关选项，能明显减少溢出到共享显存。

### 3. 命令行训练（可选，无 UI）

```bat
.venv\Scripts\python.exe run.py config\你的配置.yaml
```

`config/` 目录里有可直接照抄的配置示例；训练日志和 loss 曲线会写到 `output/` 和 `logs/`。

### 4. 出问题先看这几条

| 现象 | 处理 |
| --- | --- |
| `torch.xpu.is_available()` 为 False | 更新 Intel 显卡驱动；确认用的是 Arc 独显而不是核显 |
| 下载卡住 / 超时 | 把 `USE_CN_MIRROR` 改成 `1`，重跑 `setup_xpu.bat` |
| 打开 8675 没反应 | 先跑完 `setup_xpu.bat`（装 UI 依赖）再 `run_xpu.bat`；残留进程占端口就跑 `stop_xpu.bat` 再重开 |
| 跑完想彻底关掉 | 双击 `stop_xpu.bat`：只结束本目录下的 node/python（含正在跑的训练、TensorBoard），不会误杀其它程序 |
| 显存溢出到共享显存 | 调高「层级卸载」比例、降低分辨率/桶尺寸；原理与调优见 [XPU_ADAPTATION_GUIDE.md](XPU_ADAPTATION_GUIDE.md) |
| 想换 Python 版本 | `uv python pin 3.12` 然后 `uv sync`，不用删环境 |

## 版本

| 部分 | 版本 |
| --- | --- |
| 后端代码基线 | 上游 `0.12.27` → 本 fork 记为 **`0.12.27+xpu`**（见 `version.py` / `pyproject.toml`） |
| UI | 同步到上游 **0.13.6**（中文词条在 `ui_i18n/`） |
| 关键依赖 | torch `2.13.0+xpu`、torchvision `0.28.0+xpu`、torchao `0.17.0+xpu`、triton-xpu `3.7.2` |
| Python | `>=3.11,<3.14`，`.python-version` 默认钉 `3.13` |

模型支持：本 fork 已跟进上游的 **`toolkit/models/v2` 模型栈**，因此
**Qwen-Image 2.1（含参考图训练）已在 A770 上实测可训练**（512 档 5.7~6.6 s/it、显存峰值 10.3GB），
Ming-Image 也已接入但本机无权重、未实测。详见 [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md)。

## 可选：XPU int8 加速后端（需手动安装，按显卡架构选）

从 **[Releases](https://github.com/JWLHS/ai-toolkit-xpu/releases/tag/omni-wheels-0.2.0)**
下载**可选**的加速轮子（`omni_xpu_kernel`），装上之后
Web UI 的「量化」下拉里会多出 `xpu_int8` / `xpu_fp8` 两个选项；**不装也完全不影响**
（默认照旧走 torchao int8）。

| 你的显卡 | 下载哪个（Release 附件） |
| --- | --- |
| Intel Arc **A 系列**（A770/A750/A580/A380） | `omni_xpu_kernel-0.2.0b1+torch213.dg2-cp313-*.whl` |
| Intel Arc **B 系列**（B580/B570 等） | `omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-*.whl` |

```bat
:: 1) 到 Release 页面下载对应架构的 whl
:: 2) 安装（必须 Python 3.13 + torch 2.13，仓库默认就是）
.venv\Scripts\python.exe -m pip install --no-deps <下载好的 whl 路径>
```

**回退**：在 UI 里选了 `xpu_int8` 但没装轮子/装错架构/版本不符时，会自动回退到
torchao int8 并在日志里给出原因，训练不会失败。

**不需要任何环境变量**，也不需要装 oneAPI。细节（含实测数据）见
[wheels/README.md](wheels/README.md)。

版本号沿用上游基线的原因是这个 fork 是"跟着上游走"的改造版：改动越少越容易跟上
官方更新，`+xpu` 只用于区分"这是改造版"。

## 关于本仓库

本仓库**修改自原版** [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
（代码基线官方 0.12.27，UI 已对齐官方 0.13.6，[Apache-2.0](LICENSE) 许可），并非独立项目。
与原版的主要差异：

- 新增 5 个 XPU 补丁（Intel Arc / oneAPI 支持，见 [xpu_patches/](xpu_patches/)）
- 依赖改用 `pyproject.toml` + `uv.lock` 锁定：torch / torchvision / torchaudio / torchao / triton-xpu
  指向 XPU 轮子（torch 2.13.0+xpu、torchao 0.17.0+xpu），其余依赖跟随官方版本
- 新增一键脚本 `setup_xpu.bat` / `run_xpu.bat`，克隆后自动补全环境与 UI 依赖
- 中文 UI：词条词典 [ui_i18n/zh-CN.json](ui_i18n/zh-CN.json) + 应用脚本 [scripts/localize_ui.py](scripts/localize_ui.py)，
  UI 代码本体同步到官方 0.13.6，后续跟版只补词典
- 修复 XPU 显存一直涨/溢出到共享显存（分块搬运 + 分配器缓存回收），
  以及 XPU 不支持 fp64 的算子路径（rope 等改为 fp32）
- 移除 `docker/`：本 fork 的依赖文件面向 XPU，容器（NVIDIA/CUDA）部署请用上游仓库

文档：

- [XPU_ADAPTATION_GUIDE.md](XPU_ADAPTATION_GUIDE.md)：改造说明（环境、补丁、显存调优、FAQ）
- [AITOOLKIT_XPU_MINIMAL_GUIDE.md](AITOOLKIT_XPU_MINIMAL_GUIDE.md)：素材准备与最小训练配置
- [xpu_patches/](xpu_patches/)：从官方仓库重新打补丁的 5 个补丁

## 致谢 / Credits

- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)：上游项目，本仓库的全部训练能力都来自它。
- [allanmeng/ComfyUI-XPUSYS-Monitor](https://github.com/allanmeng/ComfyUI-XPUSYS-Monitor)：`scripts/xpu_metrics_zes.py`
  的 Level Zero Sysman 用法（结构体偏移、初始化顺序）参考了该项目的 `providers/intel.py`。
- [lodestone-rock/RamTorch](https://github.com/lodestone-rock/RamTorch)：层级卸载（把权重放到内存）功能来自上游。
- Intel oneAPI / PyTorch XPU：`torch 2.13.0+xpu`、`torchao 0.17.0+xpu` 等轮子的提供方。

原版 README 保留在下方（"Ostris AI Toolkit" 起），版权归原项目所有。

---

# Ostris AI Toolkit

AI Toolkit is an easy to use all in one training suite for diffusion models. I try to support all the latest models on consumer grade hardware. Image and video models. It can be run as a GUI or CLI. It is designed to be easy to use but still have every feature imaginable. Free and open source.



## Supported Models

### Image
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream I1)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [zhen-nan/L2P](https://huggingface.co/zhen-nan/L2P) (Z-Image L2P)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (ERNIE-Image)
- [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Nucleus-Image)
- [Boogu/Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) (Boogu Image 0.1)
- [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) (HiDream O1)
- [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (Ideogram 4 FP8)
- [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) (PRXPixel)
- [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers) (Anima)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo)
- [microsoft/Mage-Flow-Base](https://huggingface.co/microsoft/Mage-Flow-Base) (Mage-Flow)

### Instruction / Edit
- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (FLUX.1-Kontext-dev)
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Qwen-Image-Edit)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Qwen-Image-Edit-2509)
- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Qwen-Image-Edit-2511)
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) (HiDream E1)
- [Boogu/Boogu-Image-0.1-Edit](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit) (Boogu Image Edit)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2 Edit Training)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo Edit Training)
- [microsoft/Mage-Flow-Edit-Base](https://huggingface.co/microsoft/Mage-Flow-Edit-Base) (Mage-Flow Edit)

### Video
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMaxAI/MiniMax-H3)

### Audio
- [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (Ace Step 1.5)
- [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base) (Ace Step 1.5 XL)

### Experimental
- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)

## Installation

### Install with the AI Toolkit Manager (experimental)

The recommended way to install and run AI Toolkit is with the **AI Toolkit
Manager**, built into this repo. The manager detects your hardware and sets up
the right PyTorch build, creates the python environment, and grabs local copies
of Node.js and FFmpeg — everything stays inside the ai-toolkit folder, nothing
is installed system-wide. On every launch the manager checks for updates and
applies them (your local changes are never overwritten — if you have modified
files, the update is skipped with a warning), then starts the UI at
`http://localhost:8675`.

The manager is still **experimental** — please let me know if you have any
issues with it. The manual instructions below still work if you prefer them
or run into problems.

The only requirement is **git** (on Windows the manager can even fetch a
portable git for updates, but you need one installed to clone the repo first).

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
```

Then start the manager with the script for your platform:

Linux:
```bash
chmod +x run_linux.sh
./run_linux.sh
```

MacOS (Apple Silicon, experimental):
```bash
chmod +x run_mac.zsh
./run_mac.zsh
```

Windows: double-click `run_windows.bat` (or run it from a terminal).

You can also use the manager directly from a terminal (handy on headless
servers):

```bash
python3 -m manager install   # first-time setup
python3 -m manager update    # pull updates + sync dependencies
python3 -m manager launch    # start the UI
python3 -m manager doctor    # diagnose problems
```

### Manual installation

Requirements:
- python >=3.10 (3.12 recommended)
- Nvidia GPU with enough ram to do what you need
- python venv
- git


Linux:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install torch first
pip3 install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip3 install -r requirements.txt
```

For devices running **DGX OS** (including DGX Spark), follow [these](dgx_instructions.md) instructions.


Windows:

If you are having issues with Windows. I recommend using the easy install script at [https://github.com/Tavris1/AI-Toolkit-Easy-Install](https://github.com/Tavris1/AI-Toolkit-Easy-Install)

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```


# AI Toolkit UI

<img src="https://ostris.com/wp-content/uploads/2025/02/toolkit-ui.jpg" alt="AI Toolkit UI" width="100%">

The AI Toolkit UI is a web interface for the AI Toolkit. It allows you to easily start, stop, and monitor jobs. It also allows you to easily train models with a few clicks. It also allows you to set a token for the UI to prevent unauthorized access so it is mostly safe to run on an exposed server.

## Running the UI

Requirements:
- Node.js > 20

The UI does not need to be kept running for the jobs to run. It is only needed to start/stop/monitor jobs. The commands below
will install / update the UI and it's dependencies and start the UI. 

```bash
cd ui
npm run build_and_start
```

You can now access the UI at `http://localhost:8675` or `http://<your-ip>:8675` if you are running it on a server.

## Securing the UI

If you are hosting the UI on a cloud provider or any network that is not secure, I highly recommend securing it with an auth token. 
You can do this by setting the environment variable `AI_TOOLKIT_AUTH` to super secure password. This token will be required to access
the UI. You can set this when starting the UI like so:

```bash
# Linux
AI_TOOLKIT_AUTH=super_secure_password npm run build_and_start

# Windows
set AI_TOOLKIT_AUTH=super_secure_password && npm run build_and_start

# Windows Powershell
$env:AI_TOOLKIT_AUTH="super_secure_password"; npm run build_and_start
```

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all 
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

IMPORTANT. If you press crtl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving

### Need help?

Please do not open a bug report unless it is a bug in the code. You are welcome to [Join my Discord](https://discord.gg/VXmU2f5WEU)
and ask for help there. However, please refrain from PMing me directly with general question or support. Ask in the discord
and I will answer when I can.

## Ostris Cloud

You can use many cloud providers to rent GPUs. If you want to help support this project in the largest way possible, please consider using [Ostris Cloud](https://cloud.ostris.com). Ostris Cloud is owned and operated by me, Ostris, and every dollar earned goes directly back into funding the development of this project.

<a href="https://cloud.ostris.com" target="_blank"><img src="https://cloud.ostris.com/api/og" alt="Ostris Cloud" style="max-width:100%;width:600px;height:auto;"></a>


## Training in RunPod
If you would like to use Runpod, but have not signed up yet, please consider using [my Runpod affiliate link](https://runpod.io?ref=h0y9jyr2) to help support this project.


I maintain an official Runpod Pod template here which can be accessed [here](https://console.runpod.io/deploy?template=0fqzfjy6f3&ref=h0y9jyr2).

I have also created a short video showing how to get started using AI Toolkit with Runpod [here](https://youtu.be/HBNeS-F6Zz8).

## Training in Modal

### 1. Setup
#### ai-toolkit:
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```
#### Modal:
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`).

#### Hugging Face:
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, or .png images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config file located at ```config/examples/modal``` to the `config` folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:
  
   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function` _(default is A100 40GB and 2 hour timeout)_.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`. 
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

### Screenshot from Modal

<img width="1728" alt="Modal Traning Screenshot" src="https://github.com/user-attachments/assets/7497eb38-0090-49d6-8ad9-9c8ea7b5388b">

---

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Currently, the only supported
formats are jpg, jpeg, and png. Webp currently has issues. The text files should be named the same as the images
but with a `.txt` extension. For example `image2.jpg` and `image2.txt`. The text file should contain only the caption.
You can add the word `[trigger]` in the caption file and if you have `trigger_word` in your config, it will be automatically
replaced. 

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios. 


## Training Specific Layers

To train specific layers with LoRA, you can use the `only_if_contains` network kwargs. For instance, if you want to train only the 2 layers
used by The Last Ben, [mentioned in this post](https://x.com/__TheBen/status/1829554120270987740), you can adjust your
network kwargs like so:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks.7.proj_out"
            - "transformer.single_transformer_blocks.20.proj_out"
```

The naming conventions of the layers are in diffusers format, so checking the state dict of a model will reveal 
the suffix of the name of the layers you want to train. You can also use this method to only train specific groups of weights.
For instance to only train the `single_transformer` for FLUX.1, you can use the following:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks."
```

You can also exclude layers by their names by using `ignore_if_contains` network kwarg. So to exclude all the single transformer blocks,


```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          ignore_if_contains:
            - "transformer.single_transformer_blocks."
```

`ignore_if_contains` takes priority over `only_if_contains`. So if a weight is covered by both,
if will be ignored.

## LoKr Training

To learn more about LoKr, read more about it at [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md). To train a LoKr model, you can adjust the network type in the config file like so:

```yaml
      network:
        type: "lokr"
        lokr_full_rank: true
        lokr_factor: 8
```

Everything else should work the same including layer targeting.


## Support My Work

If you enjoy my projects or use them commercially, please consider sponsoring me. Every bit helps! 💖

<a href="https://ostris.com/sponsors" target="_blank"><img src="https://ostris.com/wp-content/uploads/2025/05/support-banner2.png" alt="Support my work" style="max-width:100%;height:auto;"></a>

### Current Sponsors

All of these people / organizations are the ones who selflessly make this project possible. Thank you!!

<a href="https://ostris.com/sponsors"><img src="https://ostris.com/sponsors.svg" alt="Sponsors" style="width:100%;height:auto;"></a>
