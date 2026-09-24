# 可选的 XPU int8 加速轮子（omni_xpu_kernel）

这两个 wheel 是**可选的加速后端**，装不装都不影响正常训练；不装时一切走原来的
torchao int8（见下面"回退行为"）。

> **从 Release 下载**（推荐，仓库里不再保存二进制）：
> <https://github.com/JWLHS/ai-toolkit-xpu/releases/tag/omni-wheels-0.2.0>
>
> ```bat
> :: A 系列（dg2）
> curl -L -O https://github.com/JWLHS/ai-toolkit-xpu/releases/download/omni-wheels-0.2.0/omni_xpu_kernel-0.2.0b1+torch213.dg2-cp313-cp313-win_amd64.whl
> :: B 系列（bmg）
> curl -L -O https://github.com/JWLHS/ai-toolkit-xpu/releases/download/omni-wheels-0.2.0/omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-cp313-win_amd64.whl
> ```
>
> 下面的文件名 / 架构对照表在 Release 页面同样适用。

| 文件（在 Release 附件里） | 适用显卡 | 适用 torch | 来源 |
| --- | --- | --- | --- |
| `omni_xpu_kernel-0.2.0b1+torch213.dg2-cp313-cp313-win_amd64.whl` | **Arc A 系列**（A770/A750/A580/A380） | **2.13（当前仓库默认）** | 本仓库编译 |
| `omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-cp313-win_amd64.whl` | **Arc B 系列**（B580/B570 等） | 2.13 | 上游提供 |
| （可选）`omni_xpu_kernel-0.2.0b1+torch213.dg2.2-...whl` | Arc A 系列，上游更新的小版本 | 2.13 | [上游 Release](https://github.com/Blackwood416/omni-xpu-kernel/releases) |

> **没有 `+torch214` 的轮子**：本仓库钉 torch 2.13（原因见
> [XPU_ADAPTATION_GUIDE.md 第 8 节](../XPU_ADAPTATION_GUIDE.md)），
> 2.14 在本机实测训练必崩（驱动级 0xC0000005），因此不发 2.14 的轮子。
> 上游 [Blackwood416/omni-xpu-kernel Releases](https://github.com/Blackwood416/omni-xpu-kernel/releases)
> 有 `+torch214` 构建，但同样会撞上该崩溃——除非你是纯推理场景。

## 安装前必须对上的三件事

轮子文件名就是约束，**任一条不符就别装**（装了也会被检测拒绝并回退）：

1. **Python 3.13**（`cp313`）—— 本仓库 `uv sync` 装的默认就是 3.13；
2. **PyTorch 版本要跟轮子标签一致**：仓库当前锁的是 **torch 2.13.0+xpu**，
   所以要用 `+torch213` 的轮子（文件名里带标签，装错会被自动检测并回退）；
3. **显卡架构匹配**（`dg2` = A 系列，`bmg` = B 系列）。

**不需要**安装 oneAPI：轮子运行时不依赖 oneAPI（自带 `dnnl.dll`，SYCL 运行时由
torch 的 XPU 依赖包提供）；oneAPI 只在**自己编译**时才需要。

## 安装

```bat
:: A 系列（A770/A750/A580/A380）—— 先把 whl 下载到当前目录
.venv\Scripts\python.exe -m pip install --no-deps omni_xpu_kernel-0.2.0b1+torch213.dg2-cp313-cp313-win_amd64.whl

:: B 系列（B580/B570 等）
.venv\Scripts\python.exe -m pip install --no-deps omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-cp313-win_amd64.whl
```

用 uv 环境也可以：

```bat
uv pip install --python .venv\Scripts\python.exe --no-deps <下载好的 whl>
```

自检（会打印目标架构、精度、速度，并确认与你的显卡匹配）：

```bat
.venv\Scripts\python.exe scripts\verify_omni_xpu_backend.py
```

## 装完之后有什么可选项

- **Web UI**：训练任务的「量化」下拉里会多出
  **`xpu_int8（Intel XPU 加速）`** 和 **`xpu_fp8（实验）`**，直接选即可，不用改代码；
- **CLI**：配置里写 `qtype: xpu_int8`。

实测（A770，krea2，768 档，100% 层级卸载）：

| 量化 | 步时 | 训练稳态显存 | 训练进程内存 |
| --- | --- | --- | --- |
| `int8`（torchao，原路径） | 24.97 s/it | 6.6 GB | ~19.5 GB |
| **`xpu_int8`（本后端）** | **11.11 s/it（2.25×）** | **5.0 GB** | **16.2 GB** |

## 回退行为（没装 / 装错 / 版本不符时）

在 UI 里手选 `xpu_int8` 之后：

1. 后端会做三项检查：能否 import、`__xpu_target__` 是否与你的显卡架构一致、
   轮子编译用的 torch 主版本是否与当前一致；
2. **任一条不满足 → 自动回退到 torchao int8**，训练照常进行；
3. 日志里会打印一行原因，例如
   `[XPU Int8] 回退到 torchao int8：未安装 omni_xpu_kernel（ModuleNotFoundError）`
   或 `...：wheel 是 bmg 版，当前设备是 dg2 系列（装错了架构）`。

所以"选了但没装"是安全的，不会跑不起来。

## 环境变量（都可选）

| 变量 | 作用 |
| --- | --- |
| `AI_TOOLKIT_XPU_INT8=off` | 强制关掉该后端（排查用），选了也会走 torchao |
| `AI_TOOLKIT_MEM_DEBUG=1` | 打印量化阶段的内存/大张量诊断信息 |

日常使用**不需要设置任何变量**。
