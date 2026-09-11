# UI 汉化方案（词典式）

## 为什么这样做

以前是把中文**直接写进** UI 源码（约 20 个文件、300+ 处），结果是：

- 每次上游更新 UI 都要手工合并（否则丢新功能），
- 我们的 UI 长期落后（曾整缺 17 个上游文件：数据集管理、实时监控等）。

现在改成：**上游 UI 保持原样 + 一份词典 + 一个应用脚本**。

## 用法

```bash
# 合并上游 UI 后（ui/src 为上游原版），套用中文
python scripts/localize_ui.py

# 只检查不写入
python scripts/localize_ui.py --check

# 对别处的目录套用（例如验证用的副本）
python scripts/localize_ui.py --src /path/to/ui/src
```

脚本会输出：词典条目数、更新文件数、命中条目数、**未命中条目**（说明上游改了文案，需要补词典）。

## 文件

| 文件 | 说明 |
|---|---|
| `ui_i18n/zh-CN.json` | 词典：`英文原文 -> 中文`（283 条，2026-09-11） |
| `scripts/localize_ui.py` | 应用脚本（幂等，可重复运行） |
| `ui_i18n/pending-review.json` | 自动提取时未配对的条目（含我们自加文案） |

## 当前状态（2026-09-11）

- UI 已整体切换为上游 **0.13.6** 版本（144 个文件），词典套用后 67/138 个源文件含中文。
- 核心界面已中文：任务创建 / 任务列表 / 仪表盘 / 数据集页 / 设置 / 侧边栏 / 损失曲线相关。
- 仍在英文的部分：新增页面的少量文案（监控、音频、标注批处理）、`docs.tsx` 内的长文说明。
  可用 `python - <<'EOF'` 扫描（见下）或直接跑 `localize_ui.py --check` 看未命中项。

## 非上游的自定义改动（合并时需注意）

这些是**我们自己的 UI 定制**，不属于翻译，上游没有对应字符串，合并新版后需要单独回贴：

1. **XPU 显卡信息**：`src/components/GPUWidget.tsx`、`src/components/GPUMonitor.tsx`、
   `src/app/api/gpu/route.ts` —— 用于在 Intel Arc/XPU 上正确显示显存、核心频率、功耗。
   （上游版本面向 NVIDIA / Apple GPU；合并后如显卡面板数据异常，从这里回贴。）
2. **时间步/损失的说明文案**：`src/app/jobs/new/SimpleJob.tsx` 里形如
   `linear ｜ 线性 ｜ 均匀采样噪声强度（0→1）｜ …` 的下拉说明（我们自加，上游没有）。
3. **侧边栏致谢**：`src/components/Sidebar.tsx` 的 `由 Doc_workBox 汉化`。
4. **曲线图日志默认值**：`src/app/jobs/new/jobConfig.ts` 的
   `logging: { log_every: 1, use_ui_logger: true }`
   —— **上游 0.13.6 已自带**，无需再回贴。

## 后续增量翻译流程

```bash
# 1) 看还有哪些英文没翻（脚本内置报告）
python scripts/localize_ui.py --check
# 2) 把缺失的英文原文 -> 中文写进 ui_i18n/zh-CN.json
# 3) 重新套用并重建
python scripts/localize_ui.py && (cd ui && npm run build)
```
