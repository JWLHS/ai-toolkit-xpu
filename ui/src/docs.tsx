import React from 'react';
import { ConfigDoc } from '@/types';
import { IoFlaskSharp } from 'react-icons/io5';

const docs: { [key: string]: ConfigDoc } = {
  'config.name': {
    title: '训练名称',
    description: (
      <>
        训练任务的名称。系统用它来标识任务，也是最终模型的文件名。名称必须唯一，只能包含字母、数字、下划线和短横线，不能有空格或特殊字符。
      </>
    ),
  },
  gpuids: {
    title: 'GPU 编号',
    description: (
      <>
        用于训练的显卡。目前 UI 里一个任务只能选一张显卡，但可以同时启动多个任务，各自使用不同的显卡。
      </>
    ),
  },
  'config.process[0].trigger_word': {
    title: '触发词',
    description: (
      <>
        可选。这是用来触发你的概念或角色的词（触发词）。
        <br />
        <br />
        设置了触发词后：如果标注里没有它，会自动加在标注开头；如果完全没有标注，标注就只剩触发词。想让触发词出现在标注的不同位置，可以在标注里写 [trigger] 占位符，训练时会被替换成你的触发词。
        <br />
        <br />
        触发词不会自动加到测试提示词里，你需要手动添加，或在测试提示词里同样使用 [trigger] 占位符。
      </>
    ),
  },
  'config.process[0].model.name_or_path': {
    title: '名称或路径',
    description: (
      <>
        HuggingFace 上 diffusers 仓库的名字，或本地底模的路径。大多数模型需要 diffusers 格式的文件夹；SDXL、SD1 这类模型也可以直接填整合版 safetensors 的路径。
      </>
    ),
  },
  'datasets.control_path': {
    title: '控制数据集',
    description: (
      <>
        控制数据集里的文件名要与训练数据集一一对应，成对使用。这些图会作为控制图/输入图参与训练，并被缩放到与训练图相同的尺寸。
      </>
    ),
  },
  'datasets.multi_control_paths': {
    title: '多控制数据集',
    description: (
      <>
        控制数据集里的文件名要与训练数据集一一对应，成对使用。
        <br />
        <br />
        多个控制数据集会按列表顺序依次生效。如果模型不要求控制图与目标图同分辨率（例如 Qwen/Qwen-Image-Edit-2509），控制图不必匹配目标图的尺寸或长宽比，会自动缩放到模型/目标图最合适的分辨率。
      </>
    ),
  },
  'datasets.num_frames': {
    title: '帧数',
    description: (
      <>
        视频数据集缩放后的帧数。如果数据集是图片，填 1（一帧）。如果数据集全是视频，会从每个视频中等间隔抽帧。
        <br />
        <br />
        建议训练前先把视频裁成合适的长度。Wan 是每秒 16 帧，81 帧约等于 5 秒，所以最好把素材都裁到 5 秒左右。
        <br />
        <br />
        例如：设为 81，数据集里有两个视频（2 秒和 90 秒），两者都会抽出 81 帧，结果 2 秒的看起来变慢、90 秒的看起来飞快。
      </>
    ),
  },
  'datasets.do_i2v': {
    title: '启用 I2V',
    description: (
      <>
        对同时支持 I2V（图生视频）和 T2V（文生视频）的视频模型，这个选项会让该数据集按 I2V 训练：从视频里取第一帧作为起始图。不勾选则按 T2V 处理。
      </>
    ),
  },
  'datasets.do_audio': {
    title: '处理音频',
    description: (
      <>
        对支持音视频的模型，会从视频里读出音频并调整到与视频序列匹配。由于视频会被缩放，音频的音高可能被拉高或压低。训练前请把素材裁成正确的长度。
      </>
    ),
  },
  'datasets.audio_normalize': {
    title: '音频归一化',
    description: (
      <>
        加载音频时把音量归一化到峰值最大值。适合素材音量大小不一致的情况。注意：如果你的片段里有需要保留的纯静音，不要开启，它会把静音片段也放大。
      </>
    ),
  },
  'datasets.audio_preserve_pitch': {
    title: '保持音高',
    description: (
      <>
        当音频长度与训练目标帧数不一致时，这个选项会保持音高不变。建议素材本身就与目标长度一致，因为拉伸音频可能引入失真。
      </>
    ),
  },
  'datasets.flip': {
    title: '水平/垂直翻转',
    description: (
      <>
        可以在训练时即时做数据增强：水平翻转（X 轴）和/或垂直翻转（Y 轴）。翻转一个轴相当于把数据集翻倍（原图 + 翻转图）。很有用，但也要小心：把人上下翻转毫无意义，左右翻转人脸也可能让模型困惑（人的左右脸并不完全一样），文字翻转更是明显有害。
        <br />
        <br />
        控制图会跟着一起翻转，保证像素级对齐。
      </>
    ),
  },
  'train.unload_text_encoder': {
    title: '卸载文本编码器',
    description: (
      <>
        卸载文本编码器：只缓存触发词和采样提示词，然后把文本编码器从显存里卸掉。此时数据集里的标注会被忽略。
      </>
    ),
  },
  'train.cache_text_embeddings': {
    title: '缓存文本嵌入',
    description: (
      <>
        缓存文本嵌入：把文本编码器对所有标注算出的嵌入缓存到磁盘，之后把文本编码器从显存卸载。
        <br />
        <br />
        注意：它不适用于会动态改变提示词的设置（触发词、标注丢弃率等）。
      </>
    ),
  },
  'model.multistage': {
    title: '训练阶段',
    description: (
      <>
        有些模型是多阶段网络，去噪时分别使用不同阶段。最常见的是两阶段：一个负责高噪声，一个负责低噪声。你可以同时训练两个阶段，也可以只练其中一个。同时训练时，训练器每隔若干步在两个阶段间交替，并输出两个不同的 LoRA；只选一个阶段则只训练它并输出单个 LoRA。
      </>
    ),
  },
  'train.switch_boundary_every': {
    title: '切换边界间隔',
    description: (
      <>
        训练多阶段模型时，这个值决定训练器多久在阶段之间切换一次。
        <br />
        <br />
        低显存模式下，当前不训练的模型会被卸载出显存以省内存，而卸载/加载需要时间，所以低显存时建议少切换（10 或 20 这类值）。
        <br />
        <br />
        切换发生在批次（batch）层面，也就是在梯度累积的步骤之间切换。想在一个 step 内训练两个阶段，可以设为每 1 步切换、并把梯度累积设为 2。
      </>
    ),
  },
  'train.force_first_sample': {
    title: '强制首次采样',
    description: (
      <>
        开启后，训练器启动时一定会先出一张采样图。默认情况下，只有当没有任何已训练内容时才会出首张采样图，续训时不会。这个选项让每次启动训练都强制出一张，适合改了采样提示词、想立刻看到效果的场景。
      </>
    ),
  },
  'model.layer_offloading': {
    title: (
      <>
        层级卸载{' '}
        <span className="text-yellow-500">
          ( <IoFlaskSharp className="inline text-yellow-500" name="Experimental" /> Experimental)
        </span>
      </>
    ),
    description: (
      <>
        这是一个基于{' '}
        <a className="text-blue-500" href="https://github.com/lodestone-rock/RamTorch" target="_blank">
          RamTorch
        </a>
         的实验性功能。它还处于早期阶段，后续会频繁更新与调整，因此不同版本间可能表现不一致，并且只适用于部分模型。
        <br />
        <br />
        层级卸载会使用 CPU 内存来承载模型的大部分权重，而不是占用显存。只要内存足够，就能在小显存的显卡上训练更大的模型。它比纯显存训练慢，但内存更便宜、也更容易升级。优化器状态和 LoRA 权重仍然需要显存，所以显卡通常还是要够大。
        <br />
        <br />
        你也可以选择需要卸载的层数百分比。一般来说，为了性能最好尽量少卸载（接近 0%）；显存不足时再往上调。
      </>
    ),
  },
  'model.qie.match_target_res': {
    title: '匹配目标分辨率',
    description: (
      <>
        This setting will make the control images match the resolution of the target image. The official inference
        example for Qwen-Image-Edit-2509 feeds the control image is at 1MP resolution, no matter what size you are
        generating. Doing this makes training at lower res difficult because 1MP control images are fed in despite how
        large your target image is. Match Target Res will match the resolution of your target to feed in the control
        images allowing you to use less VRAM when training with smaller resolutions. You can still use different aspect
        ratios, the image will just be resizes to match the amount of pixels in the target image.
      </>
    ),
  },
  'train.diff_output_preservation': {
    title: '差异化输出保持',
    description: (
      <>
        DOP（差异化输出保持）是一种在训练中保住原模型对该类概念认知的技术。每一步除了正常训练，还会用带类提示词、LoRA 关闭状态下的预测（先验预测）再跑一步，教 LoRA 保留该类别的知识。它既能提升训练效果，也能让你写出「Alice 站在一个女人旁边」这种提示词时，不会把两个人都画成 Alice。
      </>
    ),
  },
  'train.blank_prompt_preservation': {
    title: '空提示词保持',
    description: (
      <>
        BPP（空提示词保持）用来保住模型在无提示词时的原有能力。每一步都会用空白提示词、LoRA 关闭状态做一次先验预测，再用这个预测作为额外一步训练的目标。它能让模型更灵活，在推理端用 CFG 时对概念质量也有帮助，避免模型过度依赖提示词、丢掉泛化能力。
      </>
    ),
  },
  'train.do_differential_guidance': {
    title: '差分引导',
    description: (
      <>
        差分引导会把模型预测与目标之间的差放大，构造出一个新的目标；差分引导强度就是放大的倍数。功能仍属实验性，但在我的测试里，它让模型学得更快、细节也更好。
        <br />
        <br />
        原理是：普通训练每一步只会朝目标靠近一点（受学习率限制，永远差一点）。放大差值后，新目标会超过真实目标，于是模型会学着「命中甚至略微超过」目标，而不是总是差一点。
        <br />
        <br />
        <img src="/imgs/diff_guidance_cn_clean.svg" alt="差分引导原理图" className="max-w-full mx-auto rounded-lg shadow-lg" />
      </>
    ),
  },
  'dataset.num_repeats': {
    title: '重复次数',
    description: (
      <>
        重复次数：让数据集里的样本在训练中被重复若干遍。多个数据集搭配使用时，可以用它来平衡各自的出现频率。例如小数据集 10 张、大数据集 100 张，把小数据集设为重复 10 次，两者在训练中出现的概率就一样了。
      </>
    ),
  },
  'train.audio_loss_multiplier': {
    title: '音频损失倍率',
    description: (
      <>
        训练音视频时，视频损失的数值有时会远大于音频损失，导致音频学不好甚至失真。出现这种情况可以调高音频损失倍率（例如 2.0、10.0）。注意：调太高会过拟合并损伤模型。
      </>
    ),
  },
  'datasets.auto_frame_count': {
    title: '自动帧数',
    description: (
      <>
        自动帧数：为数据集里的每个视频单独决定帧数，而不是统一用固定帧数。这样可以在同一数据集里放不同长度的视频，且不会被加速或减速。注意长视频会占用更多显存；目前 batch size 大于 1 时不可用。
      </>
    ),
  },
  'model.model_kwargs.kv_cache': {
    title: 'KV 缓存',
    description: (
      <>
        为支持 KV 缓存的模型开启控制图 KV 缓存。用它训练出来的 LoRA 在推理时也要开启（反之亦然）。它不影响训练速度，但推理时控制图只需处理一次而不是每一步都处理，能显著加速推理。
      </>
    ),
  },
  'train.guidance_loss_target': {
    title: '引导损失目标',
    description: (
      <>
        用于对比引导损失：这是要把预测放大到的目标 CGF 值。
      </>
    ),
  },
  'datasets.caption_dropout_rate': {
    title: '全局打标丢弃率',
    description: (
      <>
        标注丢弃率：每一步训练时，某张图的标注被丢弃（替换成空标注）的概率。例如 0.05 表示大约 5% 的步数会丢掉标注。
        <br />
        <br />
        丢弃标注能让模型不依赖文字去学概念，并保住无提示词生成的能力。如果设了触发词，丢弃标注时仍会保留触发词，所以模型依然把丢标注的样本和触发词关联起来；正则化图片（没有触发词的图）会直接变成全空标注。
        <br />
        <br />
        缓存文本嵌入时同样支持标注丢弃：会额外缓存一份丢弃后的嵌入（空标注，或只有触发词），训练时按这个概率随机替换。
      </>
    ),
  },
};

export const getDoc = (key: string | null | undefined): ConfigDoc | null => {
  if (key && key in docs) {
    return docs[key];
  }
  return null;
};

export default docs;
