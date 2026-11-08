# CheXficient 图像分支接入检查（2026-11-08）

本轮先完成工程接入，随后于北京时间 **2026-11-08 14:37:16** 正式注册冻结特征实验；本接入报告中的合成输出不代表真实 AUC。当前线上最高反馈仍为 CXR-LT 的 **78.98**。公开模型的来源、固定 revision、24 项文件哈希及许可证见 [来源清单](chexficient_source_manifest_20261108.json)。

`StanfordAIMI/CheXficient@34bb5ea905c4cb504d2b73c9a6687a5d5082f52a` 的完整 checkpoint 已本地下载，780,793,036 字节，SHA-256 `f691262c3f77c3c850bebff420180602cf4ca5d5214449e377366f3205548336`。模型卡 frontmatter 写 small，但作者源码、配置与权重维度对应无 register token 的 ViT-Base/14，因此按实际权重构造。

适配器只读取 `image_encoder.model.*` 与 `image_projection.projection.*`，严格加载 177 个张量，共 86,482,688 个图像参数。无需构造文本模型或下载外部初始化。输入为 RGB 378×378，输出为 512 维 L2 归一化图像 embedding，同时可返回 768 维 CLS；这些输出不是本地十类预测。后续性能实验需要独立锁定方案，用官方 train 学习本地头并执行患者 OOF 门槛。

预处理遵循固定配置的 `BitImageProcessor`：最短边 bicubic 缩放到 378，长边取整数截断；中心裁剪使用 floor 偏移，随后 `/255` 和 CLIP mean/std 归一化。使用下载并固定哈希的 Transformers v4.51.3 原始 transform 函数体，对合成 RGB 与灰度输入逐元素比较，最大误差 **0**。该对照借用已安装库的 helper globals，NumPy 输入的 TF/JAX 类型谓词固定返回 false；没有替换或重写原始 transform 函数体。

项目 `.venv` 下六项合成测试全部通过。实际权重 CPU 检查得到 `2×768` CLS 与 `2×512` embedding，输出有限、冻结且 eval；重复 forward 与手工投影归一化的最大误差均为 **0**。报告为 `outputs/chexficient-preparation-20261108/cpu_engineering_check-v2.json`，绑定适配器 SHA-256 `f45aca9b42d21953aa459e43e5b665df1f22f0228e6457f5583bfd1d1ee3e6c3`。早期报告保留，不覆盖历史证据。

北京时间 **2026-11-08 14:06:45** 完成实际权重的合成 GPU 检查：固定 batch=4、预处理线程=2、CUDA float16 AMP，合成 RGB/灰度经处理后为 `4×3×378×378`，得到有限的 `4×768` CLS 与 `4×512` embedding。重复 forward 最大误差为 **0**，输出 L2 范数约为 1，模型冻结且 eval；峰值 allocated **570,921,472 字节**、reserved **717,225,984 字节**。报告 `outputs/chexficient-preparation-20261108/gpu_engineering_check.json` 绑定同一适配器与 checkpoint 哈希，没有读取真实图像、标签或计算 AUC。

后续真实数据实验的两个候选、患者分组、固定头参数与逐阶段晋级要求另见 [检查级增量实验方案](CHEXFICIENT_STUDY_PROBE_PLAN_20261108.md)。GPU 合成检查仅支持上述固定执行设置能够运行。

正式 runner 的 **21 项定向合成测试**与项目 **379 项 CPU 回归测试**全部通过，另有只读源码审查。注册目录 `outputs/chexficient-study-probe-20261108`，计划 SHA-256 为 `f71f2c370ecb5c78228884d0dd315a2c3481cf4875adc310af88ac5ae8e8acb4`，锁定 20 项源码文件及 127 项输入快照文件；valid/test CXR-LT 矩阵按阶段延迟物化。启动信息与实际性能结果以实验目录的 `experiment.json`、特征 `progress.json` 和后续 OOF 报告为准。

这项工程检查不证明性能收益、数据无重叠或赛事来源合规。其医学预训练来源及官方要求见 [规则核对](OFFICIAL_SCORING_20261108.md)；不隐藏来源或许可声明。
