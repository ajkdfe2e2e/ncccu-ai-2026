# CheXficient 检查级增量实验固定方案（2026-11-08）

本方案在真实图像特征提取、训练与候选 AUC 计算之前写定。当前保留的 CXR-LT 模型为 `outputs/cxrlt-supervised-study-probe-20261108`，训练患者完整 OOF Macro-AUC 0.8270384304066101、official valid 0.8335505264258105、用户反馈线上 78.98。患者聚合候选虽有更高 valid 点估计，但未晋级；不作为本轮采用的基准，不重试其系数。

## 假设与唯一一组候选

检验冻结的 CheXficient 图文对比学习图像表征是否提供当前监督 CXR-LT 特征之外的可迁移信息。仅运行两个完整十类 BCE 线性头：

1. `chexficient`：每图 512 维已 L2 归一化的投影 embedding，同检查全部图像特征等权均值，共 512 维。
2. `joint`：按相同图像与检查顺序，将当前 CXR-LT 的 1,054 维检查均值特征与上述 512 维特征拼接，共 1,566 维。

不增加未投影 CLS 候选、概率融合系数、患者聚合、TTA、视图加权、按类选型或阈值搜索。编码器完全冻结并保持 eval；每次检查一个训练样本。头为普通十输出 BCE、L2=0.01、max_iter=200、seed=20261108、CPU float64 LBFGS、class_balance=False。每个 held 组拟合全新头，标准化仅由拟合患者计算，常数列规则与原已验证的 `fit_linear_probe` 一致。

## 公开来源与工程前提

- 固定 `StanfordAIMI/CheXficient@34bb5ea905c4cb504d2b73c9a6687a5d5082f52a`，完整 checkpoint SHA-256 `f691262c3f77c3c850bebff420180602cf4ca5d5214449e377366f3205548336`，780,793,036 字节。来源、24 项文件哈希及版权声明见 [来源清单](chexficient_source_manifest_20261108.json)。不重新下载或使用浮动版本。
- 通过 `cxr/chexficient_transfer.py` 仅严格加载 177 个图像分支张量，实际为无 register 的 ViT-Base/14、768 维 CLS、线性投影到 512 维并逐图 L2 归一化；不加载文本分支或外部初始化。
- RGB，最短边 bicubic 缩放到 378，长边整数截断，中心裁剪 floor 偏移，`/255` 与固定 CLIP 均值/标准差归一化。不得替换为原 CXR-LT 灰度 512 预处理。
- 固定提取 batch=4、预处理线程=2、CUDA float16 AMP，存储有限 float32 特征。正式数据前的合成 RGB/灰度 GPU 检查通过：4×512 输出、重放最大误差 0、峰值 allocated 570,921,472 字节 / reserved 717,225,984 字节。证据 `outputs/chexficient-preparation-20261108/gpu_engineering_check.json`，适配器 SHA-256 `f45aca9b42d21953aa459e43e5b665df1f22f0228e6457f5583bfd1d1ee3e6c3`。CPU 严格加载与固定原始预处理对照已另行通过。
- 该工程证据不表示真实 AUC 收益。公开预训练与匿名本地数据的重叠未知，医学预训练研究的规则边界见 [官方评分口径](OFFICIAL_SCORING_20261108.md)。完整保留来源及许可。

## 输入注册、患者隔离和恢复

启动前要求 EVA-X v2 和患者聚合 v2 均正常结束、未晋级、未导出新测试 CSV，对应独立 OOF/数值核验通过。核验当前保留模型已完成、已晋级、分类头与配方哈希一致。不得同时运行其它模型训练/特征提取。

使用用户原始 train：30,344 张图、18,513 次检查、11,055 位患者及同一四组患者划分 `(0,1,2,"audit")`。读取 data_root、旧融合 OOF 与 valid 路径等信息时，以当前模型注册计划为准。audit 组曾被历史选优复用，现为训练交叉验证组，不称为新独立保留集。

正式运行前在全新目录保存计划、SHA-256、协议与源码闭包快照、全部必要输入快照、环境版本、24 项模型与来源文件锁、GPU/CPU 工程检查锁。锁定 train/valid/test manifests、患者分组、当前 OOF IDs/targets/probabilities、配方/头/状态、当前与旧融合 valid、完整 train CXR-LT 特征缓存及有序图像身份（DICOM ID、解析后路径、大小、mtime）。旧融合 22 组 IDs/targets 证据按其原始哈希复制，核验与本地 train 标签严格相等及完整覆盖。拒绝数值/重复/缺失/冲突身份、标签不一致或跨 split 患者及检查重叠。

prepare 阶段不读取、复制或哈希任何 test 预测 CSV/特征矩阵；只锁定 test manifest 和当前模型已有的 test 核验记录，登记后续需要的 CXR-LT test cache 原注册文件路径及既有 SHA-256。valid CXR-LT cache 同样可以登记为延迟输入，只有 OOF 通过后才复制、比哈希和读取。新 CheXficient train/valid/test 缓存按 split 分开，额外绑定模型版本/权重/适配器/plan/图像身份，不能仅依靠 generic cache 的图像行序区分编码器。

正式计算从冻结源码快照执行。恢复只允许同计划、同源码、同输入与同缓存身份；从已提交的 feature cursor 继续，完成缓存核验完整矩阵哈希。保存四组各自分类头、训练患者集、特征身份、头哈希和 held 预测哈希；重载重放须精确相等。不得覆盖旧实验、权重、原始数据或旧 CSV。

## 完整患者 OOF 晋级

每个候选最多四个头，共八个 CV 头。每次用另外三组拟合、仅预测 held 组，显式核验患者交集为零；完整 18,513 次检查恰有一次 OOF 预测。当前、旧融合与候选严格按 `(subject_id,study_id)` 一对一对齐，标签逐项核验，不能以折平均代替完整拼接 AUC。

只有总体及四组十类 AUC 全部可定义、相对当前和旧融合的完整 OOF Macro-AUC 各增加至少 0.001、至少三组相对当前提高的候选才有资格晋级。选择符合条件者中完整 OOF AUC 最高者，精确并列优先较小的 `chexficient` 512 维。无合格候选则正常结束，不拟合全量头、不评估 official valid/test、不生成新 test CSV。结果出现后不改变本轮候选、门槛或拟合参数。

## 一次 official valid 与测试导出

只对选中的一个候选拟合一次全量 train 头，保存、重载并登记不可变配方；随后才提取 valid 特征。与当前和旧融合分别在相同的 859 次检查上比较：两次都要求十类 AUC 全部定义、点增益至少 0.001、seed=20261108 的 1,000 次患者配对 bootstrap 描述性 95% 差值区间下界严格大于零、至少 950 次完整十类 replicate。Gate 固定 0.5。任何一次失败都保留当前模型，不回看另一候选或按 valid 追加搜索。历史 valid 复用未被区间校正，不宣称无偏显著性或线上保证。

两次比较都通过后，才解除 test cache 的延迟锁、核验原登记哈希、提取所需 test 特征并导出独立 CSV。完整检查 1,414 次检查 / 14,140 行、每检查十类、精确 ID 与顺序、有限 [0,1] 概率、Gate=0.5；保存头到 CSV 重放精确相等。无 test 标签，不计算或推测 test AUC，不拟合 test 统计量，不自动提交平台。

实验状态、输入/源码/缓存/头/预测哈希、OOF 与 valid 报告和晋级结论分别保存。新候选 online_score 在实际反馈前为 null；当前线上 78.98 文件保持原始字节。该次实验是持续提升目标的一步，不能以工程通过或实验结束代替性能提升。
