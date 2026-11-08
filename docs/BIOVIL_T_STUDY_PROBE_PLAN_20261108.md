# BioViL-T 检查级增量实验固定方案（2026-11-08）

本方案在 BioViL-T 真实图像特征提取、分类头训练和候选 AUC 计算之前写定。当前保留模型为 `outputs/cxrlt-supervised-study-probe-20261108`：完整患者 OOF Macro-AUC **0.8270384304066101**、official valid **0.8335505264258105**、用户反馈线上 **78.98**。已完成的 CXR-LT+CheXficient 候选 OOF 为 **0.8315443007087662**、valid 为 **0.8394749557581884**，但未通过其固定 valid 门槛，未晋级、未提取 test；其已完成特征可以复用，其分类头和分数不能冒充新的独立基准。

## 假设与固定候选

检验 BioViL-T 的冻结图像表征是否提供当前监督特征及已知 CheXficient 表征之外的互补信息。只拟合两个候选：

1. `joint`：CXR-LT **1,054D** 与 BioViL-T **384D** 检查均值特征依次拼接，共 **1,438D**。
2. `triple`：CXR-LT **1,054D**、CheXficient **512D**、BioViL-T **384D** 检查均值特征依次拼接，共 **1,950D**。

每个编码器先逐图提取，使用同次检查全部图像的等权均值，再按上述固定顺序拼接。BioViL-T 使用前 256 维空间特征与逐图 L2 归一化的 128 维投影；不使用常量的缺失前图半部。编码器冻结且保持 eval。没有单独 BioViL-T 候选、未投影变体、患者聚合、TTA、按类选型、额外融合比例或阈值搜索。

每次检查是一个训练样本。固定十输出普通 BCE 线性头，L2=0.01、max_iter=200、seed=20261108、class_balance=False、CPU float64 LBFGS。标准化只由每次拟合的患者计算，常数列规则沿用已验证的 `fit_linear_probe`。达到预算的头保留其原终态，不按分数追加训练轮次。

## 公开资产与提取设置

- BioViL-T 固定为 `microsoft/BiomedVLP-BioViL-T@301f526e823b805d3fe712d0cf06a66f042789c5`，图像权重 `biovil_t_image_model_proj_size_128.pt`，109,745,561 字节，SHA-256 `b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77`。作者代码固定为 `microsoft/hi-ml@b67c1d27c6b17d8e8ff01f8c507f3cabdb307388`，22 项资产及许可见 [来源清单](biovil_t_source_manifest_20261108.json)。
- 使用 `cxr/biovil_t_transfer.py`，当前适配器 SHA-256 `e6ad1ce0325d507c363db9de60f0bc6f42d12dd9fd1d0cbe5e42f59c727befb8`。加载前校验固定清单及本地 provenance，安全反序列化、strict=True，仅加载图像分支，不触发自动下载。
- BioViL-T 官方默认预处理为逐图 min/max remap 到 uint8、灰度、短边双线性 Resize512、CenterCrop448、ToTensor、复制三通道，无 ImageNet/CLIP 归一化；常量图明确映射全零。单图不执行时间 Transformer，不声称该路径含时序融合。
- BioViL-T 固定 batch=2、预处理线程=2、CUDA float16 AMP，存储有限 float32 特征。CPU/GPU 合成检查已通过，GPU batch2 输出384D、重复误差0；合成检查只支持工程可运行性，不代表真实 AUC 或全量速度。
- CheXficient 训练特征直接复用 `outputs/chexficient-study-probe-20261108/chexficient_features/train`，矩阵 SHA-256 `373aef3b1c79c8815b12ebda51e358973ebd0e8628596101ee1dd1755738ded2`。权重、源码、预处理及缓存 binding 均沿用该已完成实验，不重提训练图像。若后续唯一选中 `triple` 且 valid 晋级，需要新的 CheXficient test 特征时，使用原固定 378 输入、batch4、预处理线程2、CUDA float16 AMP；先释放 BioViL-T 模型，再加载该图像模型，不并发占用 GPU。
- 当前 CXR-LT 缓存、公开资产与各模型预处理分别绑定，不能仅凭特征行数相同或 generic cache 命中复用另一编码器的特征。

来源、许可和医学预训练的研究边界保持公开，见 [官方规则核对](OFFICIAL_SCORING_20261108.md)。公开预训练与匿名数据是否重叠仍未知，本实验不承诺符合比赛模型来源要求。

## 启动前注册与不可变输入

启动前要求 CheXficient 正式实验正常完成、`candidate_promoted=false`、没有提取 test 或导出新 test CSV，其完整 OOF 和终态 valid 独立复核均通过。核验其 plan_lock、源码/输入/产物登记以及原患者标签对齐证据。当前 CXR-LT 基准仍须 complete、promoted、配方与头哈希一致。不得同时运行其他真实模型训练或特征提取。

使用原始 train 的 **30,344 张图、18,513 次检查、11,055 位患者**，患者分组沿用 `(0,1,2,"audit")`。audit 已被历史实验复用，属于训练交叉验证组，不是新独立保留集。官方 valid 为859次检查、228位患者；test为1,414次检查，不含已知标签。

在全新 `outputs/biovil-t-study-probe-20261108` 中保存方案与 SHA-256、源码闭包快照、软件环境、全部必要输入快照、工程检查记录和模型来源锁。BioViL-T 模型快照包含22项来源资产及可核验 provenance。锁定 train/valid/test manifests、患者分组、基准 OOF 概率/标签/ID、头与配方/状态、两基线 valid、完整 train CXR-LT 缓存、CheXficient train 缓存及其 binding/registration/input/progress。各缓存按有序图像身份核验 DICOM ID、解析路径、大小、mtime，检查级 ID 与标签须逐项一致；拒绝重复/缺失/冲突 ID、跨 split 患者或检查重叠。

CheXficient 已完成的 plan、artifacts、独立证据和必要来源登记仅作为固定输入复用，不修改其原始字节。22 组旧融合训练标签证据覆盖全部训练检查，并与原 train 标签严格一致。

prepare 不读取、复制或哈希 test 预测 CSV、test 特征矩阵。可锁定原 test manifest 与基准已有的 test 核验元数据；原 CXR-LT valid/test 矩阵与 CheXficient valid 矩阵登记为延迟输入，只记录其原已注册哈希，OOF 通过并确定候选后才解除必要的 valid 锁。若选中 `joint`，不物化 CheXficient valid。旧 CheXficient 实验不存在 test 缓存，不能将 valid 或 train 缓存冒作 test。

正式计算从本轮冻结源码执行。恢复仅允许同一计划、源码、输入和特征 binding；从已提交 feature cursor 继续。缓存完成后核验完整矩阵哈希，保存每个 CV 头、拟合患者、特征身份、头与 held 预测哈希并重载精确重放。不得覆盖已有实验、数据、模型权重或提交 CSV。

## 完整患者 OOF 选择

两个候选各四个全新头，共八个 CV 头。每次使用其余三组拟合，仅预测 held 组，显式核验患者交集为零；全部18,513次检查各有且仅有一个 OOF 预测。候选、当前与旧融合按 `(subject_id,study_id)` 一对一对齐并核验标签。完整 OOF 使用拼接概率计算，不能用四组 AUC 均值代替。

候选须满足总体及四组十类 AUC 均可定义、相对当前和旧融合完整 OOF Macro-AUC 各至少提高0.001、至少三组相对当前提高。按符合条件者中完整 OOF AUC 最高选择唯一候选，精确并列优先较小的 `joint` 1,438D。另行描述与已完成 CheXficient 联合候选的 OOF 差异，不能隐藏该更强本地参考或用新的拟合内分数替代它。结果出现后不改变候选、门槛或拟合参数。

无合格候选则正常结束，不拟合全量头、不评价 valid/test、不生成新提交。选中一个候选后，只用全量 train 拟合一次头，保存、重载并登记不可变配方。

## 一次 official valid 与必要的测试导出

只有 OOF 选定候选并登记全量头后才提取 BioViL-T valid，物化所需 CXR-LT valid，以及仅在 `triple` 时物化原 CheXficient valid。按固定顺序构造唯一候选，与当前及旧融合分别比较。两次均要求十类 AUC 全部定义、点增益至少0.001、seed=20261108 的1,000次患者配对 bootstrap 描述性95%差值区间下界严格大于零、至少950个完整十类 replicate。Gate固定0.5。历史 valid 复用没有被区间校正，不将其称为无偏显著性或线上保证。

任一 valid 比较失败，保留78.98基准，不回看另一个候选、不改参数、不提取 test、不导出新 CSV。双基线都通过后才物化 CXR-LT test 并提取 BioViL-T test；若选中 `triple`，另用冻结 CheXficient 图像分支提取新 test 缓存，顺序运行且绑定本轮计划与模型身份。核验测试患者/检查隔离、1,414次检查/14,140行、精确 ID 顺序、每检查十类、有限[0,1]概率、Gate=0.5，保存头到 CSV 的重放须精确一致。

没有 test 标签，不计算或推测 test AUC，不用 test 统计量拟合标准化或模型，不自动提交平台。新方案线上分数保持 null，直到用户真实反馈；78.98文件原字节保持。记录完整执行状态、输入/源码/缓存/头/预测哈希与 OOF/valid 门槛结果，并独立复核终态。该实验是持续提升的一步，工程成功或实验结束不等于性能提升。
