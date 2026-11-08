# CheXpert DenseNet121 三轮迁移实验

日期：2026-11-04。按用户最新要求，以尽快提高验证集 Macro-AUC 为目标，单独执行医学预训练迁移与固定融合实验。

## 范围与比较口径

- 本轮使用外部医学预训练，属于独立研究路线，不纳入此前仅允许自然图像预训练的比赛方案。
- 官方 train 训练，valid 选模；不读取 test 图像，不改原始数据，不覆盖原 ResNet18 或正在训练的 ConvNeXt 实验。
- 本地标识已经匿名化，没有来源声明，无法排除与 CheXpert 的数据重叠。该限制随所有结果保留；验证分数不能证明独立泛化。
- 固定 ResNet18 比较对象：`weights/resnet18-finetune-v1/resnet18-finetune-v1.pt`；全量 valid CSV：`outputs/dataset-verification-20261104-180020/valid.csv`。十类 Macro-AUC=0.7806561629989964，F1@0.5=0.4184504656096208。

## 权重与输入

- TorchXRayVision 固定 1.5.7，模型 `densenet121-res224-chex`，只选 CheXpert 专属权重，不使用混合 `all` 权重。
- 官方模型共有18个分类槽，11个具名病种、7个空槽；本任务前9类按名称映射到原索引 `(17,3,1,8,4,10,0,16,7)`，不压缩源索引。`Pleural Effusion` 映射 `Effusion`。
- 第10类 `No Finding` 独立随机初始化，随后用 train 标签学习；不从其余类别推导。
- 读取原图并按 TorchXRayVision 示例归一化到 `[-1024,1024]`、RGB通道算术均值转单通道、中心正方形裁剪、skimage 缩放224。训练与验证都不使用 ImageNet RGB 归一化。
- 自定义前向仅输出原始分类器 logits，跳过官方 sigmoid/op_threshs/op_norm，供 BCEWithLogitsLoss 使用；验证时一次 sigmoid，再做检查内图像概率均值。
- 保存官方 release 元信息、下载字节数、本地 SHA-256 和软件许可证。上游旧资产没有公开 SHA-256 时如实记录；转换为 tensor-only state_dict 后，训练和重载均使用 `weights_only=True`。

## 固定实验预算

1. 不训练，先对映射9类计算完整 valid 的逐类 AUC，并与 ResNet18 相同9类比较；此阶段不报告新模型的十类 Macro-AUC。
2. 冻结主干一轮，仅训练十维分类头，lr=1e-3；主干 BatchNorm 保持 eval，避免冻结阶段修改统计量。
3. 解冻全模型两轮，主干 lr=1e-5、分类头 lr=1e-4，不追加训练、不搜索学习率。
4. AMP、有效 batch=16（优先物理2、累积8，兼顾当前并行任务显存），seed=2026，workers=0。沿用现有检查内图像权重、正类权重与梯度裁剪。
5. 按每轮完整 valid 的十类 Macro-AUC 选最佳单模型，独立重载后重新预测；新模型与固定 ResNet18 的检查级概率按0.5/0.5平均，不搜索融合权重或阈值。
6. 输出单模型和融合的完整验证报告、CSV及500次患者配对bootstrap描述区间；它们使用了选模验证集，不作独立测试性能声明。

## 工程验收

- 合成测试覆盖9头映射、No Finding 独立初始化、灰度预处理、raw logits、冻结BN、分组学习率和按检查标识对齐的固定融合。
- 真实 train/valid 小子集完成训练、保存与独立重载；完整验证前检查 GPU 资源。
- 确定性医学预处理缓存与RGB补边缓存分开，缓存后再随机增强；原图和缓存的张量逐位核验。
- 正式执行固定源码快照、环境版本、配置和权重SHA；日志与状态保存在独立输出目录。启动不等同于训练完成，结果以实际报告为准。
