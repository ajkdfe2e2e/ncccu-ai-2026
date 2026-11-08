# BioViL-T 后续方向的来源核验（2026-11-08）

本轮只准备公开资产与 CPU 工程接入，没有注册 BioViL-T 真实数据实验，没有读取本地真实图像、拟合分类头、计算候选 AUC 或生成预测 CSV。正在运行的 CheXficient 实验继续使用其原固定方案，不增加候选或改变门槛。当前保留的线上反馈最好仍为 CXR-LT 的 **78.98**。

公开图像权重固定为 `microsoft/BiomedVLP-BioViL-T@301f526e823b805d3fe712d0cf06a66f042789c5` 的 `biovil_t_image_model_proj_size_128.pt`，**109,745,561 字节**，SHA-256 `b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77`。已独立对照该 revision 的 Hugging Face API/LFS 元数据并核验实际下载字节；仓库无需 gated 访问。图像权重、模型卡与官方代码共 22 项来源和哈希见 [公开资产清单](biovil_t_source_manifest_20261108.json)。未下载文本模型。

作者实现来自 `microsoft/hi-ml@b67c1d27c6b17d8e8ff01f8c507f3cabdb307388`，保留 Microsoft 版权声明及 MIT 许可证。模型通过胸片图文及时间信息预训练；这提供了另一个表征假设，但不能从公开基准或模型名称推断本地增益。其医学预训练与比赛来源要求的边界见 [官方规则核对](OFFICIAL_SCORING_20261108.md)，本路线属于研究准备，不声称符合比赛模型来源要求。

核对作者 `image/utils.py` 后，BioViL-T 默认推理是**短边缩放 512、中心裁剪 448**。`create_chest_xray_transform_for_inference` 的泛用 512 裁剪示例不是 BioViL-T 默认设置。作者 `image/data/io.py` 还会先按单张图像的最小值、最大值将像素重映射为 uint8，再转灰度；之后使用 torchvision 默认双线性 Resize、CenterCrop、ToTensor、灰度复制三通道，没有 ImageNet 或 CLIP 均值/标准差归一化。常量图像明确返回全零 uint8，避免原公式除零与无效转换的警告；非恒定合成 RGB/灰度图与官方函数和变换逐元素一致。

图像模型由 ResNet-50、空间投影、时间 Transformer 与 128 维投影器组成。单图 `previous_image=None` 路径在作者 `MultiImageEncoder.forward` 中**不执行时间 Transformer**，而是把当前图像的 256 维空间特征与学习到的 256 维 `missing_previous_emb` 拼接；后半部对所有单图为常量。作者 `ImageModelOutput.img_embedding` 为 512 维，`projected_global_embedding` 为 128 维。适配器保留两者以供工程核验，最终特征输出采用**前 256 维空间特征与 L2 归一化的 128 维投影拼接，共 384 维**；在真实实验前固定移除已知常量半部，避免冗余列进入标准化。该向量不是十类疾病概率，单图路径没有时序融合。

适配器 `cxr/biovil_t_transfer.py` 已完成本地源码构造与严格加载。导入源码前，核对本地 provenance 与仓库来源清单一致，校验全部 22 项资产的字节数与 SHA-256，以及固定 checkpoint 哈希；使用 `torch.load(weights_only=True, map_location="cpu")` 和 `strict=True`，不执行自动下载入口，不加载文本分支。无需为现有 JPG/PNG 流程安装医学影像 IO 依赖。

4 项独立于外部资产的合成回归测试与 Python 编译通过。实际权重的 4 线程 CPU 工程检查使用两张合成 RGB/灰度图：输入形状 `2×3×448×448`，输出 `2×384`，官方预处理对照最大误差 **0**，重复输出最大误差 **0**，参数全部冻结、模型保持 eval、投影范数为 1；被移除的 256 维与池化后的 `missing_previous_emb` 最大差为 `1.1176e-8`。本地证据在 `outputs/biovil-t-preparation-20261108/cpu_engineering_check.json`，未使用 GPU、真实图像或标签，未拟合分类头或计算 AUC。

正式患者 OOF 方案、GPU 设置和真实数据提取要在当前 CheXficient 实验结束后另行冻结；当前尚无 BioViL-T 性能证据，也没有新预测 CSV。
