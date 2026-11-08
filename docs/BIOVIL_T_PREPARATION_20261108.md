# BioViL-T 后续方向的来源核验（2026-11-08）

准备阶段只使用公开资产与合成输入；其后正式实验于北京时间 **15:56:47** 从冻结源码启动。启动时尚无 BioViL-T 候选 AUC 或新预测 CSV，当前保留的线上反馈最好仍为 CXR-LT 的 **78.98**。CPU 合成检查在 CheXficient 提取训练特征时完成；GPU 合成检查在该实验正常结束、进程退出后才执行。CheXficient 的候选和门槛保持原固定方案。

公开图像权重固定为 `microsoft/BiomedVLP-BioViL-T@301f526e823b805d3fe712d0cf06a66f042789c5` 的 `biovil_t_image_model_proj_size_128.pt`，**109,745,561 字节**，SHA-256 `b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77`。已独立对照该 revision 的 Hugging Face API/LFS 元数据并核验实际下载字节；仓库无需 gated 访问。图像权重、模型卡与官方代码共 22 项来源和哈希见 [公开资产清单](biovil_t_source_manifest_20261108.json)。未下载文本模型。

作者实现来自 `microsoft/hi-ml@b67c1d27c6b17d8e8ff01f8c507f3cabdb307388`，保留 Microsoft 版权声明及 MIT 许可证。模型通过胸片图文及时间信息预训练；这提供了另一个表征假设，但不能从公开基准或模型名称推断本地增益。其医学预训练与比赛来源要求的边界见 [官方规则核对](OFFICIAL_SCORING_20261108.md)，本路线属于研究准备，不声称符合比赛模型来源要求。

核对作者 `image/utils.py` 后，BioViL-T 默认推理是**短边缩放 512、中心裁剪 448**。`create_chest_xray_transform_for_inference` 的泛用 512 裁剪示例不是 BioViL-T 默认设置。作者 `image/data/io.py` 还会先按单张图像的最小值、最大值将像素重映射为 uint8，再转灰度；之后使用 torchvision 默认双线性 Resize、CenterCrop、ToTensor、灰度复制三通道，没有 ImageNet 或 CLIP 均值/标准差归一化。常量图像明确返回全零 uint8，避免原公式除零与无效转换的警告；非恒定合成 RGB/灰度图与官方函数和变换逐元素一致。

图像模型由 ResNet-50、空间投影、时间 Transformer 与 128 维投影器组成。单图 `previous_image=None` 路径在作者 `MultiImageEncoder.forward` 中**不执行时间 Transformer**，而是把当前图像的 256 维空间特征与学习到的 256 维 `missing_previous_emb` 拼接；后半部对所有单图为常量。作者 `ImageModelOutput.img_embedding` 为 512 维，`projected_global_embedding` 为 128 维。适配器保留两者以供工程核验，最终特征输出采用**前 256 维空间特征与 L2 归一化的 128 维投影拼接，共 384 维**；在真实实验前固定移除已知常量半部，避免冗余列进入标准化。该向量不是十类疾病概率，单图路径没有时序融合。

适配器 `cxr/biovil_t_transfer.py` 已完成本地源码构造与严格加载。导入源码前，核对本地 provenance 与仓库来源清单一致，校验全部 22 项资产的字节数与 SHA-256，以及固定 checkpoint 哈希；使用 `torch.load(weights_only=True, map_location="cpu")` 和 `strict=True`，不执行自动下载入口，不加载文本分支。无需为现有 JPG/PNG 流程安装医学影像 IO 依赖。

4 项独立于外部资产的合成回归测试与 Python 编译通过。实际权重的 4 线程 CPU 工程检查使用两张合成 RGB/灰度图：输入形状 `2×3×448×448`，输出 `2×384`，官方预处理对照最大误差 **0**，重复输出最大误差 **0**，参数全部冻结、模型保持 eval、投影范数为 1；被移除的 256 维与池化后的 `missing_previous_emb` 最大差为 `1.1176e-8`。本地证据在 `outputs/biovil-t-preparation-20261108/cpu_engineering_check.json`，未使用 GPU、真实图像或标签，未拟合分类头或计算 AUC。

北京时间 **15:14:33** 的合成 GPU 检查随后通过：RTX 4060 Laptop、batch2、CUDA float16 AMP、4 CPU 线程，输入为不同的合成 RGB/灰度图，输出有限且冻结的 384D 特征；投影范数约为 1，两次输出最大误差 **0**。峰值 allocated **164,187,136 字节**、reserved **182,452,224 字节**，单次工程过程约 0.77 秒。证据 `outputs/biovil-t-preparation-20261108/gpu_engineering_check.json` 绑定适配器 SHA-256 `e6ad1ce0325d507c363db9de60f0bc6f42d12dd9fd1d0cbe5e42f59c727befb8` 与固定权重。检查前后均没有其它项目 CUDA worker；桌面应用的显存未被关闭或清理。该检查未读取真实数据、未训练或计算 AUC，合成 batch 的内存和速度不作为全量真实提取的计时保证。

随后使用实际通用特征提取器，对四张合成 PNG 运行 batch2、预处理线程2、CUDA float16 AMP 集成检查。主动在已提交2/4行后中断，再续跑；续跑与从头提取、完成缓存重载的最大误差均为 **0**，矩阵有限且为4×384。报告 `outputs/biovil-t-preparation-20261108/extraction_integration_check.json` 绑定相同适配器哈希；检查没有读取真实图像或标签、拟合分类头或计算AUC。

正式患者 OOF 的两候选方案见 [固定方案](BIOVIL_T_STUDY_PROBE_PLAN_20261108.md)：CXR-LT+BioViL-T 1,438D，以及 CXR-LT+CheXficient+BioViL-T 1,950D，后者严格按CXR-LT1054、CheXficient512、BioViL-T384的顺序拼接。只用train学习检查级十类线性头，四患者组完整OOF后最多选择一个official valid候选，通过双基线固定门槛才提取test。

北京时间 **15:53:22** 完成全新 `outputs/biovil-t-study-probe-20261108` 注册，计划 SHA-256 `ee6efa98450f5e26e47672d2c5e59d457903faa474da478c14033f46ac3cd981`；冻结源码可独立加载，201项输入、20项源码、186项快照锁核验通过。完整CPU回归396项通过，另补的成功prepare→冻结plan加载测试通过；九项执行测试含两条合成1,414次检查CSV导出、失败门槛禁止test及缓存重放检查。独立注册v2核验198项非test输入、20项源码、183项非test快照及36项CheXficient产物，排除6项test元数据和CheXficient valid特征矩阵，并核实30,344张训练图像的身份/顺序与CXR-LT缓存完全一致。

首次独立注册核验器的相对路径匹配漏项，意外流式哈希了应延迟读取的CheXficient valid特征矩阵；未用NumPy加载矩阵进行拟合、预测或评分。原报告保持，修正后的 `outputs/biovil-t-study-probe-20261108-independent-registration-verification-v2.json` 明确记录该工程偏差并跳过此矩阵；正式执行器原有的延迟访问逻辑未改，原实验方案不变。

**15:56:47** 以本地venv Python、`-B -u` 从冻结源码启动，launcher PID20960、实际worker PID**14512**；**15:56:56** 状态 `running / load_and_validate_train_cache`、`error=null`，随后进入 `extract_biovil_train`。**15:58:52** 已提交 **102/30,344** 张真实训练图像特征，进程存活，stderr仅有timm弃用提示。字节码写入关闭，避免修改冻结源码缓存。尚无本轮OOF/valid AUC、新CSV或新线上分数；后续实时状态以实验目录 `experiment.json` 和特征 `progress.json` 为准。
