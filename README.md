# Chest X-ray：检查级多标签分类基线

面向 NCCCU 2026 人工智能挑战赛的研究代码，**不用于临床诊断**。原始比赛数据、模型权重、预测结果均不随 Git 发布。

## 当前研究进度（2026-11-08）

当前保留采用的 CXR-LT 模型，本地官方 valid 十类 Macro-AUC 为 **0.8335505264**，来自冻结的公开 CXR-LT ConvNeXtV2 特征与本地训练集学习的检查级十类线性头；训练集患者分组完整 OOF Macro-AUC 为 **0.8270384304**。两个分数对应不同评估人群，不直接互比。公开模型固定为 `hieuphamha/cxrlt2026-task1-convnextv2@24bf8bdfd8d0262899167b030be16ab89f78ae20`。

该保留模型的本地测试预测为 `outputs/submission_test_1414_best_auc_083355_20261108.csv`，覆盖 1,414 次检查、14,140 行，分类头重载重放最大概率误差为 0。2026-11-08 用户反馈该文件线上成绩 **78.98**，较旧融合 **78.53** 提高 **0.45 分**，成为当前已反馈的线上最高；Rad-DINO A+EMA 为 **77.34**。附件与交付文件的 SHA-256 完全一致。官方主指标已核实为 Macro-AUC；若该反馈来自主分栏，其展示精度对应 **0.7898**，尚未独立读取用户评分页面。详见 [线上结果记录](docs/ONLINE_RESULTS_20261108.md) 与 [官方评分口径](docs/OFFICIAL_SCORING_20261108.md)。预测、权重和逐例核验文件仅保存在本地。

后续固定患者聚合候选的完整患者 OOF 为 **0.8282252394**，四组均提高；official valid 点估计为 **0.8360009305**，高于当前保留模型。但相对当前的 valid AUC 增益描述性 95% 区间 **[-0.0018000031, 0.0068323706]** 跨零，未满足事先固定的晋级规则，因此未读取本轮 test 概率、未导出新 CSV，保留线上 78.98 文件。更高的本地点估计不等于已晋级或更高线上成绩，详见 [患者聚合结果](docs/PATIENT_SMOOTHING_RESULTS_20261108.md)。

后续嵌套分类头、语义瓶颈及末阶段微调均未替换当前模型。末阶段微调完整 OOF 为 **0.8169781872**。EVA-X 于北京时间 13:10:03 完成：独立 EVA 特征 OOF **0.8153107264**，与 CXR-LT 联合特征 OOF **0.8274166581**，后者相对当前仅增 **0.0003782277**，未达到固定的 0.001 晋级门槛。独立 NumPy 头重放与完整 OOF 重算通过，本轮未评估 valid/test 或导出 CSV，详见 [EVA-X 结果](docs/EVA_STUDY_PROBE_RESULTS_20261108.md)。持续优化目标为 active；早期上传快照保留于 [进度记录](docs/PROGRESS_SNAPSHOT_20261108.md)。

CheXficient 固定实验于北京时间 **15:09:06** 正常结束。独立特征完整患者 OOF 为 **0.8286274797**，联合 CXR-LT 特征为 **0.8315443007**，后者相对当前提高 **0.0045058703**，四组均提高；按预设 OOF 规则选中联合特征。其唯一 official valid AUC 为 **0.8394749558**，相对当前提高 **0.0059244293**，但患者 bootstrap 增益描述性 95% 区间 **[-0.0011930675, 0.0133285088]** 跨零，未满足固定的双基线 valid 替换条件。当前采用及用户反馈线上最好仍为 **78.98**；本轮没有提取 test 或导出新 CSV。独立 OOF、终态 valid 重放及两组各 1,000 次患者 bootstrap 复核均通过，预测最大误差 0；详见 [实验结果](docs/CHEXFICIENT_STUDY_PROBE_RESULTS_20261108.md)和 [固定方案](docs/CHEXFICIENT_STUDY_PROBE_PLAN_20261108.md)。

BioViL-T 固定实验已于北京时间 **15:56:47** 从冻结源码启动，worker PID **14512**；**15:58:52** 为 `running / extract_biovil_train`，已保存 **102/30,344** 张训练图像特征、`error=null`，尚无本轮 AUC 或新 CSV。只比较 CXR-LT+BioViL-T **1,438D** 与 CXR-LT+CheXficient+BioViL-T **1,950D** 两个固定候选，复用已登记的训练特征，并以患者四组完整 OOF 选一个 official valid 候选。计划 SHA-256 `ee6efa98450f5e26e47672d2c5e59d457903faa474da478c14033f46ac3cd981`。完整 CPU 回归 396 项及后补的注册成功路径测试通过；冻结快照加载和独立注册 v2 核验通过。单图预处理为短边512、中心裁剪448，384D 特征不含时间 Transformer。首次独立核验曾因路径匹配漏项意外流式哈希验证矩阵，未做矩阵预测或评分；原记录保留，v2 明确跳过该延迟输入。见 [固定方案](docs/BIOVIL_T_STUDY_PROBE_PLAN_20261108.md) 与 [接入及启动记录](docs/BIOVIL_T_PREPARATION_20261108.md)。

可复核的聚合结果、公开来源和模型/预测 SHA-256 见 [当前配方](docs/best_study_probe_recipe_20261108.json)、[监督特征方案](docs/SUPERVISED_STUDY_PROBE_PLAN_20261108.md)及 [末阶段微调结果](docs/CXRLT_STAGE_FINETUNE_RESULTS_20261108.md)。本地验证集参与过多轮历史选优，公开预训练数据与匿名比赛数据的重叠情况未知；本地验证分数不代表线上分数或无偏泛化估计。

## 历史研究结果（2026-11-05）

在原最高分方案上继续增加公开权重预测与 Rad-DINO 翻转推理，完整验证集 **859 次检查、十类 Macro-AUC 从 0.819362 提高到 0.820325**。单个最佳 Rad-DINO 检查点使用原图/翻转图概率均值后，十类 AUC 从 0.803207 提高到 0.807447。所有分数来自本地完整 valid；五类公开模型的分数单独列示。

本次是用户授权的外部医学预训练研究路线。原自然图像预训练基线仍独立保留，研究结果不代表满足原比赛材料/大小/耗时要求。详细结果、固定配方、重现命令与局限见 [医学模型验证记录](docs/MEDICAL_RESULTS_20261105.md)，公开来源、固定版本和 SHA-256 见 [模型清单](docs/public_model_manifest.json)。

融合系数使用了 valid 标签；增量的患者 bootstrap 95% 区间跨零，患者分组诊断没有确认稳定收益。0.820325 表示本轮已完成实验中的最高验证分数，不是独立泛化保证。原 0.819362 方案及其文件保持不变。

## 当前实现

- 严格解析官方 CSV：`Predict_class` 是分号分隔的 **1–10** 类 ID。
- 官方 train/valid 分离；不合并验证集，不用测试集做调参、训练或归一化统计估计。
- 原基线：ResNet18 或 ConvNeXt-Tiny + 10 个 sigmoid 输出，仅使用 torchvision ImageNet-1K 权重；医学预训练研究使用独立脚本与输出。
- 保持长宽比缩放 + 黑边补齐；训练随机仿射，推理固定变换。
- BCEWithLogitsLoss，按训练集检查级频次计算正类权重，上限 10；按每次检查图数的倒数对图像损失加权。
- 同次检查多图概率算术平均；不跨患者或跨检查自适应。
- 检查级 Macro-AUC 选模，Macro-F1 辅助报告。AUC 不可定义的类明确列为 null，并报告有效类数。
- CSV 列为 `Subject_id,Study_id,Predict_class,Probability,Gate`；每次检查必须 10 行。
- **Gate 是阈值，不是 0/1 预测标签。**当前固定为 0.5。
- `predict.py` 和 `cost.py` 共用 batch=1 Predictor；推理 eval、固定参数，无隐式权重下载。

## 环境

推荐 Python 3.12；本机使用 `.venv`，不修改系统 Python。CUDA 依赖下载较大。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

默认 ResNet18 使用 batch 8、224 像素、训练混合精度。2026-11-04 的新实验机器为 RTX 4060 Laptop 8GB；实际可用显存取决于其他程序占用。本地耗时不能代替官方 T4 测试。ConvNeXt 支持物理 batch=1，并可用梯度累积保持有效 batch；ResNet 的训练 batch 至少为 2。

## 数据布局

```text
cxr_first/
  train.csv
  valid.csv
  test.csv
  train/subject_x/study_x/pic_x.jpg
  valid/subject_x/study_x/pic_x.jpg
  test/subject_x/study_x/pic_x.jpg
```

数据检查只读取 CSV 和文件路径；测试集不做内容分析或统计拟合。

```powershell
.\.venv\Scripts\python.exe scripts/audit_data.py --data-root cxr_first --output outputs/audit.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 权重来源及训练

```powershell
.\.venv\Scripts\python.exe scripts/fetch_pretrained.py
.\.venv\Scripts\python.exe train.py --pretrained weights/resnet18-f37072fd.pth --output outputs/resnet18-run1
```

权重来自 PyTorch 官方公开 URL；脚本验证官方文件名 hash 前缀，保存完整 SHA-256、许可证和来源记录到 `weights/`。软件许可证不自动代表所有权重/数据授权，提交前仍须核对赛事要求与材料。

每次训练必须用新的或空的输出目录，避免混淆日志和模型。`best.pt` 根据验证集检查级 AUC 选择，`run.json` 记录配置，`history.jsonl` 记录每轮指标、学习率和速度。默认 batch=1 验证，与共享推理器保持一致。不合并验证集。

### ConvNeXt 与显存受限训练

第一阶段的固定预算、对照指标和后续路线见 [AGGRESSIVE_OPTIMIZATION_PLAN.md](AGGRESSIVE_OPTIMIZATION_PLAN.md)。下面使用物理 batch=4、累积4次、有效 batch=16；启动前先完成 CUDA 和小子集检查。

```powershell
.\.venv\Scripts\python.exe scripts/check_cuda.py
.\.venv\Scripts\python.exe scripts/fetch_pretrained.py --architecture convnext_tiny
.\.venv\Scripts\python.exe train.py --architecture convnext_tiny --pretrained weights/convnext_tiny-983f1562.pth --data-root 'G:\cxr_first(1)\cxr_first' --output outputs/convnext384-new --image-size 384 --batch-size 4 --accumulation-steps 4 --epochs 20 --lr 0.0001 --weight-decay 0.01 --scheduler cosine --warmup-epochs 2 --min-lr 0.000001 --ema-decay 0.999 --image-cache .cache/letterbox384 --workers 0 --device cuda
```

`--gradient-checkpointing` 可重计算 ConvNeXt 特征阶段以减少显存；`--ema-device cpu` 将 EMA 权重存于 CPU，验证时移回模型设备。二者可能增加耗时。EMA 只在成功的优化器更新后更新，验证和 `best.pt` 使用 EMA 权重；`last-training.pt` 同时保存原模型和 EMA。梯度累积按每个窗口实际图像数归一化，最后不足一个窗口也会更新。改变优化过程的新增参数参与恢复配置检查；旧 ResNet 状态缺失这些字段时按原默认值解释。

`--image-cache .cache/letterbox384` 可把 RGB 转换和确定性缩放结果以无损 PNG 缓存在原始数据目录之外。缓存键含源文件路径、大小、修改时间、Pillow 版本和输入尺寸；随机仿射仍在每次训练读取时执行。缓存只影响读取成本，不是模型输入参数，因此恢复时可更换缓存目录。`scripts/profile_training.py` 会分别报告原图读取、缓存命中与 GPU 纯计算速度，并检查采样真实图像的张量逐位相同。

若要在训练结束后自动完成验证预测、患者 bootstrap 对比、推理文件清单和本地计时，可使用 `scripts/run_aggressive_experiment.py --data-root <原始数据目录> --output <全新实验目录> --baseline <已验证的基线valid.csv> --image-cache .cache/letterbox384`。执行器先保存代码快照，再从快照运行固定的20轮方案；状态和独立日志在实验父目录，模型及验证报告在其 `training/` 子目录。后台启动后查看 `experiment.json` 的实际阶段与状态，不能把进程已创建视为实验已完成。

### 继续微调与 epoch 边界恢复

- `--init-checkpoint`：从现有 `best.pt` 初始化，**重新创建优化器**；不是恢复历史优化器状态。严格检查架构、类别和输入尺寸。
- `--resume`：使用新格式 `last-training.pt` 恢复模型、优化器、AMP scaler、学习率调度及随机状态；要求 `workers=0`，相同训练配置、总轮数和 CSV 哈希，在新的输出目录继续。只保证同一软硬件环境的 epoch 边界恢复，不支持任意 batch 恢复。
- `--scheduler cosine --min-lr 0.000001`：可选余弦衰减。
- `--stop-after-epochs 1`：本次调用完成1轮后正常停止，保留原总轮数/调度范围，方便验证恢复。
- `last-training.pt` 是训练状态，不是部署权重；推理仍仅用 `best.pt`。

```powershell
.\.venv\Scripts\python.exe train.py --init-checkpoint outputs/resnet18-baseline-20261104-022511/best.pt --epochs 4 --lr 0.0001 --scheduler cosine --min-lr 0.000001 --image-size 224 --batch-size 8 --workers 0 --output outputs/finetune-new
# 恢复时填写同样的总轮数和超参数；不要用 best.pt 作为 resume 输入。
.\.venv\Scripts\python.exe train.py --resume outputs/interrupted/last-training.pt --epochs 4 --lr 0.0001 --scheduler cosine --min-lr 0.000001 --image-size 224 --batch-size 8 --workers 0 --output outputs/resumed-new
.\.venv\Scripts\python.exe scripts/verify_resume.py --device cuda
```

候选与基准使用 `scripts/compare_runs.py` 严格对齐验证 CSV，按患者配对 bootstrap 评估指标差异。置信区间只是验证集上的描述性结果，模型已经使用该集选优，不代表独立检验或泛化保证。

快速闭环测试（只用于工程检查，不代表模型成绩）：

```powershell
.\.venv\Scripts\python.exe train.py --from-scratch --epochs 1 --smoke-studies 8 --image-size 64 --batch-size 2 --output outputs/smoke-01
.\.venv\Scripts\python.exe predict.py --split valid --limit-studies 8 --checkpoint outputs/smoke-01/best.pt --output outputs/smoke-01/valid.csv
```

也可一键运行同样的闭环（使用验证子集，并加入局部耗时/一致性检查）：

```powershell
.\.venv\Scripts\python.exe scripts/smoke_pipeline.py --device cuda --pretrained weights/resnet18-f37072fd.pth
```

安装训练依赖前可先用系统 Python 执行 `python scripts/preflight.py`（需要 numpy/Pillow），验证权重完整性及训练图像解码。`python scripts/prior_baseline.py --output outputs/prior-new` 可运行常数先验的验证流程检查，预期 AUC=0.5，不是训练模型性能。

## 验证与最终预测

```powershell
.\.venv\Scripts\python.exe predict.py --split valid --checkpoint outputs/resnet18-run1/best.pt --output outputs/resnet18-run1/valid.csv
# 方案锁定后才运行测试集；不得用结果修改模型或参数。
.\.venv\Scripts\python.exe predict.py --split test --checkpoint outputs/resnet18-run1/best.pt --output outputs/submission.csv
```

完整测试集应输出 1414×10=14140 行数据，另有一行表头，共14141行。程序检查精确 ID 覆盖、无重复、类别 1–10、概率/阈值范围及有限值，并保存报告。原始数据目录拒绝作为输出路径。`valid.csv` 的859次检查预测仅供本地验证，不能作为平台的测试集提交文件。

2026-11-05 旧医学融合方案可按保存的固定配方导出完整测试集。下面命令对应旧融合；2026-11-08 当前保留采用的 CXR-LT 配方见本文开头：

```powershell
.\.venv\Scripts\python.exe scripts/predict_best_ensemble.py --data-root 'G:\cxr_first(1)\cxr_first' --output outputs/best-ensemble-test-20261105 --split test --device cuda
.\.venv\Scripts\python.exe scripts/validate_submission.py outputs/best-ensemble-test-20261105/submission.csv --metadata 'G:\cxr_first(1)\cxr_first\test.csv'
```

导出器核验公开权重和本地检查点的SHA-256，分组件运行完整推理，并用测试清单检查输出ID、十类覆盖与CSV概率往返一致性。中断后可复用指纹一致且完整的组件。报告明确记录`split=test`；没有测试标签，不报告测试集AUC。

## 耗时测试：当前是本地辅助实现

```powershell
.\.venv\Scripts\python.exe cost.py outputs/submission.csv pictures_100 pictures_5 --checkpoint outputs/resnet18-run1/best.pt --metadata cxr_first/test.csv --image-root cxr_first/test
```

- 5 张不重复预热图；100 张样本，batch=1 串行，连续 3 轮；包含读图、预处理、GPU 同步及单图输出，不包含加载模型。
- 样本和预热 ID 必须互斥，且图像字节与源文件一致。
- 模型输出按检查聚合，而官方计时单位为图片：**随机抽到一次检查的部分视图时，不能直接把单图概率当作该检查最终概率来对比。**本地工具在计时外补读同检查其余视图核对 CSV。
- 因此本地报告不冒充官方完全合规验证；正式提交前必须与组委会确认多视图聚合的计时和一致性核验口径，并按其接口调整。
- `--local-smoke` 仅放宽 100/5 数量供本地测试，输出明确标记非正式结果。
- 概率误差 >1e-3、二值决策或类内排序改变会失败；本地比较以检查为单位，官方核验细则须再次对齐。

## 合规与数据安全

- `.gitignore` 排除 `cxr_first/`、原始压缩包、CSV、权重、虚拟环境、输出等，禁止 `git add -f` 绕过。
- 不上传患者影像、标注、预测或原始路径报告。只公开代码不代表已获得比赛数据传播许可。
- 模型及必要推理配套文件解压后总计 ≤500MB；耗时须由官方指定硬件复测。
- 训练输出仅用于比赛研究；不提供临床诊断建议。
- 规则主文档附录 A 要求区域赛/省赛截止前提交材料包；**不能仅依据网页“初赛只交 CSV”而省略材料**。请按 PDF 及官方平台落实提交。

官方来源：
- https://www.ncccu.org.cn/index/Paper/case1.html
- https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf
- https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261027/b23d6a1f19602da2a9bd4048108175d0.pdf

原上游：https://github.com/hu-ihu/Chest-X-ray 。2026-11-08 上传时该地址返回 404；进度保存至当前账号的私有赛事仓库 https://github.com/ajkdfe2e2e/ncccu-ai-2026 ，使用独立进度分支，原上游配置保留。
