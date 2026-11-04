# Chest X-ray：检查级多标签分类基线

面向 NCCCU 2026 人工智能挑战赛的研究代码，**不用于临床诊断**。原始比赛数据、模型权重、预测结果均不随 Git 发布。

## 当前实现

- 严格解析官方 CSV：`Predict_class` 是分号分隔的 **1–10** 类 ID。
- 官方 train/valid 分离；不合并验证集，不用测试集做调参、训练或归一化统计估计。
- ResNet18 + 10 个 sigmoid 输出；可用 torchvision ImageNet-1K 权重，禁止医学预训练权重。
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

本机 GPU 为 RTX 3050 4GB，默认 batch 8、224 像素、训练混合精度。RTX 3050 的本地耗时不能代替官方 T4 测试。发生显存不足时先减小 batch（至少 2）。

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

完整测试集应输出 1414×10=14140 行数据，不含表头。程序检查精确 ID 覆盖、无重复、类别 1–10、概率/阈值范围及有限值，并保存报告。原始数据目录拒绝作为输出路径。

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

远程仓库：https://github.com/hu-ihu/Chest-X-ray 。初始化不代表已经推送。
