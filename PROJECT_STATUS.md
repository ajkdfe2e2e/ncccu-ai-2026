# 项目状态：优化实验01完成并通过本地核验（2026-11-04）

## 最终结果与推荐

推荐候选：outputs/resnet18-finetune-20261104-103301/best.pt，原基线保留不变。

| 指标 | 原基线 | 4轮微调最佳候选 |
|---|---:|---:|
| 完整859检查 Macro-AUC | 0.7338196439 | **0.7806437017** |
| Macro-F1（固定0.5阈值） | 0.3584969252 | **0.4184504656** |
| 权重字节 | 44,795,083 | 44,801,419 |
| 本地100图×3轮平均耗时 | 55.7163ms | 57.5179ms |

- 最佳为微调第4轮；每轮AUC依次为0.76477、0.76955、0.77691、0.78064。保存模型重载结果与最后一轮验证完全一致。四轮累计训练/验证记录耗时约2小时26分（含一次较长运行停顿）。
- AUC绝对提升0.0468241；228名患者、500次配对cluster bootstrap差值95%区间[0.0205620, 0.0626096]。F1提升0.0599535，区间[0.0405706, 0.0777177]。这些区间未修正验证集选模偏差，属于描述性分析，不是独立显著性或测试泛化证明。
- 10类AUC均比原基线提高。仍较弱：纵隔心影增大0.68214、肺炎0.67244、肺部不透明影0.71182。
- 推理计时使用同一组100张验证图和5张预热图，RTX3050、batch1、3轮；对应63次检查一致性核验：不一致0、最大误差0、排序变化false。
- scripts/verify_candidate.py 本地证据核验通过：模型/CSV/metadata哈希、指标、覆盖、来源、耗时协议、大小一致。报告 outputs/resnet18-finetune-20261104-103301/acceptance.json。
- 候选哈希：8141045a6846d210c772623e2a672f1701b7bcb7b544bb6746136c7854d3626c。
- 未使用测试图像，未提交比赛或推送仓库；原基线哈希再次核对不变。
- 官方T4测试及多视图聚合计时口径仍待确认；本地小于100ms不代表官方认证。未因结果上涨追加训练。

具体证据均在候选目录：run.json、history.jsonl、valid.report.json、comparison.json、benchmark-valid/cost.json、inference_manifest.json、acceptance.json。部署使用best.pt，不包括last-training.pt训练恢复文件。

## 优化实验01过程记录（下列中间状态仅供追溯，最终以上述结果为准）

- 固定方案见 OPTIMIZATION_PLAN.md：从原基线权重微调4轮，LR 1e-4 cosine到1e-6，224/batch8/workers0；不使用测试图像。
- 原基线 SHA256 已复核未改变：1012cb6b95d61928dc1629b2a1d6ad70bf1c544f3a418678a02f1cdd7952cf31。
- 新增训练状态恢复、原子保存、患者配对 bootstrap 比较。全套测试89项：88通过，1项符号链接权限跳过。
- GPU端到端恢复核验通过：outputs/resume-verification-20261104-103153-972929/report.json。两轮连续 vs 中断恢复的模型、optimizer、scaler、scheduler及best payload精确一致。
- 正在训练 outputs/resnet18-finetune-20261104-103301，自动流程依次训练、重载预测、对比、耗时、大小核验。微调第3轮完整验证已完成：Macro-AUC=0.7769120535、Macro-F1@0.5=0.4007318703（第1轮AUC=0.7647669608，第2轮AUC=0.7695539564/F1=0.4103252827）；第4轮进行中，选模仍按事先确定的AUC，不因F1回落改选模标准；高于原基线0.7338196439，但这是中间训练验证结果，尚待4轮结束后重载最佳权重及最终核验。
- 旧基线没有optimizer状态，因此本次是从权重继续微调、重新建立优化器，不是历史断点恢复。新训练可以epoch边界恢复。
- 对比程序用原基线与自身做了完整859次检查/228名患者、500次患者bootstrap自检：AUC/F1差值及95%区间均为0，指标与原报告完全一致；报告 outputs/comparison-selfcheck.json。
- 本次训练使用的代码与依赖SHA256清单保存在 outputs/resnet18-finetune-20261104-103301/source_hashes.json。
- 新增 scripts/verify_candidate.py 最终本地证据交叉核对：权重/CSV hash、验证覆盖、指标一致、模型来源、100/5/3计时协议、预测一致性与大小；9项独立内存合成测试全部通过（含拒绝旧模型/旧CSV计时报告）。cost.py 现保存checkpoint、CSV、metadata哈希并拒绝计时期间输入变化；不改变推理计算。后处理代码版本单独记在 postprocess_hashes.json，训练代码未变。真实候选验收须等待训练及后处理结束后执行，该脚本不证明官方合规。


## 已验证完成

- Python 3.12.14 / torch 2.10.0+cu128 / torchvision 0.25.0+cu128 安装完成，RTX 3050 Laptop 4GB 的 CUDA 正反向检查通过。
- 精确包版本记录在 requirements-lock.txt，未修改系统 Python。
- 训练/验证/测试元数据和文件路径审核通过。原始数据未修改。
- 全部训练/验证图片 30,344 + 1,422 = 31,766 张完整解码通过，0 损坏；测试图像打开数为 0。
- 63 项单元测试：62 通过，1 项因 Windows 符号链接权限跳过。
- 小样本工程闭环通过：训练→保存→加载→检查级 CSV→预测一致性及本地计时。
- 全训练集一轮 ResNet18 ImageNet 基线训练完成；训练 3793 steps，约 30 分钟，完整验证集 859 次检查。
- Git main 已初始化、origin 为 https://github.com/hu-ihu/Chest-X-ray.git；未创建提交，未推送。
- 数据、原始压缩包、CSV、权重、训练输出、虚拟环境均被 .gitignore 排除。

## 首轮基线结果

模型目录：outputs/resnet18-baseline-20261104-022511

| 项目 | 结果 |
|---|---:|
| 检查级 Macro-AUC（保存模型重载，batch=1） | 0.7338196438718676 |
| Macro-F1（所有类别阈值 0.5） | 0.35849692521173865 |
| 定义有效的 AUC 类别 | 10/10 |
| 验证输出检查数 / CSV 数据行数 | 859 / 8590 |
| checkpoint 大小 | 44,795,083 字节，约 44.8 MB |
| 本地 100 图、5 图预热、3 轮、batch=1 平均耗时 | 55.7163 ms/图 |
| 计时样本对应检查的一致性核验 | 63 次检查，不一致 0，最大概率误差 0 |

训练期间 batch=8 验证 Macro-AUC 为 0.7338171255，与重载后 batch=1 的 0.7338196439 有极微小浮点排序差异。对外基线结果以 batch=1 共享推理器报告为准。predict.py 与 cost.py 共用该推理路径，100 图核验实测完全一致。

本地耗时来自 RTX 3050，且只计单图前向及预处理；检查级核验在计时外补齐同次检查的其他视图。它不是官方 T4 合规认证，不能据此保证正式比赛的 100ms 条件已通过。

## 本地证据

- outputs/audit.json：元数据、分组、标签和路径审计。
- outputs/image_integrity.json：训练/验证图像解码检查。
- outputs/smoke-20261104-022441-776244/smoke_summary.json：小样本工程闭环。
- outputs/resnet18-baseline-20261104-022511/run.json：训练配置。
- outputs/resnet18-baseline-20261104-022511/history.jsonl：一轮训练记录。
- outputs/resnet18-baseline-20261104-022511/best.pt：当前首轮基线。
- outputs/resnet18-baseline-20261104-022511/valid.report.json：完整检查级验证报告。
- outputs/resnet18-baseline-20261104-022511/benchmark-valid/cost.json：100 图本地计时与一致性。
- outputs/resnet18-baseline-20261104-022511/inference_manifest.json：推理权重大小与 SHA-256。
- weights/provenance.json 和 weights/torchvision-LICENSE.txt：公开 ImageNet 权重来源、完整 hash、软件许可证留存。仍需核对权重/数据授权和组委会要求。

## 仍需后续完成的比赛工作

1. 已完成首轮基线加4轮微调。后续若继续优化，应另立有界实验，优先关注纵隔心影增大和肺炎；当前结果不代表最终比赛性能，避免无休止验证集搜索。
2. 对当前 859 次验证检查注意患者聚类与小样本波动；不要因频繁调参过拟合验证集。
3. 新训练现已支持经 GPU 小样本等价性验证的 epoch 边界 resume（optimizer/scaler/scheduler/RNG完整状态）；旧首轮模型仍只有推理权重，只能通过 init-checkpoint 开始新的微调。恢复必须保持原总轮数和配置。
4. 确认官方多视图检查聚合与单图计时核验口径，再适配最终 cost.py 接口，在官方指定 T4 环境复测。
5. 准备附录 A 材料包及许可证核验；不能把本地工程验证当作组委会认可。
6. 配置锁定后才运行测试集预测。本次没有打开测试图像、生成正式 submission 或向比赛平台提交。

## 常用入口

训练 train.py；验证/预测 predict.py；本地计时 cost.py；严格提交校验 scripts/validate_submission.py；部署文件大小与 hash 清单 scripts/inference_manifest.py；发布前 Git 索引检查 scripts/check_publish.py。操作命令见 README.md。
