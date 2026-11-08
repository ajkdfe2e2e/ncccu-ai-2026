# 研究进度上传快照（2026-11-08）

采集时间：**2026-11-08T12:50:36+08:00**（北京时间）。此文件保存聚合指标、固定配方与当时状态；本地数据、权重、特征缓存、逐例预测和患者/检查标识不进入 Git。

## 当前已完成的最高验证 AUC 方案

| 评估范围 | 方案 | Macro-AUC | 状态 |
| --- | --- | ---: | --- |
| train 患者完整 OOF | 冻结 CXR-LT 1054D + 检查级 BCE/L2=.01 头 | 0.8270384304 | 有限候选选型估计 |
| 官方 valid，859 次检查 | 同一 CXR-LT 方案 | 0.8335505264 | 当前已完成实验最高本地 valid AUC |
| 官方 valid，859 次检查 | 2026-11-05 旧融合 | 0.8203251713 | 历史方案 |
| train 患者完整 OOF | CXR-LT 末阶段微调 | 0.8169781872 | 筛选失败，未替换当前方案 |

四个 OOF held 组均按患者隔离，完整 OOF 是拼接折外预测后计算。OOF 与官方 valid 是不同人群；不直接用两种分数的大小判断提升。监督特征来源、完整超参数、聚合/逐类 OOF 指标、模型与预测哈希见 [固定配方](best_study_probe_recipe_20261108.json)；实现见 `cxr/cxrlt_transfer.py`、`cxr/study_probe.py` 和 `scripts/run_frozen_study_probe.py`。

最高 valid AUC 模型的测试文件已在本地交付：`outputs/submission_test_1414_best_auc_083355_20261108.csv`。覆盖 2,218 张图、1,414 次检查、14,140 行，Gate=0.5，分类头保存重载重放最大概率误差为 0。CSV SHA-256：`7590ef61172423348bcf9a0c9afdb578b5416396d41a1c9357b03b0df232ec68`。它尚无线上成绩。用户反馈旧融合线上 **78.53**、Rad-DINO A+EMA **77.34**，线上指标定义尚未确认，均不可当成本轮测试 AUC。

当前方案 valid Macro-F1@0.5 为 **0.3763337400**，旧融合为 **0.4373950055**；更高 AUC 未带来该固定阈值下更高 F1。没有把线上测试结果用于重新拟合本轮分类头。

## 已完成的后续尝试

- 嵌套分类头优化未通过官方 valid 晋级门槛，保留原 BCE/L2=.01 头，详见 [嵌套分类头结果](NESTED_STUDY_HEAD_RESULTS_20261108.md)。
- PCA128+30 语义瓶颈的嵌套 OOF 为 0.8311058848，最终 valid 为 0.8304071711，未替换当前方案，详见 [瓶颈结果](BOTTLENECK_STUDY_HEAD_RESULTS_20261108.md)。
- 末阶段微调完整 OOF 为 0.8169781872，比当前方案低 0.0100602432，仅 1/4 held 组正增益；3,473 次有效更新、0 AMP 跳过。没有进行失败候选的 final train、官方 valid/test 推理或 CSV 导出，详见 [末阶段结果](CXRLT_STAGE_FINETUNE_RESULTS_20261108.md)。

## EVA-X 仍在运行

- 实验：`outputs/eva-study-probe-20261108-v2`；此刻状态 `running`，阶段 `extract_eva_train`，error 为 `None`。
- 最新进度文件时间 `2026-11-08T04:50:34.838576+00:00`，真实 train 图像已提取 **12,112 / 30,344**；这只是工程进度，**尚无该候选真实 AUC**。
- 已冻结 plan SHA-256：`7da5f802321840d5fc87a74cbb225c0619882e6644d66138fc50d66b5d3c712b`。仅比较 EVA389D 与 CXR-LT+EVA1443D 两个检查级头，使用四患者组 OOF 筛选，最多一个候选进入一次官方 valid 比较；不得把运行中状态记作完成。
- 公开权重 `MapleF/eva_x@35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee`；权重 SHA-256 `1a5b63570ca4094c1118018e44d3012a95b904132942a8edb2fc0e5dc8aa0e97`。作者源码 `hustvl/EVA-X@eedd642c85339b25682ceb574b644cb9afe88e62`。见 [固定实验方案](EVA_STUDY_PROBE_PLAN_20261108.md) 和 [准备记录](EVA_X_PREPARATION_20261108.md)。
- 两次工程启动失败的原始现场保留；v2 已开始真实特征提取。上传没有改动当前实验锁定的执行源码或统计方案。

## 上传范围与续接限制

当前账号已有私有赛事仓库 `ajkdfe2e2e/ncccu-ai-2026`，具备 ADMIN/push 权限。原 `hu-ihu/Chest-X-ray` 通过 Git 与 GitHub API 均返回 404；保留 `origin`，新增 `github-progress` 指向私有赛事仓库，上传分支为 `experiment/progress-20261108`。

本次保存源码、测试、方案、聚合结果与公开权重来源，遵守原 `.gitignore`，不强制加入忽略文件。新电脑克隆只获得工程与实验记录；复现或续跑还需要另行迁移本地数据、权重、冻结实验目录和缓存，并核对计划中的路径、环境与哈希。当前持续提升目标为 active，本次优先完成进度上传，既有特征提取继续运行。

## 本次上传验证

- 2026-11-08 全仓 `unittest discover -s tests`：**338 项通过，0 failures，0 errors，0 skipped**。使用 CPU，进程内关闭 CUDA，未竞争后台 EVA-X 的 GPU。
- 对 155 个 Git 候选文件完成 Python AST / JSON 解析、大文件与凭证模式扫描，没有发现问题；暂存后 `scripts/check_publish.py` 对 155 个 index 路径检查通过，`git diff --cached --check` 通过。
- 当前最佳本地分类头和交付 CSV 的 SHA-256 已重新计算，与固定配方/既有核验记录一致；没有重新训练或重跑完整真实数据推理。
- EVA-X prepare 成功路径的合成测试改为自行构造 provenance，不再读取被忽略的本机权重目录；测试与公开数据/权重解耦。

公开预训练与比赛数据的重叠未知；训练 OOF 用于候选选型且历史 audit 组已参与研究；官方 valid 经多轮历史选优。以上结果不构成独立泛化保证。源模型引用与许可记录保留。
