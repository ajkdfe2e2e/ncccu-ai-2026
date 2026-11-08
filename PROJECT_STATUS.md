# 当前进度：保留线上78.98，BioViL-T 固定实验已启动（2026-11-08）

用户最新反馈 `submission_test_1414_best_auc_083355_20261108.csv` 的线上成绩为 **78.98**，超过旧融合 **78.53**，提高 **0.45 分**。已核对用户附件与本地交付 CSV 的 SHA-256 完全一致，该 CXR-LT 方案为当前保留模型及已反馈的线上最好。官方主指标为 Macro-AUC，若反馈来自主分栏，则展示精度对应 **0.7898**；用户页面与榜单划分尚未独立读取。详见 [线上结果记录](docs/ONLINE_RESULTS_20261108.md) 和 [官方评分口径及医学预训练研究边界](docs/OFFICIAL_SCORING_20261108.md)。后续实验以当前模型比较；既有冻结方案与原始产物保持不变。

固定患者聚合 v2 于北京时间 **13:47:47** 正常结束：完整患者 OOF **0.8282252394**，相对当前 **+0.0011868090**，四组均提高，OOF 门槛通过；official valid 点估计 **0.8360009305**，相对当前 **+0.0024504041**。该增益的患者 bootstrap 描述性 95% 区间 **[-0.0018000031, 0.0068323706]** 跨零，未满足预先固定的双基线 valid 晋级规则。独立实现已重算 OOF、valid 及两组各 1,000 次患者 bootstrap，结果一致。`candidate_promoted=false`、`test_read=false`、`test_csv_exported=false`；本地更高点估计与保留采用的模型明确分开，没有新测试 CSV。首次 prepare 的文件名引用失败与 v2 正式实验均保留，详见 [患者聚合结果](docs/PATIENT_SMOOTHING_RESULTS_20261108.md)。

EVA-X 本轮已于北京时间 13:10:03 正常完成，两个候选均未晋级：EVA389D 完整患者 OOF **0.8153107264**；CXR-LT+EVA1443D 为 **0.8274166581**，相对当前 **+0.0003782277**，不足固定的 +0.001 门槛。独立 verifier 重算全部 OOF 与八个头通过，最大重放误差 `4.996e-16`；未拟合全量头、未评估 valid/test、未导出新 CSV。见 [EVA-X 结果](docs/EVA_STUDY_PROBE_RESULTS_20261108.md)。

CheXficient 实验于北京时间 **15:09:06** 正常结束，`status=complete`、`error=null`。CheXficient512D 完整患者 OOF 为 **0.8286274797**；CXR-LT+CheXficient1566D 为 **0.8315443007**，相对当前 **+0.0045058703**，四组均提高，两候选都通过 OOF 门槛，按固定规则选中联合特征。唯一 official valid AUC **0.8394749558**，相对当前 **+0.0059244293**，患者 bootstrap 描述性 95% 增益区间 **[-0.0011930675, 0.0133285088]** 跨零；相对旧融合的区间为 **[0.0033159681, 0.0336507941]**，但双基线规则要求同时通过，因此未晋级。`candidate_promoted=false`、`test_extracted=false`、`test_csv_exported=false`，当前采用模型与线上最好仍为 **78.98**。独立 OOF 已复核全部八个头和完整 AUC，重放最大误差 0；终态 valid 重放及两组各 1,000 次患者 bootstrap 也已复核通过，valid 预测最大误差 0，保留原不晋级结论。终态核验覆盖 139 项输入、20 项源码、124 项快照、12 项当前模型源码和 37 项产物；明确排除 6 个 test 相关锁，未读取 test 特征矩阵或 test 预测 CSV 内容。全量头运行 188/200 轮，联合候选 held 组 2 达到固定 max_iter=200，保留原训练预算。计划 SHA-256 `f71f2c370ecb5c78228884d0dd315a2c3481cf4875adc310af88ac5ae8e8acb4` 及冻结源码/输入不变。详见 [结果](docs/CHEXFICIENT_STUDY_PROBE_RESULTS_20261108.md)和 [固定方案](docs/CHEXFICIENT_STUDY_PROBE_PLAN_20261108.md)。

BioViL-T 正式实验于北京时间 **15:56:47** 从冻结快照启动，实际 worker PID **14512**，随后进入 `extract_biovil_train`、`error=null`；**15:58:52** 已保存 **102/30,344** 张真实训练图像特征，进程存活。输出目录 `outputs/biovil-t-study-probe-20261108`，计划 SHA-256 `ee6efa98450f5e26e47672d2c5e59d457903faa474da478c14033f46ac3cd981`。固定比较 CXR-LT+BioViL-T 1,438D、CXR-LT+CheXficient+BioViL-T 1,950D；特征按检查均值、普通 BCE/L2=.01、max_iter200，四患者组共八个 CV 头，沿用双基线 OOF/valid 晋级规则。当前没有 BioViL-T AUC 或新 CSV，线上最好仍为78.98。

启动前完整 CPU 回归 **396 项通过**，随后新增的输入注册成功路径在五项输入测试中通过；九项执行测试覆盖两条合成导出路径，四项 adapter 测试及实际权重合成 GPU 中断续跑也通过。冻结计划加载核验201项输入、20项源码、186项快照锁；独立注册v2另核验198项非test输入、20项源码、183项非test快照、36项CheXficient产物，明确列出6个test元数据锁和1个延迟valid矩阵的排除范围。首次独立核验漏匹配一个相对路径，曾流式计算CheXficient valid矩阵哈希，未加载矩阵进行拟合、预测或评分；原报告保留，修正版v2记录该工程偏差，正式执行方案未改。公开资产22项与原训练缓存身份/order一致；新进程以 `-B` 禁止写字节码缓存。见 [固定方案](docs/BIOVIL_T_STUDY_PROBE_PLAN_20261108.md) 和 [准备及启动记录](docs/BIOVIL_T_PREPARATION_20261108.md)。持续提升目标仍为 active。以下为上传时快照和保留的历史实验记录，其中“最高 valid”“尚无线上分数”“仍在提取”等描述属于当时状态，当前采用模型与最新候选以上方结果为准。

---

# GitHub 研究快照（2026-11-08）

采集时间 **2026-11-08T12:50:36+08:00**。当前本地最高官方 valid Macro-AUC **0.8335505264**，患者完整 train OOF **0.8270384304**；测试 CSV 已本地交付、尚无线上分数。EVA-X v2 状态 `running` / `extract_eva_train`，train 特征 **12,112/30,344**，尚无真实候选 AUC。持续提升目标为 active。

本次先按用户要求上传已有进度到私有赛事仓库的独立分支。详见 [聚合快照](docs/PROGRESS_SNAPSHOT_20261108.md) 和 [最高 valid AUC 配方](docs/best_study_probe_recipe_20261108.json)。以下为保留的历史实验记录，可能含当时的 active/暂停或尚未上传描述；当前状态以上方快照为准。

---

<!-- cxrlt-stage-status-20261108:start -->
# 项目状态：末阶段微调完成，保留当前最高 AUC 模型（2026-11-08）

北京时间 **2026-11-08 12:22:01**，`outputs/cxrlt-stage-finetune-20261108` 正常完成，`status=complete`、`error=null`、`candidate_promoted=false`、`test_csv_exported=false`。完整患者 OOF Macro-AUC **0.8169781872476143**，低于当前模型 **0.8270384304066101**，差值 **-0.010060243158995785**；只有 **1/4** held 组提升，未通过预先固定门槛。没有做本轮官方 valid/test 推理或全量 train 微调，没有导出失败候选 CSV。

- 四个 held 组 AUC 依次为 0.8220596395、0.8279154380、0.8243647762、0.8327069543；相对当前模型同组差值 -0.0014075878、+0.0008531774、-0.0020747602、-0.0009353367。全部有完整十类 AUC，训练/held 患者交集均为零。完整 OOF 是拼接 held 预测后计算，不能用四折平均代替。
- 固定方案及冻结源码不变，plan SHA256 `08b85902dc97fd88ac8a24f698fc20589b48d10ee2b41357baaabc314364de42`；四折共 3,473 次有效更新、0 次 AMP 跳过。四个保存模型哈希与独立 OOF 重算通过。核验 `outputs/cxrlt-stage-partial-oof-check-20261108.json` 已标注 `complete_oof=true`；正式产物清单在该实验 `artifacts.json`。
- 当前最高官方 valid Macro-AUC 仍为 **0.8335505264258105**，对应 **`outputs/submission_test_1414_best_auc_083355_20261108.csv`**。SHA256 `7590ef61172423348bcf9a0c9afdb578b5416396d41a1c9357b03b0df232ec68`。重载本地分类头按用户 test 元数据重放，2,218 张图 / 1,414 次检查 / 14,140 行、完整十类与最大概率误差 0 已核验。该文件尚无线上分数；用户反馈线上最高仍为旧融合 **78.53**。
- EVA-X-S 公开权重与作者源码已经固定并保存来源许可。实际架构为 EVA02/SwiGLU/RoPE，恢复 26 个位置缓冲；CPU 严格加载/合成重放通过，7 项 adapter 测试通过。12:23:25 的合成 GPU 检查以 batch16 生成有限 16x389 特征，两次完全一致；峰值 allocated 142.21 MiB / reserved 170 MiB，没有真实数据推理或 AUC。证据 `outputs/eva-x-preparation-20261108/gpu_engineering_check.json`。
- EVA-X 实验 **`outputs/eva-study-probe-20261108-v2` 已于 12:37:13 启动**，真实 worker PID **39632**，12:37:52 状态 `running`、`phase=load_and_validate_train_cache`、`error=null`。冻结计划 SHA256 `7da5f802321840d5fc87a74cbb225c0619882e6644d66138fc50d66b5d3c712b`；14 个执行源码、68 个输入快照文件已注册。25 项定向 runner/adapter 测试、编译及合成 GPU smoke 通过，缺失缓存绑定时拒绝采用既有矩阵，完整 prepare→缓存绑定合成成功路径通过。固定方案 `docs/EVA_STUDY_PROBE_PLAN_20261108.md` 仅比较 EVA389D 与 CXR-LT+EVA1443D 两个检查级 BCE/L2=.01 头，同四患者组 OOF 筛选，最多一次官方 valid 候选；尚无该方向真实 AUC。首轮注册前环境记录错误、首次执行的 Windows 分隔符错误均在真实特征提取/拟合/AUC前停止，原始记录分别保留于 `outputs/eva-study-probe-20261108-prepare-incomplete` 与 `outputs/eva-study-probe-20261108`，未修改其冻结源码或统计方案。

本轮结果 `docs/CXRLT_STAGE_FINETUNE_RESULTS_20261108.md`。现有模型、数据、CSV 和 F 盘前缀缓存保留；47 项 stage 工程测试和先前 smoke 记录保留。公开预训练重叠未知、历史选优等限制仍在。以上是微调完成时的历史记录；当前目标与上传状态以上方最新进度快照为准。
<!-- cxrlt-stage-status-20261108:end -->

---

<!-- bottleneck-head-status-20261108:start -->
# 项目状态：最高验证 AUC 模型的测试 CSV 已整理（2026-11-08）

用户明确要求提供验证 AUC 最高模型的测试预测。已从当前 CXR-LT 全量训练头重新重放测试缓存，导出 `outputs/submission_test_1414_best_auc_083355_20261108.csv`。对应官方 valid Macro-AUC **0.8335505264258105**，覆盖 **2,218 张测试图 / 1,414 次检查 / 14,140 行**，每次检查十类，Gate=0.5。格式、覆盖和概率范围通过核验，重载与模型输出逐元素相等，亦与原正式 CSV 内容完全相同。

- CSV SHA-256：`7590ef61172423348bcf9a0c9afdb578b5416396d41a1c9357b03b0df232ec68`；核验文件 `outputs/submission_test_1414_best_auc_083355_20261108.verification.json`。原实验 `outputs/cxrlt-supervised-study-probe-20261108`、权重、原 CSV 均保留。
- 本轮语义瓶颈实验 `outputs/cxrlt-bottleneck-head-20261108` 已于北京时间 **2026-11-08 10:31:33** 正常完成，未通过官方 valid 晋级门槛，`candidate_promoted=false`、`test_csv_exported=false`。30D 语义头完整训练四折 AUC=0.8200003790；PCA128+30 方案四个外层均被选中，嵌套 OOF=0.8311058848，相对当前 OOF 增益 +0.0040674544，4/4 组提高。
- PCA128+30 最终固定候选 valid AUC=0.8304071711，低于当前 0.8335505264，差值 -0.0031433553，患者配对 bootstrap 描述性 95% 差值区间 [-0.0103605081, 0.0032266307]；F1@0.5 为 0.3438792023。共 20 个缓存 CV 头 + 1 个全量 train 头、11 次 PCA，无头达到 max_iter。没有导出失败候选的测试预测。
- 拟合前 45 项定向测试通过；验证器另有 4 项合成边界测试通过。独立 valid AUC/F1 与两基线各 1,000 次 bootstrap 已复现，最终 head 到 valid CSV 精确重放。复核记录 `outputs/cxrlt-bottleneck-head-validation-replay-20261108.json`。输入、执行源码和方案仍冻结，历史选优与公开预训练重叠未知的限制保留。
- 完成后的只读核验器重算 46 个输入、11 个源码、54 个产物哈希，PCA 训练统计及全部患者 CV 预测重放通过。报告 `outputs/cxrlt-bottleneck-head-postrun-verification-20261108.json`；结果 `docs/BOTTLENECK_STUDY_HEAD_RESULTS_20261108.md`。
- 用户已反馈线上最高仍为旧融合 **78.53**；本次交付的 CXR-LT CSV 尚无线上分数。本轮没有提交比赛、提交代码或推送仓库。
<!-- bottleneck-head-status-20261108:end -->

---

<!-- nested-head-status-20261108:start -->
# 项目状态：CXR-LT 分类头嵌套优化已完成，保留当前模型（2026-11-08）

北京时间 **2026-11-08 10:01:22**，`outputs/cxrlt-nested-head-20261108` 正常完成，`status=complete`、`error=null`。本轮未通过官方 valid 晋级条件，**保留原普通 BCE / L2=0.01 分类头，没有生成新的测试 CSV**。

| 同一评估人群内比较 | 当前 CXR-LT | 本轮普通 BCE / L2=0.1 | 差值 |
|---|---:|---:|---:|
| train 患者折外：18,513 次检查 / 11,055 位患者 | 0.8270384304 | 0.8352666978 | +0.0082282673 |
| 官方 valid：859 次检查 / 228 位患者 | 0.8335505264 | 0.8330900995 | -0.0004604269 |

- 固定比较六个配置；四个外层患者组都由内层选中普通 BCE / L2=0.1，且四组训练 OOF 均提升。完整 train 四折最终选参也选中同一配置。共拟合 60 个缓存 CV 头 + 1 个全量 train 头。
- 固定最终配方后只评价一个 valid 候选。相对当前模型的 1,000 次患者配对 bootstrap 描述性 95% 差值区间为 **[-0.0079335381, 0.0057550412]**，包含零，点增益也未达到 +0.001。valid F1@0.5 为 0.3763337400 → 0.3487373211。
- **32 项定向测试通过**。46 个输入锁、10 个源码锁、144 个完成产物哈希一致；训练特征重新汇总、患者隔离、内外层与六候选四折预测重放通过。普通 BCE / L2=0.01 的完整 OOF 与原缓存逐元素相同。主代理独立计算 valid AUC/F1，并重新运行两基线各 1,000 次患者 bootstrap，原报告精确复现。
- 当前已完成 CXR-LT 测试文件仍为 `outputs/cxrlt-supervised-study-probe-20261108/submission_test_1414_cxrlt_study_20261108.csv`，原 SHA-256 未变；当前模型 valid AUC **0.8335505264**。用户已反馈线上最高仍为旧融合 **78.53**，新 CXR-LT 尚无线上分数。
- 详细结果：`docs/NESTED_STUDY_HEAD_RESULTS_20261108.md`；固定方案：`docs/NESTED_STUDY_HEAD_PLAN_20261108.md`；核验：`outputs/cxrlt-nested-head-postrun-verification-20261108.json`、实验目录 `validation_replay_20261108.json` / `final_verification_20261108.json`。历史选优及公开预训练重叠未知的限制仍在。本轮没有提交比赛、提交代码或推送仓库。
<!-- nested-head-status-20261108:end -->

---

# 历史记录：监督胸片 ConvNeXtV2 迁移通过本地门槛，测试 CSV 已生成（2026-11-08）

北京时间 **2026-11-08 09:03:50**，`outputs/cxrlt-supervised-study-probe-20261108` 正常完成，`status=complete`、`error=null`，`candidate_promoted=true`、`test_csv_exported=true`。本轮四个固定候选由训练集患者折外结果选型，最终选择**所有图像特征均值 + 本地十类线性分类头，alpha=1.0**，即新模型单独预测。下列每行均在同一批检查上比较旧版与新版；两行使用不同人群，不能相互排名。

| 比较人群 | 固定旧版 Macro-AUC | 新候选 Macro-AUC | 差值 |
|---|---:|---:|---:|
| train 患者折外：18,513 次检查 / 11,055 位患者 | 0.8182590867 | **0.8270384304** | **+0.0087793437** |
| 官方 valid：859 次检查 / 228 位患者 | 0.8203251713 | **0.8335505264** | **+0.0132253552** |

- 四个 held 组的完整十类 AUC 均提高，增益分别为 **+0.007713、+0.006681、+0.011291、+0.009544**。OOF 是有限候选选择结果，原 audit 现在属于训练交叉验证组，不是嵌套无偏验证。
- 最终头只使用用户 train 拟合。固定最终配方和头的哈希后，仅评估一个官方 valid 候选。1,000 次患者配对 bootstrap 的描述性 95% AUC 差值区间为 **[0.0001264, 0.0264018]**；全部 1,000 次均有完整十类 AUC。旧 valid 及旧配方有历史选优，区间下界接近零，不能据此保证线上收益。
- 官方 valid 有 7 类 AUC 提高、3 类下降。Consolidation 从 **0.8546981122 降至 0.8098828814**；Macro-F1@0.5 从 **0.4373950055 降至 0.3763337400**。本轮按主指标 Macro-AUC 晋级，没有依据 valid 逐类回退、重新选比例或调阈值。
- 测试预测：`outputs/cxrlt-supervised-study-probe-20261108/submission_test_1414_cxrlt_study_20261108.csv`，覆盖 **2,218 张图 / 1,414 次检查 / 14,140 行**，每次检查十类，导出回读与模型概率逐元素相等。SHA-256：`7590ef61172423348bcf9a0c9afdb578b5416396d41a1c9357b03b0df232ec68`。
- 冻结作者监督预训练 ConvNeXtV2 及 BN，每图提取 1,024 维空间均值 + 30 维原始分类 logits，汇总为检查级特征，在本地学习完整十类头。保留公开来源、MIT 标注和固定 revision，不把公开 `Normal` 直接等同本地 `No Finding`。外部预训练患者与本地匿名数据是否重叠仍未知。
- 训练特征 30,344×1,054 已缓存；**22 项定向测试通过**，真实灰度样本与作者预处理逐元素一致。输入和执行源码指纹、患者隔离、候选预测重放、验证与测试导出均留有核验记录。配方 SHA-256：`468bafdf7eaea3e2665ba6147e2f47313722e5559022cbe639a48dda1d50e17b`；最终头 SHA-256：`6a90b170225f23ca3e602acaff32907b96a5307a1dc86b90280ad26081813d32`。
- 完成后由 Luna 只读复核 13 项输入锁、12 项源码锁、全部特征缓存哈希、四候选总体及各组 AUC，以及最终测试特征到 CSV 的精确重放；主代理另用正负样本成对比较独立重算 valid AUC/F1，并重新运行固定种子的 1,000 次 CPU 患者 bootstrap，完整比较结果与原报告完全一致。该复现仍是描述性验证，不能消除历史选择偏差。
- 主要证据：实验目录 `report.json`、`validation_report.json`、`submission_verification.json`、`validation_replay_20261108.json`、`postrun_verification_20261108.json`；固定方案：`docs/SUPERVISED_STUDY_PROBE_PLAN_20261108.md`。当前新候选**没有线上分数**；用户已反馈的线上最高分仍为旧融合 **78.53**，对应旧 CSV 保留原哈希。本轮没有提交比赛、提交代码或推送远端。

---

# 历史记录：监督胸片 ConvNeXtV2 的检查级迁移实验已启动（2026-11-08）

北京时间 **2026-11-08 08:43:44** 启动 `outputs/cxrlt-supervised-study-probe-20261108`。**08:44** 已观察冻结源码执行器进入 `extract_train`，进度超过100张，`status=running`、`error=null`。这是启动快照，实时状态以实验目录 `experiment.json` / `train_features/progress.json` 为准，尚无新 AUC。

- 新骨干为作者公开的 `hieuphamha/cxrlt2026-task1-convnextv2`；模型卡描述先在 MIMIC-CXR 14类监督训练，再在 PadChest 30类微调。固定 revision `24bf8bdfd8d0262899167b030be16ab89f78ae20`，351,075,736字节 safetensors 已完整校验 SHA-256 `6950f7b97422b807e16a821a954e5d62b6868f8fbfb1e841f6d090c91e4ca17b`。保留作者README、模型代码、MIT标注及本地provenance。
- 最终30类头不是本地十类头；不将 `Normal` 直接映成 `No Finding`。冻结官方骨干及BN，取1,024维空间均值和30维原始logits，使用用户train学习完整十类检查级分类头。
- 固定比较均值/分视图两种汇总，各比较单模型与 `0.8*旧融合+0.2*新模型`。沿用18,513次检查/11,055位患者的四个held组；OOF增益至少0.001、至少三组提高才晋级一个固定官方valid候选。官方valid再通过AUC、十类完整性及描述性患者bootstrap门槛，才导出新测试CSV。
- **22项定向测试通过**；39张真实GPU图完成1054维提取与缓存精确重载；六张跨训练数据位置样本与作者OpenCV灰度预处理逐元素一致。batch=2、预处理线程=4是在AUC评分前的工程测速中确定，预处理张量在0/2/4线程下逐元素一致。
- 方案：`docs/SUPERVISED_STUDY_PROBE_PLAN_20261108.md`；启动核验：实验目录 `startup_verification.json`；日志：`runner.stdout.log` / `runner.stderr.log`。公开预训练患者重叠未知，旧valid及旧配方有历史选优；本次OOF选型不是嵌套无偏验证。
- 当前线上已反馈最高分仍为旧融合 **78.53**。下面的CXformer路线已明确失败；本次新监督骨干尚无本地评分或线上成绩。

---

# 历史记录：冻结 CXformer 实验已完成，未超过旧融合（2026-11-08）

北京时间 **2026-11-08 08:21:49**，`outputs/cxformer-study-probe-20261108` 正常完成，`status=complete`、`error=null`。四个候选均未通过预先固定的训练 OOF 晋级门槛，**没有进入官方 valid / test 推理，没有生成新测试 CSV**。

| 同一 train 患者折外表：18,513 次检查 / 11,055 位患者 | 十类 Macro-AUC | 相对旧基线 |
|---|---:|---:|
| 固定旧融合 OOF | **0.8182590867** | — |
| CXformer 全视图均值分类头 | 0.6854368551 | -0.1328222316 |
| CXformer 分视图分类头 | 0.6904278158 | -0.1278312709 |
| 0.8 旧融合 + 0.2 全视图分类头 | 0.8177491375 | -0.0005099492 |
| 0.8 旧融合 + 0.2 分视图分类头 | 0.8181295853 | -0.0001295014 |

- 30,344 张训练图的 1,536 维特征已全部提取、保存，可供后续独立方案复用；SHA-256 为 `0e3417bcfa82b15f46bc27b0f0bd426f51f9b75effca4465b937226b38463a6d`。不重跑已完成缓存。
- 已复核 13 个输入锁、10 个源码锁、四个候选的总体 / 各 held 组 AUC 和融合重放。另独立重载模型，对原缓存开始、中间、末段共六张图重新提取，逐元素相等、最大误差 0。特征有限且存在差异，训练梯度较小；这些检查暂未发现明显工程错误，不能据此排除所有模型或预处理问题。
- 本次仅检验“最后一层特征 + 冻结线性分类头”的两种检查级汇总，不代表官方四层特征或微调方案的结果。公开论文基准分不等于本地实测提升。
- 当前已反馈线上最高分仍为旧融合 **78.53**；旧官方 valid Macro-AUC 为 **0.8203251713**。本段 OOF 与官方 valid、线上分数使用不同人群，不能直接排名。旧测试 CSV 保持原 SHA-256。
- 结果：`outputs/cxformer-study-probe-20261108/report.json`；六图重算：`outputs/cxformer-study-probe-diagnostic-20261108/extraction_replay.json`。下一方向优先核验自带监督分类头的公开权重，并在固定患者组上筛选。

---

# 历史记录：冻结 CXformer 的检查级实验已启动（2026-11-08）

北京时间 **2026-11-08 07:33:54** 启动 `outputs/cxformer-study-probe-20261108`。**07:35:35** 已核查冻结源码执行器实际进程存活，训练特征已完成 **802 / 30,344 张图**，`status=running`、`phase=extract_train`、`error=null`。这是启动快照；实际进度以该目录 `experiment.json` 和 `train_features/progress.json` 为准，尚无新 AUC。

- 本轮使用已有公开 `m42-health/CXformer-base`，冻结整个编码器，先汇总同一检查的图像特征，再仅用用户 train 的 **18,513 次检查 / 11,055 位患者**训练十输出小分类头。每次检查只贡献一个训练样本。此前尚未在此项目评估过该编码器。
- 固定比较两种输入：所有图像特征均值，以及正位 / 侧位 / 其他视图分组均值；各自仅比较单模型与 `0.8*旧融合+0.2*新模型`。使用 train 内四个患者 held 组选择一次，原 audit 现在属于交叉验证组。没有按官方 valid 调整分类头参数或融合比例。
- 训练 OOF 总体 Macro-AUC 增益至少 0.001、至少三组正增益才晋级。候选分类头及配方保存重载并登记哈希后，仅评估一个固定官方 valid 候选；最终增益至少 0.001、十类 AUC 均定义且患者 bootstrap 描述性区间下界大于零才导出新测试 CSV。失败保留旧提交。
- **17 项定向测试通过**，覆盖患者隔离、总体基线计算、缓存恢复、保存重载、晋级门槛与产物篡改拒绝。39 张真实 GPU 图像的串行 / 两线程预处理特征逐元素一致，最大误差 0；该小样本只用于工程验证。
- 已冻结 10 份源码 / 方案文件、13 个输入锁及软件环境。模型 revision 为 `f778e2ac267fdd651205e07b6fd1113754b146bb`；方案 SHA-256 为 `deeb0fd6fa3f4efeb474f3d1647f581514e73801d331bc6ebaec710620614c12`。输入 batch=2，最多两线程预处理，支持从已提交特征游标恢复。
- 方案：`docs/FROZEN_STUDY_PROBE_PLAN_20261108.md`；启动核验：实验目录 `startup_verification.json`；日志：`runner.stdout.log` / `runner.stderr.log`。后续阶段会自动串行执行，当前没有新验证成绩或测试提交文件。
- 当前已反馈线上最高分仍为旧融合 **78.53**，其 CSV 与权重保留。旧 valid 曾参与历史选优，公开预训练数据与本地数据是否重叠未知，尚不能将本实验称为实际提升。

---

# 历史记录：完整旧融合实验已完成，保留旧提交（2026-11-08）

`outputs/legacy-full-oof-20261107` 于北京时间 **2026-11-07 23:48:44** 完成，八个阶段全部正常退出，`status=complete`、`error=null`。本轮在患者折外预测上训练旧八来源融合的权重，并为折外表补训 ResNet18；没有继续微调旧版全部部署主干。

| 同一官方 valid：859 次检查 / 228 位患者 | 旧融合 | 本轮候选 |
|---|---:|---:|
| 十类 Macro-AUC | **0.8203251713** | 0.8181880641 |
| Macro-F1@0.5 | 0.4373950055 | **0.4467832117** |

- 主指标 AUC 差值为 **-0.0021371072**，1,000 次患者配对 bootstrap 描述性 95% 区间为 **[-0.0051707437, 0.0002797303]**。未通过预先固定的提升门槛，`guardrail_passed=false`、`candidate_promoted=false`；保留旧版，**本轮没有导出新测试提交 CSV**。F1 上升不能作为 Macro-AUC 改善的证据。
- 最大单类下降为 Consolidation：0.8546981122 → 0.8340552971。本轮没有依据该验证结果重新调参或逐类回退。元模型训练表上的 AUC 0.8182590867 → 0.8218102916 仅属训练诊断，不能称为独立验证提升。
- 2026-11-08 完成事后复核：179 个输入锁及八阶段产物指纹一致；18,513 次检查 / 11,055 位患者的折外覆盖、患者隔离和预测矩阵一致性通过；旧版和候选验证预测均精确重放，重新计算的指标与报告一致。旧测试提交 SHA-256 仍为 `23f27c9f319ad7153cd1bd9b12064b2ed54111fdb6b388b794246e72e3a6d285`。
- 结果报告：`outputs/legacy-full-oof-20261107/report.json`；逐类比较：同目录 `validation_comparison.json`；复核记录：同目录 `postrun_verification_20261108.json`。`candidate_valid.csv` 仅为验证集预测，不是测试提交。
- 目前已反馈的线上最高分仍为旧融合的 **78.53**，对应 `outputs/submission_test_1414_best_20261105.csv`；本轮候选没有线上分数。旧配方及官方 valid 曾参与历史选优，本次比较与区间不构成无偏泛化证明，公开预训练数据是否重叠仍未知。

---

# 历史记录：完整旧融合的患者折外实验已启动（2026-11-07）

北京时间 **2026-11-07 17:11:27** 启动 `outputs/legacy-full-oof-20261107`，已核查执行器及第一组 `resnet_cv_A_fold0` 训练子进程均存活。**17:16:13** 已观察到正式第一轮 **100/2,049 批次**，有限训练损失 0.9516060704，学习率 3e-4；该组训练 5,896 位患者，留出 2,948 位患者，`smoke=false`。首段冷缓存速度不能作为后续各轮或整个实验的稳定 ETA。此处是启动快照，实时阶段和进度以实验目录的 `experiment.json`、`runs/<stage>/progress.json` 为准，不能把本段当作训练完成证据。

- 在旧八来源融合上学习完整权重，包含 ResNet18、Rad-DINO 原图、Rad-DINO 翻转均值和五个公开冻结来源。保留旧配方 80% 的权重约束，允许另外 20% 在所有可用来源间调整。上一轮只学习公开子融合的失败结果保留在下方。
- 仅使用用户 train 的 **30,344 张图 / 18,513 次检查 / 11,055 位患者**学习参数。复用原患者分区，四个 held 组覆盖所有训练患者；前三组分别由 5,896 位其他开发患者训练，原保留组由全部 8,844 位开发患者训练。每位患者的本地基础预测均不得使用包含该患者标签的训练模型。原保留组现在属于训练数据。
- 正在从公开 ImageNet 初始化补训四组 ResNet18，每组固定 1+4 轮；随后复用已完成的 A/raw 检查点补齐四组 Rad-DINO 原图及 TTA 预测。基础折外表上的融合拟合指标只是元模型训练诊断，不作为独立验证成绩。
- 34 项定向测试通过；真实 GPU ResNet18 小样本完成五轮及保存重载预测（20 次更新尝试，其中 19 次成功、1 次 AMP 溢出跳过）。Rad-DINO 36 张图 / 21 次完整检查的 batch=2 对齐测试通过，原图预测与历史缓存最大误差为 **0**。小样本不会进入正式拟合。
- 179 个输入文件指纹及代码快照已经冻结，包含患者分区、训练清单、公开和本地检查点、旧配方和旧 CSV。固定拟合参数在 `docs/LEGACY_FULL_OOF_PLAN_20261107.md`；计划 SHA-256 为 `82e87d2bdc219c59c32ee466fb65436d5258b284e39d8f37bf935d293ef9da7c`。
- 融合配方先拟合并登记哈希，再执行同一官方 valid 比较。仅当十类 AUC 均有定义、增益至少 0.001 且患者配对 bootstrap 描述性 95% 区间下限大于零时，自动导出单独的新测试 CSV；否则保留旧提交。没有新的实测 AUC 或线上成绩，旧版用户反馈的 **78.53** 仍是当前提交基准。
- 阶段日志位于 `outputs/legacy-full-oof-20261107/logs/`，总日志为 `runner.stdout.log` 和 `runner.stderr.log`。训练按 epoch 保存恢复点；未恢复的中断现场和所有旧权重保留。此前中断的 A+EMA 全量重训没有恢复。

---

# 历史记录：公开子融合训练已完成，候选未通过保留门槛（2026-11-07）

按用户“用旧版融合为底子训练”的要求，已在完整 train 的 **30,344 张图 / 18,513 次检查 / 11,055 位患者**上完成融合权重训练。复用五个公开模型的预测缓存，固定旧本地 ResNet18、Rad-DINO 原图及 TTA 的贡献；每类公开部分最多重新分配 20% 的归一化权重。本轮没有微调模型主干。全部方案和参数在拟合前固定，最终配方先保存并哈希，再读取官方 valid 标签。

| 同一官方 valid：859 次检查 / 228 位患者 | 旧融合 | 训练后的候选 |
|---|---:|---:|
| 十类 Macro-AUC | **0.8203251713** | 0.8189976511 |
| Macro-F1@0.5 | 0.4373950055 | 0.4383155202 |

- AUC 差值 **-0.0013275201**，1,000 次患者配对 bootstrap 描述性 95% 区间 **[-0.0030665234, 0.0001961731]**；均有十类 AUC。未通过预先固定的 AUC 增益及区间门槛，**保留旧版 78.53 的提交文件**，本轮未生成新的测试提交 CSV。
- 公开子融合的三折患者折外 AUC 分别提高 0.002187、0.002778、0.002911；这些只评价公开子融合的七个活跃类，不能当作完整融合收益。完整融合的下降主要来自 Consolidation：0.8546981122 → 0.8415155891。未按该验证结果逐类回退或重新调参。
- 23 项定向测试通过；独立重算候选验证指标和 CSV 精确回读一致；70 个输入锁均未变化；本地模型权重系数逐元素未变；公开权重为零的 1、8、10 类概率逐元素未变。旧提交 SHA-256 保持 `23f27c9f319ad7153cd1bd9b12064b2ed54111fdb6b388b794246e72e3a6d285`。
- 实验报告：`outputs/legacy-public-anchored-train-20261107/report.json`；完整逐类比较：同目录 `validation_comparison.json`；训练方案：`docs/LEGACY_ANCHORED_TRAINING_20261107.md`；解释与重现：`docs/LEGACY_ANCHORED_RESULTS_20261107.md`。
- 旧 A + EMA 全量 refit **已中断，当前未运行**。2026-11-07 16:33（北京时间）按进程命令行核查无对应训练进程，日志最后进度为 15:57 的 epoch 2 / batch 900；第一轮结束的 `train/resume.pt` 仍在，806,443,075 字节。旧 `run.json` 的 `training` 为残留状态，不代表仍在训练。原因未确认，本轮未自动恢复该任务。

当前研究结论：这次有界训练没有证明改善旧完整融合；仍以用户已反馈的旧融合线上 78.53 为基准。旧配方曾用官方 valid 选优，因此本次比较和区间不构成无偏泛化证明；候选没有线上分。

---

# 历史记录：A + EMA 线上回退，旧融合保留为已反馈最高分（2026-11-07）

用户反馈 `submission_test_1414_rad_dino_A_ema_20261107.csv` 线上为 **77.34**，低于旧融合 `submission_test_1414_best_20261105.csv` 的 **78.53**，下降 **1.19 分**。两项线上分数来自用户反馈，未直接访问评分后台。当前不能继续把 A + EMA 称为优于旧融合的提交模型。

- 用户附件与原 A + EMA 导出 CSV 的 SHA-256 完全一致。两版 CSV 的 1,414 次检查 / 14,140 行、类别覆盖及概率范围均通过重新检查；A + EMA 的全部概率按患者、检查、类别与保存的预测矩阵精确一致。该检查排除了文件替换及矩阵到 CSV 的导出差异，不等于完成模型或标签映射的全部审计。
- 旧融合的本地 0.8203251713 来自参与选优的官方 valid（859 次检查、228 位患者）；A + EMA 的 0.8270335041 来自训练集内预留患者 audit（3,657 次检查、2,211 位患者）。两者不能直接排名。新实验报告明确仅比较新 A/B 训练方案及 EMA，没有证明优于旧融合。
- 目前已反馈的两份提交以旧融合 78.53 为较高分基准。线上下降的具体原因尚未确定，不能仅凭这两项总分归因到 EMA、学习率、某个类别或过拟合。
- 当时的三轮全量 refit 是未验证研究候选，没有独立验证分或线上分，也不继承旧 audit 分数；其后中断状态及本轮新训练结果见文件顶部。
- 本次证据：`outputs/submission-gap-audit-20261107/online-regression-7734.json`；历史 78.53 与本次 77.34 已分别记入 `diagnostic.json`。

---

# 历史记录：A + EMA 全量重训已启动（2026-11-07）

北京时间 2026-11-07 13:44:50 曾启动固定三轮全量重训，后已中断，详情见文件顶部。数据为 `train.csv` 的 30,344 张图、18,513 次检查、11,055 位患者；方案来自已完成的患者分组三折实验，选择 A + EMA。`outputs/rad-dino-A-ema-fulltrain-20261107/run.json` 和 `train/progress.json` 保留中断前状态，需结合实际进程核查，不表示任务仍在运行。

- 原开发数据模型的三折平均 Macro-AUC 为 0.8171091466，独立 audit Macro-AUC 为 0.8270335041（2,211 位患者）；六模型学习融合的 audit 为 0.8259501845，未替换 A + EMA。它们与下方旧官方 valid 指标使用不同人群，不能直接按数值排名。
- 本轮从固定的公开 `yaLLMan/rad-dino-chexpert` 初始化训练，沿用已选定参数，改用 workers=0 降低 Windows 内存压力，保存每轮恢复点。启动前 32 张真实图 GPU 试跑及本轮 9 项训练/流程检查已通过。
- 本轮包含原 audit 患者，属于全量提交模型重训，没有新的独立验证分数；旧 audit 分数不会写成新模型实测成绩。训练不读取官方 valid 或 test 图像。
- 训练完成后，独立进程重载 EMA，预测 2,218 张 test 图，按检查平均得到 1,414 次检查的十类概率，校验后自动生成 `outputs/submission_test_1414_rad_dino_A_ema_fulltrain_20261107.csv`（14,140 行）。当前尚未生成该文件。
- 上一版可用预测：`outputs/submission_test_1414_rad_dino_A_ema_20261107.csv`；SHA-256 为 `97de8c07a2fe806b8d7535c7a483c2157f35917844a9c23057fa5086fe193870`。上一版和更早预测均保留。
- 训练日志：`outputs/rad-dino-A-ema-fulltrain-20261107/logs/training.log`。固定方案与指标解释见 `docs/FULL_REFIT_PLAN_20261107.md`。本轮未提交或推送代码。

---

# 历史记录：最高分方案增量验证已完成（2026-11-05）

当前已验证最高十类 Macro-AUC 为 **0.8203251712650804**，在原最高分 **0.8193622646357342** 上提升 **0.0009629066293461719**。覆盖完整 valid 的 1,422 张图、859 次检查、228 名患者，10 类 AUC 均有定义；9 类提高、1 类持平。固定阈值 0.5 的 Macro-F1 为 0.43739500546651655。

- 本轮直接以旧最高分检查级概率为固定基准，评估最佳 Rad-DINO 的原图/翻转图均值、RussellALA ConvNeXt，以及两个新 MedMAE 种子/分辨率候选。系数网格固定为 0、0.25、0.5、0.75、1，每类保留原方案作为选项。
- 新 MedMAE 768/seed1337 在最终配方的 Consolidation、Edema 中被采用；448/seed7 未进入最终配方。Rad-DINO TTA 的完整十类单模型 AUC 为 0.807447133833848。
- 主要证据：`outputs/best-ensemble-improvement-20261105/report.json`、`recipe.json`、`comparison.json`、`verification.json`；公开摘要与重现方法见 [医学模型验证记录](docs/MEDICAL_RESULTS_20261105.md)。原最高分目录 `outputs/ensemble-valid/` 保持不变。
- 500 次患者配对 bootstrap 的 AUC 差值 95% 区间为 [-0.0023959544, 0.0043975543]，跨零。组合选择的患者分组三折诊断 AUC 为 0.8058196352，未确认稳定收益；固定基准先前已使用完整 valid 选优，该诊断同样不是无偏测试。
- 训练参数仅由 train 学习，当前实验不读取 test 图像。外部预训练与本地匿名数据是否重叠仍未知；医学预训练路线与原自然图像预训练比赛基线分开记录。
- 本轮完整测试 142 项通过、无跳过。使用同一软件环境重放固定配方并独立重算指标，核对原方案哈希和输出 CSV。
- 旧 ConvNeXt 日志中的 `running` 是历史残留状态；本轮检查时其记录 PID 已不存在，不将其当作仍在运行或已完成的更高分实验。

---

# 历史记录：ConvNeXt-Tiny 384 正式实验启动（2026-11-04）

本机本轮实验于北京时间 2026-11-04 20:19:10 启动，当前进入第一轮训练。完整状态以 `outputs/convnext384-experiment-20261104-201910/experiment.json` 为准；尚未完成 20 轮，不能据此声明优于当前基线。

启动核验已观察到第一轮 **300/7,586 步、75次优化器更新**，累计端到端速度 **7.93张/秒**，错误日志为空。该速度包含首次原图解码和缓存写入，不用于外推后续缓存命中速度；核验快照：`outputs/convnext384-experiment-20261104-201910/startup-verification.json`，其中 `startup_verified=true`。实际后续进度以训练日志为准。

- 固定方案：ConvNeXt-Tiny ImageNet-1K，384 输入，20 轮，物理 batch=4、累积4次、有效 batch=16，AdamW lr=1e-4、weight_decay=0.01，两轮预热和余弦衰减，EMA=0.999。
- 数据：30,344 张 train 图训练，1,422 张 valid 图选模；无 train/valid 患者交叉，不读取 test 像素。无损输入缓存位于 `.cache/letterbox384`，原始数据不变。
- 当前比较基线：原 ResNet18 部署权重完整 valid 重跑 Macro-AUC **0.7806561629989964**、Macro-F1@0.5 **0.4184504656096208**。本轮统一使用 `outputs/dataset-verification-20261104-180020/valid.csv` 对比，避免混用历史运行的数值。
- 已通过：全套124项测试（无跳过）、CUDA正反向、CUDA三轮精确断点恢复、真实384小样本训练/独立预测、缓存开关前后完整训练状态精确一致。证据入口见 `AGGRESSIVE_OPTIMIZATION_PLAN.md`。
- 实际配置：`outputs/convnext384-experiment-20261104-201910/training/run.json`；训练日志：`outputs/convnext384-experiment-20261104-201910/logs/train.stdout.log`；错误日志同目录 `train.stderr.log`。
- 22份源码快照及SHA在实验目录 `source/`；配置与源文件哈希交叉核验通过。训练结束后执行器自动完成全量valid独立推理、500次患者配对bootstrap对比、推理文件清单和本机计时；本机计时不能代替官方T4认证。
- 本机环境：Python 3.12.10、torch 2.10.0+cu128、torchvision 0.25.0+cu128，RTX4060 Laptop 8GB；完整依赖版本见 `outputs/aggressive-environment-20261104.txt`。
- 本轮由 Luna 子模型协助实现训练和骨干支持。原始数据和原ResNet18模型保留；未提交代码、推送仓库或提交比赛。

---

# 历史记录：优化实验01完成并通过本地核验（2026-11-04）

以下保留原项目的 ResNet18 实验与当时环境记录。其“正在训练”过程文字属于历史追溯；当前 ConvNeXt 实验以本文件顶部及新的 `experiment.json` 为准。

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
