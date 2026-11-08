# 最高分方案的增量实验（2026-11-05）

本轮从已验证的最高分方案继续优化。所有完整任务分数均来自相同的 valid：1,422 张图、859 次检查、228 名患者、十类输出。模型参数只用 train 学习；本轮没有读取 test 图像。公开模型原始基准分数不作为本地实验成绩。

## 完整十类结果

| 方案 | 检查级 Macro-AUC | Macro-F1@0.5 |
| --- | ---: | ---: |
| 已重跑 ResNet18 基线 | 0.7806561630 | 0.4184504656 |
| 本地微调 Rad-DINO 最佳第3轮，独立重载 | 0.8032069539 | 见本地报告 |
| 同一 Rad-DINO，原图/水平翻转图概率均值 | 0.8074471338 | 0.4422656777 |
| 原最高分逐类融合（固定基准） | 0.8193622646 | 0.4324272014 |
| **本轮增量逐类融合** | **0.8203251713** | **0.4373950055** |

新方案相对原最高分的 AUC 增量为 **0.0009629066**，即约 **0.0963 个百分点**。它是本轮完成的实验中最高的验证分数。原最高分目录和检查点保留，新增输出写入独立目录。

## 本轮固定配方

令 `B` 为原 0.8193622646 方案的检查级概率，`T` 为本地最佳 Rad-DINO 的原图/翻转图平均概率，`M` 为公开 `mamounyosef/chest-xray-bench/medmae_vitb_nih_B_768_s2_seed1337` 的五类输出，`C` 为 `RussellALA/convnext-chexpert-seed1` 的五类输出。每次检查的图像先做概率算术平均，再按下表计算。

| 类别 | 原 AUC | 新 AUC | 固定计算 |
| --- | ---: | ---: | --- |
| Enlarged Cardiomediastinum | 0.7071856287 | 0.7093812375 | 0.50 T + 0.50 B |
| Pneumothorax | 0.8836719932 | 0.8836719932 | B |
| Consolidation | 0.8536017969 | 0.8546981122 | 0.25 M + 0.75 B |
| Pneumonia | 0.7356101733 | 0.7357403138 | 0.25 T + 0.75 B |
| Edema | 0.8929990650 | 0.8937353904 | 0.50 M + 0.50 B |
| Cardiomegaly | 0.7945640108 | 0.7970989280 | 0.25 T + 0.75 B |
| Atelectasis | 0.7906922467 | 0.7914624469 | 0.25 T + 0.75 B |
| Lung Opacity | 0.7364630877 | 0.7371296095 | 0.50 T + 0.50 B |
| Pleural Effusion | 0.9208488506 | 0.9220110440 | 0.50 C + 0.50 B |
| No Finding | 0.8779857935 | 0.8783226371 | 0.25 T + 0.75 B |

候选系数网格预先固定为 `(0, 0.25, 0.5, 0.75, 1)`。每类都保留 `B`，同分优先原方案。每个候选均与固定原方案比较，没有将新增候选依次叠加搜索。原方案的逐类配方、最终系数、源版本和预测文件指纹保存在 [固定配方 JSON](best_ensemble_recipe_20261105.json)。

本地 Rad-DINO 的已有训练方案：上游14类分类头按标签名复制十类；前十个 DINO block 冻结，仅训练最后两个 block、layernorm 和分类头；AdamW 主干 lr=1e-5、分类头 lr=1e-3、weight decay=1e-4，三轮余弦衰减到1e-6，物理 batch=2、累积8次、seed=2026，AMP。输入保持上游518处理器、mean=0.5307、std=0.2583。最佳为第3轮。本轮未继续训练这个检查点，只增加了翻转推理和公开候选评估。这些是实际运行参数，不代表全局最优参数。

## 新公开候选的完整本地评估

以下模型仅提供 Atelectasis、Cardiomegaly、Consolidation、Edema、Pleural Effusion 五类，表中第一列指标仅为这五类平均；完整十类固定融合中，未覆盖类别保留原方案。

| 候选 | 五类 Macro-AUC | 与原最高分固定0.5融合后的十类 AUC | 是否进入最终配方 |
| --- | ---: | ---: | --- |
| `medmae_vitb_nih_B_768_s2_seed1337` | 0.8234662194 | 0.8178060112 | 是，Consolidation/Edema |
| `medmae_vitb_nih_B_448_s1_seed7` | 0.8255487154 | 0.8172119602 | 否 |
| `RussellALA/convnext-chexpert-seed1` | 0.8189414387 | 0.8169671556 | 是，Pleural Effusion |

这两种新 MedMAE 的配置和权重固定到同一上游版本，448候选在该版本没有 `thresholds.json`；AUC 计算使用连续概率，不依赖该文件。没有把它的五类平均分与完整十类最高分直接比较。

## 稳定性与解释范围

- 500 次患者配对 bootstrap，AUC 增量95%描述区间 **[-0.0023959544, 0.0043975543]**，跨零；F1增量区间也跨零。因此不能将小幅验证提升表述为已证明稳定收益。
- 按患者分组三折，在另外两折选择候选与系数后生成留出预测，组合选择诊断 AUC 为 **0.8058196352**。它没有确认这次逐类选择可以稳定迁移。
- 原方案本身已在全部 valid 上选优，当前 bootstrap 和三折诊断均未消除其先前选模偏差；这些结果不是独立测试成绩。
- 匿名本地数据与 CheXpert/其他外部预训练数据是否重叠未知，无法据此声明独立泛化。公开来源、许可证声明、固定版本和完整SHA-256见 [模型清单](public_model_manifest.json)。保留上游署名与来源。
- 本轮属于外部医学预训练研究路线。多模型融合的大小、延迟和比赛材料条件没有在本轮重新验收；不能直接套用原 ResNet18 的计时与部署结论。CXformer 和另一个冻结特征迁移草稿没有完成本地全量比较，未进入成绩表或最终配方。

## 重现与本地证据

项目根目录使用 Python3.12 的 `.venv`。医学模型依赖独立于原基础环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-medical.txt
.\.venv\Scripts\python.exe scripts/fetch_public_models.py
```

下载脚本固定公开仓库 revision 并核验文件 SHA-256；`--repo` 可重复指定，只下载需要的仓库。已有本地最佳微调检查点后，在全新目录运行独立 TTA 验证：

```powershell
.\.venv\Scripts\python.exe scripts/eval_rad_dino_checkpoint.py --horizontal-flip --checkpoint outputs/rad-dino-finetune-20261105/best.pt --output outputs/rad-dino-tta-reproduction-new
.\.venv\Scripts\python.exe scripts/eval_chest_xray_bench.py --models medmae_vitb_nih_B_768_s2_seed1337 medmae_vitb_nih_B_448_s1_seed7 --output outputs/bench-reproduction-new
```

本轮融合选择命令如下。它从已保存的检查级预测工作，不重新训练；输入目录须包含 `study_ids.json`、`study_targets.npy`、`study_probabilities.npy` 和来源报告。不要覆盖已完成的输出目录。

```powershell
.\.venv\Scripts\python.exe scripts/improve_best_ensemble.py --baseline outputs/ensemble-valid --candidate outputs/rad-dino-best-tta-valid-20261105 --candidate outputs/russellala-convnext-valid --candidate outputs/chest-xray-bench-new-valid-20261105/medmae_vitb_nih_B_768_s2_seed1337 --candidate outputs/chest-xray-bench-new-valid-20261105/medmae_vitb_nih_B_448_s1_seed7 --data-root 'G:\cxr_first(1)\cxr_first' --output outputs/ensemble-reproduction-new
```

本机实际证据目录为 `outputs/best-ensemble-improvement-20261105/`：

| 文件 | 用途 |
| --- | --- |
| `report.json` | 十类最终指标、固定0.5对照、逐候选和组合三折诊断 |
| `recipe.json` | 固定逐类系数、原基准配方、候选版本和文件指纹 |
| `comparison.json` | 对原最高分的500次患者配对bootstrap |
| `verification.json` | 未重新选参的独立配方重放、官方valid对齐、指标复算、原方案文件哈希 |
| `valid.csv` | 完整859次检查、8590行的本地验证输出 |

原始数据、模型权重、患者级预测及这些本地输出不加入Git。公开仓库只包含代码、聚合实验结论、配方和公开来源元数据。

验证命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_publish.py
git diff --check
```

本轮完整单元测试142项通过、无跳过；另由指定的 `gpt-6-luna` 子代理完成增量选择器的只读检查。独立复算同时重放原方案与新配方；没有通过重选系数来核验已保存结果。
