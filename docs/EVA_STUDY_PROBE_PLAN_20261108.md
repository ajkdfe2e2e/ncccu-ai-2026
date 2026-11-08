# EVA-X study feature experiment — fixed protocol, 2026-11-08

## Execution boundary

This protocol prepares the next bounded experiment. The CXR-LT final-stage experiment in `outputs/cxrlt-stage-finetune-20261108` has priority over GPU and system memory. Do not register an immutable EVA-X plan, launch extraction or perform real-data fitting while that experiment is running. Code preparation and synthetic tests may proceed separately.

The named incumbent for this protocol is `outputs/cxrlt-supervised-study-probe-20261108`: train patient OOF Macro-AUC 0.8270384304066101 and official valid Macro-AUC 0.8335505264258105. Before plan registration, require the stage experiment to have terminal `status=complete` and `candidate_promoted=false`; lock its final `experiment.json`, `plan.json`, `plan_lock.json` and `artifacts.json`, plus its final recipe/model hashes if present. The execution/resume entry must recheck these locks and the terminal decision. A missing, failed, running or promoted stage result blocks this particular protocol. If the stage experiment promotes another model, revise the comparison protocol to include the newly promoted incumbent before registering a new EVA-X plan. A runner and this document alone are not evidence that the experiment has started. Actual execution begins only after the prerequisite passes and a new immutable plan, its input/source hashes and an execution snapshot are registered.

## Hypothesis and fixed candidates

The supervised EVA-X transformer may provide useful information beyond the frozen supervised ConvNeXtV2 representation. Test exactly two complete ten-class heads:

1. `eva`: 389 EVA-X image features, averaged over all images of each study.
2. `joint`: the existing 1,054 CXR-LT image features followed by the 389 EVA-X image features, averaged over the identical images of each study; total width 1,443.

Use ordinary BCE with L2=0.01, max_iter=200, seed=20261108 and class_balance=False for both. Standardize each feature column using the fitting patients alone, including the existing constant-column rule. Every fold fits a fresh head. No full-train head initializes a CV model. Each study contributes one training example. Keep both image encoders frozen and in eval mode. Do not add threshold search, per-class choices, blend ratios, architecture variants or regularization search to this experiment.

## Public model and feature contract

- Model: `MapleF/eva_x`, revision `35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee`.
- Weight: `eva_x_small_patch16_merged520k_mim_chexpert_ft.pth`; 260,339,517 bytes; SHA256 `1a5b63570ca4094c1118018e44d3012a95b904132942a8edb2fc0e5dc8aa0e97`.
- Author source: `hustvl/EVA-X@eedd642c85339b25682ceb574b644cb9afe88e62`; local `eva_x.py` SHA256 `5c21eb02cd977c35d4b8cf2984081b88b73db49953c1b4c2235f30df29df8219`.
- Use the verified `cxr/eva_x_transfer.py` adapter and its exact restoration of all 26 stored RoPE buffers. No permissive learned-weight loading.
- RGB, shorter edge 256 with bicubic resize, center crop 224, ToTensor, per-channel mean 0.49185243 and std 0.28509309.
- Features: normalized mean-pooled patch representation of width 384 plus five original raw classification logits. These five outputs do not replace the required local ten-class head.
- CUDA extraction uses AMP float16; stored features are finite float32. Fix extraction batch size **16** and preprocessing workers **2**, registered before real-data extraction. The synthetic-only GPU check at 2026-11-08 12:23:25 Asia/Shanghai produced finite 16x389 features, repeated exactly (max error 0), and peaked at 142.21 MiB allocated / 170 MiB reserved; evidence is `outputs/eva-x-preparation-20261108/gpu_engineering_check.json`. These settings were selected without real-data inference, fitting or AUC.
- Preserve the pinned model card, author source, license files and `provenance.json`. Engineering evidence is in `outputs/eva-x-preparation-20261108/cpu_engineering_check.json`; it establishes implementation parity, not a performance gain.

## Inputs, identity and recovery

Use the user's train/valid/test manifests and the same four patient groups `(0, 1, 2, "audit")` as the incumbent. Read data, partition and legacy baseline paths from the incumbent's registered plan. Lock the manifests, partition, public model/source/provenance, source experiment plan and selected recipe/head, source OOF targets/IDs/probabilities, both baseline valid predictions, all reused feature caches and the new runner's transitive source dependencies. Copy execution code into an immutable source snapshot and register the new plan's SHA256 before running it.

The 1,054D cache must be complete, float32 and finite, with its complete matrix hash matching its registration. Compare every ordered image identity (DICOM ID, resolved path, size and mtime), not just row count. Reject duplicate DICOM IDs, inconsistent study labels/subjects and changed images. Joint inputs require exact row identity between the two caches before concatenation. Study IDs must match the incumbent OOF `[[subject_id, study_id], ...]` exactly, and OOF targets must be elementwise equal.

Use a distinct EVA-X cache directory, with an additional binding to the frozen plan and encoder identity; the generic feature cache's image identity alone does not identify its encoder. Resume from committed rows, recompute uncommitted work, verify completed matrix hashes and refuse changed cache bindings. Save each fitted fold head with its exact fitting-fold set, plan hash, feature identity and head hash. Resume only a matching fit and verify saved-head prediction replay. Do not modify any previous experiment, weight, data or CSV.

## Patient OOF selection

Fit each candidate on three patient groups and predict the remaining group, for all four held groups. Explicitly reject any patient intersection between fitting and held studies. There are at most eight CV head fits.

For each candidate, require finite predictions covering all training studies and all ten defined AUCs overall and within each held group. A candidate is eligible only if its overall train OOF Macro-AUC exceeds both the named incumbent and legacy fusion by at least 0.001, and at least three of four held groups improve over the named incumbent. Select the eligible candidate with the highest overall OOF Macro-AUC; break an exact tie in favor of `eva`. If none qualifies, stop without fitting a full-train head or evaluating official valid/test.

These OOF results select among two fixed candidates. They are not a nested unbiased estimate, and previous reuse of these patient groups introduces additional selection history.

## One fixed official validation gate

Fit the selected candidate once on all training studies, then save/reload its head and freeze its feature recipe and hashes before official valid inference. Compare this one candidate against both the named incumbent and legacy fusion on the identical official valid studies.

For each baseline, require all ten AUCs defined, candidate Macro-AUC gain at least 0.001, and a paired patient bootstrap with 1,000 replicates and seed 20261108 whose descriptive 95% percentile difference interval has lower bound strictly above zero. At least 950 replicates must have all ten AUCs defined. Keep the thresholds fixed. A failure against either baseline retains the incumbent; do not revisit the other candidate or tune settings on official valid.

The official valid set has historical selection reuse. Bootstrap intervals are descriptive and do not eliminate that reuse or establish an online performance guarantee. Public pretraining overlap with the anonymized local data remains unknown.

## Test export and reporting

Only a candidate passing both gates may extract its required test features and export a new test submission. Confirm test manifest coverage, study/class order, finite probabilities in [0,1], expected 1,414 studies / 14,140 prediction rows for this dataset, complete ten-class coverage and exact saved-head-to-CSV replay. Register the CSV SHA256 and report online_score=null until the user supplies an actual score.

Report engineering checks, train OOF, official valid and online scores as distinct evidence. Keep the existing highest-AUC CSV and the user-reported online best (old fusion, 78.53) intact. This protocol does not authorize a competition submission, git commit or push.
