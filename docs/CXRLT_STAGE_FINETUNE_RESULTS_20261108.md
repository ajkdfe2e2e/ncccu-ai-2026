# CXR-LT final-stage fine-tuning result — 2026-11-08

The fixed one-epoch final-stage candidate failed the train patient OOF gate. Retain `outputs/cxrlt-supervised-study-probe-20261108`; its official valid Macro-AUC remains **0.8335505264258105**. This experiment did not evaluate official valid or test, fit a final all-train model, or export a new submission.

## Result on the identical train patient OOF studies

All 18,513 studies / 11,055 patients are covered once, with all ten AUCs defined. The complete OOF score is computed from concatenated held predictions, not an average of the four fold AUCs.

| Evaluation | Incumbent | Final-stage candidate | Difference |
| --- | ---: | ---: | ---: |
| Complete patient OOF Macro-AUC | 0.8270384304066101 | 0.8169781872476143 | -0.010060243158995785 |
| Fold 0, 4,868 studies | 0.8234672273056935 | 0.8220596395029123 | -0.0014075878027812738 |
| Fold 1, 5,000 studies | 0.8270622606336108 | 0.8279154380279202 | +0.0008531773943094567 |
| Fold 2, 4,988 studies | 0.8264395363378480 | 0.8243647761873485 | -0.002074760150499455 |
| Audit group, 3,657 studies | 0.8336422909967315 | 0.8327069543289586 | -0.0009353366677729014 |

Only one of four held groups improved. The candidate also fell below the legacy fusion OOF score of 0.8182590866984774 by 0.0012808994508630711. It failed the preregistered overall gain and group consistency requirements. This result applies to the fixed tested recipe; it does not establish that every fine-tuning strategy would fail.

## Execution and verification

- Started at **2026-11-08 11:17:31 Asia/Shanghai**; completed at **12:22:01.818643**, `status=complete`, `candidate_promoted=false`, `test_csv_exported=false`, `error=null`.
- Frozen plan SHA256: `08b85902dc97fd88ac8a24f698fc20589b48d10ee2b41357baaabc314364de42`.
- Public checkpoint: `hieuphamha/cxrlt2026-task1-convnextv2@24bf8bdfd8d0262899167b030be16ab89f78ae20`; source and provenance preserved.
- Frozen stem and first three stages; trained stage 3 and the ten-class local head. Study-level mean pooling, one epoch per patient fold, effective batch 16; stage LR 1e-5 / weight decay .01, head LR 1e-4 / weight decay 0, ordinary BCE plus original L2 .01, seed 20261108. Each fold initialized from its corresponding training-patients-only incumbent head.
- Four folds completed 853, 845, 846 and 929 effective updates, **3,473 total, zero AMP skips**. Saved model hashes and zero train/held patient intersection were independently checked for every fold. Independent recomputation exactly agrees with the experiment's complete OOF report.
- The 30,344-image frozen prefix cache is retained on F:; complete matrix SHA256 `751811857ab428f6c5ef1e6b5f0f18ee922b84e0e4beb8071072544e97e5de1c`.
- Previously passed 47 targeted engineering tests and GPU smoke. This postrun check used the NumPy-only `outputs/check_cxrlt_stage_progress_20261108.py`; results are in `outputs/cxrlt-stage-partial-oof-check-20261108.json` with `complete_oof=true` and `partial_results_only=false`.
- Formal reports and artifact registration: `outputs/cxrlt-stage-finetune-20261108/oof_report.json`, `experiment.json`, `plan_lock.json`, `artifacts.json`.

## Retained output and limits

The highest official-valid-AUC test file is still `outputs/submission_test_1414_best_auc_083355_20261108.csv`, SHA256 `7590ef61172423348bcf9a0c9afdb578b5416396d41a1c9357b03b0df232ec68`. It covers 1,414 test studies and 14,140 class rows and has no reported online score. The user-reported online best remains the old fusion at 78.53. Train OOF, official valid and online scores describe different evaluation populations.

Historical patient-group selection and public-pretraining overlap remain limitations. No competition submission, git commit or push was performed. The continuing improvement goal stays active; the next fixed protocol is `docs/EVA_STUDY_PROBE_PLAN_20261108.md`, subject to its engineering and immutable-plan checks.
