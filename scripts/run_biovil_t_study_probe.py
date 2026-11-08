"""Run the fixed BioViL-T incremental frozen-feature experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.safety import safe_output
from cxr.study_probe import fit_linear_probe, pool_study_features, predict_linear_probe
from scripts.evaluate_frozen_study_probe import FOLDS, _fold_memberships, _load_head, _save_head
from scripts.run_chexficient_study_probe import (
    now, read_json, sha256, atomic_json, _json_hash, _array_hash,
    _validate_matrix, require_exact_image_identity, _identity_rows,
    _split_inputs, _materialize_deferred_cxrlt, _validate_legacy_targets,
    _study_key_rows, _validate_split_disjointness, _validate_probabilities,
    candidate_gate, _artifact_manifest, _gate_then_export,
)
from scripts.run_bottleneck_study_probe import legacy_oof, validation_gate

SEED = 20261108
MAX_ITER = 200
L2 = 0.01
CLASS_BALANCE = False
CANDIDATES = ("joint", "triple")
FEATURE_WIDTHS = {"joint": 1438, "triple": 1950}
CHEXFICIENT_REFERENCE_OOF = 0.8315443007087662
DEFAULT_SOURCE = ROOT / "outputs/cxrlt-supervised-study-probe-20261108"
DEFAULT_OUTPUT = ROOT / "outputs/biovil-t-study-probe-20261108"


class FoldHeadCache:
    """Register exactly replayable patient-held-out heads bound to the plan/features."""

    def __init__(self, folder, features, targets, folds, subjects,
                 plan_sha256, feature_identity):
        self.folder = Path(folder)
        self.x, self.y = features, targets
        self.folds, self.subjects = np.asarray(folds, dtype=object), np.asarray(subjects)
        self.plan_sha256, self.feature_identity = plan_sha256, feature_identity

    def predict(self, candidate, held_fold):
        if candidate not in CANDIDATES or held_fold not in FOLDS:
            raise ValueError("unsupported candidate or held fold")
        held = self.folds == held_fold
        training_folds = tuple(fold for fold in FOLDS if fold != held_fold)
        train = np.isin(self.folds, np.asarray(training_folds, dtype=object))
        held_indices = np.flatnonzero(held)
        if not held_indices.size or not train.any():
            raise ValueError("fold requires nonempty training and held rows")
        if np.any(train[held_indices]) or set(self.subjects[train]) & set(self.subjects[held]):
            raise ValueError("patient leakage in requested fold fit")
        identity = {
            "plan_sha256": self.plan_sha256,
            "feature_identity": self.feature_identity,
            "candidate": candidate,
            "width": FEATURE_WIDTHS[candidate],
            "training_folds": list(training_folds),
            "held_fold": held_fold,
            "training_patients": sorted(set(map(str, self.subjects[train]))),
            "l2": L2, "max_iter": MAX_ITER, "seed": SEED,
            "class_balance": CLASS_BALANCE,
        }
        folder = self.folder / candidate / str(held_fold)
        head_path, metadata_path = folder / "head.npz", folder / "fit.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if (metadata.get("identity") != identity or not head_path.is_file()
                    or sha256(head_path) != metadata.get("head_sha256")):
                raise ValueError("cached fold head binding or hash mismatch")
            payload = _load_head(head_path)
        else:
            if head_path.exists():
                head_path.unlink()
            payload = fit_linear_probe(
                self.x[train], self.y[train], l2=L2, max_iter=MAX_ITER,
                seed=SEED, class_balance=CLASS_BALANCE,
            )
            folder.mkdir(parents=True, exist_ok=True)
            _save_head(head_path, payload)
            reloaded = _load_head(head_path)
            expected = predict_linear_probe(payload, self.x[held])
            replay = predict_linear_probe(reloaded, self.x[held])
            if not np.array_equal(expected, replay):
                raise ValueError("saved fold head does not replay exactly")
            metadata = {"identity": identity, "head_sha256": sha256(head_path),
                        "held_prediction_sha256": _array_hash(replay)}
            atomic_json(metadata_path, metadata)
        probabilities = predict_linear_probe(payload, self.x[held])
        if metadata.get("held_prediction_sha256") != _array_hash(probabilities):
            raise ValueError("cached fold head prediction replay mismatch")
        if not np.isfinite(probabilities).all():
            raise ValueError("fold head produced non-finite probabilities")
        return probabilities


def _fit_final_head(output, plan, candidate, features, targets, feature_identity):
    head_path, recipe_path = output / "final_head.npz", output / "recipe.json"
    plan_sha = sha256(output / "plan.json")
    if recipe_path.exists():
        recipe = read_json(recipe_path)
        if (recipe.get("plan_sha256") != plan_sha
                or recipe.get("feature_identity") != feature_identity
                or recipe.get("candidate") != candidate
                or not head_path.is_file()
                or sha256(head_path) != recipe.get("head_sha256")):
            raise ValueError("saved final head is bound to another plan, candidate, or feature cache")
        payload = _load_head(head_path)
    else:
        if head_path.exists():
            head_path.unlink()
        payload = fit_linear_probe(
            features, targets, l2=L2, max_iter=MAX_ITER,
            seed=SEED, class_balance=CLASS_BALANCE,
        )
        _save_head(head_path, payload)
        reloaded = _load_head(head_path)
        if not np.array_equal(predict_linear_probe(payload, features),
                              predict_linear_probe(reloaded, features)):
            raise ValueError("saved final head does not replay exactly")
        recipe = {
            "candidate": candidate,
            "width": FEATURE_WIDTHS[candidate],
            "plan_sha256": plan_sha,
            "feature_identity": feature_identity,
            "source_plan_sha256": plan["incumbent_plan_sha256"],
            "l2": L2, "max_iter": MAX_ITER, "seed": SEED,
            "class_balance": CLASS_BALANCE,
            "head_file": head_path.name,
            "head_sha256": sha256(head_path),
        }
        atomic_json(recipe_path, recipe)
    return recipe, payload


def select_candidate(rows):
    eligible = [row for row in rows if row["gate"]["passed"]]
    order = {name: index for index, name in enumerate(CANDIDATES)}
    return min(eligible, key=lambda row: (
        -row["gate"]["candidate"]["macro_auc"], order[row["candidate"]],
    )) if eligible else None


def pool_candidates(records, views, cxrlt_images, biovil_images, identities,
                    biovil_ids, chexficient_images=None, chexficient_ids=None):
    """Pool each encoder separately, then concatenate in the registered order."""
    require_exact_image_identity(identities, biovil_ids)
    _validate_matrix(cxrlt_images, len(records), 1054, "CXR-LT image features")
    _validate_matrix(biovil_images, len(records), 384, "BioViL-T image features")
    studies, cxrlt, targets = pool_study_features(records, cxrlt_images, views, "mean")

    def pooled(values):
        other_studies, features, other_targets = pool_study_features(records, values, views, "mean")
        if _study_key_rows(other_studies) != _study_key_rows(studies):
            raise ValueError("encoders produced different ordered study IDs")
        if ((targets is None) != (other_targets is None)
                or targets is not None and not np.array_equal(targets, other_targets)):
            raise ValueError("encoders produced different study targets")
        return features

    biovil = pooled(biovil_images)
    candidates = {"joint": np.concatenate((cxrlt, biovil), axis=1)}
    if chexficient_images is not None:
        if chexficient_ids is None:
            raise ValueError("CheXficient features require registered image identities")
        require_exact_image_identity(identities, chexficient_ids)
        _validate_matrix(chexficient_images, len(records), 512, "CheXficient image features")
        candidates["triple"] = np.concatenate((cxrlt, pooled(chexficient_images), biovil), axis=1)
    elif chexficient_ids is not None:
        raise ValueError("CheXficient identities were supplied without their features")
    for name, features in candidates.items():
        if features.shape != (len(studies), FEATURE_WIDTHS[name]) or not np.isfinite(features).all():
            raise ValueError(f"invalid pooled {name} features")
    return studies, candidates, targets


def _feature_identity(candidate, cxrlt_sha, biovil_sha, chexficient_sha, study_keys):
    return _json_hash({
        "candidate": candidate, "width": FEATURE_WIDTHS[candidate],
        "order": ["cxrlt", "biovil"] if candidate == "joint" else ["cxrlt", "chexficient", "biovil"],
        "cxrlt_sha256": cxrlt_sha, "biovil_sha256": biovil_sha,
        "chexficient_sha256": chexficient_sha if candidate == "triple" else None,
        "study_keys_sha256": _json_hash(study_keys),
    })


def _binding(plan, split, encoder, identities):
    if split not in ("train", "valid", "test") or encoder not in ("biovil", "chexficient"):
        raise ValueError("unsupported feature-cache split or encoder")
    if encoder == "chexficient" and split != "test":
        raise ValueError("existing CheXficient train/valid features must be reused")
    if encoder == "biovil":
        from cxr.biovil_t_transfer import MODEL_REVISION, WEIGHTS_SHA256, FEATURE_POOLING
        width, size, batch = 384, 448, 2
        adapter = "cxr/biovil_t_transfer.py"
        manifest = "docs/biovil_t_source_manifest_20261108.json"
    else:
        from cxr.chexficient_transfer import MODEL_REVISION, FEATURE_POOLING
        from scripts.run_chexficient_study_probe import CHEXFICIENT_WEIGHT_SHA256 as WEIGHTS_SHA256
        width, size, batch = 512, 378, 4
        adapter = "cxr/chexficient_transfer.py"
        manifest = "docs/chexficient_source_manifest_20261108.json"
    return {
        "plan_sha256": plan["plan_sha256"], "model_revision": MODEL_REVISION,
        "weight_sha256": WEIGHTS_SHA256, "adapter_sha256": plan["source_locks"][adapter],
        "manifest_sha256": plan["source_locks"][manifest],
        "image_identity_sha256": _json_hash(_identity_rows(identities)),
        "feature_size": width, "input_size": size, "feature_pooling": FEATURE_POOLING,
        "batch_size": batch, "preprocess_workers": 2, "device": plan["device"],
    }


def _extract_split(plan, records, identities, split, encoder, bundle):
    from scripts.run_frozen_study_probe import extract_features
    cache = Path(plan["output_dir"]) / f"{encoder}_features" / split
    cache.mkdir(parents=True, exist_ok=True)
    binding = _binding(plan, split, encoder, identities)
    path = cache / "binding.json"
    if path.exists():
        if read_json(path) != binding:
            raise ValueError(f"{encoder} {split} feature cache binding mismatch")
    else:
        if any(cache.iterdir()):
            raise ValueError("feature cache payload exists without a model binding")
        atomic_json(path, binding)
    model, processor, device = bundle
    values = extract_features(model, processor, records, cache,
                              batch_size=binding["batch_size"], device=device,
                              preprocess_workers=binding["preprocess_workers"])
    require_exact_image_identity(identities, read_json(cache / "input.json")["images"])
    progress = read_json(cache / "progress.json")
    matrix_sha = sha256(cache / "image_features.npy")
    if (progress.get("status") != "complete" or progress.get("completed") != len(records)
            or progress.get("sha256") != matrix_sha):
        raise ValueError("feature extraction is incomplete or its matrix changed")
    _validate_matrix(values, len(records), binding["feature_size"], f"{encoder} {split}")
    registration = {
        "plan_sha256": plan["plan_sha256"], "matrix_sha256": matrix_sha,
        **{f"{name}_sha256": sha256(cache / f"{name}.json")
           for name in ("binding", "input", "progress")},
    }
    registered = cache / "registration.json"
    if registered.exists() and read_json(registered) != registration:
        raise ValueError("completed feature-cache registration changed")
    atomic_json(registered, registration)
    return values


def _extract_encoder(plan, records, identities, split, encoder="biovil"):
    if encoder == "biovil":
        from cxr.biovil_t_transfer import load_encoder
        model_dir = plan["model_dir"]
    else:
        from cxr.chexficient_transfer import load_encoder
        model_dir = plan["chex_model_dir"]
    model, processor = load_encoder(model_dir, plan["device"])
    try:
        return _extract_split(plan, records, identities, split, encoder,
                              (model, processor, torch.device(plan["device"])))
    finally:
        del model, processor
        if torch.device(plan["device"]).type == "cuda":
            torch.cuda.empty_cache()


def run(output):
    from scripts.biovil_t_probe_inputs import load_registered_plan, verify_plan, _load_reused_chex
    from scripts.compare_runs import compare_arrays, load_submission
    from cxr.submission import validate_submission, write_submission

    output = Path(output).resolve()
    plan = load_registered_plan(output)
    if (tuple(plan.get("candidates", ())) != CANDIDATES
            or plan.get("feature_widths") != FEATURE_WIDTHS
            or plan.get("fit") != {"l2": L2, "max_iter": MAX_ITER, "seed": SEED,
                                   "class_balance": CLASS_BALANCE}
            or tuple(plan.get("folds", ())) != FOLDS
            or plan.get("batch_size") != 2 or plan.get("preprocess_workers") != 2):
        raise ValueError("registered plan differs from the fixed execution protocol")
    state = read_json(output / "experiment.json")
    torch.set_num_threads(4)
    if state.get("status") == "complete":
        for name, digest in read_json(output / "artifacts.json").items():
            if sha256(output / name) != digest:
                raise ValueError(f"completed artifact changed: {name}")
        return

    def stage(phase, **extra):
        state.update(status="running", phase=phase, updated_utc=now(), pid=os.getpid(), error=None, **extra)
        atomic_json(output / "experiment.json", state)

    def finish(decision, *, promoted=False, test_exported=False):
        verify_plan(output, read_json(output / "plan.json"))
        atomic_json(output / "artifacts.json", _artifact_manifest(output))
        state.update(status="complete", phase=None, updated_utc=now(), decision=decision,
                     candidate_promoted=promoted, test_csv_exported=test_exported, error=None)
        atomic_json(output / "experiment.json", state)

    try:
        stage("load_and_validate_train_cache")
        records, views, identities, cxr_images = _split_inputs(plan, "train")
        incumbent = Path(plan["incumbent_snapshot"])
        source_ids = read_json(incumbent / "oof_study_ids.json")
        targets = np.load(incumbent / "oof_targets.npy", allow_pickle=False)
        baseline = np.load(incumbent / "oof_mean_probabilities.npy", allow_pickle=False)
        chex_images = _load_reused_chex(plan, "train", identities)
        stage("extract_biovil_train")
        bio_images = _extract_encoder(plan, records, identities, "train")
        bio_cache = output / "biovil_features/train"
        bio_ids = read_json(bio_cache / "input.json")["images"]
        studies, features, pooled_targets = pool_candidates(
            records, views, cxr_images, bio_images, identities, bio_ids, chex_images, identities)
        if _study_key_rows(studies) != source_ids or not np.array_equal(pooled_targets, targets):
            raise ValueError("candidate pooled study IDs or labels differ from incumbent OOF")
        legacy_audit = _validate_legacy_targets(output, plan, studies, targets)
        _validate_probabilities(baseline, targets.shape, "incumbent OOF")
        legacy = legacy_oof({"legacy_oof": plan["legacy_oof"]}, studies)
        _validate_probabilities(legacy, targets.shape, "legacy OOF")
        folds = _fold_memberships(studies, read_json(plan["partition"])["subject_partition"])
        subjects = [str(row["subject_id"]) for row in studies]
        cxr_sha = sha256(incumbent / "train_features/image_features.npy")
        bio_sha = sha256(bio_cache / "image_features.npy")
        chex_sha = plan["reused_chex"]["train_registration"]["matrix_sha256"]
        del cxr_images, bio_images, chex_images
        rows, feature_ids = [], {}
        for candidate in CANDIDATES:
            feature_ids[candidate] = _feature_identity(candidate, cxr_sha, bio_sha, chex_sha, source_ids)
            fits = FoldHeadCache(output / "fits", features[candidate], targets, folds, subjects,
                                 plan["plan_sha256"], feature_ids[candidate])
            prediction = np.full_like(targets, np.nan, dtype=np.float64)
            for fold in FOLDS:
                stage("train_patient_grouped_heads", candidate=candidate, held_fold=fold)
                prediction[folds == fold] = fits.predict(candidate, fold)
            _validate_probabilities(prediction, targets.shape, f"{candidate} OOF")
            gate = candidate_gate(targets, baseline, legacy, prediction, folds)
            rows.append({"candidate": candidate, "gate": gate,
                         "feature_identity": feature_ids[candidate],
                         "delta_vs_completed_chex_joint_oof":
                         gate["candidate"]["macro_auc"] - CHEXFICIENT_REFERENCE_OOF})
            np.save(output / f"oof_{candidate}_probabilities.npy", prediction, allow_pickle=False)
            del fits, prediction
        atomic_json(output / "oof_study_ids.json", source_ids)
        np.save(output / "oof_targets.npy", targets, allow_pickle=False)
        selected = select_candidate(rows)
        atomic_json(output / "oof_report.json", {
            "candidates": rows, "legacy_target_audit": legacy_audit,
            "selected": None if selected is None else selected["candidate"],
            "selection": "highest eligible complete OOF Macro-AUC; exact tie prefers joint1438",
            "completed_chex_joint_oof_reference": CHEXFICIENT_REFERENCE_OOF,
            "eligibility": {"all_ten_classes": True, "gain_vs_incumbent_min": 0.001,
                            "gain_vs_legacy_min": 0.001, "positive_incumbent_groups_min": 3},
        })
        if selected is None:
            finish("retain incumbent; fixed training OOF gate failed")
            return
        candidate = selected["candidate"]
        state["selected_candidate"] = candidate
        stage("fit_and_register_full_train_head")
        recipe, head = _fit_final_head(output, plan, candidate, features[candidate], targets,
                                      feature_ids[candidate])
        recipe_sha, head_sha = sha256(output / "recipe.json"), sha256(output / "final_head.npz")
        del features, head

        def candidate_split(split):
            _materialize_deferred_cxrlt(plan, split)
            split_records, split_views, split_ids, split_cxr = _split_inputs(plan, split)
            _validate_split_disjointness(records, split_records, split)
            if split == "test":
                _validate_split_disjointness(valid_records, split_records, split)
            bio = _extract_encoder(plan, split_records, split_ids, split)
            bio_identity = read_json(output / f"biovil_features/{split}/input.json")["images"]
            chex = None
            if candidate == "triple":
                if split == "valid":
                    plan["chex_valid_unlocked"] = True
                chex = (_load_reused_chex(plan, "valid", split_ids) if split == "valid" else
                        _extract_encoder(plan, split_records, split_ids, "test", "chexficient"))
            result = pool_candidates(split_records, split_views, split_cxr, bio, split_ids, bio_identity,
                                     chex, split_ids if chex is not None else None)
            return split_records, result

        stage("extract_biovil_valid")
        valid_records, (valid_studies, valid_features, valid_targets) = candidate_split("valid")
        if valid_targets is None:
            raise ValueError("validation records must have labels")
        if sha256(output / "recipe.json") != recipe_sha or sha256(output / "final_head.npz") != head_sha:
            raise ValueError("registered final head or recipe changed before validation")
        valid_prediction = predict_linear_probe(_load_head(output / "final_head.npz"), valid_features[candidate])
        _validate_probabilities(valid_prediction, valid_targets.shape, "validation probabilities")
        current, _, _ = load_submission(plan["current_valid"], valid_studies)
        old, _, _ = load_submission(plan["legacy_valid"], valid_studies)
        valid_csv = output / "candidate_valid.csv"
        if not valid_csv.exists():
            write_submission(valid_csv, valid_studies, valid_prediction)
        replay, _, _ = load_submission(valid_csv, valid_studies)
        if not np.array_equal(valid_prediction, replay):
            raise ValueError("validation CSV does not replay exactly")
        comparisons = {name: compare_arrays(
            [row["subject_id"] for row in valid_studies], valid_targets, previous,
            valid_prediction, bootstrap=1000, seed=SEED)
            for name, previous in (("incumbent", current), ("legacy", old))}
        report = {"candidate": candidate, "comparisons": comparisons,
                  "promoted": validation_gate(comparisons), "recipe_sha256": recipe_sha,
                  "head_sha256": head_sha, "candidate_valid_csv_sha256": sha256(valid_csv),
                  "new_online_score": None, "bootstrap": {"replicates": 1000, "seed": SEED}}
        report_path = output / "validation_report.json"
        if report_path.exists() and read_json(report_path) != report:
            raise ValueError("registered validation result changed during resume")
        atomic_json(report_path, report)
        del valid_features
        if not validation_gate(comparisons):
            finish("retain incumbent; fixed official validation gate failed")
            return

        def export_test():
            stage("extract_test_and_export", test_access_started=True)
            test_records, (test_studies, test_features, test_targets) = candidate_split("test")
            _validate_split_disjointness(valid_records, test_records, "test")
            if test_targets is not None or len(test_studies) != 1414:
                raise ValueError("test labels or coverage violate the registered contract")
            prediction = predict_linear_probe(_load_head(output / "final_head.npz"), test_features[candidate])
            _validate_probabilities(prediction, (1414, 10), "test probabilities")
            csv_path = output / "submission_test_1414_biovil_t_study_20261108.csv"
            write_submission(csv_path, test_studies, prediction)
            verification = validate_submission(csv_path, test_studies)
            replay, _, _ = load_submission(csv_path, test_studies)
            if not np.array_equal(prediction, replay):
                raise ValueError("test CSV does not replay exactly")
            atomic_json(output / "submission_verification.json", {
                **verification, "sha256": sha256(csv_path), "online_score": None,
                "replay_max_error": 0.0, "recipe_sha256": recipe_sha, "head_sha256": head_sha})
            state["test_extracted"] = True

        exported = _gate_then_export(comparisons, export_test)
        finish("promote locally; both fixed validation comparisons passed",
               promoted=exported, test_exported=exported)
    except Exception as error:
        state.update(status="failed", updated_utc=now(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(output / "experiment.json", state)
        raise


def main(argv=None):
    from scripts.biovil_t_probe_inputs import prepare
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-experiment", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.run and args.resume:
        parser.error("choose either --run or --resume")
    output = safe_output(args.output)
    if args.resume:
        if not (output / "plan.json").is_file():
            parser.error("--resume requires a prepared directory")
    else:
        output = prepare(output, args.source_experiment)
    if not (args.run or args.resume):
        print({"status": "prepared", "output": str(output)}, flush=True)
        return
    snapshot = output / "source/scripts/run_biovil_t_study_probe.py"
    if Path(__file__).resolve() != snapshot.resolve():
        raise SystemExit(subprocess.call([sys.executable, "-u", str(snapshot),
                                         "--output", str(output), "--resume"]))
    run(output)


if __name__ == "__main__":
    main()
