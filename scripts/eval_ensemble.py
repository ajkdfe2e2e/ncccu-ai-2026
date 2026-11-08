"""Search validation-only, study-aligned ensembles of public and local models.

The per-class search uses valid labels and is intentionally reported as a
validation-selected result, not as unbiased generalization evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.metrics import _rank_auc, evaluate
from cxr.safety import fresh_run_directory
from cxr.submission import write_submission
from scripts.compare_runs import load_submission, load_validation_metadata

LOCAL_LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]
WEIGHT_GRID = np.linspace(0.0, 1.0, 51)


def _load(folder: Path):
    probabilities = np.load(folder / "study_probabilities.npy").astype(np.float64)
    targets = np.load(folder / "study_targets.npy").astype(np.float64)
    ids = json.loads((folder / "study_ids.json").read_text(encoding="utf-8"))
    keys = [(str(row[0]), str(row[1])) for row in ids]
    return probabilities, targets, keys


def _align(folder: Path, expected_keys, expected_targets):
    probabilities, targets, keys = _load(folder)
    positions = {key: i for i, key in enumerate(keys)}
    if len(positions) != len(keys) or set(positions) != set(expected_keys):
        raise ValueError(f"study IDs differ from official validation studies: {folder}")
    order = np.asarray([positions[key] for key in expected_keys])
    if targets.shape != expected_targets.shape or not np.array_equal(targets[order], expected_targets):
        raise ValueError(f"targets differ from official validation targets after ID alignment: {folder}")
    return probabilities[order]


def _rank_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for j in range(values.shape[1]):
        order = np.argsort(values[:, j], kind="mergesort")
        sorted_values = values[order, j]
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        ends = np.r_[starts[1:], len(values)]
        average_ranks = (starts + ends - 1) / 2.0
        result[order, j] = np.repeat(average_ranks, ends - starts) / max(1, len(values) - 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"G:\cxr_first(1)\cxr_first"))
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/dataset-verification-20261104-180020/valid.csv")
    parser.add_argument("--rad-folder", type=Path, default=ROOT / "outputs/rad-dino-finetuned-valid-independent")
    parser.add_argument("--densenet-folder", type=Path, default=ROOT / "outputs/itsomk-densenet121-valid")
    parser.add_argument("--bench-root", type=Path, default=ROOT / "outputs/chest-xray-bench-valid")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ensemble-valid")
    args = parser.parse_args()
    args.output = fresh_run_directory(args.output, args.data_root)

    studies, targets = load_validation_metadata(args.data_root / "valid.csv")
    expected_keys = [(str(row["subject_id"]), str(row["study_id"])) for row in studies]
    baseline, _, _ = load_submission(args.baseline, studies)
    sources = {"resnet18_baseline": baseline,
               "rad_dino_finetuned": _align(args.rad_folder, expected_keys, targets)}
    if (args.densenet_folder / "study_probabilities.npy").is_file():
        sources["densenet121_chexpert"] = _align(args.densenet_folder, expected_keys, targets)

    benchmarks = {}
    for probabilities_path in sorted(args.bench_root.glob("*/study_probabilities.npy")):
        folder = probabilities_path.parent
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        probabilities = _align(folder, expected_keys, targets)
        indices = [int(index) for index in report["local_label_indices"]]
        if probabilities.shape != (len(studies), len(indices)):
            raise ValueError(f"benchmark model output/label dimensions differ: {folder}")
        benchmarks[report["model"].split("/")[-1]] = {
            "probabilities": probabilities, "indices": indices,
            "mapped_labels": report["mapped_labels"],
        }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "valid_studies": len(studies), "selection": "validation-only; blend weights selected on valid labels",
        "models": {}, "global_pair_blends": {}, "benchmark_global_blends": {},
        "per_class_selection": [],
    }
    for name, probabilities in sources.items():
        summary["models"][name] = evaluate(targets, probabilities)

    best_name = max(sources, key=lambda name: summary["models"][name]["macro_auc"])
    best_probabilities = sources[best_name].copy()
    best_score = summary["models"][best_name]["macro_auc"]

    source_names = list(sources)
    for i, left_name in enumerate(source_names):
        for right_name in source_names[i + 1:]:
            left, right = sources[left_name], sources[right_name]
            trials = [(evaluate(targets, alpha * left + (1.0 - alpha) * right)["macro_auc"], alpha)
                      for alpha in WEIGHT_GRID]
            score, alpha = max(trials, key=lambda row: row[0])
            probabilities = alpha * left + (1.0 - alpha) * right
            key = f"{left_name}__{right_name}"
            summary["global_pair_blends"][key] = {
                "left": left_name, "right": right_name, "alpha_left": float(alpha),
                "macro_auc": float(score),
            }
            if score > best_score:
                best_name, best_probabilities, best_score = key, probabilities, score

    rad = sources["rad_dino_finetuned"]
    for name, payload in benchmarks.items():
        mapped = payload["indices"]
        external = payload["probabilities"]
        position = {index: col for col, index in enumerate(mapped)}
        trials = []
        for alpha in WEIGHT_GRID:
            candidate = rad.copy()
            candidate[:, mapped] = alpha * external + (1.0 - alpha) * rad[:, mapped]
            trials.append((evaluate(targets, candidate)["macro_auc"], float(alpha), candidate))
        score, alpha, candidate = max(trials, key=lambda row: row[0])
        summary["benchmark_global_blends"][name] = {
            "mapped_labels": payload["mapped_labels"], "alpha_external": alpha,
            "macro_auc": float(score),
        }
        if score > best_score:
            best_name, best_probabilities, best_score = f"rad_dino+{name}", candidate, score

    # Search each class among all full-model pairs and every mapped external head.
    selected = np.empty_like(rad)
    for class_index, label in enumerate(LOCAL_LABELS):
        choices = [(float(_rank_auc(targets[:, class_index], values[:, class_index])),
                    values[:, class_index], {"kind": "single_model", "model": name})
                   for name, values in sources.items()]
        for i, left_name in enumerate(source_names):
            for right_name in source_names[i + 1:]:
                left, right = sources[left_name][:, class_index], sources[right_name][:, class_index]
                trials = [(_rank_auc(targets[:, class_index], alpha * left + (1.0 - alpha) * right), alpha)
                          for alpha in WEIGHT_GRID]
                auc, alpha = max(trials, key=lambda row: row[0])
                choices.append((float(auc), alpha * left + (1.0 - alpha) * right,
                                {"kind": "full_model_blend", "left": left_name, "right": right_name,
                                 "alpha_left": float(alpha)}))
        for model_name, payload in benchmarks.items():
            if class_index not in payload["indices"]:
                continue
            external_column = payload["indices"].index(class_index)
            external = payload["probabilities"][:, external_column]
            for base_name, values in sources.items():
                base = values[:, class_index]
                trials = [(_rank_auc(targets[:, class_index], alpha * external + (1.0 - alpha) * base), alpha)
                          for alpha in WEIGHT_GRID]
                auc, alpha = max(trials, key=lambda row: row[0])
                choices.append((float(auc), alpha * external + (1.0 - alpha) * base,
                                {"kind": "external_blend", "external_model": model_name,
                                 "base_model": base_name, "alpha_external": float(alpha)}))
        auc, scores, choice = max(choices, key=lambda row: row[0])
        selected[:, class_index] = scores
        summary["per_class_selection"].append({"class": class_index + 1, "label": label,
                                                "auc": auc, **choice})

    classwise_metrics = evaluate(targets, selected)
    summary["per_class_blend"] = classwise_metrics
    summary["rank_average_full_models"] = evaluate(
        targets, np.mean([_rank_normalize(value) for value in sources.values()], axis=0)
    )
    if summary["rank_average_full_models"]["macro_auc"] > best_score:
        best_name = "rank_average_full_models"
        best_probabilities = np.mean([_rank_normalize(value) for value in sources.values()], axis=0)
        best_score = summary["rank_average_full_models"]["macro_auc"]
    if classwise_metrics["macro_auc"] > best_score:
        best_name, best_probabilities, best_score = "validation_selected_per_class", selected, classwise_metrics["macro_auc"]

    args.output.mkdir(parents=True, exist_ok=True)
    write_submission(args.output / "valid.csv", studies, best_probabilities)
    summary["selected"] = {
        "name": best_name, "macro_auc": float(best_score),
        "delta_vs_resnet18": float(best_score - summary["models"]["resnet18_baseline"]["macro_auc"]),
        "metrics": evaluate(targets, best_probabilities),
    }
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.save(args.output / "study_probabilities.npy", best_probabilities)
    np.save(args.output / "study_targets.npy", targets)
    (args.output / "study_ids.json").write_text(json.dumps(expected_keys, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
