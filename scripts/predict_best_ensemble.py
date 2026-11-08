"""Replay the frozen medical ensemble on a complete split, then verify its CSV.

The default is the unlabeled test split. This command never selects models,
coefficients or thresholds. Component inference uses the same entry points as
the validation experiment and can resume only fingerprint-matched outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cxr.data import _read_metadata
from cxr.submission import validate_submission, write_submission
from scripts.compare_runs import load_submission


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def unique_studies(records):
    by_id = {}
    for row in records:
        by_id.setdefault((row["subject_id"], row["study_id"]), row)
    if not by_id:
        raise ValueError("the split must contain at least one study")
    return list(by_id.values())


def load_scores(folder: Path, studies, filename="study_probabilities.npy"):
    keys = [tuple(row) for row in read_json(folder / "study_ids.json")]
    expected = [(row["subject_id"], row["study_id"]) for row in studies]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError(f"component IDs do not match the requested split: {folder}")
    positions = {key: index for index, key in enumerate(keys)}
    values = np.load(folder / filename, allow_pickle=False)
    report = read_json(folder / "report.json")
    indices = report.get("local_label_indices", list(range(10)))
    if len(set(indices)) != len(indices) or any(i not in range(10) for i in indices):
        raise ValueError(f"invalid component class mapping: {folder}")
    if values.shape != (len(keys), len(indices)):
        raise ValueError(f"invalid component probability dimensions: {folder}")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"invalid component probabilities: {folder}")
    full = np.full((len(expected), 10), np.nan, dtype=np.float64)
    full[:, indices] = values[[positions[key] for key in expected]]
    return full


def replay_recipe(sources, recipe):
    """Apply recorded arithmetic only; missing inputs/classes fail closed."""
    first = np.asarray(next(iter(sources.values())), dtype=np.float64)
    if first.ndim != 2 or first.shape[1] != 10:
        raise ValueError("sources must be study-by-ten matrices")
    shape = first.shape

    def column(name, index):
        values = np.asarray(sources[name], dtype=np.float64)
        if values.shape != shape:
            raise ValueError(f"source shape differs: {name}")
        result = values[:, index]
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError(f"source has no valid prediction for class {index + 1}: {name}")
        return result

    def coefficient(value):
        alpha = float(value)
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("recipe coefficient must be finite and in [0, 1]")
        return alpha

    def checked_entries(entries):
        if len(entries) != 10 or {item["class"] for item in entries} != set(range(1, 11)):
            raise ValueError("recipe must contain exactly one entry for each class 1..10")
        return entries

    baseline = np.empty(shape, dtype=np.float64)
    for item in checked_entries(recipe["baseline_per_class_selection"]):
        j = item["class"] - 1
        if item["kind"] == "full_model_blend":
            left, right = item["left"], item["right"]
            alpha = coefficient(item["alpha_left"])
        elif item["kind"] == "external_blend":
            left, right = item["external_model"], item["base_model"]
            alpha = coefficient(item["alpha_external"])
        elif item["kind"] == "single_model":
            left = right = item["model"]
            alpha = 1.0
        else:
            raise ValueError(f"unsupported baseline formula: {item['kind']}")
        baseline[:, j] = alpha * column(left, j) + (1.0 - alpha) * column(right, j)

    result = baseline.copy()
    names = {
        "yaLLMan/rad-dino-chexpert fine-tuned on the local train split": "rad_dino_tta",
        "RussellALA/convnext-chexpert-seed1": "russellala_convnext",
        "mamounyosef/chest-xray-bench/medmae_vitb_nih_B_768_s2_seed1337": "medmae_vitb_nih_B_768_s2_seed1337",
    }
    for item in checked_entries(recipe["classes"]):
        j = item["class"] - 1
        alpha = coefficient(item["alpha_candidate"])
        if alpha == 0:
            continue
        name = names[item["source"]]
        result[:, j] = alpha * column(name, j) + (1.0 - alpha) * baseline[:, j]
    return baseline, result


def run_stage(name, command, stage_dir, artifacts, signature):
    marker = stage_dir / "completed.json"
    if marker.is_file():
        saved = read_json(marker)
        if saved["signature"] != signature:
            raise ValueError(f"refusing stale cached component: {name}")
        for relative, expected_sha in saved["artifacts"].items():
            if sha256(stage_dir / relative) != expected_sha:
                raise ValueError(f"cached component changed: {name}/{relative}")
        print(f"[{name}] verified completed component; reusing", flush=True)
        return
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise ValueError(f"incomplete component directory must be inspected before retry: {stage_dir}")
    print(f"[{name}] starting frozen-model inference", flush=True)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    log_path = stage_dir.parent / (stage_dir.name + ".log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"component {name} failed with exit {code}; see {log_path}")
    hashes = {str(path.relative_to(stage_dir)): sha256(path) for path in artifacts}
    write_json(marker, {"signature": signature, "artifacts": hashes})
    print(f"[{name}] complete", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="dedicated, resumable submission directory")
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    data_root, output = args.data_root.resolve(), args.output.resolve()
    if output == ROOT or output.is_relative_to(data_root):
        raise ValueError("output must be a dedicated directory outside the source data")
    recipe_path = ROOT / "docs/best_ensemble_recipe_20261105.json"
    manifest_path = data_root / f"{args.split}.csv"
    records = _read_metadata(manifest_path)
    if args.split == "test" and any(row["labels"] is not None for row in records):
        raise ValueError("test prediction must use an unlabeled test manifest")
    studies = unique_studies(records)
    recipe = read_json(recipe_path)
    public_manifest = read_json(ROOT / "docs/public_model_manifest.json")["repositories"]
    required = {
        "yaLLMan/rad-dino-chexpert": ("config.json", "preprocessor_config.json", "modeling_rad_dino.py", "pytorch_model.bin"),
        "itsomk/chexpert-densenet121": ("config.json", "pytorch_model.safetensors"),
        "RussellALA/convnext-chexpert-seed1": ("config.json", "model.safetensors"),
        "mamounyosef/chest-xray-bench": ("modeling.py",),
    }
    bench_names = ("rad_dino_vitB_768", "medmae_vitb_nih_B_768_s2", "medmae_vitb_nih_B_768_s2_seed1337")
    required["mamounyosef/chest-xray-bench"] += tuple(f"{name}/{file}" for name in bench_names
                                                          for file in ("config.json", "model.safetensors"))
    fingerprints = {}
    for repo, files in required.items():
        payload = public_manifest[repo]
        for file in files:
            path = ROOT / payload["directory"] / file
            actual = sha256(path)
            if actual != payload["files"][file]:
                raise ValueError(f"public model fingerprint differs: {repo}/{file}")
            fingerprints[str(path.relative_to(ROOT))] = actual
    resnet = ROOT / "weights/resnet18-finetune-v1/resnet18-finetune-v1.pt"
    rad_checkpoint = ROOT / "outputs/rad-dino-finetune-20261105/best.pt"
    for path, expected in (
        (resnet, "8141045a6846d210c772623e2a672f1701b7bcb7b544bb6746136c7854d3626c"),
        (rad_checkpoint, "ed88d2dfb23b1a6cb9ea82128642dd4b24b781f912b90ce5dde17616b9473e73"),
    ):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"local checkpoint fingerprint differs: {path}")
        fingerprints[str(path.relative_to(ROOT))] = actual
    signature = {"split": args.split, "metadata_sha256": sha256(manifest_path),
                 "recipe_sha256": sha256(recipe_path), "model_fingerprints": fingerprints,
                 "images": len(records), "studies": len(studies), "device": args.device}
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = output / "run_manifest.json"
    if run_manifest.is_file() and read_json(run_manifest) != signature:
        raise ValueError("output belongs to a different split, recipe or checkpoint set")
    if not run_manifest.is_file() and any(output.iterdir()):
        raise ValueError("output is not empty and has no matching run manifest")
    write_json(run_manifest, signature)
    components = output / "components"
    components.mkdir(exist_ok=True)
    common = ["--data-root", str(data_root), "--split", args.split, "--device", args.device]
    stages = [
        ("resnet18", "predict.py", ["--checkpoint", str(resnet)], "csv"),
        ("rad_dino", "scripts/eval_rad_dino_checkpoint.py", ["--checkpoint", str(rad_checkpoint), "--horizontal-flip"], "tta"),
        ("densenet121", "scripts/eval_densenet121_public.py", [], "full"),
        *[(name, "scripts/eval_chest_xray_bench.py", ["--models", name], "bench") for name in bench_names],
        ("russellala_convnext", "scripts/eval_russellala_convnext.py", [], "full"),
    ]
    folders = {}
    for name, script, extra, kind in stages:
        stage_dir = components / name
        stage_output = stage_dir / f"{args.split}.csv" if kind == "csv" else stage_dir
        command = [sys.executable, "-u", str(ROOT / script), *common, *extra, "--output", str(stage_output)]
        artifact_dir = stage_dir / name if kind == "bench" else stage_dir
        folders[name] = artifact_dir
        if kind == "csv":
            artifacts = [stage_output, stage_output.with_suffix(".report.json")]
        else:
            artifacts = [artifact_dir / file for file in ("report.json", "study_ids.json", "study_probabilities.npy")]
            if kind == "tta":
                artifacts += [artifact_dir / file for file in ("original_study_probabilities.npy", "flipped_study_probabilities.npy")]
        stage_signature = {**signature, "script_sha256": sha256(ROOT / script), "command": command}
        run_stage(name, command, stage_dir, artifacts, stage_signature)
    sources = {"resnet18_baseline": load_submission(folders["resnet18"] / f"{args.split}.csv", studies)[0],
               "rad_dino_finetuned": load_scores(folders["rad_dino"], studies, "original_study_probabilities.npy"),
               "rad_dino_tta": load_scores(folders["rad_dino"], studies),
               "densenet121_chexpert": load_scores(folders["densenet121"], studies),
               "russellala_convnext": load_scores(folders["russellala_convnext"], studies)}
    sources.update({name: load_scores(folders[name], studies) for name in bench_names})
    baseline, probabilities = replay_recipe(sources, recipe)
    destination = output / "submission.csv"
    write_submission(destination, studies, probabilities, thresholds=0.5)
    validation = validate_submission(destination, records)
    csv_scores, gates, _ = load_submission(destination, studies)
    if not np.array_equal(csv_scores, probabilities) or not np.all(gates == 0.5):
        raise ValueError("CSV did not round-trip the frozen ensemble probabilities")
    np.save(output / "baseline_study_probabilities.npy", baseline)
    np.save(output / "study_probabilities.npy", probabilities)
    write_json(output / "study_ids.json", [[row["subject_id"], row["study_id"]] for row in studies])
    write_json(output / "recipe.json", recipe)
    report = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              **validation, **signature, "submission_sha256": sha256(destination),
              "coefficient_refitting": False, "threshold": 0.5, "csv_roundtrip_exact": True,
              "test_metrics": None, "selection_metrics_are_validation_only": True}
    write_json(output / "submission.report.json", report)
    print(json.dumps({"submission": str(destination), **validation, "sha256": report["submission_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
