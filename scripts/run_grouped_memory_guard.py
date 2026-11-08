"""Explicit runtime amendment wrapper for grouped runner/trainer snapshots."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_helper_path = ROOT / "cxr/evaluation_memory.py"
_helper_spec = importlib.util.spec_from_file_location("_evaluation_memory_guard", _helper_path)
if _helper_spec is None or _helper_spec.loader is None:
    raise RuntimeError("cannot load runtime evaluation helper")
evaluation_memory = importlib.util.module_from_spec(_helper_spec)
sys.modules[_helper_spec.name] = evaluation_memory
_helper_spec.loader.exec_module(evaluation_memory)


def load_module(path, name):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load immutable module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("loaded module path changed unexpectedly")
    return module


def rewrite_trainer_command(command, guard_path):
    command = list(command)
    trainer_positions = [i for i, part in enumerate(command)
                         if Path(str(part)).name == "train_rad_dino_grouped.py"
                         and i > 0 and Path(str(part)).parent.name == "scripts"
                         and Path(str(part)).parent.parent.name == "source"]
    if not trainer_positions:
        return command
    if len(trainer_positions) != 1:
        raise RuntimeError("stage command contains multiple grouped trainer paths")
    index = trainer_positions[0]
    trainer_path = command[index]
    return [*command[:index], str(guard_path), "--trainer", trainer_path,
            *command[index + 1:]]


def verify_registration_context(amendment, output, plan):
    output = Path(output).resolve(strict=True)
    if Path(amendment["experiment_output"]).resolve(strict=True) != output:
        raise RuntimeError("runtime amendment experiment_output mismatch")
    if amendment["experiment_plan_sha256"] != evaluation_memory.sha256(output / "plan.json"):
        raise RuntimeError("runtime amendment experiment plan hash mismatch")
    fusion = "source_experiment" in plan
    source = Path(plan["source_experiment"]).resolve(strict=True) if fusion else output
    source_plan_path = source / "plan.json"
    source_plan_hash = evaluation_memory.sha256(source_plan_path)
    if amendment.get("source_plan_sha256") != source_plan_hash:
        raise RuntimeError("runtime amendment source plan hash mismatch")
    expected_fusion_hash = evaluation_memory.sha256(output / "plan.json") if fusion else amendment.get("fusion_plan_sha256")
    if amendment.get("fusion_plan_sha256") != expected_fusion_hash:
        raise RuntimeError("runtime amendment fusion plan hash mismatch")
    partition = evaluation_memory.sha256(source / "partition.json")
    inventory = evaluation_memory.sha256(source / "image_inventory.json")
    if amendment.get("partition_sha256") != partition:
        raise RuntimeError("runtime amendment partition hash mismatch")
    if amendment.get("inventory_sha256") != inventory:
        raise RuntimeError("runtime amendment inventory hash mismatch")


def runner_entry(output, argv):
    output = Path(output).resolve(strict=True)
    plan_path = output / "plan.json"
    state_path = output / "experiment.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not state_path.is_file():
        raise RuntimeError("experiment state is missing")
    fusion = "source_experiment" in plan
    runner_name = "run_oof_fusion.py" if fusion else "run_grouped_generalization.py"
    runner_path = output / "source/scripts" / runner_name
    amendment_path = output / evaluation_memory.AMENDMENT_NAME
    amendment = evaluation_memory.verify_amendment(
        amendment_path, Path(__file__), ROOT / "cxr/evaluation_memory.py",
    )
    verify_registration_context(amendment, output, plan)
    expected_key = "fusion_runner_sha256" if fusion else "source_runner_sha256"
    if expected_key in amendment and amendment[expected_key] != evaluation_memory.sha256(runner_path):
        raise RuntimeError(f"runtime amendment mismatch for {expected_key}")
    module = load_module(runner_path, f"_immutable_{runner_path.stem}_{output.name}")
    original_atomic_json = module.atomic_json
    amendment_binding = {
        "path": str(amendment_path.resolve()),
        "sha256": evaluation_memory.sha256(amendment_path),
        "policy": evaluation_memory.POLICY,
        "guard_sha256": evaluation_memory.sha256(Path(__file__)),
        "helper_sha256": evaluation_memory.sha256(_helper_path),
    }

    def amend_root_report(path, data):
        path = Path(path)
        if path.resolve() in {(output / "report.json").resolve(),
                              (output / "experiment.json").resolve()} and isinstance(data, dict):
            data = {**data, "runtime_amendment": amendment_binding}
        return original_atomic_json(path, data)

    module.atomic_json = amend_root_report
    if fusion:
        original_run_stage = module.run_stage

        def guarded_run_stage(stage_output, state, name, command, folder, artifacts):
            rewritten = rewrite_trainer_command(command, Path(__file__).resolve())
            return original_run_stage(stage_output, state, name, rewritten, folder, artifacts)

        module.run_stage = guarded_run_stage
    else:
        original_stage_command = module.stage_command

        def guarded_stage_command(*args, **kwargs):
            name, command = original_stage_command(*args, **kwargs)
            return name, rewrite_trainer_command(command, Path(__file__).resolve())

        module.stage_command = guarded_stage_command
    module.main(argv)


def trainer_entry(trainer_path, argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    known, _ = parser.parse_known_args(argv)
    amendment_path = evaluation_memory.find_amendment(known.output)
    amendment = evaluation_memory.verify_amendment(
        amendment_path, Path(__file__), ROOT / "cxr/evaluation_memory.py",
    )
    amendment_root = amendment_path.parent.resolve(strict=True)
    expected_trainer = amendment_root / "source/scripts/train_rad_dino_grouped.py"
    if Path(trainer_path).resolve(strict=True) != expected_trainer.resolve(strict=True):
        raise RuntimeError("trainer path does not match the registered immutable snapshot")
    plan = json.loads((amendment_root / "plan.json").read_text(encoding="utf-8"))
    verify_registration_context(amendment, amendment_root, plan)
    module = load_module(trainer_path, "_immutable_grouped_trainer")
    if "trainer_sha256" in amendment and amendment["trainer_sha256"] != evaluation_memory.sha256(trainer_path):
        raise RuntimeError("runtime amendment mismatch for trainer_sha256")
    cfg = module.parser().parse_args(argv)
    evaluation_memory.install(
        module, cfg, cfg.output, {}, amendment_path, Path(__file__),
        ROOT / "cxr/evaluation_memory.py",
    )
    module.main(argv)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--trainer" in args:
        index = args.index("--trainer")
        if index + 1 >= len(args):
            raise SystemExit("--trainer requires an immutable trainer path")
        trainer_path = args[index + 1]
        trainer_entry(trainer_path, [*args[:index], *args[index + 2:]])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    known, _ = parser.parse_known_args(args)
    runner_entry(known.output, args)


if __name__ == "__main__":
    main()
