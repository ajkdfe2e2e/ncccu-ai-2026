"""Fetch pinned public CheXpert weights and record their provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

ASSETS = {
    "yaLLMan/rad-dino-chexpert": {
        "revision": "db02e1b7234dd83c6d7c4485963ef5b22df9e5db",
        "directory": "weights/medical/rad-dino-chexpert-upstream",
        "license": "Apache-2.0 per upstream model card; dataset terms also apply",
        "files": {
            "README.md": "ff35ad29ebe4c3c662d86d3879712daf42b1babc0c60afa07aa0ddd78d24764c",
            "config.json": "2475e5d0a4e76fe6fec83c9e18e4a6229e7e8ecf24263616cdb2c7e5680fd7a7",
            "modeling_rad_dino.py": "831eb4af477714ed48f62546681dc2b79b4105633a62de457038b3ab7f1bbb84",
            "preprocessor_config.json": "334577e6af24cca3598cad2b9fffc5411628950db231b965c0641e154c74d96a",
            "pytorch_model.bin": "a81fea07272e6cff836b838ba4c12dd26d128b5fb16b3eb2e8c089a4a5ca14f8",
        },
    },
    "itsomk/chexpert-densenet121": {
        "revision": "81be45e4d43011e8f50e17d002190c76cb069edd",
        "directory": "weights/medical/itsomk-chexpert-densenet121",
        "license": "MIT repository license; CheXpert dataset terms also apply",
        "files": {
            "README.md": "0b86bcb14ed4fa616cebe4c67fef395ad91d00e6c8f70d04022e159649ae05fd",
            "LICENSE": "84348ac5922e54fbd464f7bd4846323fcf917c1cfda6cbb46e79122b3ff35e7d",
            "config.json": "55e983ec2197cecccb81aee05d3464fe9ea483a94ed2476b00e8c1e42b7c19d1",
            "pytorch_model.safetensors": "f6fa96434c2b4822b0e1f6c3ea4f236466788e49749c05fd5c30a83a263450f2",
        },
    },
    "RussellALA/convnext-chexpert-seed1": {
        "revision": "0444dbe62e26e0206b485c7771b06530b19e66f7",
        "directory": "weights/medical/russellala-convnext-chexpert-seed1",
        "license": "chexpert-research-use per upstream model card",
        "files": {
            "README.md": "0c8b1b2674f2d60adf969836f0b2780b62ae2f2cbc694765b50cecb62269d8bf",
            "config.json": "477ca025973770e256e8e44f63f6f7ad9474b0db62418047ae1a47706bda4a58",
            "model.safetensors": "d31eb9c739fe19d2aa8ed14a440886f39e4ce547f0d24d6133735c06c9a85c45",
        },
    },
    "mamounyosef/chest-xray-bench": {
        "revision": "1735afb88371d98f87667c5f9966ed47bf4ef678",
        "directory": "weights/medical/chest-xray-bench",
        "license": "CC BY-NC 4.0 per upstream model card",
        "files": {
            "README.md": "95a9106de02cdcb608b00d4f6937d57db2ca2fb60a2b968c746f08a636112bee",
            "modeling.py": "c93895c7b646c272dec861e40a0cb2f010b2c019ae6099c457192027ff96ae38",
            "rad_dino_vitB_768/config.json": "56eccbfbe95f62f49c75f27ef86ffc4d9d1577e95cd493604d2df12a7af0dbea",
            "rad_dino_vitB_768/model.safetensors": "04c02d377bc62bdc91d555832c1c007839a9777762ebc548a87375ac8fdff7d3",
            "rad_dino_vitB_768/thresholds.json": "61baccd34a8c9b3b1054d96e386be7dab228d5278fde5f6020781a12f48a3720",
            "medmae_vitb_nih_B_768_s2/config.json": "3283b4a3b4fa17591f8127018d14d36365150988f514fed0ef976fbe39e5e12f",
            "medmae_vitb_nih_B_768_s2/model.safetensors": "78039c74ddf09617b83e1b4a64a6a100e3b5d54eecf1c7d30061e10339b383d3",
            "medmae_vitb_nih_B_768_s2/thresholds.json": "70ed158c5835079b296388132e126d18f47ad71e4ffe3c9fdb2896e2b855d9f2",
            "convnext_base_22k_1600x1312/config.json": "2a22b589ccd7a6fe6dbd9408dc9fd3ad6c503bb3f8076206c549606e5db40e4c",
            "convnext_base_22k_1600x1312/model.safetensors": "24b8017d05761e3ddc42d3dd6e9c010b2c514ea0bef0a7d145d31f274f40dcdb",
            "convnext_base_22k_1600x1312/thresholds.json": "4d560359c7dc419e2163ae9acec981556019abf5fb1a039d1b66bafb159dad1b",
            "medmae_vitb_nih_B_768_s2_seed1337/config.json": "4e0e6c5a565341fb8f480096a9206b37a2fe68308b2bc87ebff336994bd430ba",
            "medmae_vitb_nih_B_768_s2_seed1337/model.safetensors": "a8a4247e448cefa65207a9041d16eda29f3dd853084a5db3e1ca3684dd4b8dee",
            "medmae_vitb_nih_B_768_s2_seed1337/thresholds.json": "5bdd5586099dcb8a921ea889aee1e402f37a3595f5635556cc52ff12dc7985c0",
            "medmae_vitb_nih_B_448_s1_seed7/config.json": "3ed0d1e9a4c7e6dee843dd550211065afea124b9c51192640c33094f9c12f221",
            "medmae_vitb_nih_B_448_s1_seed7/model.safetensors": "cc29550b7ab9310ad3d7416600e6633661385fa982fb96038553777e38064926",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", choices=list(ASSETS),
                        help="fetch only this pinned repository; repeat for multiple repositories")
    args = parser.parse_args()
    for repo_id in args.repo or ASSETS:
        spec = ASSETS[repo_id]
        destination = ROOT / spec["directory"]
        destination.mkdir(parents=True, exist_ok=True)
        file_records = {}
        for filename, expected in spec["files"].items():
            output = destination / filename
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.is_file() or sha256(output) != expected:
                cached = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=spec["revision"]))
                shutil.copy2(cached, output)
            actual = sha256(output)
            if expected and actual != expected:
                raise ValueError(f"SHA-256 mismatch for {repo_id}/{filename}: {actual}")
            file_records[filename] = {"bytes": output.stat().st_size, "sha256": actual}
            print(f"{repo_id}@{spec['revision']} {filename}: {output.stat().st_size} bytes", flush=True)
        provenance = {
            "repo_id": repo_id,
            "revision": spec["revision"],
            "license": spec["license"],
            "files": file_records,
            "data_overlap_status": "unknown for the local dataset",
        }
        (destination / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
