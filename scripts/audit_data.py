"""Audit CSV metadata and image completeness without opening any images.

Run from the project root:
    python scripts/audit_data.py --data-root cxr_first --output outputs/audit.json

Test data usage is strictly IDs, paths and completeness. Only training and
validation CSVs contribute label prevalence or dimension summaries.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

# Support the requested direct script invocation without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxr.data import _attach_paths, _read_metadata  # noqa: E402


def _dimensions(csv_path: Path) -> dict:
    """Summarize declared train/valid CSV dimensions, not decoded image sizes."""
    rows, columns = [], []
    invalid = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            try:
                height, width = int(record["Rows"]), int(record["Columns"])
                if height <= 0 or width <= 0:
                    raise ValueError("nonpositive dimension")
            except (ValueError, TypeError, KeyError):
                invalid += 1
                continue
            rows.append(height)
            columns.append(width)
    def describe(values: list[int]) -> dict | None:
        if not values:
            return None
        return {"min": min(values), "median": statistics.median(values), "max": max(values)}
    return {
        "source": "CSV Rows/Columns only; no image decoding or verification",
        "valid_rows": len(rows), "invalid_or_missing_rows": invalid,
        "rows": describe(rows), "columns": describe(columns),
    }


def audit_data(data_root: str | Path) -> dict:
    root = Path(data_root).resolve()
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "policy": {
            "raw_files_modified": False,
            "images_opened_or_decoded": False,
            "test_usage": "IDs/path/completeness only; no test labels, dimensions, pixel statistics or modeling",
            "label_prevalence_unit": "unique study (not image); class 1 maps to vector index 0",
        },
        "splits": {}, "overlaps": {}, "risks": [],
    }
    all_records = {}
    for split in ("train", "valid", "test"):
        csv_path = root / f"{split}.csv"
        metadata = _read_metadata(csv_path)
        records, missing = _attach_paths(metadata, root / split, allow_missing=True)
        all_records[split] = records
        studies = {record["study_id"]: record for record in records}
        summary = {
            "images": len(records), "studies": len(studies),
            "patients": len({record["subject_id"] for record in records}),
            "present_images": len(records) - len(missing),
            "missing_images_count": len(missing), "missing_images": missing,
            "metadata_validation": "passed: unique Dicom_id, safe IDs, consistent study subjects and available labels",
        }
        if split != "test":
            summary["study_label_prevalence"] = {
                str(label + 1): {
                    "count": sum(int(record["labels"][label]) for record in studies.values()),
                    "fraction": (
                        sum(record["labels"][label] for record in studies.values()) / len(studies)
                        if studies else 0.0
                    ),
                }
                for label in range(10)
            }
            summary["declared_dimensions"] = _dimensions(csv_path)
        report["splits"][split] = summary
        if missing:
            report["risks"].append(f"{split}: {len(missing)} missing images")
        if not records:
            report["risks"].append(f"{split}: empty CSV")

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = {}
        for field, name in (("subject_id", "patients"), ("study_id", "studies"), ("dicom_id", "dicom_ids")):
            shared = sorted(
                {row[field] for row in all_records[left]} & {row[field] for row in all_records[right]}
            )
            overlap[name] = {"count": len(shared), "examples": shared[:20]}
            if shared:
                report["risks"].append(f"{left}/{right}: {len(shared)} overlapping {name}")
        report["overlaps"][f"{left}_{right}"] = overlap

    # Cross-split study conflicts are separate from within-manifest validation.
    subjects_by_study = {}
    labels_by_study = {}
    subject_conflicts, label_conflicts = set(), set()
    for split, records in all_records.items():
        for row in records:
            study, subject = row["study_id"], row["subject_id"]
            if study in subjects_by_study and subjects_by_study[study] != subject:
                subject_conflicts.add(study)
            subjects_by_study[study] = subject
            if split != "test":
                labels = tuple(row["labels"])
                if study in labels_by_study and labels_by_study[study] != labels:
                    label_conflicts.add(study)
                labels_by_study[study] = labels
    report["cross_split_conflicts"] = {
        "study_subjects": {"count": len(subject_conflicts), "examples": sorted(subject_conflicts)[:20]},
        "train_valid_study_labels": {"count": len(label_conflicts), "examples": sorted(label_conflicts)[:20]},
    }
    if subject_conflicts or label_conflicts:
        report["risks"].append("Cross-split study metadata conflicts detected; resolve before modeling")
    report["notes"] = [
        "Counts are manifest counts; missing files are checked in actual subject/study directories.",
        "Label prevalence is descriptive only; test metadata must not inform model choices.",
        "Dimension ranges describe CSV metadata, not actual decoded image dimensions.",
        "No overlapping IDs does not establish independence beyond the provided identifiers.",
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("cxr_first"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit.json"))
    args = parser.parse_args(argv)
    root, output = args.data_root.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        parser.error("--output must be outside the raw data root")
    try:
        report = audit_data(root)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Audit failed: {error}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for split, summary in report["splits"].items():
        print(f"{split}: {summary['images']} images, {summary['studies']} studies, "
              f"{summary['patients']} patients, {summary['missing_images_count']} missing images")
    print(f"Audit report: {output}")
    for risk in report["risks"]:
        print(f"RISK: {risk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
