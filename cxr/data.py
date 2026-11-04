"""Strict, standard-library-only indexing of the chest X-ray CSV manifests.

``read_records(csv_path, image_root)`` takes the *split* image directory, e.g.
``cxr_first/train``. Returned paths are resolved absolute strings. Labels are
10-element multi-hot float lists (class 1 is index 0), or None for test.csv.
No image contents are opened. Missing or ambiguous images fail closed.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re


_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"})
_ID_COLUMNS = ("Dicom_id", "Subject_id", "Study_id")


def _identifier(value: str | None, column: str, line: int) -> str:
    """Require a single portable filesystem component, not a supplied path."""
    if not value or value != value.strip():
        raise ValueError(f"line {line}: missing or whitespace-padded {column}")
    if (
        value in {".", ".."}
        or any(c in value for c in '/\\:<>\"|?*')
        or any(ord(c) < 32 for c in value)
        or value.endswith((".", " "))
        or value.split(".")[0].rstrip(" ").upper() in {
            "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
            *(f"COM{n}" for n in "123456789¹²³"),
            *(f"LPT{n}" for n in "123456789¹²³"),
        }
    ):
        raise ValueError(f"line {line}: unsafe {column}: {value!r}")
    return value


def _labels(value: str | None, line: int) -> list[float]:
    if value is None or not value.strip():
        raise ValueError(f"line {line}: missing Predict_class for labeled CSV")
    tokens = [part.strip() for part in value.split(";")]
    if any(re.fullmatch(r"(?:[1-9]|10)", token) is None for token in tokens):
        raise ValueError(f"line {line}: invalid Predict_class {value!r}; expected classes 1..10")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"line {line}: duplicate class in Predict_class {value!r}")
    result = [0.0] * 10
    for token in tokens:
        result[int(token) - 1] = 1.0
    return result


def _read_metadata(csv_path: str | Path) -> list[dict]:
    """Validate IDs/labels without accessing images; shared by the audit CLI.

    Only a file named test.csv may omit Predict_class. Test metadata other than
    IDs is deliberately ignored, and a labeled test manifest is rejected.
    """
    csv_path = Path(csv_path)
    is_test = csv_path.name.casefold() == "test.csv"
    records = []
    seen_ids: set[str] = set()
    studies: dict[str, tuple[str, tuple[float, ...] | None]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{csv_path}: missing or duplicate CSV headers")
        missing = set(_ID_COLUMNS) - set(fields)
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")
        if is_test and "Predict_class" in fields:
            raise ValueError(f"{csv_path}: test.csv must not contain Predict_class")
        if not is_test and "Predict_class" not in fields:
            raise ValueError(f"{csv_path}: missing Predict_class for labeled CSV")
        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{csv_path}, line {line}: malformed CSV row")
            dicom, subject, study = (
                _identifier(row[column], column, line) for column in _ID_COLUMNS
            )
            if dicom in seen_ids:
                raise ValueError(f"line {line}: duplicate Dicom_id {dicom!r}")
            seen_ids.add(dicom)
            labels = None if is_test else _labels(row["Predict_class"], line)
            signature = (subject, None if labels is None else tuple(labels))
            if study in studies and studies[study] != signature:
                raise ValueError(f"line {line}: conflicting subject or labels for study {study!r}")
            studies[study] = signature
            records.append({
                "dicom_id": dicom,
                "subject_id": subject,
                "study_id": study,
                "labels": labels,
            })
    return records


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"image path escapes image_root: {path}")
    return resolved


def _attach_paths(records: list[dict], image_root: str | Path, *, allow_missing: bool = False) -> tuple[list[dict], list[dict]]:
    """Resolve exact ID stems under subject/study, with one scan per study.

    The audit-only allow_missing mode returns expected .jpg paths plus a list
    of missing IDs; the public reader never returns a nonexistent image.
    """
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"image_root is not an existing directory: {root}")
    cache: dict[tuple[str, str], dict[str, list[Path]]] = {}
    result, missing = [], []
    for record in records:
        key = (record["subject_id"], record["study_id"])
        directory = _inside(root / key[0] / key[1], root)
        if key not in cache:
            indexed: dict[str, list[Path]] = {}
            if directory.is_dir():
                for entry in directory.iterdir():
                    if entry.suffix.casefold() in _IMAGE_SUFFIXES:
                        resolved = _inside(entry, root)
                        if resolved.is_file():
                            indexed.setdefault(entry.stem, []).append(resolved)
            cache[key] = indexed
        candidates = cache[key].get(record["dicom_id"], [])
        if len(candidates) > 1:
            raise ValueError(f"ambiguous images for Dicom_id {record['dicom_id']!r}: {candidates}")
        if not candidates:
            expected = _inside(directory / (record["dicom_id"] + ".jpg"), root)
            detail = {"dicom_id": record["dicom_id"], "expected_path": str(expected)}
            if not allow_missing:
                raise FileNotFoundError(f"missing image for Dicom_id {record['dicom_id']!r}: {expected}")
            missing.append(detail)
            path = expected
        else:
            path = candidates[0]
        result.append({**record, "path": str(path)})
    return result, missing


def read_records(csv_path: str | Path, image_root: str | Path) -> list[dict]:
    """Return validated image records; never decode images or infer labels.

    Raises ValueError for bad metadata, unsafe paths, inconsistent studies or
    ambiguous image files, and FileNotFoundError for absent image roots/images.
    Only test.csv may be unlabeled; all other CSVs require valid Predict_class.
    """
    records, _ = _attach_paths(_read_metadata(csv_path), image_root)
    return records
