"""Synthetic tests for patient-level CXR split construction."""

import copy
import unittest

from cxr.patient_splits import audit_records, make_patient_partition, partition_records


def records_fixture(patient_count=40):
    records = []
    for patient_index in range(patient_count):
        labels = [
            float((label_index - patient_index) % 10 < 5)
            for label_index in range(10)
        ]
        for study_index in range(2):
            study = f"study_{patient_index}_{study_index}"
            for image_index in range(1 + (study_index == 0)):
                records.append({
                    "subject_id": f"patient_{patient_index:03d}",
                    "study_id": study,
                    "dicom_id": f"image_{patient_index}_{study_index}_{image_index}",
                    "path": f"/synthetic/{patient_index}/{study_index}/{image_index}.jpg",
                    "labels": labels.copy(),
                })
    return records


class PatientSplitTests(unittest.TestCase):
    def test_deterministic_row_order_independent_partition_and_summary(self):
        records = [
            {key: value for key, value in row.items() if key != "path"}
            for row in records_fixture()
        ]
        original = copy.deepcopy(records)
        first = make_patient_partition(records)
        second = make_patient_partition(list(reversed(records)))

        self.assertEqual(first, second)
        self.assertEqual(records, original)
        self.assertEqual(first["seed"], 20261105)
        self.assertEqual(first["folds"], 3)
        self.assertEqual(first["audit_fraction"], 0.2)
        self.assertEqual(set(first["subject_partition"].values()), {"audit", 0, 1, 2})
        self.assertEqual(sum(row["images"] for row in first["summary"].values()), len(records))
        self.assertEqual(sum(row["studies"] for row in first["summary"].values()), 80)
        self.assertEqual(sum(row["patients"] for row in first["summary"].values()), 40)
        for row in first["summary"].values():
            self.assertEqual(set(row["study_positive_counts"]), {str(i) for i in range(1, 11)})
            self.assertEqual(set(row["positive_patient_counts"]), {str(i) for i in range(1, 11)})
            self.assertTrue(all(0 < count < row["studies"] for count in row["study_positive_counts"].values()))

    def test_seeded_shuffle_has_exact_patient_capacities_and_approximate_label_mix(self):
        records = []
        for patient_index in range(60):
            labels = [float(patient_index % 5 == 0)] + [
                float((patient_index + label_index * 7) % 4 != 0)
                for label_index in range(1, 10)
            ]
            for study_index in range(1 + patient_index % 3):
                study_id = f"heterogeneous_study_{patient_index}_{study_index}"
                for image_index in range(1 + patient_index % 2):
                    records.append({
                        "subject_id": f"heterogeneous_patient_{patient_index:03d}",
                        "study_id": study_id,
                        "dicom_id": f"heterogeneous_image_{patient_index}_{study_index}_{image_index}",
                        "labels": labels.copy(),
                    })

        partition = make_patient_partition(records)

        self.assertEqual(
            {name: summary["patients"] for name, summary in partition["summary"].items()},
            {"audit": 12, "0": 16, "1": 16, "2": 16},
        )
        total_rare_positive_patients = sum(
            summary["positive_patient_counts"]["1"]
            for summary in partition["summary"].values()
        )
        self.assertEqual(total_rare_positive_patients, 12)
        for name, summary in partition["summary"].items():
            target = summary["patients"] * total_rare_positive_patients / 60
            actual = summary["positive_patient_counts"]["1"]
            self.assertLessEqual(abs(actual - target), 2.5, msg=f"{name}: {actual} vs {target}")

    def test_same_patient_never_crosses_sets_and_helpers_exclude_audit(self):
        records = records_fixture()
        partition = make_patient_partition(records)
        partition["train_metadata_sha256"] = "metadata-digest"
        partition["public_checkpoint_sha256"] = "checkpoint-digest"

        for patient in {row["subject_id"] for row in records}:
            assignments = {partition["subject_partition"][row["subject_id"]]
                           for row in records if row["subject_id"] == patient}
            self.assertEqual(len(assignments), 1)
        for fold in range(3):
            development, evaluation = partition_records(records, partition, fold)
            development_patients = {row["subject_id"] for row in development}
            evaluation_patients = {row["subject_id"] for row in evaluation}
            self.assertFalse(development_patients & evaluation_patients)
            self.assertTrue(all(partition["subject_partition"][p] != "audit" for p in development_patients))
            self.assertTrue(all(partition["subject_partition"][p] == fold for p in evaluation_patients))
            self.assertEqual(len(development) + len(evaluation) + sum(
                row["subject_id"] in {
                    patient for patient, value in partition["subject_partition"].items() if value == "audit"
                } for row in records
            ), len(records))

        development, audit = audit_records(records, partition)
        self.assertFalse({row["subject_id"] for row in development} & {row["subject_id"] for row in audit})
        self.assertTrue(all(partition["subject_partition"][row["subject_id"]] == "audit" for row in audit))
        self.assertEqual(len(development) + len(audit), len(records))

    def test_duplicate_images_ambiguous_ownership_and_inconsistent_labels_fail(self):
        base = records_fixture(40)
        duplicate = copy.deepcopy(base)
        duplicate.append(copy.deepcopy(duplicate[0]))
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            make_patient_partition(duplicate)

        ownership = copy.deepcopy(base)
        ownership.append({**copy.deepcopy(ownership[0]), "subject_id": "other"})
        ownership[-1]["dicom_id"] = "different_image"
        with self.assertRaisesRegex(ValueError, "ambiguous patient ownership"):
            make_patient_partition(ownership)

        labels = copy.deepcopy(base)
        labels.append({**copy.deepcopy(labels[0]), "dicom_id": "different_image", "labels": [0.0] * 10})
        with self.assertRaisesRegex(ValueError, "inconsistent labels"):
            make_patient_partition(labels)

    def test_unlabeled_or_nonbinary_records_fail(self):
        records = records_fixture(40)
        records[0]["labels"] = None
        with self.assertRaisesRegex(ValueError, "ten binary labels"):
            make_patient_partition(records)
        records = records_fixture(40)
        records[0]["labels"][0] = 0.5
        with self.assertRaisesRegex(ValueError, "ten binary labels"):
            make_patient_partition(records)

    def test_insufficient_class_support_and_invalid_partition_fail(self):
        records = records_fixture(40)
        # Class 1 has only one positive patient, so four partitions cannot all
        # receive positive studies.
        for record in records:
            if record["subject_id"] != "patient_000":
                record["labels"][0] = 0.0
        with self.assertRaisesRegex(ValueError, "insufficient positive or negative patients"):
            make_patient_partition(records)

        records = records_fixture()
        partition = make_patient_partition(records)
        partition["subject_partition"].pop(next(iter(partition["subject_partition"])))
        with self.assertRaisesRegex(ValueError, "cover records exactly"):
            partition_records(records, partition, 0)

    def test_invalid_fold_and_too_small_input_fail(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            make_patient_partition(records_fixture(3))
        records = records_fixture()
        partition = make_patient_partition(records)
        with self.assertRaisesRegex(ValueError, "fold must"):
            partition_records(records, partition, 3)
        with self.assertRaisesRegex(ValueError, "folds must"):
            make_patient_partition(records, folds=1)


if __name__ == "__main__":
    unittest.main()
