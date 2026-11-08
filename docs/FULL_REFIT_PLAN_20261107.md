# Full training-data deployment refit — 2026-11-07

## Fixed decision

The completed patient-grouped experiment selected configuration A with EMA.
This run refits that fixed recipe from the original public initialization on
all records in `train.csv`. It is a deployment refit, not a new validation
experiment. Its training patients include the former audit patients.

- Source experiment: `outputs/generalization-rad-dino-20261105-v2`.
- Public initialization: `yaLLMan/rad-dino-chexpert`, upstream revision
  `db02e1b7234dd83c6d7c4485963ef5b22df9e5db`.
- Public checkpoint SHA-256:
  `a81fea07272e6cff836b838ba4c12dd26d128b5fb16b3eb2e8c089a4a5ca14f8`.
- Training: 30,344 images, 18,513 studies, 11,055 patients.
- Previous deployment checkpoint used 24,375 images and 8,844 patients.
- Only `train.csv` supplies training labels. Official `valid.csv` is not used.
- The unlabeled `test.csv` is read only by the export stage after training.

## Frozen training recipe

Input size 518; the original public image processor; ten copied classifier
rows in the existing local class order; last two encoder blocks, layer norm,
and classifier trainable; weighted BCE; AdamW; backbone LR 1e-5; head LR 1e-3;
weight decay 1e-4; three fixed epochs; cosine LR with minimum 1e-6;
batch size 2; accumulation 8; AMP; gradient clipping 1.0; seed 2026;
EMA decay 0.999; existing mild affine augmentation.

DataLoader workers change from four to zero to reduce Windows memory pressure
and allow epoch-boundary recovery. Full refits use an explicitly seeded loader
generator. The stochastic trajectory is not claimed to reproduce the earlier
development-only run.

No early stopping, validation-driven checkpoint selection, TTA search, threshold
search, or ensemble-weight search is part of this run.

## Reliability and output

Lock source reports, metadata, public files, image inventory and code snapshot
hashes before launch. Preserve previous CSV files and record their hashes.
Write an atomic recovery checkpoint after each completed epoch, including raw
weights, EMA, optimizer, scheduler, AMP scaler, RNG streams and training history.
Resume requires the same locked configuration and software versions.

Training and test inference run in separate processes. The supervisor records
child PIDs and exit codes, keeps the system awake while active, and releases
the request when finished or failed. Training failures stop the pipeline.

After three epochs, export the EMA checkpoint with validation metrics set to
null. Independently reload it, predict 2,218 test images and average probabilities
within 1,414 studies. Write ten rows per study (14,140 rows), Gate=0.5. Validate
exact IDs, class coverage, unique rows, finite probabilities in [0,1], and exact
CSV probability roundtrip. Keep the earlier submitted file intact.

## Interpretation

The previous 0.8270335041 audit Macro-AUC belongs to the development-only
checkpoint. It is not the full-refit score and must not be copied into the
new checkpoint as a measured result. The new test set has no local labels.
Full-data weights are ineligible for development OOF prediction or for claims
of an independent result on the former audit cohort.

Further model research must return to the existing development folds with
fold-specific encoders. The next structural hypothesis is study-level feature
pooling; it is separate from this deployment refit.
