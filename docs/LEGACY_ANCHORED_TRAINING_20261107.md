# Train from the existing ensemble — 2026-11-07

The user requested training from the old fusion after reporting online scores
78.53 for that fusion and 77.34 for the later single model. These are user
reports, not independently retrieved backend results. Preserve both files.

## Fixed scope before fitting

Use the exact recipe corresponding to
`submission_test_1414_best_20261105.csv`, not the later six-source experiment.
Keep its local ResNet18 and fine-tuned Rad-DINO contributions (including the
distinct original and flip-averaged channels) fixed. Those models were trained
on the full local training set, so their in-sample predictions must not become
features or offsets for this weight-fitting experiment.

Only retrain the public-source subensemble, using the five frozen public
checkpoints already in the old model. Their saved development and audit
predictions jointly cover the entire training manifest. Both cohorts are now
training data for this experiment; neither is a new unopened audit. Verify
cache hashes, model fingerprints, IDs, targets and class mappings before use.
Public pretraining overlap with these patients remains unknown.

For each class, let `s` be the old public weight mass and `q0` its normalized
public weights. Keep `s` fixed. Fit
`q = 0.8*q0 + 0.2*v`, where `v` is a nonnegative simplex over the existing
public sources with that class available. Initialize `v=q0`. Classes with
`s=0` remain bitwise unchanged. Do not change the local-model coefficients.

The objective is the mean sampled positive-negative logistic ranking loss on
the normalized public subensemble plus `0.5 * 1.0 * ||q-q0||^2`. Fix temperature
0.1, 32,768 pairs per active class, seed 20261107, maximum 300 projected
gradient iterations and tolerance 1e-8. No hyperparameter or model search.

This public-only ranking objective is not the ranking objective of the full
ensemble, whose fixed local contribution also changes the ordering. This is
the deliberate cost of avoiding local in-sample predictions during fitting.

## Patient cross-fitting and final fit

Assign all training patients to three folds by sorting SHA-256 of
`20261107:subject_id` and distributing them round-robin. Each fold trains the
public weights on the other patients. Report only public-subensemble metrics
on active classes; these are not full-ensemble OOF metrics. No local backbone
is retrained. Fit the same fixed algorithm once more on the entire training
set, save the recipe, and hash it before opening validation labels.

## Fixed comparison and delivery gate

Use the saved official-valid scores of the old ensemble as the fixed anchor.
Apply only the public weight delta to that anchor. Compare both on the same
859 studies / 228 patients. No per-class fallback, coefficient search,
threshold tuning or parameter changes after seeing the result.

Create a separate candidate test CSV only if all ten AUCs are defined,
the same-validation Macro-AUC gain is at least 0.001, and the lower endpoint
of a descriptive paired patient-bootstrap 95% interval is above zero
(1,000 repeats, seed 20261107). Otherwise retain the old submission and report
the rejected experiment. The old recipe used these validation labels during
selection; this gate and interval do not provide unbiased significance or a
guaranteed hidden-test improvement.

Verify the frozen old test recipe replays its saved matrix and delivered CSV
before applying a new recipe. Export a fresh filename with exact ID/class
coverage, finite probabilities, Gate=0.5 and exact numeric roundtrip. Never
overwrite either earlier submitted CSV. All model sources remain recorded.
