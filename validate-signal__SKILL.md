---
name: validate-signal
description: >-
  Score materialized candidate features for real signal about a multiclass label
  and write a markdown report with charts. Trigger whenever the user has a built
  feature table + labels and asks which features carry signal, "is this feature
  any good", "screen/validate my features", "keep or drop", or has just
  materialized a backlog from ideate-features. Runs a deterministic harness
  (permutation null, effect size, baseline-model importance, time stability) all
  under leakage-safe splits. Screening only — not final model evaluation.
---

# validate-signal

Output: `validation/report.md` + `validation/figures/*.png` — per-feature table
(MI, permutation-null, beats-null, best one-vs-rest AUC, baseline importance,
coverage, time stability) with a keep / investigate / drop call and reasons,
plus baseline macro-F1/AUC and a confusion matrix.

Signal = beats a shuffled-label null (not chance) + effect size + adds value in a
model that has the other features (incremental) + stable across folds/time. No
single number is trusted; the script layers all of these. Screening only — the
final word is full-model out-of-sample performance.

## Steps

1. **Get inputs.** Feature table + label table. Spark `--features-table` /
   `--labels-table` accept a catalog table (`matcha.feat`) OR a path on HDFS /
   S3 / S3A / GCS / DBFS (`s3://bucket/feat`, `hdfs://…`, `/lake/feat/*.parquet`;
   `--format` to override). Or local `--features-file` / `--labels-file`. Plus
   `--id-col`, `--label-col`, and if available `--time-col` (stability) and
   `--group-col` (grouped CV, e.g. entity/connected groups).

2. **Run the harness** (it does all the statistics — do not compute metrics
   yourself):
   ```bash
   python "${CLAUDE_SKILL_DIR}/scripts/signal_panel.py" \
       --features-table T_FEAT --labels-table T_LAB \   # tables OR s3://… / hdfs://… paths
       --id-col ID --label-col LABEL [--time-col TS] [--group-col GID] \
       [--remote sc://HOST:PORT] [--format parquet] --out validation/report.md
   ```
   Over Spark Connect it samples + joins as DataFrame ops, pulls a stratified
   sample to the driver for the sklearn/scipy metrics. Discrete columns are used
   as NATIVE categorical features in the baseline model (ids / text-derived
   categories, not arbitrary codes).

3. **Summarize for the user.** Read `validation/report.md`. Report keep /
   investigate / drop counts, the baseline macro-F1/AUC, and call out: features
   that didn't beat the null (noise), redundant pairs, unstable-over-time
   features, and any "suspiciously strong" feature — flag to investigate, not
   celebrate. Two leak detectors: high one-vs-rest **AUC** (numeric) and high
   **mi_ratio** = MI ÷ label-entropy (catches near-deterministic *categorical*
   leaks like an id/self-report column that nearly equals the label).

Screening only. Feed the kept features into the model; loop the dropped/weak ones
back to ideate-features for a fresh batch.
