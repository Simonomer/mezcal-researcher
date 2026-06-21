---
name: validate-signal
description: >-
  Score materialized candidate features for real signal about a multiclass
  label, then write a markdown report with charts. Trigger when the user has a
  feature table + labels and asks which features carry signal, "is this feature
  any good", "keep or drop", or just materialized a backlog from
  ideate-features. Screening only — not final model evaluation.
---

# validate-signal

Screen features for signal about a multiclass label. The script does ALL the
statistics under leakage-safe splits — do not compute metrics yourself.

Signal = beats a shuffled-label null + effect size (MI / best one-vs-rest AUC) +
adds value in a baseline model that has the other features (incremental) + stable
across folds/time. No single number is trusted.

Leakage-safe by construction: train-only neighbor labels via out-of-fold
(`cross_val_predict` over StratifiedKFold, or GroupKFold with `--group-col`); a
time fence via `--time-col` (stability across time slices); the label column is
never feature-ized. Discrete columns are used as NATIVE categorical features in
the baseline model.

## Steps

1. **Collect inputs.** Need a feature source, a label source, plus `--id-col`
   and `--label-col` (both required). Add `--time-col` for time stability and
   `--group-col` for grouped CV when available.
   - Tables/paths: `--features-table` / `--labels-table` accept a catalog table
     (`matcha.feat`) OR a path on HDFS / S3 / S3A / GCS / DBFS
     (`s3://bucket/feat`, `hdfs://...`, `/lake/feat/*.parquet`). Use `--format`
     to override the read format. Set the Spark endpoint via `--remote
     sc://HOST:PORT` (or the `SPARK_REMOTE` env var / an active session).
   - Local files: `--features-file` / `--labels-file`.

2. **Run the harness:**
   ```bash
   python "${CLAUDE_SKILL_DIR}/scripts/signal_panel.py" \
       --features-table T_FEAT --labels-table T_LAB \
       --id-col ID --label-col LABEL \
       [--features-file F.parquet --labels-file L.parquet] \
       [--time-col TS] [--group-col GID] \
       [--remote sc://HOST:PORT] [--format parquet] \
       [--features COL1 COL2] [--sample 200000] [--perm 30] \
       --out validation/report.md
   ```
   Output: `validation/report.md` + `validation/figures/*.png` (figures dir is
   derived from `--out`; there is no `--figdir` flag). The report is a per-feature
   keep / investigate / drop table with reasons, plus baseline macro-F1/AUC and a
   confusion matrix.

3. **Summarize.** Read `validation/report.md`. Report keep / investigate / drop
   counts, baseline macro-F1/AUC, and the two leak flags:
   - one-vs-rest **AUC > 0.97** (numeric) -> investigate: suspiciously strong.
   - **mi_ratio** = MI ÷ label-entropy **> 0.80** (categorical) -> investigate:
     explains most of the label (near-deterministic id/self-report leak).
   If a feature is flagged, say investigate — do not celebrate it.

4. **Hand off.** Screening only — the final word is full-model out-of-sample
   performance. Feed the KEPT features into the model. Loop the dropped/weak ones
   back to ideate-features for a fresh batch.
