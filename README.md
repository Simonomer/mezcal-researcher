# mezcal-researcher — automatic feature ideation with LLMs

A worked, end-to-end demonstration of a **closed-loop method for exploratory data
analysis**: an LLM enumerates a feature-hypothesis backlog grounded in the real
schema, a deterministic harness screens every candidate for real signal, results
loop back into re-ideation, and the human stays at just two gates — **confirm the
data wiring** and **set the keep threshold**.

📄 **Read the article:**
[`article/automatic-feature-ideation-with-llms.md`](article/automatic-feature-ideation-with-llms.md)
· [PDF (landscape)](article/automatic-feature-ideation-with-llms.pdf)

## The problem

Predict each civilian's `home_city` (5 imbalanced classes) from a communication
graph plus nine auxiliary tables. It is deliberately **hard**: noisy homophily
(~63% of contacts share your city), ~12% travelers, confusable sister-cities
(A↔B, C↔D), overlapping language/behavior, and two kinds of leakage traps — a
per-region `weather_logs` table that can only be joined to a civilian *through
their home region* (the answer), and near-decisive location features (tower
region, merchant city) that look like leaks but are legitimate.

## How to run it

The loop is the same everywhere — **profile → ideate → materialize → validate →
(loop back)** — and each step's output is the next step's input. Pick your setup:

### A. Real world: your Spark tables (Spark Connect)

Your tables already exist, so you start at *ideate*. Point the skills at catalog
table names and a Spark Connect URL; heavy work stays on the cluster and only a
**bounded sample** is pulled to the driver.

```bash
pip install "pyspark[connect]" pandas scikit-learn scipy matplotlib pyarrow
export SPARK_REMOTE=sc://YOUR-HOST:15002          # your Spark Connect endpoint
```

**1 · Ideate** — in the Claude Code chat box, name your catalog tables:

```
/ideate-features predict home_city for civ_id;
  tables telco.comms telco.civilians telco.home_city telco.tower_pings
         telco.towers telco.transactions telco.merchants telco.app_events
         telco.device_info telco.weather_logs
```

→ runs `profile_spark_tables.py … --keys` (≈100k-row sample per table, profiled
**cluster-side**), proposes the wiring, you confirm in plain text → writes
[`features/backlog_homecity.md`](features/backlog_homecity.md).

**2 · Materialize** — build the feature table in Spark and write it to the
catalog. Adapt [`features/build_features_spark.py`](features/build_features_spark.py)
(the Spark counterpart of the pandas builder — same leakage-safe out-of-fold
neighbor encoding) and run it in your notebook or via the Connect client:

```python
feat.write.mode("overwrite").saveAsTable("telco.home_city_features")
```

**3 · Validate** — joins happen cluster-side; only a stratified sample comes to
the driver for the sklearn/scipy metrics:

```bash
python scripts/signal_panel.py \
  --features-table telco.home_city_features --labels-table telco.home_city \
  --id-col civ_id --label-col home_city --time-col ref_ts --perm 40 \
  --out validation/report.md            # SPARK_REMOTE is picked up automatically
```

> **In-notebook (non-Connect) `spark`?** A script Claude launches runs in a
> separate process and can't attach to your kernel's session. Either expose a
> Spark Connect endpoint (then the above just works), or run the profiler's
> emitted cell in your notebook and `saveAsTable` your feature/label tables so the
> harness can read them. (Connection logic: `getActiveSession()` → `--remote` /
> `$SPARK_REMOTE` → otherwise it prints a notebook cell.)

### B. The reproducible demo in this repo (local, no cluster)

No Spark needed — generate a synthetic dataset and run the whole loop on local
parquet (this is what produced the results below):

```bash
pip install pandas numpy scikit-learn scipy matplotlib networkx pyarrow
python data/generate_data.py                          # 1. -> data/tables/*.parquet
python scripts/inspect_tables.py data/tables          # 2a. profile (grounds ideation)
#    -> features/backlog_homecity.md                  # 2b. the LLM backlog
python features/build_features.py                     # 3. -> features/feature_table.parquet
python scripts/signal_panel.py \                      # 4. screen
  --features-file features/feature_table.parquet \
  --labels-file data/tables/home_city.parquet \
  --id-col civ_id --label-col home_city --time-col ref_ts --perm 40 \
  --out validation/report.md
```

**Local vs Spark:** `inspect_tables.py` (local) reads files on disk — CSV by its
first `--nrows` rows, parquet in full. `profile_spark_tables.py` and the validator
take catalog tables + `--remote`/`$SPARK_REMOTE` and never do a full pull. On
Windows, run the validator as `python -X utf8 …` (it emits an em-dash that crashes
cp1252).

## Results

The screen runs twice — once on the full table (with the planted leak) and once
after the closed loop drops it — to separate "looks perfect" from "is real."

| run | features | baseline macro-F1 | macro-AUC | keep / investigate / drop |
|---|---|---|---|---|
| **Full table** (leak present) | 44 | **1.000** | 1.000 | 16 / 15 / 13 |
| **Screened** (leak removed) | 42 | **0.934** | 0.994 | 16 / 13 / 13 |

The leaked `macro-F1 = 1.000` looks perfect — and is a lie. The honest baseline is
**0.934**, with the residual error landing exactly on the built-in confusable
city-pairs (A↔B, C↔D) while the isolated class E stays clean.

**What the harness did, with the human only confirming wiring and ruling on flags:**

- **Caught the planted leak.** The two `weather_logs`-derived features scored a
  one-vs-rest **AUC of 1.000** and were the *only* features with non-zero model
  importance (every legitimate feature read 0.000 — the tell that one feature
  explains everything) → flagged *investigate*.
- **Rejected all 13 red-herrings.** Every `device_info` column, app-telemetry
  volume, `age`, and raw degree/spend counts failed to beat the permutation null
  → *drop*.
- **Flagged 9 near-decisive features** (the train-only neighbor and tower fractions,
  AUC 0.97–1.0) and a **redundant pair** (`deg_total` ↔ `contact_entropy`) for a
  human to bless or cut — shrinking the adjudication set from 44 to 15.

See [`validation/report.md`](validation/report.md) (full table) and
[`validation/screened/report.md`](validation/screened/report.md) (closed-loop
re-screen, leak removed).

## Repo layout

| path | what |
|---|---|
| [`data/generate_data.py`](data/generate_data.py) | synthetic 10-table generator (hard, with leakage traps) |
| [`data/tables/`](data/tables) | parquet output |
| [`features/backlog_homecity.md`](features/backlog_homecity.md) | the LLM's ranked feature-hypothesis backlog |
| [`features/build_features.py`](features/build_features.py) | materialization (local/pandas); OOF train-only neighbor encoding |
| [`features/build_features_spark.py`](features/build_features_spark.py) | **real-world** Spark/Connect materialization template (writes a catalog table) |
| [`scripts/inspect_tables.py`](scripts/inspect_tables.py) | local schema profiler (parquet/CSV) |
| [`scripts/profile_spark_tables.py`](scripts/profile_spark_tables.py) | Spark/Connect schema profiler (catalog tables, bounded sample) |
| [`scripts/signal_panel.py`](scripts/signal_panel.py) | the deterministic screening harness (local files **or** Spark tables) |
| [`validation/report.md`](validation/report.md) | per-feature signal report + figures |
| [`article/`](article) | the method article (markdown + PDF) |

## Note on the validation command

The harness's `--group-col` feeds the column to `GroupKFold`. Passing the **label**
(`home_city`) as the group would hold out an entire class per fold and collapse
the baseline; this problem has one row per `civ_id` and no connected-entity
groups, so the default `StratifiedKFold` is the correct, leakage-safe split and
`--group-col` is omitted. `--time-col ref_ts` (account-creation time) enables the
stability panel.
