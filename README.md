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

## Pipeline — four steps, four artifacts

```bash
pip install pandas numpy scikit-learn scipy matplotlib networkx pyarrow

# 1. Generate the 10-table dataset  ->  data/tables/*.parquet
python data/generate_data.py

# 2. Ideate: profile + the LLM backlog (human confirms wiring)
python scripts/inspect_tables.py data/tables          # schema grounding
#  -> features/backlog_homecity.md  (ranked feature hypotheses)

# 3. Materialize the kept rows (leakage-safe)  ->  features/feature_table.parquet
python features/build_features.py

# 4. Screen every feature for signal  ->  validation/report.md + figures
python scripts/signal_panel.py \
  --features-file features/feature_table.parquet \
  --labels-file data/tables/home_city.parquet \
  --id-col civ_id --label-col home_city --time-col ref_ts --perm 40 \
  --out validation/report.md
```

Each step's output is the next step's input:
**generate → ideate → materialize → validate → (loop back)**.

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
| [`features/build_features.py`](features/build_features.py) | materialization; OOF train-only neighbor encoding |
| [`scripts/inspect_tables.py`](scripts/inspect_tables.py) | schema profiler (grounds ideation) |
| [`scripts/signal_panel.py`](scripts/signal_panel.py) | the deterministic screening harness |
| [`validation/report.md`](validation/report.md) | per-feature signal report + figures |
| [`article/`](article) | the method article (markdown + PDF) |

## Note on the validation command

The harness's `--group-col` feeds the column to `GroupKFold`. Passing the **label**
(`home_city`) as the group would hold out an entire class per fold and collapse
the baseline; this problem has one row per `civ_id` and no connected-entity
groups, so the default `StratifiedKFold` is the correct, leakage-safe split and
`--group-col` is omitted. `--time-col ref_ts` (account-creation time) enables the
stability panel.
