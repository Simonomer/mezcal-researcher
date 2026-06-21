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

Predict a person's `home_city` across **50 cities — but with ground truth for only
30** of them (a normal-shaped, heavily imbalanced label; the other 20 cities'
residents appear only as *unlabeled* graph/text context). The signal lives mostly
in the **text people exchange**; supporting tables add graph, geography, sparse
**documents**, and noise. It is deliberately hard and **messy**: ~60% homophily,
~12% travelers, 50 cities in 10 confusable metros, ~5% label noise, nulls,
duplicate rows/edges, inconsistent city casing/typos, mixed types — plus three
leak traps (`ip_geo` ≈ the answer, self-reported `raw_city_text`, region-joined
`weather`).

## How to run it

Same loop everywhere — **profile → ideate → materialize → validate → (loop back)**.

### A. Real world: your Spark tables (catalog, HDFS, or S3)

The skills read a catalog table **or** a path on HDFS / S3 / S3A / GCS / DBFS;
heavy work stays on the cluster, only a bounded sample reaches the driver.

```bash
pip install "pyspark[connect]" pandas scikit-learn scipy matplotlib pyarrow
export SPARK_REMOTE=sc://YOUR-HOST:15002          # your Spark Connect endpoint

# 1. Ideate — profile catalog tables OR paths, confirm wiring, get the backlog
python scripts/profile_spark_tables.py warehouse.messages warehouse.people \
    warehouse.home_city warehouse.tower_pings ... --keys          # or s3://bucket/messages …

# 2. Materialize on the cluster (text extraction, OOF neighbor) -> a feature table
SOURCE=warehouse python features/build_features_spark.py          # SOURCE may be s3://bucket/db

# 3. Validate — joins cluster-side, stratified sample to the driver
python scripts/signal_panel.py \
    --features-table warehouse.home_city_features --labels-table warehouse.home_city \
    --id-col person_id --label-col home_city --perm 25 --out validation/report.md
#   --features-table also accepts s3://… / hdfs://… paths (--format to override)
```

> **In-notebook (non-Connect) `spark`?** A script Claude launches is a separate
> process and can't attach to your kernel's session — expose a Spark Connect
> endpoint, or run the profiler's emitted cell in your notebook and `saveAsTable`
> the feature/label tables so the harness can read them.

### B. The reproducible demo in this repo (local, no cluster)

```bash
pip install pandas numpy scikit-learn scipy matplotlib pyarrow
python data/generate_data.py                          # 13 messy tables -> data/tables/
python scripts/inspect_tables.py data/tables          # profile (grounds ideation)
python features/build_features.py                     # -> features/feature_table.parquet (41 feats)
python -X utf8 scripts/signal_panel.py \              # screen (─X utf8: Windows console)
  --features-file features/feature_table.parquet \
  --labels-file data/tables/home_city.parquet \
  --id-col person_id --label-col home_city --time-col ref_ts --perm 25 \
  --out validation/report.md
```

## Results

The screen runs twice — on the full table, and after the closed loop drops the
three flagged leaks:

| run | features | baseline macro-F1 | macro-AUC | keep / investigate / drop |
|---|---|---|---|---|
| **Full table** (leaks present) | 41 | **0.091** | 0.526 | 10 / 7 / 24 |
| **Screened** (leaks removed)   | 38 | **0.845** | 0.830 | 9 / 6 / 23 |

The headline is the **0.091 → 0.845** jump. Leakage is usually framed as *inflating*
a score; here the high-cardinality leak columns (`ip_city`, `declared_city`)
*wrecked* a naive 30-class baseline — so "train on everything and read the
importances" gives a misleadingly hopeless 0.091. The aggregate model score lies
in both directions; the **per-feature screen** tells the truth and says what to cut.

**What the harness did, with the human only confirming wiring and ruling on flags:**

- **Found the signal feature-by-feature** even though the all-features model failed:
  MI ranks `nb_modal_city` (2.61), `merch_modal_city` (2.49), `ip_city` (2.21),
  `tower_modal_region` (2.00), `txt_flavor_top_city` (1.72) at the top.
- **Flagged the leaks.** A **normalized-MI** test (MI ÷ label entropy) — added to
  catch *categorical* leaks an AUC test can't see — flagged `nb_modal_city` (0.86)
  and `merch_modal_city` (0.82); `weather` tripped the numeric **AUC** flag (0.997);
  `ip_city`/`declared_city` topped the MI ranking for the human to cut.
- **Rejected all 24 red-herrings** (device, app telemetry, age, message-style
  stats, document length, raw degree) against the permutation null.

See [`validation/report.md`](validation/report.md) and
[`validation/screened/report.md`](validation/screened/report.md).

## Repo layout

| path | what |
|---|---|
| [`data/generate_data.py`](data/generate_data.py) | 13-table generator: text, docs, graph, geo, messy, 3 leak traps |
| [`features/backlog_homecity.md`](features/backlog_homecity.md) | the LLM's ranked feature-hypothesis backlog |
| [`features/build_features.py`](features/build_features.py) | materialization (local/pandas): text + OOF train-only neighbor (partial labels) |
| [`features/build_features_spark.py`](features/build_features_spark.py) | **real-world** Spark/Connect materialization (reads catalog/HDFS/S3) |
| [`scripts/inspect_tables.py`](scripts/inspect_tables.py) | local schema profiler (parquet/CSV) |
| [`scripts/profile_spark_tables.py`](scripts/profile_spark_tables.py) | Spark/Connect profiler — catalog tables **or** HDFS/S3 paths |
| [`scripts/signal_panel.py`](scripts/signal_panel.py) | screening harness (local files or Spark tables/paths; native categorical; AUC + normalized-MI leak flags) |
| [`validation/report.md`](validation/report.md) | per-feature signal report + figures |
| [`article/`](article) | the method article (markdown + landscape PDF) |

## Notes on the validation command

- `--time-col ref_ts` enables the stability panel; **omit `--group-col`** (feeding
  the `home_city` label to `GroupKFold` would hold out whole classes).
- On Windows run the harness as `python -X utf8 …` (it writes an em-dash that
  crashes cp1252).
- Generated parquet (`data/tables/`, `features/feature_table.parquet`) is
  gitignored — regenerable, deterministic (seed 11).
