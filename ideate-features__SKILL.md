---
name: ideate-features
description: >-
  Build a prioritized feature-hypothesis backlog for entity classification
  (predict a node's type/segment) on Spark graph/transaction tables. Trigger
  when the user names tables + a GT label and asks what predicts the segment,
  asks "what features" / "where's the signal", or starts a modeling task with
  no features. Profiles tables on a sample (never full pulls). Backlog only —
  no compute.
---

# ideate-features

Output file: `features/backlog_<task>.md`. It contains:
- one data-map paragraph (which table is edges/entities/labels, join keys, grain).
- a ranked candidate table with columns: name, family, why-it-separates-classes, Spark compute sketch, scale tag, leakage check, priority.
- the 5–8 candidates to build first.

Principle: entity class = fingerprint. Favor structural role and behavioral *shape* (ratios / entropy / CV, not raw totals) and train-only neighbor class mix.

## Steps

1. **Ask the user** for: table names; the entity and its classes; the prediction cutoff (time column + cutoff value). Do not guess these.

2. **Profile the tables on a sample — never pull full tables.** Pick ONE command.

   If the tables are Spark catalog tables OR paths on HDFS/S3/S3A/GCS/DBFS, run:
   ```
   python "${CLAUDE_SKILL_DIR}/scripts/profile_spark_tables.py" T1 T2 --keys --remote sc://HOST:PORT
   ```
   - Each `T` is a catalog table (`matcha.t1`) OR a path (`s3://bucket/t1`, `hdfs://…`, `/lake/t1/*.parquet`). Format is inferred from the extension; override with `--format parquet|csv|json`.
   - Flags: `--sample N` (sample rows/table, default 100000), `--count` (exact full row count), `--keys` (approx-distinct on key-like columns), `--format`, `--remote sc://host:port`.
   - If you omit `--remote`, it uses `SPARK_REMOTE`. If there is no session, the script prints a notebook cell — run that cell instead.

   If the tables are purely local parquet/csv files, run:
   ```
   python "${CLAUDE_SKILL_DIR}/scripts/inspect_tables.py" PATH1 PATH2
   ```
   - Each path is a data dir or table file. Only flag: `--nrows N` (max CSV rows to sample, default 50000; `0` = all).

3. **CONFIRM the wiring before ideating (GATE).** State back to the user, and get an explicit YES:
   - which table is edges, which is entities, which is labels;
   - the join keys. WARNING: edge keys often differ from the entity key (e.g. `src_id`/`dst_id` vs `entity_id`). Wrong key = garbage.
   - grain; time column; cutoff value.
   Do not ideate until the user confirms.

4. **Enumerate candidates** over the real columns. Tag each with a scale family:
   - `spark-native`: e.g. degree, amount CV/entropy, in÷out ratio, neighbor aggregation.
   - `graphframes`: e.g. PageRank, connected components, label propagation. NOTE: Spark Connect cannot run GraphFrames from the CLI — reformulate as iterative DataFrame joins, or run cluster-side.
   - `sampled-egonet`: e.g. betweenness, struc2vec/RolX roles, node2vec. NOTE: sample the subgraph over Connect, run the local graph lib on the driver.
   Cover: node aggregates; topology/role; neighborhood (incl. train-only labeled-neighbor class mix + label propagation); embeddings; multi-relational; counterparty diversity.

5. **Leakage-check every candidate** (mark the result in the leakage-check column):
   - Time fence: use only data with `timestamp < cutoff`. No aggregate may span the cutoff.
   - Label-derived features (neighbor labels, label propagation, centroids): TRAIN-FOLD ONLY (out-of-fold / train-only neighbor labels).
   - Never feature-ize the GT label itself.
   - Flag group leakage (connected / repeat entities across splits).

6. **Rank** candidates by signal ÷ cost. Promote structural role, behavioral shape, and train-only neighborhood features. Write `features/backlog_<task>.md`, report the path, and name the first 5–8 to build.

STOP: backlog only. If asked to compute or score features, that is the next stage — stop unless the user confirms.
