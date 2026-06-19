---
name: ideate-features
description: >-
  Build a prioritized feature-hypothesis backlog for entity classification
  (predict a node's type/segment) on Spark graph/transaction tables. Trigger
  whenever the user names tables + a GT label and wants what predicts the
  segment, asks "what features" / "where's the signal", or starts a modeling
  task with no features. Profiles tables on a sample (never full pulls), tags
  each feature by scale, checks leakage. Backlog only — no compute.
---

# ideate-features

Output: `features/backlog_<task>.md` — a data-map paragraph + a ranked table of
candidates (name, family, why-it-separates-classes, Spark compute sketch, scale
tag, leakage check, priority) + the 5–8 to build first.

Principle: entity class = fingerprint. Favor structural role, behavioral *shape*
(ratios / entropy / CV, not raw totals), and train-only neighbor class mix.

## Steps

1. **Get + profile.** From the user: table names, entity + classes, prediction
   cutoff. Then run (never pull full tables to pandas):
   `python "${CLAUDE_SKILL_DIR}/scripts/profile_spark_tables.py" T1 T2 ... --keys --remote sc://HOST:PORT`
   (or set SPARK_REMOTE; if no session it prints a notebook cell — use that.)
   For local files (parquet/csv) instead of catalog tables, use `scripts/inspect_tables.py PATH` (pandas).

2. **Infer + CONFIRM.** State back, and get a yes before ideating: which table is
   edges / entities / labels; the join keys (edge keys often differ from the
   entity key — e.g. src_id/dst_id vs entity_id; wrong key = garbage); grain;
   time column; cutoff. Frame all compute via the DAL, not raw reads.

3. **Enumerate** candidates in real columns across these families, each tagged by
   scale:
   - `spark-native`: degree, txn count/sum/mean, amount CV/skew, RFM, in÷out
     ratio, inter-event timing, counterparty entropy, neighbor aggregation.
   - `graphframes`: PageRank, connected components, label prop, triangles, k-core.
     (Spark Connect can't run GraphFrames from the CLI — reformulate as iterative
     DataFrame joins, or run cluster-side.)
   - `sampled-egonet`: betweenness, struc2vec/RolX roles, communities,
     node2vec/GraphSAGE. Sample subgraph over Connect, run local lib on driver.
   Cover: node aggregates, topology/role, neighborhood (incl. train-only
   labeled-neighbor class mix + label propagation), embeddings, multi-relational,
   counterparty diversity.

4. **Leakage-check each row:**
   - time fence: only data with `timestamp < cutoff`; no aggregate spans it.
   - label-derived (neighbor labels, label prop, centroids): train-fold only.
   - never feature-ize the GT.
   - flag group leakage (connected / repeat entities across splits).

5. **Rank** by signal ÷ cost (promote role, behavioral-shape, train-only
   neighborhood). Write the file, report the path, name the first 5–8.

Backlog only. If asked to compute or score features, that's the next stage — stop
unless the user confirms.
