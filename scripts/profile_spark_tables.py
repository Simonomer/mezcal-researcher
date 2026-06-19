#!/usr/bin/env python3
"""Spark-native grounding profiler for ideate-features.

Profiles catalog tables (e.g. matcha.t123) on a bounded sample — no full pulls.
Prints per-column dtype / null% / cardinality / sample, timestamp flags, and
cross-table join keys. Session: active -> getOrCreate -> --remote / SPARK_REMOTE
(sc://...); if none, prints a notebook cell to run in your live session.

Usage: python profile_spark_tables.py matcha.t1 matcha.t2 --keys --remote sc://host:15002
"""

import argparse
import os
import sys


def get_spark(remote=None):
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None, "pyspark not importable (Spark Connect needs pyspark[connect])"
    active = SparkSession.getActiveSession()
    if active is not None:
        return active, None
    remote = remote or os.environ.get("SPARK_REMOTE")
    try:
        builder = SparkSession.builder.appName("ideate-features-profile")
        if remote:
            builder = builder.remote(remote)  # Spark Connect: sc://host:port
        return builder.getOrCreate(), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def looks_like_time(name, dtype, sample_vals):
    if dtype in ("timestamp", "date") or "timestamp" in dtype or "date" in dtype:
        return True
    if any(k in name.lower() for k in ("time", "date", "ts", "_at")):
        try:
            import pandas as pd
            pd.to_datetime(pd.Series([v for v in sample_vals if v is not None][:50]),
                           errors="raise")
            return True
        except Exception:  # noqa: BLE001
            return False
    return False


def fmt_sample(vals, k=3):
    out = []
    for v in vals:
        if v is None:
            continue
        s = str(v)
        out.append(s if len(s) <= 24 else s[:21] + "...")
        if len(out) >= k:
            break
    return ", ".join(out) if out else "—"


def profile_one(spark, name, sample_rows, do_count, do_keys):
    from pyspark.sql import functions as F
    try:
        df = spark.table(name)
    except Exception as e:  # noqa: BLE001
        print(f"\n## {name}\n\n> could not read table: {e}\n")
        return name, set()

    dtypes = dict(df.dtypes)
    # Bounded sample pulled to the driver — limit() keeps it cheap (no full scan).
    pdf = df.limit(sample_rows).toPandas()
    n_s = len(pdf)

    print(f"\n## {name}")
    line = f"\nsample rows: {n_s:,} (stats below are on this sample)"
    if do_count:
        try:
            line += f"  ·  full row count: {df.count():,}"
        except Exception as e:  # noqa: BLE001
            line += f"  ·  full count failed: {e}"
    print(line)

    time_cols, key_cols = [], set()
    print("\n| column | dtype | null % (sample) | cardinality (sample) | sample values |")
    print("|---|---|---|---|---|")
    for col in pdf.columns:
        s = pdf[col]
        dtype = dtypes.get(col, str(s.dtype))
        null_pct = 100.0 * s.isna().mean() if n_s else 0.0
        nuniq = s.nunique(dropna=True)
        svals = list(s.dropna().unique())
        is_time = looks_like_time(col, dtype, svals)
        if is_time:
            time_cols.append(col)
        if (not is_time and null_pct < 5 and n_s > 0
                and nuniq / max(n_s, 1) > 0.30
                and ("int" in dtype or "string" in dtype or s.dtype == object)):
            key_cols.add(col)
        flag = " ⏱" if is_time else ""
        print(f"| {col}{flag} | {dtype} | {null_pct:.1f} | {nuniq:,} | {fmt_sample(svals)} |")

    if time_cols:
        print(f"\nlikely time columns: {', '.join(time_cols)}")

    if do_keys and key_cols:
        # full-table approx distinct only for the few key-like columns — cheap & distributed
        try:
            exprs = [F.approx_count_distinct(c).alias(c) for c in key_cols]
            row = df.agg(*exprs).collect()[0].asDict()
            print("\napprox distinct (full table) for key-like columns: "
                  + ", ".join(f"{c}={row[c]:,}" for c in key_cols))
        except Exception as e:  # noqa: BLE001
            print(f"\n> approx-distinct skipped: {e}")
    return name, key_cols


def notebook_bridge_cell(names):
    joined = ", ".join(f'"{n}"' for n in names)
    return f'''
# --- run this in your live (Spark-connected) notebook, then share the output ---
from pyspark.sql import functions as F
for name in [{joined}]:
    df = spark.table(name)
    print("##", name)
    df.printSchema()
    df.limit(5).show(truncate=40)
    # light stats on a sample:
    s = df.limit(100000)
    print("sample rows:", s.count())
# Optionally write to a file the skill can read:
# (e.g. dump schemas to /tmp/schema_dump.txt and point the skill at it)
'''


def main():
    ap = argparse.ArgumentParser(description="Spark-native profiler for feature ideation.")
    ap.add_argument("tables", nargs="+", help="catalog table names, e.g. matcha.table123")
    ap.add_argument("--sample", type=int, default=100000, help="sample rows per table (default 100000)")
    ap.add_argument("--count", action="store_true", help="also compute exact full row count (full scan)")
    ap.add_argument("--keys", action="store_true", help="approx-distinct on key-like columns over full table")
    ap.add_argument("--remote", default=None,
                    help="Spark Connect URL, e.g. sc://host:15002 (else uses SPARK_REMOTE)")
    args = ap.parse_args()

    spark, err = get_spark(args.remote)
    if spark is None:
        print("# No Spark session available", file=sys.stderr)
        print(f"# reason: {err}", file=sys.stderr)
        print("# Spark Connect: pass --remote sc://host:15002 (or set SPARK_REMOTE) and", file=sys.stderr)
        print("# install pyspark[connect]. Otherwise use notebook-bridge mode below:")
        print(notebook_bridge_cell(args.tables))
        sys.exit(2)

    print(f"# Table inventory ({len(args.tables)} tables) — Spark-native, sampled")
    keys_by_table = {}
    for name in args.tables:
        tname, keys = profile_one(spark, name, args.sample, args.count, args.keys)
        keys_by_table[tname] = keys

    shared = {}
    for name, keys in keys_by_table.items():
        for k in keys:
            shared.setdefault(k, []).append(name)
    edges = {k: v for k, v in shared.items() if len(v) > 1}

    print("\n## Candidate join keys (the edges of your graph)")
    if edges:
        print("\n| key column | appears in |")
        print("|---|---|")
        for k, names in sorted(edges.items()):
            print(f"| {k} | {', '.join(names)} |")
    else:
        print("\n> No shared key *names* detected. Edge keys are often named "
              "differently from the entity key (e.g. src_id/dst_id vs entity_id) "
              "— confirm the wiring manually before ideating.")


if __name__ == "__main__":
    main()
