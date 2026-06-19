#!/usr/bin/env python3
"""
inspect_tables.py — schema/grounding profiler for the ideate-features skill.

Profiles CSV / TSV / Parquet tables so feature ideation is grounded in real
columns instead of guesses. Prints, per table: row count, every column with
dtype / null% / cardinality / sample values, flags likely timestamp columns,
and lists candidate join keys shared across tables (the edges of your graph).

Usage:
    python inspect_tables.py <path>...
    # <path> can be a directory (scanned for .csv/.tsv/.parquet) or individual files

Examples:
    python inspect_tables.py ./data
    python inspect_tables.py edges.parquet entities.csv labels.csv

Notes:
    - Large CSVs are sampled (first --nrows rows, default 50000) for speed.
    - Parquet needs pyarrow or fastparquet installed.
    - Output is markdown; pipe to a file or let the skill read it directly.
"""

import argparse
import os
import sys
import glob

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas (and pyarrow for parquet)")

TABLE_EXTS = (".csv", ".tsv", ".parquet")


def find_tables(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in TABLE_EXTS:
                files.extend(sorted(glob.glob(os.path.join(p, f"*{ext}"))))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"> skipping (not found): {p}", file=sys.stderr)
    # de-dup, preserve order
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def load(path, nrows):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)  # parquet: no cheap row cap, read full
    sep = "\t" if ext == ".tsv" else ","
    return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)


def looks_like_time(series, name):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if any(k in name.lower() for k in ("time", "date", "ts", "timestamp", "_at")):
        sample = series.dropna().head(200)
        if sample.empty:
            return False
        try:
            pd.to_datetime(sample, errors="raise")
            return True
        except (ValueError, TypeError):
            return False
    return False


def sample_values(series, k=3):
    vals = series.dropna().unique()[:k]
    out = []
    for v in vals:
        s = str(v)
        out.append(s if len(s) <= 24 else s[:21] + "...")
    return ", ".join(out) if out else "—"


def profile_table(path, nrows):
    name = os.path.basename(path)
    try:
        df = load(path, nrows)
    except Exception as e:  # noqa: BLE001
        print(f"\n## {name}\n\n> could not read: {e}\n")
        return name, set()

    n = len(df)
    print(f"\n## {name}")
    print(f"\nrows read: {n:,}  ·  columns: {len(df.columns)}"
          + ("  (sampled)" if nrows and n >= nrows else ""))

    time_cols, key_cols = [], set()
    print("\n| column | dtype | null % | cardinality | sample values |")
    print("|---|---|---|---|---|")
    for col in df.columns:
        s = df[col]
        null_pct = 100.0 * s.isna().mean()
        nuniq = s.nunique(dropna=True)
        is_time = looks_like_time(s, col)
        flag = " ⏱" if is_time else ""
        if is_time:
            time_cols.append(col)
        # candidate join key: high cardinality id-ish column, low null
        if (not is_time and null_pct < 5 and n > 0
                and nuniq / max(n, 1) > 0.30
                and (s.dtype == object or pd.api.types.is_integer_dtype(s))):
            key_cols.add(col)
        print(f"| {col}{flag} | {s.dtype} | {null_pct:.1f} | {nuniq:,} "
              f"| {sample_values(s)} |")

    if time_cols:
        print(f"\nlikely time columns: {', '.join(time_cols)}")
    return name, key_cols


def main():
    ap = argparse.ArgumentParser(description="Profile tables for feature ideation.")
    ap.add_argument("paths", nargs="+", help="data dir or table files")
    ap.add_argument("--nrows", type=int, default=50000,
                    help="max CSV rows to sample (default 50000; 0 = all)")
    args = ap.parse_args()

    nrows = args.nrows if args.nrows > 0 else None
    tables = find_tables(args.paths)
    if not tables:
        sys.exit("no .csv/.tsv/.parquet tables found at the given path(s).")

    print(f"# Table inventory ({len(tables)} tables)")
    keys_by_table = {}
    for path in tables:
        name, keys = profile_table(path, nrows)
        keys_by_table[name] = keys

    # cross-table candidate join keys (shared column names that look key-like)
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
        print("\n> No obvious shared keys detected. Join keys may be named "
              "differently across tables — confirm the entity/edge keys manually.")


if __name__ == "__main__":
    main()
