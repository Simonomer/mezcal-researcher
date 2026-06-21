#!/usr/bin/env python3
"""
build_features_spark.py — REAL-WORLD materialization on Spark / Spark Connect.

This is the cluster-side counterpart of the local `build_features.py`. Same
features, same leakage rules, but everything is a distributed DataFrame op and
the output is written back to the catalog — nothing large is pulled to the
driver. Run it inside your Spark notebook (it picks up the active session) or
as a client against Spark Connect (set SPARK_REMOTE=sc://host:15002).

It is a TEMPLATE: point DB/table names at your catalog and extend with the rest
of your backlog rows following the same patterns. (Adapted from the tested
pandas builder; tailor to your schema before running.)

    spark-submit build_features_spark.py        # or run cell-by-cell in a notebook
"""

import os
from functools import reduce
from operator import add

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# --- session: use the notebook's active session, else Spark Connect ----------
spark = (SparkSession.getActiveSession()
         or SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).getOrCreate())

DB = os.environ.get("FEATURE_DB", "telco")     # your catalog/schema
K = 5                                          # CV folds for the OOF neighbor feature
CITIES = ["city_A", "city_B", "city_C", "city_D", "city_E"]
REGIONS = [f"reg_{c[-1]}" for c in CITIES]


# --- helpers -----------------------------------------------------------------
def cat_fractions(df, key, cat_col, categories, prefix):
    """Per-entity normalized category counts -> *_frac_<cat> + *_entropy."""
    counts = df.groupBy(key).pivot(cat_col, categories).count().na.fill(0)
    total = reduce(add, [F.col(c) for c in categories])
    out = counts
    for c in categories:
        out = out.withColumn(f"{prefix}_frac_{c}", (F.col(c) / total))
    ent = reduce(add, [F.when(F.col(c) > 0, -(F.col(c) / total) * F.log(F.col(c) / total))
                        .otherwise(F.lit(0.0)) for c in categories])
    out = out.withColumn(f"{prefix}_entropy", ent)
    return out.drop(*categories)


def top_category(df, key, cat_col, out_name):
    """Modal category per entity (argmax of the counts)."""
    w = Window.partitionBy(key).orderBy(F.desc("cnt"))
    return (df.groupBy(key, cat_col).count().withColumnRenamed("count", "cnt")
            .withColumn("rk", F.row_number().over(w)).filter("rk = 1")
            .select(key, F.col(cat_col).alias(out_name)))


# --- load --------------------------------------------------------------------
comms = spark.table(f"{DB}.comms")
labels = spark.table(f"{DB}.home_city").select("civ_id", "home_city")
civ = spark.table(f"{DB}.civilians")

# === tower-region geography (near-leak strong; no labels needed) =============
tp = (spark.table(f"{DB}.tower_pings")
      .join(spark.table(f"{DB}.towers").select("tower_id", "region_code"), "tower_id"))
tower = cat_fractions(tp, "civ_id", "region_code", REGIONS, "tower")
tower = tower.join(top_category(tp, "civ_id", "region_code", "tower_top"), "civ_id", "left")

# === merchant-city geography (partial; null city for online merchants) =======
tx = (spark.table(f"{DB}.transactions")
      .join(spark.table(f"{DB}.merchants").select("merchant_id", "merchant_city", "is_online"),
            "merchant_id"))
merch_online = tx.groupBy("civ_id").agg(F.avg(F.col("is_online").cast("double")).alias("merch_online_frac"))
in_store = tx.filter(F.col("merchant_city").isNotNull())
merch = cat_fractions(in_store, "civ_id", "merchant_city", CITIES, "merch")

# === activity-hour (faint timezone proxy) + degree (structural) ==============
hr = spark.table(f"{DB}.app_events").withColumn("h", F.hour("ts"))
act = hr.groupBy("civ_id").agg(F.avg("h").alias("act_hour_mean"), F.stddev("h").alias("act_hour_std"))
deg = comms.groupBy(F.col("caller_id").alias("civ_id")).agg(F.count("*").alias("deg_out"))

# === NEIGHBOR-CITY MIX — out-of-fold, TRAIN-LABELS-ONLY (the leak-critical one)
# Deterministic fold per civ; a civilian's neighbor counts use ONLY contacts in
# *other* folds -> its own label (and its own fold's labels) can never enter.
# Carry this same `fold` column into evaluation for strict OOF consistency.
folds = labels.withColumn("fold", F.pmod(F.hash("civ_id"), F.lit(K)))
edges = (comms.select(F.col("caller_id").alias("a"), F.col("callee_id").alias("b"))
         .unionByName(comms.select(F.col("callee_id").alias("a"), F.col("caller_id").alias("b"))))
labeled = (edges
           .join(folds.selectExpr("civ_id as b", "home_city as b_city", "fold as b_fold"), "b")
           .join(folds.selectExpr("civ_id as a", "fold as a_fold"), "a")
           .filter(F.col("a_fold") != F.col("b_fold")))          # <-- train-fold only
nb = (cat_fractions(labeled, "a", "b_city", CITIES, "nb")
      .withColumnRenamed("a", "civ_id"))           # -> nb_frac_city_A.., nb_entropy
nb = nb.join(top_category(labeled, "a", "b_city", "nb_modal_city")
             .withColumnRenamed("a", "civ_id"), "civ_id", "left")

# === DELIBERATE LEAK (built so the screen catches it) ========================
# weather_logs is per-region; the ONLY way to attach it to a civ is through their
# home region -- i.e. the label. Region annual mean temp keyed on the answer.
reg_temp = spark.table(f"{DB}.weather_logs").groupBy("region_code").agg(F.avg("temp_c").alias("t"))
civ_region = labels.withColumn("region_code", F.concat(F.lit("reg_"), F.substring("home_city", -1, 1)))
wx = (civ_region.join(reg_temp, "region_code")
      .select("civ_id", F.col("t").alias("wx_home_region_temp")))

# === passthrough demographics + stability time axis ==========================
base = civ.select("civ_id", "age", "language_pref", "device_type",
                  F.col("account_created_at").alias("ref_ts"))

# === assemble (left joins on civ_id) + write back to the catalog =============
feat = base
for part in [tower, merch, merch_online, act, deg, nb, wx]:
    feat = feat.join(part, "civ_id", "left")

(feat.write.mode("overwrite").saveAsTable(f"{DB}.home_city_features"))
print(f"wrote {DB}.home_city_features  ({len(feat.columns)} columns)")
print("next: python scripts/signal_panel.py "
      f"--features-table {DB}.home_city_features --labels-table {DB}.home_city "
      "--id-col civ_id --label-col home_city --time-col ref_ts --perm 40 "
      "--out validation/report.md   # (SPARK_REMOTE picked up automatically)")
