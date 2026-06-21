#!/usr/bin/env python3
"""
build_features_spark.py — REAL-WORLD materialization on Spark / Spark Connect for
the 50-city / 30-labeled text problem. Cluster-side counterpart of the pandas
build_features.py; everything is a distributed DataFrame op and the output is
written back to the catalog (or a path). Nothing large is pulled to the driver.

Reads from a catalog schema OR a base path on HDFS / S3 / GCS / DBFS — set SOURCE:
    export SOURCE=telco                      # catalog: spark.table("telco.messages")
    export SOURCE=s3://bucket/telco          # path:    spark.read.parquet("s3://…/messages")
    export SPARK_REMOTE=sc://host:15002      # if running as a Connect client

It is a TEMPLATE showing the patterns (text extraction, out-of-fold train-only
neighbor labels with PARTIAL coverage, geo modal, the deliberate leaks). Extend
with the remaining backlog rows and tailor names/paths to your warehouse.

    spark-submit build_features_spark.py     # or run cell-by-cell in a notebook
"""

import os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = (SparkSession.getActiveSession()
         or SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).getOrCreate())
SOURCE = os.environ.get("SOURCE", "telco")
OUT = os.environ.get("FEATURE_OUT", f"{SOURCE}.home_city_features"
                     if "://" not in SOURCE and not SOURCE.startswith("/")
                     else f"{SOURCE.rstrip('/')}/home_city_features")
K = 5
CITY_NUM = r"city_(\d{2})"


def read(name):
    """Catalog table (schema.name) OR parquet under a base path (HDFS/S3/GCS/DBFS)."""
    if "://" in SOURCE or SOURCE.startswith(("/", "dbfs:")):
        return spark.read.parquet(f"{SOURCE.rstrip('/')}/{name}")
    return spark.table(f"{SOURCE}.{name}")


def modal(df, key, cat_col, out_name):
    """argmax category per entity + its share + entropy."""
    g = df.groupBy(key, cat_col).count()
    w = Window.partitionBy(key)
    g = (g.withColumn("tot", F.sum("count").over(w))
         .withColumn("p", F.col("count") / F.col("tot"))
         .withColumn("rk", F.row_number().over(w.orderBy(F.desc("count")))))
    top = g.filter("rk = 1").select(key, F.col(cat_col).alias(out_name),
                                    F.col("p").alias(f"{out_name}_frac"))
    entdf = (g.groupBy(key).agg((-F.sum(F.col("p") * F.log("p"))).alias(f"{out_name}_entropy")))
    return top.join(entdf, key)


# --- load --------------------------------------------------------------------
messages = read("messages").dropDuplicates()
people = read("people").dropDuplicates(["person_id"])
labels = read("home_city").dropDuplicates(["person_id"]).select("person_id", "home_city")

feat = people.select("person_id")

# === TEXT content ============================================================
# tokenize -> per-token regex; flavor tokens reveal the city, explicit mentions
# are a near-leak. (regexp/rlike/explode are portable across Spark + Connect.)
toks = messages.select(F.col("sender_id").alias("person_id"),
                       F.explode(F.split(F.coalesce("text", F.lit("")), r"\s+")).alias("tok"))
flav = (toks.filter(F.col("tok").rlike(r"^(land|team|slang)_city_\d{2}$"))
        .withColumn("fcity", F.concat(F.lit("city_"), F.regexp_extract("tok", CITY_NUM, 1))))
feat = feat.join(modal(flav, "person_id", "fcity", "txt_flavor_top_city"), "person_id", "left")
ment = (toks.filter(F.col("tok").rlike(r"^cityname_city_\d{2}$"))
        .withColumn("mcity", F.concat(F.lit("city_"), F.regexp_extract("tok", CITY_NUM, 1))))
feat = feat.join(modal(ment, "person_id", "mcity", "txt_modal_mention_city"), "person_id", "left")
feat = feat.join(messages.groupBy(F.col("sender_id").alias("person_id"))
                 .agg(F.count(F.lit(1)).alias("txt_n_messages"),
                      F.first("lang").alias("txt_lang_dominant")), "person_id", "left")

# === GRAPH: degree + OUT-OF-FOLD, TRAIN-ONLY neighbor city (partial labels) ===
edges = (messages.select(F.col("sender_id").alias("a"), F.col("recipient_id").alias("b"))
         .unionByName(messages.select(F.col("recipient_id").alias("a"), F.col("sender_id").alias("b"))))
feat = feat.join(edges.groupBy(F.col("a").alias("person_id")).agg(F.countDistinct("b").alias("deg_total")),
                 "person_id", "left")
# deterministic fold per LABELED person; unlabeled -> -1. Only labeled neighbors
# in a DIFFERENT fold contribute -> a person's own/fold label never leaks in.
folds = labels.withColumn("fold", F.pmod(F.hash("person_id"), F.lit(K)))
allf = people.select("person_id").join(folds.select("person_id", "fold"), "person_id", "left") \
             .fillna({"fold": -1})
nbr = (edges.join(folds.selectExpr("person_id as b", "home_city as b_city", "fold as b_fold"), "b")
       .join(allf.selectExpr("person_id as a", "fold as a_fold"), "a")
       .filter(F.col("a_fold") != F.col("b_fold")))          # train-fold-only
feat = feat.join(modal(nbr, "a", "b_city", "nb_modal_city")
                 .withColumnRenamed("a", "person_id"), "person_id", "left")

# === GEO: tower region + merchant city (normalize messy casing/space) ========
tp = read("tower_pings").join(read("towers").select("tower_id", "region_code"), "tower_id")
feat = feat.join(modal(tp, "person_id", "region_code", "tower_modal_region"), "person_id", "left")
merch = read("merchants").withColumn(
    "mc", F.concat(F.lit("city_"), F.regexp_extract(F.lower(F.trim("merchant_city")), CITY_NUM, 1)))
tx = read("transactions").join(merch.select("merchant_id", "mc", "is_online"), "merchant_id")
feat = feat.join(modal(tx.filter(F.col("mc") != "city_"), "person_id", "mc", "merch_modal_city"),
                 "person_id", "left")

# === DELIBERATE LEAKS (built so the screen flags them) =======================
feat = feat.join(read("ip_geo").select("person_id", F.col("ip_city")), "person_id", "left")  # ≈ answer
feat = feat.join(people.select("person_id", F.concat(
    F.lit("city_"), F.regexp_extract(F.lower(F.trim("raw_city_text")), CITY_NUM, 1)).alias("declared_city")),
    "person_id", "left")
# weather joined via the person's HOME region (= the label) -> metro-level leak
reg_temp = read("weather_logs").groupBy("region_code").agg(F.avg("temp_c").alias("wx_home_region_temp"))
home_region = labels.withColumn(
    "region_code", F.concat(F.lit("metro_"), (F.regexp_extract("home_city", CITY_NUM, 1).cast("int") / 5).cast("int")))
feat = feat.join(home_region.join(reg_temp, "region_code").select("person_id", "wx_home_region_temp"),
                 "person_id", "left")

# === demographics + write ====================================================
feat = feat.join(people.selectExpr("person_id", "age", "declared_language", "device",
                                    "signup_at as ref_ts"), "person_id", "left")
(feat.write.mode("overwrite").saveAsTable(OUT) if "://" not in str(OUT) and not str(OUT).startswith("/")
 else feat.write.mode("overwrite").parquet(OUT))
print(f"wrote {OUT}  ({len(feat.columns)} columns)")
print(f"screen it:  python scripts/signal_panel.py --features-table {OUT} "
      f"--labels-table {SOURCE}{'.' if '://' not in SOURCE else '/'}home_city "
      "--id-col person_id --label-col home_city --perm 40 --out validation/report.md")
