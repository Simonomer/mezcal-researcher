#!/usr/bin/env python3
"""Feature-signal validation harness for validate-signal.

Screens candidate features for signal about a multiclass label, then writes a
markdown report + PNG charts. Does NOT compute features — it scores existing
ones (the materialized backlog).

What "signal" means here (layered, no single number trusted):
  - beats a shuffled-label permutation null  -> not chance
  - effect size (MI / best one-vs-rest AUC)  -> big enough to matter
  - permutation importance in a baseline model -> incremental, not redundant
  - stable across time slices (if --time-col)  -> not overfit/drift
All under a leakage-safe split (StratifiedKFold, or GroupKFold via --group-col).

Inputs (file mode is fully local; table mode uses Spark Connect):
  --features-file F.parquet|csv   OR  --features-table matcha.feat
  --labels-file   L.parquet|csv   OR  --labels-table   matcha.labels
  --id-col entity_id  --label-col segment
  [--time-col ts] [--group-col entity_id] [--remote sc://host:15002]
  [--sample 200000] [--perm 30] [--out validation/report.md]

Deps: pandas numpy scikit-learn scipy matplotlib
Usage:
  python signal_panel.py --features-table matcha.feat --labels-table matcha.labels \
      --id-col entity_id --label-col segment --time-col ts --remote sc://host:15002
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------- loading ----------------------------------------------------------

def _read_local(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    sep = "\t" if ext == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def get_spark(remote):
    from pyspark.sql import SparkSession
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    b = SparkSession.builder.appName("validate-signal")
    remote = remote or os.environ.get("SPARK_REMOTE")
    if remote:
        b = b.remote(remote)
    return b.getOrCreate()


def _looks_like_path(ref):
    return ("://" in ref or ref.startswith(("/", "dbfs:", "file:"))
            or ref.endswith((".parquet", ".csv", ".tsv", ".json")) or "*" in ref)


def spark_read(spark, ref, fmt=None):
    """Read a catalog table OR a path on HDFS / S3 / S3A / GCS / DBFS / local.

    Paths (anything with a scheme, leading slash, glob, or data extension) are
    read with spark.read.<format>; everything else is treated as a catalog
    table. Format is inferred from the extension unless --format is given.
    """
    if not _looks_like_path(ref):
        return spark.table(ref)
    if fmt is None:
        fmt = ("csv" if ref.endswith((".csv", ".tsv")) else
               "json" if ref.endswith(".json") else "parquet")
    reader = spark.read
    if fmt == "csv":
        sep = "\t" if ref.endswith(".tsv") else ","
        reader = reader.option("header", True).option("inferSchema", True).option("sep", sep)
    return reader.format(fmt).load(ref)


def load_frame(args):
    """Return one pandas df with id + label (+ time) + feature columns."""
    if args.features_file:
        feats = _read_local(args.features_file)
        labels = _read_local(args.labels_file)
        df = feats.merge(labels[[args.id_col, args.label_col]], on=args.id_col, how="inner")
        if len(df) > args.sample:
            df = df.groupby(args.label_col, group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, args.sample // df[args.label_col].nunique())),
                                   random_state=0))
        return df.reset_index(drop=True)

    spark = get_spark(args.remote)
    from pyspark.sql import functions as F
    feats = spark_read(spark, args.features_table, args.format)
    labels = spark_read(spark, args.labels_table, args.format).select(args.id_col, args.label_col)
    joined = feats.join(labels, on=args.id_col, how="inner")
    # stratified sample toward --sample rows
    try:
        counts = {r[args.label_col]: r["c"] for r in
                  joined.groupBy(args.label_col).agg(F.count(F.lit(1)).alias("c")).collect()}
        per = max(1, args.sample // max(1, len(counts)))
        fracs = {k: min(1.0, per / v) for k, v in counts.items() if v}
        sdf = joined.sampleBy(args.label_col, fractions=fracs, seed=0)
    except Exception:
        sdf = joined.limit(args.sample)
    return sdf.limit(args.sample).toPandas()


# ---------- metric helpers ---------------------------------------------------

def split_columns(df, id_col, label_col, time_col, explicit):
    drop = {id_col, label_col}
    if time_col:
        drop.add(time_col)
    cols = explicit if explicit else [c for c in df.columns if c not in drop]
    num, cat = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 10:
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def encode_matrix(df, num, cat):
    """Numeric matrix + discrete mask for mutual_info_classif."""
    parts, mask, names = [], [], []
    for c in num:
        parts.append(pd.to_numeric(df[c], errors="coerce").fillna(df[c].median()
                     if pd.api.types.is_numeric_dtype(df[c]) else 0).to_numpy())
        mask.append(False)
        names.append(c)
    for c in cat:
        # +1 so missing (factorize sentinel -1) -> 0; all codes non-negative,
        # which lets the baseline model use them as NATIVE categorical features.
        codes = (pd.factorize(df[c].astype("object"))[0] + 1).astype(float)
        parts.append(codes)
        mask.append(True)
        names.append(c)
    X = np.vstack(parts).T if parts else np.empty((len(df), 0))
    return X, np.array(mask), names


def best_ovr_auc(x, y, classes):
    from sklearn.metrics import roc_auc_score
    best = 0.5
    xv = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    ok = ~np.isnan(xv)
    if ok.sum() < 20:
        return np.nan
    for c in classes:
        yc = (y[ok] == c).astype(int)
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        try:
            a = roc_auc_score(yc, xv[ok])
            best = max(best, a, 1 - a)
        except Exception:
            pass
    return best


def kruskal_stat(x, y, classes):
    from scipy.stats import kruskal
    xv = pd.to_numeric(pd.Series(x), errors="coerce")
    groups = [xv[y == c].dropna().to_numpy() for c in classes]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return np.nan
    try:
        return float(kruskal(*groups).statistic)
    except Exception:
        return np.nan


def mi_all(X, y, mask, seed=0):
    from sklearn.feature_selection import mutual_info_classif
    if X.shape[1] == 0:
        return np.array([])
    return mutual_info_classif(X, y, discrete_features=mask, random_state=seed)


def permutation_null(X, y, mask, k):
    """Return per-feature 95th percentile of MI under shuffled labels."""
    if X.shape[1] == 0 or k <= 0:
        return np.zeros(X.shape[1])
    rng = np.random.default_rng(0)
    null = np.empty((k, X.shape[1]))
    for i in range(k):
        null[i] = mi_all(X, rng.permutation(y), mask, seed=i)
    return np.percentile(null, 95, axis=0)


def baseline_model(X, y, names, mask=None, group=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split, cross_val_predict
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import label_binarize

    # treat the discrete columns as NATIVE categorical features so the model can
    # actually use ids / text-derived cities (not arbitrary codes). Cap at 255
    # categories (HGB's limit); anything higher falls back to continuous.
    cat_feats = None
    if mask is not None and np.any(mask):
        card = np.array([np.unique(X[:, i]).size for i in range(X.shape[1])])
        cat_feats = mask & (card <= 255)
        if not np.any(cat_feats):
            cat_feats = None
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=0,
                                         categorical_features=cat_feats)
    if group is not None:
        cv = GroupKFold(n_splits=min(5, len(np.unique(group))))
        splits = cv.split(X, y, group)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        splits = cv.split(X, y)
    pred = cross_val_predict(clf, X, y, cv=list(splits))
    macro_f1 = f1_score(y, pred, average="macro")
    classes = np.unique(y)
    try:
        proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")
        yb = label_binarize(y, classes=classes)
        macro_auc = roc_auc_score(yb, proba, average="macro", multi_class="ovr")
    except Exception:
        macro_auc = np.nan
    cm = confusion_matrix(y, pred, labels=classes)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    fit = clf.fit(Xtr, ytr)
    imp = permutation_importance(fit, Xte, yte, n_repeats=5, random_state=0, scoring="f1_macro")
    return dict(macro_f1=macro_f1, macro_auc=macro_auc, cm=cm, classes=classes,
                importance=dict(zip(names, imp.importances_mean)))


def stability(df, num, cat, label_col, time_col, n_slices=4):
    """MI per feature across time slices -> coefficient of variation."""
    s = pd.to_datetime(df[time_col], errors="coerce")
    if s.notna().sum() < len(df) * 0.5:
        return {}
    bins = pd.qcut(s.rank(method="first"), n_slices, labels=False, duplicates="drop")
    per_slice = []
    for b in sorted(pd.unique(bins.dropna())):
        sub = df[bins == b]
        if sub[label_col].nunique() < 2 or len(sub) < 50:
            continue
        X, mask, names = encode_matrix(sub, num, cat)
        per_slice.append(pd.Series(mi_all(X, sub[label_col].to_numpy(), mask), index=names))
    if len(per_slice) < 2:
        return {}
    M = pd.concat(per_slice, axis=1)
    cv = (M.std(axis=1) / (M.mean(axis=1) + 1e-9)).abs()
    return cv.to_dict()


# ---------- plots ------------------------------------------------------------

def make_plots(res, df, num, label_col, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    paths = {}
    t = res["table"].sort_values("mi", ascending=False)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * min(20, len(t)))))
    top = t.head(20).iloc[::-1]
    ax.barh(top["feature"], top["mi"], color="#1D9E75")
    ax.set_xlabel("mutual information"); ax.set_title("Signal ranking (MI)")
    fig.tight_layout(); p = os.path.join(figdir, "signal_ranking.png"); fig.savefig(p, dpi=110); plt.close(fig)
    paths["ranking"] = p

    imp = res["baseline"]["importance"]
    if imp:
        s = pd.Series(imp).sort_values().tail(20)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(s))))
        ax.barh(s.index, s.values, color="#534AB7")
        ax.set_xlabel("permutation importance (macro-F1 drop)"); ax.set_title("Baseline importance")
        fig.tight_layout(); p = os.path.join(figdir, "importance.png"); fig.savefig(p, dpi=110); plt.close(fig)
        paths["importance"] = p

    cm = res["baseline"]["cm"]; classes = res["baseline"]["classes"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Baseline confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    p = os.path.join(figdir, "confusion.png"); fig.savefig(p, dpi=110); plt.close(fig)
    paths["confusion"] = p

    keep_num = [c for c in t.head(4)["feature"] if c in num]
    if keep_num:
        fig, axes = plt.subplots(1, len(keep_num), figsize=(3 * len(keep_num), 3.2))
        axes = np.atleast_1d(axes)
        classes_all = sorted(df[label_col].dropna().unique())
        for ax, c in zip(axes, keep_num):
            data = [pd.to_numeric(df.loc[df[label_col] == k, c], errors="coerce").dropna() for k in classes_all]
            ax.boxplot(data, labels=[str(k) for k in classes_all], showfliers=False)
            ax.set_title(c, fontsize=9); ax.tick_params(axis="x", labelsize=8)
        fig.suptitle("Top features by class"); fig.tight_layout()
        p = os.path.join(figdir, "by_class.png"); fig.savefig(p, dpi=110); plt.close(fig)
        paths["by_class"] = p

    if len(num) > 1:
        corr = df[num].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
        fig, ax = plt.subplots(figsize=(min(9, 0.5 * len(num) + 2),) * 2)
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=90, fontsize=7)
        ax.set_yticks(range(len(num))); ax.set_yticklabels(num, fontsize=7)
        ax.set_title("Feature redundancy (Spearman)"); fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); p = os.path.join(figdir, "redundancy.png"); fig.savefig(p, dpi=110); plt.close(fig)
        paths["redundancy"] = p
    return paths


# ---------- recommendation + report -----------------------------------------

def recommend(row):
    if not row["beats_null"] or row["coverage"] < 0.2:
        return "drop", "no signal over null" if not row["beats_null"] else "too sparse"
    if not np.isnan(row.get("best_auc", np.nan)) and row["best_auc"] > 0.97:
        return "investigate", "suspiciously strong — check leakage"
    # normalized MI catches near-deterministic CATEGORICAL leaks (no AUC), e.g.
    # an id/self-report column that nearly equals the label.
    if row.get("mi_ratio", 0.0) > 0.80:
        return "investigate", "explains most of the label — check leakage"
    if row.get("redundant_with"):
        return "investigate", f"redundant with {row['redundant_with']}"
    if not np.isnan(row.get("stability_cv", np.nan)) and row["stability_cv"] > 1.0:
        return "investigate", "unstable over time"
    if row["importance"] <= 0 and row["mi"] < row["null_p95"] * 1.5:
        return "investigate", "weak incremental value"
    return "keep", "beats null, contributes in model"


def write_report(args, df, res, plots, num):
    t = res["table"].copy().sort_values(["recommend", "mi"], ascending=[True, False])
    b = res["baseline"]
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    figrel = os.path.relpath(args.figdir, os.path.dirname(out) or ".")

    L = []
    L.append(f"# Signal validation — {args.label_col}\n")
    L.append(f"- sample rows: {len(df):,}  ·  classes: {df[args.label_col].nunique()}  "
             f"·  features: {len(res['table'])}")
    L.append(f"- baseline (HGB): macro-F1 = {b['macro_f1']:.3f}, "
             f"macro-AUC = {b['macro_auc']:.3f}" if not np.isnan(b['macro_auc'])
             else f"- baseline (HGB): macro-F1 = {b['macro_f1']:.3f}")
    keep = (t["recommend"] == "keep").sum()
    L.append(f"- recommendation: keep {keep} · "
             f"investigate {(t['recommend']=='investigate').sum()} · "
             f"drop {(t['recommend']=='drop').sum()}\n")

    L.append("![ranking](%s/signal_ranking.png)\n" % figrel)
    if "importance" in plots:
        L.append("![importance](%s/importance.png)\n" % figrel)
    L.append("![confusion](%s/confusion.png)\n" % figrel)
    if "by_class" in plots:
        L.append("![by_class](%s/by_class.png)\n" % figrel)
    if "redundancy" in plots:
        L.append("![redundancy](%s/redundancy.png)\n" % figrel)

    L.append("\n## Per-feature\n")
    cols = ["feature", "recommend", "reason", "mi", "mi_ratio", "null_p95", "beats_null",
            "best_auc", "importance", "coverage", "stability_cv"]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in t.iterrows():
        def f(v):
            if isinstance(v, float):
                return "—" if np.isnan(v) else f"{v:.3f}"
            return str(v)
        L.append("| " + " | ".join(f(r[c]) for c in cols) + " |")

    L.append("\n_Signal = beats shuffled-label null, has effect size, contributes "
             "incrementally in the model, and is stable. Screening only — the final "
             "word is full-model out-of-sample performance._\n")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    return out


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Validate feature signal -> markdown report.")
    ap.add_argument("--features-file"); ap.add_argument("--labels-file")
    ap.add_argument("--features-table"); ap.add_argument("--labels-table")
    ap.add_argument("--id-col", required=True); ap.add_argument("--label-col", required=True)
    ap.add_argument("--time-col"); ap.add_argument("--group-col")
    ap.add_argument("--remote"); ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--perm", type=int, default=30)
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--format", default=None,
                    help="spark read format for path inputs (parquet/csv/json); inferred if omitted")
    ap.add_argument("--out", default="validation/report.md")
    args = ap.parse_args()
    args.figdir = os.path.join(os.path.dirname(args.out) or ".", "figures")

    if not (args.features_file or args.features_table):
        sys.exit("provide --features-file or --features-table (and matching labels).")

    df = load_frame(args)
    y = df[args.label_col].to_numpy()
    if df[args.label_col].nunique() < 2:
        sys.exit("label has <2 classes in the sample.")
    classes = np.unique(y)
    num, cat = split_columns(df, args.id_col, args.label_col, args.time_col, args.features)
    if not num and not cat:
        sys.exit("no feature columns found.")
    X, mask, names = encode_matrix(df, num, cat)

    mi = mi_all(X, y, mask)
    _, _cnt = np.unique(y, return_counts=True)            # label entropy (nats)
    _p = _cnt / _cnt.sum()
    Hy = max(float(-(_p * np.log(_p)).sum()), 1e-9)
    mi_ratio = {names[i]: float(mi[i] / Hy) for i in range(len(names))}
    null95 = permutation_null(X, y, mask, args.perm)
    aucs = {c: (best_ovr_auc(df[c].to_numpy(), y, classes) if c in num else np.nan) for c in names}
    cov = {c: float(df[c].notna().mean()) for c in names}
    stab = stability(df, num, cat, args.label_col, args.time_col) if args.time_col else {}
    base = baseline_model(X, y, names, mask=mask,
                          group=df[args.group_col].to_numpy() if args.group_col else None)

    # redundancy: flag each numeric feature's strongest partner > 0.9
    redundant = {}
    if len(num) > 1:
        corr = df[num].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
        cv = corr.to_numpy(copy=True)
        np.fill_diagonal(cv, 0)
        corr = pd.DataFrame(cv, index=corr.index, columns=corr.columns)
        for c in num:
            j = corr[c].idxmax()
            if corr[c].max() > 0.9:
                redundant[c] = j

    rows = []
    for i, c in enumerate(names):
        row = dict(feature=c, mi=float(mi[i]), null_p95=float(null95[i]),
                   beats_null=bool(mi[i] > null95[i]), best_auc=float(aucs[c]),
                   mi_ratio=mi_ratio[c],
                   importance=float(base["importance"].get(c, 0.0)),
                   coverage=cov[c], stability_cv=float(stab.get(c, np.nan)),
                   redundant_with=redundant.get(c))
        rec, reason = recommend(row)
        row["recommend"], row["reason"] = rec, reason
        rows.append(row)
    table = pd.DataFrame(rows)

    res = {"table": table, "baseline": base}
    plots = make_plots(res, df, num, args.label_col, args.figdir)
    path = write_report(args, df, res, plots, num)
    print(f"wrote {path}  (+ figures in {args.figdir})")
    print(table.sort_values("mi", ascending=False)[
        ["feature", "recommend", "mi", "importance", "beats_null"]].to_string(index=False))


if __name__ == "__main__":
    main()
