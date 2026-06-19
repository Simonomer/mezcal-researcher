#!/usr/bin/env python3
"""
build_features.py — materialize the kept backlog rows into one feature table
keyed by civ_id, with leakage rules enforced in code.

Leakage handling:
  * Neighbor-label features (per-city contact fraction, modal contact city,
    entropy) are computed OUT-OF-FOLD: for each civilian we only count the
    labels of contacts that fall in the *training* folds. A civilian's own
    label and any test-fold label can never enter its own feature. This is a
    CV-fold loop, not a global groupby — that distinction is the whole point.
    The fold scheme (StratifiedKFold, 5, shuffle, seed 0, on civ_id-sorted
    order) matches the validation harness's CV, so a test civ's label also
    never reaches the model through a neighbor's feature.
  * Tower-region and merchant-city aggregates are NEAR-LEAK strong features:
    legitimate (observed per-civ, before the label) but geographically almost
    decisive. Kept on purpose; the screen is expected to flag them.
  * device_*, app-volume, age = red-herring controls. weather = a DELIBERATE
    leak (per-region table joined to the civ through their home region = the
    answer). Built so the screen demonstrably rejects / flags them.

Output: features/feature_table.parquet  (civ_id + features + ref_ts).
home_city stays in data/tables/home_city.parquet, joinable by civ_id.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(__file__)
TAB = os.path.join(HERE, "..", "data", "tables")
CITIES = ["city_A", "city_B", "city_C", "city_D", "city_E"]
REGION = {c: f"reg_{c[-1]}" for c in CITIES}
REGIONS = [REGION[c] for c in CITIES]
jit = np.random.default_rng(0)


def L(name):
    return pd.read_parquet(os.path.join(TAB, f"{name}.parquet"))


def row_entropy(frac):
    """Shannon entropy per row of a fraction matrix (NaN where the row is empty)."""
    p = np.where(frac > 0, frac, np.nan)
    e = -(np.nan_to_num(p * np.log(p))).sum(axis=1)
    e[np.isnan(frac).all(axis=1)] = np.nan
    return e


def frac_table(df, key, cat_col, categories, prefix):
    """Per-entity normalized category counts -> frac columns + top + entropy."""
    g = (df.groupby([key, cat_col]).size().unstack(fill_value=0)
         .reindex(columns=categories, fill_value=0))
    tot = g.sum(axis=1)
    frac = g.div(tot.where(tot > 0), axis=0)
    out = frac.copy()
    out.columns = [f"{prefix}_frac_{c}" for c in categories]
    out[f"{prefix}_top"] = frac.idxmax(axis=1).where(tot > 0)
    out[f"{prefix}_entropy"] = row_entropy(frac.to_numpy())
    out[f"{prefix}_n_cat"] = (g > 0).sum(axis=1).where(tot > 0)
    return out


# ---------------------------------------------------------------- load -------
civ = L("civilians").sort_values("civ_id").reset_index(drop=True)
labels = L("home_city").set_index("civ_id")["home_city"]
comms, towers = L("comms"), L("towers")
tower_pings, merchants = L("tower_pings"), L("merchants")
transactions, app_events = L("transactions"), L("app_events")
device_info, weather = L("device_info"), L("weather_logs")

civ_ids = civ["civ_id"].to_numpy()
N = len(civ_ids)
idx = {c: i for i, c in enumerate(civ_ids)}
city_code = {c: i for i, c in enumerate(CITIES)}
y = labels.reindex(civ_ids).map(city_code).to_numpy()

feat = pd.DataFrame(index=pd.Index(civ_ids, name="civ_id"))

# ----------------------------------------- tower geography (near-leak strong)-
tp = tower_pings.merge(towers[["tower_id", "region_code"]], on="tower_id", how="left")
feat = feat.join(frac_table(tp, "civ_id", "region_code", REGIONS, "tower"))

# ----------------------------------------- merchant geography (partial) ------
tx = transactions.merge(merchants[["merchant_id", "merchant_city", "is_online"]],
                        on="merchant_id", how="left")
feat["merch_online_frac"] = tx.groupby("civ_id")["is_online"].mean()
feat["merch_txn_count"] = tx.groupby("civ_id").size()
in_store = tx[tx["merchant_city"].notna()]
feat = feat.join(frac_table(in_store, "civ_id", "merchant_city", CITIES, "merch"))

# ----------------------------------------- activity-hour (faint tz proxy) ----
hr = app_events.assign(h=app_events["ts"].dt.hour)
feat["act_hour_mean"] = hr.groupby("civ_id")["h"].mean()
feat["act_hour_std"] = hr.groupby("civ_id")["h"].std()
feat["act_peak_hour"] = pd.crosstab(hr["civ_id"], hr["h"]).idxmax(axis=1)
feat["app_event_count"] = app_events.groupby("civ_id").size()             # red-herring volume
feat["app_n_event_types"] = app_events.groupby("civ_id")["event_type"].nunique()

# ----------------------------------------- degree / structural (weak) --------
feat["deg_out"] = comms.groupby("caller_id").size()
feat["deg_in"] = comms.groupby("callee_id").size()
und = pd.concat([
    comms[["caller_id", "callee_id"]].rename(columns={"caller_id": "a", "callee_id": "b"}),
    comms[["callee_id", "caller_id"]].rename(columns={"callee_id": "a", "caller_id": "b"})])
feat["deg_total"] = und.groupby("a")["b"].nunique()
feat["call_frac"] = comms.assign(c=comms["channel"].eq("call")).groupby("caller_id")["c"].mean()
# counterparty diversity = entropy of per-contact contact counts
cc = und.groupby(["a", "b"]).size().rename("n").reset_index()
ctot = cc.groupby("a")["n"].transform("sum")
cc["p"] = cc["n"] / ctot
feat["contact_entropy"] = cc.assign(e=-cc["p"] * np.log(cc["p"])).groupby("a")["e"].sum()

# ----------------------------------------- demographic / device -------------
civ_i = civ.set_index("civ_id")
feat["age"] = civ_i["age"]
feat["language_pref"] = civ_i["language_pref"]
feat["device_type"] = civ_i["device_type"]
dev = device_info.set_index("civ_id")
feat["device_screen_size"] = dev["screen_size_in"]                        # red-herring
feat["device_battery_health"] = dev["battery_health"]                     # red-herring
feat["os_version"] = dev["os_version"]                                    # red-herring

# ----------------------------------------- NEIGHBOR LABEL (OOF, train-only) --
# Undirected contact lists (with multiplicity = edge weight).
a = comms["caller_id"].map(idx).to_numpy()
b = comms["callee_id"].map(idx).to_numpy()
src = np.concatenate([a, b])
dst = np.concatenate([b, a])
contacts = [[] for _ in range(N)]
for s, d in zip(src.tolist(), dst.tolist()):
    contacts[s].append(d)

nb_counts = np.zeros((N, 5))      # train-fold contact-city counts
nb_cov = np.zeros(N)             # # of contacts that were labeled (train fold)
nb_deg = np.array([len(c) for c in contacts], dtype=float)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for _, test_idx in skf.split(np.zeros(N), y):       # iterate by TEST fold
    in_train = np.ones(N, dtype=bool)
    in_train[test_idx] = False                      # this fold's labels are hidden
    for i in test_idx:                              # each civ handled exactly once
        for j in contacts[i]:
            if in_train[j]:                         # only TRAIN-fold neighbor labels
                nb_counts[i, y[j]] += 1.0
                nb_cov[i] += 1.0
# (a civ's own label is never read; a contact in the same fold contributes nothing)

tot = nb_counts.sum(axis=1)
nb_frac = np.divide(nb_counts, np.where(tot[:, None] > 0, tot[:, None], np.nan))
nb = pd.DataFrame(index=feat.index)
for k, c in enumerate(CITIES):
    nb[f"nb_frac_{c}"] = nb_frac[:, k]
nb["nb_modal_city"] = np.where(tot > 0, np.array(CITIES)[nb_counts.argmax(1)], "unknown")
nb["nb_city_entropy"] = row_entropy(nb_frac)
nb["nb_label_coverage"] = np.where(nb_deg > 0, nb_cov / np.maximum(nb_deg, 1), np.nan)
feat = feat.join(nb)

# ----------------------------------------- WEATHER (DELIBERATE LEAK) ---------
# weather_logs is per region_code; the ONLY way to attach it to a civilian is
# through their home region — i.e. the label. We do that on purpose so the
# screen flags it. Region annual means + tiny jitter (keeps it continuous so
# the harness treats it as numeric and computes the give-away AUC).
reg_temp = weather.groupby("region_code")["temp_c"].mean()
reg_precip = weather.groupby("region_code")["precip_mm"].mean()
civ_region = labels.reindex(civ_ids).map(REGION)                 # <-- uses the answer
feat["wx_home_region_temp"] = civ_region.map(reg_temp).to_numpy() + jit.normal(0, 0.3, N)
feat["wx_home_region_precip"] = civ_region.map(reg_precip).to_numpy() + jit.normal(0, 2.0, N)

# ----------------------------------------- assemble + write ------------------
feat["ref_ts"] = civ_i["account_created_at"]      # stability time axis (not signal)
feat = feat.reset_index().sort_values("civ_id").reset_index(drop=True)
out = os.path.join(HERE, "feature_table.parquet")
feat.to_parquet(out, index=False)

nfeat = feat.shape[1] - 2  # minus civ_id, ref_ts
print(f"wrote {out}")
print(f"rows={len(feat):,}  features={nfeat}")
print("families:",
      "neighbor(OOF train-only), tower-region, merchant-city, activity-hour, "
      "language, degree/structural, device(red-herring), app-volume(red-herring), "
      "weather(leak)")
cov = feat[[c for c in feat.columns if c.startswith("nb_frac_")]].notna().any(axis=1).mean()
print(f"neighbor-feature coverage (>=1 labeled train contact): {cov:.1%}")
print(f"mean nb_frac on true city: "
      f"{np.nanmean([nb_frac[i, y[i]] for i in range(N) if tot[i] > 0]):.3f} "
      f"(~matches the raw comm homophily; OOF estimate is unbiased)")
