#!/usr/bin/env python3
"""
build_features.py — materialize features for the 50-city / 30-labeled problem.
Local/pandas demo; the Spark counterpart is build_features_spark.py.

Keyed by person_id over ALL people (labeled + unlabeled) so the graph/text
context is complete; home_city (30 labeled cities) is joined separately by the
screen. Leakage rules enforced in code:

  - Neighbor-label features use TRAIN-FOLD labels only (out-of-fold CV loop), and
    only LABELED neighbors contribute — a person's own/test-fold label never
    enters. Unlabeled people carry fold = -1 (they are never evaluated).
  - Text/doc/graph/geo features are derived from observable content only.
  - Deliberate near-leaks / leaks kept on purpose so the screen flags them:
    ip_city (≈ the answer), region weather (joined via the answer), and the
    self-reported declared_city (strong but messy).

Messy real-world handling: de-dup people/messages, normalize inconsistent city
strings, coerce types, tolerate nulls.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(__file__)
TAB = os.path.join(HERE, "..", "data", "tables")
CITY_RE = re.compile(r"city[_ ]?(\d{2})", re.I)
FLAVOR_RE = re.compile(r"(?:land|team|slang)_city_(\d{2})")
MENTION_RE = re.compile(r"cityname_city_(\d{2})")


def L(name):
    return pd.read_parquet(os.path.join(TAB, f"{name}.parquet"))


def norm_city(s):
    """Parse a messy city string ('City 07', ' CITY_07', typos) -> 'city_07'/None."""
    if not isinstance(s, str):
        return None
    m = CITY_RE.search(s.strip())
    return f"city_{int(m.group(1)):02d}" if m else None


def ent(series_counts):
    p = series_counts / series_counts.sum()
    return float(-(p * np.log(p)).sum())


def _city_counts(texts, pattern):
    """Long frame [key, num, c] of per-person counts of cities matched by pattern.
    Uses reset_index + named columns (NOT groupby(level=0)) to avoid the
    integer-named-level trap that silently misaligns the result."""
    key = texts.index.name or "key"
    ex = texts.str.extractall(pattern)
    if ex.empty:
        return None, key
    d = ex.rename(columns={0: "num"}).reset_index()
    return d.groupby([key, "num"]).size().rename("c").reset_index(), key


def per_person_textstats(texts):
    """texts: Series indexed by person_id of concatenated message text."""
    out = pd.DataFrame(index=texts.index)
    fc, key = _city_counts(texts, FLAVOR_RE)               # per-city flavor-token counts
    if fc is not None:
        tot = fc.groupby(key)["c"].transform("sum")
        fc["p"] = fc["c"] / tot
        top = fc.loc[fc.groupby(key)["c"].idxmax()].set_index(key)
        out["txt_flavor_top_city"] = "city_" + top["num"]
        out["txt_flavor_top_frac"] = top["c"] / fc.groupby(key)["c"].sum()
        out["txt_flavor_entropy"] = fc.assign(e=-fc["p"] * np.log(fc["p"])).groupby(key)["e"].sum()
    mc, key = _city_counts(texts, MENTION_RE)              # explicit city mentions (near-leak)
    if mc is not None:
        top = mc.loc[mc.groupby(key)["c"].idxmax()].set_index(key)
        out["txt_modal_mention_city"] = "city_" + top["num"]
        out["txt_n_city_mentions"] = mc.groupby(key)["c"].sum()
    toks = texts.str.split()                               # cheap style stats (noise)
    out["txt_total_tokens"] = toks.map(len)
    out["txt_vocab_richness"] = toks.map(lambda t: len(set(t)) / max(1, len(t)))
    out["txt_url_frac"] = texts.str.count(r"http|www\.") / toks.map(len).clip(lower=1)
    out["txt_emoji_frac"] = texts.str.count(r"[\U0001F300-\U0001FAFF]") / toks.map(len).clip(lower=1)
    return out


# ---------------------------------------------------------------- load -------
people = L("people").drop_duplicates("person_id").set_index("person_id")
labels = L("home_city").drop_duplicates("person_id").set_index("person_id")["home_city"]
messages = L("messages").drop_duplicates()
docs = L("person_docs").drop_duplicates("person_id").set_index("person_id")
ip = L("ip_geo").drop_duplicates("person_id").set_index("person_id")

pid = people.index.to_numpy()
N = len(pid)
pos = {p: i for i, p in enumerate(pid)}
feat = pd.DataFrame(index=people.index)

# ---------------------------------------------------------------- TEXT -------
msg_txt = messages.dropna(subset=["text"]).groupby("sender_id")["text"].agg(" ".join)
feat = feat.join(per_person_textstats(msg_txt))
feat["txt_n_messages"] = messages.groupby("sender_id").size()
feat["txt_lang_dominant"] = (messages.dropna(subset=["lang"])
                             .groupby("sender_id")["lang"].agg(lambda s: s.mode().iat[0]))

# ---------------------------------------------------------------- DOCS -------
feat["doc_present"] = feat.index.isin(docs.index).astype(int)
dtxt = docs["doc_text"]
feat["doc_len"] = dtxt.str.split().map(len)
dfc, dk = _city_counts(dtxt, FLAVOR_RE)
if dfc is not None:
    dtop = dfc.loc[dfc.groupby(dk)["c"].idxmax()].set_index(dk)
    feat["doc_flavor_top_city"] = "city_" + dtop["num"]
dmc, dk = _city_counts(dtxt, MENTION_RE)
if dmc is not None:
    feat["doc_n_city_mentions"] = dmc.groupby(dk)["c"].sum()

# ---------------------------------------------------------------- GRAPH ------
und = pd.concat([
    messages[["sender_id", "recipient_id"]].rename(columns={"sender_id": "a", "recipient_id": "b"}),
    messages[["recipient_id", "sender_id"]].rename(columns={"recipient_id": "a", "sender_id": "b"})])
feat["deg_out"] = messages.groupby("sender_id").size()
feat["deg_in"] = messages.groupby("recipient_id").size()
feat["deg_total"] = und.groupby("a")["b"].nunique()
# reciprocity: share of out-contacts who also messaged back
pairs = set(zip(messages["sender_id"], messages["recipient_id"]))
recip = {a: 0 for a in pid}
outc = messages.groupby("sender_id")["recipient_id"].agg(set)
for a, bs in outc.items():
    if len(bs):
        recip[a] = np.mean([(b, a) in pairs for b in bs])
feat["reciprocity"] = pd.Series(recip)

# --- OOF train-only neighbor city (partial labels) ---------------------------
lab_ids = labels.index.to_numpy()
y_lab = labels.to_numpy()
fold = pd.Series(-1, index=people.index)                     # unlabeled -> -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for k, (_, te) in enumerate(skf.split(lab_ids, y_lab)):
    fold.loc[lab_ids[te]] = k
fold_arr = fold.to_numpy()
lab_map = labels.to_dict()                                   # labeled id -> city
und_arr = und.to_numpy()
nb_counts = {}                                               # person -> {city: n}
nb_seen = np.zeros(N)
for a, b in und_arr:
    ia = pos.get(a)
    if ia is None or b not in lab_map:                       # neighbor must be labeled
        continue
    if fold_arr[ia] == fold.get(b, -2):                      # exclude same-fold (and own) labels
        continue
    nb_counts.setdefault(a, {}); d = nb_counts[a]
    cb = lab_map[b]; d[cb] = d.get(cb, 0) + 1
    nb_seen[ia] += 1
nb_modal, nb_topfrac, nb_entropy, nb_cov = {}, {}, {}, {}
deg_tot = feat["deg_total"].to_dict()
for a, d in nb_counts.items():
    s = pd.Series(d); tot = s.sum()
    nb_modal[a] = s.idxmax(); nb_topfrac[a] = s.max() / tot; nb_entropy[a] = ent(s)
    nb_cov[a] = tot / max(1, deg_tot.get(a, 1))
feat["nb_modal_city"] = pd.Series(nb_modal)
feat["nb_top_frac"] = pd.Series(nb_topfrac)
feat["nb_entropy"] = pd.Series(nb_entropy)
feat["nb_labeled_frac"] = pd.Series(nb_cov)

# ---------------------------------------------------------------- GEO --------
tp = L("tower_pings").merge(L("towers")[["tower_id", "region_code"]], on="tower_id", how="left")
tg = tp.groupby(["person_id", "region_code"]).size()
feat["tower_modal_region"] = tg.groupby(level=0).idxmax().map(lambda t: t[1])
feat["tower_top_frac"] = tg.groupby(level=0).max() / tg.groupby(level=0).sum()
feat["tower_entropy"] = tg.groupby(level=0).agg(ent)

merch = L("merchants")[["merchant_id", "merchant_city", "is_online"]].copy()
merch["mc"] = merch["merchant_city"].map(norm_city)          # normalize messy casing/space
tx = L("transactions").merge(merch, on="merchant_id", how="left")
feat["merch_online_frac"] = tx.groupby("person_id")["is_online"].mean()
mg = tx.dropna(subset=["mc"]).groupby(["person_id", "mc"]).size()
feat["merch_modal_city"] = mg.groupby(level=0).idxmax().map(lambda t: t[1])
feat["merch_top_frac"] = mg.groupby(level=0).max() / mg.groupby(level=0).sum()

# ---------------------------------------------------------------- TEMPORAL ---
ae = L("app_events")
ae_h = ae.assign(h=ae["ts"].dt.hour)
feat["act_hour_mean"] = ae_h.groupby("person_id")["h"].mean()
feat["act_hour_std"] = ae_h.groupby("person_id")["h"].std()
feat["app_event_count"] = ae.groupby("person_id").size()
sign = pd.to_datetime(people["signup_at"], errors="coerce")
feat["tenure_days"] = (pd.Timestamp("2025-07-01") - sign).dt.days

# ---------------------------------------------------------------- RED-HERRING
feat["age"] = pd.to_numeric(people["age"], errors="coerce")
feat["declared_language"] = people["declared_language"]
feat["device"] = people["device"]
dev = L("device_info").drop_duplicates("person_id").set_index("person_id")
feat["screen_in"] = dev["screen_in"]
feat["battery"] = dev["battery"]

# ---------------------------------------------------------------- LEAK TRAPS -
feat["ip_city"] = ip["ip_city"]                              # TRAP: ~ the answer
feat["declared_city"] = people["raw_city_text"].map(norm_city)  # near-leak self-report
# weather joined via the person's HOME region (= the answer) -> deliberate leak.
# region annual-mean temp + tiny jitter so it stays continuous (numeric AUC fires).
reg_temp = L("weather_logs").groupby("region_code")["temp_c"].mean()
city_region = {f"city_{i:02d}": f"metro_{i // 5}" for i in range(50)}
home_region = labels.map(city_region)                        # uses the label
jit = pd.Series(np.random.default_rng(0).normal(0, 0.3, len(labels)), index=labels.index)
feat["wx_home_region_temp"] = home_region.map(reg_temp) + jit

# ---------------------------------------------------------------- assemble ---
feat["ref_ts"] = sign                                        # stability time axis
feat = feat.reset_index().rename(columns={"index": "person_id"})
feat = feat.sort_values("person_id").reset_index(drop=True)
out = os.path.join(HERE, "feature_table.parquet")
feat.to_parquet(out, index=False)

nfeat = feat.shape[1] - 2
ncov = feat["nb_modal_city"].notna().mean()
print(f"wrote {out}")
print(f"rows={len(feat):,} (all people)  features={nfeat}")
print(f"labeled rows for screening: {labels.index.isin(feat['person_id']).sum():,} "
      f"over {labels.nunique()} cities")
print(f"neighbor-feature coverage (>=1 labeled OOF neighbor): {ncov:.1%}")
print("families: text-content, documents, graph(OOF neighbor), tower/merchant geo, "
      "temporal, demographic/device(red-herring), ip_geo/weather/declared(leaks)")
