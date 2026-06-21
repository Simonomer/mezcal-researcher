#!/usr/bin/env python3
"""
generate_data.py — HARD node-classification problem: predict a person's
`home_city` (50 cities, but GT on only 30) from the *text* they exchange, the
communication graph, sparse documents about them, and messy auxiliary tables.

What makes it hard / realistic:
  - 50 cities grouped into 10 metros (5 each). Within-metro cities are
    confusable "sisters"; a per-person home-affinity blurs ALL channels at once.
  - GT exists for only 30 of the 50 cities, with a NORMAL-shaped (bell, heavily
    imbalanced) label frequency + a floor; the other 20 cities' residents appear
    only as unlabeled graph/text context. Partial coverage + ~5% label noise.
  - The predictive signal lives mostly in TEXT: per-city flavor tokens, explicit
    city mentions (a near-leak), and language. Most messages are generic noise.
  - Messiness everywhere: nulls, duplicate edges/rows, inconsistent city casing &
    typos, mixed-type columns, online merchants with null city.
  - Three leak traps: per-person `ip_geo` (≈ the answer), per-region `weather`,
    and self-reported `raw_city_text` (strong but messy near-leak).

Writes one parquet per table to data/tables/ and prints a tagged data dictionary.
Deterministic; runtime < ~90s.
"""

import os
import numpy as np
import pandas as pd

SEED = 11
rng = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- topology ---
N = 9000
N_CITIES, METRO_SIZE = 50, 5
CITIES = [f"city_{i:02d}" for i in range(N_CITIES)]
METRO = {c: i // METRO_SIZE for i, c in enumerate(CITIES)}          # 10 metros
REGIONS = [f"metro_{m}" for m in range(N_CITIES // METRO_SIZE)]
CITY_REGION = {c: f"metro_{METRO[c]}" for c in CITIES}
BASE_TEMP = {r: 6.0 + 2.7 * i for i, r in enumerate(REGIONS)}       # distinct climates
BASE_PRECIP = {r: 30.0 + 11.0 * ((i * 7) % 10) for i, r in enumerate(REGIONS)}
LANGS = [f"lang_{i}" for i in range(6)]
# per-metro base language mix (overlapping); city perturbs it slightly
METRO_LANG = {m: rng.dirichlet(np.ones(6) * 1.3) for m in range(10)}

BASE_DATE = np.datetime64("2024-01-01")
SPAN = 540


def rand_ts(n, null_frac=0.0):
    ts = BASE_DATE + rng.integers(0, SPAN * 86400, n).astype("timedelta64[s]")
    out = pd.Series(ts)
    if null_frac:
        out[rng.random(n) < null_frac] = pd.NaT
    return out


def save(df, name, tag):
    df.to_parquet(os.path.join(OUT, f"{name}.parquet"), index=False)
    print(f"[{tag:14}] {name:14} rows={len(df):>7,}  cols=({', '.join(df.columns)})")


# ---------------------------------------------------------- labels & people ---
labeled_cities = sorted(rng.choice(CITIES, size=30, replace=False))
# normal-shaped counts across the 30 labeled cities, floored so 5-fold CV works
rank = np.arange(30)
wbell = np.exp(-((rank - 14.5) / 7.0) ** 2)
counts = np.maximum(18, (wbell / wbell.sum() * 5200).round().astype(int))
labeled_true = np.repeat(labeled_cities, counts)
n_lab = len(labeled_true)
n_unlab = N - n_lab
# unlabeled residents over ALL 50 cities (so the 20 GT-less cities are populated)
cw = np.exp(-((np.arange(N_CITIES) - 24.5) / 13.0) ** 2) + 0.25
unlab_true = rng.choice(CITIES, size=n_unlab, p=cw / cw.sum())
true_city = np.concatenate([labeled_true, unlab_true])
is_labeled = np.concatenate([np.ones(n_lab, bool), np.zeros(n_unlab, bool)])
order = rng.permutation(N)
true_city, is_labeled = true_city[order], is_labeled[order]
person_id = np.array([f"p_{i:05d}" for i in range(N)])
true_idx = np.array([CITIES.index(c) for c in true_city])

# observed label = true city, with ~5% noise to a same-metro sister that is ALSO
# a labeled city (so GT stays confined to the 30 labeled cities).
labeled_set = set(labeled_cities)
home_obs = np.array(true_city, dtype=object)
noise = is_labeled & (rng.random(N) < 0.05)
for i in np.where(noise)[0]:
    sis = [c for c in CITIES if METRO[c] == METRO[true_city[i]]
           and c != true_city[i] and c in labeled_set]
    if sis:
        home_obs[i] = rng.choice(sis)
home_obs[~is_labeled] = None

# per-person home affinity: low = "border", blurs every channel toward sisters
home_aff = np.clip(rng.normal(0.62, 0.20, N), 0.15, 0.95)
is_traveler = rng.random(N) < 0.12
home_aff[is_traveler] *= 0.6

# --------------------------------------------------------------- text vocab ---
GENERIC = ("hey hi ok yeah lol thanks later tonight lunch coffee meeting work "
           "tired busy sure maybe call text soon home tomorrow weekend game "
           "dinner train bus rain sun cold market kids school deadline").split()
JUNK = ["", "...", "??", " ", "asdf", "??!", "  "]
EMOJI = ["", "", "", "", "\U0001F600", "\U0001F44D", "\U0001F525"]


def flavor_tokens(c):
    return [f"land_{c}", f"team_{c}", f"slang_{c}", f"dia_metro_{METRO[c]}"]


def off_city(c):
    """A confusable other city: usually a same-metro sister."""
    if rng.random() < 0.7:
        sis = [x for x in CITIES if METRO[x] == METRO[c] and x != c]
        return rng.choice(sis)
    return CITIES[rng.integers(N_CITIES)]


def make_text(p):
    """Build one message's text for person p (true city c), flavored + noisy."""
    c = true_city[p]
    L = int(rng.integers(4, 13))
    toks = list(rng.choice(GENERIC, size=L))
    if rng.random() < home_aff[p] * 0.5:                 # city-flavored tokens
        src = c if rng.random() < home_aff[p] else off_city(c)
        toks += list(rng.choice(flavor_tokens(src), size=rng.integers(1, 3)))
    if rng.random() < 0.06:                              # explicit city mention (near-leak)
        ment = c if rng.random() < 0.75 else off_city(c)
        toks.append(f"cityname_{ment}")
    if rng.random() < 0.15:
        toks.append(rng.choice(["http://t.co/x", "www.site/q"]))  # url noise
    toks.append(rng.choice(EMOJI))
    toks.append(rng.choice(JUNK))
    s = " ".join(t for t in toks if t)
    return s if rng.random() > 0.04 else None            # ~4% null/empty text


def person_lang(p):
    base = METRO_LANG[METRO[true_city[p]]].copy()
    return rng.choice(LANGS, p=base / base.sum())


# ---------------------------------------------------------------- people -----
age = rng.normal(38, 13, N)
age[rng.random(N) < 0.05] = np.nan                       # messy: missing ages
age = np.where(np.isnan(age), np.nan, np.clip(age, 16, 92))
declared = np.array([None] * N, dtype=object)            # raw self-reported city (messy)
for i in range(N):
    r = rng.random()
    if r < 0.30:
        continue                                         # ~30% null
    city = true_city[i] if rng.random() < 0.82 else off_city(true_city[i])
    txt = city.replace("city_", "City ")                 # "City 07"
    if rng.random() < 0.25:
        txt = txt.upper()
    if rng.random() < 0.15:
        txt = "  " + txt + " "                            # stray whitespace
    if rng.random() < 0.10:
        txt = txt[:-1] + "x"                              # typo
    declared[i] = txt
people = pd.DataFrame(dict(
    person_id=person_id, age=age,
    signup_at=rand_ts(N, null_frac=0.03),
    device=rng.choice(["android", "ios", "kaios", None], N, p=[0.55, 0.33, 0.07, 0.05]),
    declared_language=[person_lang(i) for i in range(N)],
    raw_city_text=declared))
# messy: a few duplicate person rows
dups = people.iloc[rng.choice(N, 60, replace=False)]
people = pd.concat([people, dups], ignore_index=True)
save(people, "people", "MIXED/MESSY")

# LABEL — only labeled people; 30 cities; never a feature
lab = pd.DataFrame(dict(person_id=person_id[is_labeled],
                        home_city=home_obs[is_labeled].astype(str)))
save(lab, "home_city", "LABEL")

# ---------------------------------------------------------------- messages ---
# graph edges biased by home affinity (same city) else a confusable sister;
# each carries TEXT content + a language tag. The core signal + the graph.
N_MSG = 78_000
snd = rng.integers(0, N, N_MSG)
same = rng.random(N_MSG) < home_aff[snd]
by_city = {c: np.where(true_city == c)[0] for c in CITIES}
rcv = np.empty(N_MSG, dtype=int)
for e in range(N_MSG):
    s = snd[e]
    pool = by_city[true_city[s]] if same[e] else by_city[off_city(true_city[s])]
    rcv[e] = pool[rng.integers(len(pool))] if len(pool) else rng.integers(N)
keep = snd != rcv
snd, rcv = snd[keep], rcv[keep]
M = len(snd)
messages = pd.DataFrame(dict(
    sender_id=person_id[snd], recipient_id=person_id[rcv],
    ts=rand_ts(M, null_frac=0.02),
    text=[make_text(s) for s in snd],
    lang=[person_lang(s) for s in snd],
    channel=rng.choice(["chat", "sms", "email"], M, p=[0.6, 0.3, 0.1])))
messages = pd.concat([messages, messages.iloc[rng.choice(M, 800, replace=False)]],
                     ignore_index=True)                  # messy: duplicate messages
save(messages, "messages", "RELEVANT/TEXT")

contacts = {}
for a, b in zip(snd, rcv):
    contacts.setdefault(a, []).append(b)
    contacts.setdefault(b, []).append(a)

# ------------------------------------------------------------- person_docs ---
# Sparse free-text documents ABOUT people (~25% coverage): bios/reports.
doc_ids, doc_txt = [], []
for i in range(N):
    if rng.random() > 0.25:
        continue
    c = true_city[i]
    body = list(rng.choice(GENERIC, size=rng.integers(8, 20)))
    body += list(rng.choice(flavor_tokens(c if rng.random() < home_aff[i] else off_city(c)),
                            size=rng.integers(1, 4)))
    if rng.random() < 0.20:
        body.append(f"cityname_{c}")                     # doc sometimes names the city
    doc_ids.append(person_id[i]); doc_txt.append("report: " + " ".join(body))
save(pd.DataFrame(dict(person_id=doc_ids, doc_text=doc_txt)), "person_docs", "RELEVANT/TEXT")

# ------------------------------------------------------------- towers/pings ---
N_TOWERS = 400
tw_city = rng.choice(CITIES, N_TOWERS)
tw_region = np.array([CITY_REGION[c] if rng.random() < 0.85
                      else rng.choice(REGIONS) for c in tw_city])
tower_id = np.array([f"tw_{i:04d}" for i in range(N_TOWERS)])
save(pd.DataFrame(dict(tower_id=tower_id, region_code=tw_region,
                       lat=(10 + rng.random(N_TOWERS) * 40).round(4),
                       lon=(-90 + rng.random(N_TOWERS) * 30).round(4))), "towers", "DIM")
tw_by_region = {r: np.where(tw_region == r)[0] for r in REGIONS}
pc, pt = [], []
for i in range(N):
    n = rng.integers(5, 14)
    for _ in range(n):
        reg = CITY_REGION[true_city[i]] if rng.random() < min(0.95, home_aff[i] * 1.2) \
            else CITY_REGION[off_city(true_city[i])]
        pool = tw_by_region[reg] if len(tw_by_region[reg]) else range(N_TOWERS)
        pt.append(tower_id[pool[rng.integers(len(pool))]]); pc.append(person_id[i])
save(pd.DataFrame(dict(person_id=pc, tower_id=pt, ts=rand_ts(len(pc))),
                  ), "tower_pings", "RELEVANT")

# ------------------------------------------------------ merchants/transactions
N_MERCH = 700
m_online = rng.random(N_MERCH) < 0.22
m_city = np.array([None if m_online[j] else rng.choice(CITIES) for j in range(N_MERCH)])
# messy: inconsistent casing / whitespace on merchant_city
m_city_msgy = np.array([None if v is None else
                        (v.upper() if rng.random() < 0.3 else
                         (" " + v if rng.random() < 0.3 else v)) for v in m_city], dtype=object)
mid = np.array([f"m_{i:04d}" for i in range(N_MERCH)])
save(pd.DataFrame(dict(merchant_id=mid, merchant_city=m_city_msgy,
                       category=rng.choice(["grocery", "fuel", "food", "retail", "transit"], N_MERCH),
                       is_online=m_online)), "merchants", "DIM")
m_by_city = {c: np.where(m_city == c)[0] for c in CITIES}
m_online_idx = np.where(m_online)[0]
tc, tm = [], []
for i in range(N):
    for _ in range(rng.integers(3, 18)):
        if rng.random() < 0.15:
            j = m_online_idx[rng.integers(len(m_online_idx))]
        else:
            cc = true_city[i] if rng.random() < min(0.9, home_aff[i]) else off_city(true_city[i])
            pool = m_by_city[cc] if len(m_by_city[cc]) else m_online_idx
            j = pool[rng.integers(len(pool))]
        tc.append(person_id[i]); tm.append(mid[j])
save(pd.DataFrame(dict(person_id=tc, merchant_id=tm,
                       amount=np.round(rng.lognormal(3, 0.8, len(tc)), 2),
                       ts=rand_ts(len(tc)))), "transactions", "PARTIAL")

# ------------------------------------------------------------- app/device ----
ec, et = [], []
for i in range(N):
    n = rng.integers(8, 30)
    ec.extend([person_id[i]] * n)
    et.extend(rng.choice(["open", "scroll", "search", "share", "notif"], n))
save(pd.DataFrame(dict(person_id=ec, event_type=et, ts=rand_ts(len(ec)))),
     "app_events", "RED-HERRING")
save(pd.DataFrame(dict(person_id=person_id,
                       os_version=rng.choice(["12", "13", "14", "15", "16"], N),
                       screen_in=np.round(np.clip(rng.normal(6.1, 0.6, N), 4, 7.9), 2),
                       battery=rng.integers(55, 101, N))), "device_info", "RED-HERRING")

# ------------------------------------------------------------- leak traps -----
# weather: per region+date (TRAP — joinable to a person only via their region).
dates = pd.date_range("2024-01-01", periods=SPAN, freq="D")
season = 6 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.0)
wx = [pd.DataFrame(dict(region_code=r, date=dates,
                        temp_c=(BASE_TEMP[r] + season + rng.normal(0, 1, SPAN)).round(2),
                        precip_mm=np.clip(BASE_PRECIP[r] + rng.normal(0, 14, SPAN), 0, None).round(1)))
      for r in REGIONS]
save(pd.concat(wx, ignore_index=True), "weather_logs", "TRAP/LEAK")
# ip_geo: per-person last-seen IP city (TRAP — basically the answer, ~88% exact)
ip_city = np.array([true_city[i] if rng.random() < 0.88 else CITIES[rng.integers(N_CITIES)]
                    for i in range(N)], dtype=object)
ip_city[rng.random(N) < 0.08] = None                     # messy nulls
save(pd.DataFrame(dict(person_id=person_id, ip_city=ip_city,
                       ip_asn=rng.integers(1000, 9999, N))), "ip_geo", "TRAP/LEAK")

# ----------------------------------------------------------------- sanity ----
homo = np.mean([np.mean(true_city[contacts[i]] == true_city[i])
                for i in range(N) if i in contacts])
lc = pd.Series(home_obs[is_labeled]).value_counts()
print(f"\npeople={N:,}  labeled={is_labeled.sum():,} over {lc.size} cities "
      f"(of {N_CITIES})  ·  unlabeled context={(~is_labeled).sum():,}")
print(f"label counts: max={lc.max()} min={lc.min()} median={int(lc.median())} (normal-shaped)")
print(f"realized comm homophily (same-city contacts): {homo:.1%}")
print(f"wrote 13 tables to {OUT}")
