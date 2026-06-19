#!/usr/bin/env python3
"""
generate_data.py — synthetic "home-city" classification problem.

Task: predict each civilian's home_city (5 imbalanced classes) from a
communication graph + auxiliary tables. The data is deliberately HARD:

  - noisy homophily: only ~65% of a civilian's contacts share their city
  - travelers (~12%): mobility/spend smeared across cities
  - overlapping classes: language + behavior distributions overlap
  - leakage traps: a per-region weather table that can only be joined to a
    civilian *through their home region* (i.e. through the answer), and
    near-leak strong location features (tower region, merchant city).

Writes one parquet per table to data/tables/ and prints a one-line data
dictionary (with a relevance tag) per table. Deterministic; runtime < 2 min.
"""

import os
import numpy as np
import pandas as pd

SEED = 7
rng = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- topology ---
N_CIV = 7000
CITIES = ["city_A", "city_B", "city_C", "city_D", "city_E"]
CITY_P = np.array([0.35, 0.25, 0.20, 0.12, 0.08])           # imbalanced
REGION = {c: f"reg_{c[-1]}" for c in CITIES}                # 1:1 city<->region
REGIONS = [REGION[c] for c in CITIES]
# geographic centers (arbitrary lat/lon) per city
CENTER = {c: (10 + 4 * i, -70 - 5 * i) for i, c in enumerate(CITIES)}
# distinct base climate per region -> makes a region-joined weather feature a
# near-deterministic proxy for the label (the trap).
BASE_TEMP = dict(zip(REGIONS, [30.0, 22.0, 15.0, 8.0, 35.0]))
BASE_PRECIP = dict(zip(REGIONS, [20.0, 90.0, 140.0, 60.0, 10.0]))
# overlapping, only mildly city-correlated language mix (3 languages)
LANGS = ["lang_x", "lang_y", "lang_z"]
LANG_MIX = {
    "city_A": [0.55, 0.30, 0.15], "city_B": [0.45, 0.35, 0.20],
    "city_C": [0.40, 0.35, 0.25], "city_D": [0.35, 0.40, 0.25],
    "city_E": [0.30, 0.40, 0.30],
}
# small per-city timezone offset (hours) -> a faint activity-hour signal
TZ_OFFSET = dict(zip(CITIES, [0, 1, 2, 2, 3]))

# Where off-city activity leaks (over OTHER cities). This is what makes the
# problem HARD: instead of spreading by population, a civilian's non-home
# contacts/pings/spend concentrate on a "sister" city -> A<->B and C<->D are
# confusable, so no single feature decides them. E leaks broadly -> it stays
# separable (its features will look near-leak-strong and get flagged).
CONFUSE = {
    "city_A": {"city_B": 0.70, "city_C": 0.12, "city_D": 0.10, "city_E": 0.08},
    "city_B": {"city_A": 0.70, "city_C": 0.12, "city_D": 0.10, "city_E": 0.08},
    "city_C": {"city_D": 0.68, "city_A": 0.15, "city_B": 0.12, "city_E": 0.05},
    "city_D": {"city_C": 0.68, "city_A": 0.15, "city_B": 0.12, "city_E": 0.05},
    "city_E": {"city_A": 0.35, "city_B": 0.27, "city_C": 0.22, "city_D": 0.16},
}
CONF_C = {c: list(d.keys()) for c, d in CONFUSE.items()}
CONF_P = {c: np.array(list(d.values())) for c, d in CONFUSE.items()}

BASE_DATE = np.datetime64("2024-01-01")
SPAN_DAYS = 540  # ~18 months of activity


def off_city(home, size):
    """Sample `size` non-home cities for `home`, weighted toward its sister."""
    return rng.choice(CONF_C[home], size=size, p=CONF_P[home])


def rand_ts(n):
    """Random timestamps over the activity window."""
    offs = rng.integers(0, SPAN_DAYS * 24 * 3600, size=n)
    return BASE_DATE + offs.astype("timedelta64[s]")


def save(df, name, tag):
    path = os.path.join(OUT, f"{name}.parquet")
    df.to_parquet(path, index=False)
    cols = ", ".join(df.columns)
    print(f"[{tag:13}] {name:13} rows={len(df):>7,}  cols=({cols})")


# ---------------------------------------------------------------- civilians ---
civ_id = np.array([f"civ_{i:05d}" for i in range(N_CIV)])
home_city = rng.choice(CITIES, size=N_CIV, p=CITY_P)
is_traveler = rng.random(N_CIV) < 0.12                       # ~12% travelers
# Per-civilian home affinity drives ALL channels (comms, towers, spend) together,
# so a low-affinity ("border") civilian looks half-sister *everywhere* at once.
# That makes the channels' noise correlated and the confusion irresolvable —
# the real source of difficulty (independent per-channel noise would average out).
base_homophily = np.clip(rng.normal(0.66, 0.18, N_CIV), 0.20, 0.92)
base_homophily[is_traveler] *= 0.6                           # travelers roam most

age = np.clip(rng.normal(38, 14, N_CIV), 18, 90).round().astype(int)
created = rand_ts(N_CIV)
device_type = rng.choice(["android", "ios", "kaios"], N_CIV, p=[0.6, 0.35, 0.05])
language_pref = np.array([
    rng.choice(LANGS, p=LANG_MIX[c]) for c in home_city
])

civilians = pd.DataFrame(dict(
    civ_id=civ_id, age=age, account_created_at=created,
    device_type=device_type, language_pref=language_pref))
save(civilians, "civilians", "WEAK/PARTIAL")

# LABEL — written separately, never to be used as a feature.
save(pd.DataFrame(dict(civ_id=civ_id, home_city=home_city)), "home_city", "LABEL")

# ----------------------------------------------------------------- comms -----
# Each edge: with prob rho a same-city callee; otherwise a callee in a
# confusion-weighted OTHER city (mostly the home city's sister, via CONFUSE).
# rho is lower for travelers. Noise is NOT population-spread, so sister cities
# stay confusable -> the neighbor feature is strong but not decisive.
N_EDGES = 60_000
by_city = {c: np.where(home_city == c)[0] for c in CITIES}
callers = rng.integers(0, N_CIV, N_EDGES)
same_city = rng.random(N_EDGES) < base_homophily[callers]    # per-civ homophily
caller_city = home_city[callers]
callee_city = caller_city.copy()
for c in CITIES:                                             # vectorized off-city draw
    msk = (~same_city) & (caller_city == c)
    if msk.any():
        callee_city[msk] = off_city(c, int(msk.sum()))
callees = np.array([by_city[cc][rng.integers(len(by_city[cc]))] for cc in callee_city])
# drop self-loops
keep = callers != callees
callers, callees = callers[keep], callees[keep]
m = len(callers)
comms = pd.DataFrame(dict(
    caller_id=civ_id[callers], callee_id=civ_id[callees], ts=rand_ts(m),
    duration_s=np.clip(rng.exponential(120, m), 1, 3600).round().astype(int),
    channel=rng.choice(["call", "sms"], m, p=[0.45, 0.55])))
save(comms, "comms", "RELEVANT")

# realized per-civilian homophily, for sanity (printed below).
contacts = {}
for a, b in zip(callers, callees):
    contacts.setdefault(a, []).append(b)
    contacts.setdefault(b, []).append(a)

# ------------------------------------------------------------- towers/pings ---
N_TOWERS = 220
tower_city = rng.choice(CITIES, N_TOWERS, p=CITY_P)          # bigger city -> more towers
# soft region: a tower mostly carries its city's region, sometimes a neighbor's
tower_region = np.array([
    REGION[tc] if rng.random() < 0.88 else REGION[rng.choice(CITIES)]
    for tc in tower_city])
tw_lat = np.array([CENTER[c][0] + rng.normal(0, 0.4) for c in tower_city])
tw_lon = np.array([CENTER[c][1] + rng.normal(0, 0.4) for c in tower_city])
tower_id = np.array([f"tw_{i:04d}" for i in range(N_TOWERS)])
towers = pd.DataFrame(dict(
    tower_id=tower_id, lat=tw_lat.round(4), lon=tw_lon.round(4),
    region_code=tower_region))
save(towers, "towers", "DIM")

tw_by_city = {c: np.where(tower_city == c)[0] for c in CITIES}
ping_civ, ping_tw = [], []
for i in range(N_CIV):
    n = rng.integers(8, 16)
    hc = home_city[i]
    # ping home-share scales with the civ's affinity (towers a touch stronger
    # than comms); border/traveler civs ping the sister region just as often.
    p_home = min(0.95, base_homophily[i] * 1.15)
    chosen = np.where(rng.random(n) < p_home, hc, off_city(hc, n))
    for c in chosen:
        pool = tw_by_city[c] if len(tw_by_city[c]) else tw_by_city[hc]
        ping_tw.append(tower_id[pool[rng.integers(len(pool))]])
        ping_civ.append(civ_id[i])
tower_pings = pd.DataFrame(dict(
    civ_id=ping_civ, tower_id=ping_tw, ts=rand_ts(len(ping_civ))))
save(tower_pings, "tower_pings", "RELEVANT-STRONG")

# ------------------------------------------------------ merchants/transactions
N_MERCH = 450
is_online = rng.random(N_MERCH) < 0.20
merch_city = np.array([
    None if is_online[j] else rng.choice(CITIES, p=CITY_P) for j in range(N_MERCH)])
merch_id = np.array([f"mch_{i:04d}" for i in range(N_MERCH)])
merchants = pd.DataFrame(dict(
    merchant_id=merch_id, merchant_city=merch_city,
    category=rng.choice(["grocery", "fuel", "retail", "food", "transit"], N_MERCH),
    is_online=is_online))
save(merchants, "merchants", "DIM")

mch_by_city = {c: np.where(merch_city == c)[0] for c in CITIES}
mch_online = np.where(is_online)[0]
tx_civ, tx_mch, tx_amt = [], [], []
for i in range(N_CIV):
    n = rng.integers(5, 22)
    hc = home_city[i]
    p_home = min(0.90, base_homophily[i] * 0.95)   # noisier location cue than towers
    for _ in range(n):
        if rng.random() < 0.15:                              # online (null city)
            j = mch_online[rng.integers(len(mch_online))]
        else:
            cc = hc if rng.random() < p_home else off_city(hc, 1)[0]
            pool = mch_by_city[cc] if len(mch_by_city[cc]) else mch_online
            j = pool[rng.integers(len(pool))]
        tx_civ.append(civ_id[i]); tx_mch.append(merch_id[j])
        tx_amt.append(round(float(rng.lognormal(3.0, 0.8)), 2))
transactions = pd.DataFrame(dict(
    civ_id=tx_civ, merchant_id=tx_mch, amount=tx_amt, ts=rand_ts(len(tx_civ))))
save(transactions, "transactions", "PARTIAL")

# ------------------------------------------------------------- app_events ----
# Only signal is activity-hour (a faint timezone proxy); event_type is noise.
ev_civ, ev_type, ev_ts = [], [], []
ev_types = ["open", "scroll", "search", "notif_open", "share"]
for i in range(N_CIV):
    n = rng.integers(10, 40)
    # a city-independent personal schedule dominates; the per-city timezone
    # offset is only a faint shift on top -> activity-hour is a weak location cue.
    personal = rng.normal(13, 5)
    base_hour = rng.normal(personal + TZ_OFFSET[home_city[i]], 2.5, n) % 24
    day = rng.integers(0, SPAN_DAYS, n)
    ts = BASE_DATE + (day * 24 * 3600 + (base_hour * 3600).astype(int)).astype("timedelta64[s]")
    ev_civ.extend([civ_id[i]] * n)
    ev_type.extend(rng.choice(ev_types, n))
    ev_ts.extend(ts)
app_events = pd.DataFrame(dict(civ_id=ev_civ, event_type=ev_type, ts=ev_ts))
save(app_events, "app_events", "WEAK/RED-HERRING")

# ------------------------------------------------------------- device_info ---
# Pure distractor: no location signal at all.
device_info = pd.DataFrame(dict(
    civ_id=civ_id,
    os_version=rng.choice(["12.1", "13.0", "14.2", "15.1", "16.0"], N_CIV),
    screen_size_in=np.clip(rng.normal(6.1, 0.6, N_CIV), 4.0, 7.8).round(2),
    battery_health=rng.integers(60, 101, N_CIV)))
save(device_info, "device_info", "RED-HERRING")

# ------------------------------------------------------------- weather_logs --
# TRAP: keyed by region_code+date, NOT by civilian. Joining it to a civilian
# requires their region — i.e. the answer. Distinct per-region climate makes the
# (leaky) region-joined feature a near-perfect label proxy.
dates = pd.date_range("2024-01-01", periods=SPAN_DAYS, freq="D")
doy = dates.dayofyear.to_numpy()
season = 6.0 * np.sin(2 * np.pi * (doy - 80) / 365.0)        # shared seasonal swing
rows = []
for reg in REGIONS:
    temp = BASE_TEMP[reg] + season + rng.normal(0, 1.0, SPAN_DAYS)
    precip = np.clip(BASE_PRECIP[reg] + rng.normal(0, 15, SPAN_DAYS), 0, None)
    rows.append(pd.DataFrame(dict(region_code=reg, date=dates,
                                  temp_c=temp.round(2), precip_mm=precip.round(1))))
weather_logs = pd.concat(rows, ignore_index=True)
save(weather_logs, "weather_logs", "TRAP")

# ----------------------------------------------------------------- sanity ----
homo = np.mean([
    np.mean(home_city[contacts[i]] == home_city[i])
    for i in range(N_CIV) if i in contacts])
print(f"\nrealized mean per-civilian comm homophily: {homo:.2%}  "
      f"(travelers={is_traveler.mean():.1%})")
print(f"class balance: " + ", ".join(
    f"{c}={np.mean(home_city==c):.0%}" for c in CITIES))
print(f"wrote 10 tables to {OUT}")
