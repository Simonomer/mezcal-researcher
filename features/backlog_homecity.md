# Feature backlog — predict `home_city`

**Task.** Classify each civilian's `home_city` (5 imbalanced classes:
A≈35% / B≈25% / C≈20% / D≈13% / E≈8%) from a communication graph plus auxiliary
tables. Local parquet in `data/tables/`. Profiled with
`scripts/inspect_tables.py data/tables`.

## Data map (wiring — CONFIRMED before ideating)

- **Entity** = `civ_id` (7,000 civilians; one row per civilian in the feature
  table). **Label** = `home_city.home_city` — *never* a feature.
- **Graph / edges** = `comms.caller_id → callee_id` (≈60k directed edges; treat
  undirected for "contacts"). The auto key-scan only matched `civ_id`; the edge
  keys (`caller_id`,`callee_id`) and the dim keys below are **named differently
  from the entity key**, so they were confirmed by hand — wrong key = garbage.
- **Dim joins:** `tower_pings.tower_id → towers.tower_id` (gives each ping a
  `region_code`); `transactions.merchant_id → merchants.merchant_id` (gives each
  txn a `merchant_city`, null for `is_online=True`).
- **Grain:** `comms`, `tower_pings`, `transactions`, `app_events` are event-level
  (many rows per civ); `civilians`, `device_info`, `home_city` are one row per
  civ; `towers`, `merchants` are dims; `weather_logs` is keyed by
  `region_code × date` (NOT by civ — see trap below).
- **Time.** No per-entity prediction cutoff in this problem, so the time fence is
  not the active leakage rule. **The active rule is the train-only neighbor-label
  constraint:** any "city of your contacts" feature is label-derived and must be
  computed with *training-fold labels only* (out-of-fold), never the entity's own
  label or any test-fold label. `account_created_at` is kept only as a stability
  time axis for the screen, not as signal.

## Candidates (ranked by signal ÷ cost)

Scale tags: `spark-native` (groupBy/join aggregates), `graphframes`
(label-prop / centrality, reformulate as iterative joins on Connect),
`sampled-egonet` (driver-side on a sampled subgraph). Everything below is
materialized in pandas for this local worked example; the sketch is the Spark
shape.

| # | feature | family | why it separates cities | compute sketch | scale | leakage check | prio |
|---|---|---|---|---|---|---|---|
| 1 | `nb_frac_city_{A..E}` | neighbor / homophily | ~64% of contacts share your city → the per-city contact mix peaks on your own city | join comms to train labels, groupBy civ, normalize city counts | spark-native | **LABEL-DERIVED → train-fold-only (OOF); exclude own + test labels** | **1** |
| 2 | `nb_modal_city` | neighbor / homophily | the single most-common contact city is your city ~most of the time | argmax of #1 | spark-native | same as #1 (OOF) | **1** |
| 3 | `nb_city_entropy` | neighbor / homophily | travelers / weakly-homophilic civs have flatter contact-city mixes | entropy of #1 | spark-native | OOF (derived from labels) | 2 |
| 4 | `tower_frac_reg_{A..E}` | tower geography | pings concentrate on home-region towers (~90% for non-travelers) | `tower_pings⋈towers`, groupBy civ, normalize region counts | spark-native | **near-leak**: geography≈city but observed pre-label & per-civ → keep, expect high AUC, let screen flag | **1** |
| 5 | `tower_top_region` | tower geography | modal ping region ≈ home region | argmax of #4 | spark-native | near-leak (as #4) | 2 |
| 6 | `tower_region_entropy` | tower geography | travelers ping multiple regions → higher entropy | entropy of #4 | spark-native | none (shape, not label) | 2 |
| 7 | `merch_frac_city_{A..E}` | merchant geography | ~70% of in-store spend is in-city, but online (null city) + travelers blur it | `transactions⋈merchants` (drop null city), groupBy civ, normalize | spark-native | **near-leak**, noisier than towers → keep, screen | 2 |
| 8 | `merch_top_city` | merchant geography | modal spend city ≈ home city (noisy) | argmax of #7 | spark-native | near-leak | 3 |
| 9 | `merch_online_frac` | behavioral shape | fraction of spend online — pure habit, no location | `mean(is_online)` per civ | spark-native | none — expected weak/noise | 4 |
| 10 | `act_peak_hour`, `act_hour_mean/std` | activity timing | small per-city timezone offset shifts the activity-hour peak (faint) | hour-of-day stats over `app_events.ts` | spark-native | none — faint tz proxy | 3 |
| 11 | `language_pref` | demographic | city language mixes overlap → weak, non-decisive | passthrough from `civilians` | spark-native | none | 3 |
| 12 | `deg_total/in/out`, `call_sms_ratio` | degree / structural | sociability volume — not location; included as structural control | degree + channel ratio over `comms` | spark-native | none — expected weak | 4 |
| 13 | `contact_entropy` | counterparty diversity | shape of who-you-talk-to; weak location proxy | entropy of per-contact call counts | spark-native | none | 4 |
| 14 | PageRank / k-core / triangles | topology / role | structural role rarely encodes *location*; deferred (cost) | iterative joins (Connect) or driver-side on sampled egonet | graphframes / sampled-egonet | none | 5 (defer) |

### Controls / red-herrings — built on purpose so the screen can reject them

| # | feature | tag | expectation |
|---|---|---|---|
| 15 | `device_screen_size`, `device_battery_health`, `device_type`, `os_version` | RED-HERRING | no location signal → should **fail the null → drop** |
| 16 | `app_event_count`, `app_n_event_types` | RED-HERRING | raw telemetry volume → should **drop** |
| 17 | `age` | WEAK | ~no city signal → drop |
| 18 | `wx_home_region_temp`, `wx_home_region_precip` | **TRAP / LEAK** | `weather_logs` is per-`region_code`; the only way to attach it to a civ is via their region = **the answer**. Joined deliberately → expect AUC≈1.0 → screen must **flag as investigate (leakage)** |

## Build first (5–8)

1. **`nb_frac_city_{A..E}` + `nb_modal_city` + `nb_city_entropy`** — the
   train-only homophily mix (OOF). The headline graph feature; leakage-critical.
2. **`tower_frac_reg_{A..E}` + `tower_region_entropy` + `tower_top_region`** —
   the strongest legitimate location signal (near-leak; let the screen flag it).
3. **`merch_frac_city_{A..E}` + `merch_top_city`** — partial location signal.
4. **`act_peak_hour` + `language_pref`** — the weak/overlapping cues.
5. **The full control set (#15–18)** — device, app-volume, age, and the leaky
   weather join — so the screen demonstrably rejects noise and flags the leak.

_Backlog only — no compute here. Materialization is the next stage
(`features/build_features.py`), and the train-only OOF rule for #1–3 is enforced
in code there._
