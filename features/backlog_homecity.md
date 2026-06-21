# Feature backlog — predict `home_city` (50 cities, GT on 30)

**Task.** Classify a person's `home_city` from the **text they exchange**, the
communication graph, sparse documents, and messy auxiliary tables. The label
universe is **50 cities** but ground truth exists for only **30** (normal-shaped,
heavily imbalanced; counts 14–420). People in the other 20 cities appear only as
**unlabeled graph/text context**. ~9,000 people, 5,240 labeled. Local parquet in
`data/tables/`; profiled with `scripts/inspect_tables.py` (or
`scripts/profile_spark_tables.py` on catalog tables / HDFS / S3).

## Data map (wiring — CONFIRMED before ideating)

- **Entity** = `person_id`. **Label** = `home_city.home_city` (30 cities) — never a
  feature; built over ALL people, screened on the labeled subset.
- **Graph / edges** = `messages.sender_id → recipient_id` (~78k), each edge
  carrying **`text` + `lang`** — the core signal is the text content, not just
  the topology.
- **Dim joins:** `tower_pings.tower_id → towers.tower_id` (→ metro `region_code`);
  `transactions.merchant_id → merchants.merchant_id` (→ `merchant_city`).
- **Per-entity text:** `person_docs.person_id` (sparse, ~25%).
- **No prediction cutoff** → the active leakage rule is **train-only neighbor
  labels** (out-of-fold), now with PARTIAL coverage (most neighbors unlabeled).
- **Messy:** nulls, duplicate rows/edges, inconsistent city casing/typos in
  `raw_city_text` and `merchant_city`, mixed types, ~5% label noise.

## Candidates (ranked by signal ÷ cost)

| # | feature | family | why it separates cities | scale | leakage check | prio |
|---|---|---|---|---|---|---|
| 1 | `nb_modal_city`, `nb_top_frac`, `nb_entropy`, `nb_labeled_frac` | graph / neighbor | ~60% of contacts share your city → modal labeled-neighbor city | spark-native | **train-fold-only OOF; only labeled neighbors; partial coverage** | **1** |
| 2 | `txt_flavor_top_city`, `_top_frac`, `_entropy` | **text content** | city-flavored tokens (landmarks/slang/dialect) in messages | spark-native (tokenize + regex) | none (observable text) | **1** |
| 3 | `txt_modal_mention_city`, `txt_n_city_mentions` | **text content** | people sometimes name a city in chat | spark-native | **near-leak** (self-mention) → keep, let screen flag | 2 |
| 4 | `txt_lang_dominant`, `declared_language` | language | metro-correlated, overlapping language mix | spark-native | none | 3 |
| 5 | `tower_modal_region`, `tower_top_frac`, `tower_entropy` | geography | pings concentrate on the home metro | spark-native | near-leak (geography ≈ metro) → keep, screen | 2 |
| 6 | `merch_modal_city`, `merch_top_frac`, `merch_online_frac` | geography | in-store spend is mostly in-city (online null; messy casing) | spark-native | near-leak; normalize casing first | 2 |
| 7 | `doc_present`, `doc_len`, `doc_flavor_top_city`, `doc_n_city_mentions` | **documents** | bios/reports sometimes reveal the city (sparse) | sampled-egonet / batch NLP | near-leak via mentions | 3 |
| 8 | `deg_in`, `deg_out`, `deg_total`, `reciprocity` | degree / structural | sociability volume — weak location control | spark-native | none — expect weak/unstable | 4 |
| 9 | `act_hour_mean/std`, `tenure_days` | timing | faint timezone / account-age cues | spark-native | none | 4 |

### Controls / traps — built on purpose so the screen rejects / flags them

| feature | tag | expectation |
|---|---|---|
| `device`, `screen_in`, `battery`, `os_version`, `app_event_count`, `age`, `txt_total_tokens`, `txt_vocab_richness`, `txt_url_frac`, `txt_emoji_frac` | RED-HERRING | no location signal → **drop** (fail the null) |
| `ip_city` (last-seen IP) | **TRAP/LEAK** | ≈ the answer at a different time → high MI, flag/cut |
| `declared_city` (self-report) | **TRAP/LEAK** | messy self-report ≈ the answer → strong near-leak |
| `wx_home_region_temp` (region weather) | **TRAP/LEAK** | joinable only through the home region = the answer |

## Build first (5–8)

1. **`nb_*` (OOF, train-only, partial labels)** — the homophily mix; leakage-critical.
2. **`txt_flavor_*` + `txt_modal_mention_*` + `txt_lang_dominant`** — the text-content core.
3. **`tower_modal_region` + `merch_modal_city`** — geography.
4. **`doc_*`** — sparse document signal.
5. **The control + trap set** (`device`/`app`/`age`/style; `ip_city`/`declared_city`/`weather`) — so the screen demonstrably rejects noise and flags the leaks.

_Backlog only. Materialization is the next stage (`build_features.py` /
`build_features_spark.py`); the OOF train-only rule for #1 is enforced in code._
