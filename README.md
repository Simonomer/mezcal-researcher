# mezcal-researcher — automatic feature ideation with LLMs

A worked, end-to-end demonstration of a **closed-loop method for exploratory data
analysis**: an LLM enumerates a feature-hypothesis backlog grounded in the real
schema, a deterministic harness screens every candidate for real signal, results
loop back into re-ideation, and the human stays at just two gates — **confirm the
data wiring** and **set the keep threshold**.

📄 **Read the article:**
[`article/automatic-feature-ideation-with-llms.md`](article/automatic-feature-ideation-with-llms.md)
· [PDF (landscape)](article/automatic-feature-ideation-with-llms.pdf)

## The problem

Predict a person's `home_city` across **50 cities — but with ground truth for only
30** of them (a normal-shaped, heavily imbalanced label; the other 20 cities'
residents appear only as *unlabeled* graph/text context). The signal lives mostly
in the **text people exchange**; supporting tables add graph, geography, sparse
**documents**, and noise. It is deliberately hard and **messy**: ~60% homophily,
~12% travelers, 50 cities in 10 confusable metros, ~5% label noise, nulls,
duplicate rows/edges, inconsistent city casing/typos, mixed types — plus three
leak traps (`ip_geo` ≈ the answer, self-reported `raw_city_text`, region-joined
`weather`).

## How to use it

In normal use the workflow is **two prompts** in the Claude Code chat: the skills
run their scripts and return the report. The only manual steps are the two
judgment gates — **confirming the data wiring** and **setting the keep
threshold**. The loop is **ideate → materialize → validate → (loop back)**.

### Setup (once)

Install the two skills so `/ideate-features` and `/validate-signal` become real
commands (they live as `*__SKILL.md` files in this repo's root):

```bash
mkdir -p ~/.claude/skills/ideate-features/scripts ~/.claude/skills/validate-signal/scripts
cp ideate-features__SKILL.md  ~/.claude/skills/ideate-features/SKILL.md
cp scripts/inspect_tables.py scripts/profile_spark_tables.py ~/.claude/skills/ideate-features/scripts/
cp validate-signal__SKILL.md ~/.claude/skills/validate-signal/SKILL.md
cp scripts/signal_panel.py   ~/.claude/skills/validate-signal/scripts/

pip install pandas scikit-learn scipy matplotlib pyarrow      # + "pyspark[connect]" for Spark
export SPARK_REMOTE=sc://YOUR-HOST:15002                       # Spark only: your Connect endpoint
```

### The workflow

**1 · Ideate** — point the skill at your tables. Use catalog names:

> `/ideate-features predict home_city for person_id; tables warehouse.messages warehouse.people warehouse.home_city warehouse.tower_pings … — Spark Connect at sc://my-host:15002`

…or HDFS / S3 paths instead of a catalog:

> `/ideate-features predict home_city for person_id from s3://lake/telco/messages s3://lake/telco/people s3://lake/telco/home_city hdfs:///telco/tower_pings … — no time cutoff`

Claude profiles them on a 100k-row cluster-side sample (no full pull), proposes the wiring for your confirmation, then writes the backlog. Set the endpoint once with `export SPARK_REMOTE=sc://my-host:15002` to omit it from the prompt; a live Connect session needs no URL at all.

**2 · Materialize** — this step has no dedicated skill, since the feature logic is specific to your tables. Ask Claude to build it:

> `materialize the kept backlog rows into a feature table`

Claude writes and runs the feature code — which you review for the leakage-safe parts — and writes the table to the catalog, S3, or HDFS (on Spark), or to local parquet. [`build_features_spark.py`](features/build_features_spark.py) is the template.

**3 · Validate** — screen it:

> `/validate-signal screen warehouse.home_city_features vs warehouse.home_city — id person_id, label home_city`

Claude runs the harness and returns **`validation/report.md`** plus `validation/figures/`. The analysis is written wherever `--out` points; on Spark that is the driver, not back to S3/HDFS.

The human is at **two gates only**: confirm the wiring (step 1), and set the keep threshold and adjudicate the flagged features (step 3).

> **In-notebook (non-Connect) `spark`?** A script Claude launches is a separate
> process and can't attach to your kernel's session — expose a Spark Connect
> endpoint, or run the profiler's emitted cell in your notebook and `saveAsTable`
> the feature/label tables so the harness can read them.

<details>
<summary><b>Reproduce the bundled demo without installing the skills (raw scripts)</b></summary>

The skills run these for you. `generate_data.py` is demo scaffolding only — in
real use your tables already exist, so you start at *ideate*.

```bash
pip install pandas numpy scikit-learn scipy matplotlib pyarrow
python data/generate_data.py                          # 13 messy demo tables -> data/tables/
python scripts/inspect_tables.py data/tables          # profile  (what /ideate-features runs)
python features/build_features.py                     # materialize -> features/feature_table.parquet
python -X utf8 scripts/signal_panel.py \              # screen   (what /validate-signal runs)
  --features-file features/feature_table.parquet --labels-file data/tables/home_city.parquet \
  --id-col person_id --label-col home_city --time-col ref_ts --perm 25 --out validation/report.md
```
</details>

## Results

The screen runs twice — on the full table, and after the closed loop drops the
three flagged leaks:

| run | features | baseline macro-F1 | macro-AUC | keep / investigate / drop |
|---|---|---|---|---|
| **Full table** (leaks present) | 41 | **0.091** | 0.526 | 10 / 7 / 24 |
| **Screened** (leaks removed)   | 38 | **0.845** | 0.830 | 9 / 6 / 23 |

The headline is the **0.091 → 0.845** jump. Leakage is usually framed as *inflating*
a score; here the high-cardinality leak columns (`ip_city`, `declared_city`)
*wrecked* a naive 30-class baseline — so "train on everything and read the
importances" gives a misleadingly hopeless 0.091. The aggregate model score lies
in both directions; the **per-feature screen** tells the truth and says what to cut.

**What the harness did — the human only confirmed the wiring and adjudicated the flags:**

- **Found the signal feature-by-feature** even though the all-features model failed:
  MI ranks `nb_modal_city` (2.61), `merch_modal_city` (2.49), `ip_city` (2.21),
  `tower_modal_region` (2.00), `txt_flavor_top_city` (1.72) at the top.
- **Flagged the leaks.** A **normalized-MI** test (MI ÷ label entropy) — added to
  catch *categorical* leaks an AUC test can't see — flagged `nb_modal_city` (0.86)
  and `merch_modal_city` (0.82); `weather` tripped the numeric **AUC** flag (0.997);
  `ip_city`/`declared_city` topped the MI ranking for the human to cut.
- **Rejected all 24 red-herrings** (device, app telemetry, age, message-style
  stats, document length, raw degree) against the permutation null.

See [`validation/report.md`](validation/report.md) and
[`validation/screened/report.md`](validation/screened/report.md).

## Repo layout

| path | what |
|---|---|
| [`data/generate_data.py`](data/generate_data.py) | 13-table generator: text, docs, graph, geo, messy, 3 leak traps |
| [`features/backlog_homecity.md`](features/backlog_homecity.md) | the LLM's ranked feature-hypothesis backlog |
| [`features/build_features.py`](features/build_features.py) | materialization (local/pandas): text + OOF train-only neighbor (partial labels) |
| [`features/build_features_spark.py`](features/build_features_spark.py) | **real-world** Spark/Connect materialization (reads catalog/HDFS/S3) |
| [`scripts/inspect_tables.py`](scripts/inspect_tables.py) | local schema profiler (parquet/CSV) |
| [`scripts/profile_spark_tables.py`](scripts/profile_spark_tables.py) | Spark/Connect profiler — catalog tables **or** HDFS/S3 paths |
| [`scripts/signal_panel.py`](scripts/signal_panel.py) | screening harness (local files or Spark tables/paths; native categorical; AUC + normalized-MI leak flags) |
| [`validation/report.md`](validation/report.md) | per-feature signal report + figures |
| [`article/`](article) | the method article (markdown + landscape PDF) |

## Notes (mostly relevant only if you run the harness directly)

`/validate-signal` handles these for you; they matter if you bypass the skill:

- Pass `--time-col ref_ts` for the stability panel, and **omit `--group-col`**
  (feeding the `home_city` label to `GroupKFold` would hold out whole classes).
- On Windows run the harness as `python -X utf8 …` (it writes an em-dash that
  crashes cp1252).
- Generated parquet (`data/tables/`, `features/feature_table.parquet`) is
  gitignored — regenerable, deterministic (seed 11).
