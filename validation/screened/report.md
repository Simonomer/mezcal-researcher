# Signal validation — home_city

- sample rows: 5,240  ·  classes: 30  ·  features: 38
- baseline (HGB): macro-F1 = 0.845, macro-AUC = 0.830
- recommendation: keep 9 · investigate 6 · drop 23

![ranking](figures/signal_ranking.png)

![importance](figures/importance.png)

![confusion](figures/confusion.png)

![redundancy](figures/redundancy.png)


## Per-feature

| feature | recommend | reason | mi | mi_ratio | null_p95 | beats_null | best_auc | importance | coverage | stability_cv |
|---|---|---|---|---|---|---|---|---|---|---|
| txt_flavor_top_frac | drop | no signal over null | 0.013 | 0.004 | 0.021 | False | 0.585 | 0.001 | 0.839 | 0.914 |
| txt_flavor_entropy | drop | no signal over null | 0.013 | 0.004 | 0.018 | False | 0.590 | 0.001 | 0.839 | 1.417 |
| act_hour_std | drop | no signal over null | 0.011 | 0.004 | 0.022 | False | 0.616 | 0.008 | 1.000 | 1.921 |
| txt_vocab_richness | drop | no signal over null | 0.010 | 0.003 | 0.015 | False | 0.604 | 0.008 | 0.999 | 2.000 |
| device | drop | no signal over null | 0.009 | 0.003 | 0.010 | False | — | 0.001 | 0.952 | 0.134 |
| txt_n_city_mentions | drop | no signal over null | 0.009 | 0.003 | 0.013 | False | — | -0.000 | 0.387 | 0.079 |
| txt_emoji_frac | drop | no signal over null | 0.006 | 0.002 | 0.013 | False | 0.597 | 0.004 | 0.999 | 1.909 |
| app_event_count | drop | no signal over null | 0.005 | 0.002 | 0.014 | False | 0.647 | 0.003 | 1.000 | 2.000 |
| doc_n_city_mentions | drop | no signal over null | 0.003 | 0.001 | 0.004 | False | — | 0.000 | 0.048 | 0.193 |
| battery | drop | no signal over null | 0.003 | 0.001 | 0.012 | False | 0.609 | 0.003 | 1.000 | 1.265 |
| act_hour_mean | drop | no signal over null | 0.003 | 0.001 | 0.009 | False | 0.584 | 0.004 | 1.000 | 1.026 |
| doc_present | drop | no signal over null | 0.003 | 0.001 | 0.004 | False | — | -0.000 | 1.000 | 0.229 |
| doc_len | drop | no signal over null | 0.000 | 0.000 | 0.013 | False | 0.841 | -0.000 | 0.244 | 0.559 |
| txt_total_tokens | drop | no signal over null | 0.000 | 0.000 | 0.014 | False | 0.597 | 0.005 | 0.999 | 0.995 |
| txt_url_frac | drop | no signal over null | 0.000 | 0.000 | 0.016 | False | 0.647 | 0.002 | 0.999 | 1.562 |
| txt_n_messages | drop | no signal over null | 0.000 | 0.000 | 0.023 | False | 0.600 | 0.000 | 1.000 | 0.898 |
| deg_out | drop | no signal over null | 0.000 | 0.000 | 0.012 | False | 0.600 | 0.000 | 1.000 | 2.000 |
| tower_top_frac | drop | no signal over null | 0.000 | 0.000 | 0.010 | False | 0.606 | 0.002 | 1.000 | 1.240 |
| tower_entropy | drop | no signal over null | 0.000 | 0.000 | 0.013 | False | 0.610 | -0.001 | 1.000 | 1.153 |
| merch_online_frac | drop | no signal over null | 0.000 | 0.000 | 0.020 | False | 0.620 | -0.001 | 1.000 | 1.313 |
| tenure_days | drop | no signal over null | 0.000 | 0.000 | 0.019 | False | 0.656 | 0.002 | 0.971 | 0.681 |
| age | drop | no signal over null | 0.000 | 0.000 | 0.018 | False | 0.625 | 0.007 | 0.949 | 1.897 |
| screen_in | drop | no signal over null | 0.000 | 0.000 | 0.013 | False | 0.690 | 0.002 | 1.000 | 1.387 |
| nb_modal_city | investigate | explains most of the label — check leakage | 2.613 | 0.862 | 0.085 | True | — | 0.229 | 0.999 | 0.007 |
| merch_modal_city | investigate | explains most of the label — check leakage | 2.486 | 0.820 | 0.132 | True | — | 0.153 | 1.000 | 0.011 |
| nb_entropy | investigate | redundant with nb_top_frac | 0.197 | 0.065 | 0.017 | True | 0.855 | 0.003 | 0.999 | 0.191 |
| nb_top_frac | investigate | redundant with nb_entropy | 0.155 | 0.051 | 0.016 | True | 0.918 | 0.005 | 0.999 | 0.226 |
| reciprocity | investigate | unstable over time | 0.032 | 0.010 | 0.016 | True | 0.782 | -0.000 | 1.000 | 1.533 |
| merch_top_frac | investigate | unstable over time | 0.021 | 0.007 | 0.012 | True | 0.563 | 0.003 | 1.000 | 1.304 |
| tower_modal_region | keep | beats null, contributes in model | 1.996 | 0.659 | 0.029 | True | — | 0.116 | 1.000 | 0.004 |
| txt_flavor_top_city | keep | beats null, contributes in model | 1.724 | 0.569 | 0.139 | True | — | 0.002 | 0.839 | 0.022 |
| txt_modal_mention_city | keep | beats null, contributes in model | 0.892 | 0.294 | 0.119 | True | — | -0.008 | 0.387 | 0.030 |
| txt_lang_dominant | keep | beats null, contributes in model | 0.467 | 0.154 | 0.018 | True | — | 0.000 | 1.000 | 0.033 |
| doc_flavor_top_city | keep | beats null, contributes in model | 0.439 | 0.145 | 0.114 | True | — | 0.001 | 0.218 | 0.040 |
| declared_language | keep | beats null, contributes in model | 0.222 | 0.073 | 0.017 | True | — | -0.000 | 1.000 | 0.063 |
| nb_labeled_frac | keep | beats null, contributes in model | 0.092 | 0.030 | 0.010 | True | 0.934 | 0.004 | 0.999 | 0.073 |
| deg_in | keep | beats null, contributes in model | 0.059 | 0.020 | 0.020 | True | 0.872 | 0.002 | 0.999 | 0.451 |
| deg_total | keep | beats null, contributes in model | 0.038 | 0.013 | 0.013 | True | 0.790 | 0.001 | 1.000 | 0.926 |

_Signal = beats shuffled-label null, has effect size, contributes incrementally in the model, and is stable. Screening only — the final word is full-model out-of-sample performance._
