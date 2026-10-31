# Cross-Model Exploratory Comparison

## Five-model comparison in the current experiment manuscript

The current manuscript reports two runs for each of three shared papers across five models. The table below summarizes model-level means and average absolute differences between the two runs.

| Model | Mean score across three papers | Mean absolute difference between runs |
|---|---:|---:|
| Qwen3.8-Flash | 89.99 | 3.83 |
| Qwen3.5-35B-A3B | 86.55 | 2.35 |
| Qwen3.7-Flash | 86.14 | 1.11 |
| DeepSeek-Flash | 78.37 | 0.61 |
| Kimi-K2.5 | 77.91 | 3.83 |

The three shared papers had the same rank order across the models with complete records. Evidence was re-extracted and external literature was retrieved on each run, so run-to-run differences combine evidence and model-judgment variation. There were no independent expert scores; the results are descriptive and do not establish model accuracy or general superiority.

The manuscript also records partial outcomes for Qwen-Flash, Qwen3.8-Max, and DeepSeek-V4-Pro. The DeepSeek-V4-Pro run used a different rule profile and is not directly comparable with the five-model table.

## Earlier archived two-model comparison

## Design

The archived comparison covers three papers, two models, and two runs per paper/model: 12 valid runs in total. Both models used the V16 profile. The source summary combines runs collected at different times and endpoints; evidence retrieval was not fixed between models. Treat the results as descriptive model-sensitivity evidence.

## Aggregate scores

| Anonymous sample | Qwen3.7 Flash mean | DeepSeek Flash mean | Difference |
|---|---:|---:|---:|
| Sample A | 79.33 | 67.38 | +11.95 |
| Sample B | 85.42 | 76.31 | +9.12 |
| Sample C | 93.67 | 91.42 | +2.25 |
| Mean across 6 runs per model | 86.14 | 78.37 | +7.77 |

Within-model absolute score differences across the two runs ranged from 0.56 to 1.97 points for Qwen3.7 Flash and from 0.00 to 1.33 points for DeepSeek Flash across the three samples.

## Interpretation

The archived comparison shows model choice can shift scores, with the size of the difference varying by paper and dimension. Because the runs used evidence retrieved at different times, the observed differences do not isolate model effects.

The source summary also records a GLM-5.3 attempt that failed before producing a score because the provider returned an insufficient-balance response. It is logged as a service-access event and excluded from the 12 valid runs.
