# V16 Cross-Model Descriptive Comparison

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
