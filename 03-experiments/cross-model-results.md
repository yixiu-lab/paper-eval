# Exploratory Comparison Across Five Models

## Study design

The experiment compared five models on three shared papers, with two recorded runs for each paper–model pair (30 scores in total). All runs used the same expert-calibrated scoring rules. Each run extracted evidence and performed external-literature retrieval independently, so the comparison captures variation in both model judgments and evidence inputs. It is descriptive and does not establish model accuracy or general superiority.

## Results by paper and model

Scores are on a 0–100 scale. The run difference is the absolute difference between the two recorded runs for that paper–model pair.

| Paper | Model | Run 1 | Run 2 | Mean | Absolute run difference |
|---|---|---:|---:|---:|---:|
| Mikalef and Gupta (2021) | Qwen3.8-Flash | 82.98 | 84.86 | 83.92 | 1.88 |
| Mikalef and Gupta (2021) | Qwen3.5-35B-A3B | 77.77 | 76.47 | 77.12 | 1.30 |
| Mikalef and Gupta (2021) | Qwen3.7-Flash | 79.73 | 78.93 | 79.33 | 0.80 |
| Mikalef and Gupta (2021) | DeepSeek-Flash | 68.05 | 66.72 | 67.38 | 1.33 |
| Mikalef and Gupta (2021) | Kimi-K2.5 | 74.45 | 72.95 | 73.70 | 1.50 |
| Wesche and Sonderegger (2021) | Qwen3.8-Flash | 89.59 | 93.13 | 91.36 | 3.53 |
| Wesche and Sonderegger (2021) | Qwen3.5-35B-A3B | 91.05 | 87.83 | 89.44 | 3.22 |
| Wesche and Sonderegger (2021) | Qwen3.7-Flash | 85.70 | 85.14 | 85.42 | 0.56 |
| Wesche and Sonderegger (2021) | DeepSeek-Flash | 76.06 | 76.56 | 76.31 | 0.50 |
| Wesche and Sonderegger (2021) | Kimi-K2.5 | 79.80 | 73.98 | 76.89 | 5.81 |
| Yan et al. (2024) | Qwen3.8-Flash | 97.75 | 91.66 | 94.70 | 6.09 |
| Yan et al. (2024) | Qwen3.5-35B-A3B | 94.38 | 91.84 | 93.11 | 2.53 |
| Yan et al. (2024) | Qwen3.7-Flash | 94.66 | 92.68 | 93.67 | 1.97 |
| Yan et al. (2024) | DeepSeek-Flash | 91.42 | 91.42 | 91.42 | 0.00 |
| Yan et al. (2024) | Kimi-K2.5 | 81.06 | 85.23 | 83.15 | 4.17 |

## Model-level summary

| Model | Mean across three papers | Mean absolute run difference |
|---|---:|---:|
| Qwen3.8-Flash | 89.99 | 3.83 |
| Qwen3.5-35B-A3B | 86.55 | 2.35 |
| Qwen3.7-Flash | 86.14 | 1.11 |
| DeepSeek-Flash | 78.37 | 0.61 |
| Kimi-K2.5 | 77.91 | 3.83 |

## Interpretation

Across these three papers, the model-level means span 12.08 points, from 77.91 for Kimi-K2.5 to 89.99 for Qwen3.8-Flash. The paper-level ordering is the same across all five models: Yan et al. (2024) receives the highest mean, followed by Wesche and Sonderegger (2021), then Mikalef and Gupta (2021). This shared ordering describes only these three papers.

Repeat-run differences vary by model and paper. DeepSeek-Flash has the smallest model-level mean absolute difference (0.61), while Qwen3.8-Flash and Kimi-K2.5 have the largest (3.83 each). The largest individual paired difference is 6.09 for Qwen3.8-Flash on Yan et al. (2024); the smallest is 0.00 for DeepSeek-Flash on the same paper. Because evidence was re-extracted and retrieval repeated on each run, these differences cannot be attributed to model choice alone.

The full experiment sequence is indexed in the [experiment results and index](README.md#manuscript-sequence). The single expert-calibrated scoring-rule set is documented in the [system rules](../02-system/rules/README.md).
