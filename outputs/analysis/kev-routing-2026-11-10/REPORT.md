# Kev routing analysis

## Main results

The saved 4B and 9B probabilities were swept on 715 development records. Five development cost budgets were specified in advance of this sweep; for each router and budget, the highest-accuracy eligible development setting was selected, breaking ties by lower cost and then lower μ. These settings were applied unchanged to the 1,213-record in-distribution (ID) test and 278-record OOD test. The ID and OOD results had been inspected before this analysis, so neither is a fresh, untouched confirmation set for the wider research project.

The 9B setting selected at a $0.005/query development budget achieved 53.83% accuracy at $0.00239/query on ID, compared with 53.42% at $0.00083 for fixed Qwen 235B. Its paired accuracy difference was +0.41 percentage points (95% cluster-bootstrap interval −1.57 to +2.39); 80 queries were correct only for Kev and 75 only for Qwen. The measured cost increase was approximately 2.9-fold. The ID test result therefore does not establish an accuracy advantage at this operating point.

The three-epoch 9B argmax policy achieved 56.31% at $0.01115/query, compared with Qwen's 53.42% at $0.00083. The paired accuracy difference is +2.89 points (95% interval +0.66 to +5.11); Kev uniquely succeeded on 112 queries, and Qwen on 77. This is evidence of a positive ID accuracy difference for this observed test sample, with much higher downstream cost. The 4B three-epoch argmax difference versus Qwen was +2.72 points (95% interval +0.49 to +4.86).

Three-epoch 4B minus 9B argmax accuracy was −0.16 points (95% interval −2.06 to +1.65); 4B uniquely succeeded on 66 queries, 9B on 68. One-epoch 4B minus 9B was −0.16 points (−0.99 to +0.66); unique successes 13 and 15. There is no clear accuracy winner by size here.

Three-epoch minus one-epoch argmax differences were −0.49 points for both sizes, with intervals spanning zero: 4B −2.72 to +1.65 and 9B −2.64 to +1.65. These runs differ in *both* learning rate and epoch count (three epochs at 2e-4 versus one at 2e-5), so this is not a controlled epoch comparison. Three-epoch mean ID cost was lower by about $0.0031–$0.0035/query.

## Development-selected ID policies

| Router | Development budget | Selected μ | ID accuracy | ID cost/query |
|---|---:|---:|---:|---:|
| 4B | $0.001 | 73.7904 | 46.74% | $0.000892 |
| 4B | $0.002 | 15.2757 | 48.39% | $0.001517 |
| 4B | $0.005 | 0.7516 | 53.83% | $0.004753 |
| 4B | $0.010 | 0.0871 | 54.66% | $0.009012 |
| 4B | $0.020 | 0 | 56.14% | $0.011175 |
| 9B | $0.001 | 73.7904 | 46.08% | $0.000918 |
| 9B | $0.002 | 1.5849 | 53.34% | $0.001842 |
| 9B | $0.005 | 1.2706 | 53.83% | $0.002393 |
| 9B | $0.010 | 0.1117 | 55.32% | $0.009398 |
| 9B | $0.020 | 0.0023 | 56.39% | $0.011142 |
| Fixed Qwen 235B | — | — | 53.42% | $0.000829 |
| Fixed GPT-5 | — | — | 56.55% | $0.02933 |
| Fixed Gemini 2.5 Pro | — | — | 58.37% | $0.06039 |

Budgets are empirical averages on development, not guarantees for individual queries or other splits. The connecting lines in the plots join selected settings for visualization; they are not evidence that intermediate policies achieve the interpolated values.

## Where routing helps and costs arise

Three-epoch 9B argmax compared with fixed Qwen: HLE 32/212 versus 20/212, LiveCodeBench 81/105 versus 72/105, MMLU-Pro 239/299 versus 240/299, and SWE-Bench 11/52 for both. Small benchmark groups, including AIME (5) and LiveMathBench (13), are too small for strong conclusions. Training excluded SWE-Bench records.

The development-defined prompt-length quartile cutpoints are 24, 128 and 300 cl100k_base tokens. For shortest through longest ID quartiles, three-epoch 9B accuracy was 60.54%, 50.00%, 59.06%, 54.83%; fixed Qwen was 55.52%, 48.94%, 57.02%, 51.38%. Task type and length are correlated, and these descriptive comparisons do not establish a causal length effect.

Three-epoch 9B selected GPT-5 on 180 ID queries. These accounted for $9.47 of the router's $13.53 total recorded ID downstream cost; GPT-5 answered 69 of those 180 correctly. These queries were selected by the router and can differ in difficulty from the rest, so selected-model accuracy is not an isolated comparison of model quality. The CSV breaks results down by benchmark, development-defined length quartile, selected model and decision-time proxy cost tier.

## OOD transfer is exploratory

The OOD dataset has 278 rows but 167 distinct query hashes; repeated query text may occur in different scenarios. The paired bootstrap clusters by query hash. Three-epoch 9B argmax scored 38.13% at $0.04392/query; fixed Qwen scored 40.29% at $0.04198. The one-epoch 4B and 9B routers selected Qwen on all 278 rows.

The 4B tier gate scored 48.92% at $0.10027/query while choosing Kimi on 273 rows; fixed Kimi scored 48.92% at $0.10036. Fixed GLM-4.6 scored 53.96% at $0.09931, exceeding the 4B gate's observed accuracy at lower recorded cost. The OOD fixed-baseline CSV contains all 12 available models; GPT-5-chat has no OOD evaluations and is not a candidate there. These post hoc comparisons do not prove performance on unseen OOD data.

## Method and artifacts

- Decision rule: choose the eligible model maximizing calibrated probability minus μ times a fixed model cost proxy. The proxy costs are stored in `analysis_metadata.json`; outcome costs use each benchmark row's recorded per-model cost. Model eligibility is taken from the benchmark evaluations, so offline masking is not a prospective availability guarantee. Router inference cost is excluded.
- Development sweep: μ=0 plus a logarithmic grid from 0.0001 to 100 and μ=2.5, 503 distinct values. Three-epoch models only: the one-epoch files contain decisions but not full probabilities. All saved μ=0 and μ=2.5 choices were reproduced before analysis.
- Paired bootstrap: 5,000 resamples of query-hash clusters, percentile 95% intervals, seed 20261110. Test comparisons use the same sampled clusters for both methods; records are weighted by their count in each sampled cluster. Bootstrap intervals quantify sampling uncertainty conditional on the saved models and decisions, not variation from retraining, model selection, or multiple comparisons.
- [Development accuracy–cost plot](development_accuracy_cost.png), [ID plot](test_id_accuracy_cost.png), [OOD plot](test_ood_accuracy_cost.png).
- [All fixed baselines](fixed_baselines.csv), [policy results](policy_results.csv), [development sweep](development_sweep.csv), [selected settings](development_selected.csv), [paired intervals](paired_bootstrap.csv), [ID breakdowns](id_breakdowns.csv), and [metadata](analysis_metadata.json).
- Reproduce from the project root with `kevrouter_venv/bin/python scripts/analyze_kev_results.py`.
