# Results

## Pre-registered hypotheses

| id | claim | n | Δ [95% CI] | McNemar exact p (Bonf.) | result |
|---|---|---|---|---|---|
| H1 | few_shot_3 > zero_shot on GSM8K for Qwen2.5-0.5B | 300 | +0.087 [+0.023, +0.143] | 0.044 | supported |
| H2 | Llama-3.2-1B > Qwen2.5-0.5B on HellaSwag acc_norm (zero-shot) | 500 | +0.138 [+0.106, +0.172] | <0.001 | supported |
| H3 | cot > zero_shot on GSM8K for Llama-3.2-1B | 300 | +0.013 [-0.010, +0.037] | 1.000 | not supported — underpowered — no conclusion |

## HellaSwag · generation (metric: acc)

500 examples; bootstrap B = 1000, seed 42.

### Cells

| model | prompt | acc [95% CI] | n_ok | n_error | parse failures | hit token budget |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | zero_shot | 0.350 [0.308, 0.392] | 500 | 0 | 0.000 | 0.012 |
| qwen2.5-0.5b | few_shot_3 | 0.326 [0.288, 0.368] | 500 | 0 | 0.000 | 0.000 |
| qwen2.5-0.5b | cot | 0.100 [0.076, 0.128] | 500 | 0 | 0.702 | 0.672 |
| llama-3.2-1b | zero_shot | 0.258 [0.222, 0.298] | 500 | 0 | 0.002 | 0.002 |
| llama-3.2-1b | few_shot_3 | 0.236 [0.200, 0.274] | 500 | 0 | 0.000 | 0.000 |
| llama-3.2-1b | cot | 0.034 [0.020, 0.050] | 500 | 0 | 0.866 | 0.784 |

### Per model (averaged over prompt variants)

| model | acc [95% CI] | cells averaged | n |
|---|---|---|---|
| qwen2.5-0.5b | 0.259 [0.234, 0.285] | 3 | 500 |
| llama-3.2-1b | 0.176 [0.157, 0.196] | 3 | 500 |

### Per prompt variant (averaged over models)

| prompt_variant | acc [95% CI] | cells averaged | n |
|---|---|---|---|
| zero_shot | 0.304 [0.273, 0.339] | 2 | 500 |
| few_shot_3 | 0.281 [0.250, 0.311] | 2 | 500 |
| cot | 0.067 [0.052, 0.083] | 2 | 500 |

### Pairwise comparisons (9 confirmatory, Bonferroni)

| A | B | factor | acc_A − acc_B [95% CI] | A only / B only | Wilcoxon p (Bonf.) | Holm | McNemar exact p (Bonf.) | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b · zero_shot | qwen2.5-0.5b · few_shot_3 | prompt | +0.024 [-0.022, +0.072] | 93 / 81 | 1.000 | 0.726 | 1.000 | underpowered — no conclusion |  |
| qwen2.5-0.5b · zero_shot | qwen2.5-0.5b · cot | prompt | +0.250 [+0.206, +0.294] | 144 / 19 | <0.001 | <0.001 | <0.001 | A > B |  |
| qwen2.5-0.5b · zero_shot | llama-3.2-1b · zero_shot | model | +0.092 [+0.040, +0.140] | 104 / 58 | 0.003 | 0.001 | 0.003 | A > B |  |
| qwen2.5-0.5b · few_shot_3 | qwen2.5-0.5b · cot | prompt | +0.226 [+0.182, +0.272] | 134 / 21 | <0.001 | <0.001 | <0.001 | A > B |  |
| qwen2.5-0.5b · few_shot_3 | llama-3.2-1b · few_shot_3 | model | +0.090 [+0.040, +0.144] | 115 / 70 | 0.008 | 0.003 | 0.010 | A > B |  |
| qwen2.5-0.5b · cot | llama-3.2-1b · cot | model | +0.066 [+0.034, +0.098] | 48 / 15 | <0.001 | <0.001 | <0.001 | A > B |  |
| llama-3.2-1b · zero_shot | llama-3.2-1b · few_shot_3 | prompt | +0.022 [-0.026, +0.074] | 92 / 81 | 1.000 | 0.726 | 1.000 | underpowered — no conclusion |  |
| llama-3.2-1b · zero_shot | llama-3.2-1b · cot | prompt | +0.224 [+0.186, +0.266] | 122 / 10 | <0.001 | <0.001 | <0.001 | A > B |  |
| llama-3.2-1b · few_shot_3 | llama-3.2-1b · cot | prompt | +0.202 [+0.162, +0.244] | 114 / 13 | <0.001 | <0.001 | <0.001 | A > B |  |

Significance matrix (Bonferroni-adjusted Wilcoxon p; ★ = verdict under the decision rule; · = pair differs in both factors, exploratory, not tested):

|  | qwen2.5-0.5b · zero_shot | qwen2.5-0.5b · few_shot_3 | qwen2.5-0.5b · cot | llama-3.2-1b · zero_shot | llama-3.2-1b · few_shot_3 | llama-3.2-1b · cot |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b · zero_shot | — | 1.000 | <0.001★ | 0.003★ | · | · |
| qwen2.5-0.5b · few_shot_3 |  | — | <0.001★ | · | 0.008★ | · |
| qwen2.5-0.5b · cot |  |  | — | · | · | <0.001★ |
| llama-3.2-1b · zero_shot |  |  |  | — | 1.000 | <0.001★ |
| llama-3.2-1b · few_shot_3 |  |  |  |  | — | <0.001★ |
| llama-3.2-1b · cot |  |  |  |  |  | — |

**Power.** With n = 500 and a discordance rate of 0.32, a confirmatory comparison at α = 0.05/9 has 80% power to detect differences of **0.092** or more; smaller differences are "underpowered — no conclusion", never "no difference".

### Variance decomposition

| source | share of cell-mean variance [95% CI] |
|---|---|
| model | 0.130 [0.064, 0.207] |
| prompt | 0.867 [0.779, 0.931] |
| interaction | 0.003 [0.000, 0.029] |

Per-example linear-probability ANOVA, `correct ~ C(example) + C(model) * C(prompt)`:

| term | df | share of total SS | F | p |
|---|---|---|---|---|
| example (item difficulty) | 499 | 0.248 | 1.839 | <0.001 |
| model | 1 | 0.010 | 37.144 | <0.001 |
| prompt | 2 | 0.067 | 123.799 | <0.001 |
| model × prompt | 2 | 0.000 | 0.379 | 0.684 |
| residual | 2495 | 0.675 |  |  |

## GSM8K · generation (metric: exact_match)

300 examples; bootstrap B = 1000, seed 42.

### Cells

| model | prompt | acc [95% CI] | n_ok | n_error | parse failures | hit token budget |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | zero_shot | 0.300 [0.250, 0.353] | 300 | 0 | 0.030 | 0.237 |
| qwen2.5-0.5b | few_shot_3 | 0.387 [0.333, 0.443] | 300 | 0 | 0.003 | 0.053 |
| qwen2.5-0.5b | cot | 0.373 [0.317, 0.427] | 300 | 0 | 0.000 | 0.227 |
| llama-3.2-1b | zero_shot | 0.017 [0.003, 0.033] | 300 | 0 | 0.073 | 0.223 |
| llama-3.2-1b | few_shot_3 | 0.040 [0.020, 0.063] | 300 | 0 | 0.003 | 0.167 |
| llama-3.2-1b | cot | 0.030 [0.013, 0.050] | 300 | 0 | 0.007 | 0.563 |

### Per model (averaged over prompt variants)

| model | acc [95% CI] | cells averaged | n |
|---|---|---|---|
| qwen2.5-0.5b | 0.353 [0.312, 0.398] | 3 | 300 |
| llama-3.2-1b | 0.029 [0.018, 0.041] | 3 | 300 |

### Per prompt variant (averaged over models)

| prompt_variant | acc [95% CI] | cells averaged | n |
|---|---|---|---|
| zero_shot | 0.158 [0.132, 0.185] | 2 | 300 |
| few_shot_3 | 0.213 [0.183, 0.247] | 2 | 300 |
| cot | 0.202 [0.173, 0.230] | 2 | 300 |

### Pairwise comparisons (9 confirmatory, Bonferroni)

| A | B | factor | acc_A − acc_B [95% CI] | A only / B only | Wilcoxon p (Bonf.) | Holm | McNemar exact p (Bonf.) | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b · zero_shot | qwen2.5-0.5b · few_shot_3 | prompt | -0.087 [-0.143, -0.023] | 27 / 53 | 0.033 | 0.022 | 0.044 | B > A |  |
| qwen2.5-0.5b · zero_shot | qwen2.5-0.5b · cot | prompt | -0.073 [-0.127, -0.020] | 27 / 49 | 0.105 | 0.058 | 0.139 | underpowered — no conclusion | CI and test disagree |
| qwen2.5-0.5b · zero_shot | llama-3.2-1b · zero_shot | model | +0.283 [+0.230, +0.337] | 88 / 3 | <0.001 | <0.001 | <0.001 | A > B |  |
| qwen2.5-0.5b · few_shot_3 | qwen2.5-0.5b · cot | prompt | +0.013 [-0.040, +0.067] | 36 / 32 | 1.000 | 0.983 | 1.000 | underpowered — no conclusion |  |
| qwen2.5-0.5b · few_shot_3 | llama-3.2-1b · few_shot_3 | model | +0.347 [+0.287, +0.407] | 109 / 5 | <0.001 | <0.001 | <0.001 | A > B |  |
| qwen2.5-0.5b · cot | llama-3.2-1b · cot | model | +0.343 [+0.287, +0.403] | 108 / 5 | <0.001 | <0.001 | <0.001 | A > B |  |
| llama-3.2-1b · zero_shot | llama-3.2-1b · few_shot_3 | prompt | -0.023 [-0.047, +0.000] | 4 / 11 | 0.636 | 0.283 | 1.000 | underpowered — no conclusion |  |
| llama-3.2-1b · zero_shot | llama-3.2-1b · cot | prompt | -0.013 [-0.037, +0.010] | 4 / 8 | 1.000 | 0.745 | 1.000 | underpowered — no conclusion |  |
| llama-3.2-1b · few_shot_3 | llama-3.2-1b · cot | prompt | +0.010 [-0.017, +0.037] | 11 / 8 | 1.000 | 0.983 | 1.000 | underpowered — no conclusion |  |

Significance matrix (Bonferroni-adjusted Wilcoxon p; ★ = verdict under the decision rule; · = pair differs in both factors, exploratory, not tested):

|  | qwen2.5-0.5b · zero_shot | qwen2.5-0.5b · few_shot_3 | qwen2.5-0.5b · cot | llama-3.2-1b · zero_shot | llama-3.2-1b · few_shot_3 | llama-3.2-1b · cot |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b · zero_shot | — | 0.033★ | 0.105 | <0.001★ | · | · |
| qwen2.5-0.5b · few_shot_3 |  | — | 1.000 | · | <0.001★ | · |
| qwen2.5-0.5b · cot |  |  | — | · | · | <0.001★ |
| llama-3.2-1b · zero_shot |  |  |  | — | 0.636 | 1.000 |
| llama-3.2-1b · few_shot_3 |  |  |  |  | — | 1.000 |
| llama-3.2-1b · cot |  |  |  |  |  | — |

**Power.** With n = 300 and a discordance rate of 0.25, a confirmatory comparison at α = 0.05/9 has 80% power to detect differences of **0.105** or more; smaller differences are "underpowered — no conclusion", never "no difference".

### Variance decomposition

| source | share of cell-mean variance [95% CI] |
|---|---|
| model | 0.972 [0.928, 0.993] |
| prompt | 0.021 [0.005, 0.049] |
| interaction | 0.008 [0.001, 0.027] |

Per-example linear-probability ANOVA, `correct ~ C(example) + C(model) * C(prompt)`:

| term | df | share of total SS | F | p |
|---|---|---|---|---|
| example (item difficulty) | 299 | 0.269 | 2.424 | <0.001 |
| model | 1 | 0.170 | 458.135 | <0.001 |
| prompt | 2 | 0.004 | 4.873 | 0.008 |
| model × prompt | 2 | 0.001 | 1.843 | 0.159 |
| residual | 1495 | 0.556 |  |  |

## HellaSwag · log-likelihood (metric: acc_norm)

500 examples; bootstrap B = 1000, seed 42.

### Cells

| model | prompt | acc [95% CI] | n_ok | n_error |
|---|---|---|---|---|
| qwen2.5-0.5b | zero_shot | 0.494 [0.450, 0.536] | 500 | 0 |
| qwen2.5-0.5b | few_shot_3 | 0.490 [0.448, 0.532] | 500 | 0 |
| llama-3.2-1b | zero_shot | 0.632 [0.590, 0.674] | 500 | 0 |
| llama-3.2-1b | few_shot_3 | 0.636 [0.594, 0.676] | 500 | 0 |

### Per model (averaged over prompt variants)

| model | acc [95% CI] | cells averaged | n |
|---|---|---|---|
| qwen2.5-0.5b | 0.492 [0.450, 0.534] | 2 | 500 |
| llama-3.2-1b | 0.634 [0.595, 0.674] | 2 | 500 |

### Per prompt variant (averaged over models)

| prompt_variant | acc [95% CI] | cells averaged | n |
|---|---|---|---|
| zero_shot | 0.563 [0.523, 0.604] | 2 | 500 |
| few_shot_3 | 0.563 [0.523, 0.602] | 2 | 500 |

### Pairwise comparisons (4 confirmatory, Bonferroni)

| A | B | factor | acc_A − acc_B [95% CI] | A only / B only | Wilcoxon p (Bonf.) | Holm | McNemar exact p (Bonf.) | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b · zero_shot | qwen2.5-0.5b · few_shot_3 | prompt | +0.004 [-0.014, +0.022] | 11 / 9 | 1.000 | 1.000 | 1.000 | underpowered — no conclusion |  |
| qwen2.5-0.5b · zero_shot | llama-3.2-1b · zero_shot | model | -0.138 [-0.172, -0.106] | 7 / 76 | <0.001 | <0.001 | <0.001 | B > A |  |
| qwen2.5-0.5b · few_shot_3 | llama-3.2-1b · few_shot_3 | model | -0.146 [-0.180, -0.114] | 5 / 78 | <0.001 | <0.001 | <0.001 | B > A |  |
| llama-3.2-1b · zero_shot | llama-3.2-1b · few_shot_3 | prompt | -0.004 [-0.030, +0.020] | 20 / 22 | 1.000 | 1.000 | 1.000 | underpowered — no conclusion |  |

Significance matrix (Bonferroni-adjusted Wilcoxon p; ★ = verdict under the decision rule; · = pair differs in both factors, exploratory, not tested):

|  | qwen2.5-0.5b · zero_shot | qwen2.5-0.5b · few_shot_3 | llama-3.2-1b · zero_shot | llama-3.2-1b · few_shot_3 |
|---|---|---|---|---|
| qwen2.5-0.5b · zero_shot | — | 1.000 | <0.001★ | · |
| qwen2.5-0.5b · few_shot_3 |  | — | · | <0.001★ |
| llama-3.2-1b · zero_shot |  |  | — | 1.000 |
| llama-3.2-1b · few_shot_3 |  |  |  | — |

**Power.** With n = 500 and a discordance rate of 0.12, a confirmatory comparison at α = 0.05/4 has 80% power to detect differences of **0.053** or more; smaller differences are "underpowered — no conclusion", never "no difference".

### Variance decomposition

| source | share of cell-mean variance [95% CI] |
|---|---|
| model | 0.999 [0.971, 1.000] |
| prompt | 0.000 [0.000, 0.016] |
| interaction | 0.001 [0.000, 0.021] |

Per-example linear-probability ANOVA, `correct ~ C(example) + C(model) * C(prompt)`:

| term | df | share of total SS | F | p |
|---|---|---|---|---|
| example (item difficulty) | 499 | 0.799 | 13.263 | <0.001 |
| model | 1 | 0.020 | 169.753 | <0.001 |
| prompt | 1 | 0.000 | 0.000 | 1.000 |
| model × prompt | 1 | 0.000 | 0.135 | 0.714 |
| residual | 1497 | 0.181 |  |  |

## Test–retest

3 same-seed reruns of each config. Agreement = share of (cell, example) pairs whose prediction is identical in every rerun; conclusion stability = share of confirmatory verdicts that are identical in every rerun.

| provider | scoring | identical predictions | (cell, example) pairs | agreement |
|---|---|---|---|---|
| hf | generation | 4800 | 4800 | 1.0000 |
| hf | loglikelihood | 2000 | 2000 | 1.0000 |

Overall agreement: **100.00%** (6800 of 6800). Conclusion stability: **100%** (22 of 22 confirmatory verdicts).
