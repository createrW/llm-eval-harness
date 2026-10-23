# Results

## Pre-registered hypotheses

| id | claim | n | Δ [95% CI] | McNemar exact p (Bonf.) | result |
|---|---|---|---|---|---|
| H1 | few_shot_3 > zero_shot on GSM8K for Qwen2.5-0.5B |  |  |  | not run |
| H2 | Llama-3.2-1B > Qwen2.5-0.5B on HellaSwag acc_norm (zero-shot) | 500 | +0.138 [+0.106, +0.172] | <0.001 | supported |
| H3 | cot > zero_shot on GSM8K for Llama-3.2-1B |  |  |  | not run |

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

No config was rerun yet.
