# Paper statistics (from data/results)

Metric tag: absmatch; pooling: dropF_punctL

## 1. Privilege (native - random) per target, one observation per model

| Target | n | random | native | optimal | gap mean [95% CI] | d_z | t | p_t | W | p_W (exact) | models > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Model-to-model | 12 | 0.2466 | 0.2677 | 0.4264 | +0.0211 [+0.0156, +0.0265] | 2.45 | 8.50 | 3.7e-06 | 0 | 4.9e-04 | 12/12 |
| Pereira Exp 2 | 12 | 0.0366 | 0.0440 | 0.0568 | +0.0074 [+0.0048, +0.0100] | 1.80 | 6.22 | 6.5e-05 | 0 | 4.9e-04 | 12/12 |
| Pereira Exp 3 | 12 | 0.0361 | 0.0467 | 0.0670 | +0.0105 [+0.0060, +0.0151] | 1.48 | 5.12 | 3.4e-04 | 0 | 4.9e-04 | 12/12 |
| Wug | 12 | 0.1891 | 0.2270 | 0.6781 | +0.0379 [+0.0224, +0.0533] | 1.56 | 5.40 | 2.2e-04 | 1 | 9.8e-04 | 11/12 |
| BLiMP | 12 | 0.3483 | 0.3667 | 0.7779 | +0.0184 [+0.0096, +0.0272] | 1.33 | 4.61 | 7.5e-04 | 0 | 4.9e-04 | 12/12 |
| Hypernymy | 12 | 0.1563 | 0.2019 | 0.5903 | +0.0456 [+0.0202, +0.0709] | 1.14 | 3.96 | 0.002 | 4 | 0.003 | 10/12 |
| Perceptual | 12 | 0.2159 | 0.2702 | 0.5978 | +0.0543 [+0.0274, +0.0812] | 1.28 | 4.44 | 1.0e-03 | 1 | 9.8e-04 | 11/12 |
| Affective | 12 | 0.1942 | 0.2258 | 0.6033 | +0.0317 [+0.0117, +0.0516] | 1.01 | 3.50 | 0.005 | 9 | 0.016 | 9/12 |

Holm-adjusted Wilcoxon p across the 8 targets: Model-to-model: 0.004; Pereira Exp 2: 0.004; Pereira Exp 3: 0.004; Wug: 0.004; BLiMP: 0.004; Hypernymy: 0.007; Perceptual: 0.004; Affective: 0.016

## 2. Ceiling (optimal - native) per target

| Target | gap mean [95% CI] | native/optimal | t | p_t | W | p_W | models > 0 |
|---|---|---|---|---|---|---|---|
| Model-to-model | +0.1587 [+0.1440, +0.1733] | 0.63 | 23.87 | 8.0e-11 | 0 | 4.9e-04 | 12/12 |
| Pereira Exp 2 | +0.0128 [+0.0113, +0.0143] | 0.77 | 19.17 | 8.4e-10 | 0 | 4.9e-04 | 12/12 |
| Pereira Exp 3 | +0.0203 [+0.0188, +0.0219] | 0.70 | 28.16 | 1.3e-11 | 0 | 4.9e-04 | 12/12 |
| Wug | +0.4512 [+0.4228, +0.4795] | 0.33 | 35.00 | 1.2e-12 | 0 | 4.9e-04 | 12/12 |
| BLiMP | +0.4112 [+0.3965, +0.4259] | 0.47 | 61.61 | 2.5e-15 | 0 | 4.9e-04 | 12/12 |
| Hypernymy | +0.3884 [+0.3535, +0.4233] | 0.34 | 24.47 | 6.1e-11 | 0 | 4.9e-04 | 12/12 |
| Perceptual | +0.3276 [+0.2993, +0.3559] | 0.45 | 25.47 | 3.9e-11 | 0 | 4.9e-04 | 12/12 |
| Affective | +0.3775 [+0.3495, +0.4056] | 0.37 | 29.61 | 7.7e-12 | 0 | 4.9e-04 | 12/12 |

## 3. Monotonic decline with alpha (per-model Kendall tau over 11 alphas, exact one-sided p, Stouffer-combined; Spearman rho descriptive; OLS slope tested across models)

| Target | models rho<0 | models rho<=-0.9 | mean rho | Stouffer Z | p (one-sided) | slope mean | t | p_t | p_W |
|---|---|---|---|---|---|---|---|---|---|
| Model-to-model | 12/12 | 12/12 | -0.97 | 16.51 | 1.5e-61 | -0.0238 | -8.21 | 5.1e-06 | 4.9e-04 |
| Pereira Exp 2 | 12/12 | 7/12 | -0.89 | 13.42 | 2.4e-41 | -0.0082 | -6.12 | 7.6e-05 | 4.9e-04 |
| Pereira Exp 3 | 12/12 | 8/12 | -0.92 | 14.43 | 1.7e-47 | -0.0120 | -5.25 | 2.7e-04 | 4.9e-04 |
| Wug | 11/12 | 11/12 | -0.84 | 14.37 | 4.1e-47 | -0.0439 | -5.20 | 2.9e-04 | 9.8e-04 |
| BLiMP | 12/12 | 9/12 | -0.88 | 13.99 | 8.8e-45 | -0.0212 | -4.47 | 9.4e-04 | 4.9e-04 |
| Hypernymy | 10/12 | 7/12 | -0.68 | 11.44 | 1.3e-30 | -0.0528 | -3.85 | 0.003 | 0.003 |
| Perceptual | 11/12 | 8/12 | -0.80 | 13.43 | 2.1e-41 | -0.0636 | -4.34 | 0.001 | 9.8e-04 |
| Affective | 10/12 | 7/12 | -0.60 | 9.96 | 1.1e-23 | -0.0358 | -3.50 | 0.005 | 0.009 |

## 4. Privilege vs model size (Spearman with log10 params)

| Target | n | rho | p |
|---|---|---|---|
| Model-to-model | 11 | +0.32 | 0.340 |
| Pereira Exp 2 | 12 | +0.20 | 0.527 |
| Pereira Exp 3 | 12 | -0.17 | 0.587 |
| Wug | 12 | -0.18 | 0.572 |
| BLiMP | 12 | +0.38 | 0.217 |
| Hypernymy | 12 | +0.01 | 0.983 |
| Perceptual | 12 | +0.33 | 0.297 |
| Affective | 12 | +0.50 | 0.095 |

## 5. Per-model values (native / random / optimal; gap = native - random)

| label                          |   Affective |    BLiMP |   Hypernymy |   Model-to-model |   Perceptual |   Pereira Exp 2 |   Pereira Exp 3 |      Wug |
|:-------------------------------|------------:|---------:|------------:|-----------------:|-------------:|----------------:|----------------:|---------:|
| GPT-2                          |     -0.0085 |   0.0057 |     -0.0105 |           0.0141 |       0.018  |          0.0054 |          0.0315 |   0.0261 |
| GPT-2$_{\mathrm{L}}$           |     -0.0059 |   0.0035 |      0.0555 |           0.0126 |       0.0241 |          0.0016 |          0.0069 |   0.0272 |
| GPT-2$_{\mathrm{M}}$           |     -0.0026 |   0.0151 |      0.0251 |           0.0111 |       0.0075 |          0.0029 |          0.0126 |   0.04   |
| GPT-2$_{\mathrm{XL}}$          |      0.0228 |   0.0079 |     -0.0035 |           0.0114 |      -0.0012 |          0.0048 |          0.0052 |  -0.0046 |
| Gemma 3$_{\mathrm{1B}}$        |      0.0294 |   0.0448 |      0.0536 |           0.0245 |       0.0678 |          0.0164 |          0.0124 |   0.0784 |
| LLaMA 3.2$_{\mathrm{1B}}$      |      0.0723 |   0.0189 |      0.0159 |           0.0254 |       0.07   |          0.0088 |          0.0091 |   0.0342 |
| LLaMA 3.2$_{\mathrm{3B}}$      |      0.0773 |   0.0271 |      0.0483 |           0.0321 |       0.1056 |          0.0091 |          0.0088 |   0.0166 |
| Mamba$_{\mathrm{130M}}$        |      0.0478 |   0.0203 |      0.0882 |           0.0192 |       0.0451 |          0.0099 |          0.0085 |   0.06   |
| Mamba$_{\mathrm{370M}}$        |      0.0621 |   0.0163 |      0.0891 |           0.0254 |       0.1025 |          0.0082 |          0.0072 |   0.0623 |
| Mamba$_{\mathrm{790M}}$        |      0.0281 |   0.0177 |      0.1175 |           0.0268 |       0.1102 |          0.0089 |          0.01   |   0.0491 |
| Pythia$_{\mathrm{160M-Seed2}}$ |    nan      | nan      |    nan      |           0.0136 |     nan      |        nan      |        nan      | nan      |
| Pythia$_{\mathrm{160M}}$       |      0.0003 |   0.0019 |      0.0084 |         nan      |       0.0083 |          0.0029 |          0.0034 |   0.0095 |
| Qwen 3$_{\mathrm{0.6B}}$       |      0.0568 |   0.0417 |      0.0594 |           0.0364 |       0.0937 |          0.0095 |          0.0109 |   0.0553 |

(Full per-model table with native/random/optimal: data/results/stats_voxavg_all/per_model_scores.csv)

## 6. Model-to-model detail

Seed pair: native 0.302, random 0.289, gap +0.0136, optimal 0.492
Other 11 pairs: native 0.265, random 0.243, gap +0.0217 (range +0.0111 to +0.0364); seed-pair gap z relative to the other pairs: -0.94

## 7. Pereira: ceiling audit over (layer, participant) cells

Exp 2 ceiling audit over (layer, participant) cells, mean across models: optimal < native in 28% (range 22-35%), optimal <= 0 in 3%; native > random in 69% of cells

Exp 3 ceiling audit over (layer, participant) cells, mean across models: optimal < native in 8% (range 1-17%), optimal <= 0 in 2%; native > random in 77% of cells

## 8. BLiMP per sub-task (Wilcoxon across models)

| Sub-task | gap mean | models > 0 | p_W |
|---|---|---|---|
| adjunct_island | +0.0289 | 11/12 | 0.001 |
| anaphor_gender_agreement | +0.0084 | 7/12 | 0.233 |
| determiner_noun_agreement_1 | +0.0371 | 11/12 | 9.8e-04 |
| ellipsis_n_bar_1 | +0.0096 | 7/12 | 0.151 |
| existential_there_object_raising | +0.0108 | 10/12 | 0.005 |
| existential_there_quantifiers_1 | +0.0088 | 8/12 | 0.129 |
| irregular_past_participle_adjectives | +0.0449 | 12/12 | 4.9e-04 |
| irregular_plural_subject_verb_agreement_1 | +0.0246 | 9/12 | 0.016 |
| matrix_question_npi_licensor_present | +0.0182 | 9/12 | 0.016 |
| principle_A_c_command | +0.0032 | 7/12 | 0.151 |
| transitive | +0.0129 | 9/12 | 0.021 |
| wh_questions_subject_gap | +0.0132 | 10/12 | 0.016 |

8/12 sub-tasks significant at p < 0.05 (uncorrected)

## 9. Training dynamics (Pythia seed pairs; gap = native - random, mean over layers)

| Size | steps | gap step 0 | layers > 0 at step 0 | gap last step | rho(log step, gap) | p | step of max gap |
|---|---|---|---|---|---|---|---|
| 14m | 29 | -0.0023 | 1/4 | +0.0120 | +0.84 | 1.2e-08 | 143000 |
| 160m | 29 | +0.0064 | 3/4 | +0.0129 | +0.72 | 8.8e-06 | 112000 |
| 31m | 29 | +0.0014 | 3/4 | +0.0174 | +0.91 | 9.4e-12 | 143000 |
| 410m | 29 | +0.0055 | 4/4 | +0.0093 | +0.42 | 0.022 | 87000 |
| 70m | 29 | +0.0039 | 4/4 | +0.0167 | +0.95 | 7.4e-15 | 87000 |

Step 0 across sizes: mean +0.0030 [-0.0014, +0.0073], t(4) = 1.90, p_t = 0.130, Wilcoxon p = 0.188, 4/5 > 0; all (size, layer) cells at step 0: 15/20 > 0 (binomial p = 0.041)
Last step across sizes: mean +0.0136 [+0.0095, +0.0178], t(4) = 9.10, p_t = 8.1e-04
Last - first: mean +0.0107, t(4) = 4.54, p_t = 0.010, Wilcoxon p = 0.062, 5/5

## 10. Readout sparsity: median native kurtosis vs null of the median (10 rotation seeds)

| Model | d | discrete native | null mean +- SD | z | sig | continuous native | null | z | sig | pooled native | z | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 | 768 | 0.170 | -0.020 +- 0.014 | 13.2 | * | 0.396 | -0.005 +- 0.022 | 18.6 | * | 0.283 | 27.5 | * |
| GPT-2$_{\mathrm{M}}$ | 1024 | 0.139 | -0.012 +- 0.016 | 9.4 | * | 0.124 | -0.000 +- 0.016 | 7.8 | * | 0.131 | 12.9 | * |
| GPT-2$_{\mathrm{L}}$ | 1280 | 0.081 | -0.014 +- 0.015 | 6.3 | * | 0.485 | -0.005 +- 0.017 | 28.1 | * | 0.283 | 26.8 | * |
| GPT-2$_{\mathrm{XL}}$ | 1600 | 0.028 | -0.007 +- 0.007 | 4.7 | * | 0.088 | -0.010 +- 0.018 | 5.3 | * | 0.058 | 6.7 | * |
| LLaMA 3.2$_{\mathrm{1B}}$ | 2048 | 0.061 | -0.017 +- 0.010 | 7.7 | * | 2.582 | -0.002 +- 0.009 | 300.7 | * | 1.322 | 275.6 | * |
| LLaMA 3.2$_{\mathrm{3B}}$ | 3072 | 0.023 | -0.004 +- 0.005 | 5.7 | * | 2.579 | -0.002 +- 0.008 | 331.6 | * | 1.301 | 317.7 | * |
| Gemma 3$_{\mathrm{1B}}$ | 1152 | 0.437 | -0.013 +- 0.006 | 73.8 | * | 2.579 | -0.006 +- 0.011 | 236.5 | * | 1.508 | 228.3 | * |
| Mamba$_{\mathrm{130M}}$ | 768 | 0.392 | -0.012 +- 0.009 | 45.7 | * | 0.741 | +0.004 +- 0.021 | 34.4 | * | 0.567 | 56.0 | * |
| Mamba$_{\mathrm{370M}}$ | 1024 | 0.228 | -0.016 +- 0.011 | 23.1 | * | 0.740 | -0.008 +- 0.023 | 32.6 | * | 0.484 | 39.3 | * |
| Mamba$_{\mathrm{790M}}$ | 1536 | 0.066 | -0.013 +- 0.005 | 14.6 | * | 0.968 | -0.004 +- 0.018 | 53.4 | * | 0.517 | 60.7 | * |
| Qwen 3$_{\mathrm{0.6B}}$ | 1024 | 0.605 | -0.017 +- 0.011 | 55.6 | * | 1.609 | -0.009 +- 0.016 | 99.4 | * | 1.107 | 108.9 | * |
| Pythia$_{\mathrm{160M}}$ | 768 | 0.250 | -0.029 +- 0.020 | 14.2 | * | 0.459 | -0.017 +- 0.020 | 23.4 | * | 0.355 | 26.7 | * |

Discrete: 12/12 above null; continuous: 12/12; pooled: 12/12
alpha = 1 endpoint: per-cell null mean over all cells = -0.0045 (SD across cells 0.042); expected under a uniformly random rotation -6/(d+1) per cell, i.e. -0.0050 averaged over the same cells (a median is more negative).

