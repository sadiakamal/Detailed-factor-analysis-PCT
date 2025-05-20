#### Data Analysis

We want to prove/disprove certain hypotheses.

##### Hypothesis 1
RQ1: Does fine-tuning affect the political bias of a model as measured by the PCT test? We test this through a single factor anova with the fine-tuning status of a model as the independent variable and the Social Libertarian/Authoratrian score or Economic Left/Right score as being the dependent variable. The design dictates that there are $9$ groups in the independent variable as the models can be queried after fine-tuning on $8$ datasets or as it is.

```commandline
Predicted_var: Social Libertarian/Authoritarian, Model: Llama3
------------------------------
                          sum_sq     df         F    PR(>F)    eta_sq
C(fine_tune_dataset)    7.121646    8.0  1.422386  0.183352  0.015752
Residual              444.982164  711.0       NaN       NaN  0.984248
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Mistral
------------------------------
                          sum_sq     df        F    PR(>F)    eta_sq
C(fine_tune_dataset)   47.485889    8.0  5.17912  0.000003  0.055065
Residual              814.869844  711.0      NaN       NaN  0.944935
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Gemma
------------------------------
                          sum_sq     df          F        PR(>F)    eta_sq
C(fine_tune_dataset)  634.519779    8.0  57.361848  6.046453e-72  0.392253
Residual              983.108931  711.0        NaN           NaN  0.607747
==============================
Predicted_var: Economic Left/Right, Model: Llama3
------------------------------
                          sum_sq     df         F    PR(>F)    eta_sq
C(fine_tune_dataset)   17.736065    8.0  2.464097  0.012285  0.026977
Residual              639.703932  711.0       NaN       NaN  0.973023
==============================
Predicted_var: Economic Left/Right, Model: Mistral
------------------------------
                           sum_sq     df          F        PR(>F)   eta_sq
C(fine_tune_dataset)   717.318801    8.0  51.547732  9.108699e-66  0.36709
Residual              1236.750985  711.0        NaN           NaN  0.63291
==============================
Predicted_var: Economic Left/Right, Model: Gemma
------------------------------
                           sum_sq     df         F        PR(>F)    eta_sq
C(fine_tune_dataset)   476.446765    8.0  37.67872  5.548877e-50  0.297729
Residual              1123.822830  711.0       NaN           NaN  0.702271
==============================
```

Using Levene's test (Null hypothesis is that the in-group variance is identical) we can see that the unadjusted 1-way anova is not the best approach (as the homoscedasticity assumption is violated), so we use Welch's Anova that adjusts the degrees of freedom in the calculation of the F-statistic to account for unequal variances.

```commandline
Predicted_var: Social Libertarian/Authoritarian, Model: Llama3
------------------------------
Levene’s Test: Statistic = 2.8558, p-value = 0.003946
Welch
              Source  ddof1       ddof2         F     p-unc       np2
0  fine_tune_dataset      8  296.052349  1.159349  0.323711  0.015752
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Mistral
------------------------------
Levene’s Test: Statistic = 4.0420, p-value = 0.000103
Welch
              Source  ddof1       ddof2         F         p-unc       np2
0  fine_tune_dataset      8  294.866382  5.811853  6.750669e-07  0.055065
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Gemma
------------------------------
Levene’s Test: Statistic = 6.2885, p-value = 6.908e-08
Welch
              Source  ddof1       ddof2          F         p-unc       np2
0  fine_tune_dataset      8  295.339497  62.991599  2.008933e-59  0.392253
==============================
Predicted_var: Economic Left/Right, Model: Llama3
------------------------------
Levene’s Test: Statistic = 1.4257, p-value = 0.182
                          sum_sq     df         F    PR(>F)    eta_sq
C(fine_tune_dataset)   17.736065    8.0  2.464097  0.012285  0.026977
Residual              639.703932  711.0       NaN       NaN  0.973023
==============================
Predicted_var: Economic Left/Right, Model: Mistral
------------------------------
Levene’s Test: Statistic = 26.9897, p-value = 1.127e-36
Welch
              Source  ddof1       ddof2          F         p-unc      np2
0  fine_tune_dataset      8  295.236885  88.606452  6.646968e-74  0.36709
==============================
Predicted_var: Economic Left/Right, Model: Gemma
------------------------------
Levene’s Test: Statistic = 5.2920, p-value = 1.835e-06
Welch
              Source  ddof1       ddof2          F         p-unc       np2
0  fine_tune_dataset      8  295.793504  40.309997  3.579873e-43  0.297729
==============================
```

We don't see any effect of finetuning on the dependent variables in the Llama3 model in the Social Libertarian/Authoritarian case. However, both Gemma and Mistral models have high F-staistic with very low p-values for both dependent variables, indicating a clear effect of finetuning on the political bias of the models.

#### RQ2
Do other factors of significant effect?

##### tmp
```commandline
Predicted_var: Social Libertarian/Authoritarian, Model: Llama3
------------------------------
Levene’s Test: Statistic = 0.6819, p-value = 0.4092
              sum_sq     df        F    PR(>F)    eta_sq
C(tmp)      0.527583    1.0  0.83885  0.360033  0.001167
Residual  451.576226  718.0      NaN       NaN  0.998833
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Mistral
------------------------------
Levene’s Test: Statistic = 0.7536, p-value = 0.3856
              sum_sq     df         F    PR(>F)    eta_sq
C(tmp)      1.580157    1.0  1.318059  0.251323  0.001832
Residual  860.775576  718.0       NaN       NaN  0.998168
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Gemma
------------------------------
Levene’s Test: Statistic = 1.4935, p-value = 0.2221
               sum_sq     df         F    PR(>F)    eta_sq
C(tmp)       4.403911    1.0  1.960054  0.161939  0.002722
Residual  1613.224799  718.0       NaN       NaN  0.997278
==============================
Predicted_var: Economic Left/Right, Model: Llama3
------------------------------
Levene’s Test: Statistic = 11.1014, p-value = 0.0009068
Welch
  Source  ddof1      ddof2        F     p-unc      np2
0    tmp      1  675.19334  5.05382  0.024894  0.00699
==============================
Predicted_var: Economic Left/Right, Model: Mistral
------------------------------
Levene’s Test: Statistic = 7.5050, p-value = 0.006306
Welch
  Source  ddof1       ddof2         F     p-unc       np2
0    tmp      1  705.546495  5.227185  0.022532  0.007228
==============================
Predicted_var: Economic Left/Right, Model: Gemma
------------------------------
Levene’s Test: Statistic = 0.0144, p-value = 0.9045
               sum_sq     df         F    PR(>F)    eta_sq
C(tmp)       0.438080    1.0  0.196609  0.657605  0.000274
Residual  1599.831515  718.0       NaN       NaN  0.999726
==============================
```

#### top_k

```commandline
Predicted_var: Social Libertarian/Authoritarian, Model: Llama3
------------------------------
Levene’s Test: Statistic = 0.0186, p-value = 0.8916
             sum_sq     df         F    PR(>F)   eta_sq
C(top_k)    0.00889    1.0  0.014119  0.905449  0.00002
Residual  452.09492  718.0       NaN       NaN  0.99998
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Mistral
------------------------------
Levene’s Test: Statistic = 0.3491, p-value = 0.5548
              sum_sq     df         F    PR(>F)    eta_sq
C(top_k)    1.209500    1.0  1.008448  0.315613  0.001403
Residual  861.146233  718.0       NaN       NaN  0.998597
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Gemma
------------------------------
Levene’s Test: Statistic = 0.3636, p-value = 0.5467
               sum_sq     df         F    PR(>F)    eta_sq
C(top_k)     0.012417    1.0  0.005511  0.940841  0.000008
Residual  1617.616293  718.0       NaN       NaN  0.999992
==============================
Predicted_var: Economic Left/Right, Model: Llama3
------------------------------
Levene’s Test: Statistic = 0.6319, p-value = 0.4269
              sum_sq     df        F   PR(>F)    eta_sq
C(top_k)    0.003827    1.0  0.00418  0.94847  0.000006
Residual  657.436171  718.0      NaN      NaN  0.999994
==============================
Predicted_var: Economic Left/Right, Model: Mistral
------------------------------
Levene’s Test: Statistic = 3.0102, p-value = 0.08317
               sum_sq     df         F    PR(>F)    eta_sq
C(top_k)     6.712542    1.0  2.474947  0.116114  0.003435
Residual  1947.357244  718.0       NaN       NaN  0.996565
==============================
Predicted_var: Economic Left/Right, Model: Gemma
------------------------------
Levene’s Test: Statistic = 0.1315, p-value = 0.7169
               sum_sq     df        F    PR(>F)    eta_sq
C(top_k)     1.722845    1.0  0.77383  0.379329  0.001077
Residual  1598.546750  718.0      NaN       NaN  0.998923
==============================
```

#### n_beams

```commandline
Predicted_var: Social Libertarian/Authoritarian, Model: Llama3
------------------------------
Levene’s Test: Statistic = 0.5543, p-value = 0.4568
                sum_sq     df         F    PR(>F)    eta_sq
C(n_beams)    5.340611    1.0  8.582978  0.003501  0.011813
Residual    446.763199  718.0       NaN       NaN  0.988187
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Mistral
------------------------------
Levene’s Test: Statistic = 0.0934, p-value = 0.76
                sum_sq     df          F        PR(>F)    eta_sq
C(n_beams)   62.228040    1.0  55.840753  2.290294e-13  0.072161
Residual    800.127693  718.0        NaN           NaN  0.927839
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Gemma
------------------------------
Levene’s Test: Statistic = 0.7788, p-value = 0.3778
                sum_sq     df         F    PR(>F)    eta_sq
C(n_beams)     9.31840    1.0  4.160025  0.041754  0.005761
Residual    1608.31031  718.0       NaN       NaN  0.994239
==============================
Predicted_var: Economic Left/Right, Model: Llama3
------------------------------
Levene’s Test: Statistic = 70.4318, p-value = 2.527e-16
Welch
    Source  ddof1       ddof2         F     p-unc       np2
0  n_beams      1  567.949403  8.202409  0.004338  0.011295
==============================
Predicted_var: Economic Left/Right, Model: Mistral
------------------------------
Levene’s Test: Statistic = 10.4026, p-value = 0.001315
Welch
    Source  ddof1       ddof2          F         p-unc       np2
0  n_beams      1  693.765256  39.907649  4.757708e-10  0.052655
==============================
Predicted_var: Economic Left/Right, Model: Gemma
------------------------------
Levene’s Test: Statistic = 0.1965, p-value = 0.6577
                 sum_sq     df         F    PR(>F)    eta_sq
C(n_beams)    13.197709    1.0  5.970716  0.014785  0.008247
Residual    1587.071886  718.0       NaN       NaN  0.991753
==============================
```

#### prompt

```commandline
Predicted_var: Social Libertarian/Authoritarian, Model: Llama3
------------------------------
Levene’s Test: Statistic = 2.6657, p-value = 0.004782
Welch
   Source  ddof1       ddof2           F          p-unc       np2
0  prompt      9  288.931052  203.788591  1.464810e-119  0.704523
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Mistral
------------------------------
Levene’s Test: Statistic = 13.6438, p-value = 2.459e-20
Welch
   Source  ddof1       ddof2          F         p-unc       np2
0  prompt      9  288.485341  50.263304  4.879377e-54  0.508211
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Gemma
------------------------------
Levene’s Test: Statistic = 4.0450, p-value = 4.524e-05
Welch
   Source  ddof1       ddof2          F         p-unc       np2
0  prompt      9  289.042978  33.238745  8.071618e-40  0.304213
==============================
Predicted_var: Economic Left/Right, Model: Llama3
------------------------------
Levene’s Test: Statistic = 7.8454, p-value = 4.404e-11
Welch
   Source  ddof1       ddof2         F         p-unc      np2
0  prompt      9  288.163097  9.119671  3.783798e-12  0.12149
==============================
Predicted_var: Economic Left/Right, Model: Mistral
------------------------------
Levene’s Test: Statistic = 5.1657, p-value = 8.283e-07
Welch
   Source  ddof1       ddof2          F         p-unc       np2
0  prompt      9  288.830717  13.317017  8.083695e-18  0.157889
==============================
Predicted_var: Economic Left/Right, Model: Gemma
------------------------------
Levene’s Test: Statistic = 2.4350, p-value = 0.009916
Welch
   Source  ddof1       ddof2          F         p-unc       np2
0  prompt      9  288.907479  17.650332  2.973064e-23  0.180624
==============================
```

The final result is:

| independent_variable | dependent_variable               | model   | F-statistic | p-value     | test          |
|----------------------|----------------------------------|---------|-------------|-------------|---------------|
| tmp                  | Social Libertarian/Authoritarian | Llama3  | 0.83885     | 0.360033    | one way anova |
| tmp                  | Social Libertarian/Authoritarian | Mistral | 1.318059    | 0.251323    | one way anova |
| tmp                  | Social Libertarian/Authoritarian | Gemma   | 1.960054    | 0.161939    | one way anova |
| **tmp**              | **Economic Left/Right**          | **Llama3**  | **5.05382**  | **0.024894** | **welch**     |
| **tmp**              | **Economic Left/Right**          | **Mistral** | **5.227185** | **0.022532** | **welch**     |
| tmp                  | Economic Left/Right              | Gemma   | 0.196609    | 0.657605    | one way anova |
|----------------------|----------------------------------|---------|-------------|-------------|---------------|
| fine_tune_dataset    | Social Libertarian/Authoritarian | Llama3  | 1.159349    | 0.323711    | welch         |
| **fine_tune_dataset**| **Social Libertarian/Authoritarian** | **Mistral** | **5.811853** | **6.75E-07** | **welch**     |
| **fine_tune_dataset**| **Social Libertarian/Authoritarian** | **Gemma**   | **62.991599**| **2.01E-59** | **welch**     |
| **fine_tune_dataset**| **Economic Left/Right**          | **Llama3**  | **2.464097** | **0.012285** | **one way anova** |
| **fine_tune_dataset**| **Economic Left/Right**          | **Mistral** | **88.606452**| **6.65E-74** | **welch**     |
| **fine_tune_dataset**| **Economic Left/Right**          | **Gemma**   | **40.309997**| **3.58E-43** | **welch**     |
|----------------------|----------------------------------|---------|-------------|-------------|---------------|
| top_k                | Social Libertarian/Authoritarian | Llama3  | 0.014119    | 0.905449    | one way anova |
| top_k                | Social Libertarian/Authoritarian | Mistral | 1.008448    | 0.315613    | one way anova |
| top_k                | Social Libertarian/Authoritarian | Gemma   | 0.005511    | 0.940841    | one way anova |
| top_k                | Economic Left/Right              | Llama3  | 0.00418     | 0.94847     | one way anova |
| top_k                | Economic Left/Right              | Mistral | 2.474947    | 0.116114    | one way anova |
| top_k                | Economic Left/Right              | Gemma   | 0.77383     | 0.379329    | one way anova |
|----------------------|----------------------------------|---------|-------------|-------------|---------------|
| **n_beams**          | **Social Libertarian/Authoritarian** | **Llama3**  | **8.582978** | **0.003501** | **one way anova** |
| **n_beams**          | **Social Libertarian/Authoritarian** | **Mistral** | **55.840753**| **2.29E-13** | **one way anova** |
| **n_beams**          | **Social Libertarian/Authoritarian** | **Gemma**   | **4.160025** | **0.041754** | one way anova |
| **n_beams**          | **Economic Left/Right**          | **Llama3**  | **8.202409** | **0.004338** | **welch**     |
| **n_beams**          | **Economic Left/Right**          | **Mistral** | **39.907649**| **4.76E-10** | **welch**     |
| **n_beams**          | **Economic Left/Right**          | **Gemma**   | **5.970716** | **0.014785** | **one way anova** |
|----------------------|----------------------------------|---------|-------------|-------------|---------------|
| **prompt**           | **Social Libertarian/Authoritarian** | **Llama3**  | **203.788591** | **1.46E-119** | **welch**     |
| **prompt**           | **Social Libertarian/Authoritarian** | **Mistral** | **50.263304** | **4.88E-54**  | **welch**     |
| **prompt**           | **Social Libertarian/Authoritarian** | **Gemma**   | **33.238745** | **8.07E-40**  | **welch**     |
| **prompt**           | **Economic Left/Right**          | **Llama3**  | **9.119671**  | **3.78E-12**  | **welch**     |
| **prompt**           | **Economic Left/Right**          | **Mistral** | **13.317017** | **8.08E-18**  | **welch**     |
| **prompt**           | **Economic Left/Right**          | **Gemma**   | **17.650332** | **2.97E-23**  | **welch**     |