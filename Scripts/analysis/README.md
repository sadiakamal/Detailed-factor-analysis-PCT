#### Data Analysis

We want to prove/disprove certain hypotheses.

##### Hypothesis 1
RQ1: Is fine-tuning a contributing factor toward changing the political bias of a model as measured by PCT tests? Among all generation factors that affect a model's response on the PCT scale, the most important one is whether the model was fine-tuned or not.

We test this using factorial anova, with the following factors: a) fine_tuned: captures which dataset a model was finetuned on, including no finetuning, b) prompt, c) top_k, d) number of beams, and e) temperature.

Preliminary results [factorial-anova.py](factorial-anova.py):
```asdl
Predicted_var: Social Libertarian/Authoritarian, Model: Llama3
------------------------------
                          sum_sq     df           F         PR(>F)    eta_sq
C(fine_tune_dataset)    7.121646    8.0    5.160183   2.841545e-06  0.015752
C(n_beams)              5.340611    1.0   30.957488   3.759009e-08  0.011813
C(tmp)                  0.527583    1.0    3.058200   8.076954e-02  0.001167
C(top_k)                0.008890    1.0    0.051533   8.204837e-01  0.000020
C(prompt)             318.517539    9.0  205.147195  1.575121e-189  0.704523
Residual              120.587540  699.0         NaN            NaN  0.266725
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Mistral
------------------------------
                          sum_sq     df           F         PR(>F)    eta_sq
C(fine_tune_dataset)   47.485889    8.0   13.315672   5.229199e-18  0.055065
C(n_beams)             62.228040    1.0  139.596527   1.693617e-29  0.072161
C(tmp)                  1.580157    1.0    3.544775   6.014833e-02  0.001832
C(top_k)                1.209500    1.0    2.713279   9.996570e-02  0.001403
C(prompt)             438.258435    9.0  109.238635  5.472152e-127  0.508211
Residual              311.593712  699.0         NaN            NaN  0.361329
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Gemma
------------------------------
                          sum_sq     df           F         PR(>F)    eta_sq
C(fine_tune_dataset)  634.519779    8.0  116.162978  5.955558e-123  0.392253
C(n_beams)              9.318400    1.0   13.647526   2.377568e-04  0.005761
C(tmp)                  4.403911    1.0    6.449872   1.131151e-02  0.002722
C(top_k)                0.012417    1.0    0.018185   8.927668e-01  0.000008
C(prompt)             492.103678    9.0   80.080479  1.859217e-101  0.304213
Residual              477.270525  699.0         NaN            NaN  0.295043
==============================
Predicted_var: Economic Left/Right, Model: Llama3
------------------------------
                          sum_sq     df          F        PR(>F)    eta_sq
C(fine_tune_dataset)   17.736065    8.0   2.828895  4.278287e-03  0.026977
C(n_beams)              7.425742    1.0   9.475224  2.164115e-03  0.011295
C(tmp)                  4.595209    1.0   5.863472  1.571180e-02  0.006990
C(top_k)                0.003827    1.0   0.004884  9.443074e-01  0.000006
C(prompt)              79.872164    9.0  11.324070  1.212395e-16  0.121490
Residual              547.806990  699.0        NaN           NaN  0.833243
==============================
Predicted_var: Economic Left/Right, Model: Mistral
------------------------------
                          sum_sq     df          F        PR(>F)    eta_sq
C(fine_tune_dataset)  717.318801    8.0  77.906762  1.346033e-91  0.367090
C(n_beams)            102.891601    1.0  89.399039  4.781369e-20  0.052655
C(tmp)                 14.123202    1.0  12.271174  4.892968e-04  0.007228
C(top_k)                6.712542    1.0   5.832301  1.599018e-02  0.003435
C(prompt)             308.526972    9.0  29.785408  4.312624e-44  0.157889
Residual              804.496668  699.0        NaN           NaN  0.411703
==============================
Predicted_var: Economic Left/Right, Model: Gemma
------------------------------
                          sum_sq     df          F        PR(>F)    eta_sq
C(fine_tune_dataset)  476.446765    8.0  50.803869  9.837088e-65  0.297729
C(n_beams)             13.197709    1.0  11.258251  8.355408e-04  0.008247
C(tmp)                  0.438080    1.0   0.373702  5.411918e-01  0.000274
C(top_k)                1.722845    1.0   1.469666  2.258080e-01  0.001077
C(prompt)             289.047548    9.0  27.396758  9.017820e-41  0.180624
Residual              819.416648  699.0        NaN           NaN  0.512049
==============================

```
