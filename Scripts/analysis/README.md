#### Data Analysis

We want to prove/disprove certain hypotheses.

##### Hypothesis 1
RQ1: Is fine-tuning a contributing factor toward changing the political bias of a model as measured by PCT tests? Among all generation factors that affect a model's response on the PCT scale, the most important one is whether the model was fine-tuned or not.

We test this using factorial anova, with the following factors: a) fine_tuned: captures whether a model is fine_tuned or not, b) prompt, c) top_k, d) number of beams, and e) temperature.

Preliminary results [factorial-anova.py](factorial-anova.py):
```asdl
Predicted_var: Social Libertarian/Authoritarian, Model: Llama3
------------------------------
                           sum_sq     df            F         PR(>F)    eta_sq
Intercept             1621.563853    1.0  8991.582816   0.000000e+00  0.781979
C(fine_tune_dataset)     0.387434    1.0     2.148322   1.431710e-01  0.000187
C(n_beams)               5.340611    1.0    29.613726   7.278998e-08  0.002575
C(tmp)                   0.527583    1.0     2.925454   8.763203e-02  0.000254
C(top_k)                 0.008890    1.0     0.049296   8.243573e-01  0.000004
C(prompt)              318.517539    9.0   196.242440  1.668961e-185  0.153601
Residual               127.321752  706.0          NaN            NaN  0.061399
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Mistral
------------------------------
                           sum_sq     df            F         PR(>F)    eta_sq
Intercept             1361.400066    1.0  2711.053231  5.976885e-244  0.612208
C(fine_tune_dataset)     4.550065    1.0     9.060869   2.704453e-03  0.002046
C(n_beams)              62.228040    1.0   123.919143   1.245471e-26  0.027983
C(tmp)                   1.580157    1.0     3.146679   7.651262e-02  0.000711
C(top_k)                 1.209500    1.0     2.408564   1.211211e-01  0.000544
C(prompt)              438.258435    9.0    96.970593  3.877673e-117  0.197080
Residual               354.529537  706.0          NaN            NaN  0.159428
==============================
Predicted_var: Social Libertarian/Authoritarian, Model: Gemma
------------------------------
                           sum_sq     df           F        PR(>F)    eta_sq
Intercept              488.244181    1.0  322.880686  9.793176e-60  0.231849
C(fine_tune_dataset)    44.212071    1.0   29.237878  8.768409e-08  0.020995
C(n_beams)               9.318400    1.0    6.162350  1.328114e-02  0.004425
C(tmp)                   4.403911    1.0    2.912350  8.834346e-02  0.002091
C(top_k)                 0.012417    1.0    0.008211  9.278230e-01  0.000006
C(prompt)              492.103678    9.0   36.159223  1.001212e-52  0.233682
Residual              1067.578233  706.0         NaN           NaN  0.506953
==============================
Predicted_var: Economic Left/Right, Model: Llama3
------------------------------
                          sum_sq     df           F         PR(>F)    eta_sq
Intercept             764.266957    1.0  954.144320  3.264177e-133  0.537570
C(fine_tune_dataset)    0.039010    1.0    0.048702   8.254007e-01  0.000027
C(n_beams)              7.425742    1.0    9.270622   2.415174e-03  0.005223
C(tmp)                  4.595209    1.0    5.736860   1.687205e-02  0.003232
C(top_k)                0.003827    1.0    0.004778   9.449108e-01  0.000003
C(prompt)              79.872164    9.0   11.079545   2.896909e-16  0.056180
Residual              565.504045  706.0         NaN            NaN  0.397764
==============================
Predicted_var: Economic Left/Right, Model: Mistral
------------------------------
                           sum_sq     df           F         PR(>F)    eta_sq
Intercept             1951.381335    1.0  930.533534  5.153269e-131  0.499656
C(fine_tune_dataset)    41.293627    1.0   19.691233   1.055689e-05  0.010573
C(n_beams)             102.891601    1.0   49.064774   5.783703e-12  0.026346
C(tmp)                  14.123202    1.0    6.734775   9.651500e-03  0.003616
C(top_k)                 6.712542    1.0    3.200935   7.402474e-02  0.001719
C(prompt)              308.526972    9.0   16.347092   1.553010e-24  0.078999
Residual              1480.521842  706.0         NaN            NaN  0.379091
==============================
Predicted_var: Economic Left/Right, Model: Gemma
------------------------------
                           sum_sq     df          F        PR(>F)    eta_sq
Intercept              122.139226    1.0  68.721934  5.713395e-16  0.070912
C(fine_tune_dataset)    41.092358    1.0  23.120715  1.860340e-06  0.023857
C(n_beams)              13.197709    1.0   7.425723  6.589021e-03  0.007662
C(tmp)                   0.438080    1.0   0.246487  6.197143e-01  0.000254
C(top_k)                 1.722845    1.0   0.969363  3.251764e-01  0.001000
C(prompt)              289.047548    9.0  18.070368  3.565451e-27  0.167816
Residual              1254.771056  706.0        NaN           NaN  0.728498
==============================
```
