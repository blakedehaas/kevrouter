# Kev Decision Model Benchmark Comparison Matrix

| Model Size | Variant | Split | Accuracy (%) | Optimal Match (%) | Avg Cost / Query | Cost / Correct Ans | Latency (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| KEV-0.8B   | Base (Zero-Shot)           | val        |        45.84% |              5.19% | $       0.01189 | $         0.02594 |        0.2 ms |
| KEV-0.8B   | Fine-Tuned (CE Baseline)   | val        |        49.47% |              4.61% | $       0.04010 | $         0.08107 |        0.2 ms |
| KEV-0.8B   | Optimized (Cost-Aggr, mu=2.5) | val        |        45.19% |             11.93% | $       0.00171 | $         0.00379 |        0.1 ms |
| KEV-0.8B   | Base (Zero-Shot)           | test_id    |        46.74% |              6.02% | $       0.01031 | $         0.02206 |        0.2 ms |
| KEV-0.8B   | Fine-Tuned (CE Baseline)   | test_id    |        49.22% |              5.69% | $       0.03862 | $         0.07846 |        0.1 ms |
| KEV-0.8B   | Optimized (Cost-Aggr, mu=2.5) | test_id    |        45.18% |             11.79% | $       0.00159 | $         0.00353 |        0.1 ms |
| KEV-0.8B   | Base (Zero-Shot)           | test_ood   |        41.73% |              5.04% | $       0.09641 | $         0.23106 |        0.2 ms |
| KEV-0.8B   | Fine-Tuned (CE Baseline)   | test_ood   |        37.77% |              9.71% | $       0.16884 | $         0.44703 |        0.1 ms |
| KEV-0.8B   | Optimized (Cost-Aggr, mu=2.5) | test_ood   |        28.78% |             12.95% | $       0.03440 | $         0.11955 |        0.1 ms |
| KEV-4B     | Base (Zero-Shot)           | val        |        46.17% |              5.19% | $       0.00414 | $         0.00897 |        0.3 ms |
| KEV-4B     | Fine-Tuned (CE Baseline)   | val        |        46.50% |              7.16% | $       0.01203 | $         0.02586 |        0.4 ms |
| KEV-4B     | Optimized (Cost-Aggr, mu=2.5) | val        |        44.12% |              9.30% | $       0.00348 | $         0.00788 |        0.4 ms |
| KEV-4B     | Base (Zero-Shot)           | test_id    |        46.50% |              6.84% | $       0.00440 | $         0.00946 |        0.4 ms |
| KEV-4B     | Fine-Tuned (CE Baseline)   | test_id    |        45.59% |              7.50% | $       0.01285 | $         0.02819 |        0.4 ms |
| KEV-4B     | Optimized (Cost-Aggr, mu=2.5) | test_id    |        43.28% |              9.73% | $       0.00343 | $         0.00793 |        0.4 ms |
| KEV-4B     | Base (Zero-Shot)           | test_ood   |        44.96% |              6.12% | $       0.12356 | $         0.27480 |        0.4 ms |
| KEV-4B     | Fine-Tuned (CE Baseline)   | test_ood   |        32.01% |              6.47% | $       0.05819 | $         0.18178 |        0.3 ms |
| KEV-4B     | Optimized (Cost-Aggr, mu=2.5) | test_ood   |        24.46% |              6.12% | $       0.03795 | $         0.15516 |        0.3 ms |
| KEV-9B     | Base (Zero-Shot)           | val        |        44.86% |              8.56% | $       0.00740 | $         0.01649 |        0.5 ms |
| KEV-9B     | Fine-Tuned (CE Baseline)   | val        |        47.49% |              4.03% | $       0.00709 | $         0.01494 |        0.5 ms |
| KEV-9B     | Optimized (Cost-Aggr, mu=2.5) | val        |        44.86% |              8.15% | $       0.00165 | $         0.00367 |        0.5 ms |
| KEV-9B     | Base (Zero-Shot)           | test_id    |        45.18% |              7.17% | $       0.00754 | $         0.01670 |        0.5 ms |
| KEV-9B     | Fine-Tuned (CE Baseline)   | test_id    |        47.40% |              4.53% | $       0.00698 | $         0.01473 |        0.5 ms |
| KEV-9B     | Optimized (Cost-Aggr, mu=2.5) | test_id    |        46.66% |              9.48% | $       0.00164 | $         0.00352 |        0.4 ms |
| KEV-9B     | Base (Zero-Shot)           | test_ood   |        41.37% |              8.27% | $       0.09551 | $         0.23087 |        0.5 ms |
| KEV-9B     | Fine-Tuned (CE Baseline)   | test_ood   |        37.77% |              6.47% | $       0.08305 | $         0.21987 |        0.5 ms |
| KEV-9B     | Optimized (Cost-Aggr, mu=2.5) | test_ood   |        14.75% |              4.32% | $       0.03142 | $         0.21304 |        0.5 ms |
