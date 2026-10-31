# KevRouter Benchmark Comparison Matrix

| Model | Decision Rule | Split | Queries | Accuracy (%) | Optimal Match (%) | Avg Cost / Query | Cost / Correct Ans | Latency (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Kev-0.8B (Base) | ArgMax (Zero-Shot) | val | 1,215 | 47.49% | 3.46% | $0.00712 | $0.01499 | 91.5 |
| Kev-0.8B (Base) | ArgMax (Zero-Shot) | test_id | 1,213 | 45.75% | 3.63% | $0.00765 | $0.01672 | 92.9 |
| Kev-0.8B (Base) | ArgMax (Zero-Shot) | test_ood | 278 | 15.47% | 2.88% | $0.06859 | $0.44341 | 90.9 |
| Kev-0.8B (Fine-Tuned) | Calibrated ArgMax | val | 1,215 | 59.42% | 39.01% | $0.01577 | $0.02654 | 90.8 |
| Kev-0.8B (Fine-Tuned) | Cost-Utility (mu=2.5) | val | 1,215 | 54.07% | 36.38% | $0.00098 | $0.00181 | 90.8 |
| Kev-0.8B (Fine-Tuned) | Cascading (tau=0.6) | val | 1,215 | 59.42% | 39.01% | $0.01577 | $0.02654 | 90.8 |
| Kev-0.8B (Fine-Tuned) | Calibrated ArgMax | test_id | 1,213 | 56.72% | 36.44% | $0.01447 | $0.02551 | 98.9 |
| Kev-0.8B (Fine-Tuned) | Cost-Utility (mu=2.5) | test_id | 1,213 | 53.09% | 35.28% | $0.00087 | $0.00164 | 98.9 |
| Kev-0.8B (Fine-Tuned) | Cascading (tau=0.6) | test_id | 1,213 | 56.72% | 36.44% | $0.01447 | $0.02551 | 98.9 |
| Kev-0.8B (Fine-Tuned) | Calibrated ArgMax | test_ood | 278 | 40.29% | 25.18% | $0.04198 | $0.10421 | 95.7 |
| Kev-0.8B (Fine-Tuned) | Cost-Utility (mu=2.5) | test_ood | 278 | 40.29% | 25.18% | $0.04198 | $0.10421 | 95.7 |
| Kev-0.8B (Fine-Tuned) | Cascading (tau=0.6) | test_ood | 278 | 40.29% | 25.18% | $0.04198 | $0.10421 | 95.7 |
