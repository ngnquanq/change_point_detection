# Synthetic Reproducibility Summary

| Experiment | NN Acc | NN Power | NN FPR | CUSUM Acc | CUSUM Power | CUSUM FPR |
|---|---:|---:|---:|---:|---:|---:|
| mlp_s1 | 0.9190 | 0.8650 | 0.0270 | 0.8915 | 0.9750 | 0.1920 |
| rescnn_s1_paper | 0.9255 | 0.8740 | 0.0230 | 0.8915 | 0.9750 | 0.1920 |
| mlp_s1prime | 0.7290 | 0.6940 | 0.2360 | 0.5005 | 1.0000 | 0.9990 |
| rescnn_s1prime_paper | 0.7435 | 0.6140 | 0.1270 | 0.5005 | 1.0000 | 0.9990 |
| mlp_s2 | 0.7540 | 0.6730 | 0.1650 | 0.5000 | 1.0000 | 1.0000 |
| rescnn_s2_paper | 0.7525 | 0.6750 | 0.1700 | 0.5000 | 1.0000 | 1.0000 |
| mlp_s3 | 0.5970 | 0.7730 | 0.5790 | 0.5110 | 1.0000 | 0.9780 |
| rescnn_s3_paper | 0.9475 | 0.9220 | 0.0270 | 0.5110 | 1.0000 | 0.9780 |

Localization is intentionally not part of the fixed-window summary table.
The deterministic localization demo is regenerated in `models/mlp_s1/plots/fig4_localization_demo.png`.
