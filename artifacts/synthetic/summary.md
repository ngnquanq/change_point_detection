# Synthetic Reproducibility Summary

| Experiment | NN Acc | NN Power | NN FPR | CUSUM Acc | CUSUM Power | CUSUM FPR |
|---|---:|---:|---:|---:|---:|---:|
| mlp_s1 | 0.9190 | 0.8650 | 0.0270 | 0.8915 | 0.9750 | 0.1920 |
| mlp_s1prime | 0.7290 | 0.6940 | 0.2360 | 0.5005 | 1.0000 | 0.9990 |
| mlp_s2 | 0.7540 | 0.6730 | 0.1650 | 0.5000 | 1.0000 | 1.0000 |
| mlp_s3 | 0.5970 | 0.7730 | 0.5790 | 0.5110 | 1.0000 | 0.9780 |

Localization is intentionally not part of the fixed-window summary table.
The deterministic localization demo is regenerated in `models/mlp_s1/plots/fig4_localization_demo.png`.
