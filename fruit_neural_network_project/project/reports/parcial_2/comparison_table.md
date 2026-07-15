# Parcial 2 — Base vs Tuned Model Comparison

| Metric | Base Model | Tuned Model |
|---|---:|---:|
| Accuracy | 0.5429 | 0.6100 |
| Precision (macro) | 0.5513 | 0.6037 |
| Recall (macro) | 0.5429 | 0.6100 |
| F1 (macro) | 0.5122 | 0.6000 |
| Best validation loss | 1.0962 | 1.0090 |

## Winning hyperparameters (Optuna)

```json
{
  "model_name": "fruit_cnn",
  "activation": "leaky_relu",
  "dropout_rate": 0.14881529393791154,
  "base_channels": 32,
  "learning_rate": 0.00020914981329035596,
  "weight_decay": 9.565499215943819e-05,
  "batch_size": 32
}
```