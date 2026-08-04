# Parcial 3 - Model Comparison

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| Base CNN (from scratch) | 0.5429 | 0.5513 | 0.5429 | 0.5122 |
| Tuned CNN (Optuna) | 0.6100 | 0.6037 | 0.6100 | 0.6000 |
| ResNet18 (frozen backbone) | 0.8445 | 0.8437 | 0.8445 | 0.8439 |
| ResNet18 (fine-tuned, unfrozen) | 0.9143 | 0.9145 | 0.9143 | 0.9139 |