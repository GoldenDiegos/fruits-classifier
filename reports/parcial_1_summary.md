# Parcial 1 Summary - Automatic Fruit Classification using Computer Vision

## 1. Project Overview

This project prepares a fruit image dataset for multiclass image classification using Computer Vision. The goal is to build an organized, validated, and preprocessed dataset that can support future Computer Vision classification work.

## 2. Parcial 1 Objective

Parcial 1 covers dataset selection, exploratory data analysis (EDA), image preprocessing, resize verification, and train/validation/test split. No model training is performed in this stage.

## 3. Dataset Information

| Field | Details |
|---|---|
| Dataset name | Fruits Classification |
| Source | Kaggle |
| URL | https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification |
| License | CC0-1.0 |
| Task type | Multiclass image classification |
| Data type | Images |
| Classes | Apple, Banana, Grape, Mango, Strawberry |

## 4. Exploratory Data Analysis

| Class | Images | Percentage |
|---|---:|---:|
| Apple | 2000 | 20.0% |
| Banana | 2000 | 20.0% |
| Grape | 2000 | 20.0% |
| Mango | 2000 | 20.0% |
| Strawberry | 2000 | 20.0% |
| **Total** | **10000** | **100.0%** |

Sample images were visualized for each class during EDA.

## 5. Class Balance Analysis

| Metric | Result |
|---|---|
| Largest class | All classes tied (2000 images each) |
| Smallest class | All classes tied (2000 images each) |
| Imbalance ratio | 1.0x |
| Balance status | BALANCED |

The dataset has an equal number of images across all five classes.

## 6. Image Quality Verification

| Metric | Result |
|---|---:|
| Total images checked | 10000 |
| Corrupted images found | 0 |
| Readable images | 10000 |

No corrupted or unreadable images were found.

## 7. Image Preprocessing

| Step | Details |
|---|---|
| Color format | Converted to RGB |
| Resize | 224x224 pixels (LANCZOS) |
| Images processed | 10000 |
| Resize errors | 0 |
| Resize verification | Passed |

All images were standardized to 224x224 pixels for future Computer Vision classification work.

## 8. Dataset Split

Split ratios: 70% train / 15% validation / 15% test (random seed = 42)

| Subset | Images | Percentage |
|---|---:|---:|
| Train | 6995 | 69.95% |
| Validation | 1500 | 15.00% |
| Test | 1505 | 15.05% |
| **Total** | **10000** | **100.0%** |

The small difference from exact ratios is expected because `train_test_split` performs integer splitting per class (2000 images -> 1399 train / 300 val / 301 test).

### Images per class per subset

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Apple | 1399 | 300 | 301 | 2000 |
| Banana | 1399 | 300 | 301 | 2000 |
| Grape | 1399 | 300 | 301 | 2000 |
| Mango | 1399 | 300 | 301 | 2000 |
| Strawberry | 1399 | 300 | 301 | 2000 |
| **Total** | **6995** | **1500** | **1505** | **10000** |

Split verification: Passed. Class distribution is preserved across all subsets.

## 9. Final Conclusions

- The dataset contains 10,000 images across 5 classes with exactly 2,000 images per class.
- The dataset is perfectly balanced with an imbalance ratio of 1.0x.
- No corrupted or unreadable images were found in the dataset.
- All 10,000 images were successfully resized to 224x224 pixels with zero errors.
- The dataset was split into train, validation, and test subsets while preserving class distribution.
- The dataset is organized, validated, preprocessed, and ready for the next course stage.
