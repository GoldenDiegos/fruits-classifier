# Automatic Fruit Classification using Computer Vision

**Institution:** Global University — Ciberseguridad y Desarrollo de Software

**Course:** Inteligencia Artificial y Machine Learning

**Team:** Hector Oropeza Pelcastre, Sharon Daniela Escobedo Davila, Diego David Lara Martínez

**Professor:** Jorge Antonio Delgado Magallanes

## General Description

This project builds an automatic fruit classification system using Computer Vision, developed in two course stages.

**Parcial 1** covers project definition, dataset selection, exploratory data analysis, image preprocessing, and dataset splitting.

**Parcial 2** covers building a Convolutional Neural Network from scratch, training it, reporting baseline metrics, fine-tuning hyperparameters, and exporting the final optimized model.

API development and production deployment are not part of either stage.

## Problem Statement

In areas such as inventory management, agricultural product classification, retail systems, and automated visual recognition, it can be useful to identify fruits automatically from images.

The problem consists of preparing a structured image dataset that can later support a multiclass fruit classification system.

## General Objective

Prepare and structure a fruit image dataset for a future Computer Vision classification model.

## Parcial 1 Objective

The objective of Parcial 1 is to define the project, document the selected dataset, perform exploratory data analysis, preprocess the images, and split the dataset into training, validation, and testing subsets.

At this stage, the model is not trained. The focus is only on making the dataset organized, validated, and ready for the next course stage.

## Dataset Information

| Field | Details |
|---|---|
| Dataset name | Fruits Classification |
| Source | Kaggle |
| Dataset URL | https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification |
| Classes | Apples, Bananas, Grapes, Mangoes, Strawberries |
| Task type | Multiclass image classification |
| Data type | Images |

## Why Computer Vision?

Computer Vision is suitable for this project because fruits have visual characteristics that can be analyzed from images.

Relevant characteristics include:

- Color
- Shape
- Texture
- Edges
- Visual patterns

These characteristics are useful for future image classification work.

## Repository Structure

```text
fruit-classification-ai/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- notebooks/
|   |-- 01_EDA_Preprocessing.ipynb
|
|-- reports/
|   |-- parcial_1_summary.md
|
|-- scripts/
|   |-- resize_images.py
|   |-- split_dataset.py
|
|-- data/
|   |-- raw/
|   |   |-- .gitkeep
|   |-- processed/
|   |   |-- .gitkeep
|   |-- split/
|       |-- train/
|       |   |-- .gitkeep
|       |-- val/
|       |   |-- .gitkeep
|       |-- test/
|           |-- .gitkeep
|
|-- fruit_neural_network_project/
    |-- project/                     # Parcial 2 — see below
        |-- main.py
        |-- models/
        |-- training/
        |-- evaluation/
        |-- deployment/
        |-- scripts/
        |-- tests/
        |-- configs/
        `-- notebooks/
            `-- 02_Training_Colab.ipynb
```

## Dataset Note

The complete image dataset must not be uploaded directly to this repository.

This repository includes:

- The dataset URL.
- The Google Colab notebook.
- The exploratory data analysis process.
- The image preprocessing process.
- The train, validation, and test split process.

Downloaded images should remain inside the local or Colab runtime data folders and should not be committed to GitHub.

## Parcial 1 Process

The first stage of the project includes:

1. Select and document the dataset.
2. Load the dataset from Kaggle in Google Colab.
3. Explore the dataset folder structure.
4. Count the total number of images.
5. Count images per class.
6. Display a class distribution table.
7. Display a class distribution bar chart.
8. Visualize sample images per class.
9. Verify corrupted or unreadable images.
10. Analyze class balance.
11. Resize all images to 224x224 pixels.
12. Verify resized image dimensions.
13. Split the dataset into train, validation, and test subsets.
14. Count images per subset.
15. Count images per class per subset.
16. Print the final EDA summary and conclusions.

## Image Preprocessing

All images are converted to RGB format and resized to:

```text
224x224 pixels
```

This standardizes the dataset dimensions and color format for future Computer Vision experiments.

## Train / Validation / Test Split

The dataset is divided using the following proportion:

| Subset | Percentage | Purpose |
|---|---:|---|
| Train | 70% | Data reserved for future model training |
| Validation | 15% | Data reserved for future validation |
| Test | 15% | Data reserved for future final evaluation |

The split is performed per class to preserve the class distribution across all subsets.

## EDA Results

| Metric | Result |
|---|---:|
| Total images | 10000 |
| Number of classes | 5 |
| Apple | 2000 |
| Banana | 2000 |
| Grape | 2000 |
| Mango | 2000 |
| Strawberry | 2000 |

## Class Balance Analysis

| Metric | Result |
|---|---|
| Largest class | All classes tied (2000 images each) |
| Smallest class | All classes tied (2000 images each) |
| Imbalance ratio | 1.0x |
| Balance status | BALANCED |

The dataset has an equal number of images across all five classes.

## Conclusions

The exploratory data analysis confirmed that the dataset contains 10,000 images distributed equally across 5 fruit classes, with 2,000 images per class.

The class balance analysis showed that the dataset is balanced, with an imbalance ratio of 1.0x. No corrupted or unreadable images were found during the image quality verification step.

All images were converted to RGB format and resized successfully to 224x224 pixels. The final dataset split was verified with 6,995 images for train, 1,500 for validation, and 1,505 for test.

The dataset is organized, validated, preprocessed, and ready for the next course stage.

## Parcial 1 Status

- EDA results obtained from Google Colab execution.
- All images resized to 224x224 pixels successfully.
- Dataset split verified: 6995 train / 1500 val / 1505 test.
- Final report available in reports/parcial_1_summary.md.
- Parcial 1 complete.

## Parcial 2 Objective

The objective of Parcial 2 is to build a Convolutional Neural Network from scratch (no pretrained backbones), train it on the prepared dataset, report baseline metrics, improve those metrics through hyperparameter fine-tuning, and export the final optimized model.

This stage lives in [`fruit_neural_network_project/project/`](fruit_neural_network_project/project/), a self-contained PyTorch project designed to run in Google Colab with GPU while keeping a clean, modular, testable codebase (`models/`, `training/`, `evaluation/`, `deployment/`, `tests/`).

## Balancing and Preprocessing

Parcial 1's EDA already confirmed the dataset is perfectly balanced (2,000 images per class, imbalance ratio 1.0x), so techniques such as SMOTE, undersampling, or oversampling do not apply. The balancing technique implemented instead is class weighting inside `CrossEntropyLoss` (`training/losses.py::compute_class_weights`), computed directly from the training split. Since the classes are equally represented, the resulting weights come out close to 1.0 for every class — evidence that the mechanism is correctly wired in and correctly diagnosed as unnecessary for this particular dataset.

## Model Architecture

`FruitCNN` (`models/fruit_cnn.py`) is built entirely from scratch using `Conv2d` / `BatchNorm2d` / activation / `MaxPool2d` / `Dropout2d` blocks followed by an adaptive average pool and a dense classifier head. It returns raw logits; Softmax is only applied at inference time. Activation function, dropout rate, and base channel width are all configurable and tunable.

## Parcial 2 Process

1. Verify class balance and implement class weighting as the balancing technique.
2. Train `FruitCNN` from scratch on the real 10,000-image split.
3. Evaluate the base model on the held-out test set: Accuracy, Precision, Recall, F1-Score.
4. Plot and save train/validation loss and accuracy curves.
5. Run an Optuna hyperparameter search (learning rate, dropout rate, weight decay, activation, base channels, batch size).
6. Retrain with the winning configuration for the full epoch budget.
7. Compare base vs. tuned metrics to confirm measurable improvement.
8. Export the final tuned model as a `.pt` checkpoint with full metadata.

## Fine-Tuning

Hyperparameter search is implemented with Optuna (`training/tuning.py`, `scripts/tune.py`), minimizing validation loss across trials with a `TPESampler` and `MedianPruner`. Each trial is a short-budget run of the exact same training loop used for the base model, so the base run and the tuned retrain never drift apart.

## Model Export

The final tuned model is serialized as `models/checkpoints/tuned_model.pt` via `torch.save`, containing the model's `state_dict` plus metadata: class names, architecture hyperparameters, the winning Optuna parameters, and final test metrics.

## Parcial 2 Status

- Training, fine-tuning, and export pipeline implemented and covered by unit tests (`pytest`, all passing).
- End-to-end smoke-tested locally on a synthetic dataset (CPU) to validate wiring.
- Not yet executed on the real dataset — pending a full run in Google Colab with GPU via [`notebooks/02_Training_Colab.ipynb`](fruit_neural_network_project/project/notebooks/02_Training_Colab.ipynb).
- Base vs. tuned model comparison and final exported checkpoint: Pending.
