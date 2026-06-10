# Automatic Fruit Classification using Computer Vision

## General Description

This project focuses on preparing an image dataset for automatic fruit classification using Computer Vision.

The repository corresponds to Parcial 1. This stage covers project definition, dataset selection, exploratory data analysis, image preprocessing, and dataset splitting.

No model training, API development, fine-tuning, or deployment is included in this stage.

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
    |-- raw/
    |   |-- .gitkeep
    |-- processed/
    |   |-- .gitkeep
    |-- split/
        |-- train/
        |   |-- .gitkeep
        |-- val/
        |   |-- .gitkeep
        |-- test/
            |-- .gitkeep
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

This section will be completed after running the Google Colab notebook.

| Metric | Result |
|---|---:|
| Total images | Pending |
| Number of classes | 5 |
| Apples | Pending |
| Bananas | Pending |
| Grapes | Pending |
| Mangoes | Pending |
| Strawberries | Pending |

## Class Balance Analysis

This section will be completed after calculating the number of images per class.

| Metric | Result |
|---|---:|
| Largest class | Pending |
| Smallest class | Pending |
| Imbalance ratio | Pending |
| Balance status | Pending |

The goal is to determine whether the dataset is balanced, moderately imbalanced, or imbalanced.

## Conclusions

This section will be completed after running the exploratory data analysis and preprocessing notebook.

The final conclusions will summarize:

- The total number of images.
- The number of images per class.
- Whether the dataset is balanced or imbalanced.
- The result of corrupted image verification.
- The preprocessing steps applied.
- The final train, validation, and test split.
- The readiness of the dataset for the next course stage.

## Current Project Status

- Dataset selected.
- Repository structure created.
- README file updated for Parcial 1.
- Google Colab EDA and preprocessing notebook available.
- EDA results pending execution in Google Colab.
- Final report pending.
