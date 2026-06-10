# Automatic Fruit Classification using Computer Vision

## General Description

This project focuses on the development of an Artificial Intelligence and Machine Learning solution for automatic fruit image classification using Computer Vision and Deep Learning techniques.

The system aims to classify fruit images into predefined categories based on their visual features such as color, shape, texture, and edges.

## Problem Statement

In areas such as inventory management, agricultural product classification, retail systems, and automated visual recognition, it can be useful to identify fruits automatically from images.

The problem consists of building a system capable of receiving a fruit image and classifying it into one of the available fruit categories.

## General Objective

Prepare and structure a fruit image dataset for the future development of an automatic classification model using Computer Vision and Deep Learning.

## Parcial 1 Objective

The objective of this first stage is to define the project, select the dataset, perform exploratory data analysis, preprocess the images, and split the dataset into training, validation, and testing subsets.

At this stage, the main model will not be trained yet. The focus is on preparing the dataset correctly for the next phase of the project.

## Dataset

| Field     | Details                                                               |
| --------- | --------------------------------------------------------------------- |
| Name      | Fruits Classification                                                 |
| Source    | Kaggle                                                                |
| URL       | https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification |
| Classes   | Apples, Bananas, Grapes, Mangoes, Strawberries                        |
| Task Type | Multiclass image classification                                       |
| Data Type | Images                                                                |

## Why Computer Vision?

Computer Vision is suitable for this project because fruits have distinct visual characteristics that can be analyzed from images.

Some of these characteristics include:

* Color
* Shape
* Texture
* Edges
* Visual patterns

These features can be learned by Machine Learning and Deep Learning models to classify fruit images automatically.

## Repository Structure

```text
fruit-classification-ai/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── 01_EDA_Preprocessing.ipynb
│
├── reports/
│   └── parcial_1_summary.md
│
├── scripts/
│   ├── resize_images.py
│   └── split_dataset.py
│
└── data/
    ├── raw/
    │   └── .gitkeep
    ├── processed/
    │   └── .gitkeep
    └── split/
        ├── train/
        │   └── .gitkeep
        ├── val/
        │   └── .gitkeep
        └── test/
            └── .gitkeep
```

## Dataset Note

The complete image dataset will not be uploaded directly to this repository if its size is too large.

Instead, this repository includes:

* The dataset URL.
* The Google Colab notebook.
* The preprocessing code.
* The exploratory data analysis process.
* The train, validation, and test split process.

## Parcial 1 — Process

The following steps will be completed during the first stage of the project:

1. Create the GitHub repository.
2. Select and document the dataset.
3. Load the dataset in Google Colab.
4. Explore the dataset structure.
5. Count the total number of images.
6. Count the number of images per class.
7. Generate a distribution table by class.
8. Generate a distribution chart by class.
9. Visualize sample images from the dataset.
10. Analyze class balance.
11. Resize all images to 224x224 pixels.
12. Verify the final image dimensions.
13. Split the dataset into train, validation, and test subsets.
14. Document the EDA results and conclusions.

## Image Preprocessing

All images will be converted to RGB format and resized to:

```text
224x224 pixels
```

This size is commonly used in Computer Vision models and allows the dataset to have a standardized input format for future Deep Learning training.

## Train / Validation / Test Split

The dataset will be divided using the following proportion:

| Subset     | Percentage | Purpose                                           |
| ---------- | ---------: | ------------------------------------------------- |
| Train      |        70% | Used to train the model                           |
| Validation |        15% | Used to validate the model during training        |
| Test       |        15% | Used to evaluate the final model with unseen data |

## EDA Results

This section will be completed after running the Google Colab notebook.

| Metric            |  Result |
| ----------------- | ------: |
| Total images      | Pending |
| Number of classes |       5 |
| Apples            | Pending |
| Bananas           | Pending |
| Grapes            | Pending |
| Mangoes           | Pending |
| Strawberries      | Pending |

## Class Balance Analysis

This section will be completed after calculating the number of images per class.

The goal is to determine whether the dataset is balanced or imbalanced and explain how this could affect the future model training process.

## Conclusions

This section will be completed after running the exploratory data analysis and preprocessing steps.

The final conclusions will summarize:

* The total number of images.
* The number of images per class.
* Whether the dataset is balanced or imbalanced.
* The quality of the images.
* The preprocessing steps applied.
* The final train, validation, and test split.
* The readiness of the dataset for the next stage of the project.

## Current Project Status

* Dataset selected.
* GitHub repository structure planned.
* README file created.
* EDA notebook pending.
* Image preprocessing pending.
* Dataset split pending.
