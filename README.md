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

## Parcial 2 Results

Trained and fine-tuned on the real 10,000-image split via [`notebooks/02_Training_Colab.ipynb`](fruit_neural_network_project/project/notebooks/02_Training_Colab.ipynb) on Google Colab GPU. Full artifacts (metrics, curves, confusion matrices, Optuna trial log) are in [`fruit_neural_network_project/project/reports/parcial_2/`](fruit_neural_network_project/project/reports/parcial_2/).

| Metric | Base Model | Tuned Model |
|---|---:|---:|
| Accuracy | 0.5429 | 0.6100 |
| Precision (macro) | 0.5513 | 0.6037 |
| Recall (macro) | 0.5429 | 0.6100 |
| F1 (macro) | 0.5122 | 0.6000 |
| Best validation loss | 1.0962 | 1.0090 |

Winning hyperparameters (Optuna): `base_channels=32`, `activation=leaky_relu`, `learning_rate≈2.09e-4`, `dropout_rate≈0.149`, `weight_decay≈9.57e-5`, `batch_size=32`.

The improvement is concentrated in the two classes the base model struggled with most: Apple recall went from 15% to 33%, and Mango recall from 37% to 68% (Banana and Grape recall dipped slightly as a capacity trade-off, Strawberry stayed ~85%).

**Note:** this run used a reduced Optuna budget (5 trials / 5 search epochs / 10 final epochs) due to a time constraint during the training session. A fuller search (20 trials / 12 search epochs / 30 final epochs, the notebook's default) is expected to improve on these numbers further.

## Parcial 2 Status

- Training, fine-tuning, and export pipeline implemented and covered by unit tests (`pytest`, all passing).
- Executed end-to-end on the real dataset in Google Colab with GPU.
- Base vs. tuned model comparison: complete (see table above and `reports/parcial_2/comparison_table.md`).
- Final exported checkpoint: [`fruit_neural_network_project/project/models/checkpoints/tuned_model.pt`](fruit_neural_network_project/project/models/checkpoints/tuned_model.pt), containing the state dict plus metadata (class names, winning hyperparameters, Optuna best value, final test metrics).
- A fuller Optuna search (more trials/epochs) is planned to replace the current reduced-budget results.

## Parcial 3 Objective

The objective of Parcial 3 is to further improve accuracy using transfer learning (now allowed, unlike Parcial 2), build an applicable use case, and present the work from Parcial 2 and Parcial 3 together.

## Transfer Learning

`models/pretrained_resnet.py::build_resnet18` loads a ResNet18 pretrained on ImageNet and replaces its classifier head with a new `Linear` layer for the 5 fruit classes. Two configurations were trained and compared:
- **Frozen backbone**: only the new classifier head is trained (`requires_grad=False` on every pretrained parameter).
- **Fine-tuned (unfrozen backbone)**: the entire network is trained with a lower learning rate, letting the backbone adapt specifically to fruit images.

Both reuse the exact same balancing (class weights), dataset split, and evaluation pipeline established in Parcial 2, for a fair comparison.

## Parcial 3 Results

Trained via [`notebooks/03_Transfer_Learning_and_Demo.ipynb`](fruit_neural_network_project/project/notebooks/03_Transfer_Learning_and_Demo.ipynb) on the same real 10,000-image split. Full artifacts are in [`fruit_neural_network_project/project/reports/parcial_3/`](fruit_neural_network_project/project/reports/parcial_3/).

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| CNN Base (Parcial 2, from scratch) | 0.5429 | 0.5513 | 0.5429 | 0.5122 |
| CNN Tuned (Parcial 2, Optuna) | 0.6100 | 0.6037 | 0.6100 | 0.6000 |
| ResNet18 (frozen backbone) | 0.8445 | 0.8437 | 0.8445 | 0.8438 |
| **ResNet18 (fine-tuned, unfrozen)** | **0.9143** | **0.9145** | **0.9143** | **0.9139** |

The frozen-backbone run took ~6 minutes (15 epochs, no early stop) and already improved +23.5 points over the best Parcial 2 model. Unfreezing the backbone with a lower learning rate (`1e-4` vs. `1e-3`) pushed accuracy to 91.4% before early stopping triggered at epoch 7 (validation loss stopped improving after epoch 2, at which point training accuracy kept climbing — a sign of the model starting to overfit, correctly caught by `EarlyStopping`).

## Applicable Use Case: Gradio Demo

[`scripts/gradio_demo.py`](fruit_neural_network_project/project/scripts/gradio_demo.py) launches an interactive demo: upload a fruit photo, get back the predicted class and a per-class confidence breakdown. It reuses `FruitPredictor` (`deployment/inference.py`), which now reconstructs the exact model architecture from the checkpoint's own metadata — fixing a bug where it previously always assumed default architecture hyperparameters, which would have crashed loading `tuned_model.pt`. Run with `python scripts/gradio_demo.py --model-path models/checkpoints/resnet18_finetuned_model.pt --share` for a temporary public link (used from Colab in the demo notebook).

Tested with a real photo (not from the training dataset) of two bananas on a blue surface: correctly classified as Banana with 65% confidence (Mango was the second guess at 26%, a reasonable confusion given similar shape/color).

## Presentation

Slide-by-slide content covering both Parcial 2 and Parcial 3 is in [`fruit_neural_network_project/project/reports/presentation_parcial2_3.md`](fruit_neural_network_project/project/reports/presentation_parcial2_3.md).

## Parcial 3 Status

- ResNet18 transfer learning (both frozen and fine-tuned) implemented, tested, and executed end-to-end on the real dataset in Google Colab with GPU.
- Applicable use case (Gradio demo) implemented, tested locally against the existing `tuned_model.pt`, and verified with a real (non-dataset) photo during the Colab run.
- Presentation content written with real results (no pending placeholders).
- Final exported checkpoint: [`fruit_neural_network_project/project/models/checkpoints/resnet18_finetuned_model.pt`](fruit_neural_network_project/project/models/checkpoints/resnet18_finetuned_model.pt).
