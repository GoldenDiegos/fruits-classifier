# Fruit Neural Network Project

**Institución:** Global University — Ciberseguridad y Desarrollo de Software

**Materia:** Inteligencia Artificial y Machine Learning

**Equipo:** Hector Oropeza Pelcastre, Sharon Daniela Escobedo Davila, Diego David Lara Martínez

**Profesor:** Jorge Antonio Delgado Magallanes

Proyecto educativo y profesional para entender los fundamentos de una red neuronal aplicada a clasificación de frutas.

Repositorio base analizado: `https://github.com/GoldenDiegos/fruits-classifier`

## Objetivo

Diseñar una red neuronal en Python para clasificación multiclase de imágenes de frutas.

Clases esperadas:

- Apples
- Bananas
- Grapes
- Mangoes
- Strawberries

## Arquitectura del proyecto

```text
project/
├── data/
├── notebooks/
├── configs/
├── models/
├── training/
├── evaluation/
├── deployment/
├── tests/
└── main.py
```

## Modelos incluidos

### 1. SimpleDenseFruitNet

Modelo didáctico basado en capas densas.
Sirve para entender:

- Neuronas
- Pesos
- Bias
- Activaciones
- Forward pass
- Loss
- Backpropagation
- Optimizer step

### 2. FruitCNN

Modelo recomendado para imágenes.
Usa:

- Conv2D
- BatchNorm
- ReLU / LeakyReLU / GELU / Swish
- MaxPooling
- Dropout
- Dense classifier
- CrossEntropyLoss
- AdamW

## Crear ambiente en VS Code

Desde la carpeta `project/`:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Estructura esperada del dataset

El loader usa `torchvision.datasets.ImageFolder`, por eso los folders deben verse así:

```text
data/split/train/
├── Apples/
├── Bananas/
├── Grapes/
├── Mangoes/
└── Strawberries/

/data/split/val/
├── Apples/
├── Bananas/
├── Grapes/
├── Mangoes/
└── Strawberries/
```

## Ejecutar entrenamiento

```bash
python main.py --train-dir data/split/train --val-dir data/split/val --epochs 30 --batch-size 32 --model-name fruit_cnn
```

Entrenamiento rápido de prueba:

```bash
python main.py --train-dir data/split/train --val-dir data/split/val --epochs 2 --batch-size 8
```

## Ejecutar tests

```bash
pytest
```

## Ejecutar API placeholder

```bash
uvicorn deployment.api:app --reload
```

Abrir:

```text
http://127.0.0.1:8000/health
```

## Flujo de aprendizaje de una red neuronal

1. Forward pass
2. Compute loss
3. Backpropagation
4. Update weights
5. Repeat

Código clave en `training/trainer.py`:

```python
logits = model(images)
loss = loss_function(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Diseño recomendado

Para este problema, el baseline recomendado es:

```text
Problem: Multiclass image classification
Model: FruitCNN
Loss: CrossEntropyLoss
Optimizer: AdamW
Regularization: Dropout + BatchNorm + Weight Decay
Prevention: EarlyStopping
Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
```

## Proporción profesional de esfuerzo

```text
Data preparation   70%
Architecture       15%
Training           10%
Deployment          5%
```

La arquitectura importa, pero los datos mandan. Garbage in, garbage out. Clásico, brutal y todavía vigente.

## Scripts incluidos

Crear dataset sintético mínimo para validar el pipeline sin descargar datos reales:

```bash
python scripts/create_sample_dataset.py
```

Clonar el repo de referencia dentro de `external/fruits-classifier`:

```bash
python scripts/clone_reference_repo.py
```

## Nota sobre ejecución en CPU

`FruitCNN` está configurado como una CNN compacta para aprendizaje y ejecución local. En CPU puede ser lento con imágenes de 224x224; en GPU mejora bastante. Para el siguiente proyecto serio de entrenamiento, el salto natural es transfer learning con MobileNetV2, ResNet18 o EfficientNet.
