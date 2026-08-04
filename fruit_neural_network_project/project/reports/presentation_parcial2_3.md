# Presentación — Parcial 2 y Parcial 3

Contenido para armar la presentación (slide por sección). Los valores marcados como `[PLACEHOLDER]` se llenan después de correr `notebooks/03_Transfer_Learning_and_Demo.ipynb` con el dataset real.

---

## Slide 1: Portada

- Automatic Fruit Classification using Computer Vision
- Global University — Ciberseguridad y Desarrollo de Software
- Materia: Inteligencia Artificial y Machine Learning
- Equipo: Hector Oropeza Pelcastre, Sharon Daniela Escobedo Davila, Diego David Lara Martínez
- Profesor: Jorge Antonio Delgado Magallanes

---

## Slide 2: Problema y Dataset

- Clasificación automática de frutas a partir de imágenes (visión por computadora)
- Dataset: Fruits Classification (Kaggle) — 10,000 imágenes, 5 clases
- Clases: Apple, Banana, Grape, Mango, Strawberry (2,000 imágenes cada una — balanceado)
- Split: 70% train / 15% validación / 15% test, seed=42
- Imágenes normalizadas a 224x224 RGB

---

## Slide 3: Parcial 2 — Balanceo y Preprocesamiento

- Dataset verificado balanceado 1:1 (ratio de imbalance 1.0x) desde la EDA de Parcial 1
- SMOTE / undersampling / oversampling no aplican — no hay desbalance que corregir
- Técnica implementada igual: class weights dentro de `CrossEntropyLoss` (`compute_class_weights`)
- Verificado sobre el dataset real: peso = 1.0000 en las 5 clases

---

## Slide 4: Parcial 2 — Arquitectura desde Cero

- `FruitCNN`: red convolucional construida enteramente desde cero (sin modelos preentrenados)
- Bloques Conv2D → BatchNorm2D → activación → MaxPool2D → Dropout2D
- Canales base configurables (`base_channels`), activación intercambiable (ReLU/LeakyReLU/GELU/Swish)
- AdaptiveAvgPool + clasificador denso → logits crudos (softmax solo en inferencia)
- Entrenamiento: AdamW + CrossEntropyLoss (con class weights) + EarlyStopping

---

## Slide 5: Parcial 2 — Resultados

| Métrica | Modelo Base | Modelo Tuned (Optuna) |
|---|---:|---:|
| Accuracy | 54.3% | 61.0% |
| Precision (macro) | 55.1% | 60.4% |
| Recall (macro) | 54.3% | 61.0% |
| F1 (macro) | 51.2% | 60.0% |

- Fine-tuning con Optuna (búsqueda de hiperparámetros): learning rate, dropout, weight decay, activación, `base_channels`, batch size
- Configuración ganadora: `base_channels=32`, `leaky_relu`
- Mejora concentrada en las clases más débiles del modelo base: Apple (15%→33% recall), Mango (37%→68% recall)

---

## Slide 6: Parcial 3 — Motivación (Transfer Learning)

- Objetivo: mejorar el accuracy más allá de lo logrado con la CNN desde cero
- En esta etapa ya se permite usar modelos preentrenados
- Se eligió **ResNet18** (preentrenado en ImageNet) sobre MobileNetV2 — mejor accuracy típico en datasets de este tamaño (10,000 imágenes, 5 clases), la diferencia de tamaño/velocidad no es relevante a esta escala

---

## Slide 7: Parcial 3 — Enfoque

- Backbone de ResNet18 congelado (pesos preentrenados de ImageNet, `requires_grad=False`)
- Solo se entrena una capa clasificadora nueva (`Linear(512, 5)`)
- Una sola corrida directa (no búsqueda de Optuna) — transfer learning con backbone congelado converge rápido y es poco sensible a hiperparámetros
- Misma técnica de balanceo (class weights) y mismo dataset/split que Parcial 2, para comparación justa

---

## Slide 8: Parcial 3 — Resultados ResNet18

`[PLACEHOLDER — llenar con reports/parcial_3/resnet18_metrics.json después de correr el notebook]`

| Métrica | ResNet18 (Transfer Learning) |
|---|---:|
| Accuracy | `[PLACEHOLDER]` |
| Precision (macro) | `[PLACEHOLDER]` |
| Recall (macro) | `[PLACEHOLDER]` |
| F1 (macro) | `[PLACEHOLDER]` |

---

## Slide 9: Comparación de los 3 Modelos

`[PLACEHOLDER — llenar con reports/parcial_3/comparison_table.md después de correr el notebook]`

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| CNN Base (desde cero) | 54.3% | 55.1% | 54.3% | 51.2% |
| CNN Tuned (Optuna) | 61.0% | 60.4% | 61.0% | 60.0% |
| ResNet18 (Transfer Learning) | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

---

## Slide 10: Caso Aplicable — Demo Interactiva

- Demo funcional con Gradio: subir una foto de fruta → clase predicha + confianza + desglose de probabilidades por clase
- Reutiliza `FruitPredictor` (reconstruye automáticamente la arquitectura correcta desde los metadatos del checkpoint)
- Se ejecuta directo en Colab: `python scripts/gradio_demo.py --share` genera un link público temporal
- Ideal para mostrar en vivo durante la presentación o grabar un video corto

---

## Slide 11: Conclusiones y Siguientes Pasos

- El dataset balanceado + class weights se mantuvo consistente en las 3 etapas del proyecto
- La CNN desde cero (Parcial 2) y el transfer learning con ResNet18 (Parcial 3) parten del mismo pipeline de datos, permitiendo una comparación justa
- `[PLACEHOLDER — conclusión final sobre cuál modelo ganó y por cuánto, una vez con números reales]`
- Siguientes pasos posibles: descongelar el backbone de ResNet18 para fine-tuning completo, probar otros backbones preentrenados, completar el endpoint `/predict` de FastAPI para un despliegue más formal
