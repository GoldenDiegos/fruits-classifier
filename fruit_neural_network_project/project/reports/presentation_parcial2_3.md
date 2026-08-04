# Presentación — Parcial 2 y Parcial 3

Contenido para armar la presentación (slide por sección). Todos los números son reales, obtenidos corriendo `notebooks/02_Training_Colab.ipynb` y `notebooks/03_Transfer_Learning_and_Demo.ipynb` sobre el dataset real de Kaggle en Google Colab (GPU).

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
- Se eligió **ResNet18** (preentrenado en ImageNet, 1.4 millones de imágenes) sobre MobileNetV2 — mejor accuracy típico en datasets de este tamaño
- El backbone ya sabe detectar bordes, texturas, formas y colores — solo hace falta enseñarle a distinguir las 5 frutas

---

## Slide 7: Parcial 3 — Fase 1: Backbone Congelado

- ResNet18 preentrenado con backbone congelado (`requires_grad=False`)
- Solo se entrena una capa clasificadora nueva (`Linear(512, 5)`)
- Una sola corrida directa (no búsqueda de Optuna) — converge rápido y es poco sensible a hiperparámetros
- Misma técnica de balanceo (class weights) y mismo dataset/split que Parcial 2, para comparación justa
- **Resultado: 6 minutos de entrenamiento, 15 épocas**

| Métrica | ResNet18 (Backbone Congelado) |
|---|---:|
| Accuracy | 84.5% |
| Precision (macro) | 84.4% |
| Recall (macro) | 84.5% |
| F1 (macro) | 84.4% |

- +23.5 puntos sobre el mejor modelo de Parcial 2 (61.0%)

---

## Slide 8: Parcial 3 — Fase 2: Fine-Tuning Completo

- Se descongeló el backbone completo (`--unfreeze-backbone`) para ajustar también las capas internas a fotos de fruta específicamente
- Learning rate mucho más bajo (`1e-4` vs. `1e-3`) para no destruir los pesos preentrenados
- Early stopping se activó en la época 7 — el modelo empezó a sobreajustar (train accuracy subía, validación se estancaba) y el mecanismo cortó a tiempo

| Métrica | ResNet18 (Fine-Tuning Completo) |
|---|---:|
| Accuracy | **91.4%** |
| Precision (macro) | 91.5% |
| Recall (macro) | 91.4% |
| F1 (macro) | 91.4% |

- +6.9 puntos adicionales sobre el backbone congelado
- +30.4 puntos en total sobre el mejor modelo de Parcial 2

---

## Slide 9: Comparación de los 4 Modelos

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| CNN Base (desde cero) | 54.3% | 55.1% | 54.3% | 51.2% |
| CNN Tuned (Optuna) | 61.0% | 60.4% | 61.0% | 60.0% |
| ResNet18 (backbone congelado) | 84.5% | 84.4% | 84.5% | 84.4% |
| **ResNet18 (fine-tuning completo)** | **91.4%** | **91.5%** | **91.4%** | **91.4%** |

- Progresión clara: desde cero → tuning de hiperparámetros → transfer learning → fine-tuning completo
- Cada etapa aportó una mejora medible y explicable

---

## Slide 10: Caso Aplicable — Demo Interactiva

- Demo funcional con Gradio: subir una foto de fruta → clase predicha + confianza + desglose de probabilidades por clase
- Reutiliza `FruitPredictor` (reconstruye automáticamente la arquitectura correcta desde los metadatos del checkpoint)
- Se ejecuta directo en Colab: `python scripts/gradio_demo.py --model-path models/checkpoints/resnet18_finetuned_model.pt --share` genera un link público temporal
- **Probado con una foto real (no del dataset)**: dos bananas maduras sobre una superficie azul → clasificada correctamente como Banana (65% de confianza), con Mango como segunda opción (26%, forma/color similares)
- Ideal para mostrar en vivo durante la presentación o grabar un video corto

---

## Slide 11: Conclusiones y Siguientes Pasos

- El dataset balanceado + class weights se mantuvo consistente en las 3 etapas del proyecto, permitiendo comparaciones justas
- El salto más grande vino de usar transfer learning (61.0% → 84.5%), y el fine-tuning completo lo llevó aún más lejos (84.5% → 91.4%)
- El modelo final (ResNet18 fine-tuned) queda exportado en `models/checkpoints/resnet18_finetuned_model.pt` con toda su metadata (hiperparámetros, métricas de test)
- Siguientes pasos posibles: probar otros backbones preentrenados (EfficientNet, MobileNetV2), completar el endpoint `/predict` de FastAPI para un despliegue más formal, ampliar el dataset con más variedad de fondos/ángulos para mejorar la robustez en el mundo real
