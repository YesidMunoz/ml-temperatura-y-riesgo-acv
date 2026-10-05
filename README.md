# Aprendizaje automático en salud: temperatura por termografía y riesgo de ACV

Proyecto de aprendizaje automático en Python con dos problemas clínicos reales:

1. **Regresión:** estimar la temperatura oral a partir de la temperatura facial medida con cámaras térmicas (termografía infrarroja).
2. **Clasificación:** predecir el riesgo de accidente cerebrovascular (ACV) a partir de datos clínicos y demográficos.

Proyecto del curso Probabilidad y Estadística · Bioingeniería · Universidad de Antioquia (2025-2).

## Parte 1 · Regresión: temperatura oral desde termografía

Se usaron las mediciones del termógrafo FLIR de un estudio clínico con más de mil voluntarios. Se seleccionaron las **4 zonas faciales con mayor correlación** con la temperatura oral (temperatura máxima del rostro y tres medidas del canto interno del ojo, la zona más usada para estimar temperatura central) y se compararon modelos con **validación cruzada K-Fold**:

| Modelo | RMSE (°C) | R² | Términos |
| --- | --- | --- | --- |
| Lineal con interacciones + Lasso | 0,251 | 0,748 | 11 |
| Lineal con interacciones | 0,252 | 0,746 | 15 |
| Polinómico grado 2 (curva de validación) | 0,252 | 0,744 | 14 |
| Lineal simple + Lasso | 0,272 | 0,703 | 2 |
| **Polinómico grado 2 + Lasso (elegido)** | **0,269** | **0,708** | **3** |

Se eligió un modelo compacto de 3 términos con un error aceptable (alrededor de 0,27 °C), en lugar del de 11 términos que tenía métricas apenas mejores:

```
T_oral = 21,0607 + 0,011552·T_max² + 0,000297·T_canto_der² + 0,000421·T_max·T_canto_der
```

## Parte 2 · Clasificación: riesgo de ACV

El conjunto de datos tiene un fuerte **desbalance**: solo el 4,9 % de los pacientes tuvo un ACV (249 de 5.110). Por eso la exactitud global engaña, y la métrica clave es el **recall de la clase ACV**: en un contexto clínico es más grave no detectar a un paciente en riesgo que generar una falsa alarma.

Pasos: imputación de IMC faltante por clase, codificación *one-hot*, estandarización, prueba de K-Means y KNN como generadores de nuevas características (descartados por bajo desempeño) y ajuste de hiperparámetros con curvas de validación.

| Modelo | Exactitud | Recall ACV | F1 ACV |
| --- | --- | --- | --- |
| Árbol de decisión | 0,92 | 0,22 | 0,24 |
| Random Forest | 0,94 | 0,06 | 0,10 |
| SVM (kernel RBF, clases balanceadas) | 0,62 | 0,37 | 0,10 |
| **Regresión logística (elegido)** | **0,75** | **0,73** | **0,25** |

La regresión logística detecta cerca de 3 de cada 4 pacientes con ACV, a costa de más falsos positivos.

## Tecnologías

Python · scikit-learn · pandas · NumPy · Matplotlib · seaborn · Jupyter

## Cómo ejecutarlo

```
pip install -r requirements.txt
jupyter notebook regresion_y_clasificacion.ipynb
```

## Datos

- [Facial and oral temperature data from a large set of human subject volunteers](https://physionet.org/content/face-oral-temp-data/1.0.0/) (PhysioNet), datos anonimizados.
- [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle).

## Autores

Trabajo en equipo de:

- **Dumar Yesid Londoño Muñoz** · [GitHub](https://github.com/YesidMunoz)
- Valentina Morales
- Brayan Ramírez
