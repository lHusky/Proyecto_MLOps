# Detección de Fraude en Transacciones con Tarjeta de Crédito: Comparación entre un Enfoque Tradicional y MLOps

## Descripción del Proyecto

Este proyecto tiene como objetivo analizar el impacto de la adopción de prácticas MLOps en el ciclo de vida de un modelo de Machine Learning para la detección de fraude en transacciones con tarjeta de crédito.

Para ello se implementan dos enfoques:

- **No MLOps:** entrenamiento, validación y almacenamiento del modelo mediante procesos manuales.
- **MLOps:** automatización de tareas, validación de datos, seguimiento de métricas y organización modular del ciclo de vida del modelo.

El estudio permite comparar ambos enfoques y evaluar las mejoras que aporta MLOps en términos de reproducibilidad, mantenimiento y despliegue.

---

## Problematica

Las entidades financieras procesan miles de transacciones diariamente. Identificar transacciones fraudulentas de forma rápida y precisa es fundamental para reducir pérdidas económicas y mejorar la seguridad de los clientes.

El modelo desarrollado clasifica cada transacción como:

- Clase 0: Transacción legítima.
- Clase 1: Transacción fraudulenta.

---

## Dataset Utilizado

**Credit Card Fraud Detection Dataset**

Fuente:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Características principales

- 284,807 transacciones.
- 492 casos de fraude.
- Dataset altamente desbalanceado.
- Variables V1–V28 anonimizadas mediante PCA.
- Variable objetivo: `Class`.

---

## Modelos Evaluados

### Logistic Regression

```python
LogisticRegression(
    max_iter=1000,
    class_weight={0:1, 1:10},
    solver="liblinear",
    random_state=42
)
```

### HistGradientBoostingClassifier

```python
HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42
)
```

---

## Métricas de Evaluación

Debido al fuerte desbalance de clases, se emplearon métricas especializadas:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

Además, se implementó una búsqueda automática del umbral óptimo utilizando la curva Precision-Recall para maximizar el F1-Score bajo una restricción mínima de precisión.

---

## Ejecución
### Levantar entorno

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecutar entrenamiento - NO MLOPS

```bash
cd no_mlops
python train_model.py
```

### Resultado

Se generará:

```text
models/best_fraud_model_no_mlops.pkl
```

Y se mostrarán en consola:

- Matriz de confusión
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

---

## Comparación entre Enfoques

| Característica | No MLOps | MLOps |
|---------------|----------|--------|
| Entrenamiento | Manual | Automatizado |
| Validación de datos | Manual | Automática |
| Registro de métricas | Consola | Persistente |
| Reproducibilidad | Media | Alta |
| Versionado de modelos | Limitado | Completo |
| Escalabilidad | Baja | Alta |
| Mantenimiento | Complejo | Simplificado |
| Despliegue | Manual | Automatizado |

---

## Integrantes

- Luis Angel Pumayucra Chauca - 20214035
- Gerson Ademir Oviedo Soto - 20211960
- Yerson Daniel Mamani Valverde - 20211590
- Joaquin Alonso Nuñez Gonzales - 20204706
- Leonardo Miguel Pachas Cleonares - 20211961
- Diana Sofia Villalva Gomez - 20212903
