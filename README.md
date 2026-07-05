# Deteccion de Fraude en Transacciones con Tarjeta de Credito: Comparacion entre un Enfoque Tradicional y MLOps

## Descripcion del Proyecto

Este proyecto tiene como objetivo analizar el impacto de la adopcion de practicas MLOps en el ciclo de vida de un modelo de Machine Learning para la deteccion de fraude en transacciones con tarjeta de credito.

Para ello se implementan dos enfoques:

- **No MLOps:** entrenamiento, validacion y almacenamiento del modelo mediante procesos manuales.
- **MLOps:** automatizacion del pipeline, versionado de datos, registro de metricas, validacion automatica, registro de modelos, API de inferencia y contenerizacion con Docker.

El estudio permite comparar ambos enfoques y evaluar las mejoras que aporta MLOps en terminos de trazabilidad, reproducibilidad, mantenimiento, control de calidad y despliegue.

---

## Problematica

Las entidades financieras procesan miles de transacciones diariamente. Identificar transacciones fraudulentas de forma rapida y precisa es fundamental para reducir perdidas economicas y mejorar la seguridad de los clientes.

El modelo desarrollado clasifica cada transaccion como:

- **Clase 0:** Transaccion legitima.
- **Clase 1:** Transaccion fraudulenta.

Debido a que los casos de fraude son muy pocos en comparacion con las transacciones legitimas, el problema presenta un fuerte desbalance de clases. Por ello, no basta con evaluar el modelo usando solo accuracy; se utilizan metricas mas adecuadas como precision, recall, F1-score, ROC-AUC y PR-AUC.

---

## Dataset Utilizado

**Credit Card Fraud Detection Dataset**

Fuente:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

### Caracteristicas principales

- 284,807 transacciones originales.
- 492 casos de fraude en el dataset original.
- Dataset altamente desbalanceado.
- Variables `V1` a `V28` anonimizadas mediante PCA.
- Variables adicionales: `Time`, `Amount`.
- Variable objetivo: `Class`.

En el pipeline MLOps, luego de eliminar duplicados, el dataset quedo con:

- 283,726 registros.
- 1,081 duplicados eliminados.
- 0 valores nulos.
- 473 casos de fraude despues de la limpieza.

---

## Modelos Evaluados

### Logistic Regression

```python
LogisticRegression(
    max_iter=1000,
    class_weight={0: 1, 1: 10},
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

## Metricas de Evaluacion

Debido al fuerte desbalance de clases, se emplearon metricas especializadas:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Matriz de confusion

Ademas, se implemento una busqueda automatica del umbral optimo utilizando la curva Precision-Recall para maximizar el F1-Score bajo una restriccion minima de precision.

---

## Resultado del Mejor Modelo

El pipeline MLOps selecciono como mejor modelo:

```text
HistGradientBoosting + Early Stopping
```

Metricas obtenidas:

```text
ROC-AUC:   0.9596
PR-AUC:    0.7935
Precision: 0.9577
Recall:    0.7158
F1-score:  0.8193
Threshold: 0.8517
```

Matriz de confusion:

```text
[[56648,     3],
 [   27,    68]]
```

Interpretacion:

- El modelo logro una precision alta para la clase fraude.
- El recall indica que detecto una proporcion importante de transacciones fraudulentas.
- El uso de PR-AUC es relevante porque el dataset esta fuertemente desbalanceado.
- El threshold seleccionado permite controlar el balance entre falsos positivos y falsos negativos.

---

## Arquitectura MLOps Implementada

El flujo MLOps implementado se organiza de la siguiente forma:

```text
Datos -> DVC -> Preprocesamiento -> Entrenamiento -> Validacion -> Registro MLflow -> API FastAPI -> Docker
```

Componentes principales:

- **Git:** versionado del codigo fuente y archivos de configuracion.
- **DVC:** versionado del dataset y automatizacion del pipeline reproducible.
- **MLflow Tracking:** registro de experimentos, parametros, metricas y artefactos.
- **MLflow Model Registry:** registro de versiones del modelo entrenado.
- **FastAPI:** exposicion del modelo como servicio REST.
- **Docker:** empaquetado de la API y sus dependencias.
- **GitHub Actions:** automatizacion del pipeline CI/CD.

---

## Estructura del Proyecto

```text
Proyecto_MLOps/
|-- api/
|   |-- main.py
|-- artifacts/
|   |-- data_validation.json
|   |-- metrics.json
|   |-- model_info.json
|-- data/
|   |-- raw/
|   |   |-- creditcard.csv.dvc
|   |-- processed/
|-- models/
|   |-- best_fraud_model.pkl
|-- src/
|   |-- features.py
|   |-- train.py
|   |-- validate.py
|   |-- register_model.py
|-- .dvc/
|-- .github/
|   |-- workflows/
|       |-- mlops.yml
|-- Dockerfile
|-- dvc.yaml
|-- dvc.lock
|-- requirements.txt
|-- README.md
|-- README_MLOPS.md
```

---

## Ejecucion del Entorno

### Crear y activar entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Enfoque No MLOps

El enfoque tradicional ejecuta el entrenamiento de forma manual y guarda el modelo como un archivo local.

```powershell
python baseline_no_mlops.py
```

Salida esperada:

```text
models/best_fraud_model_no_mlops.pkl
```

En este enfoque, las metricas se visualizan principalmente por consola y no existe un control completo de versiones de datos, experimentos ni modelos.

---

## Enfoque MLOps

### 1. Inicializar Git y DVC

```powershell
git init
dvc init
```

### 2. Versionar el dataset con DVC

El archivo CSV real no se sube directamente a Git. Se versiona con DVC:

```powershell
dvc add data/raw/creditcard.csv
git add data/raw/creditcard.csv.dvc data/raw/.gitignore .dvc/config
git commit -m "Versionar dataset crudo con DVC"
```

---

## Pipeline DVC

El pipeline se define en `dvc.yaml` y se ejecuta con:

```powershell
dvc repro
```

Etapas implementadas:

```text
preprocess -> train -> validate -> register
```

### Etapa 1: Preprocesamiento

Archivo principal:

```text
src/features.py
```

Funciones principales:

- Carga del dataset `creditcard.csv`.
- Eliminacion de duplicados.
- Validacion de valores nulos.
- Creacion de variables `Amount_log` y `Hour`.
- Division en train, validation y test.
- Generacion del archivo `artifacts/data_validation.json`.

### Etapa 2: Entrenamiento

Archivo principal:

```text
src/train.py
```

Funciones principales:

- Entrenamiento de modelos candidatos.
- Comparacion de resultados.
- Seleccion del mejor modelo por F1-score.
- Registro de parametros y metricas en MLflow.
- Guardado del modelo en `models/best_fraud_model.pkl`.
- Generacion de `artifacts/metrics.json` y `artifacts/model_info.json`.

### Etapa 3: Validacion automatica

Archivo principal:

```text
src/validate.py
```

Funciones principales:

- Lectura de `artifacts/metrics.json`.
- Comparacion contra umbrales minimos.
- Bloqueo del flujo si el modelo no cumple la calidad requerida.

Resultado obtenido:

```text
pr_auc: OK
precision: OK
recall: OK
f1: OK
El modelo supera los umbrales definidos. Puede promoverse.
```

### Etapa 4: Registro del modelo

Archivo principal:

```text
src/register_model.py
```

Funcion principal:

- Registrar el mejor modelo en MLflow Model Registry bajo el nombre:

```text
creditcard-fraud-model
```

Comando de registro manual:

```powershell
python src\register_model.py --model models\best_fraud_model.pkl --test data\processed\test.csv --target Class --name creditcard-fraud-model
```

---

## MLflow Tracking

Para abrir la interfaz de MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Luego ingresar a:

```text
http://127.0.0.1:5000
```

En la seccion **Experiments** se observa el experimento:

```text
fraude-mlops
```

Dentro del experimento se pueden visualizar:

- Parametros.
- Metricas.
- Artefactos.
- Ejecuciones de entrenamiento.
- Comparacion entre modelos.

Metricas registradas:

```text
roc_auc
pr_auc
precision
recall
f1
threshold
```

---

## MLflow Model Registry

El modelo fue registrado en MLflow Model Registry como:

```text
creditcard-fraud-model
```

En la interfaz de MLflow se puede ingresar a:

```text
http://127.0.0.1:5000/#/models
```

Alli se visualiza la seccion **Registered Models**, donde aparece el modelo registrado y sus versiones.

En la version actual de MLflow, la promocion del modelo puede gestionarse mediante aliases. Por ejemplo:

```text
production
champion
candidate
```

Esto permite identificar que version del modelo se encuentra aprobada para uso productivo.

---

## API con FastAPI

La API permite consumir el modelo entrenado mediante un servicio REST.

Archivo principal:

```text
api/main.py
```

Ejecutar localmente:

```powershell
uvicorn api.main:app --reload
```

Luego ingresar a:

```text
http://127.0.0.1:8000/docs
```

Endpoint principal:

```text
POST /predict
```

Ejemplo de entrada:

```json
{
  "data": {
    "Time": 0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.5516,
    "V12": -0.617801,
    "V13": -0.99139,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
  }
}
```

Respuesta esperada:

```json
{
  "prediction": 0,
  "fraud_probability": 0.0123,
  "threshold": 0.8517
}
```

---

## Docker

Construir imagen:

```powershell
docker build --no-cache -t fraude-api .
```

Ejecutar contenedor:

```powershell
docker run -p 8000:8000 fraude-api
```

Abrir documentacion interactiva:

```text
http://127.0.0.1:8000/docs
```

### Nota importante sobre Docker

Antes de construir la imagen, debe existir el modelo:

```text
models/best_fraud_model.pkl
```

Si no existe, se debe ejecutar:

```powershell
dvc repro
```

Como alternativa de prueba, se puede montar la carpeta local de modelos:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}\models:/app/models" fraude-api
```

Para entrega final, el `Dockerfile` debe copiar explicitamente el modelo entrenado dentro de la imagen.

---

## GitHub Actions CI/CD

El proyecto incluye un workflow en:

```text
.github/workflows/mlops.yml
```

Funciones del workflow:

- Descargar el repositorio.
- Configurar Python.
- Instalar dependencias.
- Ejecutar el pipeline DVC.
- Mostrar metricas generadas.

---

---

## Comparacion entre Enfoques

| Caracteristica | No MLOps | MLOps |
|---------------|----------|-------|
| Entrenamiento | Manual | Automatizado |
| Validacion de datos | Manual | Automatica |
| Registro de metricas | Consola | MLflow Tracking |
| Registro de modelos | Manual | MLflow Model Registry |
| Versionado de datos | No | DVC |
| Reproducibilidad | Media | Alta |
| Automatizacion | Baja | Alta |
| Despliegue | Manual | FastAPI + Docker |
| Escalabilidad | Baja | Alta |
| Mantenimiento | Complejo | Simplificado |

---

## Integrantes

- Luis Angel Pumayucra Chauca - 20214035
- Gerson Ademir Oviedo Soto - 20211960
- Yerson Daniel Mamani Valverde - 20211590
- Joaquin Alonso Nuñez Gonzales - 20204706
- Leonardo Miguel Pachas Cleonares - 20211961
- Gabriel Andres Tineo Morales - 20214245
- Diana Sofia Villalva Gomez - 20212903
