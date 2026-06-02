import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_sample_weight


DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/best_fraud_model_no_mlops.pkl"


def add_features(df):
    df = df.copy()

    # Monto con transformación logarítmica.
    # Ayuda porque Amount suele tener distribución muy sesgada.
    df["Amount_log"] = np.log1p(df["Amount"])

    # Time está en segundos desde la primera transacción.
    # Convertimos a hora del día aproximada.
    df["Hour"] = (df["Time"] // 3600) % 24

    return df


def clean_data(df):
    df = df.copy()

    # Eliminar duplicados completos.
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"Filas originales: {before}")
    print(f"Filas después de eliminar duplicados: {after}")
    print(f"Duplicados eliminados: {before - after}")

    # Validación básica de nulos.
    nulls = df.isnull().sum().sum()
    print(f"Valores nulos encontrados: {nulls}")

    return df


def find_best_threshold(y_true, y_proba, min_precision=0.75):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    for precision, recall, threshold in zip(precisions[:-1], recalls[:-1], thresholds):
        if precision >= min_precision:
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

            if f1 > best_f1:
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
                best_f1 = f1

    return best_threshold, best_precision, best_recall, best_f1


def evaluate_model(name, model, X_val, y_val, X_test, y_test, min_precision):
    y_val_proba = model.predict_proba(X_val)[:, 1]

    threshold, val_precision, val_recall, val_f1 = find_best_threshold(
        y_val,
        y_val_proba,
        min_precision=min_precision
    )

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print("\n" + "=" * 70)
    print(f"MODELO: {name}")
    print("=" * 70)

    print("\n=== UMBRAL SELECCIONADO EN VALIDACIÓN ===")
    print(f"Threshold: {threshold:.4f}")
    print(f"Precision validación: {val_precision:.4f}")
    print(f"Recall validación: {val_recall:.4f}")
    print(f"F1 validación: {val_f1:.4f}")

    print("\n=== MATRIZ DE CONFUSIÓN TEST ===")
    print(confusion_matrix(y_test, y_test_pred))

    print("\n=== REPORTE DE CLASIFICACIÓN TEST ===")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    print("\n=== MÉTRICAS TEST ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Precision fraude: {precision:.4f}")
    print(f"Recall fraude: {recall:.4f}")
    print(f"F1 fraude: {f1:.4f}")
    print(f"Threshold usado: {threshold:.4f}")

    return {
        "name": name,
        "model": model,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    df = clean_data(df)
    df = add_features(df)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print("\n=== DISTRIBUCIÓN DE CLASES ===")
    print(y.value_counts())
    print(y.value_counts(normalize=True))

    # Train / validation / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp
    )

    results = []

    # ============================================================
    # Modelo 1: Logistic Regression
    # ============================================================

    columns_to_scale = ["Time", "Amount", "Amount_log", "Hour"]

    preprocessor_lr = ColumnTransformer(
        transformers=[
            ("scale_numeric", StandardScaler(), columns_to_scale)
        ],
        remainder="passthrough"
    )

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor_lr),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight={0: 1, 1: 10},
                solver="liblinear",
                random_state=42
            ))
        ]
    )

    logistic_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            name="LogisticRegression + Feature Engineering",
            model=logistic_model,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            min_precision=0.75
        )
    )

    # ============================================================
    # Modelo 2: HistGradientBoosting con Early Stopping
    # ============================================================

    sample_weight = compute_sample_weight(
        class_weight={0: 1, 1: 10},
        y=y_train
    )

    hgb_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    )

    hgb_model.fit(X_train, y_train, sample_weight=sample_weight)

    results.append(
        evaluate_model(
            name="HistGradientBoosting + Early Stopping",
            model=hgb_model,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            min_precision=0.75
        )
    )

    # Seleccionar mejor modelo por F1 de fraude
    best_result = max(results, key=lambda x: x["f1"])

    artifact = {
        "model_name": best_result["name"],
        "model": best_result["model"],
        "threshold": best_result["threshold"],
        "metrics": {
            "roc_auc": best_result["roc_auc"],
            "pr_auc": best_result["pr_auc"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"]
        }
    }

    joblib.dump(artifact, MODEL_PATH)

    print("\n" + "=" * 70)
    print("MEJOR MODELO SELECCIONADO")
    print("=" * 70)
    print(f"Modelo: {best_result['name']}")
    print(f"F1 fraude: {best_result['f1']:.4f}")
    print(f"Precision fraude: {best_result['precision']:.4f}")
    print(f"Recall fraude: {best_result['recall']:.4f}")
    print(f"PR-AUC: {best_result['pr_auc']:.4f}")
    print(f"Threshold: {best_result['threshold']:.4f}")
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()