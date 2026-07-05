import argparse
import json
import os
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def find_best_threshold(y_true, y_proba, min_precision: float = 0.75) -> tuple[float, float, float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    for precision, recall, threshold in zip(precisions[:-1], recalls[:-1], thresholds):
        if precision >= min_precision:
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            if f1 > best_f1:
                best_threshold = float(threshold)
                best_precision = float(precision)
                best_recall = float(recall)
                best_f1 = float(f1)

    return best_threshold, best_precision, best_recall, best_f1


def evaluate_model(name: str, model: Any, X_val, y_val, X_test, y_test, min_precision: float) -> dict:
    y_val_proba = model.predict_proba(X_val)[:, 1]
    threshold, val_precision, val_recall, val_f1 = find_best_threshold(
        y_val,
        y_val_proba,
        min_precision=min_precision,
    )

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    report = classification_report(y_test, y_test_pred, digits=4, zero_division=0, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()

    return {
        "name": name,
        "model": model,
        "threshold": threshold,
        "validation_precision": val_precision,
        "validation_recall": val_recall,
        "validation_f1": val_f1,
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "pr_auc": float(average_precision_score(y_test, y_test_proba)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "confusion_matrix": conf_matrix,
        "classification_report": report,
    }


def log_candidate_to_mlflow(result: dict, params: dict, artifact_dir: str) -> None:
    with mlflow.start_run(run_name=result["name"], nested=True):
        for key, value in params.items():
            mlflow.log_param(key, value)

        metric_keys = [
            "threshold",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
        ]
        for key in metric_keys:
            mlflow.log_metric(key, result[key])

        model_dir_name = result["name"].replace(" ", "_").replace("+", "plus")
        mlflow.sklearn.log_model(result["model"], f"model_{model_dir_name}")

        candidate_report = os.path.join(artifact_dir, f"{model_dir_name}_report.json")
        with open(candidate_report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": result["name"],
                    "threshold": result["threshold"],
                    "metrics": {k: result[k] for k in metric_keys},
                    "confusion_matrix": result["confusion_matrix"],
                    "classification_report": result["classification_report"],
                },
                f,
                indent=4,
            )
        mlflow.log_artifact(candidate_report)


def main(
    train_file: str,
    val_file: str,
    test_file: str,
    target_col: str,
    model_out: str,
    metrics_out: str,
    model_info_out: str,
    min_precision: float,
) -> None:
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    os.makedirs(os.path.dirname(model_info_out), exist_ok=True)

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    mlflow.set_experiment("fraude-mlops")

    results = []
    params_by_model = {}

    scale_cols = [col for col in ["Time", "Amount", "Amount_log", "Hour"] if col in X_train.columns]
    preprocessor_lr = ColumnTransformer(
        transformers=[("scale_numeric", StandardScaler(), scale_cols)],
        remainder="passthrough",
    )

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor_lr),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight={0: 1, 1: 10},
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    logistic_model.fit(X_train, y_train)

    logistic_result = evaluate_model(
        "LogisticRegression + Feature Engineering",
        logistic_model,
        X_val,
        y_val,
        X_test,
        y_test,
        min_precision,
    )
    results.append(logistic_result)
    params_by_model[logistic_result["name"]] = {
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "class_weight_0": 1,
        "class_weight_1": 10,
        "solver": "liblinear",
        "min_precision": min_precision,
    }

    sample_weight = compute_sample_weight(class_weight={0: 1, 1: 10}, y=y_train)
    hgb_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )
    hgb_model.fit(X_train, y_train, sample_weight=sample_weight)

    hgb_result = evaluate_model(
        "HistGradientBoosting + Early Stopping",
        hgb_model,
        X_val,
        y_val,
        X_test,
        y_test,
        min_precision,
    )
    results.append(hgb_result)
    params_by_model[hgb_result["name"]] = {
        "model_type": "HistGradientBoostingClassifier",
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.1,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
        "class_weight_0": 1,
        "class_weight_1": 10,
        "min_precision": min_precision,
    }

    with mlflow.start_run(run_name="fraude-training-pipeline"):
        for result in results:
            log_candidate_to_mlflow(result, params_by_model[result["name"]], os.path.dirname(metrics_out))

        best_result = max(results, key=lambda x: x["f1"])

        artifact = {
            "model_name": best_result["name"],
            "model": best_result["model"],
            "threshold": best_result["threshold"],
            "feature_columns": list(X_train.columns),
            "target_column": target_col,
            "metrics": {
                "roc_auc": best_result["roc_auc"],
                "pr_auc": best_result["pr_auc"],
                "precision": best_result["precision"],
                "recall": best_result["recall"],
                "f1": best_result["f1"],
            },
        }

        joblib.dump(artifact, model_out)

        metrics = {
            "best_model": best_result["name"],
            "threshold": best_result["threshold"],
            "roc_auc": best_result["roc_auc"],
            "pr_auc": best_result["pr_auc"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
            "confusion_matrix": best_result["confusion_matrix"],
            "classification_report": best_result["classification_report"],
        }

        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        model_info = {
            "model_path": model_out,
            "model_name": best_result["name"],
            "threshold": best_result["threshold"],
            "feature_columns": list(X_train.columns),
            "target_column": target_col,
        }
        with open(model_info_out, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=4)

        mlflow.log_param("best_model", best_result["name"])
        for key, value in artifact["metrics"].items():
            mlflow.log_metric(f"best_{key}", value)
        mlflow.log_artifact(metrics_out)
        mlflow.log_artifact(model_info_out)
        input_example = X_train.head(5)
        prediction_example = best_result["model"].predict_proba(input_example)[:, 1]
        signature = infer_signature(input_example, prediction_example)

        mlflow.sklearn.log_model(
            sk_model=best_result["model"],
            name="best_model",
            registered_model_name="creditcard-fraud-model",
            signature=signature,
            input_example=input_example
        )

    print(json.dumps(metrics, indent=4))
    print(f"Modelo guardado en: {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", default="Class")
    parser.add_argument("--model-out", default="models/best_fraud_model.pkl")
    parser.add_argument("--metrics-out", default="artifacts/metrics.json")
    parser.add_argument("--model-info-out", default="artifacts/model_info.json")
    parser.add_argument("--min-precision", type=float, default=0.75)
    args = parser.parse_args()

    main(
        args.train,
        args.val,
        args.test,
        args.target,
        args.model_out,
        args.metrics_out,
        args.model_info_out,
        args.min_precision,
    )
