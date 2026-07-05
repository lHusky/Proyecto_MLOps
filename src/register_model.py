import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature


def main(model_path, test_file, target_col, registered_name):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraude-mlops")

    artifact = joblib.load(model_path)

    model = artifact["model"]
    threshold = artifact["threshold"]
    metrics = artifact.get("metrics", {})
    model_name = artifact.get("model_name", "best_model")

    test_df = pd.read_csv(test_file)
    X_test = test_df.drop(columns=[target_col])

    input_example = X_test.head(5)
    prediction_example = model.predict_proba(input_example)[:, 1]

    signature = infer_signature(input_example, prediction_example)

    with mlflow.start_run(run_name="registro-modelo-fraude"):
        mlflow.log_param("registered_model_name", registered_name)
        mlflow.log_param("best_model", model_name)
        mlflow.log_param("threshold", threshold)

        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        mlflow.log_artifact(model_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="best_model",
            registered_model_name=registered_name,
            signature=signature,
            input_example=input_example
        )

    print(f"Modelo registrado correctamente en MLflow: {registered_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", default="Class")
    parser.add_argument("--name", default="creditcard-fraud-model")
    args = parser.parse_args()

    main(args.model, args.test, args.target, args.name)