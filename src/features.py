import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create fraud-detection features used by the baseline model."""
    df = df.copy()

    if "Amount" in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"])

    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24

    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Basic data quality controls for traceability."""
    rows_before = len(df)
    duplicate_rows = int(df.duplicated().sum())
    null_values = int(df.isnull().sum().sum())

    df = df.drop_duplicates()

    stats = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "duplicates_removed": duplicate_rows,
        "null_values": null_values,
    }
    return df, stats


def main(input_file: str, output_dir: str, target_col: str, stats_out: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(stats_out), exist_ok=True)

    df = pd.read_csv(input_file)

    if target_col not in df.columns:
        raise ValueError(
            f"La columna objetivo '{target_col}' no existe. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df, stats = clean_data(df)
    df = add_features(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    stats.update(
        {
            "target_column": target_col,
            "target_distribution": y.value_counts().to_dict(),
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "feature_columns": list(X.columns),
        }
    )

    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target", default="Class")
    parser.add_argument("--stats-out", default="artifacts/data_validation.json")
    args = parser.parse_args()

    main(args.infile, args.outdir, args.target, args.stats_out)
