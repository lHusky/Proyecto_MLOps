import argparse
import json
import sys


def main(metrics_file: str, min_pr_auc: float, min_precision: float, min_recall: float, min_f1: float) -> None:
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    checks = {
        "pr_auc": metrics.get("pr_auc", 0) >= min_pr_auc,
        "precision": metrics.get("precision", 0) >= min_precision,
        "recall": metrics.get("recall", 0) >= min_recall,
        "f1": metrics.get("f1", 0) >= min_f1,
    }

    print("=== Validación automática del modelo ===")
    for metric, passed in checks.items():
        print(f"{metric}: {metrics.get(metric, 0):.4f} -> {'OK' if passed else 'NO CUMPLE'}")

    if not all(checks.values()):
        print("El modelo no supera los umbrales definidos. No debe promoverse a producción.")
        sys.exit(1)

    print("El modelo supera los umbrales definidos. Puede promoverse.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts/metrics.json")
    parser.add_argument("--min-pr-auc", type=float, default=0.70)
    parser.add_argument("--min-precision", type=float, default=0.75)
    parser.add_argument("--min-recall", type=float, default=0.60)
    parser.add_argument("--min-f1", type=float, default=0.65)
    args = parser.parse_args()

    main(args.metrics, args.min_pr_auc, args.min_precision, args.min_recall, args.min_f1)
