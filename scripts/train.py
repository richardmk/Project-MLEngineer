"""Script de entrenamiento del modelo de churn."""

import argparse
import joblib
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trainer import ModelTrainer


def main(processed_path: str, model_name: str, output_dir: str):
    print(f"Cargando datos procesados desde: {processed_path}")
    X_train, X_test, y_train, y_test = joblib.load(processed_path)

    trainer = ModelTrainer(model_name=model_name)
    print(f"Entrenando modelo: {model_name} ...")
    trainer.fit(X_train, y_train)

    metrics = trainer.evaluate(X_test, y_test)
    print("\nMétricas de evaluación:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs(output_dir, exist_ok=True)
    trainer.save(f"{output_dir}/{model_name}.pkl")
    with open(f"{output_dir}/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModelo y métricas guardados en: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo de clasificación de churn.")
    parser.add_argument("--data", type=str, default="artifacts/processed_data.pkl")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["logistic_regression", "decision_tree", "random_forest"])
    parser.add_argument("--output", type=str, default="artifacts")
    args = parser.parse_args()
    main(args.data, args.model, args.output)
