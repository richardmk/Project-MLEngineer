"""Script de predicción usando el modelo de churn entrenado."""

import argparse
import joblib
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trainer import ModelTrainer


def main(input_path: str, model_path: str, preprocessor_path: str, output_path: str):
    print(f"Cargando preprocesador desde: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    print(f"Cargando datos desde: {input_path}")
    df = preprocessor.load(input_path)
    df = preprocessor.clean(df)
    df = preprocessor.encode(df)
    X = df.drop(columns=["Churn"], errors="ignore")
    X_scaled = preprocessor.scaler.transform(X)

    print(f"Cargando modelo desde: {model_path}")
    trainer = ModelTrainer()
    trainer.load(model_path)
    predictions = trainer.predict(X_scaled)

    df["predicted_churn"] = predictions
    df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")
    print(f"  Total predicciones: {len(predictions)}")
    print(f"  Churn predicho: {predictions.sum()} ({predictions.mean()*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera predicciones de churn.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, default="artifacts/random_forest.pkl")
    parser.add_argument("--preprocessor", type=str, default="artifacts/preprocessor.pkl")
    parser.add_argument("--output", type=str, default="artifacts/predictions.csv")
    args = parser.parse_args()
    main(args.input, args.model, args.preprocessor, args.output)
