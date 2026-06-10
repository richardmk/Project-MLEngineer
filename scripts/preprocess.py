"""Script de preprocesamiento del dataset de churn."""

import argparse
import joblib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessor import ChurnPreprocessor


def main(input_path: str, output_dir: str):
    print(f"Cargando datos desde: {input_path}")
    preprocessor = ChurnPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(input_path)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), f"{output_dir}/processed_data.pkl")
    joblib.dump(preprocessor, f"{output_dir}/preprocessor.pkl")
    print(f"Datos procesados guardados en: {output_dir}/")
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesa el dataset de churn.")
    parser.add_argument("--input", type=str, default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output", type=str, default="artifacts")
    args = parser.parse_args()
    main(args.input, args.output)
