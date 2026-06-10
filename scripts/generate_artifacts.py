"""Script to generate all project artifacts: preprocessing + model training + plots."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import joblib
import json
import warnings
import os

warnings.filterwarnings("ignore")

# ── 1. Preprocessing ──────────────────────────────────────────────────────────
print("=== PASO 1: Preprocesamiento ===")
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Shape original: {df.shape}")

df.drop(columns=["customerID"], inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
print(f"Shape limpio: {df.shape}")

df_enc = df.copy()
label_encoders = {}
for col in df_enc.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    label_encoders[col] = le

X = df_enc.drop(columns=["Churn"])
y = df_enc["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"Train: {X_train_s.shape} | Test: {X_test_s.shape}")

os.makedirs("artifacts", exist_ok=True)
joblib.dump((X_train_s, X_test_s, y_train, y_test), "artifacts/processed_data.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(label_encoders, "artifacts/label_encoders.pkl")
joblib.dump(X.columns.tolist(), "artifacts/feature_names.pkl")
df_enc.to_csv("artifacts/churn_clean_encoded.csv", index=False)
print("Artefactos de preprocesamiento guardados.")

# ── 2. EDA plots ──────────────────────────────────────────────────────────────
print("\n=== PASO 2: Graficas EDA ===")
plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
churn_counts = df["Churn"].value_counts()
colors = ["#2ecc71", "#e74c3c"]
axes[0].bar(churn_counts.index, churn_counts.values, color=colors)
axes[0].set_title("Distribucion de Churn")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")
axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct="%1.1f%%", colors=colors)
axes[1].set_title("Proporcion de Churn")
plt.tight_layout()
plt.savefig("artifacts/churn_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

cat_cols = ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity"]
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for i, col in enumerate(cat_cols):
    churn_rate = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    churn_rate.sort_values(ascending=False).plot(kind="bar", ax=axes[i], color="#e74c3c", alpha=0.8)
    axes[i].set_title(f"Churn %\n{col}", fontweight="bold", fontsize=9)
    axes[i].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("artifacts/categorical_churn_rates.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(14, 10))
corr = df_enc.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            linewidths=0.5, annot_kws={"size": 7}, ax=ax)
ax.set_title("Matriz de Correlacion", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("artifacts/correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Graficas EDA guardadas.")

# ── 3. Model training ─────────────────────────────────────────────────────────
print("\n=== PASO 3: Entrenamiento de modelos ===")
models = {
    "Regresion Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "Arbol de Decision": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
results = {}
trained = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    results[name] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "AUC-ROC": round(roc_auc_score(y_test, y_proba), 4),
    }
    trained[name] = (model, y_pred, y_proba)
    r = results[name]
    print(f"  {name}: AUC={r['AUC-ROC']} F1={r['F1-Score']}")

results_df = pd.DataFrame(results).T

# ── 4. Model plots ────────────────────────────────────────────────────────────
print("\n=== PASO 4: Graficas de modelos ===")
bar_colors = ["#3498db", "#e74c3c", "#2ecc71"]
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(results_df.columns))
width = 0.25
for i, (mn, color) in enumerate(zip(results_df.index, bar_colors)):
    bars = ax.bar(x + i * width, results_df.loc[mn].values, width, label=mn, color=color, alpha=0.85)
    for bar, val in zip(bars, results_df.loc[mn].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(results_df.columns)
ax.set_ylim(0, 1.1)
ax.legend()
ax.set_title("Comparacion de Metricas por Modelo", fontweight="bold")
plt.tight_layout()
plt.savefig("artifacts/metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
for (name, (model, y_pred, y_proba)), color in zip(trained.items(), bar_colors):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.5)
ax.legend(loc="lower right")
ax.set_title("Curvas ROC", fontweight="bold")
plt.tight_layout()
plt.savefig("artifacts/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

best_model, y_pred_best, y_proba_best = trained["Random Forest"]

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best, display_labels=["No Churn", "Churn"], cmap="Blues", ax=ax
)
ax.set_title("Matriz de Confusion - Random Forest", fontweight="bold")
plt.tight_layout()
plt.savefig("artifacts/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

feat_imp = pd.Series(best_model.feature_importances_, index=X.columns.tolist()).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 7))
feat_imp.plot(kind="barh", ax=ax, color="#e74c3c", alpha=0.8)
ax.set_title("Importancia de Features - Random Forest", fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("artifacts/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5. Save models and metrics ────────────────────────────────────────────────
joblib.dump(best_model, "artifacts/random_forest_model.pkl")
with open("artifacts/model_metrics.json", "w") as f:
    json.dump({"best_model": "Random Forest", "all_results": results}, f, indent=2)

print("\n=== ARTEFACTOS GENERADOS ===")
for fname in sorted(os.listdir("artifacts")):
    size = os.path.getsize(f"artifacts/{fname}")
    print(f"  {fname} ({size/1024:.1f} KB)")

print("\n=== METRICAS FINALES ===")
print(results_df.to_string())
rf = results["Random Forest"]
print(f"\nBest Model (Random Forest):")
for k, v in rf.items():
    print(f"  {k}: {v}")
