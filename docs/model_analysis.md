# Análisis de Resultados del Modelo

## Comparación de Algoritmos

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Regresión Logística ⭐ | 0.7939 | 0.6243 | 0.5642 | 0.5927 | **0.8345** |
| Árbol de Decisión | 0.7818 | 0.6091 | 0.5000 | 0.5492 | 0.8186 |
| Random Forest | 0.7818 | 0.6151 | 0.4786 | 0.5383 | 0.8107 |

## Por qué ganó la Regresión Logística

Contra lo esperado, el modelo más simple superó al Random Forest. Esto ocurre por dos razones:

1. **Relaciones lineales dominantes:** Variables como `Contract`, `tenure` y `MonthlyCharges` tienen una relación casi lineal con el churn. El escalado con StandardScaler potencia esta linealidad.

2. **Tamaño del dataset:** Con ~5,600 muestras de entrenamiento, los árboles de decisión tienen mayor tendencia al overfitting comparado con un modelo regularizado.

## Variables más importantes (Random Forest - Gini)

| Feature | Importancia |
|---------|-------------|
| tenure | ~0.18 |
| MonthlyCharges | ~0.17 |
| TotalCharges | ~0.16 |
| Contract | ~0.08 |
| OnlineSecurity | ~0.05 |

## Interpretación de negocio

- Un cliente con **menos de 12 meses** de antigüedad tiene 3x más riesgo de churn
- Los contratos **mes a mes** tienen tasa de churn del ~43% vs ~3% en contratos bianuales
- Clientes con **cargo mensual > $70** y sin soporte técnico son el segmento de mayor riesgo

## Propuesta de mejora (v2.0)

- Aplicar **SMOTE** para balancear clases y mejorar Recall
- Evaluar **XGBoost / LightGBM** con hiperparámetros optimizados
- Ajustar el **threshold de decisión** de 0.5 a ~0.35 para maximizar Recall
