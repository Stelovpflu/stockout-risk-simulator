# ------------------------------
# XGBOOST FULL DATA - STOCKOUT RISK
# ------------------------------

import gc
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score
)

# ---------- Cargar data preparada ----------
from data_prep.prepare_data import load_prepared_data

df_train, df_eval, features, target = load_prepared_data()

X_train = df_train[features]
y_train = df_train[target]

X_eval = df_eval[features]
y_eval = df_eval[target]

# ---------- Separar features ----------
num_features = [
    "sale_amount",
    "discount",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "day",
    "month",
    "dayofweek"
]

cat_features = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id"
]

# ---------- Preprocessor ----------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), cat_features)
    ]
)

# ---------- Modelo final (hiperparámetros ya definidos) ----------
xgb_final = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

pipe_final = Pipeline([
    ("prep", preprocessor),
    ("model", xgb_final)
])

# ---------- Entrenamiento ----------
print("Training XGBoost on FULL DATA...")
pipe_final.fit(X_train, y_train)
gc.collect()

# ---------- Probabilidades ----------
y_proba = pipe_final.predict_proba(X_eval)[:, 1]

# ---------- Threshold operativo ----------
THRESHOLD = 0.3249
y_pred = (y_proba >= THRESHOLD).astype(int)

# ---------- Métricas ----------
precision, recall, _ = precision_recall_curve(y_eval, y_proba)
pr_auc = auc(recall, precision)
roc_auc = roc_auc_score(y_eval, y_proba)

print("\n===== XGBOOST FULL DATA (STOCKOUT RISK) =====")
print("Threshold operativo:", THRESHOLD)
print("Confusion Matrix:\n", confusion_matrix(y_eval, y_pred))
print(classification_report(y_eval, y_pred, digits=4))
print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))

# ---------- Guardar pipeline ----------
joblib.dump(pipe_final, "xgb_stockout_pipeline.pkl")
print("\nPipeline guardado: xgb_stockout_pipeline.pkl")
