# ------------------------------
# DATA PREPARATION PIPELINE
# ------------------------------
# Fuente real:
# HuggingFace -> Dingdong-Inc/FreshRetailNet-50K
#
# Este script:
# - Descarga el dataset
# - Crea el target stockout
# - Elimina leakage
# - Genera features finales
# - Devuelve data lista para modelado o app
# ------------------------------

import pandas as pd
from datasets import load_dataset


def load_prepared_data():
    """
    Carga y prepara el dataset FreshRetailNet-50K
    Retorna:
        df_train, df_eval, features, target
    """

    # ---------- Descargar dataset ----------
    ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")

    df_train = ds["train"].to_pandas()
    df_eval  = ds["eval"].to_pandas()

    # ---------- Crear target stockout ----------
    # 1 = stock disponible
    # 0 = stockout (agotado)
    df_train["stockout"] = (df_train["stock_hour6_22_cnt"] > 0).astype(int)
    df_eval["stockout"]  = (df_eval["stock_hour6_22_cnt"] > 0).astype(int)

    # ---------- Convertir fecha y crear features de calendario ----------
    for df in [df_train, df_eval]:
        df["dt"] = pd.to_datetime(df["dt"])
        df["day"] = df["dt"].dt.day
        df["month"] = df["dt"].dt.month
        df["dayofweek"] = df["dt"].dt.dayofweek

    # ---------- Eliminar leakage directo ----------
    leakage_cols = [
        "stock_hour6_22_cnt",
        "hours_stock_status"
    ]

    # ---------- Eliminar columnas no accionables ----------
    drop_cols = [
        "dt",
        "hours_sale"
    ]

    df_train = df_train.drop(columns=leakage_cols + drop_cols)
    df_eval  = df_eval.drop(columns=leakage_cols + drop_cols)

    # ---------- Definir features ----------
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

    features = num_features + cat_features
    target = "stockout"

    return df_train, df_eval, features, target


# ---------- Debug rápido ----------
if __name__ == "__main__":
    df_train, df_eval, features, target = load_prepared_data()
    print("Data preparada correctamente")
    print("Train shape:", df_train.shape)
    print("Eval shape:", df_eval.shape)
    print("Target:", target)
    print("N° Features:", len(features))
