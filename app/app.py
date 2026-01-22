# ------------------------------
# streamlit_app.py
# ------------------------------

import streamlit as st
import pandas as pd
import joblib

# ---------- Config ----------
st.set_page_config(
    page_title="Stockout Risk Simulator",
    layout="wide"
)

# ---------- Cargar pipeline ----------
pipe = joblib.load("modeling/xgb_stockout_pipeline.pkl")

THRESHOLD = 0.325

# ---------- Título ----------
st.title("📦 Stockout Risk Simulator")
st.markdown(
    "Simula escenarios *what-if* para estimar **riesgo de quiebre de stock** "
    "en retail usando un modelo XGBoost entrenado en datos reales."
)

# ---------- Sidebar ----------
st.sidebar.header("🔧 Ajusta las variables")

# --- Numéricas ---
sale_amount = st.sidebar.slider("Sale Amount", 0.0, 50.0, 1.0)
discount = st.sidebar.slider("Discount", 0.0, 1.1, 0.9)
precpt = st.sidebar.slider("Precipitation", 0.0, 20.0, 0.0)
avg_temperature = st.sidebar.slider("Temperature (°C)", 10.0, 35.0, 22.0)
avg_humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 75.0)
avg_wind_level = st.sidebar.slider("Wind Level", 0.5, 4.0, 1.5)
day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 5)
dayofweek = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 3)

# --- Categóricas ---
city_id = st.sidebar.selectbox("City ID", list(range(1, 16)))
store_id = st.sidebar.selectbox("Store ID", list(range(1, 1000)))
management_group_id = st.sidebar.selectbox("Management Group", [1, 2, 3, 4, 5, 6])
first_category_id = st.sidebar.selectbox("First Category", list(range(0, 30)))
second_category_id = st.sidebar.selectbox("Second Category", list(range(0, 80)))
third_category_id = st.sidebar.selectbox("Third Category", list(range(0, 250)))
product_id = st.sidebar.selectbox("Product ID", list(range(0, 800)))

# ---------- Input dataframe ----------
input_df = pd.DataFrame([{
    "sale_amount": sale_amount,
    "discount": discount,
    "precpt": precpt,
    "avg_temperature": avg_temperature,
    "avg_humidity": avg_humidity,
    "avg_wind_level": avg_wind_level,
    "day": day,
    "month": month,
    "dayofweek": dayofweek,
    "city_id": city_id,
    "store_id": store_id,
    "management_group_id": management_group_id,
    "first_category_id": first_category_id,
    "second_category_id": second_category_id,
    "third_category_id": third_category_id,
    "product_id": product_id
}])

# ---------- Predicción ----------
stockout_prob = pipe.predict_proba(input_df)[:, 1][0]
stockout_pred = int(stockout_prob >= THRESHOLD)

# ---------- Output ----------
st.subheader("📊 Resultado de la simulación")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Probabilidad de Stockout",
        f"{stockout_prob:.2%}"
    )

with col2:
    st.metric(
        "Predicción",
        "STOCKOUT" if stockout_pred == 1 else "STOCK OK"
    )

st.caption(
    f"Threshold operativo utilizado: {THRESHOLD}"
)

