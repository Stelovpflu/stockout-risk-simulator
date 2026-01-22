
📦 **Stockout Risk Simulator – Retail Supply Chain**

🔍 Business Problem

En retail, los quiebres de stock (stockouts) generan pérdida directa de ventas, mala experiencia del cliente e ineficiencias en planificación e inventario.

A traves de esta aplicación interactiva construida con Streamlit para estimar el riesgo de quiebre de stock (stockout) en entornos retail, utilizando un modelo de Machine Learning (XGBoost) entrenado sobre datos reales de operaciones de inventario, ventas, clima y calendario.

La app permite simular escenarios operativos (“what-if”) ajustando variables clave (demanda, descuentos, clima, producto y tienda) y observar cómo cambia la probabilidad de stockout, apoyando decisiones de planificación de inventario y reposición.

---

🎯 **Business Decision Simulated**

¿Qué combinaciones de demanda, clima, calendario y producto incrementan el riesgo de stockout, y cuándo debo anticiparme?

El output del modelo no es solo una clase, sino una:

probabilidad de stockout,

convertida en acción mediante un threshold operativo optimizado para recall.

---

🧠 **Dataset**

Fuente: HuggingFace

Nombre: Dingdong-Inc/FreshRetailNet-50K

Tipo: datos reales de retail (ventas, stock, clima, calendario)

Tamaño:

Train: ~4.5M registros

Eval: ~350K registros

El target NO viene dado: se construye a partir de información operacional real de stock.

---

🛠️ **Feature Engineering**
🎯 Target
stockout = 1  → riesgo de quiebre de stock
stockout = 0  → stock suficiente


Construido a partir de:

stock_hour6_22_cnt (conteo real de stock disponible)

🔢 Numéricas

sale_amount

discount

precpt

avg_temperature

avg_humidity

avg_wind_level

day, month, dayofweek

🏷️ Categóricas

city_id

store_id

management_group_id

first_category_id

second_category_id

third_category_id

product_id

Se elimina leakage directo y columnas no accionables para la app.

---

🤖 **Model**

Algoritmo: XGBoost (Gradient Boosted Trees)

Encoding categórico: OrdinalEncoder (eficiente y deployable)

Pipeline: preprocessing + model

🔧 Hiperparámetros finales
max_depth = 6
learning_rate = 0.1
n_estimators = 200
subsample = 0.8
colsample_bytree = 0.8


Seleccionados mediante comparación contra RF y GB + tuning focalizado.

---

📊 **Performance (Eval Set)**
Métrica	Valor
ROC-AUC	0.75
PR-AUC	0.72
Accuracy	0.62
🎯 Threshold Operativo

Threshold = 0.325

Optimizado para recall ≈ 0.87 en stockouts
(priorizando evitar quiebres, incluso con más falsos positivos)

---

🚀 **Streamlit App – Stockout Risk Simulator**

La app permite:

ajustar variables clave (demanda, clima, calendario),

seleccionar producto, tienda y categorías,

obtener:

probabilidad de stockout

decisión operativa (stockout / ok)

👉 Pensada para:

planners,

supply chain managers,

demos comerciales.

---

📂 **Project Structure**
├─ app/
│   └─ streamlit_app.py
├─ data_prep/
│   ├─ __init__.py
│   └─ prepare_data.py
├─ modeling/
│   └─ xgb_stockout_pipeline.pkl
├─ requirements.txt
└─ README.md

---

▶️ **How to Run**
pip install -r requirements.txt
streamlit run app/streamlit_app.py

---



