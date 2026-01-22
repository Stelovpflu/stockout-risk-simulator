
📦 **Stockout Risk Simulator – Retail Supply Chain**

🔍 Business Problem

En retail, los quiebres de stock (stockouts) generan pérdidas directas de ventas, una mala experiencia del cliente e ineficiencias en la planificación y el inventario.

A través de esta aplicación interactiva construida con Streamlit podra estimar el riesgo de quiebre de stock (stockout) en entornos retail, utilizando un modelo de Machine Learning (XGBoost) entrenado sobre datos reales de operaciones de inventario, ventas, clima y calendario.

La app permite simular escenarios operativos ajustando variables clave como demanda, descuentos, clima, producto y tienda y observar cómo cambia la probabilidad de stockout, apoyando decisiones de planificación de inventario y reposición.

---
🎯 **Business Decision Simulated**

¿Qué combinaciones de demanda, clima, calendario y producto incrementan el riesgo de stockout, y cuándo debo anticiparme?

El output del modelo no es solo una clase, sino una probabilidad de stockout, convertida en acción mediante un threshold operativo optimizado para recall.

---
🚀 Simulador 👉 Streamlit Cloud App

https://sl-stockout-risk-simulator.streamlit.app/

---
🧠 **Dataset**

Fuente: HuggingFace

Nombre: Dingdong-Inc/FreshRetailNet-50K

Tipo: datos reales de retail (ventas, stock, clima, calendario)

Tamaño:

Entrenamiento(Train): ~4.5M registros

Evaluación(Eval): ~350K registros

El target NO viene dado, se construye a partir de información operacional real de stock.

---
🛠️ **Feature Engineering**

🎯 **Target**

stockout = 1  → riesgo de quiebre de stock

stockout = 0  → stock suficiente


Construido a partir de:

stock_hour6_22_cnt (conteo real de stock disponible)

---
🤖 **Model**

Algoritmo: XGBoost (Gradient Boosted Trees)

---
📊 **Performance (Eval Set)**

Métrica	Valor
ROC-AUC	0.75
PR-AUC	0.72
Accuracy	0.62

🎯 **Threshold Operativo**

Threshold = 0.325


Optimizado para recall ≈ 0.87 en stockouts
(priorizando evitar quiebres, incluso con más falsos positivos)

---
🚀 **Streamlit App – Stockout Risk Simulator**

La app permite ajustar variables clave (demanda, clima, calendario), seleccionar producto, tienda y categorías y obtener probabilidad de stockout, decisión operativa (stockout / ok).

👉 Pensada para planners, supply chain managers y demos comerciales.

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
⚠️ Notas
Este repositorio contiene **solo el código de inferencia**.
El entrenamiento del modelo se realizó por separado.

---
👤 Autor
**Steve Loveday**  
Data Scientist | Business Analytics | Machine Learning



