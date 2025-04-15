import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Title
st.title("🔋 EV Battery Life Prediction Dashboard")
st.markdown("View model performance for different cell types and models.")
data = {
    "B0005": [
        {"Model": "Random Forest", "R2": 0.91, "RMSE": 27.5, "RUL": [180, 185, 190, 195, 200], "Actual": [182, 187, 192, 194, 198]},
        {"Model": "XGBoost", "R2": 0.93, "RMSE": 24.3, "RUL": [182, 188, 193, 198, 204], "Actual": [182, 187, 192, 194, 198]},
    ],
    "B0006": [
        {"Model": "MLP", "R2": 0.89, "RMSE": 29.1, "RUL": [170, 172, 174, 176, 178], "Actual": [171, 173, 175, 177, 179]},
        {"Model": "LightGBM", "R2": 0.92, "RMSE": 22.8, "RUL": [169, 172, 176, 179, 181], "Actual": [171, 173, 175, 177, 179]},
    ],
    "B0007": [
        {"Model": "AdaBoost", "R2": 0.87, "RMSE": 30.0, "RUL": [160, 162, 165, 168, 170], "Actual": [161, 164, 167, 169, 171]},
    ],
    "B0018": [
        {"Model": "CatBoost", "R2": 0.94, "RMSE": 20.1, "RUL": [200, 205, 210, 215, 220], "Actual": [202, 206, 211, 216, 221]},
    ]
}

cell = st.selectbox("Select Battery Cell", list(data.keys()))
st.subheader(f"Model Results for Cell {cell}")

# Build metrics table
df = pd.DataFrame([{"Model": d["Model"], "R2 Score": d["R2"], "RMSE": d["RMSE"]} for d in data[cell]])
st.dataframe(df.set_index("Model"))

# Plot: Bar chart of R² and RMSE
st.subheader("📊 R² and RMSE Comparison")
fig, ax1 = plt.subplots()
bar_width = 0.35
models = df["Model"]
r2 = df["R2 Score"]
rmse = df["RMSE"]
index = np.arange(len(models))

ax1.bar(index, r2, bar_width, label='R2 Score', color='skyblue')
ax1.bar(index + bar_width, rmse, bar_width, label='RMSE', color='lightgreen')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(models)
ax1.legend()
st.pyplot(fig)

# Line Chart: Predicted vs Actual RUL
st.subheader("📈 Predicted vs Actual RUL")
for entry in data[cell]:
    fig, ax = plt.subplots()
    ax.plot(entry["Actual"], label="Actual RUL", marker='o')
    ax.plot(entry["RUL"], label=f"{entry['Model']} Predicted", marker='x')
    ax.set_title(f"{entry['Model']} - Predicted vs Actual")
    ax.legend()
    st.pyplot(fig)

# Table of RUL Predictions
st.subheader("🧮 Remaining Useful Life Predictions")
for entry in data[cell]:
    rul_df = pd.DataFrame({"Cycle": list(range(1, len(entry["RUL"])+1)), "Predicted RUL": entry["RUL"], "Actual RUL": entry["Actual"]})
    st.markdown(f"#### {entry['Model']}")
    st.table(rul_df)

st.markdown("---")



