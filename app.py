import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="GridGuard AI", layout="wide")

st.title("⚡ GridGuard AI – Transformer Overload Prediction Dashboard")

model = joblib.load("gridguard_transformer_model.pkl")

expected_columns = [
    'Time (hour)',
    'Current_Load_MW',
    'Temperature_C',
    'Consumers',
    'Area_Type_Industrial',
    'Area_Type_Residential'
]

st.sidebar.header("🔧 Transformer Inputs")

time_hour = st.sidebar.slider("Time (Hour)", 0, 23, 15)
current_load = st.sidebar.number_input("Current Load (MW)", 0.0, 30.0, 18.5)
temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 34)
area = st.sidebar.selectbox(
    "Area Type",
    ["Residential", "Commercial", "Industrial"]
)
consumers = st.sidebar.number_input("Consumers", 10, 5000, 850)

input_df = pd.DataFrame({
    'Time (hour)': [time_hour],
    'Current_Load_MW': [current_load],
    'Temperature_C': [temperature],
    'Area_Type': [area],
    'Consumers': [consumers]
})

input_encoded = pd.get_dummies(input_df, columns=['Area_Type'], drop_first=True)
input_processed = input_encoded.reindex(columns=expected_columns, fill_value=0)

if st.button("⚡ Predict Transformer Status"):

    predicted_load = model.predict(input_processed)[0]

    transformer_capacity = 20
    utilization = (predicted_load / transformer_capacity) * 100

    if utilization < 80:
        risk = "NORMAL ✅"
        color = "green"
    elif utilization < 100:
        risk = "WARNING ⚠️"
        color = "orange"
    else:
        risk = "CRITICAL 🚨"
        color = "red"

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Load (Next Hour)", f"{predicted_load:.2f} MW")
    col2.metric("Utilization", f"{utilization:.1f}%")
    col3.markdown(f"### Risk: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)

    st.divider()

    st.subheader("📊 Smart Grid Visual Analytics")

    np.random.seed(42)
    temp_data = np.random.uniform(15, 40, 200)
    load_data = 0.5 * temp_data + np.random.normal(5, 2, 200)

    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=temp_data, y=load_data, ax=ax1)
    ax1.set_title("Load vs Temperature Relationship")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Load (MW)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(["Utilization"], [utilization])
    ax2.axhline(80, linestyle="--")
    ax2.axhline(100, linestyle="--")
    ax2.set_ylim(0, 120)
    ax2.set_title("Transformer Utilization Level")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.bar(["Current Load", "Predicted Load"],
            [current_load, predicted_load])
    ax3.set_title("Current vs Predicted Load")
    st.pyplot(fig3)

st.markdown("---")
st.caption("GridGuard AI | Smart Grid Intelligence System")