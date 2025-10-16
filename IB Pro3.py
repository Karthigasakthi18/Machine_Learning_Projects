# app.py - Streamlit Dashboard for Sales Forecast
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model and features
model = joblib.load('sales_model.pkl')
features = joblib.load('model_features.pkl')

st.title("📊 Store Sales Forecast Dashboard")
st.write("Predict store sales using multiple regression model")

# Sidebar inputs
st.sidebar.header("Enter Store & Date Details")

store_cluster = st.sidebar.number_input("Store Cluster", min_value=1, max_value=17, value=5)
onpromotion = st.sidebar.number_input("Number of Promotions", min_value=0, max_value=1000, value=10)
is_holiday = st.sidebar.selectbox("Is it a holiday?", [0, 1])
year = st.sidebar.selectbox("Year", [2013, 2014, 2015, 2016, 2017])
month = st.sidebar.slider("Month", 1, 12, 6)
dayofweek = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

# Create input dataframe
input_data = pd.DataFrame([[onpromotion, store_cluster, is_holiday, year, month, dayofweek]],
                          columns=['onpromotion', 'cluster', 'is_holiday', 'year', 'month', 'dayofweek'])

# Add dummy columns if model used type features
for col in features:
    if col not in input_data.columns:
        input_data[col] = 0  # default 0

# Reorder columns
input_data = input_data[features]

# Predict
if st.sidebar.button("Predict Sales"):
    predicted_sales = model.predict(input_data)[0]
    st.success(f"💰 Predicted Sales: ${predicted_sales:,.2f}")

    # Simple visualization (demo)
    actual = predicted_sales * np.random.uniform(0.8, 1.2)
    fig, ax = plt.subplots()
    ax.bar(['Predicted', 'Actual'], [predicted_sales, actual], color=['orange', 'blue'])
    ax.set_ylabel('Sales')
    ax.set_title('Actual vs Predicted Sales')
    st.pyplot(fig)
