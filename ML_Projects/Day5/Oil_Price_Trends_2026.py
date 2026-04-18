import streamlit as st
import pickle
import pandas as pd

# Load saved files
model = pickle.load(open('oil_price_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('model_columns.pkl', 'rb'))

st.title("Oil Price Prediction App")

# ------------------ USER INPUTS ------------------

conflict = st.number_input("Conflict Intensity", min_value=0, max_value=3, value=1)

region = st.selectbox(
    "Region",
    ["Asia", "Europe", "Middle East", "Africa", "America"]
)

supply = st.slider("Supply Shock", 0, 1, 10)
demand = st.slider("Demand Index", 50, 100, 75)
year = st.number_input("Year", 2000, 2030, 2026)
month = st.slider("Month", 1, 12, 4)

# ------------------ PREDICTION ------------------

if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        'Conflict_Intensity': [conflict],
        'Region': [region],
        'Supply_Shock': [supply],
        'Demand_Index': [demand],
        'Year': [year],
        'Month': [month]
    })

    # Apply same encoding as training
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Align columns with training data
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Scale input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)

    # Show result
    st.success(f"Predicted Oil Price: ${prediction[0]:.2f}")
