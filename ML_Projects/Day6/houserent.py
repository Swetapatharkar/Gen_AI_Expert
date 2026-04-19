import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# #Load Saved Files
# model=pickle.load(open('house_rent_prediction.pkl','rb'))
# scaler=pickle.load(open('scaler.pkl','rb'))
# columns=pickle.load(open('model_columns.pkl','rb'))
# =============================================================================


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("House Rent Prediction In India")

# ------------------ USER INPUTS ------------------

area = st.number_input("Area (sqft)", 100, 5000, 1000)
beds = st.number_input("Bedrooms", 1, 5, 2)
bathrooms = st.number_input("Bathrooms", 1, 5, 2)

furnishing = st.selectbox(
    "Furnishing",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

city = st.selectbox(
    "City",
    ["Pune", "Mumbai", "Delhi", "Bangalore"]
)

if st.button("Predict"):
    input_data = pd.DataFrame({
    'area': [area],
    'beds': [beds],
    'bathrooms': [bathrooms],
    'furnishing': [furnishing],
    'city': [city]
})

input_data = pd.get_dummies(input_data, drop_first=True)
input_data = input_data.reindex(columns=columns, fill_value=0)
input_data = scaler.transform(input_data)


prediction = model.predict(input_data)
prediction = np.exp(prediction)
st.success(f"Predicted Rent: ₹{prediction[0]:,.0f}")

