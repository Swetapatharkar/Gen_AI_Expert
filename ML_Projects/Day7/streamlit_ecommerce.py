import streamlit as st
import pandas as pd
import joblib

# Load models
clf_pipeline = joblib.load("clf_model.pkl")
reg_pipeline = joblib.load("reg_model.pkl")

st.title("🛒 E-commerce Purchase Prediction (Hybrid ML)")

# Inputs
time_on_site = st.slider("Time on Site (minutes)", 1, 60, 10)
pages_viewed = st.slider("Pages Viewed", 1, 30, 5)
past_purchases = st.slider("Past Purchases", 0, 20, 2)
device = st.selectbox("Device", ["mobile", "desktop"])
discount = st.slider("Discount (%)", 0, 50, 10)

threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5)

# Input dataframe
input_df = pd.DataFrame([{
    'time_on_site': time_on_site,
    'pages_viewed': pages_viewed,
    'past_purchases': past_purchases,
    'device': device,
    'discount': discount
}])

# Prediction
if st.button("Predict"):
    prob = clf_pipeline.predict_proba(input_df)[0][1]

    if prob >= threshold:
        amount = reg_pipeline.predict(input_df)[0]
        st.success(f"✅ Will Purchase\nProbability: {round(prob,2)}\nAmount: ₹{round(amount,2)}")
    else:
        st.error(f"❌ Will NOT Purchase\nProbability: {round(prob,2)}")