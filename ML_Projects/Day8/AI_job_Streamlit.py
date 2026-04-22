import streamlit as st
import pandas as pd
import joblib

# Load models
clf_model = joblib.load("clf_model.pkl")
reg_model = joblib.load("reg_model.pkl")

st.title("🤖 AI Job Impact Prediction")

st.write("Hybrid Model: Classification + Salary Prediction")

# Inputs (MATCH TRAINING)
Years_Experience = st.slider("Years of Experience", 0, 20, 3)
Education_Level = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
Industry = st.selectbox("Industry", ["IT", "Finance", "Healthcare"])
Automation_Risk = st.slider("Automation Risk (%)", 0, 100, 50)

threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Create input dataframe
input_df = pd.DataFrame([{
    'Years_Experience': int(Years_Experience),
    'Education_Level': str(Education_Level),
    'Industry': str(Industry),
    'Automation_Risk': str(Automation_Risk) 
}])

# Predict
if st.button("Predict"):

    prob = clf_model.predict_proba(input_df)[0].max()
    class_pred = clf_model.predict(input_df)[0]

    st.subheader(f"📊 Job Status: {class_pred}")
    st.write(f"Confidence: {round(prob,2)}")

    if prob >= threshold:
        reg_pred = reg_model.predict(input_df)[0]
        st.success(f"💰 Predicted Salary Before AI: ₹{round(reg_pred,2)}")
    else:
        st.warning("⚠️ Low confidence → Salary not shown")