import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.title("🏥 LASSO Regression - Healthcare Dataset")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Dataset Preview")
    st.write(dataset.head())

    # Select target column
    target_column = st.selectbox("Select Target Column", dataset.columns)

    # Split features and target
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Alpha slider
    alpha = st.slider("Select Alpha (Regularization Strength)", 0.01, 1.0, 0.1)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    st.subheader("📈 Model Performance")
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))

    # Coefficients
    st.subheader("🔍 Feature Coefficients (LASSO)")
    coefficients = pd.Series(model.coef_, index=X.columns)
    st.write(coefficients)

    # Highlight zero vs non-zero
    st.subheader("✅ Selected vs ❌ Removed Features")
    selected = coefficients[coefficients != 0]
    removed = coefficients[coefficients == 0]

    st.write("Selected Features:", list(selected.index))
    st.write("Removed Features:", list(removed.index))

    # Plot
    st.subheader("📊 Feature Importance Plot")
    fig, ax = plt.subplots()
    coefficients.plot(kind='bar', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)