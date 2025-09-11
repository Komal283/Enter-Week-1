import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
    return df

# ---------------------------
# Train models
# ---------------------------
def train_models(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical values
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train RandomForest and Linear Regression
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Evaluate
    rf_pred = rf.predict(X_test)
    lr_pred = lr.predict(X_test)

    results = {
        "RandomForest_R2": r2_score(y_test, rf_pred),
        "LinearRegression_R2": r2_score(y_test, lr_pred),
    }

    return rf, lr, scaler, results, X.columns

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🔥 Wildfire Detection & Prediction App")
st.write("Upload wildfire dataset and predict fire spread area.")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target_col = st.selectbox("Select target column (what you want to predict)", df.columns)

    if st.button("Train Models"):
        rf_model, lr_model, scaler, results, feature_names = train_models(df, target_col)

        st.subheader("Model Performance")
        st.write(results)

        st.subheader("Make Prediction")
        user_input = {}
        for col in feature_names:
            value = st.number_input(f"Enter value for {col}", value=0.0)
            user_input[col] = value

        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            rf_pred = rf_model.predict(input_scaled)[0]
            lr_pred = lr_model.predict(input_scaled)[0]

            st.success(f"RandomForest Prediction ({target_col}): {rf_pred:.2f}")
            st.success(f"LinearRegression Prediction ({target_col}): {lr_pred:.2f}")
else:
    st.info("Please upload a dataset to begin.")