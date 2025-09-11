import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wildfire Detection & Brightness Prediction", layout="wide")
st.title("🔥 Wildfire Detection & Brightness Prediction")

# --- Load Models ---
clf, scaler_clf, features_clf = joblib.load("wildfire_classifier.pkl")
reg, scaler_reg, features_reg = joblib.load("wildfire_brightness_model.pkl")

# --- Sidebar Menu ---
task = st.sidebar.selectbox("Select Task", ["Fire Classification", "Brightness Prediction"])

# --- Input Fields Function ---
def get_user_input():
    latitude = st.number_input("Latitude", value=20.0)
    longitude = st.number_input("Longitude", value=78.0)
    scan = st.number_input("Scan", value=1.0)
    track = st.number_input("Track", value=1.0)
    confidence = st.slider("Confidence", 0, 100, 50)
    frp = st.number_input("Fire Radiative Power (FRP)", value=10.0)
    bright_t31 = st.number_input("Brightness T31", value=300.0)

    data = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "scan": scan,
        "track": track,
        "confidence": confidence,
        "frp": frp,
        "bright_t31": bright_t31
    }])
    return data

# --- Fire Classification ---
if task == "Fire Classification":
    st.subheader("🌲 Fire vs No Fire Classification")
    user_input = get_user_input()

    # Scale & predict
    user_scaled = scaler_clf.transform(user_input[features_clf])
    prediction = clf.predict(user_scaled)

    if st.button("Predict Fire"):
        st.success("🔥 FIRE DETECTED!" if prediction[0] == 1 else "✅ No Fire")

# --- Brightness Prediction ---
elif task == "Brightness Prediction":
    st.subheader("☀ Predict Wildfire Brightness")
    user_input = get_user_input()

    # Scale