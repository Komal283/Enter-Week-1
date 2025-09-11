import streamlit as st
import os
import joblib

# --- Ensure 'models' folder exists ---
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Debug: list files in current directory ---
st.write("Files in current directory:", os.listdir())
st.write("Files in models folder:", os.listdir(MODEL_DIR))

# --- Function to load model safely ---
def load_model(filename):
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        st.success(f"{filename} loaded successfully!")
        return model
    else:
        st.warning(f"{filename} not found! Please upload it below.")
        uploaded_file = st.file_uploader(f"Upload {filename}", type="pkl")
        if uploaded_file is not None:
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"{filename} uploaded and saved!")
            model = joblib.load(filepath)
            return model
        return None

# --- Load models ---
clf_tuple = load_model("wildfire_classifier.pkl")
if clf_tuple is not None:
    clf, scaler_clf, features_clf = clf_tuple
else:
    clf = scaler_clf = features_clf = None

reg_tuple = load_model("wildfire_brightness_model.pkl")
if reg_tuple is not None:
    reg, scaler_reg, features_reg = reg_tuple
else:
    reg = scaler_reg = features_reg = None

# --- Example usage ---
if clf:
    st.write("Classifier is ready to use!")
if reg:
    st.write("Brightness regression is ready to use!")