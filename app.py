import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Load the saved model
model = joblib.load("heart_model.pkl")

# Severity label map
severity_map = {
    0: "No Disease",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Life-Threatening"
}

emoji_map = {
    0: "✅",
    1: "🟡",
    2: "🟠",
    3: "🔴",
    4: "⚠️"
}

# Page config
st.set_page_config(
    page_title="Heart Disease Severity Prediction",
    layout="centered"
)

# Custom CSS for black text and yellow background
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background-color: #fffcc9 !important;
            color: black !important;
        }

        .stApp {
            background-color: #f7ec54 !important;
        }
        h1 {
            color: black !important;
        }
        .stButton>button {
            background-color: #ffcc00 !important;
            color: black !important;
        }
        .stMarkdown, .stNumberInput, .stSelectbox, .stSlider {
            color: black !important;
        }
        label {
            color: black !important;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💓 Heart Disease Severity Prediction")

# Instruction text (black by default due to CSS)
st.markdown("Please enter the following health details below.")

# Sidebar: Info about inputs
with st.sidebar:
    st.header("ℹ️ Feature Guide")
    st.markdown("""
    - **Age**: Patient's age in years.
    - **Chest Pain Type**:
        - 0 = Typical Angina  
        - 1 = Atypical Angina  
        - 2 = Non-anginal Pain  
        - 3 = Asymptomatic
    - **Resting BP**: Resting blood pressure in mm Hg.
    - **Cholesterol**: Serum cholesterol (mg/dl).
    - **Max HR (Thalach)**: Maximum heart rate achieved.
    - **Oldpeak**: ST depression induced by exercise.
    - **Slope of ST**:
        - 0 = Upsloping  
        - 1 = Flat  
        - 2 = Downsloping
    - **CA**: Number of major vessels (0–3).
    """)

# Inputs (now styled with black text)
age = st.number_input("Age", min_value=1, max_value=120, value=50)

cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)

trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg) [Range: 70–250]",
    min_value=70,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol Level (mg/dL) [Range: 100–600]",
    min_value=100,
    max_value=600,
    value=230
)

thalch = st.number_input(
    "Max Heart Rate Achieved (bpm) [Range: 60–210]",
    min_value=60,
    max_value=210,
    value=150
)

oldpeak = st.number_input(
    "ST Depression (mm) [Range: 0.0–6.0]",
    min_value=0.0,
    max_value=6.0,
    value=1.4,
    step=0.1
)

slope = st.selectbox(
    "Slope of ST Segment [0 = Upsloping, 1 = Flat, 2 = Downsloping]",
    ["Upsloping", "Flat", "Downsloping"]
)
slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)

ca = st.slider(
    "Number of Major Vessels (0–3) [0 = None, 3 = All affected]",
    min_value=0,
    max_value=3,
    value=0
)
# Predict and Display
if st.button("🔍 Predict"):
    user_data = pd.DataFrame([{
        'age': age,
        'cp': cp_val,
        'trestbps': trestbps,
        'chol': chol,
        'thalch': thalch,
        'oldpeak': oldpeak,
        'slope': slope_val,
        'ca': float(ca)
    }])

    prediction = model.predict(user_data)[0]
    proba = model.predict_proba(user_data)[0]

    emoji = emoji_map[prediction]
    severity_label = severity_map[prediction]

    st.markdown(f"### 🧠 Predicted Heart Disease Severity: {emoji} **{severity_label}**")

    st.markdown("### 📊 Prediction Confidence")
    st.bar_chart(pd.DataFrame(proba, index=severity_map.values(), columns=["Probability"]))

    # Generate downloadable report
    report = f"""
Heart Disease Severity Prediction Report

Predicted Severity: {severity_label} {emoji}

Prediction Probabilities:
--------------------------
"""
    for i, prob in enumerate(proba):
        report += f"{severity_map[i]}: {prob:.2f}\n"

    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="heart_disease_prediction.txt",
        mime='text/plain'
    )
