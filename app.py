import streamlit as st
import pickle
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Heart Attack Prediction",layout="wide")

# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_models():
    model = pickle.load(open("model.sav", "rb"))
    encoder = pickle.load(open("ohe_smoking.sav", "rb"))
    scaler = pickle.load(open("scaler.sav", "rb"))
    return model, encoder, scaler

model, ohe_smoking, scaler = load_models()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Heart Attack Risk Prediction")
image = Image.open('image.png')    
st.image([image])
st.caption("AI-based cardiovascular risk assessment system")
st.divider()

# --------------------------------------------------
# Input Form
# --------------------------------------------------

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patient Information")
        age = st.number_input("Age",min_value=1,max_value=120,value=30)
        hypertension = st.selectbox("Hypertension",["No", "Yes"])
        diabetes = st.selectbox("Diabetes",["No", "Yes"])
        obesity = st.selectbox("Obesity (BMI > 30)",["No", "Yes"])
    with col2:
        st.subheader("Clinical Information")
        cholesterol_level = st.number_input("Cholesterol Level",value=180.0)
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar",value=90.0)
        previous_heart_disease = st.selectbox("Previous Heart Disease",["No", "Yes"])
        smoking_status = st.selectbox("Smoking Status",["Never", "Past", "Current"])
    st.divider()
    predict = st.form_submit_button("Predict Risk",use_container_width=True)
# --------------------------------------------------
# Prediction
# --------------------------------------------------
if predict:
    data = pd.DataFrame({
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "diabetes": [1 if diabetes == "Yes" else 0],
        "cholesterol_level": [cholesterol_level],
        "obesity": [1 if obesity == "Yes" else 0],
        "fasting_blood_sugar": [fasting_blood_sugar],
        "previous_heart_disease": [
            1 if previous_heart_disease == "Yes" else 0
        ]
    })
    smoking = ohe_smoking.transform([[smoking_status]])
    if hasattr(smoking, "toarray"):
        smoking = smoking.toarray()
    smoking_df = pd.DataFrame(smoking,columns=ohe_smoking.get_feature_names_out())
    data = pd.concat([data, smoking_df], axis=1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    st.divider()
    st.subheader("Prediction Result")
    if hasattr(model, "predict_proba"):
        risk = model.predict_proba(data_scaled)[0][1]
        c1, c2, c3 = st.columns(3)
        with c2:
            st.metric("Risk Score",f"{risk*100:.1f}%")
        st.progress(float(risk))
    if prediction[0] == 1:
        st.error(
            "⚠ High Risk of Heart Attack"
        )
        st.info(
            "Consult a healthcare professional for further evaluation."
        )
    else:
        st.success(
            "✅ Low Risk of Heart Attack"
        )
        st.info(
            "Maintain healthy lifestyle habits and regular checkups."
        )
