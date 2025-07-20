import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature columns
model = joblib.load("model.pkl")  # Make sure this matches your model filename
feature_columns = joblib.load("feature_columns.pkl")  # Must be saved during training

st.title("Cancer Survival Prediction App")

# User inputs
st.header("Enter Patient Information")

age = st.slider("Age", 20, 100, 40)
tumor_size = st.slider("Tumor Size (mm)", 0, 100, 20)
regional_node_examined = st.slider("Regional Nodes Examined", 0, 50, 3)
regional_node_positive = st.slider("Regional Nodes Positive", 0, 50, 1)

estrogen_status = st.selectbox("Estrogen Status", ["Positive", "Negative"])
progesterone_status = st.selectbox("Progesterone Status", ["Positive", "Negative"])
grade = st.selectbox("Grade", ["1", "2", "3"])
stage = st.selectbox("6th Stage", ["Stage_I", "Stage_IIA", "Stage_IIB", "Stage_IIIA", "Stage_IIIB", "Stage_IIIC"])

race = st.selectbox("Race", ["White", "Black", "Other"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Widowed", "Divorced", "Separated"])

# Build input dataframe
input_dict = {
    "Age": age,
    "Tumor Size": tumor_size,
    "Regional Node Examined": regional_node_examined,
    "Regional Node Positive": regional_node_positive,
    "Estrogen Status": estrogen_status,
    "Progesterone Status": progesterone_status,
    "Grade": grade,
    "6th Stage": stage,
    "Race": race,
    "Marital Status": marital_status
}

input_df = pd.DataFrame([input_dict])

# One-hot encode
input_encoded = pd.get_dummies(input_df)

# Reindex to match model features
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Prediction
if st.button("Predict Survival Status"):
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0]

    status = "Alive" if prediction == 1 else "Dead"
    st.subheader(f"Prediction: {status}")
    st.write(f"Probability of survival: {prob[1]*100:.2f}%")
