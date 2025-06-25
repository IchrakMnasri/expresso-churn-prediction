#Step4: 
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
region_enc = joblib.load("region_encoder.pkl")
top_pack_enc = joblib.load("top_pack_encoder.pkl")

st.title("📱 Expresso Client Churn Prediction")


with st.form("churn_form"):
    region = st.selectbox("REGION", region_enc.classes_)
    tenure = st.number_input("TENURE", min_value=0, value=10)
    montant = st.number_input("MONTANT", min_value=0.0)
    frequence = st.number_input("FREQUENCE", min_value=0.0)
    revenue = st.number_input("REVENUE", min_value=0.0)
    arpu_segment = st.number_input("ARPU_SEGMENT", min_value=0.0)
    frequence_rech = st.number_input("FREQUENCE_RECH", min_value=0.0)
    revenue_voice = st.number_input("REVENUE_VOICE", min_value=0.0)
    on_net = st.number_input("ON_NET", min_value=0.0)
    orange = st.number_input("ORANGE", min_value=0.0)
    tigo = st.number_input("TIGO", min_value=0.0)
    regularity = st.number_input("REGULARITY", min_value=0.0)
    top_pack = st.selectbox("TOP_PACK", top_pack_enc.classes_)

    submitted = st.form_submit_button("Predict CHURN")
#step 5 
if submitted:
    region_encoded = region_enc.transform([region])[0]
    top_pack_encoded = top_pack_enc.transform([top_pack])[0]

   
input_data = pd.DataFrame([{
        'REGION': region_encoded,
        'TENURE': tenure,
        'MONTANT': montant,
        'FREQUENCE': frequence,
        'REVENUE': revenue,
        'ARPU_SEGMENT': arpu_segment,
        'FREQUENCE_RECH': frequence_rech,
        'REVENUE_VOICE': revenue_voice,
        'ON_NET': on_net,
        'ORANGE': orange,
        'TIGO': tigo,
        'REGULARITY': regularity,
        'TOP_PACK': top_pack_encoded
    }])

# Predict
prob = model.predict_proba(input_data)[0][1]
st.success(f"Predicted Churn Probability: **{prob:.2%}**")
