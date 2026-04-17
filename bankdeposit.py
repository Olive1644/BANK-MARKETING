import joblib
import pandas as pdp

import numpy as np
from sklearn import pipeline
import streamlit as st

st.markdown("""
    <style>

    /* Main background */
    .stApp {
        background-color: #0E1117;
    }

    /* Title styling */
    h1 {
        color: #1E88E5;
        font-weight: 700;
    }

    /* Subheaders */
    h2, h3 {
        color: #4FC3F7;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        border: none;
    }

    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }

    /* Input boxes */
    .stNumberInput, .stSelectbox {
        border-radius: 8px;
    }

    /* Section dividers */
    hr {
        border: 1px solid #1E88E5;
    }

    </style>
""", unsafe_allow_html=True)

# load modelimport joblib
model = joblib.load("xgb.pkl")
columns = joblib.load("columns.pkl")
def main():
    st.title("Bank Marketing Prediction App")
    
    # input features
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ["admin", "blue-collar", "entrepreneur", "housemaid", "management", "self-employed", "services", "retired", "student", "technician", "unemployed"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Credit in Default?", ["yes", "no"])
    housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Has Personal Loan?", ["yes", "no"])
    contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
    day_of_week = st.selectbox("Last Contact Day of the Week", ["mon", "tue", "wed", "thu", "fri"])
    month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=10000, value=0)
    euribor3m = st.number_input("Euribor 3 Month Rate", min_value=0.0, max_value=10.0, value=0.0)
    nr_employed = st.number_input("Number of Employees", min_value=0, max_value=100000, value=0)
    emp_var_rate = st.number_input("Employment Variation Rate", min_value=-10.0, max_value=10.0, value=0.0)
    cons_price_idx = st.number_input("Consumer Price Index", min_value=0.0, max_value=100.0, value=0.0)
    cons_conf_idx = st.number_input("Consumer Confidence Index", min_value=-100.0, max_value=100.0, value=0.0)

    input_df = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day_of_week': day_of_week,
        'month': month,
        'duration': duration,
        'euribor3m': euribor3m,
        'nr_employed': nr_employed,
        'emp_var_rate': emp_var_rate,
        'cons_price_idx': cons_price_idx,
        'cons_conf_idx': cons_conf_idx
    }])
    input_df = pd.get_dummies(input_df)

    # 🔥 ALIGN WITH TRAINING COLUMNS
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # predict button
    if st.button("Predict"):
        # make prediction
        prediction = model.predict(input_df)
        
        # display result
        if prediction[0] == 1:
            st.success("The client is likely to subscribe to a term deposit.")
        else:
            st.error("The client is unlikely to subscribe to a term deposit.")
if __name__ == "__main__":
    main()




