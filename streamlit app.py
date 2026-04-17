import joblib
import numpy as np
import streamlit as st

# load model
model = joblib.load("xgb.pkl")
def main():
    st.title("Bank Marketing Prediction App")
    
    # input features
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "self-employed", "services", "retired", "student", "technician", "unemployed"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Credit in Default?", ["yes", "no"])
    balance = st.number_input("Balance", min_value=-100000, max_value=100000, value=0)
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

    # predict button
    if st.button("Predict"):
        # create input array
        input_data = np.array([[age, job, marital, education, default, balance, housing, loan, contact, day_of_week, month, duration, euribor3m, nr_employed, emp.var.rate, cons.price.idx, cons.conf.idx]])
        
        # make prediction
        prediction = model.predict(input_data)
        
        # display result
        if prediction[0] == 1:
            st.success("The client is likely to subscribe to a term deposit.")
        else:
            st.error("The client is unlikely to subscribe to a term deposit.")
if __name__ == "__main__":
    main()




