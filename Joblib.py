import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model pipeline
model_path = 'D:/Machine_Learning/Cases/HumanResource/bm_rf.joblib'
model = joblib.load(model_path)

# 2. Set up the Page UI
st.set_page_config(page_title="Employee Retention Predictor", layout="centered")
st.title("📊 Employee Attrition Prediction")
st.write("Enter employee details below to predict the likelihood of them leaving the company.")

# 3. Create Input Fields (Matching your HR dataset features)
col1, col2 = st.columns(2)

with col1:
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
    projects = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
    monthly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350, value=200)

with col2:
    tenure = st.number_input("Time Spent at Company (Years)", min_value=1, max_value=15, value=3)
    work_accident = st.selectbox("Work Accident", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    promotion = st.selectbox("Promotion in Last 5 Years", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    salary = st.selectbox("Salary Level", options=['low', 'medium', 'high'])

department = st.selectbox("Department", options=[
    'sales', 'accounting', 'hr', 'technical', 'support',
    'management', 'IT', 'product_mng', 'marketing', 'RandD'
])

# 4. Prediction Logic
if st.button("Predict Probability of Leaving"):
    # Create a dataframe for the input (must match original X column names exactly)
    input_df = pd.DataFrame([[
        satisfaction, evaluation, projects, monthly_hours,
        tenure, work_accident, promotion, department, salary
    ]], columns=[
        'satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
        'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Department', 'salary'
    ])

    # Get Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display Results
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Risk: The employee is likely to leave. (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk: The employee is likely to stay. (Probability of leaving: {probability:.2%})")