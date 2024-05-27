import streamlit as st
import pandas as pd
import numpy as np
import pickle
# Load the model
with open('rf.pkl', 'rb') as file:
    model = pickle.load(file)
st.title("Employee Attrition Prediction")

# Creating input fields
satisfaction_level = st.slider("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
previous_score = st.slider("Previous Score", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
number_of_projects = st.slider("Number of Projects", min_value=0, max_value=10, value=2, step=1)
average_monthly_hours = st.slider("Average Monthly Hours", min_value=90, max_value=310, value=200, step=10)
years_spent = st.slider("Years Spent", min_value=0, max_value=10, value=3, step=1)

departments = ('RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng', 'sales', 'support', 'technical')
department = st.selectbox("Department", departments)

salary = st.selectbox("Salary Level", ('low', 'medium', 'high'))

work_accident = st.checkbox("Work Accident")
promotion_last_5years = st.checkbox("Promotion in Last 5 Years")
# Convert department and salary to one-hot encoding
department_encoding = pd.get_dummies(department).reindex(columns=departments, fill_value=0)
salary_encoding = [1 if salary == 'low' else 0, 1 if salary == 'medium' else 0]

# Combine all the inputs
input_data = np.array([
    satisfaction_level,
    previous_score,
    number_of_projects,
    average_monthly_hours,
    years_spent,
    *department_encoding.loc[department],
    *salary_encoding,
    int(work_accident),
    int(promotion_last_5years)
]).reshape(1, -1)
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("The employee is likely to leave the organization.")
    else:
        st.success("The employee is likely to stay in the organization.")
