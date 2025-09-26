import streamlit as st
import pandas as pd
import joblib

# Load the model and preprocessor trained on top features
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/random_forest_attrition.joblib")

st.title("HR Attrition Prediction")
st.write("Enter employee details to predict attrition probability:")

# Collect employee input for top features
employee_data = {}

# Numeric Inputs
employee_data['Age'] = st.number_input("Age", min_value=18, max_value=60, value=30)
employee_data['MonthlyIncome'] = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
employee_data['YearsAtCompany'] = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
employee_data['DistanceFromHome'] = st.number_input("Distance From Home", min_value=0, max_value=50, value=10)

# Categorical Inputs
employee_data['OverTime'] = st.selectbox("OverTime", ["Yes", "No"])
employee_data['JobRole'] = st.selectbox("Job Role", [
    "Sales Executive","Research Scientist","Laboratory Technician",
    "Manufacturing Director","Healthcare Representative",
    "Manager","Sales Representative","Research Director","Human Resources"
])
employee_data['JobSatisfaction'] = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1,2,3,4])

# Predict Button
if st.button("Predict Attrition"):
    df_input = pd.DataFrame([employee_data])

    # Transform & predict
    Xp = preprocessor.transform(df_input)
    proba = model.predict_proba(Xp)[0,1]
    pred = int(proba >= 0.5)

    st.write(f"**Attrition Probability:** {proba:.2f}")
    st.write(f"**Predicted Class:** {'Yes' if pred==1 else 'No'}")
