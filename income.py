import streamlit as st
import pandas as pd
import joblib

# Load the complete pipeline
try:
    model_pipeline = joblib.load("xgboost_pipeline.pkl")
except FileNotFoundError as e:
    st.error(f"Error: Missing file. Please ensure 'xgboost_pipeline.pkl' is in the same directory.")
    st.stop()

st.title("Income Classification App")
st.write("Enter information to predict your income level.")

# Define education level to education-num mapping
education_mapping = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}

# Collect user input
age = st.number_input("Age", min_value=17, max_value=100, value=30)
workclass = st.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
])
fnlwgt = st.number_input("Sample Weight", min_value=0, value=100000)
education = st.selectbox("Education Level", list(education_mapping.keys()), index=9)  # Default to "Some-college" (10)
education_num = education_mapping[education]  # Map selected education level to education-num
marital_status = st.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
])
relationship = st.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = st.selectbox("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
native_country = st.selectbox("Native Country", [
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
    "Germany", "India", "Mexico", "Outlying-US(Guam-USVI-etc)", "Japan",
    "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines",
    "Italy", "Poland", "Jamaica", "Vietnam", "Laos", "Portugal", "El-Salvador",
    "Columbia", "Dominican-Republic", "Guatemala", "North-America", "France",
    "Yugoslavia", "Peru", "Scotland", "Trinadad&Tobago", "Haiti", "Hungary",
    "Holand-Netherlands", "Ireland", "Ecuador", "Hong", "Taiwan", "Thailand",
    "Nicaragua", "Dominica", "France", "Hungary", "Ireland", "Guatemala",
    "Nicaragua"
])

# Create a DataFrame from the raw user inputs
# Do not one-hot encode, scale, or align columns here. The pipeline will do it.
input_dict = {
    "age": [age],
    "workclass": [workclass],
    "fnlwgt": [fnlwgt],
    "education-num": [education_num],
    "marital-status": [marital_status],
    "occupation": [occupation],
    "relationship": [relationship],
    "race": [race],
    "sex": [sex],
    "capital-gain": [capital_gain],
    "capital-loss": [capital_loss],
    "hours-per-week": [hours_per_week],
    "native-country": [native_country]
}

input_df = pd.DataFrame(input_dict)

# Predict with the pipeline
if st.button("Predict Income"):
    # The pipeline will automatically apply all the transformations to input_df
    prediction = model_pipeline.predict(input_df)[0]
    prediction_proba = model_pipeline.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Predicted: >50K (High Income) — Probability: {prediction_proba:.2f}")
    else:
        st.error(f"Predicted: <=50K (Low Income) — Probability: {prediction_proba:.2f}")