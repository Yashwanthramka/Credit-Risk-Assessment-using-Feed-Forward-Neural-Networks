import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle  # Assuming you saved the preprocessor using pickle

# Load the trained model
model = load_model('Models/hmeq_loan_approval_model.keras')  # Path to your saved model

# Load the preprocessor (assuming you saved it with pickle)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Define a function to get user input from the Streamlit interface
def get_user_input():
    st.title("Loan Approval Prediction")
    
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    mort_due = st.number_input("Mortgage Due", min_value=0.0)
    property_value = st.number_input("Property Value", min_value=0.0)
    reason = st.selectbox("Reason for Loan", options=["DebtCon", "HomeImp"])
    job = st.selectbox("Job Category", options=["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"])
    yoj = st.number_input("Years at Present Job", min_value=0.0)
    derog = st.number_input("Number of Major Derogatory Reports", min_value=0)
    delinq = st.number_input("Number of Delinquent Credit Lines", min_value=0)
    clage = st.number_input("Age of Oldest Credit Line (months)", min_value=0.0)
    ninq = st.number_input("Number of Recent Credit Inquiries", min_value=0)
    clno = st.number_input("Number of Credit Lines", min_value=0)
    debtinc = st.number_input("Debt-to-Income Ratio", min_value=0.0)

    # Create a DataFrame with the user input
    user_data = pd.DataFrame({
        'LOAN': [loan_amount],
        'MORTDUE': [mort_due],
        'VALUE': [property_value],
        'REASON': [reason],
        'JOB': [job],
        'YOJ': [yoj],
        'DEROG': [derog],
        'DELINQ': [delinq],
        'CLAGE': [clage],
        'NINQ': [ninq],
        'CLNO': [clno],
        'DEBTINC': [debtinc]
    })
    return user_data

# Function to make predictions
def predict_approval(user_data):
    # Preprocess user input using the same preprocessor pipeline
    user_data_processed = preprocessor.transform(user_data)
    
    # Predict using the model
    prediction_prob = model.predict(user_data_processed).ravel()[0]
    prediction = int(prediction_prob >= 0.5)  # Assuming 0.5 threshold for binary classification
    
    # Display prediction result
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Not Approved.")
    
    # Show the prediction probability
    st.write(f"Approval Probability: {prediction_prob:.2f}")

    return prediction

# Main app
def main():
    user_data = get_user_input()
    
    if st.button("Submit"):
        prediction = predict_approval(user_data)
        
        # Explanation from Local LLM
        if prediction == 1:
            explanation = "Your loan was approved because your financial history and current standing align with our approval criteria."
        else:
            explanation = "Your loan was not approved due to certain criteria not being met, such as income, credit history, or other financial ratios."
        
        st.write("Explanation:", explanation)

if __name__ == "__main__":
    main()
