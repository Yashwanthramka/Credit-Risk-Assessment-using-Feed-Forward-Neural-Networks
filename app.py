import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle  # Assuming you saved the preprocessor using pickle
import subprocess

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





# Function to generate personalized career guidance using OpenAI
def get_llm_explanation(user_data, prediction):
     # Prepare the prompt based on prediction
    if prediction == 1:
        outcome = "approved"
    else:
        outcome = "not approved"
    
    # Construct a descriptive prompt using user data
    prompt = (
        f"You are a financial advisor. "
        f"My loan application was {outcome} based on the model's prediction.\n\n"
        f"Loan Amount: {user_data['LOAN'].iloc[0]}\n"
        f"Mortgage Due: {user_data['MORTDUE'].iloc[0]}\n"
        f"Property Value: {user_data['VALUE'].iloc[0]}\n"
        f"Reason for Loan: {user_data['REASON'].iloc[0]}\n"
        f"Job Category: {user_data['JOB'].iloc[0]}\n"
        f"Years at Present Job: {user_data['YOJ'].iloc[0]}\n"
        f"Number of Major Derogatory Reports: {user_data['DEROG'].iloc[0]}\n"
        f"Number of Delinquent Credit Lines: {user_data['DELINQ'].iloc[0]}\n"
        f"Age of Oldest Credit Line (months): {user_data['CLAGE'].iloc[0]}\n"
        f"Number of Recent Credit Inquiries: {user_data['NINQ'].iloc[0]}\n"
        f"Number of Credit Lines: {user_data['CLNO'].iloc[0]}\n"
        f"Debt-to-Income Ratio: {user_data['DEBTINC'].iloc[0]}\n\n"
        f"Provide a clear and concise explanation for why my loan was {outcome}. Answer like you are talking to me."
    )


    result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            capture_output=True,
            text=True,
            check=True,
        )
    explanation = result.stdout.strip()
 
    return explanation

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
        
        explanation = get_llm_explanation(user_data, prediction);
        
        st.write("Explanation:", explanation)

if __name__ == "__main__":
    main()
