import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the models
knn_model = joblib.load('knn_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')

# Function to preprocess data (example: replace with your data preprocessing steps)
def preprocess_data(data):
    # Add any necessary data preprocessing steps here
    # For this example, just return the data as it is
    return data

# Function to get predictions from a model
def get_predictions(data, model):
    return model.predict(data)

# Function to compare all models' predictions
def compare_models(processed_data):
    """
    Compare predictions from different models including KNN.
    """
    knn_predictions = get_predictions(processed_data, knn_model)
    rf_predictions = get_predictions(processed_data, rf_model)
    dt_predictions = get_predictions(processed_data, dt_model)
    
    # Assuming 'target' column is in the processed data
    comparison = pd.DataFrame({
        'True': processed_data['target'],  # Replace with actual target column in data
        'KNN Predictions': knn_predictions,
        'Random Forest Predictions': rf_predictions,
        'Decision Tree Predictions': dt_predictions
    })
    
    return comparison

# Streamlit Interface
st.title('Ovarian Cancer Prediction with Multiple Models')

# Sidebar for model selection
st.sidebar.header('Choose Model for Predictions:')
model_choice = st.sidebar.selectbox(
    'Select Model:',
    ['K-Nearest Neighbors', 'Random Forest', 'Decision Tree']
)

# Upload CSV file for predictions
uploaded_file = st.file_uploader("Upload Your CSV Data", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.dataframe(data)
    
    # Preprocess data
    processed_data = preprocess_data(data)
    
    if st.button('Make Predictions'):
        # Make predictions based on the selected model
        if model_choice == 'K-Nearest Neighbors':
            predictions = get_predictions(processed_data, knn_model)
        elif model_choice == 'Random Forest':
            predictions = get_predictions(processed_data, rf_model)
        elif model_choice == 'Decision Tree':
            predictions = get_predictions(processed_data, dt_model)
        else:
            st.error("Please select a valid model.")
        
        st.write("Predictions:")
        st.write(predictions)
        
        # Option to view comparison of models
        if st.checkbox('Compare Models'):
            comparison = compare_models(processed_data)
            st.write("Model Comparison:")
            st.write(comparison)
        
        # Option to view comparison chart
        if st.checkbox('Show Model Comparison Chart'):
            comparison = compare_models(processed_data)
            st.bar_chart(comparison)
