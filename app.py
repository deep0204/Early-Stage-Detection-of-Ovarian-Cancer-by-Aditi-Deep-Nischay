import streamlit as st
import pandas as pd
from model import load_data, train_knn, train_random_forest, train_decision_tree
import joblib

# Load and preprocess data (this happens once at the start)
@st.cache_data  # Cache data to avoid redundant loading
def get_data():
    return load_data()

X, y = get_data()

# Train models based on user selection
@st.cache_resource  # Cache trained models to avoid retraining
def train_model(model_type):
    if model_type == "K-Nearest Neighbors (KNN)":
        return train_knn(X, X, y, y)  # Pass X_train, X_test, y_train, y_test
    elif model_type == "Random Forest":
        return train_random_forest(X, X, y, y)
    elif model_type == "Decision Tree":
        return train_decision_tree(X, X, y, y)
    else:
        st.error("Invalid model selected.")
        return None

# Streamlit App Layout
st.title("Ovarian Cancer Detection Models")

# Sidebar for model selection
st.sidebar.header("Select a Model")
model_choice = st.sidebar.selectbox(
    "Choose a model to train and use:",
    ["K-Nearest Neighbors (KNN)", "Random Forest", "Decision Tree"]
)

# Button to train the selected model
if st.sidebar.button("Train Model"):
    st.write(f"Training {model_choice}...")
    model = train_model(model_choice)
    if model:
        st.success(f"{model_choice} has been trained successfully!")

# Section for predictions
st.header("Make Predictions")
uploaded_file = st.file_uploader("Upload a CSV file for predictions", type="csv")

if uploaded_file:
    # Load uploaded data
    user_data = pd.read_csv(uploaded_file)

    # Make predictions if a model is trained
    if 'model' in locals() and model:
        st.write("Predicting...")
        predictions = model.predict(user_data)  # Use the trained model for predictions
        st.write("Predictions:")
        st.write(predictions)
    else:
        st.warning("Train a model first!")
