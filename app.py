import streamlit as st
import pandas as pd
from model import load_data, train_knn, train_random_forest, train_decision_tree
from sklearn.model_selection import train_test_split

# Load and preprocess data (this happens once at the start)
@st.cache_data  # Cache data to avoid redundant loading
def get_data():
    return load_data()

X, y = get_data()

# Split the data into train and test sets (do this once and use for all models)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models based on user selection
@st.cache_resource  # Cache trained models to avoid retraining
def train_model(model_type):
    if model_type == "K-Nearest Neighbors (KNN)":
        return train_knn(X_train, X_test, y_train, y_test)
    elif model_type == "Random Forest":
        return train_random_forest(X_train, X_test, y_train, y_test)
    elif model_type == "Decision Tree":
        return train_decision_tree(X_train, X_test, y_train, y_test)
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
        # Save model to session state after training
        st.session_state.model = model
        st.success(f"{model_choice} has been trained successfully!")

# Section for predictions
st.header("Make Predictions")
uploaded_file = st.file_uploader("Upload a CSV file for predictions", type="csv")

if uploaded_file:
    try:
        # Try reading with default comma delimiter
        user_data = pd.read_csv(uploaded_file, encoding='utf-8', sep=',')
    except UnicodeDecodeError:
        # Fallback to other encoding if utf-8 doesn't work
        user_data = pd.read_csv(uploaded_file, encoding='ISO-8859-1', sep=',')
    except pd.errors.ParserError:
        # If there's a parser error, handle by ignoring bad lines
        st.error("There was an issue with the CSV formatting. Please check the file for irregular rows.")
        user_data = pd.read_csv(uploaded_file, encoding='utf-8', sep=',', error_bad_lines=False)
    
    # Check if model is trained and available
    if 'model' in st.session_state and st.session_state.model:
        st.write("Predicting...")
        model = st.session_state.model
        predictions = model.predict(user_data)
        st.write("Predictions:")
        st.write(predictions)
    else:
        st.warning("Train a model first!")
