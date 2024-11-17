import pandas as pd
import streamlit as st
from model import load_data, train_knn, train_random_forest, train_decision_tree

# Load and preprocess data (this happens once at the start)
@st.cache_data  # Cache data to avoid redundant loading
def get_data(uploaded_files):
    # Assuming three files: OC_Marker.csv, OC_Genarel_Chem.csv, OC_Blood_Routine.csv
    if len(uploaded_files) != 3:
        st.error("Please upload exactly 3 CSV files.")
        return None, None
    
    # Read the uploaded files
    data1 = pd.read_csv(uploaded_files[0])
    data2 = pd.read_csv(uploaded_files[1])
    data3 = pd.read_csv(uploaded_files[2])
    
    # Combine the data and preprocess as in the original load_data function
    combined_data = pd.concat(
        [data1.set_index(['Age']),
         data3.drop(columns=['TYPE', 'TYPE.1']).set_index(['Age']),
         data2.drop(columns=['TYPE', 'TYPE.1']).set_index(['Age'])],
        axis=1
    ).reset_index()

    y = combined_data['TYPE']
    X = combined_data.drop(columns=['TYPE', 'TYPE.1'], axis=1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Streamlit App Layout
st.title("Ovarian Cancer Detection Models")

# Sidebar for model selection
st.sidebar.header("Select a Model")
model_choice = st.sidebar.selectbox(
    "Choose a model to train and use:",
    ["K-Nearest Neighbors (KNN)", "Random Forest", "Decision Tree"]
)

# Allow multiple files upload
uploaded_files = st.file_uploader("Upload 3 CSV files for the model", type="csv", accept_multiple_files=True)

if uploaded_files:
    # Load data from uploaded files
    X, y = get_data(uploaded_files)

    if X is None or y is None:
        st.warning("Error loading data. Ensure you have uploaded exactly 3 files.")
    else:
        # Train models based on user selection
        @st.cache_resource  # Cache trained models to avoid retraining
        def train_model(model_type):
            if model_type == "K-Nearest Neighbors (KNN)":
                return train_knn(X, y)
            elif model_type == "Random Forest":
                return train_random_forest(X, y)
            elif model_type == "Decision Tree":
                return train_decision_tree(X, y)
            else:
                st.error("Invalid model selected.")
                return None

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
                predictions = model.predict(user_data)
                st.write("Predictions:")
                st.write(predictions)
            else:
                st.warning("Train a model first!")
