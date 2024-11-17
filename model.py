import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, confusion_matrix, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import joblib

# Load and preprocess the dataset
def load_data():
    data1 = pd.read_csv("OC_Marker.csv")
    data2 = pd.read_csv("OC_Genarel_Chem.csv")
    data3 = pd.read_csv("OC_Blood_Routine.csv")

    oc_blood_routine = data3.drop(columns=['TYPE', 'TYPE.1'])
    oc_general_chem = data2.drop(columns=['TYPE', 'TYPE.1'])

    combined_data = pd.concat(
        [data1.set_index(['Age']),
         oc_blood_routine.set_index(['Age']),
         oc_general_chem.set_index(['Age'])],
        axis=1
    ).reset_index()

    y = combined_data['TYPE']
    X = combined_data.drop(columns=['TYPE', 'TYPE.1'], axis=1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Function for evaluating a model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=1),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'Log Loss': log_loss(y_test, model.predict_proba(X_test)),
    }
    print(confusion_matrix(y_test, y_pred))
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    return model  # Return the trained model

# Train and evaluate KNN with hyperparameter tuning
def train_knn(X_train, X_test, y_train, y_test):
    param_dist_knn = {'n_neighbors': range(1, 11)}
    knn = RandomizedSearchCV(KNeighborsClassifier(), param_dist_knn, n_iter=5, cv=3, n_jobs=-1, random_state=42)
    knn.fit(X_train, y_train)
    best_knn = knn.best_estimator_

    print("\nEvaluating KNN Model...")
    evaluate_model(best_knn, X_test, y_test)
    return best_knn  # Return the trained model

# Train and evaluate Random Forest with GridSearchCV
def train_random_forest(X_train, X_test, y_train, y_test):
    param_grid_rfc = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    rfc = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rfc, cv=5)
    rfc.fit(X_train, y_train)
    best_rfc = rfc.best_estimator_

    print("\nEvaluating Random Forest Model...")
    evaluate_model(best_rfc, X_test, y_test)
    return best_rfc  # Return the trained model

# Train and evaluate Decision Tree with hyperparameter tuning
def train_decision_tree(X_train, X_test, y_train, y_test):
    param_dist_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_dist_dt, cv=5)
    dt.fit(X_train, y_train)
    best_dt = dt.best_estimator_

    print("\nEvaluating Decision Tree Model...")
    evaluate_model(best_dt, X_test, y_test)
    return best_dt  # Return the trained model

# Perform PCA and visualize results
def apply_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.7, align='center')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Ratio')
    plt.title('Explained Variance of PCA Components')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title('PCA of Dataset (Transformed Data)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

    return X_pca

# Main function to run all models
def run_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN
    print("\n--- Training KNN ---")
    knn_model = train_knn(X_train, X_test, y_train, y_test)

    # Random Forest
    print("\n--- Training Random Forest ---")
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    # Decision Tree
    print("\n--- Training Decision Tree ---")
    dt_model = train_decision_tree(X_train, X_test, y_train, y_test)

    # PCA Visualization
    print("\n--- Applying PCA ---")
    apply_pca(X, y)

    # Save models for reuse
    joblib.dump(knn_model, 'knn_model.pkl')
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(dt_model, 'decision_tree_model.pkl')

    print("\nModels have been saved successfully.")

# Run the script
if __name__ == "__main__":
    run_models()
