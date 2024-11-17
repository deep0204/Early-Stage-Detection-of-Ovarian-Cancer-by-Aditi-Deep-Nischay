from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris  # Replace with your dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load your dataset (replace this with your dataset loading code)
data1=pd.read_csv("OC_Marker.csv")
data1.head()
data2=pd.read_csv("OC_Genarel_Chem.csv")
data2.head()
data3=pd.read_csv("OC_Blood_Routine.csv")
data3.head()
oc_blood_routine =data3.drop(columns=['TYPE','TYPE.1'])
oc_general_chem = data2.drop(columns=['TYPE','TYPE.1'])
print(oc_blood_routine)
combined_data = pd.concat([data1.set_index(['Age']),
                           oc_blood_routine.set_index(['Age']),
                           oc_general_chem.set_index(['Age'])],
                          axis=1).reset_index()

# Print the resulting combined dataset
print(combined_data)
columns_t_drop = ['TYPE','TYPE.1']
x = combined_data.drop(columns=columns_t_drop,axis = 1)
x.head()
y = combined_data['TYPE']
y

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y



# Train KNN model
# Assuming x and y are your features and target variables
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=1)

# Step 1: Basic KNN Model (with n_neighbors=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain, ytrain)
ypred = knn.predict(xtest)

# Evaluate the basic KNN model
print("Basic KNN Model Evaluation:")
print(confusion_matrix(ytest, ypred))
print(classification_report(ytest, ypred))
error_rate = []
k_values = range(1, 40)

for i in k_values:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain, ytrain)
    pred_i = knn.predict(xtest)
    error_rate.append(np.mean(pred_i != ytest))

# Find the k value with the lowest error rate
optimal_k = k_values[np.argmin(error_rate)]
min_error_rate = np.min(error_rate)

# Print the optimal k and corresponding error rate
print(f"The optimal k value is {optimal_k} with the lowest error rate of {min_error_rate:.4f}")

# Plotting K vs Error rate
plt.figure(figsize=(10,6))
plt.plot(k_values, error_rate, color='blue', linestyle='--', markersize=10, markerfacecolor='red', marker='o')
plt.title('K vs Error Rate')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN hyperparameter tuning
param_dist_knn = {'n_neighbors': range(1, 11)}
knn = RandomizedSearchCV(KNeighborsClassifier(), param_dist_knn, n_iter=5, cv=3, n_jobs=-1, random_state=42)
knn.fit(X_train, y_train)
best_knn = knn.best_estimator_

# Print the best k value (this is the value of n_neighbors that gives the best performance)
print(f"The best k value is: {knn.best_params_['n_neighbors']}")

# Evaluation function
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'Log Loss': log_loss(y_test, model.predict_proba(X_test)),
    }
    print(confusion_matrix(y_test, y_pred))
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Evaluate best KNN model
print("\nEvaluating KNN Model...")
evaluate_model(best_knn, X_train, y_train, X_test, y_test)

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Assuming best_knn is already fitted with the best k value from RandomizedSearchCV

# Perform permutation importance
perm_importance = permutation_importance(best_knn, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Extract feature importance values
importance = perm_importance.importances_mean

# Sort features by importance
sorted_idx = importance.argsort()

# Select the top 3 features
top_3_idx = sorted_idx[-3:]

# Plotting the top 3 feature importances
plt.figure(figsize=(10, 6))
plt.barh(X.columns[top_3_idx], importance[top_3_idx], align='center', color='skyblue')
plt.xlabel("Permutation Importance")
plt.title("Top 3 Feature Importance (Permutation) for KNN Model")
plt.show()
# Step 1: Apply PCA to reduce dimensions and extract the most important features
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X)

# Step 2: Visualize the Explained Variance Ratio of PCA Components
plt.figure(figsize=(8, 6))
plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.7, align='center')
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')
plt.title('Explained Variance of PCA Components')
plt.show()

# Step 3: Visualize the Transformed Data in 2D (PCA)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA of Dataset (Transformed Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()  # Add color bar for target classes
plt.show()

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Step 5: Train the KNN Model on the PCA-transformed data
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Step 7: Print the evaluation metrics for the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
print("Training Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

print("Testing Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Optional: You can visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for KNN Model (Test Data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Train Random Forest model

import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import accuracy_score, confusion_matrix , precision_score, recall_score, f1_score, roc_auc_score, log_loss  # Import metrics

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)

# Assuming 'combined_data' is already defined and has 'TYPE' and 'TYPE.1' columns
X = combined_data.drop(['TYPE', 'TYPE.1'], axis=1)  # Features
y = combined_data['TYPE']  # Target

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier with GridSearch for Best Model
rfc = RandomForestClassifier(random_state=42)
param_grid_rfc = {
    'n_estimators': [200, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rfc, cv=5)
CV_rfc.fit(X_train, y_train)

# Best RFC model
m_best_rfc = CV_rfc.best_estimator_
y_pred_rfc = m_best_rfc.predict(X_test)
y_train_pred_rfc = m_best_rfc.predict(X_train)

# Metrics for Best RFC Model
train_accuracy = accuracy_score(y_train, y_train_pred_rfc)
test_accuracy = accuracy_score(y_test, y_pred_rfc)
precision = precision_score(y_test, y_pred_rfc, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred_rfc, average='weighted')
f1 = f1_score(y_test, y_pred_rfc, average='weighted')
auc = roc_auc_score(y_test, m_best_rfc.predict_proba(X_test)[:, 1])
logloss = log_loss(y_test, m_best_rfc.predict_proba(X_test))

# Printing Evaluation Results
print(f"Best RFC Model Evaluation:")
print("Training confusion matrix:")
print(confusion_matrix(y_train, y_train_pred_rfc))
print("Testing confusion matrix:")
print(confusion_matrix(y_test, y_pred_rfc))
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Log Loss: {logloss:.4f}")
# Random Forest Classifier (Overfitting example)
rfc_overfit = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42)
rfc_overfit.fit(X_train, y_train)
y_pred_rfc_overfit = rfc_overfit.predict(X_test)
rfc_overfit_train_acc = accuracy_score(y_train, rfc_overfit.predict(X_train))
rfc_overfit_test_acc = accuracy_score(y_test, y_pred_rfc_overfit)

# Print the results for the overfitting scenario
print(f"Random Forest (Overfitting):")
print(f"Train Accuracy: {rfc_overfit_train_acc:.4f}")
print(f"Test Accuracy: {rfc_overfit_test_acc:.4f}")
print(confusion_matrix(y_train, rfc_overfit.predict(X_train)))
print(confusion_matrix(y_test, y_pred_rfc_overfit))


# Random Forest Classifier (Underfitting example)
rfc_underfit = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
rfc_underfit.fit(X_train, y_train)
y_pred_rfc_underfit = rfc_underfit.predict(X_test)
rfc_underfit_train_acc = accuracy_score(y_train, rfc_underfit.predict(X_train))
rfc_underfit_test_acc = accuracy_score(y_test, y_pred_rfc_underfit)

# Print the results for the underfitting scenario
print(f"Random Forest (Underfitting):")
print(f"Train Accuracy: {rfc_underfit_train_acc:.4f}")
print(f"Test Accuracy: {rfc_underfit_test_acc:.4f}")
print(confusion_matrix(y_train, rfc_underfit.predict(X_train)))
print(confusion_matrix(y_test, y_pred_rfc_underfit))

# Data for comparison
models = ['Best RFC', 'Random Forest (Overfitting)', 'Random Forest (Underfitting)']
train_accuracies = [m_best_rfc_train_acc, rfc_overfit_train_acc, rfc_underfit_train_acc]
test_accuracies = [m_best_rfc_test_acc, rfc_overfit_test_acc, rfc_underfit_test_acc]

# Plotting
plt.figure(figsize=(10, 6))
x = np.arange(len(models))  # X-axis positions
width = 0.35  # Bar width

# Plot bars for Train and Test Accuracies
bars_train = plt.bar(x - width/2, train_accuracies, width, label='Train Accuracy')
bars_test = plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Train and Test Accuracies for RFC Models')
plt.xticks(x, models)
plt.legend()

# Adding values on top of the bars
for bar in bars_train:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,  # Positioning text slightly above the bar
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)

for bar in bars_test:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,  # Positioning text slightly above the bar
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()

# Assuming `m_best_rfc` is your trained Random Forest model
importances = m_best_rfc.feature_importances_

# Create a DataFrame to display feature importances along with their names
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top three most important features
top_features = feature_importance_df.head(3)
print("Top Three Important Features for RFC Model:")
print(top_features)

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features)  # Removed palette
plt.title('Top Three Most Important Features for RFC Model')
plt.show()
# Step 1: Apply PCA to reduce dimensions and extract the most important features
pca = PCA(n_components=2)  # Reduce to 2 components for visualization and model input
X_pca = pca.fit_transform(X)

# Step 2: Visualize the Explained Variance Ratio of PCA Components
plt.figure(figsize=(8, 6))
plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.7, align='center')
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')
plt.title('Explained Variance of PCA Components')
plt.show()

# Step 3: Visualize the Transformed Data in 2D
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA of Dataset (Transformed Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()  # Add color bar for target classes
plt.show()

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Classifier on the PCA-transformed data
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_train_pred = rfc.predict(X_train)
y_test_pred = rfc.predict(X_test)

# Step 7: Print the evaluation metrics for the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
print("Training Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

print("Testing Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Optional: You can visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Test Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



# Train Decision Tree model
# Decision Tree Model - Expanded Hyperparameter Tuning (with overfitting reduction techniques)
param_dist_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 7, 10],  # Limiting max_depth to avoid overfitting
    'min_samples_split': [2, 5, 10],  # Increasing min_samples_split
    'min_samples_leaf': [1, 2, 4],  # Increasing min_samples_leaf
    'class_weight': [None, 'balanced'],  # Optional: Use if classes are imbalanced
    'max_features': [None, 'sqrt', 'log2']  # Limiting features per split
}

dtc = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_distributions=param_dist_dt, n_iter=10, cv=3, n_jobs=-1, random_state=42)
dtc.fit(X_train, y_train)
best_dtc = dtc.best_estimator_

# Predicting the outcomes
y_train_pred = dtc.predict(X_train)
y_test_pred = dtc.predict(X_test)

# Calculate metrics
metrics = {
    'Train Accuracy': accuracy_score(y_train, y_train_pred),
    'Test Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=1),
    'Recall': recall_score(y_test, y_test_pred, average='weighted'),
    'F1 Score': f1_score(y_test, y_test_pred, average='weighted'),
    'AUC': roc_auc_score(y_test, dtc.predict_proba(X_test)[:, 1]),
    'Log Loss': log_loss(y_test, dtc.predict_proba(X_test))
}

# Print evaluation results
print(f"Best Decision Tree Model Evaluation:")
print("Training confusion matrix:\n", confusion_matrix(y_train, y_train_pred))
print("Testing confusion matrix:\n", confusion_matrix(y_test, y_test_pred))
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

dtc_overfit = DecisionTreeClassifier(max_depth=None, random_state=42)

# Train the model
dtc_overfit.fit(X_train, y_train)

# Predictions
y_train_pred = dtc_overfit.predict(X_train)
y_test_pred = dtc_overfit.predict(X_test)

# Metrics
print(f"Overfitting Decision Tree Model:")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("Training confusion matrix:\n", confusion_matrix(y_train, y_train_pred))
print("Testing confusion matrix:\n", confusion_matrix(y_test, y_test_pred))

dtc_underfit = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model
dtc_underfit.fit(X_train, y_train)

# Predictions
y_train_pred = dtc_underfit.predict(X_train)
y_test_pred = dtc_underfit.predict(X_test)

# Metrics
print(f"Underfitting Decision Tree Model:")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy:{ accuracy_score(y_test, y_test_pred):.4f}")
print("Training confusion matrix:\n", confusion_matrix(y_train, y_train_pred))
print("Testing confusion matrix:\n", confusion_matrix(y_test, y_test_pred))

import matplotlib.pyplot as plt
import numpy as np

# Data for comparison
models = ['Best DT', 'Overfitting DT', 'Underfitting DT']
train_accuracies = [0.9032, 1.0000, 0.9283]
test_accuracies = [0.8857, 0.8143, 0.8429]

# Plotting
plt.figure(figsize=(10, 6))
x = np.arange(len(models))  # X-axis positions
width = 0.35  # Bar width

# Plot bars for Train and Test Accuracies
bars_train = plt.bar(x - width/2, train_accuracies, width, label='Train Accuracy', color='b')  # Blue for Train
bars_test = plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='orange')  # Orange for Test

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Train and Test Accuracies for Decision Tree Models')
plt.xticks(x, models)
plt.legend()

# Adding values on top of the bars
for bar in bars_train:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,  # Positioning text slightly above the bar
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)

for bar in bars_test:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,  # Positioning text slightly above the bar
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()

importances = m_best_dtc.feature_importances_

# Create a DataFrame to display feature importances along with their names
feature_importance_df_dtc = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the DataFrame by importance in descending order
feature_importance_df_dtc = feature_importance_df_dtc.sort_values(by='Importance', ascending=False)

# Display the top three most important features
top_features_dtc = feature_importance_df_dtc.head(3)
print("Top Three Important Features for DT Model:")
print(top_features_dtc)

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_dtc)
plt.title('Top Three Most Important Features for Cancer Prediction using Decision Tree')
plt.show()


# Save the models
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(rfc, 'random_forest_model.pkl')
joblib.dump(dtc, 'decision_tree_model.pkl')

print("Models saved successfully.")
