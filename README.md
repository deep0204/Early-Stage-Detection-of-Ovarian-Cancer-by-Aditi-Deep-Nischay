# Early-Stage-Detection-of-Ovarian-Cancer-by-Aditi-Deep-Nischay
..

This college project explores machine learning to aid in early ovarian cancer detection. We applied K-Nearest Neighbors (KNN), Random Forest (RF), and Decision Tree (DT) models, plus an RF-DT ensemble for enhanced accuracy in future predeiction on un-seen data.

Our Contribution : 

Hyperparameter Tuning : 

•⁠  Optimized Model Parameters: Used GridSearchCV to systematically search for the best combination of hyperparameters across models (Random Forest, Decision Tree, etc.). This significantly boosted model performance      by identifying the best-fit parameters.

•⁠  Cross-Validation: Employed k-fold cross-validation to assess model stability and prevent overfitting, leading to more reliable generalization on unseen data.


Feature Selection and Engineering—Used model-based methods: 

•⁠ (e.g., feature importance from Random Forest or Decision Tree) to identify the most influential features, allowing for refined input and improving predictive performance.


Visualization and Interpretability : 

•⁠  Feature Importance Plotting: Visualized the top three or more most important features for each model, allowing domain experts to understand which factors most influence predictions.
•⁠  Metric Comparison Plotting: Compared model performance across various metrics in a single visualization, providing a clear and comprehensive comparison for decision-makers.

Integration of PCA
Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset while retaining the most important variance. It helps streamline the data by eliminating redundant and correlated features, improving model efficiency and training time.
