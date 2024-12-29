# Fraud Detection using Supervised and Unsupervised Models

This repository implements fraud detection using machine learning models, including **Logistic Regression**, **XGBoost**, and **Isolation Forest**. The goal is to evaluate and compare the performance of these models in identifying fraudulent transactions from a dataset of credit card transactions. The project also includes model evaluation through various metrics and visualizations to understand model performance and feature importance.


## Setup

To get started, clone this repository and install the required dependencies:

### Install Dependencies
1. Clone the repository:

2. Install dependencies:
   pip install -r requirements.txt

3. Download the dataset (`creditcard.csv`) and place it in the project directory.

### Dependencies
The following libraries are required to run the analysis:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- shap

## Dataset

The dataset used for this project is a collection of credit card transactions, including both fraudulent and non-fraudulent transactions. The dataset contains the following columns:

- **V1, V2, ..., V28**: Anonymized features.
- **Time**: Time elapsed since the first transaction.
- **Amount**: Transaction amount.
- **Class**: Target variable (1 for fraud, 0 for non-fraud).

Place the dataset (`creditcard.csv`) in the root directory of the project.

## Models

This project implements both supervised and unsupervised machine learning models:

1. **Supervised Models**: 
   - **Logistic Regression**: A baseline model for fraud detection.
   - **XGBoost**: A more advanced model, optimized using RandomizedSearchCV for hyperparameter tuning.

2. **Unsupervised Model**:
   - **Isolation Forest**: Anomaly detection algorithm used to identify fraudulent transactions without labeled data.

### Model Training and Evaluation

- **Logistic Regression**: A simple classifier that uses regularization and iterative optimization to predict fraud. 
- **XGBoost**: A gradient boosting algorithm that works well with imbalanced data and has been tuned for better performance.
- **Isolation Forest**: Used as an unsupervised anomaly detection model to identify fraud.

## Model Evaluation

Models are evaluated using several performance metrics:
- **Accuracy**: The proportion of correct predictions.
- **Recall**: The proportion of actual frauds correctly predicted.
- **Precision**: The proportion of predicted frauds that are actual frauds.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
- **PR-AUC**: Area under the Precision-Recall curve.

### Model Performance
- Logistic Regression and XGBoost are trained and evaluated on the same dataset.
- The Isolation Forest model is used to detect outliers (fraudulent transactions) without needing labeled data.

## Visualizations

- **Confusion Matrices**: Displayed for each model to show the number of true positives, true negatives, false positives, and false negatives.
- **ROC and PR Curves**: Plotted for each model to visualize how well the model distinguishes between fraudulent and non-fraudulent transactions.
- **Feature Importance**: Visualized for the XGBoost model, showing the most important features used for predictions.
- **SHAP Values**: Explains model predictions and provides insights into feature contributions for individual predictions.

## Results

The results include the following:
- Evaluation metrics for each model.
- ROC and PR AUC curves comparing model performance.
- Feature importance plots to understand which features influence the predictions.
- SHAP analysis to explain individual predictions made by XGBoost.
