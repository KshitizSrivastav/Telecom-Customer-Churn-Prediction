# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the datasets
train_data = pd.read_csv("churn-bigml-80.csv")
test_data = pd.read_csv("churn-bigml-20.csv")

# Data preprocessing
def preprocess_data(data):
    # Convert categorical variables to numeric using LabelEncoder
    label_encoders = {}
    for column in ['State', 'International plan', 'Voice mail plan']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    # Separate features and target variable
    # X: Contains all features except the target column (Churn).
    # y: Contains the target column (Churn), converted to integers (0 for no churn, 1 for churn).
    X = data.drop(columns=['Churn'])
    y = data['Churn'].astype(int)  # Convert target to integer (0 or 1)
    
    # Standardize numerical features
    #StandardScaler: Standardizes the features in X to have a mean of 0 and a standard deviation of 1.
    #fit_transform: Fits the scaler to the data and transforms it.
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y, label_encoders, scaler

# Preprocess training and testing data
X_train, y_train, label_encoders_train, scaler_train = preprocess_data(train_data)
X_test, y_test, _, _ = preprocess_data(test_data)

# Split training data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define ensemble models
# RandomForestClassifier: A type of ensemble model that uses multiple decision trees to make predictions.
# GradientBoostingClassifier: Another type of ensemble model that builds trees sequentially to improve accuracy.
# n_estimators: Number of trees in the forest.
# random_state: Controls the randomness of the estimator.
# learning_rate: Determines the contribution of each tree to the final prediction.
# soft: Uses predicted probabilities to make the final prediction.
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Combine models using VotingClassifier
#VotingClassifier: Combines the random forest and gradient boosting models.
#voting='soft': Uses predicted probabilities for voting.
ensemble_model = VotingClassifier(estimators=[
    ('rf', random_forest),
    ('gb', gradient_boosting)
], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train_split, y_train_split)

# Evaluate on validation set
y_val_pred = ensemble_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)
print("\nClassification Report (Validation):\n", classification_report(y_val, y_val_pred))
print("\nConfusion Matrix (Validation):\n", confusion_matrix(y_val, y_val_pred))

#Confusion Matrix is a table used to evaluate the performance of a classification model. 
#It compares the predicted labels with the actual labels and provides a breakdown of correct 
#and incorrect predictions for each class.

# Evaluate on test set
y_test_pred = ensemble_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("\nTest Accuracy:", test_accuracy)
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))

# Filter and display customers who churned out
churned_customers = test_data[y_test_pred == 1]
print("\nCustomers Who Churned Out:")
print(churned_customers)
churned_count = len(churned_customers)
print("\nCount of Customers Who Churned Out:", churned_count)

# Cross-validation on training data
cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='accuracy')
print("\nCross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

#**********************************************************************************************************************************************
#The code preprocesses the data, trains an ensemble model, 
#evaluates it on validation and test sets, and identifies customers who churned.
#It also performs cross-validation to ensure the model's robustness.
#**********************************************************************************************************************************************

