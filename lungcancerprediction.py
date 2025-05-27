# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv('surveylungcancer.csv')

# Inspect the dataset (optional)
print(data.head())
print(data.info())

# Data Cleaning
# Convert categorical 'GENDER' to numerical
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})

# Convert 'LUNG_CANCER' to numerical
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Check for missing values (optional)
print(data.isnull().sum())

# Feature and target separation
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, 'lung_cancer_model.pkl')
print("Model saved as 'lung_cancer_model.pkl'")

# Predict on new data (optional)
sample_data = np.array([[1, 65, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1]])
sample_prediction = model.predict(sample_data)
print("Prediction for sample data:", "Lung Cancer" if sample_prediction[0] == 1 else "No Lung Cancer")
