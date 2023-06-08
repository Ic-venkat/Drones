import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
data = pd.read_csv('drone_dataset.csv')

# Store the drone names separately
drone_names = data['Drone Name']

# Drop the 'Drone Name' and 'Drone Buy Website' columns from the dataset
data = data.drop(['Drone Name'], axis=1)

# Handle missing values by imputing them
imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)

# Split the dataset into features (X) and target variable (y)
X = data
y = drone_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate multiple models

# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Model accuracy: {rf_accuracy}")

# Logistic Regression Classifier
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
lr_y_pred = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print(f"Logistic Regression Model accuracy: {lr_accuracy}")

# Support Vector Classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svm_y_pred = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print(f"Support Vector Model accuracy: {svm_accuracy}")

# Save the trained models
joblib.dump(rf_classifier, 'random_forest_model.joblib')
joblib.dump(lr_classifier, 'logistic_regression_model.joblib')
joblib.dump(svm_classifier, 'support_vector_model.joblib')
