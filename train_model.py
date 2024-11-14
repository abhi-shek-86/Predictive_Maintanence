import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the data
data_path = os.path.join('data', 'sensor_data.csv')
data = pd.read_csv(data_path)

# Check for missing values in the target variable (y)
if data['maintenance_needed'].isnull().sum() > 0:
    print("Warning: Missing values found in target variable 'maintenance_needed'.")
    # Option 1: Drop rows with missing target values
    data = data.dropna(subset=['maintenance_needed'])

# Handle missing values in features (if any)
data.fillna(0, inplace=True)  # Replace missing values in features with 0

# Features and target
X = data[['sensor1', 'sensor2', 'sensor3', 'sensor4']]  # Independent variables
y = data['maintenance_needed']                          # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model_path = os.path.join('model', 'predictive_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
