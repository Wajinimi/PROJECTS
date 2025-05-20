import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load and clean the dataset
file_path = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\portfolio\\backendFlask\\HousePrediction\\house_price_dataset1.csv"

try:
    data = pd.read_csv(file_path).dropna(subset=["House_Price"])
    print("Dataset loaded and cleaned.")
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    exit()

# Drop less important features based on correlation analysis
columns_to_drop = ["Garage_Size", "Nearby_Schools_Rating", "Proximity_to_City_Center", "Number_of_Bathrooms",]
data = data.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")

# Split features and target
X = data.drop(columns=["House_Price"])
y = data["House_Price"]
print(f"Shape of X after dropping: {X.shape}, Shape of y: {y.shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into train and test sets.")

# Create and train the Random Forest model
randomForest = RandomForestRegressor(n_estimators=100, random_state=42)
randomForest.fit(X_train, y_train)
print("Random Forest model trained.")

# Make predictions
y_test_pred = randomForest.predict(X_test)
print("Predictions made.")

# Evaluate the model
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save the trained model
model_path = os.path.join(os.getcwd(), 'model.pkl')
try:
    joblib.dump(randomForest, model_path)
    print(f"Model saved to: {model_path}")
except Exception as e:
    print(f"Error saving model: {e}")