import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import joblib

file_path = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\portfolio\\backendFlask\\HousePrediction\\Plant_Estates_dataset.csv"
try:
    data = pd.read_csv(file_path).dropna(subset=["price"])
    print("Data loaded and price dropped")
except FileNotFoundError as e:
    print(f"File not found in the location specified {e}")
    exit()
    
X = data.drop(columns="price")
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state= 42)

linearegression = LinearRegression()
linearegression.fit(X_train, y_train)
formatted_coef = [f"{coef:.2f}" for coef in linearegression.coef_]
print("The models coefficient is:", formatted_coef)
print(f"The models intercept is: {linearegression.intercept_:.2f}")

y_test_predict = linearegression.predict(X_test)
print(y_test_predict)

mse = mean_squared_error(y_test, y_test_predict)
print(mse)

"""
random_forest = RandomForestRegressor(n_estimators = 100, random_state= 42)
random_forest.fit(X_train, y_train)
y_test_predict = random_forest.predict(X_test)
print(y_test_predict)
mse = mean_squared_error(y_test, y_test_predict)
print(mse) 
"""
#The mse for LinearRegression is 346740079, the mse for Random forest is  647795011"""

model_path = os.path.join(os.getcwd(), 'plantmodel.pkl')
try:
    joblib.dump(linearegression, model_path)
    print(f"Model has been saved: {model_path}")
except Exception as e:
    print(f"There is error saving the file: {e}")
