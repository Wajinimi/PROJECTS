from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import traceback

app = Flask(__name__)
CORS(app)

# Load the machine learning model if it exists
model_path = os.path.join(os.getcwd(), 'model.pkl')
print(f"Looking for model at: {model_path}")

model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
else:
    print("Warning: No model.pkl found - using mock predictions")

@app.route('/project', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)
    
    if model:
        try:
            # Extract features from the received data
            features = [
                float(data['Number_of_Bedrooms']),
                float(data['Location_Score']), 
                float(data['Square_Feet']),
                float(data['House_Age'])
            ]
            print("Extracted features:", features)
            
            # Use the machine learning model's prediction
            prediction = model.predict([features])[0]
            print("Model prediction:", prediction)
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()  # Print the stack trace to the console
            prediction = "Error during prediction"
    else:
        # Use a mock prediction if the model is not available
        prediction = 123456  # Example mock prediction value
        print("Mock prediction:", prediction)
    
    # Format the prediction with a dollar sign
    formatted_prediction = f"${prediction:,.2f}" if isinstance(prediction, (int, float)) else prediction
    print("Formatted prediction:", formatted_prediction)
    
    return jsonify({'predictedPrice': formatted_prediction})

if __name__ == '__main__':
    app.run(debug=True)
    