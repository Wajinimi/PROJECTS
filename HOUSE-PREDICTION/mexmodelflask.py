from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import traceback

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.getcwd(), 'mexmodel.pkl')
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file not found at:", model_path)

@app.route('/mexmodel', methods=["POST"])
def predict():
    data = request.json
    print("Received data:", data)
    
    try:
        features = [
            float(data["Square_Footage"]),
            float(data["Bedrooms"]),
            float(data["Bathrooms"]),
            float(data["Age"]),
            float(data["Condition"])
        ]
        print("Extracted features:", features)

        if model:
            prediction = model.predict([features])[0]
            print("Model prediction:", prediction)
        else:
            prediction = 1223345  # Mock prediction
            print("Using mock prediction:", prediction)

        # Ensure the prediction is returned as a numeric value
        numeric_prediction = float(prediction)

    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Error during prediction'}), 500

    return jsonify({'predictedPrice': numeric_prediction})

if __name__ == '__main__':
    app.run(debug=True)
