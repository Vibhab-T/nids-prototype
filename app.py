from flask import Flask, request, jsonify

import numpy as np 
import joblib

import os 
import sys
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, 'ML Model')
sys.path.append(model_dir)

from logistic_regression import LogisticRegression  

app = Flask(__name__)

# load model 
model = joblib.load("ML Model/logistic_regression_model.pkl")
transformer = joblib.load("ML Model/column_transformer.pkl")
label_encoder = joblib.load("ML Model/label_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)  # convert to 2d array 

    # Transform the features using the loaded transformer
    features_encoded = transformer.transform(features)

    # predict using the trained model 
    prediction = model.predict(features_encoded)

    #convert prediction back to the binary label: normal or anomaly 
    label = label_encoder.inverse_transform(prediction)

    return jsonify({'prediction': label[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5002)  # Run the Flask app on port 5002