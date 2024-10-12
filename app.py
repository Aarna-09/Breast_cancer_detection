from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

import joblib


model = joblib.load('xgboost_model.pkl')


# Define class mapping (0 -> 2, 1 -> 4)
class_mapping = {0: 2, 1: 4}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from form and convert to the appropriate type
        input_features = [float(x) for x in request.form.values()]
        
        # Convert input features into the format expected by the model
        features_array = np.array([input_features])

        # Make the prediction
        prediction = model.predict(features_array)[0]
        
        # Map the prediction to original classes (2 for benign, 4 for malignant)
        output = class_mapping[prediction]

        return render_template('index.html', prediction_text=f'The predicted class is: {output}')

if __name__ == "__main__":
    app.run(debug=True)
