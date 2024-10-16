from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load paths for model and encoder
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), 'encoder.pkl')

# Load the trained model and encoder
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    crop = data.get('crop')
    season = data.get('season')
    state = data.get('state')
    area = float(data.get('area'))  # Ensure the area is treated as a numeric value

    print(f"Incoming data: crop={crop}, season={season}, state={state}, area={area}")

    # Create a DataFrame for the input
    input_df = pd.DataFrame([[crop, season, state]], columns=['Crop', 'Season', 'State'])
    print("Input DataFrame:")
    print(input_df)

    # Encode categorical variables
    input_encoded = encoder.transform(input_df)
    print("Encoded DataFrame:")
    print(pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out()))

    # Combine encoded data with the area value
    input_final = np.concatenate([input_encoded, np.array([[area]])], axis=1)

    # Predict the yield
    prediction = model.predict(input_final)
    predicted_yield = float(prediction[0])
    print(f"Predicted Yield: {predicted_yield}")

    # Return the prediction as a JSON response
    return jsonify({'predicted_yield': round(predicted_yield, 2)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
