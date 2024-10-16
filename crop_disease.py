from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import traceback
import pandas as pd
from pymongo import MongoClient  # Import MongoDB client

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB connection
MONGO_URI = "mongodb+srv://221501139:gMf5Whuz8VPd9O1r@agri.xukd7.mongodb.net/"  # Replace with your MongoDB URI
client = MongoClient(MONGO_URI)
db = client["agri"]  # Replace with your database name
disease_collection = db["disease"]

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def load_class_labels(excel_path, crop_name):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file {excel_path} not found.")
    
    df = pd.read_excel(excel_path)
    if 'Crop' not in df.columns or 'Class Labels' not in df.columns:
        raise ValueError("Excel file does not contain required columns: 'Crop' and 'Class Labels'.")
    
    class_labels = df[df['Crop'] == crop_name]['Class Labels'].values
    if len(class_labels) > 0:
        return class_labels[0].split(', ')
    else:
        raise ValueError(f"No class labels found for crop: {crop_name}")

def load_trained_model(crop):
    model_path = f'{crop}_best_model.keras'
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Modify as needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_crop_disease(model, img_array):
    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        crop = request.form.get('crop')
        img = request.files.get('image')

        if not crop or not img:
            return jsonify({'error': 'Crop name and image are required'}), 400

        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        img_path = os.path.join(upload_folder, img.filename)
        img.save(img_path)

        excel_path = os.path.join('src', 'views', 'examples', 'class_labels.xlsx')
        class_labels = load_class_labels(excel_path, crop)

        model = load_trained_model(crop)
        img_array = preprocess_image(img_path)
        predicted_class = predict_crop_disease(model, img_array)
        predicted_label = class_labels[predicted_class[0]]

        # Query MongoDB for disease details
        disease_details = disease_collection.find_one({"disease": predicted_label})
        
        if not disease_details:
            return jsonify({'predicted_class': predicted_label, 'message': 'No additional details found for this disease.'}), 200

        # Format the response with MongoDB data
        response_data = {
            'predicted_class': predicted_label,
            'preventive_measures': disease_details.get('preventive_measures', []),
            'treatments': {
                'chemical': [treatment for treatment in disease_details.get('treatments', []) if treatment.get('type') == 'Chemical'],
                'biological': [treatment for treatment in disease_details.get('treatments', []) if treatment.get('type') == 'Biological']
            }
        }

        return jsonify(response_data)

    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        print(f"Internal Server Error: {error_message}")
        print(traceback_str)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True, port=4000)
