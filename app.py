from flask import Flask, request, jsonify, send_from_directory
import joblib
import os

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('disease_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Serve the index.html file
@app.route('/')
def index():
    return send_from_directory('', 'index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = [
        data['WBC'], data['LYMp'], data['MIDp'], data['NEUTp'],
        data['LYMn'], data['MIDn'], data['NEUTn'], data['RBC'],
        data['HGB'], data['HCT'], data['MCV'], data['MCH'],
        data['MCHC'], data['RDWSD'], data['RDWCV'], data['PLT'],
        data['MPV'], data['PDW'], data['PCT'], data['PLCR']
    ]
    input_data = scaler.transform([input_data])
    prediction = model.predict(input_data)
    return jsonify({'Disease': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
