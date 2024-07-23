from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

#load the trained model and scaler
model = joblib.load('disease_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = [data['RBC'], data['WBC'], data['Hemoglobin'], data['Platelets']]
    input_data = scaler.transform([input_data])
    prediction = model.predict(input_data)
    return jsonify({'Disease': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
