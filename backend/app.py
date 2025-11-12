from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from predict import predict_parkinsons

app = Flask(__name__)
CORS(app) 

@app.route('/')
def home():
    return jsonify({"message": "Parkinson’s Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Run prediction
        result = predict_parkinsons(tmp_path)
        os.remove(tmp_path)
        return jsonify({
            "prediction": "Parkinson’s Detected" if result["prediction"] == 1 else "Healthy",
            "risk_score": result["risk_score"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
