# AI-Powered Parkinson’s Disease Detection from Voice

Detect Parkinson’s disease early using voice recordings. This repository contains a complete web application that allows users to **record their voice** or **upload an audio file**, extract relevant features, and predict the **risk of Parkinson’s disease** using a trained machine learning model. The app also supports **multilingual UI** via React i18n.

---

## Repository Structure
```
Parkinson-s-disease-detection/
│
├── backend/
│ ├── app.py # Flask API endpoints for predictions
│ ├── model_def.py # Custom ensemble model definition
│ ├── predict.py # Feature extraction & prediction script
│ ├── train_model.py # Script to train the model and save predictions.csv
│ ├── parkinsons_model.pkl # Trained ML model
│ ├── predictions.csv # Predictions on the full dataset
│ └── requirements.txt # Python dependencies
│
└── frontend/
├── src/
│ ├── App.js # Main React app
│ ├── App.css # Styling and modern UI design
│ └── locales/ # i18n translation files for multilingual support
```
---

## Backend Overview

The backend is a **Flask application** that exposes an endpoint to predict Parkinson’s disease from an audio file.

- **`app.py`**  
  Flask server.  
  - `/predict` endpoint: Accepts an uploaded `.wav` file and returns a JSON response with:
    - `prediction` – "Parkinson’s Detected" or "Healthy"
    - `risk_score` – Predicted risk percentage
    - `confidence` – Model confidence
  - Example response:
    ```json
    {
      "prediction": "Parkinson’s Detected",
      "risk_score": 89.83,
      "confidence": 89.83
    }
    ```

- **`model_def.py`**  
  Defines the custom ensemble model `ParkinsonsEnsembleModel` used for training and predictions.

- **`predict.py`**  
  Handles:
  - Audio preprocessing
  - Feature extraction (pitch, jitter, shimmer, MFCC, HNR, etc.)
  - Loading the trained model
  - Predicting the risk

- **`train_model.py`**  
  Script to train the model on the UCI Parkinson’s dataset.  
  - Splits dataset into train/test
  - Performs model training
  - Evaluates performance
  - Saves `parkinsons_model.pkl` for deployment
  - Saves predictions for the full dataset as `predictions.csv`

- **`parkinsons_model.pkl`**  
  Pre-trained machine learning model (pickle file) used by the API.

- **`predictions.csv`**  
  Contains model predictions for the full dataset including:
  - Features
  - Actual label
  - Predicted label
  - Risk score

- **`requirements.txt`**  
  Backend dependencies:
  ```text
  flask
  numpy
  pandas
  librosa
  scikit-learn
  scipy
  imbalanced-learn
Frontend Overview
  The frontend is a React application that provides a user-friendly UI for uploading or recording voice and displaying predictions.

src/App.js
  Main React component:

* Allows file upload or voice recording
* Displays animated waveform during recording
* Sends audio to backend for predictions
* Shows results with status, risk score, and suggestions
* Multilingual support via i18n

src/App.css
  Styles the app using a modern blue gradient + glassmorphism theme.

src/locales/
  JSON files containing translations for supported languages.

Setup Instructions
1. Clone the repository

git clone https://github.com/R-Jayasree/Parkinson-s-disease-detection.git
cd Parkinson-s-disease-detection
2. Backend Setup

cd backend

# Create a virtual environment (optional but recommended)
python -m venv myenv
# Activate virtual environment
# Windows
myenv\Scripts\activate
# macOS/Linux
source myenv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the Flask app
python app.py
The backend will start at http://127.0.0.1:5000/.

Test the /predict endpoint using Postman or your frontend.

3️. Frontend Setup

cd ../frontend

# Install dependencies
npm install

# Start the React app
npm start
The app will run at http://localhost:3000/.

Ensure the backend is running to receive predictions.

How to Use
* Open the React app.
* Select Upload Audio or Start Recording.
* Record or upload a voice sample in .wav format.
* Click Get Predictions.
* The app displays:  Status: Healthy / Parkinson’s Detected
* Risk Score: %
* Suggestions: Based on predicted risk

Features
* AI-powered Parkinson’s detection using voice features
* Supports file upload or real-time recording
* Multilingual user interface
* Detailed predictions with risk assessment
* Glassmorphism + blue gradient modern UI

References
1. UCI Parkinson’s Dataset: https://archive.ics.uci.edu/ml/datasets/parkinsons
2. Additional dataset: Parkinson’s Disease Classification
