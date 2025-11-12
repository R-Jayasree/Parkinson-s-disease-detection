import React from "react";
import "../App.css";

function ResultCard({ result }) {
  const { prediction, risk_score, confidence, probabilities } = result;
  const color =
    prediction === "Parkinson’s Detected" ? "#f44336" : "#4caf50";

  return (
    <div className="result-card" style={{ borderColor: color }}>
      <h2>Prediction Result</h2>

      <p>
        <strong>Prediction:</strong>{" "}
        <span style={{ color }}>{prediction}</span>
      </p>

      <p>
        <strong>Risk Score:</strong>{" "}
        {risk_score ? `${risk_score.toFixed(2)}%` : "N/A"}
      </p>

      <p>
        <strong>Confidence:</strong>{" "}
        {confidence ? `${confidence.toFixed(2)}%` : "N/A"}
      </p>

      {probabilities && (
        <div className="prob-section">
          <p><strong>Probabilities:</strong></p>
          <ul>
            <li>Healthy: {(probabilities[0] * 100).toFixed(2)}%</li>
            <li>Parkinson’s: {(probabilities[1] * 100).toFixed(2)}%</li>
          </ul>
        </div>
      )}

      <p className="disclaimer">
        ⚠ This is a preliminary screening tool. Please consult a healthcare professional.
      </p>
    </div>
  );
}

export default ResultCard;
