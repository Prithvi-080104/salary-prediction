"""
Flask REST API — serves salary predictions.
Endpoint: POST /predict
          GET  /meta
          GET  /health
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "salary_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")

app = Flask(__name__)
CORS(app)

# ---------- load artifacts once at startup ----------
_pipeline = None
_meta = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_model.py first."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def get_meta():
    global _meta
    if _meta is None:
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _meta


# ---------- routes ----------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/meta", methods=["GET"])
def meta():
    """Return allowed values for categorical fields and numeric ranges."""
    return jsonify(get_meta())


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON body:
    {
        "Education":  "Bachelor",
        "Experience": 5,
        "Location":   "Urban",
        "Job_Title":  "Engineer",
        "Age":        28,
        "Gender":     "Male"
    }
    """
    body = request.get_json(force=True)
    required = ["Education", "Experience", "Location", "Job_Title", "Age", "Gender"]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = pd.DataFrame(
            [
                {
                    "Education": body["Education"],
                    "Experience": float(body["Experience"]),
                    "Location": body["Location"],
                    "Job_Title": body["Job_Title"],
                    "Age": float(body["Age"]),
                    "Gender": body["Gender"],
                }
            ]
        )
        salary = float(get_pipeline().predict(row)[0])
        return jsonify(
            {
                "predicted_salary": round(salary, 2),
                "currency": "USD",
                "inputs": body,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # Ensure model is loaded before first request
    get_pipeline()
    get_meta()
    print("[Flask API] Running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
