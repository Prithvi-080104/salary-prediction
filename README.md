# Salary Prediction Project

A full-stack ML application that predicts employee salary based on profile attributes.

## Architecture

```
salary_prediction/
├── data/
│   └── salary_data.csv          # Dataset
├── model/
│   ├── train_model.py           # ML training script (Gradient Boosting)
│   ├── salary_model.pkl         # Saved model (generated after training)
│   └── feature_meta.json        # Dropdown options & ranges (generated)
├── api/
│   └── app.py                   # Flask REST API
├── frontend/
│   └── streamlit_app.py         # Streamlit UI
├── requirements.txt
└── run.ps1                      # One-shot launcher (Windows)
```

## Features

| Feature | Details |
|---|---|
| **Model** | Gradient Boosting Regressor (scikit-learn) |
| **Inputs** | Education, Experience, Location, Job Title, Age, Gender |
| **Output** | Predicted annual salary (USD) |
| **API** | Flask REST – `POST /predict`, `GET /meta`, `GET /health` |
| **Frontend** | Streamlit with Plotly charts |

## Quick Start

### 1. Install dependencies
```powershell
cd salary_prediction
pip install -r requirements.txt
```

### 2. Train the model
```powershell
python model/train_model.py
```
Outputs:
- `model/salary_model.pkl` — trained pipeline
- `model/feature_meta.json` — allowed values for dropdowns

### 3. Start the Flask API
```powershell
python api/app.py
# Running on http://127.0.0.1:5000
```

### 4. Launch the Streamlit frontend
```powershell
streamlit run frontend/streamlit_app.py
# Opens http://localhost:8501
```

### Or run everything at once (PowerShell)
```powershell
.\run.ps1
```

## API Reference

### `POST /predict`
```json
{
  "Education":  "Bachelor",
  "Experience": 5,
  "Location":   "Urban",
  "Job_Title":  "Engineer",
  "Age":        28,
  "Gender":     "Male"
}
```
Response:
```json
{
  "predicted_salary": 95234.50,
  "currency": "USD",
  "inputs": { ... }
}
```

### `GET /meta`
Returns allowed dropdown values and numeric ranges.

### `GET /health`
Returns `{ "status": "ok" }`.
