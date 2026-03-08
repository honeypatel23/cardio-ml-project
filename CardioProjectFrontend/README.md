# CardioPredict Machine Learning Application

A modern, professionally structured full-stack ML application for assessing cardiovascular disease risk.

## Features
- **Machine Learning Inference:** Leverages a pre-trained scikit-learn model and scaler to predict the risk based on 11 health parameters.
- **Modern User Interface:** Glassmorphism UI aesthetic, dark mode gradients, fully responsive layout.
- **Robust Error Handling:** Validates numerical limits securely and catches incomplete forms safely.
- **RESTful API Endpoint:** Optionally exposes an `/api/predict` JSON endpoint for decoupled applications or mobile clients.
- **Standardized Setup:** Standardized Python application structure using `requirements.txt`.

## Getting Started

### Prerequisites
- Python 3.8+

### Installation
1. Navigate to the project directory:
   ```bash
   cd CardioProjectFrontend
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Navigate to `http://127.0.0.1:5000` in your web browser.

## API Usage
You can make POST requests to `/api/predict` with JSON payloads:
```json
{
  "age_years": 50,
  "gender": 1,
  "height": 170,
  "weight": 70,
  "ap_hi": 120,
  "ap_lo": 80,
  "cholesterol": 1,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}
```

Response format:
```json
{
  "status": "success",
  "prediction": 0,
  "risk_level": "Low",
  "message": "Prediction evaluated successfully."
}
```

## Structure
- `app.py`: Backend server logic, API integration, error catching.
- `static/style.css`: Modern visual aesthetics.
- `templates/`: HTML structure.
- `model.pkl / scaler.pkl`: Serialized intelligence and normalization tools.
