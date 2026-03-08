import os
import logging
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load scaler and model safely
try:
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Machine Learning model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {e}")
    scaler = None
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_template("result.html", prediction_text="Server Error: ML Model is not available. Please contact administrator.")
        
    try:
        # Define expected fields from the form
        fields = [
            'age_days', 'gender', 'height', 'weight', 'ap_hi', 
            'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]
        
        # Check for missing forms fields
        missing_fields = [f for f in fields if request.form.get(f) is None or request.form.get(f).strip() == ""]
        if missing_fields:
            return render_template("result.html", prediction_text=f"Validation Error: Missing required fields: {', '.join(missing_fields)}")
        
        # Parse inputs safely
        id_val = 0 # Default random ID since it doesn't matter for prediction (the model expects 12 features)
        age = float(request.form.get('age_days')) 
        gender = float(request.form.get('gender'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        ap_hi = float(request.form.get('ap_hi'))
        ap_lo = float(request.form.get('ap_lo'))
        cholesterol = float(request.form.get('cholesterol'))
        gluc = float(request.form.get('gluc'))
        smoke = float(request.form.get('smoke'))
        alco = float(request.form.get('alco'))
        active = float(request.form.get('active'))
        
        # Order must exactly match scaler's expected features:
        # ['id' 'age' 'gender' 'height' 'weight' 'ap_hi' 'ap_lo' 'cholesterol' 'gluc' 'smoke' 'alco' 'active']
        input_features = [id_val, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]
        
        # Convert to numpy array
        final_input = np.array([input_features])
        
        # Apply normalization using scaler
        final_input_scaled = scaler.transform(final_input)
        
        # Make prediction
        prediction = model.predict(final_input_scaled)

        # Interpret the prediction (1 -> High Risk, 0 -> Low Risk)
        if prediction[0] == 1:
            result = "High Risk of Cardiovascular Disease"
        else:
            result = "Low Risk of Cardiovascular Disease"
            
    except ValueError as val_err:
        logger.error(f"Value Parsing Error: {str(val_err)}")
        result = "Data Error: Invalid input data format. Please only enter numerical values."
    except Exception as e:
        logger.error(f"Prediction Pipeline Error: {str(e)}")
        result = "System Error: Unable to process input at this moment. Please try again later."
        
    return render_template("result.html", prediction_text=result)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for a professional backend setup. 
    Accepts application/json requests.
    """
    if model is None or scaler is None:
        return jsonify({"error": "Service temporarily unavailable: ML Model is not active."}), 503
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request: No JSON payload provided."}), 400
            
        fields = [
            'age_years', 'gender', 'height', 'weight', 'ap_hi', 
            'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]
        
        # Verify required API fields
        missing_fields = [f for f in fields if f not in data]
        if missing_fields:
            return jsonify({"error": f"Bad Request: Missing required data fields: {', '.join(missing_fields)}"}), 400
            
        id_val = data.get('id', 0)
        # Convert years to days internally to match dataset expected logic
        age_days = float(data['age_years']) * 365.25 
        
        input_features = [
            id_val, age_days, float(data['gender']), float(data['height']), 
            float(data['weight']), float(data['ap_hi']), float(data['ap_lo']), 
            float(data['cholesterol']), float(data['gluc']), float(data['smoke']), 
            float(data['alco']), float(data['active'])
        ]
        
        final_input = np.array([input_features])
        final_input_scaled = scaler.transform(final_input)
        prediction = model.predict(final_input_scaled)
        
        # Structure the response
        risk_level = "High" if prediction[0] == 1 else "Low"
        
        return jsonify({
            "status": "success",
            "prediction": int(prediction[0]),
            "risk_level": risk_level,
            "message": "Prediction evaluated successfully."
        }), 200
        
    except ValueError:
        return jsonify({"error": "Bad Request: Ensure all expected values are numerical."}), 400
    except Exception as e:
        logger.error(f"API Prediction Exception: {str(e)}")
        return jsonify({"error": "Internal Server Error: Unexpected issue during model evaluation."}), 500

if __name__ == "__main__":
    # In a professional setting, debug should be controlled by env variables, but we leave it True for user testing
    app.run(host="0.0.0.0", port=5000, debug=True)