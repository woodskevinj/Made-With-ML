from flask import Flask, request, jsonify
import joblib
import pandas as pd

# --- 1. Initialize Flask app ---
app = Flask(__name__)

# --- 2. Load the trained model ---
model = joblib.load("house_price_model.joblib")

# --- 3. Define prediction route ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON
        data = request.get_json(force=True)

        # Create DataFrame from input JSON
        new_data = pd.DataFrame([data])

        # Predict using the trained pipeline
        predicted_price = model.predict(new_data)[0]

        # Return JSON response
        return jsonify({
            "predicted_price": round(float(predicted_price), 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
# --- 4. Run the app ---
if __name__ == "__main__":
    app.run(debug=True)