import joblib
import pandas as pd

# --- 1. Load your saved model ---
model = joblib.load("house_price_model.joblib")

# --- 2. Define a helper function for prediction ---
def predict_house_price(bedrooms, bathrooms, square_feet, city):
    # Takes raw input values and returns a predicted house price.

    #Create a DataFrame for a single example
    new_data = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "square_feet": [square_feet],
        "city": [city]
    })

    # Run the model Pipeline (it handles encoding and scaling internally)
    predicted_price = model.predict(new_data)[0]

    # Return nicely formatted output
    return round(predicted_price, 2)

# --- 3. Try it out ---
if __name__ == "__main__":
    price = predict_house_price(4, 3, 2000, "Boston")
    print(f"🏠 Predicted Price for Boston home:  ${price:,}")

    price = predict_house_price(3, 2, 1500, "New York")
    print(f"🏠 Predicted Price for New York home:  ${price:,}")

    price = predict_house_price(5, 4, 3200, "Los Angeles")
    print(f"🏠 Predicted Price for Los Angeles home:  ${price:,}")
    