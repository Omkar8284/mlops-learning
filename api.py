from fastapi import FastAPI
import joblib
import numpy as np

# Load the saved model
model = joblib.load("house_price_model.pkl")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "House Price Prediction API is running!"}

@app.get("/predict")
def predict(area: float, bedrooms: int):
    # Predict using the loaded model
    input_features = np.array([[area, bedrooms]])
    price = model.predict(input_features)[0]
    return {"predicted_price": float(price)}

