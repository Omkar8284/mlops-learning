from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("house_price_model.pkl")

app = FastAPI()

@app.get("/") 
def root():
    return {"message": "House Price Prediction API is running!"}

@app.get("/predict")
def predict(area: float, bedrooms: int):
    input_features = np.array([[area, bedrooms]])
    price = model.predict(input_features)[0]
    return {"predicted_price": float(price)}

