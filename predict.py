import joblib
import numpy as np

model = joblib.load("house_price_model.pkl")

area = 1800
bedrooms = 3

prediction = model.predict([[area, bedrooms]])

print(f"Predicted price for a {area} sqft, {bedrooms}-bedroom house: {prediction[0]}")

