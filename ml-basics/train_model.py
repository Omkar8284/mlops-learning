import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

data = pd.DataFrame({
    "area": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4],
    "price": [70_000, 120_000, 150_000, 180_000, 220_000]
})

X = data[["area", "bedrooms"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("Model accuracy:", score)

joblib.dump(model, "house_price_model.pkl")

print("Model saved successfully.")

