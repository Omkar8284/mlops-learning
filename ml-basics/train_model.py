import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# 1) Load sample data
data = pd.DataFrame({
    "area": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4],
    "price": [70_000, 120_000, 150_000, 180_000, 220_000]
})

# 2) Separate features and target
X = data[["area", "bedrooms"]]
y = data["price"]

# 3) Train-test split (model learns on some data, we test on rest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Create the model
model = LinearRegression()

# 5) Train the model (model learns the pattern)
model.fit(X_train, y_train)

# 6) Evaluate the model
score = model.score(X_test, y_test)
print("Model accuracy:", score)

# 7) Save model for later use
joblib.dump(model, "house_price_model.pkl")

print("Model saved successfully.")

