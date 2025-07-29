import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load your historical sales data
df = pd.read_csv('Desktop/DATA TYPES/100 Sales Records.csv')
df["Order Date"] = pd.to_datetime(df["Order Date"])
df = df.sort_values("Order Date")
df["Day"] = (df["Order Date"] - df["Order Date"].min()).dt.days

# Features and target
X = df[["Day"]]
y = df["Total Revenue"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open("weekly_sales_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open('weekly_sales_model.pkl', 'rb') as file:
    model = pickle.load(file)

# When predicting:
future_days = pd.DataFrame({"Day": [df["Day"].max() + i for i in range(1, 8)]})
forecast = model.predict(future_days)
import pickle

with open('weekly_sales_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(type(model))