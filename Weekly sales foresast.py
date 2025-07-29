import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import timedelta
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Desktop/DATA TYPES/100 Sales Records.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df = df.groupby("Order Date")[["Total Revenue"]].sum().reset_index()
    df = df.sort_values("Order Date")
    df["Days Since"] = (df["Order Date"] - df["Order Date"].min()).dt.days
    return df

# Predict future sales
def predict_sales(df, days=7):
    X = df[["Days Since"]]
    y = df["Total Revenue"]
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 7 days
    last_day = df["Days Since"].max()
    future_days = pd.DataFrame({"Days Since": range(last_day+1, last_day+1+days)})
    future_preds = model.predict(future_days)
    future_dates = [df["Order Date"].max() + timedelta(days=i) for i in range(1, days+1)]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Revenue": future_preds
    })
    return forecast_df

# Load data and predict
df = load_data()
forecast = predict_sales(df)

# Streamlit UI
st.title("📈 Supermarket Sales Forecast Dashboard")
st.write("This dashboard predicts the next 7 days of revenue using linear regression.")

st.subheader("📊 Past Sales Data")
st.line_chart(df.set_index("Order Date")["Total Revenue"])

st.subheader("🔮 Next 7 Days Sales Prediction")
st.dataframe(forecast)

# Chart
st.line_chart(forecast.set_index("Date")["Predicted Revenue"])
