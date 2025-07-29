import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from PIL import Image

# Load the model
with open('weekly_sales_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(page_title="🛒 Supermarket Sales Forecast", layout="wide")

# Load image (optional - add 'supermarket.png' to your folder)
try:
    image = Image.open("supermarket.png")
    st.image(image, caption="Predict Tomorrow's Revenue Today!", use_container_width=True)
except:
    pass  # Skip image if not found

# App title
st.title("📈 Supermarket Sales Forecast")
st.markdown("""
This interactive dashboard predicts the **next 7 days of total revenue** based on historical daily sales data using a **Linear Regression** model.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Desktop/DATA TYPES/100 Sales Records.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df = df.groupby("Order Date")[["Total Revenue"]].sum().reset_index()
    df = df.sort_values("Order Date")
    df["Days Since"] = (df["Order Date"] - df["Order Date"].min()).dt.days
    return df

# Predict future revenue
def predict_sales(df, forecast_days):
    X = df[["Days Since"]]
    y = df["Total Revenue"]

    model = LinearRegression()
    model.fit(X, y)

    last_day = df["Days Since"].max()
    future_days = pd.DataFrame({"Days Since": range(last_day+1, last_day+1+forecast_days)})
    predictions = model.predict(future_days)
    future_dates = [df["Order Date"].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Revenue": predictions
    })

    return forecast_df

# Load data
df = load_data()

# UI layout
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 Historical Sales Trend")
    st.line_chart(df.set_index("Order Date")["Total Revenue"])

with col2:
    forecast_days = st.slider("📅 Select Forecast Period (Days)", min_value=3, max_value=30, value=7)

# Prediction trigger
if st.button("🔮 Generate Forecast"):
    forecast = predict_sales(df, forecast_days)

    st.markdown("---")
    st.subheader("📅 Predicted Revenue for Next {} Days".format(forecast_days))
    st.dataframe(forecast)

    # Combined chart
    combined_df = pd.concat([
        df[["Order Date", "Total Revenue"]].rename(columns={"Order Date": "Date", "Total Revenue": "Revenue"}),
        forecast.rename(columns={"Predicted Revenue": "Revenue"})
    ])
    combined_df["Label"] = ["Actual"] * len(df) + ["Forecast"] * len(forecast)
 # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, data in combined_df.groupby("Label"):
        ax.plot(data["Date"], data["Revenue"], label=label)
    ax.set_title("📉 Actual vs Forecasted Revenue")
    ax.legend()
    st.pyplot(fig)

    with st.expander("🔍 Forecast Summary"):
        st.json({
            "Start Date": str(forecast['Date'].min().date()),
            "End Date": str(forecast['Date'].max().date()),
            "Average Forecasted Revenue": round(forecast["Predicted Revenue"].mean(), 2),
            "Total Forecasted Revenue": round(forecast["Predicted Revenue"].sum(), 2)
        })

