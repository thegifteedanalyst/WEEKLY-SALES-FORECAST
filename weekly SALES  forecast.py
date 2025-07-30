import streamlit as st
import pandas as pd
from PIL import Image
import pickle
from datetime import timedelta


# Set Streamlit page config
st.set_page_config(page_title="🛒 Supermarket Sales Forecast", layout="wide")

# Load the trained model
with open('weekly_forecast_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Display header image (optional)
try:
    image = Image.open("supermarket.png")
    st.image(image, caption="Predict Tomorrow's Revenue Today!", use_container_width=True)
except:
    pass

# Title and instructions
st.title("📈 Supermarket Sales Forecast")
st.markdown("Upload your daily sales CSV file. Columns required: **Order Date** and **Total Revenue**")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Daily Sales Data (.csv)", type=["csv"])

# Load and prepare data
def load_data(file):
    df = pd.read_csv(file)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df = df.sort_values("Order Date")
    return df

# Forecast future sales using the same feature used during training: 'Day'
def predict_sales(df, forecast_days=7):
    last_date = df["Order Date"].max()
    min_date = df["Order Date"].min()
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_days = [(d - min_date).days for d in future_dates]
    
    features = pd.DataFrame({"Day": future_days})
    predicted_revenue = model.predict(features)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Revenue": predicted_revenue
    })
    return forecast_df

# Main logic
if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📊 Historical Sales Trend")
    st.line_chart(df.set_index("Order Date")["Total Revenue"])

    forecast_days = st.slider("📅 Select Forecast Period", 3, 30, 7)

    if st.button("🔮 Generate Forecast"):
        forecast = predict_sales(df, forecast_days)

        st.subheader(f"📅 Forecast for Next {forecast_days} Days")
        st.dataframe(forecast)

        # Combine for visualization
        combined_df = pd.concat([
            df[["Order Date", "Total Revenue"]].rename(columns={"Order Date": "Date", "Total Revenue": "Revenue"}),
            forecast.rename(columns={"Date": "Date", "Predicted Revenue": "Revenue"})
        ])
        combined_df["Label"] = ["Actual"] * len(df) + ["Forecast"] * len(forecast)

        # Plot actual vs forecast
        fig, ax = plt.subplots(figsize=(10, 4))
        for label, group in combined_df.groupby("Label"):
            ax.plot(group["Date"], group["Revenue"], label=label)
        ax.set_title("📉 Actual vs Forecasted Revenue")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue")
        ax.legend()
        st.pyplot(fig)

        # Forecast summary
        with st.expander("🔍 Forecast Summary"):
            st.json({
                "Start Date": str(forecast['Date'].min().date()),
                "End Date": str(forecast['Date'].max().date()),
                "Average Forecasted Revenue": round(forecast["Predicted Revenue"].mean(), 2),
                "Total Forecasted Revenue": round(forecast["Predicted Revenue"].sum(), 2)
            })
