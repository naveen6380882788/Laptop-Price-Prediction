import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="centered"
)

# Load dataset
df = pd.read_csv("laptop_price_100.csv")

# Features and target
X = df[["RAM", "Storage", "Processor", "Screen_Size"]]
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Title
st.title("💻 Laptop Price Prediction")
st.write("Enter laptop specifications to predict the price.")

st.divider()

# Input fields
ram = st.selectbox(
    "RAM (GB)",
    [4, 8, 16, 32]
)

storage = st.selectbox(
    "Storage (GB)",
    [128, 256, 512, 1024, 2048, 4096]
)

processor = st.selectbox(
    "Processor",
    [3, 5, 7, 9]
)

screen_size = st.selectbox(
    "Screen Size (inches)",
    [14.0, 15.6, 16.0, 17.3]
)

# Predict button
if st.button("🔮 Predict Price", use_container_width=True):

    new_laptop = pd.DataFrame({
        "RAM": [ram],
        "Storage": [storage],
        "Processor": [processor],
        "Screen_Size": [screen_size]
    })

    prediction = model.predict(new_laptop)[0]

    st.success(f"💰 Predicted Laptop Price: ₹{prediction:,.2f}")

# Dataset information
st.divider()
st.subheader("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Features", 4)

st.caption("Laptop Price Prediction using Machine Learning")
