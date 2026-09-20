import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="centered"
)

# -----------------------------
# Custom Title
# -----------------------------
st.title("💻 Laptop Price Prediction")
st.write("Enter your laptop specifications to predict the price.")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("laptop_price_100.csv")
    return df

df = load_data()

# -----------------------------
# Train Model
# -----------------------------
X = df[["RAM", "Storage", "Processor", "Screen_Size"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📝 Laptop Specifications")

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
    "Screen Size (inch)",
    [14.0, 15.6, 16.0, 17.3]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Laptop Price"):

    new_laptop = pd.DataFrame({
        "RAM": [ram],
        "Storage": [storage],
        "Processor": [processor],
        "Screen_Size": [screen_size]
    })

    prediction = model.predict(new_laptop)[0]

    st.success(
        f"💰 Predicted Laptop Price: ₹{prediction:,.2f}"
    )

# -----------------------------
# Dataset Information
# -----------------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df)

st.markdown("---")
st.caption("Laptop Price Prediction using Machine Learning")