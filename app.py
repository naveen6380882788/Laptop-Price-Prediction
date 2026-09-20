import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

/* Animated dark background */
.stApp {
    background: linear-gradient(
        -45deg,
        #0f0c29,
        #302b63,
        #24243e,
        #1a1a2e
    );
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    color: #e6e8f0;
}

@keyframes gradientShift {
    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}

/* Main container */
.block-container {
    max-width: 850px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Main title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #b8c1ec;
    font-size: 16px;
    margin-bottom: 35px;
}

/* Input labels */
label {
    color: #dfe5ff !important;
    font-weight: 600 !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 12px !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(90deg, #00c6ff, #7b2ff7);
    color: white;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,198,255,0.35);
}

/* Prediction result */
.result-box {
    margin-top: 25px;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.result-title {
    color: #b8c1ec;
    font-size: 16px;
}

.result-price {
    color: #00e5ff;
    font-size: 38px;
    font-weight: 800;
}

/* =========================
   CREATOR DESIGN
   ========================= */

.creator-label {
    text-align: center;
    color: #d8d8ff;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 7px;
    margin-top: 60px;
    margin-bottom: 10px;
}

.creator-name {
    text-align: center;
    color: #00e5ff;
    font-size: 38px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 15px;
}

.creator-project {
    text-align: center;
    color: #aeb6d4;
    font-size: 14px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("laptop_price_100.csv")


df = load_data()


# =========================
# TRAIN MODEL
# =========================
X = df[[
    "RAM",
    "Storage",
    "Processor",
    "Screen_Size"
]]

y = df["Price"]

model = LinearRegression()
model.fit(X, y)


# =========================
# HEADER
# =========================
st.markdown(
    '<div class="title">💻 Laptop Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based Laptop Price Prediction System'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# INPUT SECTION
# =========================
st.markdown("### ⚙️ Laptop Specifications")

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


# =========================
# PREDICT
# =========================
if st.button("🔮 Predict Laptop Price"):

    new_laptop = pd.DataFrame({
        "RAM": [ram],
        "Storage": [storage],
        "Processor": [processor],
        "Screen_Size": [screen_size]
    })

    prediction = model.predict(new_laptop)[0]

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">
                Estimated Laptop Price
            </div>
            <div class="result-price">
                ₹ {prediction:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# CREATOR
# =========================

st.markdown(
    '<div class="creator-label">CREATED BY</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="creator-name">NAVEEN RAJ</div>',
    unsafe_allow_html=True
)

st.divider()

st.markdown(
    '<div class="creator-project">'
    'Laptop Price Prediction Using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
