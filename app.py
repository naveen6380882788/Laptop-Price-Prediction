import gradio as gr
import pandas as pd
from sklearn.linear_model import LinearRegression


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("laptop_price_100.csv")


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

X = df[["RAM", "Storage", "Processor", "Screen_Size"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)


# =========================================================
# CUSTOM CSS
# =========================================================

custom_css = """

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif !important;
}


/* Background */

body,
.gradio-container {
    background: linear-gradient(
        -45deg,
        #0f0c29,
        #302b63,
        #24243e,
        #1a1a2e
    ) !important;

    background-size: 400% 400% !important;

    animation: gradientShift 18s ease infinite !important;

    min-height: 100vh;

    color: #e6e8f0 !important;
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


/* Hero */

#hero {

    text-align: center;

    padding: 55px 25px 45px 25px;

    border-radius: 24px;

    background: rgba(255,255,255,0.04);

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.10);

    box-shadow: 0 8px 40px rgba(106,17,203,0.25);

    margin-bottom: 28px;

}


#hero h1 {

    font-size: 2.6rem;

    font-weight: 800;

    margin: 0;

    background: linear-gradient(
        90deg,
        #00d4ff,
        #a855f7,
        #ec4899,
        #00d4ff
    );

    background-size: 300% 100%;

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    animation: textShine 6s linear infinite;

}


@keyframes textShine {

    0% {
        background-position: 0% 50%;
    }

    100% {
        background-position: 300% 50%;
    }

}


#hero p {

    font-size: 1.05rem;

    margin-top: 12px;

    color: #b8bcd0;

}


/* Glass Card */

.glass-card {

    background: rgba(255,255,255,0.05) !important;

    backdrop-filter: blur(16px);

    border: 1px solid rgba(255,255,255,0.10) !important;

    border-radius: 20px !important;

    padding: 26px !important;

    box-shadow: 0 8px 32px rgba(0,0,0,0.35);

}


/* Section title */

.section-title {

    font-size: 1.15rem;

    font-weight: 700;

    color: #00d4ff;

    margin-bottom: 16px;

    padding-bottom: 10px;

    border-bottom: 1px dashed rgba(0,212,255,0.25);

    text-transform: uppercase;

}


/* Labels */

.gradio-container label,
.gradio-container .label-wrap span {

    color: #00e5ff !important;

    font-weight: 600 !important;

    font-size: 0.95rem !important;

}


/* Dropdown */

.gradio-container .wrap-inner,
.gradio-container .secondary-wrap,
.gradio-container [data-testid="dropdown"],
.gradio-container [role="combobox"] {

    background: #1a1a2e !important;

    background-color: #1a1a2e !important;

    border: 1.5px solid #00d4ff !important;

    border-radius: 12px !important;

    color: #ffffff !important;

}


.gradio-container input,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container input[type="search"],
.gradio-container [role="combobox"] input,
.gradio-container [role="combobox"] span {

    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    background: #1a1a2e !important;

    font-weight: 600 !important;

}


.gradio-container svg {

    color: #00d4ff !important;

    fill: #00d4ff !important;

    stroke: #00d4ff !important;

}


/* Dropdown popup */

.gradio-container ul.options,
.gradio-container div[role="listbox"] {

    background: #1a1a2e !important;

    border: 1px solid #00d4ff !important;

    border-radius: 12px !important;

    color: #ffffff !important;

}


.gradio-container div[role="option"] {

    color: #ffffff !important;

    background: #1a1a2e !important;

    padding: 10px 14px !important;

}


.gradio-container div[role="option"]:hover {

    background: rgba(0,212,255,0.25) !important;

    color: #00e5ff !important;

}


/* Predict button */

#predict-btn {

    background: linear-gradient(
        90deg,
        #6a11cb 0%,
        #2575fc 50%,
        #00d4ff 100%
    ) !important;

    color: #ffffff !important;

    font-size: 1.15rem !important;

    font-weight: 700 !important;

    padding: 16px 22px !important;

    border-radius: 14px !important;

    border: none !important;

    box-shadow: 0 8px 24px rgba(37,117,252,0.45);

    width: 100%;

}


#predict-btn:hover {

    transform: translateY(-3px);

    box-shadow: 0 14px 34px rgba(0,212,255,0.55);

}


/* Reset button */

#reset-btn {

    background: transparent !important;

    color: #00d4ff !important;

    font-size: 1rem !important;

    font-weight: 600 !important;

    padding: 13px 22px !important;

    border-radius: 14px !important;

    border: 1.5px solid #00d4ff !important;

    width: 100%;

    margin-top: 10px;

}


#reset-btn:hover {

    background: rgba(0,212,255,0.15) !important;

}


/* Result box */

#result-box {

    background: linear-gradient(
        135deg,
        rgba(0,176,155,0.20),
        rgba(150,201,61,0.12)
    );

    border: 1.5px solid #00d4ff;

    border-radius: 22px;

    padding: 55px 28px;

    text-align: center;

    box-shadow: 0 0 40px rgba(0,212,255,0.25);

    min-height: 220px;

    display: flex;

    flex-direction: column;

    justify-content: center;

}


#result-box h2 {

    margin: 0;

    font-size: 1.15rem;

    font-weight: 600;

    color: #a0f0e0;

    text-transform: uppercase;

}


#result-box .price {

    font-size: 3rem;

    font-weight: 800;

    margin-top: 18px;

    color: #00d4ff;

}


/* Footer */

#footer {

    text-align: center;

    margin-top: 30px;

    padding: 18px;

    color: #7a7f95;

    font-size: 0.88rem;

}


#footer span {

    color: #00d4ff;

    font-weight: 600;

}


/* Mobile */

@media (max-width: 768px) {

    #hero h1 {
        font-size: 1.8rem;
    }

    #result-box .price {
        font-size: 2.2rem;
    }

}

"""


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_price(ram, storage, processor, screen_size):

    new_laptop = pd.DataFrame({

        "RAM": [float(ram)],

        "Storage": [float(storage)],

        "Processor": [float(processor)],

        "Screen_Size": [float(screen_size)]

    })

    price = model.predict(new_laptop)[0]

    return f"""
    <div id="result-box">

        <h2>💰 Predicted Laptop Price</h2>

        <div class="price">₹ {price:,.2f}</div>

    </div>
    """


# =========================================================
# RESET FUNCTION
# =========================================================

def reset_all():

    return (
        4,
        128,
        3,
        14.0,
        """
        <div id="result-box">

            <h2>💰 Predicted Laptop Price</h2>

            <div class="price">₹ --</div>

        </div>
        """
    )


# =========================================================
# GRADIO WEBSITE
# =========================================================

with gr.Blocks(
    css=custom_css,
    title="Laptop Price Predictor",
    theme=gr.themes.Base(
        neutral_hue="slate",
        primary_hue="cyan"
    )
) as demo:


    # Hero

    gr.HTML("""
        <div id="hero">

            <h1>💻 Laptop Price Predictor</h1>

            <p>
                Machine Learning Based Laptop Price Prediction System
            </p>

        </div>
    """)


    # Main area

    with gr.Row(equal_height=False):


        # Input section

        with gr.Column(
            scale=1,
            elem_classes="glass-card"
        ):

            gr.HTML(
                '<div class="section-title">'
                '⚡ Enter Specifications'
                '</div>'
            )


            ram_in = gr.Dropdown(
                choices=[4, 8, 16, 32],
                value=4,
                label="🧠 RAM (GB)"
            )


            storage_in = gr.Dropdown(
                choices=[128, 256, 512, 1024, 2048, 4096],
                value=128,
                label="💾 Storage (GB)"
            )


            processor_in = gr.Dropdown(
                choices=[3, 5, 7, 9],
                value=3,
                label="⚙️ Processor"
            )


            screen_in = gr.Dropdown(
                choices=[14.0, 15.6, 16.0, 17.3],
                value=14.0,
                label="🖥️ Screen Size (inches)"
            )


            predict_btn = gr.Button(
                "🔮 Predict Laptop Price",
                elem_id="predict-btn"
            )


            reset_btn = gr.Button(
                "♻️ Reset",
                elem_id="reset-btn"
            )


        # Result section

        with gr.Column(scale=1):

            result_html = gr.HTML("""
                <div id="result-box">

                    <h2>💰 Predicted Laptop Price</h2>

                    <div class="price">₹ --</div>

                </div>
            """)


    # Footer

    gr.HTML("""
        <div id="footer">

            Powered by
            <span>Linear Regression</span>
            · Built with
            <span>Gradio</span>

        </div>
    """)


    # Predict button action

    predict_btn.click(

        fn=predict_price,

        inputs=[
            ram_in,
            storage_in,
            processor_in,
            screen_in
        ],

        outputs=result_html

    )


    # Reset button action

    reset_btn.click(

        fn=reset_all,

        inputs=None,

        outputs=[
            ram_in,
            storage_in,
            processor_in,
            screen_in,
            result_html
        ]

    )


# =========================================================
# LAUNCH
# =========================================================

demo.launch()
