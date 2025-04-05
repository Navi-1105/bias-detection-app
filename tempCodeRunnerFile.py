import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from io import StringIO

# --- App config ---
st.set_page_config(
    page_title="Bias Detection App",
    page_icon="🧠",
    layout="centered",
)

# --- Load model & tokenizer ---
@st.cache_resource
def load_model():
    model_path = "./bias_detection_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to("cpu").eval()
    return model, tokenizer

model, tokenizer = load_model()

# --- Bias prediction function ---
def predict_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    return "🟥 Biased" if prediction == 1 else "🟩 Not Biased"

# --- Custom CSS for aesthetics ---
st.markdown("""
    <style>
        .stRadio > div {
            flex-direction: row;
        }
        .main {
            background-color: #f8f9fa;
        }
        .title {
            text-align: center;
            font-size: 36px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #666;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title and instructions ---
st.markdown("<div class='title'>🧠 Bias Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect bias in individual sentences or upload a CSV file</div>", unsafe_allow_html=True)

# --- Mode selection ---
mode = st.radio("Choose input mode:", ["📄 Single Sentence", "📁 Upload CSV File"], horizontal=True)

# --- Single sentence mode ---
if mode == "📄 Single Sentence":
    st.markdown("#### ✍️ Enter a sentence:")
    input_text = st.text_area("")

    if st.button("🔍 Detect Bias"):
        if input_text.strip():
            with st.spinner("Analyzing..."):
                prediction = predict_bias(input_text)
            st.success("Prediction complete!")
            st.markdown(f"### 🧾 Result: <span style='font-size:24px'>{prediction}</span>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a sentence to analyze.")

# --- CSV file upload mode ---
else:
    st.markdown("#### 📁 Upload a CSV with a `text` column:")
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("❌ The CSV must contain a column named 'text'.")
            else:
                with st.spinner("Analyzing the CSV..."):
                    df["Prediction"] = df["text"].apply(predict_bias)
                st.success("✅ Analysis complete!")
                st.dataframe(df[["text", "Prediction"]], use_container_width=True)

                # --- Download button ---
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="bias_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
