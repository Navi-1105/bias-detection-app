import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Bias Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Load model/tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./bias_detection_model")
    tokenizer = AutoTokenizer.from_pretrained("./bias_detection_model")
    model.to("cpu").eval()
    return model, tokenizer

model, tokenizer = load_model()

# Predict function
def predict_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    return "Biased" if prediction == 1 else "Not Biased"

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Single Text", "📁 Bulk CSV", "📊 Summary"])

st.sidebar.markdown("---")
st.sidebar.write("Made by Navneet Kaur ")

# --- App Title ---
st.markdown("<h1 style='text-align: center;'>🧠 Bias Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Analyze texts or datasets for biased language</h4><br>", unsafe_allow_html=True)

# Global session storage
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None


# --- MODE 1: Single sentence ---
if app_mode == "🏠 Single Text":
    st.subheader("✍️ Analyze a Single Sentence")
    input_text = st.text_area("Enter text to analyze:", height=100)

    if st.button("🔍 Detect Bias"):
        if input_text.strip():
            with st.spinner("Analyzing..."):
                result = predict_bias(input_text)
            st.success(f"✅ Prediction: **{result}**")
        else:
            st.warning("Please enter some text.")

# --- MODE 2: Bulk CSV file ---
elif app_mode == "📁 Bulk CSV":
    st.subheader("📥 Upload CSV with `text` column")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("The CSV must have a column named `text`.")
            else:
                with st.spinner("Analyzing..."):
                    df["Prediction"] = df["text"].apply(predict_bias)
                    st.session_state.csv_data = df

                st.success("✅ Analysis Complete!")
                st.dataframe(df[["text", "Prediction"]], use_container_width=True)

                # Download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- MODE 3: Summary / Dashboard ---
elif app_mode == "📊 Summary":
    st.subheader("📊 Prediction Summary")

    if st.session_state.csv_data is None:
        st.warning("No CSV data found. Please upload a CSV in the 'Bulk CSV' section first.")
    else:
        df = st.session_state.csv_data
        bias_counts = df["Prediction"].value_counts()

        # Grid layout: 2 columns
        col1, col2 = st.columns(2)

        with col1:
            st.metric("🔴 Biased", int(bias_counts.get("Biased", 0)))

        with col2:
            st.metric("🟢 Not Biased", int(bias_counts.get("Not Biased", 0)))

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(
            bias_counts,
            labels=bias_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=["#e74c3c", "#2ecc71"]
        )
        ax.axis("equal")
        st.pyplot(fig)

        with st.expander("📋 View Full Table"):
            st.dataframe(df[["text", "Prediction"]])
