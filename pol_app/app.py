import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="AI-Based Political Speech Analyzer", page_icon="🧠", layout="wide")

# Sidebar: Model selection
st.sidebar.title("⚙️ Model Options")
model_choice = st.sidebar.selectbox(
    "Choose Model Type:",
    ["Full (High Accuracy)", "Lightweight (Fast Inference)"],
    index=0
)

# Sidebar: Credits
st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 **Made by Navneet Kaur**")

# Custom styling
def add_custom_css():
    st.markdown(
        """
        <style>
        .main-title { color: #4CAF50; font-size: 36px; font-weight: bold; text-align: center; }
        .sub-title { color: #555555; font-size: 18px; text-align: center; }
        .footer { text-align: center; font-size: 14px; color: gray; margin-top: 50px; }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton > button:hover { background-color: #45a049; }
        </style>
        """,
        unsafe_allow_html=True
    )
add_custom_css()

st.markdown('<h1 class="main-title">🧠 AI-Based Political Speech Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analyze political statements for bias using a fine-tuned Transformer model.</p>', unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model(model_path):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            use_safetensors=True,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model.to("cpu").eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"🚨 Model loading failed: {e}")
        raise

model_path = "bias_detection_model" if model_choice == "Full (High Accuracy)" else "bias_detection_model_light"
model, tokenizer = load_model(model_path)

# Classify result
def classify_bias(score):
    return "Biased" if score > 0.5 else "Unbiased"

# Interpret graph scores
def interpret_bias_trend(scores):
    if not scores:
        return "No data available for interpretation."

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    variability = max_score - min_score

    interpretation = ""

    if avg_score > 0.7:
        interpretation += "🟥 The speech is predominantly **biased**."
    elif avg_score < 0.3:
        interpretation += "🟩 The speech appears mostly **unbiased**."
    else:
        interpretation += "🟨 The speech contains a **mix of biased and unbiased** statements."

    if variability > 0.4:
        interpretation += " ⚠️ There are **significant fluctuations**, indicating parts with strong opinions or abrupt tone shifts."
    else:
        interpretation += " ✅ The bias tone is fairly **consistent throughout**."

    return interpretation

# Word cloud
def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Chunking text for section-wise bias
def chunk_text(text, max_chars=200):
    words = text.split()
    chunks, current, count = [], [], 0
    for w in words:
        if count + len(w) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = [w]
            count = len(w)
        else:
            current.append(w)
            count += len(w) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks

# Get bias scores for list of texts
def get_bias_scores(texts, model, tokenizer):
    scores = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = model(**inputs)
            prob = torch.nn.functional.softmax(out.logits, dim=1)[0][1].item()
        scores.append(prob)
    return scores

# Input method
st.subheader("Choose Input Method")
input_option = st.radio(
    "Select how you want to provide input:",
    ("📄 Manual Text Input", "📂 Upload CSV File"),
    index=0
)

# TEXT INPUT MODE
if input_option == "📄 Manual Text Input":
    text_input = st.text_area("Enter political speech or statement:", placeholder="Type your statement here...")

    if st.button("Analyze Bias"):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text")
        else:
            cleaned_text = text_input.lower()

            try:
                inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    bias_score = probs[0][1].item()

                classification = classify_bias(bias_score)
                if classification == "Biased":
                    st.error(f"⚠️ Biased (score: {bias_score:.2f})")
                else:
                    st.success(f"✅ Unbiased (score: {bias_score:.2f})")

                st.subheader("☁️ Word Cloud of the Speech")
                generate_wordcloud(cleaned_text)

                st.subheader("📈 Bias Score Across Segments")
                chunks = chunk_text(cleaned_text)
                scores = get_bias_scores(chunks, model, tokenizer)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=scores, mode='lines+markers', name='Bias Score'))
                fig.update_layout(
                    xaxis_title='Chunk Index',
                    yaxis_title='Bias Score',
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                st.plotly_chart(fig)

                st.subheader("🧠 Interpretation")
                st.info(interpret_bias_trend(scores))

            except Exception as e:
                st.error(f"🚨 Analysis failed: {e}")

# CSV UPLOAD MODE
elif input_option == "📂 Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Uploaded CSV file preview:")
            st.dataframe(df)

            results = []
            all_text = ""

            for _, row in df.iterrows():
                text_input = str(row[0])
                all_text += f" {text_input}"

                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    bias_score = probs[0][1].item()

                classification = classify_bias(bias_score)
                results.append({
                    "Statement": text_input,
                    "Bias Score": bias_score,
                    "Result": classification
                })

            st.write("🔍 Analysis Results:")
            st.dataframe(pd.DataFrame(results))

            st.subheader("☁️ Word Cloud from Uploaded Data")
            generate_wordcloud(all_text.lower())

            st.subheader("📊 Bias Scores for Uploaded Statements")
            stmt_scores = [r["Bias Score"] for r in results]
            fig2 = go.Figure(data=[go.Bar(y=stmt_scores)])
            fig2.update_layout(
                xaxis_title="Statement Index",
                yaxis_title="Bias Score",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            st.plotly_chart(fig2)

            st.subheader("🧠 Interpretation")
            st.info(interpret_bias_trend(stmt_scores))

        except Exception as e:
            st.error(f"🚨 Error processing uploaded file: {e}")

# Feedback Section
st.markdown("---")
st.header("📣 Feedback")

with st.form("feedback_form"):
    user_feedback = st.text_area("We value your feedback! Please share your thoughts, issues, or suggestions:")
    contact_opt_in = st.checkbox("I’m open to being contacted for follow-up (optional)")
    contact_info = ""
    if contact_opt_in:
        contact_info = st.text_input("Enter your email or contact info:")

    submitted = st.form_submit_button("Submit Feedback")
    if submitted:
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n--- New Feedback ---\nFeedback: {user_feedback}\nContact: {contact_info if contact_opt_in else 'N/A'}\n")

        st.success("✅ Thank you! Your feedback has been recorded.")

# Footer
st.markdown('<div class="footer">🛠️ Built with ❤️ by <strong>Navneet Kaur</strong></div>', unsafe_allow_html=True)
