
---

# 🧠 AI-Based Political Speech Analyzer

A web app to detect **bias** in political speeches using **AI** and **BERT**! Classify speeches as **biased** or **unbiased** and visualize the results.

## 🔑 Features

** 🎯 Bias Classification**: Classify speeches as **biased** or **unbiased**.
* **📊 Visualizations**: Word clouds and bias score trends.
* **📝 Input Options**:

  * **Text Input**: Manually input speeches.
  * **📂 CSV Upload**: Batch process multiple speeches.
* **📣 Feedback**: Collect user feedback to improve.

## 🚀 Technologies

* **Python** (Backend)
* **Streamlit** (Web App)
* **Transformers & BERT** (Model)
* **PyTorch** (Model Training)
* **Plotly** & **WordCloud** (Visualizations)

## 📊 Dataset

Labeled political speeches: **0 (unbiased)** or **1 (biased)**.

| Speech                    | Label |
| ------------------------- | ----- |
| "This policy is harmful." | 1     |
| "We must work together."  | 0     |

## ⚡ How to Run

1. Install dependencies:

   ```bash
   pip install streamlit transformers torch pandas plotly wordcloud
   ```

2. Clone the repo and run the app:

   ```bash
   git clone https://github.com/your-username/ai-political-speech-analyzer.git
   cd ai-political-speech-analyzer
   streamlit run app.py
   ```

## 🏋️‍♂️ Model Training

Train BERT using:

```bash
python train_bias_detection_model.py
```

Model saves to `bias_detection_model` folder.

## 📈 Evaluation Metrics

* **Accuracy**: Proportion of correct predictions.
* **F1 Score**: Balances precision and recall.

## 📝 License

MIT License

--
