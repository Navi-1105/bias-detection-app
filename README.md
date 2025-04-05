# 🔍 Bias Detection Web App

A simple and interactive web application built using **Streamlit** and **Transformers** to detect bias in text using a fine-tuned BERT-based model.

## 🚀 Features

- Predict whether a given input text is **Biased** or **Not Biased**
- Upload `.csv` files to analyze multiple entries
- Interactive dashboard with statistics
- Powered by `bert-tiny` for fast performance on CPU

## 🧠 Model

- Fine-tuned on the [Bias-detection-combined](https://huggingface.co/datasets/pranjali97/Bias-detection-combined) dataset
- Uses HuggingFace `transformers` and `Trainer` API

## 🖥️ Tech Stack

- Python
- Streamlit
- HuggingFace Transformers
- PyTorch
- Pandas
- Matplotlib

## 📦 Installation

```bash
git clone https://github.com/Navi-1105/bias-detection-app.git
cd bias-detection-app
pip install -r requirements.txt
