import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# Configuration
MODELS = {
    "Full (High Accuracy)": "bias_detection_model",
    "Lightweight (Fast Inference)": "bias_detection_model_light"
}
TEST_DATA_PATH = "benchmark_test_data.csv"  # Located in project root

# Load dataset
df = pd.read_csv(TEST_DATA_PATH)
texts = df["text"].tolist()
true_labels = df["label"].tolist()

# Function to evaluate a model
def evaluate_model(model_name, model_path):
    print(f"\n🔍 Benchmarking: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True, local_files_only=True)
    model.to("cpu").eval()

    preds = []
    inference_times = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            bias_score = probs[0][1].item()

        inference_times.append(time.time() - start_time)
        preds.append(1 if bias_score > 0.5 else 0)

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    avg_time = sum(inference_times) / len(inference_times)

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"🎯 F1 Score: {f1:.4f}")
    print(f"⏱️ Avg Inference Time: {avg_time:.4f} sec")

# Run benchmarks
if __name__ == "__main__":
    for name, path in MODELS.items():
        evaluate_model(name, path)
