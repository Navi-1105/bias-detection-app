from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset
from sklearn.metrics import classification_report
import torch

# Load saved model and tokenizer
model_dir = "./bias_detection_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Load dataset (reloading same dataset used before)
dataset = load_dataset("pranjali97/Bias-detection-combined")
test_dataset = dataset["validation"]  # or use dataset["test"] if exists

# Tokenize the data
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_test = test_dataset.map(tokenize, batched=True)
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Prediction loop
y_true = []
y_pred = []

with torch.no_grad():
    for batch in tokenized_test:
        inputs = {
            "input_ids": batch["input_ids"].unsqueeze(0),
            "attention_mask": batch["attention_mask"].unsqueeze(0),
        }
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        y_pred.append(prediction)
        y_true.append(batch["label"])

# Evaluation report
print("\n📊 Evaluation Report:")
print(classification_report(y_true, y_pred, target_names=["Not Biased", "Biased"]))
