from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Load the saved model and tokenizer
model_dir = "./bias_detection_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Set model to evaluation mode
model.eval()

# Function to predict
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return "Biased" if predicted_class == 1 else "Not Biased"

# Test example
sample_text = "Immigrants are a burden to our society."
prediction = predict(sample_text)
print(f"Text: {sample_text}")
print(f"Prediction: {prediction}")
