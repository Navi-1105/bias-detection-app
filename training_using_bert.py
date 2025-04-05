import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

# 1. Load the dataset
dataset = load_dataset("pranjali97/Bias-detection-combined")
print(dataset)
print(dataset["train"][0])

# 2. Use a small, fast model for CPU
model_name = "prajjwal1/bert-tiny"

# 3. Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Optional: Freeze encoder to speed up training
for param in model.base_model.parameters():
    param.requires_grad = False

# 5. Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Split into train and eval sets
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# 7. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 9. Train the model
print("🚀 Training the model...")
trainer.train()

# 10. Evaluate the model
eval_results = trainer.evaluate()
print("📊 Evaluation Results:", eval_results)

# 11. Save the trained model
model.save_pretrained("./bias_detection_model")
tokenizer.save_pretrained("./bias_detection_model")
print("✅ Model training and saving complete!")

# 12. Load the model for inference
model_path = "./bias_detection_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to CPU
device = torch.device("cpu")
model.to(device)
model.eval()

# 13. Define inference function
def predict_bias(text):
    """Predict whether a given text is biased or not."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    return "Biased" if prediction == 1 else "Not Biased"


# 14. Test it
test_text = "This group of people always causes trouble."
print("🔍 Test Prediction:", predict_bias(test_text))
print(predict_bias("All immigrants are criminals."))
print(predict_bias("Cats are adorable and make great pets."))
print(predict_bias("Some people just aren’t smart enough."))

