import sys
import os

# Add the root directory to the sys.path so that Python can find 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.benchmark import benchmark_model

def main():
    # Load model & tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("bias_detection_model")
    tokenizer = AutoTokenizer.from_pretrained("bias_detection_model")

    # Text to test
    sample_text = "This is an example political speech to benchmark model inference."

    # Run benchmark
    avg_ms = benchmark_model(model, tokenizer, sample_text)
    print(f"Baseline inference time: {avg_ms:.1f} ms")

if __name__ == "__main__":
    main()
