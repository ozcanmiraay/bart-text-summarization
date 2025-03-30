import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load fine-tuned model
model = AutoModelForSeq2SeqLM.from_pretrained("checkpoints/checkpoint-2500")  # Replace with your actual checkpoint folder
tokenizer = AutoTokenizer.from_pretrained("checkpoints/checkpoint-2500")
model.to(device)
model.eval()

# Load test data
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:5]")  # First 5 examples

for i, sample in enumerate(dataset):
    article = sample["article"]
    reference = sample["highlights"]

    inputs = tokenizer(
        article,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            no_repeat_ngram_size=3
        )

    prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"\n=== Example {i+1} ===")
    print("Article (truncated):", article[:350], "...")
    print("Reference Summary:", reference)
    print("Predicted Summary:", prediction)