import argparse
import torch
import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import load_dataset


if torch.backends.mps.is_available():
    print("✅ MPS is available and will be used as backend.")
    device = torch.device("mps")
else:
    print("❌ MPS not available. Using CPU as fallback.")
    device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text summarization model on CNN/DailyMail.")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/bart-base", help="HF model checkpoint")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--max_input_length", type=int, default=256, help="Max input sequence length")
    parser.add_argument("--max_target_length", type=int, default=64, help="Max target sequence length")
    return parser.parse_args()


def main():
    args = parse_args()

    # Start an MLflow run
    mlflow.start_run()

    # Log hyperparameters to MLflow
    mlflow.log_param("model_name_or_path", args.model_name_or_path)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", args.batch_size)

    print("Loading dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # For initial draft, let's use smaller subsets
    dataset["train"] = dataset["train"].select(range(20000))
    dataset["validation"] = dataset["validation"].select(range(1000))
    dataset["test"] = dataset["test"].select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def preprocess_function(examples):
        inputs = examples["article"]
        targets = examples["highlights"]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_input_length,
            truncation=True
        )
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True
        )   
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        remove_unused_columns=True,
        predict_with_generate=True,
        push_to_hub=False
    )

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Evaluate on validation
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(metrics)
    mlflow.log_metrics(metrics)

    mlflow.end_run()


if __name__ == "__main__":
    main()