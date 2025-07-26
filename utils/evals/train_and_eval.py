import os
import json
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv(dotenv_path="/mmfs1/gscratch/ark/devinl6/preference/preference-decoding/.env")
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN not found in .env")
login(hf_token)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="User name (e.g., user1)")
parser.add_argument("--model_path", type=str, default="./reward_model", help="Path to save/load model")
parser.add_argument("--sample_sizes", nargs='+', type=int, default=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20], help="Sample sizes to train on")
parser.add_argument("--eval_size", type=int, default=100, help="Number of examples to use for evaluation")
args = parser.parse_args()

# Load dataset
dataset = load_dataset(
    "json",
    data_files={"train": f"../data/processed/{args.name}.jsonl"},
    split="train"
)

# Prepare training data
texts = []
labels = []
for ex in dataset:
    texts.append(ex["prompt"] + "\n" + ex["chosen"])
    labels.append(1)  # preferred
    texts.append(ex["prompt"] + "\n" + ex["rejected"])
    labels.append(0)  # rejected

from datasets import Dataset
rm_dataset = Dataset.from_dict({"text": texts, "label": labels})

# Setup model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors=None
    )
    tokenized["labels"] = example["label"]
    return tokenized

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluation function
def evaluate_model(model, tokenizer, eval_dataset, device):
    model.eval()
    correct = 0
    total = 0
    
    for example in eval_dataset:
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        # Score chosen response
        chosen_input = tokenizer(
            prompt + "\n" + chosen,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            chosen_score = model(**chosen_input).logits[0][1].item()
        
        # Score rejected response
        rejected_input = tokenizer(
            prompt + "\n" + rejected,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            rejected_score = model(**rejected_input).logits[0][1].item()
        
        if chosen_score > rejected_score:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load or create evaluation dataset
eval_dataset = dataset.shuffle().select(range(args.eval_size))

# Training and evaluation loop
results = []
current_model = None
cumulative_samples = 0

for n in args.sample_sizes:
    print(f"\n{'='*50}")
    print(f"Training with {n} samples (cumulative: {cumulative_samples + n})...")
    print(f"{'='*50}")
    
    # Prepare training data for this iteration (use exactly n samples)
    tokenized = rm_dataset.map(tokenize, batched=False, remove_columns=rm_dataset.column_names)
    tokenized = tokenized.shuffle().select(range(min(n * 2, len(tokenized))))
    
    # Load existing model or create new one
    if current_model is not None and os.path.exists(args.model_path):
        print(f"Loading existing model from {args.model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            device_map=None
        )
    else:
        print("Creating new model")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=torch.float32,
            device_map=None
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    current_model = model
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    trainer.train()
    trainer.save_model(args.model_path)
    
    # Evaluate
    print(f"Evaluating model with {cumulative_samples + n} training samples...")
    accuracy = evaluate_model(model, tokenizer, eval_dataset, device)
    print(f"Accuracy: {accuracy:.2%}")
    
    # Store results
    result = {
        'user': args.name,
        'n_samples': cumulative_samples + n,
        'accuracy': accuracy
    }
    results.append(result)
    
    # Update cumulative samples
    cumulative_samples += n
    
    # Append result to file
    with open("../reward_results.jsonl", "a") as f:
        f.write(json.dumps(result) + '\n')

print(f"\n{'='*50}")
print("Training and evaluation complete!")
print(f"Results saved to ../reward_results.jsonl")
print(f"{'='*50}")

# Print summary
print("\nSummary:")
for result in results:
    print(f"  {result['n_samples']} samples: {result['accuracy']:.2%} accuracy") 