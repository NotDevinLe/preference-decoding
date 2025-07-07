import os
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"
import json
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
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
args = parser.parse_args()

dataset = load_dataset(
    "json",
    data_files={"train": f"../data/processed/{args.name}.jsonl"},
    split="train"
)

texts = []
labels = []
for ex in dataset:
    texts.append(ex["prompt"] + "\n" + ex["chosen"])
    labels.append(1)  # preferred
    texts.append(ex["prompt"] + "\n" + ex["rejected"])
    labels.append(0)  # rejected

from datasets import Dataset
rm_dataset = Dataset.from_dict({"text": texts, "label": labels})

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    out = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    out["labels"] = example["label"]
    return out

tokenized = rm_dataset.map(tokenize, batched=False)
tokenized = tokenized.select(range(min(16, len(tokenized))))

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    torch_dtype=torch.float32,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

training_args = TrainingArguments(
    output_dir=f"./reward_model_{args.name}",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model(f"./reward_model_{args.name}")

