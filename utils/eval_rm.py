import os
import json
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="User name (e.g., user1)")
parser.add_argument("--folder_name", type=str, required=True, help="folder path for reward model")
args = parser.parse_args()

model_path = f"{args.folder_name}"  # or wherever you saved it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset

dataset = load_dataset("json", data_files=f"../data/processed/{args.name}.jsonl", split="train")
dataset = dataset.shuffle().select(range(38))

def score_response(prompt, response):
    inputs = tokenizer(
        prompt + "\n" + response,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        score = model(**inputs).logits[0][1].item()
    return score

correct = 0

for example in dataset:
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    chosen_score = score_response(prompt, chosen)
    rejected_score = score_response(prompt, rejected)
    print(chosen_score, rejected_score)
    if chosen_score > rejected_score:
        correct += 1

accuracy = correct / len(dataset)
print(f"Reward model accuracy: {accuracy:.2%}")


