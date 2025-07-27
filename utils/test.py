import pickle
import argparse
import json
import numpy as np
from attribute_prompts import attribute_prompts

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="user1")
args = parser.parse_args()

with open(f"../results/training_matrix/{args.name}.json", "r") as f:
    data = json.load(f)

data = np.array(data)

with open("../results/user_p.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        if entry['user'] == args.name:
            p = np.array(entry['p'])
            acc = np.sum((data @ p.reshape(-1, 1) > 0).astype(int)) / len(data)
            print(f"{args.name} accuracy: {acc}")
            
            with open("../results/approximation_accuracy.jsonl", "a") as f:
                f.write(json.dumps({'user': args.name, 'n': entry['sample_size'], 'accuracy': acc}) + "\n")

for i in range(len(attribute_prompts)):
    print(f"Attribute {i}: {attribute_prompts[i]}")
    p = np.zeros(len(attribute_prompts))
    p[i] = 1
    acc = np.sum((data @ p.reshape(-1, 1) > 0).astype(int)) / len(data)
    print(f"Attribute {i} accuracy: {acc}")
