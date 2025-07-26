import pickle
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="user1")
args = parser.parse_args()

with open(f"../results/user_test/{args.name}_toy.json", "r") as f:
    data = json.load(f)

data = np.array(data)

with open("../results/user_p_toy.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        if entry['user'] == args.name:
            p = np.array(entry['p'])
            acc = np.sum((data @ p.reshape(-1, 1) > 0).astype(int)) / len(data)
            print(f"{args.name} accuracy: {acc}")
            
            with open("../results/approximation_accuracy.jsonl", "a") as f:
                f.write(json.dumps({'user': args.name, 'n': entry['n'], 'accuracy': acc}) + "\n")