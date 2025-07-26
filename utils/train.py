import pickle
import argparse
import numpy as np
import torch
import json

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="user2")
args = parser.parse_args()

with open(f"d_{args.name}.pkl", "rb") as f:
    d = pickle.load(f)

d = np.array(d.cpu().numpy())

sample_sizes = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]

for sample_size in sample_sizes:
    curr = d[:sample_size]
    curr = np.mean(curr, axis=0)
    if np.linalg.norm(curr) > 1:
        curr = curr * (1 / np.linalg.norm(curr))
    
    adding = {'user': args.name, 'sample_size': sample_size, 'p': curr.tolist()}
    with open("../results/user_p.jsonl", "a") as f:
        f.write(json.dumps(adding) + "\n")
