import os
import torch
import numpy as np

folder = 'rewards'
curr = []

for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if filename.endswith(".pt"):   # optional: only process .pt files
        data = torch.load(filepath)
        curr.append(data)

# Concatenate all tensors along dim=0
if curr:
    merged = torch.cat(curr, dim=0)
    torch.save(merged, os.path.join(folder, "rewards.pt"))
    print(f"Saved merged tensor with shape {merged.shape}")
else:
    print("No tensors found to merge.")

