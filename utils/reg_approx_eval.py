import argparse
import json
import torch
import os
from drift import approximate_l1
import random
import pickle
import cvxpy as cp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='User name (e.g., user1)')
args = parser.parse_args()


with open('a.pkl', 'rb') as f:
    temp = pickle.load(f)

sample_sizes = [200]
lambdas = np.exp(np.linspace(-5, 4, 20))

temp = torch.tensor(temp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for sample_size in sample_sizes:
    a = torch.sum(temp[:sample_size], dim=0).cpu().numpy()

    for lambda_reg in lambdas:
        p_var = cp.Variable(len(a))
        constraints = [cp.norm2(p_var) <= 1.0]
        objective = cp.Maximize(p_var @ a - lambda_reg * cp.norm1(p_var))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        p = torch.tensor(p_var.value, device=device, dtype=torch.float32)
        if p.norm() > 0:
            p = p / p.norm(p=2)

        with open("../results/reg_p.jsonl", 'a') as f:
            f.write(json.dumps({"user": args.name, "n": sample_size, "lambda_reg": lambda_reg, "p": p.tolist()}) + "\n")