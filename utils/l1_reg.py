import argparse
import json
import torch
import numpy as np
import pickle
import cvxpy as cp

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='User name (e.g., user1)')
args = parser.parse_args()

save_path = f"../results/l1_reg/{args.name}_l1_reg.jsonl"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open(f'../results/l1_reg/{args.name}_p.pkl', 'rb') as f:
    d = pickle.load(f)

d = np.array(d)
d = d / len(d)

# Lambda should be on the same scale as ||d|| to have meaningful effect

print(f"d vector shape: {d.shape}")
print(f"d vector norm: {np.linalg.norm(d):.6f}")
print(f"d vector range: [{d.min():.6f}, {d.max():.6f}]")
print(f"Non-zero elements in d: {np.count_nonzero(np.abs(d) > 1e-8)}/{len(d)}")

# Store solutions for comparison
solutions = []

lambdas = [0.1, 0.5, 1, 2, 5, 10]

for i, lambda_ in enumerate(lambdas):
    print(f"\n--- Lambda = {lambda_} ---")

    p_var = cp.Variable(len(d))
    constraints = [cp.norm1(p_var) <= lambda_]
    objective = cp.Maximize(p_var @ d)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    print(f"Solver status: {problem.status}")
    print(f"Objective value: {problem.value}")
    
    # Check if solution was found
    if p_var.value is None:
        print(f"No solution found, skipping")
        continue
        
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Bad solver status: {problem.status}")
        continue
    
    p_raw = p_var.value
    p_norm = np.linalg.norm(p_raw)
    print(f"Raw solution norm: {p_norm:.6f}")
    print(f"Constraint violation: {max(0, p_norm - 1.0):.6f}")
    
    # Check the individual terms in the objective
    linear_term = p_raw @ d
    l1_term = np.sum(np.abs(p_raw))
    print(f"Linear term (p @ d): {linear_term:.6f}")
    print(f"L1 term (||p||_1): {l1_term:.6f}")
    print(f"Objective check: {linear_term - lambda_ * l1_term:.6f}")
    
    p = torch.tensor(p_raw, device=device, dtype=torch.float32)
    
    solutions.append({
        'lambda': lambda_,
        'p': p.clone(),
    })

    result_entry = {
        "lambda": lambda_,
        "p": p.tolist(),
    }

    with open(save_path, "a") as f:
        f.write(json.dumps(result_entry) + "\n")

# Compare solutions across different lambdas
print(f"\n--- COMPARISON ACROSS LAMBDAS ---")
if len(solutions) >= 2:
    for i in range(1, len(solutions)):
        diff = torch.norm(solutions[i]['p'] - solutions[0]['p']).item()
        print(f"Lambda {solutions[i]['lambda']} vs Lambda {solutions[0]['lambda']}: ||p_diff|| = {diff:.6f}")

print("L1 regularization completed!")