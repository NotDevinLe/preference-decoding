import pickle
import argparse
import numpy as np
import torch
import json
from drift import get_training_matrix, get_log_probs
from vllm import LLM
from transformers import AutoTokenizer
from attribute_prompts import attribute_prompts, persona_prompts, user1_reg_prompts, user2_reg_prompts, user4_reg_prompts, base_prompt
import cvxpy as cp
import tqdm

def get_p_vector(preference_data):
    # Model and tokenizer setup
    small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = vllm.LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=8192)

    tokenizer = AutoTokenizer.from_pretrained(small_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    data = []
    for j in range(len(preference_data)):
        question = preference_data[j]['prompt']
        yw = preference_data[j]['chosen']  # winning/chosen response
        yl = preference_data[j]['rejected']  # losing/rejected response
        data.append((question, yw, yl))

    print(f"Converted {len(data)} samples to drift format")
    res = []

    lambdas = [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    d = get_training_matrix(data[:200], model, tokenizer, base_prompt, attribute_prompts, device)

    for lambda_ in lambdas:
        sample_sizes = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]

        for sample_size in sample_sizes:
            curr = d[:sample_size]
            # Take mean over samples and convert to numpy
            d_mean = torch.mean(curr, dim=0).cpu().numpy()
            
            p_var = cp.Variable(len(d_mean))  # Use number of attributes, not samples
            # Remove redundant constraint since you're using L1 penalty
            
            linear_term = d_mean @ -p_var  # Use numpy array
            l1_penalty = lambda_ * cp.norm1(p_var)
            l2_penalty = 0.01 * cp.sum_squares(p_var)  # Use smaller L2 penalty
            objective = cp.Minimize(linear_term + l1_penalty + l2_penalty)
            problem = cp.Problem(objective)  # No constraints
            problem.solve()

            if p_var.value is None:
                print("Optimization failed, falling back to simple normalization")
                # Use the current sample mean, not full d
                current_norm = torch.norm(torch.mean(curr, dim=0), p=1)
                if current_norm > 1:
                    p = torch.mean(curr, dim=0) * (1 / current_norm)
                else:
                    p = torch.mean(curr, dim=0)
                p = p.cpu().numpy()
            else:
                p = p_var.value  # Use the optimization result

            adding = {'user': args.name, 'sample_size': sample_size, 'p': p.tolist(), 'lambda': lambda_}
            res.append(adding)

    return res


def get_test_matrix(preference_data):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model = LLM(model=model_id, tensor_parallel_size=1, gpu_memory_utilization=0.5, max_model_len=8192)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prompt_list = [d['prompt'] for d in preference_data]
    chosen_list = [d['chosen'] for d in preference_data]
    rejected_list = [d['rejected'] for d in preference_data]

    #  Set attribute prompts here
    attributes = attribute_prompts

    W = np.zeros((len(preference_data), len(attributes)))
    L = np.zeros((len(preference_data), len(attributes)))

    base_prompt = "You are an AI assistant."

    base_chosen, base_chosen_counts = get_log_probs(model, tokenizer, [base_prompt] * len(prompt_list), prompt_list, chosen_list, device, temperature=0.0)
    base_rejected, base_rejected_counts = get_log_probs(model, tokenizer, [base_prompt] * len(prompt_list), prompt_list, rejected_list, device, temperature=0.0)

    base_chosen = np.array(base_chosen) / np.array(base_chosen_counts)
    base_rejected = np.array(base_rejected) / np.array(base_rejected_counts)

    for i, system_prompt in tqdm.tqdm(enumerate(attributes)):
        chosen_logprobs, chosen_counts = get_log_probs(model, tokenizer, [system_prompt] * len(prompt_list), prompt_list, chosen_list, device, temperature=0.0)
        rejected_logprobs, rejected_counts = get_log_probs(model, tokenizer, [system_prompt] * len(prompt_list), prompt_list, rejected_list, device, temperature=0.0)

        W[:, i] = np.array(chosen_logprobs) / np.array(chosen_counts) - base_chosen
        L[:, i] = np.array(rejected_logprobs) / np.array(rejected_counts) - base_rejected

    full = W - L

    return full

def test_p_vector(p_vector_list, testing_matrix):
    results = []
    for entry in p_vector_list:
        p_vector = np.array(entry['p'])
        acc = np.sum((testing_matrix @ p_vector.reshape(-1, 1) > 0).astype(int)) / len(testing_matrix)
        results.append({'user': entry['user'], 'sample_size': entry['sample_size'], 'lambda': entry['lambda'], 'accuracy': acc})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="user1")
    args = parser.parse_args()

    with open(f"../data/toy/attribute/{args.name}_train.json", "r") as f:
        train_data = json.load(f)

    with open(f"../data/toy/attribute/{args.name}_test.json", "r") as f:
        test_data = json.load(f)

    p_vector_list = get_p_vector(train_data)
    test_matrix = get_test_matrix(test_data)
    test_results = test_p_vector(p_vector_list, test_matrix)
    print(test_results)