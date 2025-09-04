import torch
import json
import random
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

small_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# ---- Model / tokenizer
model = LLM(
    model=small_model_id,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_model_len=4096,
)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

# ---- Load data
with open("../data/persona_pref/user11_train.json", "r") as f:
    train_data = json.load(f)

with open("../data/persona_pref/user11_test.json", "r") as f:
    test_data = json.load(f)

# ---- Pick 8 few-shot examples (correct range)

k = 4
selected_idx = [random.sample(range(len(train_data)), k) for _ in range(8)]
examples = [[train_data[i] for i in selected_idx[j]] for j in range(8)]

def create_fewshot_preference_prompt(fewshots, final_question, resp_a, resp_b):
    # Make few-shots mirror the A/B CHOICE format exactly.
    parts = []
    parts.append(
        "You are a *strict* judge of which response better matches the user's preferences.\n"
        "For each example, you will see a QUESTION and two RESPONSES (A and B). "
        "Answer with exactly one character: A or B. No punctuation, no words."
    )
    for i, ex in enumerate(fewshots, 1):
        qa = []
        qa.append(f"# Example {i}")
        qa.append(f"QUESTION:\n{ex['prompt']}")
        qa.append(f"RESPONSE A:\n{ex['chosen']}")    # A is the preferred one in examples
        qa.append(f"RESPONSE B:\n{ex['rejected']}")
        qa.append("ANSWER: A")                       # explicit target label
        parts.append("\n".join(qa))
    # Now the actual test item
    parts.append("# Task")
    parts.append(f"QUESTION:\n{final_question}")
    parts.append(f"RESPONSE A:\n{resp_a}")
    parts.append(f"RESPONSE B:\n{resp_b}")
    parts.append("ANSWER:")  # model should output A or B next
    return "\n\n".join(parts)

# ---- Build prompts (ensure the assistant is prompted to emit just A/B)
formatted = []

for example in examples:
    for row in test_data:
        final_q = row["prompt"]
        resp_a = row["chosen"]    # ground-truth preferred
        resp_b = row["rejected"]
        msg = [{"role": "user", "content": create_fewshot_preference_prompt(example, final_q, resp_a, resp_b)}]
        formatted.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

# ---- Force a single-character answer
sampling_params = SamplingParams(
    temperature=0.0,   # deterministic
    top_p=1.0,
    max_tokens=1,      # only 'A' or 'B'
)

outputs = model.generate(formatted, sampling_params)

def extract_label(text):
    # Robustly get the first A/B in the output (case-insensitive)
    m = re.search(r"[ABab]", text)
    return m.group(0).upper() if m else None

correct = 0
for out in outputs:
    pred = extract_label(out.outputs[0].text)
    correct += 1 if pred == "A" else 0  # ground truth is A by construction

acc = correct / len(outputs)
print(f"Accuracy: {acc:.3f}")
