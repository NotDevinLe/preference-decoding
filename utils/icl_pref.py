import torch
import json
import random
import re
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

# ---- Determinism (optional)
random.seed(0)
torch.manual_seed(0)

# ---- Model / tokenizer
model = LLM(
    model=small_model_id,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_model_len=8192,
)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

parser = argparse.ArgumentParser()
parser.add_argument("--names", type=str, default="user11", help="User names for data files")
args = parser.parse_args()
names = args.names.split(",")

results = []

def flip_pair(chosen: str, rejected: str, p: float = 0.5):
    if random.random() < p:
        return rejected, chosen, "B"
    else:
        return chosen, rejected, "A"

def render_fewshot_block(examples):
    fs = examples[:]
    random.shuffle(fs)
    blocks = []
    for i, ex in enumerate(fs, 1):
        A, B, gold = flip_pair(ex["chosen"], ex["rejected"], p=0.5)
        blocks.append(
            f"# Example {i}\n"
            f"QUESTION:\n{ex['prompt']}\n"
            f"RESPONSE A:\n{A}\n"
            f"RESPONSE B:\n{B}\n"
            f"ANSWER: {gold}"
        )
    return "\n\n".join(blocks)

def create_fewshot_preference_prompt(fewshots, final_question, resp_chosen, resp_rejected):
    header = (
        "You are evaluating response quality, not content appropriateness.\n"
        "Sometimes the preferred response in the EXAMPLES is A, other times B.\n"
        "After `ANSWER:` output exactly one character: A or B.\n"
    )
    examples_text = render_fewshot_block(fewshots)

    # IMPORTANT: Flip A/B on the TASK too and return the gold.
    A_task, B_task, task_gold = flip_pair(resp_chosen, resp_rejected, p=0.5)

    task = (
        "# Task\n"
        f"QUESTION:\n{final_question}\n"
        f"RESPONSE A:\n{A_task}\n"
        f"RESPONSE B:\n{B_task}\n"
        "ANSWER:"
    )

    return header + "\n" + examples_text + "\n\n" + task, task_gold

def extract_label(text: str):
    text = text.strip()
    # Look for A or B anywhere in the text, but prefer after "ANSWER:"
    m = re.search(r"ANSWER:\s*([ABab])", text)
    if m:
        return m.group(1).upper()
    
    return None

for name in names:
    # ---- Load data
    with open(f"../data/persona_pref/{name}_train.json", "r") as f:
        train_data = json.load(f)
    with open(f"../data/persona_pref/{name}_test.json", "r") as f:
        test_data = json.load(f)

    # quick leakage check: identical prompts across train/test?
    train_prompts = {ex["prompt"] for ex in train_data}
    test_prompts = {ex["prompt"] for ex in test_data}
    overlap = train_prompts & test_prompts
    if overlap:
        print(f"[WARN] {name}: {len(overlap)} prompt(s) appear in both train/test (possible leakage).")

    k_list = [25]

    for k in k_list:
        if len(train_data) < k:
            raise ValueError(f"Need at least {k} train examples for {name}, got {len(train_data)}")
        selected_idx = [random.sample(range(len(train_data)), k) for _ in range(8)]
        examples_sets = [[train_data[i] for i in idxs] for idxs in selected_idx]

        formatted = []
        gold_labels = []

        for fewshots in examples_sets:
            for row in test_data:
                final_q = row["prompt"]
                resp_chosen = row["chosen"]
                resp_rejected = row["rejected"]
                prompt_text, gold = create_fewshot_preference_prompt(
                    fewshots, final_q, resp_chosen, resp_rejected
                )
                msg = [{"role": "user", "content": prompt_text}]
                formatted.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
                gold_labels.append(gold)

        # Keep generation tiny + stop; REMOVE logit_bias to avoid forcing 'A'
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=10,
            stop=["\n", "<|eot_id|>"]
            # no logit_bias here
        )

        outputs = model.generate(formatted, sampling_params)

        correct = 0
        valid = 0
        a_count = 0

        for out, gold in zip(outputs, gold_labels):
            gen = out.outputs[0].text
            print(f"Raw generation: '{gen}'")  # Add this line
            print(f"Gold label: {gold}")       # Add this line
            pred = extract_label(gen)
            print(f"Extracted: {pred}")        # Add this line
            print("---")

        for out, gold in zip(outputs, gold_labels):
            gen = out.outputs[0].text
            pred = extract_label(gen)
            if pred is None:
                continue  # skip non A/B
            valid += 1
            a_count += (pred == "A")
            correct += int(pred == gold)

        acc = (correct / valid) if valid > 0 else 0.0
        a_rate = (a_count / valid) if valid > 0 else 0.0
        print(f"{name}: Accuracy={acc:.3f}  ValidN={valid}  A-rate={a_rate:.2%}")

        # # write one JSON object per line (not the whole list each time)
        # out_path = f'../results/icl_pref/group1.jsonl'
        # with open(out_path, "a") as f:
        #     f.write(json.dumps({"user": name, "acc": acc, "k": k, "valid": valid, "a_rate": a_rate}) + "\n")
        # print(f"Results saved to {out_path}")
