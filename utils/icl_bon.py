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
    max_model_len=8192,
)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

# ---- Load data
with open("../data/persona_pref/user11_train.json", "r") as f:
    train_data = json.load(f)

with open("../data/bon_attributes.json", "r") as f:
    bon_data = json.load(f)

# ---- Few-shot selection
k = 4
random.seed(0)
selected_idx = [random.sample(range(len(train_data)), k) for _ in range(8)]
examples_sets = [[train_data[i] for i in idxs] for idxs in selected_idx]

def create_fewshot_preference_prompt(fewshots, final_question, resp_a, resp_b):
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
        qa.append(f"RESPONSE A:\n{ex['chosen']}")
        qa.append(f"RESPONSE B:\n{ex['rejected']}")
        qa.append("ANSWER: A")
        parts.append("\n".join(qa))
    parts.append("# Task")
    parts.append(f"QUESTION:\n{final_question}")
    parts.append(f"RESPONSE A:\n{resp_a}")
    parts.append(f"RESPONSE B:\n{resp_b}")
    parts.append("ANSWER:")
    return "\n\n".join(parts)

def extract_label(text):
    m = re.search(r"[ABab]", text or "")
    if m:
        return m.group(0).upper()
    return None

# ---- Deterministic A/B output
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1,
)

# ---- Tally wins per prompt per few-shot set (using linear tournament)
# wins_per_ex[ex_idx][prompt_idx] -> list of wins counts of length n
wins_per_ex = []
for _ in range(len(examples_sets)):
    wins_per_ex.append([])

for ex_idx, fewshots in enumerate(examples_sets):
    per_prompt = []
    for prompt_idx, row in enumerate(bon_data):
        outputs_list = row["outputs"]
        n = len(outputs_list)
        if n == 0:
            per_prompt.append([])
            continue
        if n == 1:
            per_prompt.append([0])
            continue

        wins = [0] * n
        # start champion at index 0
        champ = 0

        # linearly compare champ vs next candidate
        for t in range(1, n):
            resp_a = outputs_list[champ]
            resp_b = outputs_list[t]
            msg = [{
                "role": "user",
                "content": create_fewshot_preference_prompt(
                    fewshots, row["prompt"], resp_a, resp_b
                )
            }]
            prompt_text = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )

            gen = model.generate([prompt_text], sampling_params)[0]
            label = extract_label(gen.outputs[0].text)

            if label == "A":
                wins[champ] += 1
                # champ stays the same
            elif label == "B":
                wins[t] += 1
                champ = t  # challenger becomes new champion
            else:
                # invalid output => treat as tie -> do nothing, keep champ
                pass

        # record per-prompt wins for this few-shot set
        per_prompt.append(wins)
    wins_per_ex[ex_idx] = per_prompt

# ---- Aggregate across few-shot sets and pick argmax per prompt
final_selected = []
for prompt_idx, row in enumerate(bon_data):
    n = len(row["outputs"])
    if n == 0:
        final_selected.append({
            "prompt_idx": prompt_idx,
            "prompt": row["prompt"],
            "winner_index": None,
            "winner_wins": 0,
            "total_votes": 0,
            "winner_text": None
        })
        continue

    agg = [0] * n
    for ex_idx in range(len(examples_sets)):
        w = wins_per_ex[ex_idx][prompt_idx]
        for t in range(n):
            agg[t] += w[t]

    # argmax with simple tie-break: lowest index wins ties
    best_idx = 0
    best_val = agg[0]
    for t in range(1, n):
        if agg[t] > best_val:
            best_val = agg[t]
            best_idx = t

    final_selected.append({
        "prompt_idx": prompt_idx,
        "prompt": row["prompt"],
        "winner_index": best_idx,
        "winner_wins": best_val,
        "total_votes": sum(agg),
        "winner_text": row["outputs"][best_idx]
    })

# ---- (Optional) print a small summary
num = min(5, len(final_selected))
for item in final_selected[:num]:
    print(f"[Prompt {item['prompt_idx']}] winner idx={item['winner_index']} "
          f"with {item['winner_wins']}/{item['total_votes']} votes")

# ---- Save results
with open("../results/bon_pairwise_winners_linear.json", "w") as f:
    json.dump(final_selected, f, indent=2)

print(f"Saved winners for {len(final_selected)} prompts to ../results/bon_pairwise_winners_linear.json")
