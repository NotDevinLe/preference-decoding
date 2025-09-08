import torch
import json
import random
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

small_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
random.seed(0)
k = 16
selected_idx_sets = [random.sample(range(len(train_data)), k) for _ in range(8)]
fewshot_sets = [[train_data[i] for i in idxs] for idxs in selected_idx_sets]

def create_fewshot_prompt(fewshots, final_question):
    # Force the model to output only between <answer>...</answer>.
    # No headings in the target; few-shots are shown as paired examples.
    parts = []
    parts.append(
        "You will see examples of a QUESTION with a preferred and a dispreferred response.\n"
        "For the final QUESTION, write only the preferred response between <answer> and </answer>.\n"
        "Do not include headings, explanations, reasons, or extra text outside the tags."
    )
    for i, ex in enumerate(fewshots, 1):
        parts.append(
            f"# Example {i}\n"
            f"QUESTION:\n{ex['prompt']}\n"
            f"PREFERRED:\n{ex['chosen']}\n"
            f"DISPREFERRED:\n{ex['rejected']}\n"
        )
    parts.append("# Task")
    parts.append(f"QUESTION:\n{final_question}")
    parts.append("Write only the preferred response between the tags below:")
    parts.append("<answer>")
    # The model will continue here; we will stop at </answer>.
    return "\n\n".join(parts)

# ---- Build prompts
formatted = []
prompt_map = []  # keep mapping so we know which prompt corresponds to which output
for fewshots in fewshot_sets:
    for row in bon_data:
        final_q = row["prompt"]
        user_msg = {"role": "user", "content": create_fewshot_prompt(fewshots, final_q)}
        # IMPORTANT: add assistant prefix so model starts writing the answer immediately
        formatted.append(tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True))
        prompt_map.append(final_q)

# ---- Strong output constraints
sampling_params = SamplingParams(
    temperature=0.2,      # low temp to reduce rambling
    top_p=0.9,
    max_tokens=512,
    stop=["</answer>", "\n#", "##", "Explanation:", "Reason:"]
)

outputs = model.generate(formatted, sampling_params)

# ---- Extract clean answer text (between <answer> ... </answer>)
def extract_answer(txt: str) -> str:
    # take everything after the first <answer>, strip any trailing whitespace
    # (we stopped before </answer>, so it shouldn't be present)
    if "<answer>" in txt:
        txt = txt.split("<answer>", 1)[1]
    # remove any accidental headings if stop didn't trigger perfectly
    txt = re.split(r"\n[#<]", txt, maxsplit=1)[0]  # cut at next heading/tag if any
    return txt.strip()

all_data = []
for i, out in enumerate(outputs):
    gen = out.outputs[0].text
    clean = extract_answer(gen)
    all_data.append({
        "prompt": prompt_map[i],
        "raw_output": gen,
        "output": clean
    })

with open("../results/icl_gen.json", "w") as f:
    json.dump(all_data, f, indent=2)
