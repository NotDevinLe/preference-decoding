from vllm import LLM, SamplingParams
import json
import pickle
from transformers import AutoTokenizer
import time

def get_log_probs(model, tokenizer, system_prompts, user_prompts, completion_texts, device, temperature=0.0):
    input_ids = []
    ns = []
    completion_ids = []
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        # Apply chat template to get prompt tokens
        prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": sys_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        ns.append(len(prompt_ids))
        # Tokenize completion without skipping tokens
        completion_ids_i = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        input_ids_i = prompt_ids + completion_ids_i + [tokenizer.eos_token_id]
        input_ids.append(input_ids_i)
        completion_ids.append(completion_ids_i)
    sampling_params = SamplingParams(
        prompt_logprobs=0,
        max_tokens=1,
        temperature=temperature,
    )

    outputs = model.generate(
        prompt_token_ids=input_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    log_probs = []
    for compl, out, n in zip(input_ids, outputs, ns):
        logprobs = [
            (lxi[xi].logprob)
            for xi, lxi in zip(
                compl[1:],
                out.prompt_logprobs[1:],
            )
        ][n:]
        log_probs.append(sum(logprobs))

    token_counts = [len(compl) for compl in completion_ids]
    return log_probs, token_counts


model = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_model_len=8192,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

base_prompt = "You are a helpful assistant."
device = "cuda"

with open("data/persona_val_dataset.pkl", "rb") as f:
    data = pickle.load(f)['user_data']

test_data = []

total = 0
for user_id, pairs in data.items():
    for pair in pairs:
        test_data.append({
            "prompt": pair["prompt"],
            "output": pair["output"]
        })
        total += 1
    if total >= 100:  # Reduced for faster testing
        break

system_prompts = []
user_prompts = []
completions = []

for example in test_data:
    system_prompts.append(base_prompt)
    user_prompts.append(example["prompt"])
    completions.append(example["output"])

time_start = time.time()
log_probs, token_counts = get_log_probs(model, tokenizer, system_prompts, user_prompts, completions, device, temperature=0.0)
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
print(f"Processed {len(test_data)} examples")
print(f"Average log prob: {sum(log_probs)/len(log_probs):.3f}")
print(f"Average tokens per completion: {sum(token_counts)/len(token_counts):.1f}")