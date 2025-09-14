#!/usr/bin/env python3
# Local vLLM logprob scorer + 3x3 cross-score demo (pirate / teenager / wizard)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import math
import time

# ---------------------------
# Core scorer (same semantics as your function)
# ---------------------------
def get_log_probs(model, tokenizer, system_prompts, user_prompts, completion_texts, device, temperature=0.0):
    """
    Returns:
      log_probs: list of sum log-probs over completion tokens for each (sys,user,completion)
      token_counts: list of completion token counts
    """
    input_ids = []
    ns = []                # prompt prefix lengths (tokens)
    completion_ids_list = []

    # Build tokenized inputs: [prompt_tokens] + [completion_tokens] + [eos]
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        # Apply chat template to get prompt text
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_prompt.strip()},
                {"role": "user",   "content": user_prompt.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize prompt and completion
        prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        completion_ids = tokenizer([completion],   return_tensors=None, add_special_tokens=False)["input_ids"][0]

        ns.append(len(prompt_ids))
        completion_ids_list.append(completion_ids)

        input_ids.append(prompt_ids + completion_ids + [tokenizer.eos_token_id])

    # Ask vLLM to compute prompt_logprobs (we set max_tokens=1 just to trigger compute; we ignore any gen)
    sampling_params = SamplingParams(
        prompt_logprobs=0,   # we only need logprob of the chosen (next) token per position
        max_tokens=1,
        temperature=temperature,
    )

    outputs = model.generate(
        prompt_token_ids=input_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    # Slice out completion region and sum logprobs
    log_probs = []
    for compl_ids, out, n in zip(input_ids, outputs, ns):
        # out.prompt_logprobs is aligned with compl_ids (except first token has no prev context)
        # We zipped compl[1:] with prompt_logprobs[1:] in your code; replicate that.
        per_token = [
            lxi[xi].logprob
            for xi, lxi in zip(compl_ids[1:], out.prompt_logprobs[1:])
        ]
        # Now take only the completion segment:
        comp_token_count = len(compl_ids) - n - 1  # -1 because compl_ids includes EOS we appended
        # But we constructed compl_ids = prompt + completion + [eos]; so completion length is:
        # true_comp_len = len(completion_ids)
        # Let's use that to be precise:
        true_comp_len = len(completion_ids_list[len(log_probs)])
        # Slice: start at n (first completion token), take true_comp_len tokens
        comp_slice = per_token[n : n + true_comp_len]
        log_probs.append(sum(comp_slice))

    token_counts = [len(cids) for cids in completion_ids_list]
    return log_probs, token_counts


# ---------------------------
# Build 3x3 cross-score inputs
# ---------------------------
def build_cross_score_inputs(system_prompts, user_prompts, completions):
    """
    For S systems/users and C completions, build S*C triples so that
    entry (j,i) scores completion[i] under (system[j], user[j]).
    """
    S = len(system_prompts)
    C = len(completions)

    sys_list = []
    usr_list = []
    comp_list = []

    for j in range(S):
        for i in range(C):
            sys_list.append(system_prompts[j])
            usr_list.append(user_prompts[j])
            comp_list.append(completions[i])

    return sys_list, usr_list, comp_list, S, C


# ---------------------------
# Demo: pirate / teenager / wizard
# ---------------------------
if __name__ == "__main__":
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    # Load locally (no HTTP)
    model = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Three personas / prompts
    system_prompts = [
        "You are a grumpy pirate who always talks about treasure.",
        "You are an angsty teenager who complains about homework.",
        "You are a wise old wizard who speaks in riddles.",
    ]
    user_prompts = [
        "Introduce yourself.",
        "Tell me about your day.",
        "What is your secret of power?",
    ]
    completions = [
        "Arrr, I be Blackbeard, scourge of the seas!",
        "Ugh, school is so boring, nobody understands me.",
        "The secret lies in patience, young apprentice.",
    ]

    # Build S*C inputs
    sys_list, usr_list, comp_list, S, C = build_cross_score_inputs(system_prompts, user_prompts, completions)

    # Score
    t0 = time.time()
    log_probs, token_counts = get_log_probs(
        model, tokenizer, sys_list, usr_list, comp_list, device="cuda", temperature=0.0
    )
    t1 = time.time()

    # Arrange into SxC matrix
    scores = [[0.0 for _ in range(C)] for _ in range(S)]
    k = 0
    for j in range(S):
        for i in range(C):
            scores[j][i] = log_probs[k]
            k += 1

    # Pretty-print
    row_names = ["pirate_sys", "teen_sys", "wizard_sys"]
    col_names = ["pirate_comp", "teen_comp", "wizard_comp"]

    print("\nLogprob sum matrix  (rows = system+user style, cols = completion style)\n")
    print("{:16s}".format(""), end="")
    for h in col_names:
        print(f"{h:>20s}", end="")
    print()
    for j in range(S):
        print(f"{row_names[j]:16s}", end="")
        for i in range(C):
            print(f"{scores[j][i]:20.3f}", end="")
        print()

    # Row-wise argmax
    print("\nBest completion per system row:")
    for j in range(S):
        best_i = max(range(C), key=lambda i: scores[j][i])
        print(f"  {row_names[j]} -> {col_names[best_i]} (score={scores[j][best_i]:.3f})")

    # Summary like your script
    print(f"\nTime taken: {t1 - t0:.2f} seconds")
    print(f"Processed {len(sys_list)} examples")
    avg_lp = sum(log_probs) / len(log_probs)
    avg_toks = sum(token_counts) / len(token_counts)
    print(f"Average log prob: {avg_lp:.3f}")
    print(f"Average tokens per completion: {avg_toks:.1f}")
