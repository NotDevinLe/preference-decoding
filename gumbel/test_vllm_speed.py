#!/usr/bin/env python3
import asyncio
import aiohttp
from typing import Tuple, List
from transformers import AutoTokenizer

VLLM_URL = "http://localhost:8000/v1/completions"
MODEL_ID  = "meta-llama/Llama-3.2-1B-Instruct"
CONCURRENCY = 128   # tune as needed

# ---------- helpers ----------
def build_full_prompt(tokenizer, sys_prompt: str, user_prompt: str, completion: str) -> Tuple[str, int, int]:
    """Return: full_text (prompt+completion), n_prefix_tokens, completion_len_tokens"""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_prompt.strip()},
         {"role": "user",   "content": user_prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    comp_ids   = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    return prompt_text + completion, len(prompt_ids), len(comp_ids)

def sum_completion_logprobs(resp_json, n_prefix: int, comp_len: int) -> float:
    lp = resp_json["choices"][0]["logprobs"]["token_logprobs"]
    end = min(len(lp), n_prefix + comp_len)  # guard if server adds a token somehow
    seg = [x for x in lp[n_prefix:end] if x is not None]
    return float(sum(seg))

async def fetch_sum_lp(session: aiohttp.ClientSession, prompt: str, n_prefix: int, comp_len: int) -> float:
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "echo": True,
        "logprobs": 1,
        "max_tokens": 0,      # no generation; just score provided text
        "temperature": 0.0,
    }
    async with session.post(VLLM_URL, json=payload) as r:
        r.raise_for_status()
        data = await r.json()
        return sum_completion_logprobs(data, n_prefix, comp_len)

# ---------- main ----------
async def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Three “styles”
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

    S = len(system_prompts)
    C = len(completions)

    # Build all S x C requests: score completion i under (system j, user j)
    jobs = []  # (j, i, prompt, n_prefix, comp_len)
    for j in range(S):
        for i in range(C):
            full, n_pref, clen = build_full_prompt(
                tokenizer, system_prompts[j], user_prompts[j], completions[i]
            )
            jobs.append((j, i, full, n_pref, clen))

    # Fire off requests with concurrency cap
    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def go(job):
            j, i, full, n_pref, clen = job
            async with sem:
                val = await fetch_sum_lp(session, full, n_pref, clen)
            return j, i, val

        results = await asyncio.gather(*[go(job) for job in jobs])

    # Assemble matrix scores[j][i]
    scores = [[0.0 for _ in range(C)] for _ in range(S)]
    for j, i, val in results:
        scores[j][i] = val

    # Pretty print matrix with headers
    col_headers = [f"comp{i}: {name.split()[0]}" for i, name in enumerate(["pirate", "teenager", "wizard"])]
    row_headers = [f"sys{j}: {name.split()[0]}"  for j, name in enumerate(["pirate", "teenager", "wizard"])]

    # header row
    print("\nLogprob sum matrix  (rows = system+user style, cols = completion style)\n")
    print("{:16s}".format(""), end="")
    for h in col_headers:
        print(f"{h:>20s}", end="")
    print()
    # rows
    for j in range(S):
        print(f"{row_headers[j]:16s}", end="")
        for i in range(C):
            print(f"{scores[j][i]:20.3f}", end="")
        print()

    # Optional: show argmax per row (which completion best fits each system style)
    print("\nBest completion per system row:")
    for j in range(S):
        best_i = max(range(C), key=lambda i: scores[j][i])
        print(f"  {row_headers[j]} -> {col_headers[best_i]} (score={scores[j][best_i]:.3f})")

if __name__ == "__main__":
    asyncio.run(main())
