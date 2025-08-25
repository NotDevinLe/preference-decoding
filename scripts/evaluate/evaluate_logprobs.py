import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import numpy as np
import pickle
from dotenv import load_dotenv
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from vllm import LLM, SamplingParams
from src.core.drift import get_log_probs

def get_log_probs_updated(model, tokenizer, system_prompts, user_prompts, completion_texts, device, temperature=0.0):
    input_ids = []
    prompt_lengths = []
    completion_ids_list = []

    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        # Build prompt (system + user)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_prompt.strip()},
             {"role": "user", "content": user_prompt.strip()}],
            tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        completion_ids = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]

        # No EOS here; we only score the actual completion tokens
        input_ids_i = prompt_ids + completion_ids

        input_ids.append(input_ids_i)
        prompt_lengths.append(len(prompt_ids))
        completion_ids_list.append(completion_ids)

    sampling_params = SamplingParams(
        prompt_logprobs=1,   # <- must be >=1
        max_tokens=1,        # we don't actually need generations; 1 is fine
        temperature=temperature,
    )

    outputs = model.generate(
        prompt_token_ids=input_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    log_probs = []
    token_counts = []
    for inp, out, n_prompt, compl_ids in zip(input_ids, outputs, prompt_lengths, completion_ids_list):
        # We want log p(compl[t] | prompt + compl[:t]) for t over completion span
        # Completion occupies positions [n_prompt, n_prompt + len(compl_ids) - 1] in inp
        acc = 0.0
        for pos in range(n_prompt, n_prompt + len(compl_ids)):
            chosen_token = inp[pos]
            # vLLM returns a dict-like (or list of candidates) for each position
            lp_dict = out.prompt_logprobs[pos]  # logprobs for the token at 'pos'
            # Get logprob of the observed token; it should be present
            # If not, fall back to the candidate with matching id or error
            if isinstance(lp_dict, dict):
                acc += lp_dict[chosen_token].logprob
            else:
                # lp_dict can be a list of candidates; find match
                found = False
                for cand in lp_dict:
                    if cand.token_id == chosen_token:
                        acc += cand.logprob
                        found = True
                        break
                if not found:
                    # As a conservative fallback, skip (or raise)
                    # Here we add 0 which is pessimistic; consider raising.
                    pass
        log_probs.append(acc)
        token_counts.append(len(compl_ids))

    return log_probs, token_counts

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LLM(
    model=model_id,
    tensor_parallel_size=1,
    trust_remote_code=True,
    max_model_len=512,
)

system_prompts = ["You are an AI assistant who speaks like a gentle preschool educator. You are warm, often encouraging. You tend to respond with simple, nurturing language and relatable examples. You value patience and foundational understanding.", "You are an AI assistant who speaks like a pirate."]
user_prompts = ["What is a newspaper?", "What is a newspaper?"]
completion_texts = ["Sweetie, let me tell you about newspapers! \n\nA newspaper is a special kind of magazine that people read every day to stay informed about what's happening in the world. It's like a big, long book that tells you about all sorts of things like news, stories, and fun facts about people, places, and things.\n\nImagine you're sitting around the table with all your friends, and someone shares a story about a new park that's been built near your school. The story might tell you all about the park, like its location, what it looks like, and why it's a great place to play. That's kind of like what a newspaper does!\n\nNewspapers usually have lots of different sections, like news, sports, and features, just like a big puzzle with lots of pieces that fit together to tell a bigger story. They might also have pictures and comics to make it more fun to read.\n\nDo you think you might want to pick one up and see what's in it?"
, "Sweetie, let me tell you about newspapers! \n\nA newspaper is a special kind of magazine that people read every day to stay informed about what's happening in the world. It's like a big, long book that tells you about all sorts of things like news, stories, and fun facts about people, places, and things.\n\nImagine you're sitting around the table with all your friends, and someone shares a story about a new park that's been built near your school. The story might tell you all about the park, like its location, what it looks like, and why it's a great place to play. That's kind of like what a newspaper does!\n\nNewspapers usually have lots of different sections, like news, sports, and features, just like a big puzzle with lots of pieces that fit together to tell a bigger story. They might also have pictures and comics to make it more fun to read.\n\nDo you think you might want to pick one up and see what's in it?"]


log_probs, token_counts = get_log_probs(model, tokenizer, system_prompts, user_prompts, completion_texts, device="cuda")

print(log_probs, token_counts)

log_probs, token_counts = get_log_probs_updated(model, tokenizer, system_prompts, user_prompts, completion_texts, device="cuda", temperature=0.0)

print(log_probs, token_counts)