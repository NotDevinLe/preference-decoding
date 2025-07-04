import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

small_model_id = "meta-llama/Llama-3.2-1B-Instruct"
big_model_id   = "meta-llama/Meta-Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model_ds = AutoModelForCausalLM.from_pretrained(
    small_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model_bs = AutoModelForCausalLM.from_pretrained(
    big_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)

# Your input question
question = "What is the future of space exploration?"

# Format the prompt (LLaMA-chat-style if needed)
prompt = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
    + question.strip()
    + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model_bs.device)

# Generate output
with torch.no_grad():
    output = model_bs.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)