
import torch
from drift import approximate
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Model and tokenizer setup
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=4096)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

prompt = "Hello, how are you?"
persona = "You are a helpful assistant."
messages = [
                {"role": "system", "content": persona},
                {"role": "user", "content": prompt}
            ]

formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=512,
    top_p=0.95
)

response = model.generate(formatted, sampling_params)

print(response[0].outputs[0].text)