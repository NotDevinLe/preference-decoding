import vllm
from transformers import AutoTokenizer
from drift import get_log_probs_vllm

small_model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token
llm = vllm.LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.95, max_model_len=4096)

systems = ["You are a helpful assistant."] * 3
questions = ["What's the capital of France?"] * 3
completions = ["The capital of France is Banana.", "The capital of France is Paris.", "The capital of France is Paris. It is a beautiful city known for the Eiffel Tower, the Louvre, and its rich cultural history."]

lps = get_log_probs_vllm(systems, questions, completions, llm, tokenizer)
print(lps)