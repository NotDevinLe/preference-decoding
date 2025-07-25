from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from drift import log_prob, get_log_probs_vllm, evaluate_continuation_vllm

small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

# llm = LLM(
#     model=small_model_id,
#     dtype="float16",
#     tensor_parallel_size=1,
#     trust_remote_code=True
# )

# model = AutoModelForCausalLM.from_pretrained(small_model_id, device_map="auto", torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

out = tokenizer("What is up peoples?")
print(out)

# systems = ["You are an AI assistant.", "You are an AI assistant.", "You are a highly knowledgeable and friendly assistant. You provide accurate, concise, and well-structured answers to user questions, while maintaining a professional and approachable tone. You admit when you don't know something and avoid making up facts. Use markdown formatting where appropriate."]
# questions = ["What is the capital of France?", "What is the capital of France?", "Can you explain how photosynthesis works in simple terms?"]
# outputs = ["Well how are you doing today good sir, what is it that I can get you today?", "What is up peoples?", """Sure! Here's a simple explanation of **photosynthesis**:

# Photosynthesis is the process plants use to make their own food using **sunlight**, **carbon dioxide (CO₂)** from the air, and **water (H₂O)** from the soil. It happens mostly in the leaves, which contain a green pigment called **chlorophyll**.

# Here’s how it works:

# 1. 🌞 **Sunlight** provides energy.
# 2. 🌿 **Chlorophyll** captures the sunlight.
# 3. 💧 **Water** is absorbed by the roots.
# 4. 💨 **Carbon dioxide** enters through small pores in the leaves.
# 5. 🔬 These ingredients are converted into:
#    - **Glucose (a type of sugar)** – the plant’s food.
#    - **Oxygen** – released into the air.

# The overall simplified equation is:

# 6CO₂ + 6H₂O + sunlight → C₆H₁₂O₆ + 6O₂

# Let me know if you’d like a diagram or a deeper explanation!
# """]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# normal = log_prob(model, outputs, systems, questions, device, tokenizer)

# fast = get_log_probs_vllm(systems, questions, outputs, llm, tokenizer)

# fast_vllm = evaluate_continuation_vllm(llm, tokenizer, systems, questions, outputs)

# print(normal)
# print(fast)
# print(fast_vllm)