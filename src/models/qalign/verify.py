from src.models.remote_vllm import RemoteVLLM
from src.models.qalign.qalign import QAlign
from src.models.qalign.reward import VectorReward
from transformers import AutoTokenizer
  
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = RemoteVLLM(
	server_url="http://localhost:8080",
	model_path="meta-llama/Llama-3.2-1B-Instruct",
	max_prompt_length=1000,
	max_new_tokens=1000,
)

reward = VectorReward([1, 0], ["You are a concise assistant.", "You are a verbose assistant."], "http://localhost:8080", tokenizer, "You are a helpful assistant.", "meta-llama/Llama-3.2-1B-Instruct", "cuda")

chain = QAlign(
	model=model,
	reward=reward
)

conversations = [[{"role": "user", "content": "Can you tell me how to choose a hobby?"}]]

results =chain.run(
	conversations=conversations,
	steps=8,
	warm_start=[{'completion': "You can start by going outside!", 'reward':1000}]
)

print(results.texts[0]['outputs'])