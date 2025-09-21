from src.models.remote_vllm import RemoteVLLM
from src.models.qalign.qalign import QAlign
from src.models.qalign.reward import ConstantReward

  

model = RemoteVLLM(
	server_url="http://g3124.hyak.local:8080",
	model_path="meta-llama/Llama-3.2-1B-Instruct",
	max_prompt_length=1000,
	max_new_tokens=1000,
)

reward = ConstantReward(1.0)

chain = QAlign(
	model=model,
	reward=reward
)

results =chain.run(
	conversations=[[{"role": "user", "content": "What district is Guimarães in?"}]],
	steps=8,
)

print(results.texts[0]['outputs'])