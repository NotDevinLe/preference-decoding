from src.models.remote_vllm import RemoteVLLM
from src.models.qalign.qalign import QAlign
from src.models.qalign.reward import ConstantReward

model = RemoteVLLM(
    server_url="http://g3090.hyak.local:8080",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    max_prompt_length=1000,
    max_new_tokens=1000,
)

reward = ConstantReward(1.0)

chain = QAlign(
    model=model,
    reward=reward
)

t = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": "What district is Guimarães in?"}],
    tokenize=False,
    add_generation_prompt=True,
)

results =chain.run(
    prompts=[t],
    steps=8,
)

print(results)