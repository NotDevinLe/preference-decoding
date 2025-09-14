from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = LLM(model=model_id, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=8192)

sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=512, stop=[])

persona = "You are a software developer with extensive experience in WPF, a web developer who prefers to use the terminal for file management, a senior software engineer with expertise in C++ and the LIEF library, a passionate software engineer who actively participates in forums and shares their knowledge. You value clarity above all and make an effort to use macOS for creative work, provide feedback on the audio quality of your recordings, and emphasize the importance of ethical conduct in public service. You occasionally add fresh phrasing to your writing and cite sources, but avoid jargon and unnecessary complexity. When goals compete, you favor higher-weighted traits over lower-weighted ones."

with open("../data/bon_attributes.json", "r") as f:
    bon_data = json.load(f)

questions = []

for item in bon_data:
    questions.append(item["prompt"])

responses = []
for question in questions:
    formatted = tokenizer.apply_chat_template([
        {"role": "system", "content": persona},
        {"role": "user", "content": question}
    ], tokenize=False, add_generation_prompt=True)
    responses.append(formatted)

responses = model.generate(responses, sampling_params)
responses = [output.outputs[0].text.strip() for output in responses]

with open("../results/compressed_responses/user18.json", "w") as f:
    json.dump(responses, f)