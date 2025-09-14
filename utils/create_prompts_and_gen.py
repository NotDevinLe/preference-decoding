from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from attribute_prompts import persona_prompts_3, persona_selected

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = LLM(model=model_id, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=8192)

sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=512, stop=[])

prompt = f"""

You are a persona composer.
Your task is to compress a list of weighted attributes into a single, natural persona description for use as a system prompt.
Input:
A list of tuples in the form (weight, description), where weight ∈ [-1, 1].
Positive weights mean “prefer this behavior.”
Negative weights mean “avoid this behavior.”
Instructions:
Start with a concise role framing, e.g., “You are a helpful assistant…” or “You are a research companion…”.
Translate high-weight positives (≥0.7) into dominant persona traits (“You value clarity above all”).
Translate medium positives (0.3-0.7) into supportive tendencies (“You also make an effort to cite sources…”).
Translate low positives (<0.3) into occasional behaviors (“You occasionally add fresh phrasing…”).
Translate negatives into aversions (“You avoid jargon and unnecessary complexity”).
Reflect relative priority: stronger weights come first, weaker ones later.
Include a balancing statement for conflicts: “When goals compete, you favor higher-weighted traits over lower-weighted ones.”
Output only the final persona description text. Do not list the weights.
Output:
A single persona description suitable to be placed in the system prompt of a language model.

"""

formatted_p = []
info = {"user": "user18", "n": 150, "p": [1.1601211426051123, -0.13658330833826002, 1.0836969048571217, -0.05113611486731564, 0.17574865996324202, 0.185181127039575, 0.14685141985216757, 1.5069965187084071, 0.2692404123351374, 0.9355147613678577, 0.09216189690570133, 0.8862156661659488, 0.12891127164970428, 0.17881148402036165, 1.3584098562455118, 0.530502956175549, 0.2694282390057069, 0.3202139161269084, 0.9722407029425771, 0.7526721232917524, 0.007228077938951025, -0.07538311652632816, 0.8775912542961062, 1.7985130187294656, 0.12536252449379057, 0.15196634063954764, 0.11339751194551805, 0.16812571186617126, 0.060868352748261176, 0.14872809932961847, -0.00316280188347386, 0.7214150979124018, -0.2375642876963089, 0.3320738384097851, -0.05215546300289771, 0.3840854739533438, 0.0, 0.2164957100569795, 1.8474301233837225, 1.0382179936631364, 1.1884092410689362, 0.17155472850272396, 0.14737309459979062, 1.1080297385653524, 0.995453993386658, -0.036317564853004775, 0.04860468008564938], "lambda0": 0.0001, "system_prompt_list": "personas"}
selected_idx = persona_selected[info["lambda0"]]

for i, p in enumerate(info["p"]):
    formatted_p.append({"weight": p, "description": persona_prompts_3[selected_idx[i]]})

formatted = tokenizer.apply_chat_template([
    {"role": "system", "content": prompt},
    {"role": "user", "content": "Input: " + str(formatted_p) + "\nOutput:"}
], tokenize=False, add_generation_prompt=True)

response = model.generate(formatted, sampling_params)
print(response[0].outputs[0].text)