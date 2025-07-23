import json
import torch
import numpy as np
from drift import get_approximation_accuracy
from attribute_prompts import attribute_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ps = [
    [0.0, -0.1819930523633957, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2501052916049957, 0.0, 0.0, 0.0, -0.1779288351535797, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3061266243457794, 0.614651083946228, 0.0, 0.0, 0.3309508264064789, 0.34024953842163086],
    [0.0, 0.0, 0.0, 0.0, 0.2480524629354477, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3474026620388031, 0.0, 0.0, 0.0, 0.3189997971057892, 0.0, 0.0, 0.2870076894760132, 0.4503062069416046, 0.0, 0.0, 0.2374761700630188, 0.25566166639328003],
    [0.0, -0.23653773963451385, 0.0, 0.0, 0.21514713764190674, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.405910462141037, 0.0, 0.0, 0.0, 0.0, 0.27293097972869873, 0.0, 0.2807425558567047, 0.600579559803009, 0.0, 0.0, 0.0, 0.2517958879470825],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.2227494865655899, 0.24646762013435364, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19025957584381104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26726996898651123, 0.6611918210983276, 0.0, 0.0, 0.32031723856925964, 0.19087213277816772],
    [0.0, -0.46971526741981506, 0.0, -0.31083306670188904, 0.0, 0.0, 0.0, 0.20688557624816895, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2261766493320465, 0.0, 0.22364188730716705, 0.0, 0.0, 0.0, 0.0, 0.3168671429157257, 0.47680360078811646, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.280841588973999, 0.0, 0.0, 0.24313214421272278, 0.0, 0.0, 0.0, 0.32967981696128845, 0.0, 0.0, 0.3580660820007324, 0.5343039035797119, 0.0, 0.0, 0.2761712968349457, 0.1856825351715088]
]

# Settings
user = 'user1'  # or set as needed
sample_sizes = [(i+1)*10 for i in range(len(ps))]
test_path = f"../data/preference/{user}_test.json"
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
base_prompt = "You are an AI assistant."

# Load test data
with open(test_path, "r") as f:
    test_data = json.load(f)

test_data = test_data[:100]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)
model.eval()

eval_data = [
    (entry["prompt"], entry["chosen"], entry["rejected"])
    for entry in test_data
]

results = []
for i, p in enumerate(ps):
    n = (i+1)*10
    print(f"Evaluating for n={n}...")
    accuracy = get_approximation_accuracy(
        eval_data,
        model,
        p,
        base_prompt,
        attribute_prompts[:5],
        device,
        tokenizer,
        batch_size=8,
        normalize_by_length=True
    )
    print(f"n={n}, accuracy={accuracy:.4f}")
    results.append({
        "user": user,
        "n": n,
        "acc": accuracy,
        "p": p
    })

with open("../results/approximation_results_stable.jsonl", "a") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")

